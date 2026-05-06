[Back to Curriculum](../../README.md) | [Previous: Human in the Loop and Monitoring](../../18-Alignment-and-Safety/05-Human-in-the-Loop-and-Monitoring/notes.md) | [Next: Experiment Tracking and Reproducibility](../02-Experiment-Tracking-and-Reproducibility/notes.md)

---

# Data Versioning and Lineage

> _"A production model is only as reproducible as the data graph that produced it."_

## Overview

Data versioning and lineage turn training data from an invisible dependency into an auditable mathematical artifact.

Production ML and MLOps are the mathematical discipline of keeping a learned system useful after it leaves the notebook. The model is only one artifact in a larger graph of data, code, configuration, evaluation, deployment, monitoring, and response actions.

This chapter uses LaTeX Markdown throughout. Inline mathematics uses `$...$`, and display equations use `$$...$$`. The central habit is to turn production behavior into explicit objects: versions, hashes, traces, thresholds, queues, contracts, and release decisions.

## Prerequisites

- [Documentation and Governance](../../16-LLM-Training-Data-Pipeline/06-Documentation-and-Governance/notes.md)
- [Contamination and Dedup Audits](../../16-LLM-Training-Data-Pipeline/05-Contamination-and-Dedup-Audits/notes.md)
- [Error Analysis and Ablations](../../17-Evaluation-and-Reliability/04-Error-Analysis-and-Ablations/notes.md)
- [Human in the Loop and Monitoring](../../18-Alignment-and-Safety/05-Human-in-the-Loop-and-Monitoring/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Executable demonstrations for data versioning and lineage |
| [exercises.ipynb](exercises.ipynb) | Graded practice for data versioning and lineage |

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
  - [1.1 production ML as an artifact graph](#11-production-ml-as-an-artifact-graph)
  - [1.2 why data changes break models](#12-why-data-changes-break-models)
  - [1.3 versioning versus backups](#13-versioning-versus-backups)
  - [1.4 lineage as rollback memory](#14-lineage-as-rollback-memory)
  - [1.5 technical debt from hidden dependencies](#15-technical-debt-from-hidden-dependencies)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 dataset snapshot $\mathcal{D}_v$](#21-dataset-snapshot)
  - [2.2 artifact hash $h(a)$](#22-artifact-hash)
  - [2.3 lineage DAG $G=(V,E)$](#23-lineage-dag)
  - [2.4 provenance metadata](#24-provenance-metadata)
  - [2.5 reproducibility predicate](#25-reproducibility-predicate)
- [3. Data Versioning](#3-data-versioning)
  - [3.1 immutable snapshots](#31-immutable-snapshots)
  - [3.2 Git versus object store versioning](#32-git-versus-object-store-versioning)
  - [3.3 DVC-style pointer files](#33-dvcstyle-pointer-files)
  - [3.4 semantic dataset versions](#34-semantic-dataset-versions)
  - [3.5 data diffs and rollback](#35-data-diffs-and-rollback)
- [4. Lineage Systems](#4-lineage-systems)
  - [4.1 runs, jobs, and datasets](#41-runs-jobs-and-datasets)
  - [4.2 pipeline metadata](#42-pipeline-metadata)
  - [4.3 dependency closure](#43-dependency-closure)
  - [4.4 model-to-data linkage](#44-modeltodata-linkage)
  - [4.5 audit trails](#45-audit-trails)
- [5. Quality Gates](#5-quality-gates)
  - [5.1 schema checks](#51-schema-checks)
  - [5.2 row count and hash checks](#52-row-count-and-hash-checks)
  - [5.3 freshness service-level objectives](#53-freshness-servicelevel-objectives)
  - [5.4 privacy and license gates](#54-privacy-and-license-gates)
  - [5.5 continuous integration validation](#55-continuous-integration-validation)
- [6. ML and LLM Applications](#6-ml-and-llm-applications)
  - [6.1 training corpus lineage](#61-training-corpus-lineage)
  - [6.2 fine-tune provenance](#62-finetune-provenance)
  - [6.3 embedding index versioning](#63-embedding-index-versioning)
  - [6.4 retrieval source traceability](#64-retrieval-source-traceability)
  - [6.5 incident rollback](#65-incident-rollback)
- [7. Common Mistakes](#7-common-mistakes)
- [8. Exercises](#8-exercises)
- [9. Why This Matters for AI](#9-why-this-matters-for-ai)
- [10. Conceptual Bridge](#10-conceptual-bridge)
- [References](#references)

---

## 1. Intuition

Intuition develops the part of data versioning and lineage assigned by the approved Chapter 19 table of contents. The treatment is production-focused: every idea is connected to a versioned artifact, measurable signal, release decision, or incident response.

### 1.1 production ML as an artifact graph

Production ml as an artifact graph is part of the canonical scope of Data Versioning and Lineage. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is artifact versioning, lineage DAGs, provenance metadata, quality gates, and rollback-safe production data operations. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\mathcal{D}_v = \{(\mathbf{x}^{(i)}, y^{(i)}, t_i, m_i)\}_{i=1}^{n_v}.$$

The formula is intentionally simple. It says that production ml as an artifact graph should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of production ml as an artifact graph in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Production ml as an artifact graph is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for production ml as an artifact graph:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of production ml as an artifact graph executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for production ml as an artifact graph should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 1.2 why data changes break models

Why data changes break models is part of the canonical scope of Data Versioning and Lineage. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is artifact versioning, lineage DAGs, provenance metadata, quality gates, and rollback-safe production data operations. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$h(a) = \operatorname{SHA256}(\operatorname{bytes}(a)).$$

The formula is intentionally simple. It says that why data changes break models should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of why data changes break models in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Why data changes break models is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for why data changes break models:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of why data changes break models executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for why data changes break models should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 1.3 versioning versus backups

Versioning versus backups is part of the canonical scope of Data Versioning and Lineage. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is artifact versioning, lineage DAGs, provenance metadata, quality gates, and rollback-safe production data operations. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\operatorname{Anc}(m) = \{v \in V : v \leadsto m\}.$$

The formula is intentionally simple. It says that versioning versus backups should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of versioning versus backups in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Versioning versus backups is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for versioning versus backups:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of versioning versus backups executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for versioning versus backups should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 1.4 lineage as rollback memory

Lineage as rollback memory is part of the canonical scope of Data Versioning and Lineage. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is artifact versioning, lineage DAGs, provenance metadata, quality gates, and rollback-safe production data operations. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$R(m) = \mathbb{1}[h(\mathcal{D})=h_0] \mathbb{1}[h(c)=c_0] \mathbb{1}[h(e)=e_0].$$

The formula is intentionally simple. It says that lineage as rollback memory should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of lineage as rollback memory in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Lineage as rollback memory is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for lineage as rollback memory:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of lineage as rollback memory executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for lineage as rollback memory should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 1.5 technical debt from hidden dependencies

Technical debt from hidden dependencies is part of the canonical scope of Data Versioning and Lineage. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is artifact versioning, lineage DAGs, provenance metadata, quality gates, and rollback-safe production data operations. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\mathcal{D}_v = \{(\mathbf{x}^{(i)}, y^{(i)}, t_i, m_i)\}_{i=1}^{n_v}.$$

The formula is intentionally simple. It says that technical debt from hidden dependencies should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of technical debt from hidden dependencies in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Technical debt from hidden dependencies is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for technical debt from hidden dependencies:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of technical debt from hidden dependencies executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for technical debt from hidden dependencies should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

## 2. Formal Definitions

Formal Definitions develops the part of data versioning and lineage assigned by the approved Chapter 19 table of contents. The treatment is production-focused: every idea is connected to a versioned artifact, measurable signal, release decision, or incident response.

### 2.1 dataset snapshot $\mathcal{D}_v$

Dataset snapshot $\mathcal{d}_v$ is part of the canonical scope of Data Versioning and Lineage. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is artifact versioning, lineage DAGs, provenance metadata, quality gates, and rollback-safe production data operations. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$h(a) = \operatorname{SHA256}(\operatorname{bytes}(a)).$$

The formula is intentionally simple. It says that dataset snapshot $\mathcal{d}_v$ should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of dataset snapshot $\mathcal{d}_v$ in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Dataset snapshot $\mathcal{d}_v$ is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for dataset snapshot $\mathcal{d}_v$:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of dataset snapshot $\mathcal{d}_v$ executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for dataset snapshot $\mathcal{d}_v$ should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 2.2 artifact hash $h(a)$

Artifact hash $h(a)$ is part of the canonical scope of Data Versioning and Lineage. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is artifact versioning, lineage DAGs, provenance metadata, quality gates, and rollback-safe production data operations. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\operatorname{Anc}(m) = \{v \in V : v \leadsto m\}.$$

The formula is intentionally simple. It says that artifact hash $h(a)$ should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of artifact hash $h(a)$ in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Artifact hash $h(a)$ is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for artifact hash $h(a)$:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of artifact hash $h(a)$ executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for artifact hash $h(a)$ should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 2.3 lineage DAG $G=(V,E)$

Lineage dag $g=(v,e)$ is part of the canonical scope of Data Versioning and Lineage. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is artifact versioning, lineage DAGs, provenance metadata, quality gates, and rollback-safe production data operations. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$R(m) = \mathbb{1}[h(\mathcal{D})=h_0] \mathbb{1}[h(c)=c_0] \mathbb{1}[h(e)=e_0].$$

The formula is intentionally simple. It says that lineage dag $g=(v,e)$ should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of lineage dag $g=(v,e)$ in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Lineage dag $g=(v,e)$ is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for lineage dag $g=(v,e)$:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of lineage dag $g=(v,e)$ executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for lineage dag $g=(v,e)$ should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 2.4 provenance metadata

Provenance metadata is part of the canonical scope of Data Versioning and Lineage. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is artifact versioning, lineage DAGs, provenance metadata, quality gates, and rollback-safe production data operations. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\mathcal{D}_v = \{(\mathbf{x}^{(i)}, y^{(i)}, t_i, m_i)\}_{i=1}^{n_v}.$$

The formula is intentionally simple. It says that provenance metadata should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of provenance metadata in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Provenance metadata is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for provenance metadata:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of provenance metadata executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for provenance metadata should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 2.5 reproducibility predicate

Reproducibility predicate is part of the canonical scope of Data Versioning and Lineage. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is artifact versioning, lineage DAGs, provenance metadata, quality gates, and rollback-safe production data operations. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$h(a) = \operatorname{SHA256}(\operatorname{bytes}(a)).$$

The formula is intentionally simple. It says that reproducibility predicate should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of reproducibility predicate in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Reproducibility predicate is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for reproducibility predicate:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of reproducibility predicate executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for reproducibility predicate should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

## 3. Data Versioning

Data Versioning develops the part of data versioning and lineage assigned by the approved Chapter 19 table of contents. The treatment is production-focused: every idea is connected to a versioned artifact, measurable signal, release decision, or incident response.

### 3.1 immutable snapshots

Immutable snapshots is part of the canonical scope of Data Versioning and Lineage. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is artifact versioning, lineage DAGs, provenance metadata, quality gates, and rollback-safe production data operations. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\operatorname{Anc}(m) = \{v \in V : v \leadsto m\}.$$

The formula is intentionally simple. It says that immutable snapshots should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of immutable snapshots in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Immutable snapshots is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for immutable snapshots:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of immutable snapshots executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for immutable snapshots should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 3.2 Git versus object store versioning

Git versus object store versioning is part of the canonical scope of Data Versioning and Lineage. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is artifact versioning, lineage DAGs, provenance metadata, quality gates, and rollback-safe production data operations. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$R(m) = \mathbb{1}[h(\mathcal{D})=h_0] \mathbb{1}[h(c)=c_0] \mathbb{1}[h(e)=e_0].$$

The formula is intentionally simple. It says that git versus object store versioning should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of git versus object store versioning in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Git versus object store versioning is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for git versus object store versioning:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of git versus object store versioning executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for git versus object store versioning should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 3.3 DVC-style pointer files

Dvc-style pointer files is part of the canonical scope of Data Versioning and Lineage. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is artifact versioning, lineage DAGs, provenance metadata, quality gates, and rollback-safe production data operations. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\mathcal{D}_v = \{(\mathbf{x}^{(i)}, y^{(i)}, t_i, m_i)\}_{i=1}^{n_v}.$$

The formula is intentionally simple. It says that dvc-style pointer files should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of dvc-style pointer files in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Dvc-style pointer files is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for dvc-style pointer files:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of dvc-style pointer files executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for dvc-style pointer files should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 3.4 semantic dataset versions

Semantic dataset versions is part of the canonical scope of Data Versioning and Lineage. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is artifact versioning, lineage DAGs, provenance metadata, quality gates, and rollback-safe production data operations. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$h(a) = \operatorname{SHA256}(\operatorname{bytes}(a)).$$

The formula is intentionally simple. It says that semantic dataset versions should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of semantic dataset versions in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Semantic dataset versions is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for semantic dataset versions:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of semantic dataset versions executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for semantic dataset versions should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 3.5 data diffs and rollback

Data diffs and rollback is part of the canonical scope of Data Versioning and Lineage. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is artifact versioning, lineage DAGs, provenance metadata, quality gates, and rollback-safe production data operations. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\operatorname{Anc}(m) = \{v \in V : v \leadsto m\}.$$

The formula is intentionally simple. It says that data diffs and rollback should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of data diffs and rollback in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Data diffs and rollback is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for data diffs and rollback:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of data diffs and rollback executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for data diffs and rollback should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

## 4. Lineage Systems

Lineage Systems develops the part of data versioning and lineage assigned by the approved Chapter 19 table of contents. The treatment is production-focused: every idea is connected to a versioned artifact, measurable signal, release decision, or incident response.

### 4.1 runs, jobs, and datasets

Runs, jobs, and datasets is part of the canonical scope of Data Versioning and Lineage. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is artifact versioning, lineage DAGs, provenance metadata, quality gates, and rollback-safe production data operations. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$R(m) = \mathbb{1}[h(\mathcal{D})=h_0] \mathbb{1}[h(c)=c_0] \mathbb{1}[h(e)=e_0].$$

The formula is intentionally simple. It says that runs, jobs, and datasets should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of runs, jobs, and datasets in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Runs, jobs, and datasets is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for runs, jobs, and datasets:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of runs, jobs, and datasets executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for runs, jobs, and datasets should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 4.2 pipeline metadata

Pipeline metadata is part of the canonical scope of Data Versioning and Lineage. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is artifact versioning, lineage DAGs, provenance metadata, quality gates, and rollback-safe production data operations. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\mathcal{D}_v = \{(\mathbf{x}^{(i)}, y^{(i)}, t_i, m_i)\}_{i=1}^{n_v}.$$

The formula is intentionally simple. It says that pipeline metadata should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of pipeline metadata in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Pipeline metadata is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for pipeline metadata:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of pipeline metadata executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for pipeline metadata should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 4.3 dependency closure

Dependency closure is part of the canonical scope of Data Versioning and Lineage. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is artifact versioning, lineage DAGs, provenance metadata, quality gates, and rollback-safe production data operations. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$h(a) = \operatorname{SHA256}(\operatorname{bytes}(a)).$$

The formula is intentionally simple. It says that dependency closure should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of dependency closure in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Dependency closure is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for dependency closure:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of dependency closure executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for dependency closure should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 4.4 model-to-data linkage

Model-to-data linkage is part of the canonical scope of Data Versioning and Lineage. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is artifact versioning, lineage DAGs, provenance metadata, quality gates, and rollback-safe production data operations. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\operatorname{Anc}(m) = \{v \in V : v \leadsto m\}.$$

The formula is intentionally simple. It says that model-to-data linkage should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of model-to-data linkage in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Model-to-data linkage is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for model-to-data linkage:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of model-to-data linkage executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for model-to-data linkage should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 4.5 audit trails

Audit trails is part of the canonical scope of Data Versioning and Lineage. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is artifact versioning, lineage DAGs, provenance metadata, quality gates, and rollback-safe production data operations. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$R(m) = \mathbb{1}[h(\mathcal{D})=h_0] \mathbb{1}[h(c)=c_0] \mathbb{1}[h(e)=e_0].$$

The formula is intentionally simple. It says that audit trails should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of audit trails in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Audit trails is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for audit trails:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of audit trails executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for audit trails should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

## 5. Quality Gates

Quality Gates develops the part of data versioning and lineage assigned by the approved Chapter 19 table of contents. The treatment is production-focused: every idea is connected to a versioned artifact, measurable signal, release decision, or incident response.

### 5.1 schema checks

Schema checks is part of the canonical scope of Data Versioning and Lineage. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is artifact versioning, lineage DAGs, provenance metadata, quality gates, and rollback-safe production data operations. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\mathcal{D}_v = \{(\mathbf{x}^{(i)}, y^{(i)}, t_i, m_i)\}_{i=1}^{n_v}.$$

The formula is intentionally simple. It says that schema checks should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of schema checks in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Schema checks is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for schema checks:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of schema checks executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for schema checks should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 5.2 row count and hash checks

Row count and hash checks is part of the canonical scope of Data Versioning and Lineage. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is artifact versioning, lineage DAGs, provenance metadata, quality gates, and rollback-safe production data operations. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$h(a) = \operatorname{SHA256}(\operatorname{bytes}(a)).$$

The formula is intentionally simple. It says that row count and hash checks should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of row count and hash checks in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Row count and hash checks is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for row count and hash checks:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of row count and hash checks executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for row count and hash checks should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 5.3 freshness service-level objectives

Freshness service-level objectives is part of the canonical scope of Data Versioning and Lineage. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is artifact versioning, lineage DAGs, provenance metadata, quality gates, and rollback-safe production data operations. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\operatorname{Anc}(m) = \{v \in V : v \leadsto m\}.$$

The formula is intentionally simple. It says that freshness service-level objectives should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of freshness service-level objectives in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Freshness service-level objectives is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for freshness service-level objectives:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of freshness service-level objectives executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for freshness service-level objectives should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 5.4 privacy and license gates

Privacy and license gates is part of the canonical scope of Data Versioning and Lineage. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is artifact versioning, lineage DAGs, provenance metadata, quality gates, and rollback-safe production data operations. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$R(m) = \mathbb{1}[h(\mathcal{D})=h_0] \mathbb{1}[h(c)=c_0] \mathbb{1}[h(e)=e_0].$$

The formula is intentionally simple. It says that privacy and license gates should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of privacy and license gates in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Privacy and license gates is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for privacy and license gates:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of privacy and license gates executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for privacy and license gates should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 5.5 continuous integration validation

Continuous integration validation is part of the canonical scope of Data Versioning and Lineage. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is artifact versioning, lineage DAGs, provenance metadata, quality gates, and rollback-safe production data operations. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\mathcal{D}_v = \{(\mathbf{x}^{(i)}, y^{(i)}, t_i, m_i)\}_{i=1}^{n_v}.$$

The formula is intentionally simple. It says that continuous integration validation should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of continuous integration validation in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Continuous integration validation is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for continuous integration validation:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of continuous integration validation executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for continuous integration validation should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

## 6. ML and LLM Applications

ML and LLM Applications develops the part of data versioning and lineage assigned by the approved Chapter 19 table of contents. The treatment is production-focused: every idea is connected to a versioned artifact, measurable signal, release decision, or incident response.

### 6.1 training corpus lineage

Training corpus lineage is part of the canonical scope of Data Versioning and Lineage. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is artifact versioning, lineage DAGs, provenance metadata, quality gates, and rollback-safe production data operations. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$h(a) = \operatorname{SHA256}(\operatorname{bytes}(a)).$$

The formula is intentionally simple. It says that training corpus lineage should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of training corpus lineage in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Training corpus lineage is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for training corpus lineage:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of training corpus lineage executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for training corpus lineage should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 6.2 fine-tune provenance

Fine-tune provenance is part of the canonical scope of Data Versioning and Lineage. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is artifact versioning, lineage DAGs, provenance metadata, quality gates, and rollback-safe production data operations. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\operatorname{Anc}(m) = \{v \in V : v \leadsto m\}.$$

The formula is intentionally simple. It says that fine-tune provenance should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of fine-tune provenance in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Fine-tune provenance is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for fine-tune provenance:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of fine-tune provenance executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for fine-tune provenance should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 6.3 embedding index versioning

Embedding index versioning is part of the canonical scope of Data Versioning and Lineage. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is artifact versioning, lineage DAGs, provenance metadata, quality gates, and rollback-safe production data operations. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$R(m) = \mathbb{1}[h(\mathcal{D})=h_0] \mathbb{1}[h(c)=c_0] \mathbb{1}[h(e)=e_0].$$

The formula is intentionally simple. It says that embedding index versioning should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of embedding index versioning in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Embedding index versioning is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for embedding index versioning:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of embedding index versioning executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for embedding index versioning should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 6.4 retrieval source traceability

Retrieval source traceability is part of the canonical scope of Data Versioning and Lineage. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is artifact versioning, lineage DAGs, provenance metadata, quality gates, and rollback-safe production data operations. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\mathcal{D}_v = \{(\mathbf{x}^{(i)}, y^{(i)}, t_i, m_i)\}_{i=1}^{n_v}.$$

The formula is intentionally simple. It says that retrieval source traceability should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of retrieval source traceability in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Retrieval source traceability is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for retrieval source traceability:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of retrieval source traceability executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for retrieval source traceability should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 6.5 incident rollback

Incident rollback is part of the canonical scope of Data Versioning and Lineage. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is artifact versioning, lineage DAGs, provenance metadata, quality gates, and rollback-safe production data operations. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$h(a) = \operatorname{SHA256}(\operatorname{bytes}(a)).$$

The formula is intentionally simple. It says that incident rollback should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of incident rollback in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Incident rollback is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for incident rollback:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of incident rollback executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for incident rollback should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

## 7. Common Mistakes

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

## 8. Exercises

1. (*) Design a production ML check related to data versioning and lineage.
   - (a) Define the object being checked using mathematical notation.
   - (b) State the metric, predicate, or threshold used to decide pass/fail.
   - (c) Explain which artifact versions must be logged.
   - (d) Give one failure case and one rollback or escalation action.

2. (*) Design a production ML check related to data versioning and lineage.
   - (a) Define the object being checked using mathematical notation.
   - (b) State the metric, predicate, or threshold used to decide pass/fail.
   - (c) Explain which artifact versions must be logged.
   - (d) Give one failure case and one rollback or escalation action.

3. (*) Design a production ML check related to data versioning and lineage.
   - (a) Define the object being checked using mathematical notation.
   - (b) State the metric, predicate, or threshold used to decide pass/fail.
   - (c) Explain which artifact versions must be logged.
   - (d) Give one failure case and one rollback or escalation action.

4. (**) Design a production ML check related to data versioning and lineage.
   - (a) Define the object being checked using mathematical notation.
   - (b) State the metric, predicate, or threshold used to decide pass/fail.
   - (c) Explain which artifact versions must be logged.
   - (d) Give one failure case and one rollback or escalation action.

5. (**) Design a production ML check related to data versioning and lineage.
   - (a) Define the object being checked using mathematical notation.
   - (b) State the metric, predicate, or threshold used to decide pass/fail.
   - (c) Explain which artifact versions must be logged.
   - (d) Give one failure case and one rollback or escalation action.

6. (**) Design a production ML check related to data versioning and lineage.
   - (a) Define the object being checked using mathematical notation.
   - (b) State the metric, predicate, or threshold used to decide pass/fail.
   - (c) Explain which artifact versions must be logged.
   - (d) Give one failure case and one rollback or escalation action.

7. (***) Design a production ML check related to data versioning and lineage.
   - (a) Define the object being checked using mathematical notation.
   - (b) State the metric, predicate, or threshold used to decide pass/fail.
   - (c) Explain which artifact versions must be logged.
   - (d) Give one failure case and one rollback or escalation action.

8. (***) Design a production ML check related to data versioning and lineage.
   - (a) Define the object being checked using mathematical notation.
   - (b) State the metric, predicate, or threshold used to decide pass/fail.
   - (c) Explain which artifact versions must be logged.
   - (d) Give one failure case and one rollback or escalation action.

9. (***) Design a production ML check related to data versioning and lineage.
   - (a) Define the object being checked using mathematical notation.
   - (b) State the metric, predicate, or threshold used to decide pass/fail.
   - (c) Explain which artifact versions must be logged.
   - (d) Give one failure case and one rollback or escalation action.

10. (***) Design a production ML check related to data versioning and lineage.
   - (a) Define the object being checked using mathematical notation.
   - (b) State the metric, predicate, or threshold used to decide pass/fail.
   - (c) Explain which artifact versions must be logged.
   - (d) Give one failure case and one rollback or escalation action.

## 9. Why This Matters for AI

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

## 10. Conceptual Bridge

Data Versioning and Lineage sits after the chapters on data construction, evaluation, and alignment because production systems combine all three. Chapter 16 explains how reliable datasets are assembled. Chapter 17 explains how models are measured. Chapter 18 explains how desired behavior and safety constraints are specified. Chapter 19 asks whether those ideas survive contact with changing data, users, services, and costs.

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

- Google Cloud. MLOps: Continuous delivery and automation pipelines in machine learning. https://cloud.google.com/solutions/machine-learning/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning
- Sculley et al.. Hidden Technical Debt in Machine Learning Systems. https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-syst
- DVC. Versioning data and models. https://dvc.org/doc/use-cases/versioning-data-and-models
- OpenLineage. OpenLineage specification and documentation. https://openlineage.io/
