[Back to Curriculum](../../README.md) | [Previous: Data Format Standards](../01-Data-Format-Standards/notes.md) | [Next: Quality Checks](../03-Quality-Checks/notes.md)

---

# JSONL Generation

> _"A good generator is boring: deterministic, streamable, restartable, and easy to audit."_

## Overview

JSONL generation turns heterogeneous raw sources into deterministic streams of validated
training records. In an LLM training run, data is not an inert pile of text; it is the
empirical distribution that defines the examples, losses, risks, and capabilities the
model will see.

This section is written as LaTeX Markdown. Inline mathematics uses `$...$`, and display
equations use `$$...$$`. The goal is to connect data engineering decisions to
mathematical objects such as records $r_i$, token sequences $x_{1:T}$, filters $f(x)$,
hashes $h(x)$, mixture weights $\boldsymbol{\alpha}$, and empirical expectations.

The scope is deliberately narrow: this chapter owns the training-data pipeline.
Tokenizer design, GPU training systems, benchmark methodology, alignment objectives, and
production MLOps each have their own canonical chapters. Here we study the data objects
that those later systems consume.

## Prerequisites

- [Data Format Standards](../01-Data-Format-Standards/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Executable demonstrations for jsonl generation |
| [exercises.ipynb](exercises.ipynb) | Graded practice for jsonl generation |

## Learning Objectives

After completing this section, you will be able to:

- Define a JSONL generator as a deterministic map from source objects to records
- Implement memory-safe streaming over shards
- Preserve metadata and source trace fields during serialization
- Design quarantine paths for records that fail validation
- Measure throughput, parse failure rates, and duplicate IDs
- Explain atomic writes, resume logic, and deterministic ordering
- Separate extraction, transformation, validation, and writing stages
- Generate line-delimited JSON that can be parsed independently per line

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 JSONL as streamable training data](#11-jsonl-as-streamable-training-data)
  - [1.2 Serialization as deterministic map $g(r_i)$](#12-serialization-as-deterministic-map-gri)
  - [1.3 Why one-record-per-line matters](#13-why-onerecordperline-matters)
  - [1.4 Reproducibility](#14-reproducibility)
  - [1.5 Failure modes in large generation jobs](#15-failure-modes-in-large-generation-jobs)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Generator function](#21-generator-function)
  - [2.2 Canonical JSON encoding](#22-canonical-json-encoding)
  - [2.3 Idempotence](#23-idempotence)
  - [2.4 Deterministic ordering](#24-deterministic-ordering)
  - [2.5 Atomic write contract](#25-atomic-write-contract)
- [3. Source Extraction](#3-source-extraction)
  - [3.1 Plain text](#31-plain-text)
  - [3.2 HTML extraction preview](#32-html-extraction-preview)
  - [3.3 Code/source files](#33-codesource-files)
  - [3.4 PDF/OCR preview](#34-pdfocr-preview)
  - [3.5 Document boundaries](#35-document-boundaries)
- [4. Record Construction](#4-record-construction)
  - [4.1 Field mapping](#41-field-mapping)
  - [4.2 Metadata preservation](#42-metadata-preservation)
  - [4.3 Token count estimates](#43-token-count-estimates)
  - [4.4 Source trace](#44-source-trace)
  - [4.5 Error quarantine](#45-error-quarantine)
- [5. Streaming and Sharding](#5-streaming-and-sharding)
  - [5.1 Python generators](#51-python-generators)
  - [5.2 Memory-safe iteration](#52-memorysafe-iteration)
  - [5.3 Shard rotation](#53-shard-rotation)
  - [5.4 Compression](#54-compression)
  - [5.5 Resume/restart logic](#55-resumerestart-logic)
- [6. Validation and Performance](#6-validation-and-performance)
  - [6.1 Line-level JSON parse](#61-linelevel-json-parse)
  - [6.2 Duplicate ID detection](#62-duplicate-id-detection)
  - [6.3 Throughput metrics](#63-throughput-metrics)
  - [6.4 Multiprocessing](#64-multiprocessing)
  - [6.5 Deterministic tests](#65-deterministic-tests)
- [7. Common Mistakes](#7-common-mistakes)
- [8. Exercises](#8-exercises)
- [9. Why This Matters for AI](#9-why-this-matters-for-ai)
- [10. Conceptual Bridge](#10-conceptual-bridge)
- [References](#references)

---

## 1. Intuition

Intuition gives the conceptual and mathematical layer for jsonl generation. The local
variables in this section should be read as pipeline objects: documents, records,
tokens, filters, weights, shards, and manifests.

### 1.1 JSONL as streamable training data

JSONL as streamable training data is part of the canonical scope of jsonl generation. We
model the relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with
record-level metadata $m_i$ and text or token content $x_i$. The practical question is
whether the transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `generator`, the invariant should be explicit enough that a checker can fail fast.
If the invariant is only written in a notebook comment or an engineer's memory, it will
not protect a long-running data build.

Examples:
- A small local experiment can store this object in memory; a frontier-scale run must store it as sharded, versioned, validated records.
- The mathematical object is simple, but the operational contract must survive restarts, parallel workers, schema changes, and audits.
- The notebook for this section uses synthetic data so the same ideas can be executed without external files.

Non-examples:
- A path on disk without a manifest is not a reproducible dataset.
- A metric dashboard without record-level lineage is not a provenance system.
- A filter threshold without an audit sample is not evidence of quality.

Implementation consequence: every transformation should report both a count and a rate.
If $n_{\mathrm{in}}$ records enter the stage and $n_{\mathrm{out}}$ records leave, the
acceptance rate is

$$
a = \frac{n_{\mathrm{out}}}{n_{\mathrm{in}}}.
$$

A sudden change in $a$ is a data-drift signal even when the code still runs. This is why
pipeline math is inseparable from logging, manifests, and audit slices.

For LLM work, the token-weighted view is often more important than the document-weighted
view. A filter that removes 5 percent of documents may remove 30 percent of tokens if it
targets long documents. The corresponding token acceptance rate is

$$
a_{\mathrm{tok}} = \frac{\sum_i f(r_i)\,T_i}{\sum_i T_i},
$$

where $T_i$ is the token count or a deterministic token-count estimate. The distinction
matters for compute budgets, mixture proportions, and scaling-law interpretation.

### 1.2 Serialization as deterministic map $g(r_i)$

Serialization as deterministic map $g(r_i)$ is part of the canonical scope of jsonl
generation. We model the relevant object as a finite collection $\mathcal{D} =
\{r_i\}_{i=1}^n$ with record-level metadata $m_i$ and text or token content $x_i$. The
practical question is whether the transformation preserves the intended empirical
distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `serialization`, the invariant should be explicit enough that a checker can fail
fast. If the invariant is only written in a notebook comment or an engineer's memory, it
will not protect a long-running data build.

Examples:
- A small local experiment can store this object in memory; a frontier-scale run must store it as sharded, versioned, validated records.
- The mathematical object is simple, but the operational contract must survive restarts, parallel workers, schema changes, and audits.
- The notebook for this section uses synthetic data so the same ideas can be executed without external files.

Non-examples:
- A path on disk without a manifest is not a reproducible dataset.
- A metric dashboard without record-level lineage is not a provenance system.
- A filter threshold without an audit sample is not evidence of quality.

Implementation consequence: every transformation should report both a count and a rate.
If $n_{\mathrm{in}}$ records enter the stage and $n_{\mathrm{out}}$ records leave, the
acceptance rate is

$$
a = \frac{n_{\mathrm{out}}}{n_{\mathrm{in}}}.
$$

A sudden change in $a$ is a data-drift signal even when the code still runs. This is why
pipeline math is inseparable from logging, manifests, and audit slices.

For LLM work, the token-weighted view is often more important than the document-weighted
view. A filter that removes 5 percent of documents may remove 30 percent of tokens if it
targets long documents. The corresponding token acceptance rate is

$$
a_{\mathrm{tok}} = \frac{\sum_i f(r_i)\,T_i}{\sum_i T_i},
$$

where $T_i$ is the token count or a deterministic token-count estimate. The distinction
matters for compute budgets, mixture proportions, and scaling-law interpretation.

### 1.3 Why one-record-per-line matters

Why one-record-per-line matters is part of the canonical scope of jsonl generation. We
model the relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with
record-level metadata $m_i$ and text or token content $x_i$. The practical question is
whether the transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `shard`, the invariant should be explicit enough that a checker can fail fast. If
the invariant is only written in a notebook comment or an engineer's memory, it will not
protect a long-running data build.

Examples:
- A small local experiment can store this object in memory; a frontier-scale run must store it as sharded, versioned, validated records.
- The mathematical object is simple, but the operational contract must survive restarts, parallel workers, schema changes, and audits.
- The notebook for this section uses synthetic data so the same ideas can be executed without external files.

Non-examples:
- A path on disk without a manifest is not a reproducible dataset.
- A metric dashboard without record-level lineage is not a provenance system.
- A filter threshold without an audit sample is not evidence of quality.

Implementation consequence: every transformation should report both a count and a rate.
If $n_{\mathrm{in}}$ records enter the stage and $n_{\mathrm{out}}$ records leave, the
acceptance rate is

$$
a = \frac{n_{\mathrm{out}}}{n_{\mathrm{in}}}.
$$

A sudden change in $a$ is a data-drift signal even when the code still runs. This is why
pipeline math is inseparable from logging, manifests, and audit slices.

For LLM work, the token-weighted view is often more important than the document-weighted
view. A filter that removes 5 percent of documents may remove 30 percent of tokens if it
targets long documents. The corresponding token acceptance rate is

$$
a_{\mathrm{tok}} = \frac{\sum_i f(r_i)\,T_i}{\sum_i T_i},
$$

where $T_i$ is the token count or a deterministic token-count estimate. The distinction
matters for compute budgets, mixture proportions, and scaling-law interpretation.

### 1.4 Reproducibility

Reproducibility is part of the canonical scope of jsonl generation. We model the
relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level
metadata $m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `quarantine`, the invariant should be explicit enough that a checker can fail fast.
If the invariant is only written in a notebook comment or an engineer's memory, it will
not protect a long-running data build.

Examples:
- A small local experiment can store this object in memory; a frontier-scale run must store it as sharded, versioned, validated records.
- The mathematical object is simple, but the operational contract must survive restarts, parallel workers, schema changes, and audits.
- The notebook for this section uses synthetic data so the same ideas can be executed without external files.

Non-examples:
- A path on disk without a manifest is not a reproducible dataset.
- A metric dashboard without record-level lineage is not a provenance system.
- A filter threshold without an audit sample is not evidence of quality.

Implementation consequence: every transformation should report both a count and a rate.
If $n_{\mathrm{in}}$ records enter the stage and $n_{\mathrm{out}}$ records leave, the
acceptance rate is

$$
a = \frac{n_{\mathrm{out}}}{n_{\mathrm{in}}}.
$$

A sudden change in $a$ is a data-drift signal even when the code still runs. This is why
pipeline math is inseparable from logging, manifests, and audit slices.

For LLM work, the token-weighted view is often more important than the document-weighted
view. A filter that removes 5 percent of documents may remove 30 percent of tokens if it
targets long documents. The corresponding token acceptance rate is

$$
a_{\mathrm{tok}} = \frac{\sum_i f(r_i)\,T_i}{\sum_i T_i},
$$

where $T_i$ is the token count or a deterministic token-count estimate. The distinction
matters for compute budgets, mixture proportions, and scaling-law interpretation.

### 1.5 Failure modes in large generation jobs

Failure modes in large generation jobs is part of the canonical scope of jsonl
generation. We model the relevant object as a finite collection $\mathcal{D} =
\{r_i\}_{i=1}^n$ with record-level metadata $m_i$ and text or token content $x_i$. The
practical question is whether the transformation preserves the intended empirical
distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `idempotence`, the invariant should be explicit enough that a checker can fail fast.
If the invariant is only written in a notebook comment or an engineer's memory, it will
not protect a long-running data build.

Examples:
- A small local experiment can store this object in memory; a frontier-scale run must store it as sharded, versioned, validated records.
- The mathematical object is simple, but the operational contract must survive restarts, parallel workers, schema changes, and audits.
- The notebook for this section uses synthetic data so the same ideas can be executed without external files.

Non-examples:
- A path on disk without a manifest is not a reproducible dataset.
- A metric dashboard without record-level lineage is not a provenance system.
- A filter threshold without an audit sample is not evidence of quality.

Implementation consequence: every transformation should report both a count and a rate.
If $n_{\mathrm{in}}$ records enter the stage and $n_{\mathrm{out}}$ records leave, the
acceptance rate is

$$
a = \frac{n_{\mathrm{out}}}{n_{\mathrm{in}}}.
$$

A sudden change in $a$ is a data-drift signal even when the code still runs. This is why
pipeline math is inseparable from logging, manifests, and audit slices.

For LLM work, the token-weighted view is often more important than the document-weighted
view. A filter that removes 5 percent of documents may remove 30 percent of tokens if it
targets long documents. The corresponding token acceptance rate is

$$
a_{\mathrm{tok}} = \frac{\sum_i f(r_i)\,T_i}{\sum_i T_i},
$$

where $T_i$ is the token count or a deterministic token-count estimate. The distinction
matters for compute budgets, mixture proportions, and scaling-law interpretation.

## 2. Formal Definitions

Formal Definitions gives the conceptual and mathematical layer for jsonl generation. The
local variables in this section should be read as pipeline objects: documents, records,
tokens, filters, weights, shards, and manifests.

### 2.1 Generator function

Generator function is part of the canonical scope of jsonl generation. We model the
relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level
metadata $m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `generator`, the invariant should be explicit enough that a checker can fail fast.
If the invariant is only written in a notebook comment or an engineer's memory, it will
not protect a long-running data build.

Examples:
- A small local experiment can store this object in memory; a frontier-scale run must store it as sharded, versioned, validated records.
- The mathematical object is simple, but the operational contract must survive restarts, parallel workers, schema changes, and audits.
- The notebook for this section uses synthetic data so the same ideas can be executed without external files.

Non-examples:
- A path on disk without a manifest is not a reproducible dataset.
- A metric dashboard without record-level lineage is not a provenance system.
- A filter threshold without an audit sample is not evidence of quality.

Implementation consequence: every transformation should report both a count and a rate.
If $n_{\mathrm{in}}$ records enter the stage and $n_{\mathrm{out}}$ records leave, the
acceptance rate is

$$
a = \frac{n_{\mathrm{out}}}{n_{\mathrm{in}}}.
$$

A sudden change in $a$ is a data-drift signal even when the code still runs. This is why
pipeline math is inseparable from logging, manifests, and audit slices.

For LLM work, the token-weighted view is often more important than the document-weighted
view. A filter that removes 5 percent of documents may remove 30 percent of tokens if it
targets long documents. The corresponding token acceptance rate is

$$
a_{\mathrm{tok}} = \frac{\sum_i f(r_i)\,T_i}{\sum_i T_i},
$$

where $T_i$ is the token count or a deterministic token-count estimate. The distinction
matters for compute budgets, mixture proportions, and scaling-law interpretation.

### 2.2 Canonical JSON encoding

Canonical JSON encoding is part of the canonical scope of jsonl generation. We model the
relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level
metadata $m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `serialization`, the invariant should be explicit enough that a checker can fail
fast. If the invariant is only written in a notebook comment or an engineer's memory, it
will not protect a long-running data build.

Examples:
- A small local experiment can store this object in memory; a frontier-scale run must store it as sharded, versioned, validated records.
- The mathematical object is simple, but the operational contract must survive restarts, parallel workers, schema changes, and audits.
- The notebook for this section uses synthetic data so the same ideas can be executed without external files.

Non-examples:
- A path on disk without a manifest is not a reproducible dataset.
- A metric dashboard without record-level lineage is not a provenance system.
- A filter threshold without an audit sample is not evidence of quality.

Implementation consequence: every transformation should report both a count and a rate.
If $n_{\mathrm{in}}$ records enter the stage and $n_{\mathrm{out}}$ records leave, the
acceptance rate is

$$
a = \frac{n_{\mathrm{out}}}{n_{\mathrm{in}}}.
$$

A sudden change in $a$ is a data-drift signal even when the code still runs. This is why
pipeline math is inseparable from logging, manifests, and audit slices.

For LLM work, the token-weighted view is often more important than the document-weighted
view. A filter that removes 5 percent of documents may remove 30 percent of tokens if it
targets long documents. The corresponding token acceptance rate is

$$
a_{\mathrm{tok}} = \frac{\sum_i f(r_i)\,T_i}{\sum_i T_i},
$$

where $T_i$ is the token count or a deterministic token-count estimate. The distinction
matters for compute budgets, mixture proportions, and scaling-law interpretation.

### 2.3 Idempotence

Idempotence is part of the canonical scope of jsonl generation. We model the relevant
object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level metadata
$m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `shard`, the invariant should be explicit enough that a checker can fail fast. If
the invariant is only written in a notebook comment or an engineer's memory, it will not
protect a long-running data build.

Examples:
- A small local experiment can store this object in memory; a frontier-scale run must store it as sharded, versioned, validated records.
- The mathematical object is simple, but the operational contract must survive restarts, parallel workers, schema changes, and audits.
- The notebook for this section uses synthetic data so the same ideas can be executed without external files.

Non-examples:
- A path on disk without a manifest is not a reproducible dataset.
- A metric dashboard without record-level lineage is not a provenance system.
- A filter threshold without an audit sample is not evidence of quality.

Implementation consequence: every transformation should report both a count and a rate.
If $n_{\mathrm{in}}$ records enter the stage and $n_{\mathrm{out}}$ records leave, the
acceptance rate is

$$
a = \frac{n_{\mathrm{out}}}{n_{\mathrm{in}}}.
$$

A sudden change in $a$ is a data-drift signal even when the code still runs. This is why
pipeline math is inseparable from logging, manifests, and audit slices.

For LLM work, the token-weighted view is often more important than the document-weighted
view. A filter that removes 5 percent of documents may remove 30 percent of tokens if it
targets long documents. The corresponding token acceptance rate is

$$
a_{\mathrm{tok}} = \frac{\sum_i f(r_i)\,T_i}{\sum_i T_i},
$$

where $T_i$ is the token count or a deterministic token-count estimate. The distinction
matters for compute budgets, mixture proportions, and scaling-law interpretation.

### 2.4 Deterministic ordering

Deterministic ordering is part of the canonical scope of jsonl generation. We model the
relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level
metadata $m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `quarantine`, the invariant should be explicit enough that a checker can fail fast.
If the invariant is only written in a notebook comment or an engineer's memory, it will
not protect a long-running data build.

Examples:
- A small local experiment can store this object in memory; a frontier-scale run must store it as sharded, versioned, validated records.
- The mathematical object is simple, but the operational contract must survive restarts, parallel workers, schema changes, and audits.
- The notebook for this section uses synthetic data so the same ideas can be executed without external files.

Non-examples:
- A path on disk without a manifest is not a reproducible dataset.
- A metric dashboard without record-level lineage is not a provenance system.
- A filter threshold without an audit sample is not evidence of quality.

Implementation consequence: every transformation should report both a count and a rate.
If $n_{\mathrm{in}}$ records enter the stage and $n_{\mathrm{out}}$ records leave, the
acceptance rate is

$$
a = \frac{n_{\mathrm{out}}}{n_{\mathrm{in}}}.
$$

A sudden change in $a$ is a data-drift signal even when the code still runs. This is why
pipeline math is inseparable from logging, manifests, and audit slices.

For LLM work, the token-weighted view is often more important than the document-weighted
view. A filter that removes 5 percent of documents may remove 30 percent of tokens if it
targets long documents. The corresponding token acceptance rate is

$$
a_{\mathrm{tok}} = \frac{\sum_i f(r_i)\,T_i}{\sum_i T_i},
$$

where $T_i$ is the token count or a deterministic token-count estimate. The distinction
matters for compute budgets, mixture proportions, and scaling-law interpretation.

### 2.5 Atomic write contract

Atomic write contract is part of the canonical scope of jsonl generation. We model the
relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level
metadata $m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `idempotence`, the invariant should be explicit enough that a checker can fail fast.
If the invariant is only written in a notebook comment or an engineer's memory, it will
not protect a long-running data build.

Examples:
- A small local experiment can store this object in memory; a frontier-scale run must store it as sharded, versioned, validated records.
- The mathematical object is simple, but the operational contract must survive restarts, parallel workers, schema changes, and audits.
- The notebook for this section uses synthetic data so the same ideas can be executed without external files.

Non-examples:
- A path on disk without a manifest is not a reproducible dataset.
- A metric dashboard without record-level lineage is not a provenance system.
- A filter threshold without an audit sample is not evidence of quality.

Implementation consequence: every transformation should report both a count and a rate.
If $n_{\mathrm{in}}$ records enter the stage and $n_{\mathrm{out}}$ records leave, the
acceptance rate is

$$
a = \frac{n_{\mathrm{out}}}{n_{\mathrm{in}}}.
$$

A sudden change in $a$ is a data-drift signal even when the code still runs. This is why
pipeline math is inseparable from logging, manifests, and audit slices.

For LLM work, the token-weighted view is often more important than the document-weighted
view. A filter that removes 5 percent of documents may remove 30 percent of tokens if it
targets long documents. The corresponding token acceptance rate is

$$
a_{\mathrm{tok}} = \frac{\sum_i f(r_i)\,T_i}{\sum_i T_i},
$$

where $T_i$ is the token count or a deterministic token-count estimate. The distinction
matters for compute budgets, mixture proportions, and scaling-law interpretation.

## 3. Source Extraction

Source Extraction gives the conceptual and mathematical layer for jsonl generation. The
local variables in this section should be read as pipeline objects: documents, records,
tokens, filters, weights, shards, and manifests.

### 3.1 Plain text

Plain text is part of the canonical scope of jsonl generation. We model the relevant
object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level metadata
$m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `generator`, the invariant should be explicit enough that a checker can fail fast.
If the invariant is only written in a notebook comment or an engineer's memory, it will
not protect a long-running data build.

Examples:
- A small local experiment can store this object in memory; a frontier-scale run must store it as sharded, versioned, validated records.
- The mathematical object is simple, but the operational contract must survive restarts, parallel workers, schema changes, and audits.
- The notebook for this section uses synthetic data so the same ideas can be executed without external files.

Non-examples:
- A path on disk without a manifest is not a reproducible dataset.
- A metric dashboard without record-level lineage is not a provenance system.
- A filter threshold without an audit sample is not evidence of quality.

Implementation consequence: every transformation should report both a count and a rate.
If $n_{\mathrm{in}}$ records enter the stage and $n_{\mathrm{out}}$ records leave, the
acceptance rate is

$$
a = \frac{n_{\mathrm{out}}}{n_{\mathrm{in}}}.
$$

A sudden change in $a$ is a data-drift signal even when the code still runs. This is why
pipeline math is inseparable from logging, manifests, and audit slices.

For LLM work, the token-weighted view is often more important than the document-weighted
view. A filter that removes 5 percent of documents may remove 30 percent of tokens if it
targets long documents. The corresponding token acceptance rate is

$$
a_{\mathrm{tok}} = \frac{\sum_i f(r_i)\,T_i}{\sum_i T_i},
$$

where $T_i$ is the token count or a deterministic token-count estimate. The distinction
matters for compute budgets, mixture proportions, and scaling-law interpretation.

### 3.2 HTML extraction preview

HTML extraction preview is part of the canonical scope of jsonl generation. We model the
relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level
metadata $m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `serialization`, the invariant should be explicit enough that a checker can fail
fast. If the invariant is only written in a notebook comment or an engineer's memory, it
will not protect a long-running data build.

Examples:
- A small local experiment can store this object in memory; a frontier-scale run must store it as sharded, versioned, validated records.
- The mathematical object is simple, but the operational contract must survive restarts, parallel workers, schema changes, and audits.
- The notebook for this section uses synthetic data so the same ideas can be executed without external files.

Non-examples:
- A path on disk without a manifest is not a reproducible dataset.
- A metric dashboard without record-level lineage is not a provenance system.
- A filter threshold without an audit sample is not evidence of quality.

Implementation consequence: every transformation should report both a count and a rate.
If $n_{\mathrm{in}}$ records enter the stage and $n_{\mathrm{out}}$ records leave, the
acceptance rate is

$$
a = \frac{n_{\mathrm{out}}}{n_{\mathrm{in}}}.
$$

A sudden change in $a$ is a data-drift signal even when the code still runs. This is why
pipeline math is inseparable from logging, manifests, and audit slices.

For LLM work, the token-weighted view is often more important than the document-weighted
view. A filter that removes 5 percent of documents may remove 30 percent of tokens if it
targets long documents. The corresponding token acceptance rate is

$$
a_{\mathrm{tok}} = \frac{\sum_i f(r_i)\,T_i}{\sum_i T_i},
$$

where $T_i$ is the token count or a deterministic token-count estimate. The distinction
matters for compute budgets, mixture proportions, and scaling-law interpretation.

### 3.3 Code/source files

Code/source files is part of the canonical scope of jsonl generation. We model the
relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level
metadata $m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `shard`, the invariant should be explicit enough that a checker can fail fast. If
the invariant is only written in a notebook comment or an engineer's memory, it will not
protect a long-running data build.

Examples:
- A small local experiment can store this object in memory; a frontier-scale run must store it as sharded, versioned, validated records.
- The mathematical object is simple, but the operational contract must survive restarts, parallel workers, schema changes, and audits.
- The notebook for this section uses synthetic data so the same ideas can be executed without external files.

Non-examples:
- A path on disk without a manifest is not a reproducible dataset.
- A metric dashboard without record-level lineage is not a provenance system.
- A filter threshold without an audit sample is not evidence of quality.

Implementation consequence: every transformation should report both a count and a rate.
If $n_{\mathrm{in}}$ records enter the stage and $n_{\mathrm{out}}$ records leave, the
acceptance rate is

$$
a = \frac{n_{\mathrm{out}}}{n_{\mathrm{in}}}.
$$

A sudden change in $a$ is a data-drift signal even when the code still runs. This is why
pipeline math is inseparable from logging, manifests, and audit slices.

For LLM work, the token-weighted view is often more important than the document-weighted
view. A filter that removes 5 percent of documents may remove 30 percent of tokens if it
targets long documents. The corresponding token acceptance rate is

$$
a_{\mathrm{tok}} = \frac{\sum_i f(r_i)\,T_i}{\sum_i T_i},
$$

where $T_i$ is the token count or a deterministic token-count estimate. The distinction
matters for compute budgets, mixture proportions, and scaling-law interpretation.

### 3.4 PDF/OCR preview

PDF/OCR preview is part of the canonical scope of jsonl generation. We model the
relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level
metadata $m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `quarantine`, the invariant should be explicit enough that a checker can fail fast.
If the invariant is only written in a notebook comment or an engineer's memory, it will
not protect a long-running data build.

Examples:
- A small local experiment can store this object in memory; a frontier-scale run must store it as sharded, versioned, validated records.
- The mathematical object is simple, but the operational contract must survive restarts, parallel workers, schema changes, and audits.
- The notebook for this section uses synthetic data so the same ideas can be executed without external files.

Non-examples:
- A path on disk without a manifest is not a reproducible dataset.
- A metric dashboard without record-level lineage is not a provenance system.
- A filter threshold without an audit sample is not evidence of quality.

Implementation consequence: every transformation should report both a count and a rate.
If $n_{\mathrm{in}}$ records enter the stage and $n_{\mathrm{out}}$ records leave, the
acceptance rate is

$$
a = \frac{n_{\mathrm{out}}}{n_{\mathrm{in}}}.
$$

A sudden change in $a$ is a data-drift signal even when the code still runs. This is why
pipeline math is inseparable from logging, manifests, and audit slices.

For LLM work, the token-weighted view is often more important than the document-weighted
view. A filter that removes 5 percent of documents may remove 30 percent of tokens if it
targets long documents. The corresponding token acceptance rate is

$$
a_{\mathrm{tok}} = \frac{\sum_i f(r_i)\,T_i}{\sum_i T_i},
$$

where $T_i$ is the token count or a deterministic token-count estimate. The distinction
matters for compute budgets, mixture proportions, and scaling-law interpretation.

### 3.5 Document boundaries

Document boundaries is part of the canonical scope of jsonl generation. We model the
relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level
metadata $m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `idempotence`, the invariant should be explicit enough that a checker can fail fast.
If the invariant is only written in a notebook comment or an engineer's memory, it will
not protect a long-running data build.

Examples:
- A small local experiment can store this object in memory; a frontier-scale run must store it as sharded, versioned, validated records.
- The mathematical object is simple, but the operational contract must survive restarts, parallel workers, schema changes, and audits.
- The notebook for this section uses synthetic data so the same ideas can be executed without external files.

Non-examples:
- A path on disk without a manifest is not a reproducible dataset.
- A metric dashboard without record-level lineage is not a provenance system.
- A filter threshold without an audit sample is not evidence of quality.

Implementation consequence: every transformation should report both a count and a rate.
If $n_{\mathrm{in}}$ records enter the stage and $n_{\mathrm{out}}$ records leave, the
acceptance rate is

$$
a = \frac{n_{\mathrm{out}}}{n_{\mathrm{in}}}.
$$

A sudden change in $a$ is a data-drift signal even when the code still runs. This is why
pipeline math is inseparable from logging, manifests, and audit slices.

For LLM work, the token-weighted view is often more important than the document-weighted
view. A filter that removes 5 percent of documents may remove 30 percent of tokens if it
targets long documents. The corresponding token acceptance rate is

$$
a_{\mathrm{tok}} = \frac{\sum_i f(r_i)\,T_i}{\sum_i T_i},
$$

where $T_i$ is the token count or a deterministic token-count estimate. The distinction
matters for compute budgets, mixture proportions, and scaling-law interpretation.

## 4. Record Construction

Record Construction gives the conceptual and mathematical layer for jsonl generation.
The local variables in this section should be read as pipeline objects: documents,
records, tokens, filters, weights, shards, and manifests.

### 4.1 Field mapping

Field mapping is part of the canonical scope of jsonl generation. We model the relevant
object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level metadata
$m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `generator`, the invariant should be explicit enough that a checker can fail fast.
If the invariant is only written in a notebook comment or an engineer's memory, it will
not protect a long-running data build.

Examples:
- A small local experiment can store this object in memory; a frontier-scale run must store it as sharded, versioned, validated records.
- The mathematical object is simple, but the operational contract must survive restarts, parallel workers, schema changes, and audits.
- The notebook for this section uses synthetic data so the same ideas can be executed without external files.

Non-examples:
- A path on disk without a manifest is not a reproducible dataset.
- A metric dashboard without record-level lineage is not a provenance system.
- A filter threshold without an audit sample is not evidence of quality.

Implementation consequence: every transformation should report both a count and a rate.
If $n_{\mathrm{in}}$ records enter the stage and $n_{\mathrm{out}}$ records leave, the
acceptance rate is

$$
a = \frac{n_{\mathrm{out}}}{n_{\mathrm{in}}}.
$$

A sudden change in $a$ is a data-drift signal even when the code still runs. This is why
pipeline math is inseparable from logging, manifests, and audit slices.

For LLM work, the token-weighted view is often more important than the document-weighted
view. A filter that removes 5 percent of documents may remove 30 percent of tokens if it
targets long documents. The corresponding token acceptance rate is

$$
a_{\mathrm{tok}} = \frac{\sum_i f(r_i)\,T_i}{\sum_i T_i},
$$

where $T_i$ is the token count or a deterministic token-count estimate. The distinction
matters for compute budgets, mixture proportions, and scaling-law interpretation.

### 4.2 Metadata preservation

Metadata preservation is part of the canonical scope of jsonl generation. We model the
relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level
metadata $m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `serialization`, the invariant should be explicit enough that a checker can fail
fast. If the invariant is only written in a notebook comment or an engineer's memory, it
will not protect a long-running data build.

Examples:
- A small local experiment can store this object in memory; a frontier-scale run must store it as sharded, versioned, validated records.
- The mathematical object is simple, but the operational contract must survive restarts, parallel workers, schema changes, and audits.
- The notebook for this section uses synthetic data so the same ideas can be executed without external files.

Non-examples:
- A path on disk without a manifest is not a reproducible dataset.
- A metric dashboard without record-level lineage is not a provenance system.
- A filter threshold without an audit sample is not evidence of quality.

Implementation consequence: every transformation should report both a count and a rate.
If $n_{\mathrm{in}}$ records enter the stage and $n_{\mathrm{out}}$ records leave, the
acceptance rate is

$$
a = \frac{n_{\mathrm{out}}}{n_{\mathrm{in}}}.
$$

A sudden change in $a$ is a data-drift signal even when the code still runs. This is why
pipeline math is inseparable from logging, manifests, and audit slices.

For LLM work, the token-weighted view is often more important than the document-weighted
view. A filter that removes 5 percent of documents may remove 30 percent of tokens if it
targets long documents. The corresponding token acceptance rate is

$$
a_{\mathrm{tok}} = \frac{\sum_i f(r_i)\,T_i}{\sum_i T_i},
$$

where $T_i$ is the token count or a deterministic token-count estimate. The distinction
matters for compute budgets, mixture proportions, and scaling-law interpretation.

### 4.3 Token count estimates

Token count estimates is part of the canonical scope of jsonl generation. We model the
relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level
metadata $m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `shard`, the invariant should be explicit enough that a checker can fail fast. If
the invariant is only written in a notebook comment or an engineer's memory, it will not
protect a long-running data build.

Examples:
- A small local experiment can store this object in memory; a frontier-scale run must store it as sharded, versioned, validated records.
- The mathematical object is simple, but the operational contract must survive restarts, parallel workers, schema changes, and audits.
- The notebook for this section uses synthetic data so the same ideas can be executed without external files.

Non-examples:
- A path on disk without a manifest is not a reproducible dataset.
- A metric dashboard without record-level lineage is not a provenance system.
- A filter threshold without an audit sample is not evidence of quality.

Implementation consequence: every transformation should report both a count and a rate.
If $n_{\mathrm{in}}$ records enter the stage and $n_{\mathrm{out}}$ records leave, the
acceptance rate is

$$
a = \frac{n_{\mathrm{out}}}{n_{\mathrm{in}}}.
$$

A sudden change in $a$ is a data-drift signal even when the code still runs. This is why
pipeline math is inseparable from logging, manifests, and audit slices.

For LLM work, the token-weighted view is often more important than the document-weighted
view. A filter that removes 5 percent of documents may remove 30 percent of tokens if it
targets long documents. The corresponding token acceptance rate is

$$
a_{\mathrm{tok}} = \frac{\sum_i f(r_i)\,T_i}{\sum_i T_i},
$$

where $T_i$ is the token count or a deterministic token-count estimate. The distinction
matters for compute budgets, mixture proportions, and scaling-law interpretation.

### 4.4 Source trace

Source trace is part of the canonical scope of jsonl generation. We model the relevant
object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level metadata
$m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `quarantine`, the invariant should be explicit enough that a checker can fail fast.
If the invariant is only written in a notebook comment or an engineer's memory, it will
not protect a long-running data build.

Examples:
- A small local experiment can store this object in memory; a frontier-scale run must store it as sharded, versioned, validated records.
- The mathematical object is simple, but the operational contract must survive restarts, parallel workers, schema changes, and audits.
- The notebook for this section uses synthetic data so the same ideas can be executed without external files.

Non-examples:
- A path on disk without a manifest is not a reproducible dataset.
- A metric dashboard without record-level lineage is not a provenance system.
- A filter threshold without an audit sample is not evidence of quality.

Implementation consequence: every transformation should report both a count and a rate.
If $n_{\mathrm{in}}$ records enter the stage and $n_{\mathrm{out}}$ records leave, the
acceptance rate is

$$
a = \frac{n_{\mathrm{out}}}{n_{\mathrm{in}}}.
$$

A sudden change in $a$ is a data-drift signal even when the code still runs. This is why
pipeline math is inseparable from logging, manifests, and audit slices.

For LLM work, the token-weighted view is often more important than the document-weighted
view. A filter that removes 5 percent of documents may remove 30 percent of tokens if it
targets long documents. The corresponding token acceptance rate is

$$
a_{\mathrm{tok}} = \frac{\sum_i f(r_i)\,T_i}{\sum_i T_i},
$$

where $T_i$ is the token count or a deterministic token-count estimate. The distinction
matters for compute budgets, mixture proportions, and scaling-law interpretation.

### 4.5 Error quarantine

Error quarantine is part of the canonical scope of jsonl generation. We model the
relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level
metadata $m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `idempotence`, the invariant should be explicit enough that a checker can fail fast.
If the invariant is only written in a notebook comment or an engineer's memory, it will
not protect a long-running data build.

Examples:
- A small local experiment can store this object in memory; a frontier-scale run must store it as sharded, versioned, validated records.
- The mathematical object is simple, but the operational contract must survive restarts, parallel workers, schema changes, and audits.
- The notebook for this section uses synthetic data so the same ideas can be executed without external files.

Non-examples:
- A path on disk without a manifest is not a reproducible dataset.
- A metric dashboard without record-level lineage is not a provenance system.
- A filter threshold without an audit sample is not evidence of quality.

Implementation consequence: every transformation should report both a count and a rate.
If $n_{\mathrm{in}}$ records enter the stage and $n_{\mathrm{out}}$ records leave, the
acceptance rate is

$$
a = \frac{n_{\mathrm{out}}}{n_{\mathrm{in}}}.
$$

A sudden change in $a$ is a data-drift signal even when the code still runs. This is why
pipeline math is inseparable from logging, manifests, and audit slices.

For LLM work, the token-weighted view is often more important than the document-weighted
view. A filter that removes 5 percent of documents may remove 30 percent of tokens if it
targets long documents. The corresponding token acceptance rate is

$$
a_{\mathrm{tok}} = \frac{\sum_i f(r_i)\,T_i}{\sum_i T_i},
$$

where $T_i$ is the token count or a deterministic token-count estimate. The distinction
matters for compute budgets, mixture proportions, and scaling-law interpretation.

## 5. Streaming and Sharding

Streaming and Sharding gives the conceptual and mathematical layer for jsonl generation.
The local variables in this section should be read as pipeline objects: documents,
records, tokens, filters, weights, shards, and manifests.

### 5.1 Python generators

Python generators is part of the canonical scope of jsonl generation. We model the
relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level
metadata $m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `generator`, the invariant should be explicit enough that a checker can fail fast.
If the invariant is only written in a notebook comment or an engineer's memory, it will
not protect a long-running data build.

Examples:
- A small local experiment can store this object in memory; a frontier-scale run must store it as sharded, versioned, validated records.
- The mathematical object is simple, but the operational contract must survive restarts, parallel workers, schema changes, and audits.
- The notebook for this section uses synthetic data so the same ideas can be executed without external files.

Non-examples:
- A path on disk without a manifest is not a reproducible dataset.
- A metric dashboard without record-level lineage is not a provenance system.
- A filter threshold without an audit sample is not evidence of quality.

Implementation consequence: every transformation should report both a count and a rate.
If $n_{\mathrm{in}}$ records enter the stage and $n_{\mathrm{out}}$ records leave, the
acceptance rate is

$$
a = \frac{n_{\mathrm{out}}}{n_{\mathrm{in}}}.
$$

A sudden change in $a$ is a data-drift signal even when the code still runs. This is why
pipeline math is inseparable from logging, manifests, and audit slices.

For LLM work, the token-weighted view is often more important than the document-weighted
view. A filter that removes 5 percent of documents may remove 30 percent of tokens if it
targets long documents. The corresponding token acceptance rate is

$$
a_{\mathrm{tok}} = \frac{\sum_i f(r_i)\,T_i}{\sum_i T_i},
$$

where $T_i$ is the token count or a deterministic token-count estimate. The distinction
matters for compute budgets, mixture proportions, and scaling-law interpretation.

### 5.2 Memory-safe iteration

Memory-safe iteration is part of the canonical scope of jsonl generation. We model the
relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level
metadata $m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `serialization`, the invariant should be explicit enough that a checker can fail
fast. If the invariant is only written in a notebook comment or an engineer's memory, it
will not protect a long-running data build.

Examples:
- A small local experiment can store this object in memory; a frontier-scale run must store it as sharded, versioned, validated records.
- The mathematical object is simple, but the operational contract must survive restarts, parallel workers, schema changes, and audits.
- The notebook for this section uses synthetic data so the same ideas can be executed without external files.

Non-examples:
- A path on disk without a manifest is not a reproducible dataset.
- A metric dashboard without record-level lineage is not a provenance system.
- A filter threshold without an audit sample is not evidence of quality.

Implementation consequence: every transformation should report both a count and a rate.
If $n_{\mathrm{in}}$ records enter the stage and $n_{\mathrm{out}}$ records leave, the
acceptance rate is

$$
a = \frac{n_{\mathrm{out}}}{n_{\mathrm{in}}}.
$$

A sudden change in $a$ is a data-drift signal even when the code still runs. This is why
pipeline math is inseparable from logging, manifests, and audit slices.

For LLM work, the token-weighted view is often more important than the document-weighted
view. A filter that removes 5 percent of documents may remove 30 percent of tokens if it
targets long documents. The corresponding token acceptance rate is

$$
a_{\mathrm{tok}} = \frac{\sum_i f(r_i)\,T_i}{\sum_i T_i},
$$

where $T_i$ is the token count or a deterministic token-count estimate. The distinction
matters for compute budgets, mixture proportions, and scaling-law interpretation.

### 5.3 Shard rotation

Shard rotation is part of the canonical scope of jsonl generation. We model the relevant
object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level metadata
$m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `shard`, the invariant should be explicit enough that a checker can fail fast. If
the invariant is only written in a notebook comment or an engineer's memory, it will not
protect a long-running data build.

Examples:
- A small local experiment can store this object in memory; a frontier-scale run must store it as sharded, versioned, validated records.
- The mathematical object is simple, but the operational contract must survive restarts, parallel workers, schema changes, and audits.
- The notebook for this section uses synthetic data so the same ideas can be executed without external files.

Non-examples:
- A path on disk without a manifest is not a reproducible dataset.
- A metric dashboard without record-level lineage is not a provenance system.
- A filter threshold without an audit sample is not evidence of quality.

Implementation consequence: every transformation should report both a count and a rate.
If $n_{\mathrm{in}}$ records enter the stage and $n_{\mathrm{out}}$ records leave, the
acceptance rate is

$$
a = \frac{n_{\mathrm{out}}}{n_{\mathrm{in}}}.
$$

A sudden change in $a$ is a data-drift signal even when the code still runs. This is why
pipeline math is inseparable from logging, manifests, and audit slices.

For LLM work, the token-weighted view is often more important than the document-weighted
view. A filter that removes 5 percent of documents may remove 30 percent of tokens if it
targets long documents. The corresponding token acceptance rate is

$$
a_{\mathrm{tok}} = \frac{\sum_i f(r_i)\,T_i}{\sum_i T_i},
$$

where $T_i$ is the token count or a deterministic token-count estimate. The distinction
matters for compute budgets, mixture proportions, and scaling-law interpretation.

### 5.4 Compression

Compression is part of the canonical scope of jsonl generation. We model the relevant
object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level metadata
$m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `quarantine`, the invariant should be explicit enough that a checker can fail fast.
If the invariant is only written in a notebook comment or an engineer's memory, it will
not protect a long-running data build.

Examples:
- A small local experiment can store this object in memory; a frontier-scale run must store it as sharded, versioned, validated records.
- The mathematical object is simple, but the operational contract must survive restarts, parallel workers, schema changes, and audits.
- The notebook for this section uses synthetic data so the same ideas can be executed without external files.

Non-examples:
- A path on disk without a manifest is not a reproducible dataset.
- A metric dashboard without record-level lineage is not a provenance system.
- A filter threshold without an audit sample is not evidence of quality.

Implementation consequence: every transformation should report both a count and a rate.
If $n_{\mathrm{in}}$ records enter the stage and $n_{\mathrm{out}}$ records leave, the
acceptance rate is

$$
a = \frac{n_{\mathrm{out}}}{n_{\mathrm{in}}}.
$$

A sudden change in $a$ is a data-drift signal even when the code still runs. This is why
pipeline math is inseparable from logging, manifests, and audit slices.

For LLM work, the token-weighted view is often more important than the document-weighted
view. A filter that removes 5 percent of documents may remove 30 percent of tokens if it
targets long documents. The corresponding token acceptance rate is

$$
a_{\mathrm{tok}} = \frac{\sum_i f(r_i)\,T_i}{\sum_i T_i},
$$

where $T_i$ is the token count or a deterministic token-count estimate. The distinction
matters for compute budgets, mixture proportions, and scaling-law interpretation.

### 5.5 Resume/restart logic

Resume/restart logic is part of the canonical scope of jsonl generation. We model the
relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level
metadata $m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `idempotence`, the invariant should be explicit enough that a checker can fail fast.
If the invariant is only written in a notebook comment or an engineer's memory, it will
not protect a long-running data build.

Examples:
- A small local experiment can store this object in memory; a frontier-scale run must store it as sharded, versioned, validated records.
- The mathematical object is simple, but the operational contract must survive restarts, parallel workers, schema changes, and audits.
- The notebook for this section uses synthetic data so the same ideas can be executed without external files.

Non-examples:
- A path on disk without a manifest is not a reproducible dataset.
- A metric dashboard without record-level lineage is not a provenance system.
- A filter threshold without an audit sample is not evidence of quality.

Implementation consequence: every transformation should report both a count and a rate.
If $n_{\mathrm{in}}$ records enter the stage and $n_{\mathrm{out}}$ records leave, the
acceptance rate is

$$
a = \frac{n_{\mathrm{out}}}{n_{\mathrm{in}}}.
$$

A sudden change in $a$ is a data-drift signal even when the code still runs. This is why
pipeline math is inseparable from logging, manifests, and audit slices.

For LLM work, the token-weighted view is often more important than the document-weighted
view. A filter that removes 5 percent of documents may remove 30 percent of tokens if it
targets long documents. The corresponding token acceptance rate is

$$
a_{\mathrm{tok}} = \frac{\sum_i f(r_i)\,T_i}{\sum_i T_i},
$$

where $T_i$ is the token count or a deterministic token-count estimate. The distinction
matters for compute budgets, mixture proportions, and scaling-law interpretation.

## 6. Validation and Performance

Validation and Performance gives the conceptual and mathematical layer for jsonl
generation. The local variables in this section should be read as pipeline objects:
documents, records, tokens, filters, weights, shards, and manifests.

### 6.1 Line-level JSON parse

Line-level JSON parse is part of the canonical scope of jsonl generation. We model the
relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level
metadata $m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `generator`, the invariant should be explicit enough that a checker can fail fast.
If the invariant is only written in a notebook comment or an engineer's memory, it will
not protect a long-running data build.

Examples:
- A small local experiment can store this object in memory; a frontier-scale run must store it as sharded, versioned, validated records.
- The mathematical object is simple, but the operational contract must survive restarts, parallel workers, schema changes, and audits.
- The notebook for this section uses synthetic data so the same ideas can be executed without external files.

Non-examples:
- A path on disk without a manifest is not a reproducible dataset.
- A metric dashboard without record-level lineage is not a provenance system.
- A filter threshold without an audit sample is not evidence of quality.

Implementation consequence: every transformation should report both a count and a rate.
If $n_{\mathrm{in}}$ records enter the stage and $n_{\mathrm{out}}$ records leave, the
acceptance rate is

$$
a = \frac{n_{\mathrm{out}}}{n_{\mathrm{in}}}.
$$

A sudden change in $a$ is a data-drift signal even when the code still runs. This is why
pipeline math is inseparable from logging, manifests, and audit slices.

For LLM work, the token-weighted view is often more important than the document-weighted
view. A filter that removes 5 percent of documents may remove 30 percent of tokens if it
targets long documents. The corresponding token acceptance rate is

$$
a_{\mathrm{tok}} = \frac{\sum_i f(r_i)\,T_i}{\sum_i T_i},
$$

where $T_i$ is the token count or a deterministic token-count estimate. The distinction
matters for compute budgets, mixture proportions, and scaling-law interpretation.

### 6.2 Duplicate ID detection

Duplicate ID detection is part of the canonical scope of jsonl generation. We model the
relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level
metadata $m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `serialization`, the invariant should be explicit enough that a checker can fail
fast. If the invariant is only written in a notebook comment or an engineer's memory, it
will not protect a long-running data build.

Examples:
- A small local experiment can store this object in memory; a frontier-scale run must store it as sharded, versioned, validated records.
- The mathematical object is simple, but the operational contract must survive restarts, parallel workers, schema changes, and audits.
- The notebook for this section uses synthetic data so the same ideas can be executed without external files.

Non-examples:
- A path on disk without a manifest is not a reproducible dataset.
- A metric dashboard without record-level lineage is not a provenance system.
- A filter threshold without an audit sample is not evidence of quality.

Implementation consequence: every transformation should report both a count and a rate.
If $n_{\mathrm{in}}$ records enter the stage and $n_{\mathrm{out}}$ records leave, the
acceptance rate is

$$
a = \frac{n_{\mathrm{out}}}{n_{\mathrm{in}}}.
$$

A sudden change in $a$ is a data-drift signal even when the code still runs. This is why
pipeline math is inseparable from logging, manifests, and audit slices.

For LLM work, the token-weighted view is often more important than the document-weighted
view. A filter that removes 5 percent of documents may remove 30 percent of tokens if it
targets long documents. The corresponding token acceptance rate is

$$
a_{\mathrm{tok}} = \frac{\sum_i f(r_i)\,T_i}{\sum_i T_i},
$$

where $T_i$ is the token count or a deterministic token-count estimate. The distinction
matters for compute budgets, mixture proportions, and scaling-law interpretation.

### 6.3 Throughput metrics

Throughput metrics is part of the canonical scope of jsonl generation. We model the
relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level
metadata $m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `shard`, the invariant should be explicit enough that a checker can fail fast. If
the invariant is only written in a notebook comment or an engineer's memory, it will not
protect a long-running data build.

Examples:
- A small local experiment can store this object in memory; a frontier-scale run must store it as sharded, versioned, validated records.
- The mathematical object is simple, but the operational contract must survive restarts, parallel workers, schema changes, and audits.
- The notebook for this section uses synthetic data so the same ideas can be executed without external files.

Non-examples:
- A path on disk without a manifest is not a reproducible dataset.
- A metric dashboard without record-level lineage is not a provenance system.
- A filter threshold without an audit sample is not evidence of quality.

Implementation consequence: every transformation should report both a count and a rate.
If $n_{\mathrm{in}}$ records enter the stage and $n_{\mathrm{out}}$ records leave, the
acceptance rate is

$$
a = \frac{n_{\mathrm{out}}}{n_{\mathrm{in}}}.
$$

A sudden change in $a$ is a data-drift signal even when the code still runs. This is why
pipeline math is inseparable from logging, manifests, and audit slices.

For LLM work, the token-weighted view is often more important than the document-weighted
view. A filter that removes 5 percent of documents may remove 30 percent of tokens if it
targets long documents. The corresponding token acceptance rate is

$$
a_{\mathrm{tok}} = \frac{\sum_i f(r_i)\,T_i}{\sum_i T_i},
$$

where $T_i$ is the token count or a deterministic token-count estimate. The distinction
matters for compute budgets, mixture proportions, and scaling-law interpretation.

### 6.4 Multiprocessing

Multiprocessing is part of the canonical scope of jsonl generation. We model the
relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level
metadata $m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `quarantine`, the invariant should be explicit enough that a checker can fail fast.
If the invariant is only written in a notebook comment or an engineer's memory, it will
not protect a long-running data build.

Examples:
- A small local experiment can store this object in memory; a frontier-scale run must store it as sharded, versioned, validated records.
- The mathematical object is simple, but the operational contract must survive restarts, parallel workers, schema changes, and audits.
- The notebook for this section uses synthetic data so the same ideas can be executed without external files.

Non-examples:
- A path on disk without a manifest is not a reproducible dataset.
- A metric dashboard without record-level lineage is not a provenance system.
- A filter threshold without an audit sample is not evidence of quality.

Implementation consequence: every transformation should report both a count and a rate.
If $n_{\mathrm{in}}$ records enter the stage and $n_{\mathrm{out}}$ records leave, the
acceptance rate is

$$
a = \frac{n_{\mathrm{out}}}{n_{\mathrm{in}}}.
$$

A sudden change in $a$ is a data-drift signal even when the code still runs. This is why
pipeline math is inseparable from logging, manifests, and audit slices.

For LLM work, the token-weighted view is often more important than the document-weighted
view. A filter that removes 5 percent of documents may remove 30 percent of tokens if it
targets long documents. The corresponding token acceptance rate is

$$
a_{\mathrm{tok}} = \frac{\sum_i f(r_i)\,T_i}{\sum_i T_i},
$$

where $T_i$ is the token count or a deterministic token-count estimate. The distinction
matters for compute budgets, mixture proportions, and scaling-law interpretation.

### 6.5 Deterministic tests

Deterministic tests is part of the canonical scope of jsonl generation. We model the
relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level
metadata $m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `idempotence`, the invariant should be explicit enough that a checker can fail fast.
If the invariant is only written in a notebook comment or an engineer's memory, it will
not protect a long-running data build.

Examples:
- A small local experiment can store this object in memory; a frontier-scale run must store it as sharded, versioned, validated records.
- The mathematical object is simple, but the operational contract must survive restarts, parallel workers, schema changes, and audits.
- The notebook for this section uses synthetic data so the same ideas can be executed without external files.

Non-examples:
- A path on disk without a manifest is not a reproducible dataset.
- A metric dashboard without record-level lineage is not a provenance system.
- A filter threshold without an audit sample is not evidence of quality.

Implementation consequence: every transformation should report both a count and a rate.
If $n_{\mathrm{in}}$ records enter the stage and $n_{\mathrm{out}}$ records leave, the
acceptance rate is

$$
a = \frac{n_{\mathrm{out}}}{n_{\mathrm{in}}}.
$$

A sudden change in $a$ is a data-drift signal even when the code still runs. This is why
pipeline math is inseparable from logging, manifests, and audit slices.

For LLM work, the token-weighted view is often more important than the document-weighted
view. A filter that removes 5 percent of documents may remove 30 percent of tokens if it
targets long documents. The corresponding token acceptance rate is

$$
a_{\mathrm{tok}} = \frac{\sum_i f(r_i)\,T_i}{\sum_i T_i},
$$

where $T_i$ is the token count or a deterministic token-count estimate. The distinction
matters for compute budgets, mixture proportions, and scaling-law interpretation.

## 7. Common Mistakes

| # | Mistake | Why It Is Wrong | Fix |
| --- | --- | --- | --- |
| 1 | Trusting a file because it exists | A zero-byte or unparsable artifact can still pass a loose path check | Validate content and parseability |
| 2 | Counting documents but not tokens | Long documents dominate compute | Report both document and token rates |
| 3 | Changing schemas without versioning | Old and new records become indistinguishable | Pin schema versions in every record |
| 4 | Dropping metadata during transforms | Audits and removals become impossible | Preserve source and transform lineage |
| 5 | Using nondeterministic ordering | Rebuilds cannot be compared | Seed and record ordering rules |
| 6 | Ignoring failed records | Silent loss can bias the corpus | Quarantine and summarize failures |
| 7 | Treating filters as neutral | Filters encode preferences and tradeoffs | Ablate and audit every major filter |
| 8 | Mixing train and eval sources | Evaluation becomes contaminated | Run overlap audits before release |
| 9 | Optimizing one aggregate score | Small domains can regress | Track slice metrics |
| 10 | Skipping data cards | Users cannot judge intended use or risk | Publish structured documentation |
| 11 | Assuming licenses are uniform | Source terms can conflict | Track license at source and record level |
| 12 | Forgetting reproducible manifests | The same name can refer to different data | Use hashes and version pins |

## 8. Exercises

1. (*) Build a synthetic `generator` example, compute its validation signal, and explain which downstream stage would fail if the signal were wrong.
2. (*) Build a synthetic `serialization` example, compute its validation signal, and explain which downstream stage would fail if the signal were wrong.
3. (*) Build a synthetic `shard` example, compute its validation signal, and explain which downstream stage would fail if the signal were wrong.
4. (**) Build a synthetic `quarantine` example, compute its validation signal, and explain which downstream stage would fail if the signal were wrong.
5. (**) Build a synthetic `idempotence` example, compute its validation signal, and explain which downstream stage would fail if the signal were wrong.
6. (**) Build a synthetic `throughput` example, compute its validation signal, and explain which downstream stage would fail if the signal were wrong.
7. (**) Build a synthetic `resume` example, compute its validation signal, and explain which downstream stage would fail if the signal were wrong.
8. (***) Build a synthetic `generator` example, compute its validation signal, and explain which downstream stage would fail if the signal were wrong.
9. (***) Build a synthetic `serialization` example, compute its validation signal, and explain which downstream stage would fail if the signal were wrong.
10. (***) Build a synthetic `shard` example, compute its validation signal, and explain which downstream stage would fail if the signal were wrong.

## 9. Why This Matters for AI

| Concept | AI impact |
| --- | --- |
| generator | Controls what examples, gradients, risks, or audits the model pipeline can represent |
| serialization | Controls what examples, gradients, risks, or audits the model pipeline can represent |
| shard | Controls what examples, gradients, risks, or audits the model pipeline can represent |
| quarantine | Controls what examples, gradients, risks, or audits the model pipeline can represent |
| idempotence | Controls what examples, gradients, risks, or audits the model pipeline can represent |
| throughput | Controls what examples, gradients, risks, or audits the model pipeline can represent |
| resume | Controls what examples, gradients, risks, or audits the model pipeline can represent |

Data pipeline quality is model quality in delayed form. The model eventually converts these records into gradients; any unresolved ambiguity becomes either wasted compute, misleading evaluation, memorization risk, or irreproducible science.

## 10. Conceptual Bridge

This section connects the previous and next pieces of the curriculum as follows:

```text
raw sources -> records -> validation -> assembly -> audits -> documentation -> mixture
```

The next section is [Quality Checks](../03-Quality-Checks/notes.md). It uses the
contracts established here and moves one step further through the LLM data pipeline.

## References

- [DataComp-LM](https://arxiv.org/abs/2406.11794)
- [FineWeb](https://arxiv.org/abs/2406.17557)
- [Dolma](https://arxiv.org/abs/2402.00159)
- [WIMBD](https://arxiv.org/abs/2310.20707)
- [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499)
- [DoReMi](https://arxiv.org/abs/2305.10429)
- [Data Cards](https://arxiv.org/abs/2204.01075)
- [A Pretrainer's Guide to Training Data](https://arxiv.org/abs/2305.13169)
- [Hugging Face Datasets loading docs](https://huggingface.co/docs/datasets/loading)
