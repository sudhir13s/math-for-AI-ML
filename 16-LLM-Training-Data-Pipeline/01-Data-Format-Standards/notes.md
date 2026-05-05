[Back to Curriculum](../../README.md) | [Previous: Serving and Systems Tradeoffs](../../15-Math-for-LLMs/13-Serving-and-Systems-Tradeoffs/notes.md) | [Next: JSONL Generation](../02-JSONL-Generation/notes.md)

---

# Data Format Standards

> _"A training record is a small object with a large blast radius."_

## Overview

Data format standards define the mathematical and engineering contract between raw
examples and the training loop. In an LLM training run, data is not an inert pile of
text; it is the empirical distribution that defines the examples, losses, risks, and
capabilities the model will see.

This section is written as LaTeX Markdown. Inline mathematics uses `$...$`, and display
equations use `$$...$$`. The goal is to connect data engineering decisions to
mathematical objects such as records $r_i$, token sequences $x_{1:T}$, filters $f(x)$,
hashes $h(x)$, mixture weights $\boldsymbol{\alpha}$, and empirical expectations.

The scope is deliberately narrow: this chapter owns the training-data pipeline.
Tokenizer design, GPU training systems, benchmark methodology, alignment objectives, and
production MLOps each have their own canonical chapters. Here we study the data objects
that those later systems consume.

## Prerequisites

- [Tokenization Math](../../15-Math-for-LLMs/01-Tokenization-Math/notes.md)
- [Language Model Probability](../../15-Math-for-LLMs/05-Language-Model-Probability/notes.md)
- [Training at Scale](../../15-Math-for-LLMs/06-Training-at-Scale/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Executable demonstrations for data format standards |
| [exercises.ipynb](exercises.ipynb) | Graded practice for data format standards |

## Learning Objectives

After completing this section, you will be able to:

- Define records, schemas, token streams, shards, and provenance identifiers
- Distinguish raw documents, pretraining records, SFT messages, and preference pairs
- Validate JSONL-style examples with deterministic type and key checks
- Explain when JSONL, Parquet, Arrow, or tokenized binary formats are appropriate
- Use stable hashes to identify records and preserve reproducibility
- Design metadata fields for source, license, language, quality, and split information
- Connect schema design to downstream loss computation and evaluation isolation
- Recognize format errors that silently change the training objective

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 Data as a training contract](#11-data-as-a-training-contract)
  - [1.2 Records vs documents vs token streams](#12-records-vs-documents-vs-token-streams)
  - [1.3 Why format bugs become model bugs](#13-why-format-bugs-become-model-bugs)
  - [1.4 Pretraining, SFT, and preference formats](#14-pretraining-sft-and-preference-formats)
  - [1.5 Pipeline history from raw web text to curated corpora](#15-pipeline-history-from-raw-web-text-to-curated-corpora)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Record $r_i$](#21-record-ri)
  - [2.2 Schema $\mathcal{S}$](#22-schema-mathcals)
  - [2.3 Text field and metadata field](#23-text-field-and-metadata-field)
  - [2.4 Token sequence $x_{1:T}$](#24-token-sequence-x1t)
  - [2.5 Source, split, shard, and provenance identifiers](#25-source-split-shard-and-provenance-identifiers)
- [3. Canonical Schemas](#3-canonical-schemas)
  - [3.1 Raw text document schema](#31-raw-text-document-schema)
  - [3.2 Pretraining document schema](#32-pretraining-document-schema)
  - [3.3 Chat/SFT messages schema](#33-chatsft-messages-schema)
  - [3.4 Pairwise preference schema](#34-pairwise-preference-schema)
  - [3.5 Evaluation-holdout schema](#35-evaluationholdout-schema)
- [4. Storage Formats](#4-storage-formats)
  - [4.1 JSONL](#41-jsonl)
  - [4.2 Parquet and Arrow](#42-parquet-and-arrow)
  - [4.3 Tokenized binary arrays](#43-tokenized-binary-arrays)
  - [4.4 Sharded compressed files](#44-sharded-compressed-files)
  - [4.5 Manifest files and checksums](#45-manifest-files-and-checksums)
- [5. Validation Rules](#5-validation-rules)
  - [5.1 Required keys](#51-required-keys)
  - [5.2 Type validation](#52-type-validation)
  - [5.3 Unicode and whitespace normalization](#53-unicode-and-whitespace-normalization)
  - [5.4 Text-length constraints](#54-textlength-constraints)
  - [5.5 Deterministic IDs and hashes](#55-deterministic-ids-and-hashes)
- [6. Applications](#6-applications)
  - [6.1 Pretraining corpora](#61-pretraining-corpora)
  - [6.2 Continual pretraining](#62-continual-pretraining)
  - [6.3 SFT](#63-sft)
  - [6.4 Preference data](#64-preference-data)
  - [6.5 Data release packages](#65-data-release-packages)
- [7. Common Mistakes](#7-common-mistakes)
- [8. Exercises](#8-exercises)
- [9. Why This Matters for AI](#9-why-this-matters-for-ai)
- [10. Conceptual Bridge](#10-conceptual-bridge)
- [References](#references)

---

## 1. Intuition

Intuition gives the conceptual and mathematical layer for data format standards. The
local variables in this section should be read as pipeline objects: documents, records,
tokens, filters, weights, shards, and manifests.

### 1.1 Data as a training contract

Data as a training contract is part of the canonical scope of data format standards. We
model the relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with
record-level metadata $m_i$ and text or token content $x_i$. The practical question is
whether the transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `record`, the invariant should be explicit enough that a checker can fail fast. If
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

### 1.2 Records vs documents vs token streams

Records vs documents vs token streams is part of the canonical scope of data format
standards. We model the relevant object as a finite collection $\mathcal{D} =
\{r_i\}_{i=1}^n$ with record-level metadata $m_i$ and text or token content $x_i$. The
practical question is whether the transformation preserves the intended empirical
distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `schema`, the invariant should be explicit enough that a checker can fail fast. If
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

### 1.3 Why format bugs become model bugs

Why format bugs become model bugs is part of the canonical scope of data format
standards. We model the relevant object as a finite collection $\mathcal{D} =
\{r_i\}_{i=1}^n$ with record-level metadata $m_i$ and text or token content $x_i$. The
practical question is whether the transformation preserves the intended empirical
distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `JSONL`, the invariant should be explicit enough that a checker can fail fast. If
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

### 1.4 Pretraining, SFT, and preference formats

Pretraining, SFT, and preference formats is part of the canonical scope of data format
standards. We model the relevant object as a finite collection $\mathcal{D} =
\{r_i\}_{i=1}^n$ with record-level metadata $m_i$ and text or token content $x_i$. The
practical question is whether the transformation preserves the intended empirical
distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `metadata`, the invariant should be explicit enough that a checker can fail fast. If
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

### 1.5 Pipeline history from raw web text to curated corpora

Pipeline history from raw web text to curated corpora is part of the canonical scope of
data format standards. We model the relevant object as a finite collection $\mathcal{D}
= \{r_i\}_{i=1}^n$ with record-level metadata $m_i$ and text or token content $x_i$. The
practical question is whether the transformation preserves the intended empirical
distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `provenance`, the invariant should be explicit enough that a checker can fail fast.
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

Formal Definitions gives the conceptual and mathematical layer for data format
standards. The local variables in this section should be read as pipeline objects:
documents, records, tokens, filters, weights, shards, and manifests.

### 2.1 Record $r_i$

Record $r_i$ is part of the canonical scope of data format standards. We model the
relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level
metadata $m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `record`, the invariant should be explicit enough that a checker can fail fast. If
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

### 2.2 Schema $\mathcal{S}$

Schema $\mathcal{S}$ is part of the canonical scope of data format standards. We model
the relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-
level metadata $m_i$ and text or token content $x_i$. The practical question is whether
the transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `schema`, the invariant should be explicit enough that a checker can fail fast. If
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

### 2.3 Text field and metadata field

Text field and metadata field is part of the canonical scope of data format standards.
We model the relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with
record-level metadata $m_i$ and text or token content $x_i$. The practical question is
whether the transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `JSONL`, the invariant should be explicit enough that a checker can fail fast. If
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

### 2.4 Token sequence $x_{1:T}$

Token sequence $x_{1:T}$ is part of the canonical scope of data format standards. We
model the relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with
record-level metadata $m_i$ and text or token content $x_i$. The practical question is
whether the transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `metadata`, the invariant should be explicit enough that a checker can fail fast. If
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

### 2.5 Source, split, shard, and provenance identifiers

Source, split, shard, and provenance identifiers is part of the canonical scope of data
format standards. We model the relevant object as a finite collection $\mathcal{D} =
\{r_i\}_{i=1}^n$ with record-level metadata $m_i$ and text or token content $x_i$. The
practical question is whether the transformation preserves the intended empirical
distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `provenance`, the invariant should be explicit enough that a checker can fail fast.
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

## 3. Canonical Schemas

Canonical Schemas gives the conceptual and mathematical layer for data format standards.
The local variables in this section should be read as pipeline objects: documents,
records, tokens, filters, weights, shards, and manifests.

### 3.1 Raw text document schema

Raw text document schema is part of the canonical scope of data format standards. We
model the relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with
record-level metadata $m_i$ and text or token content $x_i$. The practical question is
whether the transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `record`, the invariant should be explicit enough that a checker can fail fast. If
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

### 3.2 Pretraining document schema

Pretraining document schema is part of the canonical scope of data format standards. We
model the relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with
record-level metadata $m_i$ and text or token content $x_i$. The practical question is
whether the transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `schema`, the invariant should be explicit enough that a checker can fail fast. If
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

### 3.3 Chat/SFT messages schema

Chat/SFT messages schema is part of the canonical scope of data format standards. We
model the relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with
record-level metadata $m_i$ and text or token content $x_i$. The practical question is
whether the transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `JSONL`, the invariant should be explicit enough that a checker can fail fast. If
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

### 3.4 Pairwise preference schema

Pairwise preference schema is part of the canonical scope of data format standards. We
model the relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with
record-level metadata $m_i$ and text or token content $x_i$. The practical question is
whether the transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `metadata`, the invariant should be explicit enough that a checker can fail fast. If
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

### 3.5 Evaluation-holdout schema

Evaluation-holdout schema is part of the canonical scope of data format standards. We
model the relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with
record-level metadata $m_i$ and text or token content $x_i$. The practical question is
whether the transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `provenance`, the invariant should be explicit enough that a checker can fail fast.
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

## 4. Storage Formats

Storage Formats gives the conceptual and mathematical layer for data format standards.
The local variables in this section should be read as pipeline objects: documents,
records, tokens, filters, weights, shards, and manifests.

### 4.1 JSONL

JSONL is part of the canonical scope of data format standards. We model the relevant
object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level metadata
$m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `record`, the invariant should be explicit enough that a checker can fail fast. If
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

### 4.2 Parquet and Arrow

Parquet and Arrow is part of the canonical scope of data format standards. We model the
relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level
metadata $m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `schema`, the invariant should be explicit enough that a checker can fail fast. If
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

### 4.3 Tokenized binary arrays

Tokenized binary arrays is part of the canonical scope of data format standards. We
model the relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with
record-level metadata $m_i$ and text or token content $x_i$. The practical question is
whether the transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `JSONL`, the invariant should be explicit enough that a checker can fail fast. If
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

### 4.4 Sharded compressed files

Sharded compressed files is part of the canonical scope of data format standards. We
model the relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with
record-level metadata $m_i$ and text or token content $x_i$. The practical question is
whether the transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `metadata`, the invariant should be explicit enough that a checker can fail fast. If
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

### 4.5 Manifest files and checksums

Manifest files and checksums is part of the canonical scope of data format standards. We
model the relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with
record-level metadata $m_i$ and text or token content $x_i$. The practical question is
whether the transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `provenance`, the invariant should be explicit enough that a checker can fail fast.
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

## 5. Validation Rules

Validation Rules gives the conceptual and mathematical layer for data format standards.
The local variables in this section should be read as pipeline objects: documents,
records, tokens, filters, weights, shards, and manifests.

### 5.1 Required keys

Required keys is part of the canonical scope of data format standards. We model the
relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level
metadata $m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `record`, the invariant should be explicit enough that a checker can fail fast. If
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

### 5.2 Type validation

Type validation is part of the canonical scope of data format standards. We model the
relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level
metadata $m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `schema`, the invariant should be explicit enough that a checker can fail fast. If
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

### 5.3 Unicode and whitespace normalization

Unicode and whitespace normalization is part of the canonical scope of data format
standards. We model the relevant object as a finite collection $\mathcal{D} =
\{r_i\}_{i=1}^n$ with record-level metadata $m_i$ and text or token content $x_i$. The
practical question is whether the transformation preserves the intended empirical
distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `JSONL`, the invariant should be explicit enough that a checker can fail fast. If
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

### 5.4 Text-length constraints

Text-length constraints is part of the canonical scope of data format standards. We
model the relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with
record-level metadata $m_i$ and text or token content $x_i$. The practical question is
whether the transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `metadata`, the invariant should be explicit enough that a checker can fail fast. If
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

### 5.5 Deterministic IDs and hashes

Deterministic IDs and hashes is part of the canonical scope of data format standards. We
model the relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with
record-level metadata $m_i$ and text or token content $x_i$. The practical question is
whether the transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `provenance`, the invariant should be explicit enough that a checker can fail fast.
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

## 6. Applications

Applications gives the conceptual and mathematical layer for data format standards. The
local variables in this section should be read as pipeline objects: documents, records,
tokens, filters, weights, shards, and manifests.

### 6.1 Pretraining corpora

Pretraining corpora is part of the canonical scope of data format standards. We model
the relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-
level metadata $m_i$ and text or token content $x_i$. The practical question is whether
the transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `record`, the invariant should be explicit enough that a checker can fail fast. If
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

### 6.2 Continual pretraining

Continual pretraining is part of the canonical scope of data format standards. We model
the relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-
level metadata $m_i$ and text or token content $x_i$. The practical question is whether
the transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `schema`, the invariant should be explicit enough that a checker can fail fast. If
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

### 6.3 SFT

SFT is part of the canonical scope of data format standards. We model the relevant
object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level metadata
$m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `JSONL`, the invariant should be explicit enough that a checker can fail fast. If
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

### 6.4 Preference data

Preference data is part of the canonical scope of data format standards. We model the
relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level
metadata $m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `metadata`, the invariant should be explicit enough that a checker can fail fast. If
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

### 6.5 Data release packages

Data release packages is part of the canonical scope of data format standards. We model
the relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-
level metadata $m_i$ and text or token content $x_i$. The practical question is whether
the transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `provenance`, the invariant should be explicit enough that a checker can fail fast.
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

1. (*) Build a synthetic `record` example, compute its validation signal, and explain which downstream stage would fail if the signal were wrong.
2. (*) Build a synthetic `schema` example, compute its validation signal, and explain which downstream stage would fail if the signal were wrong.
3. (*) Build a synthetic `JSONL` example, compute its validation signal, and explain which downstream stage would fail if the signal were wrong.
4. (**) Build a synthetic `metadata` example, compute its validation signal, and explain which downstream stage would fail if the signal were wrong.
5. (**) Build a synthetic `provenance` example, compute its validation signal, and explain which downstream stage would fail if the signal were wrong.
6. (**) Build a synthetic `token sequence` example, compute its validation signal, and explain which downstream stage would fail if the signal were wrong.
7. (**) Build a synthetic `manifest` example, compute its validation signal, and explain which downstream stage would fail if the signal were wrong.
8. (***) Build a synthetic `record` example, compute its validation signal, and explain which downstream stage would fail if the signal were wrong.
9. (***) Build a synthetic `schema` example, compute its validation signal, and explain which downstream stage would fail if the signal were wrong.
10. (***) Build a synthetic `JSONL` example, compute its validation signal, and explain which downstream stage would fail if the signal were wrong.

## 9. Why This Matters for AI

| Concept | AI impact |
| --- | --- |
| record | Controls what examples, gradients, risks, or audits the model pipeline can represent |
| schema | Controls what examples, gradients, risks, or audits the model pipeline can represent |
| JSONL | Controls what examples, gradients, risks, or audits the model pipeline can represent |
| metadata | Controls what examples, gradients, risks, or audits the model pipeline can represent |
| provenance | Controls what examples, gradients, risks, or audits the model pipeline can represent |
| token sequence | Controls what examples, gradients, risks, or audits the model pipeline can represent |
| manifest | Controls what examples, gradients, risks, or audits the model pipeline can represent |

Data pipeline quality is model quality in delayed form. The model eventually converts these records into gradients; any unresolved ambiguity becomes either wasted compute, misleading evaluation, memorization risk, or irreproducible science.

## 10. Conceptual Bridge

This section connects the previous and next pieces of the curriculum as follows:

```text
raw sources -> records -> validation -> assembly -> audits -> documentation -> mixture
```

The next section is [JSONL Generation](../02-JSONL-Generation/notes.md). It uses the
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
