[Back to Curriculum](../../README.md) | [Previous: Quality Checks](../03-Quality-Checks/notes.md) | [Next: Contamination and Dedup Audits](../05-Contamination-and-Dedup-Audits/notes.md)

---

# Full Dataset Assembly

> _"A corpus is not a pile of files; it is a reproducible sampling distribution over tokens."_

## Overview

Full dataset assembly combines accepted records into deterministic, balanced, token-
accounted corpora for training and validation. In an LLM training run, data is not an
inert pile of text; it is the empirical distribution that defines the examples, losses,
risks, and capabilities the model will see.

This section is written as LaTeX Markdown. Inline mathematics uses `$...$`, and display
equations use `$$...$$`. The goal is to connect data engineering decisions to
mathematical objects such as records $r_i$, token sequences $x_{1:T}$, filters $f(x)$,
hashes $h(x)$, mixture weights $\boldsymbol{\alpha}$, and empirical expectations.

The scope is deliberately narrow: this chapter owns the training-data pipeline.
Tokenizer design, GPU training systems, benchmark methodology, alignment objectives, and
production MLOps each have their own canonical chapters. Here we study the data objects
that those later systems consume.

## Prerequisites

- [Quality Checks](../03-Quality-Checks/notes.md)
- [Training at Scale](../../15-Math-for-LLMs/06-Training-at-Scale/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Executable demonstrations for full dataset assembly |
| [exercises.ipynb](exercises.ipynb) | Graded practice for full dataset assembly |

## Learning Objectives

After completing this section, you will be able to:

- Define source sets, mixture weights, token budgets, shards, and split assignments
- Build source manifests with checksums and version pins
- Implement weighted and stratified sampling over multiple data sources
- Compute token budgets and source proportions
- Create deterministic train/validation/test splits
- Explain sequence packing, document-boundary masks, and packed-shard statistics
- Verify assembled corpora through shard counts, token counts, and loader smoke tests
- Connect assembly decisions to curriculum and training stability

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 Assembly as turning accepted records into a trainable corpus](#11-assembly-as-turning-accepted-records-into-a-trainable-corpus)
  - [1.2 Source manifests](#12-source-manifests)
  - [1.3 Token accounting](#13-token-accounting)
  - [1.4 Data order as curriculum](#14-data-order-as-curriculum)
  - [1.5 Reproducibility at trillion-token scale](#15-reproducibility-at-trilliontoken-scale)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Source set $\mathcal{D}_k$](#21-source-set-mathcaldk)
  - [2.2 Mixture weights $\alpha_k$](#22-mixture-weights-alphak)
  - [2.3 Token budget $T$](#23-token-budget-t)
  - [2.4 Shard manifest](#24-shard-manifest)
  - [2.5 Split assignment](#25-split-assignment)
- [3. Corpus Manifests](#3-corpus-manifests)
  - [3.1 Source inventory](#31-source-inventory)
  - [3.2 Version pins](#32-version-pins)
  - [3.3 Hashes/checksums](#33-hasheschecksums)
  - [3.4 License fields](#34-license-fields)
  - [3.5 Build recipe](#35-build-recipe)
- [4. Assembly Algorithms](#4-assembly-algorithms)
  - [4.1 Concatenation](#41-concatenation)
  - [4.2 Weighted sampling](#42-weighted-sampling)
  - [4.3 Stratified sampling](#43-stratified-sampling)
  - [4.4 Train/validation/test split](#44-trainvalidationtest-split)
  - [4.5 Deterministic shuffling](#45-deterministic-shuffling)
- [5. Tokenization and Packing Interface](#5-tokenization-and-packing-interface)
  - [5.1 Token-count budgets](#51-tokencount-budgets)
  - [5.2 Sequence packing](#52-sequence-packing)
  - [5.3 Document-boundary masks](#53-documentboundary-masks)
  - [5.4 Padding/truncation](#54-paddingtruncation)
  - [5.5 Packed-shard statistics](#55-packedshard-statistics)
- [6. Final Verification](#6-final-verification)
  - [6.1 Shard count](#61-shard-count)
  - [6.2 Token count](#62-token-count)
  - [6.3 Source proportions](#63-source-proportions)
  - [6.4 Reproducible rebuild](#64-reproducible-rebuild)
  - [6.5 Smoke-test data loader](#65-smoketest-data-loader)
- [7. Common Mistakes](#7-common-mistakes)
- [8. Exercises](#8-exercises)
- [9. Why This Matters for AI](#9-why-this-matters-for-ai)
- [10. Conceptual Bridge](#10-conceptual-bridge)
- [References](#references)

---

## 1. Intuition

Intuition gives the conceptual and mathematical layer for full dataset assembly. The
local variables in this section should be read as pipeline objects: documents, records,
tokens, filters, weights, shards, and manifests.

### 1.1 Assembly as turning accepted records into a trainable corpus

Assembly as turning accepted records into a trainable corpus is part of the canonical
scope of full dataset assembly. We model the relevant object as a finite collection
$\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level metadata $m_i$ and text or token
content $x_i$. The practical question is whether the transformation preserves the
intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `source set`, the invariant should be explicit enough that a checker can fail fast.
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

### 1.2 Source manifests

Source manifests is part of the canonical scope of full dataset assembly. We model the
relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level
metadata $m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `mixture weight`, the invariant should be explicit enough that a checker can fail
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

### 1.3 Token accounting

Token accounting is part of the canonical scope of full dataset assembly. We model the
relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level
metadata $m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `token budget`, the invariant should be explicit enough that a checker can fail
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

### 1.4 Data order as curriculum

Data order as curriculum is part of the canonical scope of full dataset assembly. We
model the relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with
record-level metadata $m_i$ and text or token content $x_i$. The practical question is
whether the transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `manifest`, the invariant should be explicit enough that a checker can fail fast. If
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

### 1.5 Reproducibility at trillion-token scale

Reproducibility at trillion-token scale is part of the canonical scope of full dataset
assembly. We model the relevant object as a finite collection $\mathcal{D} =
\{r_i\}_{i=1}^n$ with record-level metadata $m_i$ and text or token content $x_i$. The
practical question is whether the transformation preserves the intended empirical
distribution.

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

## 2. Formal Definitions

Formal Definitions gives the conceptual and mathematical layer for full dataset
assembly. The local variables in this section should be read as pipeline objects:
documents, records, tokens, filters, weights, shards, and manifests.

### 2.1 Source set $\mathcal{D}_k$

Source set $\mathcal{D}_k$ is part of the canonical scope of full dataset assembly. We
model the relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with
record-level metadata $m_i$ and text or token content $x_i$. The practical question is
whether the transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `source set`, the invariant should be explicit enough that a checker can fail fast.
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

### 2.2 Mixture weights $\alpha_k$

Mixture weights $\alpha_k$ is part of the canonical scope of full dataset assembly. We
model the relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with
record-level metadata $m_i$ and text or token content $x_i$. The practical question is
whether the transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `mixture weight`, the invariant should be explicit enough that a checker can fail
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

### 2.3 Token budget $T$

Token budget $T$ is part of the canonical scope of full dataset assembly. We model the
relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level
metadata $m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `token budget`, the invariant should be explicit enough that a checker can fail
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

### 2.4 Shard manifest

Shard manifest is part of the canonical scope of full dataset assembly. We model the
relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level
metadata $m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `manifest`, the invariant should be explicit enough that a checker can fail fast. If
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

### 2.5 Split assignment

Split assignment is part of the canonical scope of full dataset assembly. We model the
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

## 3. Corpus Manifests

Corpus Manifests gives the conceptual and mathematical layer for full dataset assembly.
The local variables in this section should be read as pipeline objects: documents,
records, tokens, filters, weights, shards, and manifests.

### 3.1 Source inventory

Source inventory is part of the canonical scope of full dataset assembly. We model the
relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level
metadata $m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `source set`, the invariant should be explicit enough that a checker can fail fast.
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

### 3.2 Version pins

Version pins is part of the canonical scope of full dataset assembly. We model the
relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level
metadata $m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `mixture weight`, the invariant should be explicit enough that a checker can fail
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

### 3.3 Hashes/checksums

Hashes/checksums is part of the canonical scope of full dataset assembly. We model the
relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level
metadata $m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `token budget`, the invariant should be explicit enough that a checker can fail
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

### 3.4 License fields

License fields is part of the canonical scope of full dataset assembly. We model the
relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level
metadata $m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `manifest`, the invariant should be explicit enough that a checker can fail fast. If
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

### 3.5 Build recipe

Build recipe is part of the canonical scope of full dataset assembly. We model the
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

## 4. Assembly Algorithms

Assembly Algorithms gives the conceptual and mathematical layer for full dataset
assembly. The local variables in this section should be read as pipeline objects:
documents, records, tokens, filters, weights, shards, and manifests.

### 4.1 Concatenation

Concatenation is part of the canonical scope of full dataset assembly. We model the
relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level
metadata $m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `source set`, the invariant should be explicit enough that a checker can fail fast.
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

### 4.2 Weighted sampling

Weighted sampling is part of the canonical scope of full dataset assembly. We model the
relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level
metadata $m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `mixture weight`, the invariant should be explicit enough that a checker can fail
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

### 4.3 Stratified sampling

Stratified sampling is part of the canonical scope of full dataset assembly. We model
the relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-
level metadata $m_i$ and text or token content $x_i$. The practical question is whether
the transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `token budget`, the invariant should be explicit enough that a checker can fail
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

### 4.4 Train/validation/test split

Train/validation/test split is part of the canonical scope of full dataset assembly. We
model the relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with
record-level metadata $m_i$ and text or token content $x_i$. The practical question is
whether the transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `manifest`, the invariant should be explicit enough that a checker can fail fast. If
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

### 4.5 Deterministic shuffling

Deterministic shuffling is part of the canonical scope of full dataset assembly. We
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

## 5. Tokenization and Packing Interface

Tokenization and Packing Interface gives the conceptual and mathematical layer for full
dataset assembly. The local variables in this section should be read as pipeline
objects: documents, records, tokens, filters, weights, shards, and manifests.

### 5.1 Token-count budgets

Token-count budgets is part of the canonical scope of full dataset assembly. We model
the relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-
level metadata $m_i$ and text or token content $x_i$. The practical question is whether
the transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `source set`, the invariant should be explicit enough that a checker can fail fast.
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

### 5.2 Sequence packing

Sequence packing is part of the canonical scope of full dataset assembly. We model the
relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level
metadata $m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `mixture weight`, the invariant should be explicit enough that a checker can fail
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

### 5.3 Document-boundary masks

Document-boundary masks is part of the canonical scope of full dataset assembly. We
model the relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with
record-level metadata $m_i$ and text or token content $x_i$. The practical question is
whether the transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `token budget`, the invariant should be explicit enough that a checker can fail
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

### 5.4 Padding/truncation

Padding/truncation is part of the canonical scope of full dataset assembly. We model the
relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level
metadata $m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `manifest`, the invariant should be explicit enough that a checker can fail fast. If
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

### 5.5 Packed-shard statistics

Packed-shard statistics is part of the canonical scope of full dataset assembly. We
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

## 6. Final Verification

Final Verification gives the conceptual and mathematical layer for full dataset
assembly. The local variables in this section should be read as pipeline objects:
documents, records, tokens, filters, weights, shards, and manifests.

### 6.1 Shard count

Shard count is part of the canonical scope of full dataset assembly. We model the
relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level
metadata $m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `source set`, the invariant should be explicit enough that a checker can fail fast.
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

### 6.2 Token count

Token count is part of the canonical scope of full dataset assembly. We model the
relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level
metadata $m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `mixture weight`, the invariant should be explicit enough that a checker can fail
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

### 6.3 Source proportions

Source proportions is part of the canonical scope of full dataset assembly. We model the
relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-level
metadata $m_i$ and text or token content $x_i$. The practical question is whether the
transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `token budget`, the invariant should be explicit enough that a checker can fail
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

### 6.4 Reproducible rebuild

Reproducible rebuild is part of the canonical scope of full dataset assembly. We model
the relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-
level metadata $m_i$ and text or token content $x_i$. The practical question is whether
the transformation preserves the intended empirical distribution.

A useful local invariant is:

$$
\text{valid}(r_i, \mathcal{S}) = 1 \quad \Longrightarrow \quad r_i \text{ can be consumed by the next pipeline stage.}
$$

For `manifest`, the invariant should be explicit enough that a checker can fail fast. If
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

### 6.5 Smoke-test data loader

Smoke-test data loader is part of the canonical scope of full dataset assembly. We model
the relevant object as a finite collection $\mathcal{D} = \{r_i\}_{i=1}^n$ with record-
level metadata $m_i$ and text or token content $x_i$. The practical question is whether
the transformation preserves the intended empirical distribution.

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

1. (*) Build a synthetic `source set` example, compute its validation signal, and explain which downstream stage would fail if the signal were wrong.
2. (*) Build a synthetic `mixture weight` example, compute its validation signal, and explain which downstream stage would fail if the signal were wrong.
3. (*) Build a synthetic `token budget` example, compute its validation signal, and explain which downstream stage would fail if the signal were wrong.
4. (**) Build a synthetic `manifest` example, compute its validation signal, and explain which downstream stage would fail if the signal were wrong.
5. (**) Build a synthetic `shard` example, compute its validation signal, and explain which downstream stage would fail if the signal were wrong.
6. (**) Build a synthetic `split` example, compute its validation signal, and explain which downstream stage would fail if the signal were wrong.
7. (**) Build a synthetic `packing` example, compute its validation signal, and explain which downstream stage would fail if the signal were wrong.
8. (***) Build a synthetic `source set` example, compute its validation signal, and explain which downstream stage would fail if the signal were wrong.
9. (***) Build a synthetic `mixture weight` example, compute its validation signal, and explain which downstream stage would fail if the signal were wrong.
10. (***) Build a synthetic `token budget` example, compute its validation signal, and explain which downstream stage would fail if the signal were wrong.

## 9. Why This Matters for AI

| Concept | AI impact |
| --- | --- |
| source set | Controls what examples, gradients, risks, or audits the model pipeline can represent |
| mixture weight | Controls what examples, gradients, risks, or audits the model pipeline can represent |
| token budget | Controls what examples, gradients, risks, or audits the model pipeline can represent |
| manifest | Controls what examples, gradients, risks, or audits the model pipeline can represent |
| shard | Controls what examples, gradients, risks, or audits the model pipeline can represent |
| split | Controls what examples, gradients, risks, or audits the model pipeline can represent |
| packing | Controls what examples, gradients, risks, or audits the model pipeline can represent |

Data pipeline quality is model quality in delayed form. The model eventually converts these records into gradients; any unresolved ambiguity becomes either wasted compute, misleading evaluation, memorization risk, or irreproducible science.

## 10. Conceptual Bridge

This section connects the previous and next pieces of the curriculum as follows:

```text
raw sources -> records -> validation -> assembly -> audits -> documentation -> mixture
```

The next section is [Contamination and Dedup Audits](../05-Contamination-and-Dedup-
Audits/notes.md). It uses the contracts established here and moves one step further
through the LLM data pipeline.

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
