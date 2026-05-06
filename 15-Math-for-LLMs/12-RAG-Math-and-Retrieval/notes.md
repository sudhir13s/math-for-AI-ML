[<- Quantization and Distillation](../11-Quantization-and-Distillation/notes.md) | [Home](../../README.md) | [Serving and Systems Tradeoffs ->](../13-Serving-and-Systems-Tradeoffs/notes.md)

---

# RAG Math and Retrieval

Retrieval-augmented generation adds an external memory system to an LLM. The math is a pipeline: embed, search, rank, pack context, generate, and verify attribution.

## Overview

The central RAG conditional is:

$$
p_\theta(y\mid q,R_k(q)),
$$

where $q$ is the query and $R_k(q)$ is the set of retrieved chunks. RAG succeeds only when the right information is retrieved, ranked highly, packed into context, and used by the generator.

## Prerequisites

- Embedding-space geometry and cosine similarity
- Conditional language-model probability
- Efficient inference and context-window constraints
- Evaluation metrics and error analysis

## Companion Notebooks

| Notebook | Purpose |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Demonstrates cosine search, BM25 intuition, contrastive retrieval loss, recall@k, MMR, reranking, context packing, and RAG failure decomposition. |
| [exercises.ipynb](exercises.ipynb) | Ten practice problems for retrieval scores, recall, MRR, chunk packing, MMR, RRF, and RAG diagnostics. |

## Learning Objectives

After this section, you should be able to:

- Define RAG as conditional generation with retrieved non-parametric memory.
- Compute dot-product and cosine retrieval scores.
- Explain sparse, dense, hybrid, late-interaction, and cross-encoder retrieval.
- Write the contrastive loss for dense retriever training.
- Compute recall@k, MRR, and simple nDCG.
- Explain chunk length, overlap, MMR, and context packing.
- Explain ANN recall-latency tradeoffs.
- Diagnose RAG failures with traces and ablations.

## Table of Contents

1. [RAG as Conditional Generation](#1-rag-as-conditional-generation)
   - 1.1 [Parametric memory](#11-parametric-memory)
   - 1.2 [Non-parametric memory](#12-nonparametric-memory)
   - 1.3 [Retriever](#13-retriever)
   - 1.4 [Generator](#14-generator)
   - 1.5 [Failure decomposition](#15-failure-decomposition)
2. [Similarity Spaces](#2-similarity-spaces)
   - 2.1 [Embedding functions](#21-embedding-functions)
   - 2.2 [Dot product](#22-dot-product)
   - 2.3 [Cosine similarity](#23-cosine-similarity)
   - 2.4 [Maximum inner product search](#24-maximum-inner-product-search)
   - 2.5 [Normalization](#25-normalization)
3. [Sparse and Dense Retrieval](#3-sparse-and-dense-retrieval)
   - 3.1 [Sparse lexical retrieval](#31-sparse-lexical-retrieval)
   - 3.2 [Dense bi-encoder retrieval](#32-dense-biencoder-retrieval)
   - 3.3 [Hybrid retrieval](#33-hybrid-retrieval)
   - 3.4 [Late interaction](#34-late-interaction)
   - 3.5 [Cross-encoder reranking](#35-crossencoder-reranking)
4. [Retriever Training](#4-retriever-training)
   - 4.1 [Positive pairs](#41-positive-pairs)
   - 4.2 [Negative pairs](#42-negative-pairs)
   - 4.3 [Contrastive loss](#43-contrastive-loss)
   - 4.4 [In-batch negatives](#44-inbatch-negatives)
   - 4.5 [Hard negatives](#45-hard-negatives)
5. [Approximate Nearest Neighbor Search](#5-approximate-nearest-neighbor-search)
   - 5.1 [Exact search](#51-exact-search)
   - 5.2 [ANN search](#52-ann-search)
   - 5.3 [Recall at k](#53-recall-at-k)
   - 5.4 [Index compression](#54-index-compression)
   - 5.5 [Latency recall tradeoff](#55-latency-recall-tradeoff)
6. [Chunking and Context Packing](#6-chunking-and-context-packing)
   - 6.1 [Chunk length](#61-chunk-length)
   - 6.2 [Overlap](#62-overlap)
   - 6.3 [Packing budget](#63-packing-budget)
   - 6.4 [Diversity](#64-diversity)
   - 6.5 [Lost-in-context risk](#65-lostincontext-risk)
7. [Reranking and Fusion](#7-reranking-and-fusion)
   - 7.1 [First-stage recall](#71-firststage-recall)
   - 7.2 [Reranker precision](#72-reranker-precision)
   - 7.3 [Reciprocal rank fusion](#73-reciprocal-rank-fusion)
   - 7.4 [Score calibration](#74-score-calibration)
   - 7.5 [Citation selection](#75-citation-selection)
8. [RAG Evaluation](#8-rag-evaluation)
   - 8.1 [Retrieval metrics](#81-retrieval-metrics)
   - 8.2 [Answer metrics](#82-answer-metrics)
   - 8.3 [Attribution](#83-attribution)
   - 8.4 [Ablations](#84-ablations)
   - 8.5 [Dataset drift](#85-dataset-drift)
9. [Failure Modes](#9-failure-modes)
   - 9.1 [Missed retrieval](#91-missed-retrieval)
   - 9.2 [Bad chunk](#92-bad-chunk)
   - 9.3 [Distractor context](#93-distractor-context)
   - 9.4 [Generator ignores evidence](#94-generator-ignores-evidence)
   - 9.5 [Citation mismatch](#95-citation-mismatch)
10. [Implementation Checklist](#10-implementation-checklist)
   - 10.1 [Embedding normalization](#101-embedding-normalization)
   - 10.2 [Chunk audit](#102-chunk-audit)
   - 10.3 [Gold retrieval set](#103-gold-retrieval-set)
   - 10.4 [Context budget tests](#104-context-budget-tests)
   - 10.5 [End-to-end traces](#105-endtoend-traces)

---

## Pipeline Diagram

```text
query -> query encoder -> vector search -> top-k chunks -> reranker -> context packer -> LLM -> answer + citations
```

Each arrow can fail. The math gives you probes for each arrow.

## 1. RAG as Conditional Generation

This part studies rag as conditional generation as retrieval math for LLM systems. Keep separate the embedding space, search algorithm, ranking metric, context budget, and generator behavior.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Parametric memory](#1-parametric-memory) | knowledge stored in model weights | $p_\theta(y\mid q)$ |
| [Non-parametric memory](#1-nonparametric-memory) | knowledge stored in an external corpus | $D=\{d_i\}_{i=1}^n$ |
| [Retriever](#1-retriever) | select relevant documents for a query | $R_k(q)=\mathrm{TopK}_{d_i\in D}\ s(q,d_i)$ |
| [Generator](#1-generator) | answer conditioned on query and retrieved context | $p_\theta(y\mid q,R_k(q))$ |
| [Failure decomposition](#1-failure-decomposition) | RAG can fail at retrieval, ranking, context packing, or generation | $P(\mathrm{correct})=P(\mathrm{retrieve})P(\mathrm{use})P(\mathrm{generate})$ |

### 1.1 Parametric memory

**Main idea.** Knowledge stored in model weights.

Core relation:

$$p_\theta(y\mid q)$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 1.2 Non-parametric memory

**Main idea.** Knowledge stored in an external corpus.

Core relation:

$$D=\{d_i\}_{i=1}^n$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 1.3 Retriever

**Main idea.** Select relevant documents for a query.

Core relation:

$$R_k(q)=\mathrm{TopK}_{d_i\in D}\ s(q,d_i)$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** The generator cannot use evidence that retrieval never returns.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 1.4 Generator

**Main idea.** Answer conditioned on query and retrieved context.

Core relation:

$$p_\theta(y\mid q,R_k(q))$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 1.5 Failure decomposition

**Main idea.** Rag can fail at retrieval, ranking, context packing, or generation.

Core relation:

$$P(\mathrm{correct})=P(\mathrm{retrieve})P(\mathrm{use})P(\mathrm{generate})$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
## 2. Similarity Spaces

This part studies similarity spaces as retrieval math for LLM systems. Keep separate the embedding space, search algorithm, ranking metric, context budget, and generator behavior.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Embedding functions](#2-embedding-functions) | map queries and documents into a shared vector space | $u=f_q(q),\ v_i=f_d(d_i)$ |
| [Dot product](#2-dot-product) | inner product rewards alignment and vector norm | $s(u,v)=u^\top v$ |
| [Cosine similarity](#2-cosine-similarity) | normalize away vector length | $s(u,v)=u^\top v/(\Vert u\Vert\Vert v\Vert)$ |
| [Maximum inner product search](#2-maximum-inner-product-search) | retrieve highest-scoring vectors | $\arg\max_i u^\top v_i$ |
| [Normalization](#2-normalization) | for unit vectors, dot product and cosine are the same | $\Vert u\Vert=\Vert v\Vert=1$ |

### 2.1 Embedding functions

**Main idea.** Map queries and documents into a shared vector space.

Core relation:

$$u=f_q(q),\ v_i=f_d(d_i)$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 2.2 Dot product

**Main idea.** Inner product rewards alignment and vector norm.

Core relation:

$$s(u,v)=u^\top v$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 2.3 Cosine similarity

**Main idea.** Normalize away vector length.

Core relation:

$$s(u,v)=u^\top v/(\Vert u\Vert\Vert v\Vert)$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** Most RAG bugs start with misunderstanding what the vector store is scoring.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 2.4 Maximum inner product search

**Main idea.** Retrieve highest-scoring vectors.

Core relation:

$$\arg\max_i u^\top v_i$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 2.5 Normalization

**Main idea.** For unit vectors, dot product and cosine are the same.

Core relation:

$$\Vert u\Vert=\Vert v\Vert=1$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
## 3. Sparse and Dense Retrieval

This part studies sparse and dense retrieval as retrieval math for LLM systems. Keep separate the embedding space, search algorithm, ranking metric, context budget, and generator behavior.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Sparse lexical retrieval](#3-sparse-lexical-retrieval) | match query terms to document terms | $\mathrm{BM25}(q,d)$ |
| [Dense bi-encoder retrieval](#3-dense-biencoder-retrieval) | encode query and document separately | $s(q,d)=f_q(q)^\top f_d(d)$ |
| [Hybrid retrieval](#3-hybrid-retrieval) | combine sparse and dense scores | $s=\lambda s_\mathrm{dense}+(1-\lambda)s_\mathrm{sparse}$ |
| [Late interaction](#3-late-interaction) | score token embeddings after independent encoding | $\sum_{i\in q}\max_{j\in d} e_i^\top e_j$ |
| [Cross-encoder reranking](#3-crossencoder-reranking) | jointly encode query and document for a slower stronger score | $s=g(q,d)$ |

### 3.1 Sparse lexical retrieval

**Main idea.** Match query terms to document terms.

Core relation:

$$\mathrm{BM25}(q,d)$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 3.2 Dense bi-encoder retrieval

**Main idea.** Encode query and document separately.

Core relation:

$$s(q,d)=f_q(q)^\top f_d(d)$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 3.3 Hybrid retrieval

**Main idea.** Combine sparse and dense scores.

Core relation:

$$s=\lambda s_\mathrm{dense}+(1-\lambda)s_\mathrm{sparse}$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 3.4 Late interaction

**Main idea.** Score token embeddings after independent encoding.

Core relation:

$$\sum_{i\in q}\max_{j\in d} e_i^\top e_j$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 3.5 Cross-encoder reranking

**Main idea.** Jointly encode query and document for a slower stronger score.

Core relation:

$$s=g(q,d)$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
## 4. Retriever Training

This part studies retriever training as retrieval math for LLM systems. Keep separate the embedding space, search algorithm, ranking metric, context budget, and generator behavior.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Positive pairs](#4-positive-pairs) | train query and relevant document to be close | $(q,d^+)$ |
| [Negative pairs](#4-negative-pairs) | train irrelevant or hard-negative documents to score lower | $(q,d^-)$ |
| [Contrastive loss](#4-contrastive-loss) | softmax over one positive and negatives | $L=-\log\frac{e^{s(q,d^+)}}{e^{s(q,d^+)}+\sum_j e^{s(q,d_j^-)}}$ |
| [In-batch negatives](#4-inbatch-negatives) | other examples in a batch become negatives | $B-1$ negatives per query |
| [Hard negatives](#4-hard-negatives) | near misses improve ranking training | $s(q,d^-)$ high but label negative |

### 4.1 Positive pairs

**Main idea.** Train query and relevant document to be close.

Core relation:

$$(q,d^+)$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 4.2 Negative pairs

**Main idea.** Train irrelevant or hard-negative documents to score lower.

Core relation:

$$(q,d^-)$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 4.3 Contrastive loss

**Main idea.** Softmax over one positive and negatives.

Core relation:

$$L=-\log\frac{e^{s(q,d^+)}}{e^{s(q,d^+)}+\sum_j e^{s(q,d_j^-)}}$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is the training objective behind many dense retrievers.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 4.4 In-batch negatives

**Main idea.** Other examples in a batch become negatives.

Core relation:

$$B-1$ negatives per query$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 4.5 Hard negatives

**Main idea.** Near misses improve ranking training.

Core relation:

$$s(q,d^-)$ high but label negative$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
## 5. Approximate Nearest Neighbor Search

This part studies approximate nearest neighbor search as retrieval math for LLM systems. Keep separate the embedding space, search algorithm, ranking metric, context budget, and generator behavior.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Exact search](#5-exact-search) | compare a query to every vector | $O(nd)$ |
| [ANN search](#5-ann-search) | trade exactness for latency and memory | $\hat R_k(q)\approx R_k(q)$ |
| [Recall at k](#5-recall-at-k) | measure whether relevant docs appear in top k | $\mathrm{Recall@}k=|\mathrm{Rel}\cap R_k|/|\mathrm{Rel}|$ |
| [Index compression](#5-index-compression) | quantize or cluster vectors to reduce memory | $V\rightarrow \hat V$ |
| [Latency recall tradeoff](#5-latency-recall-tradeoff) | faster search can miss relevant documents | $T\downarrow,\ \mathrm{recall}\downarrow$ |

### 5.1 Exact search

**Main idea.** Compare a query to every vector.

Core relation:

$$O(nd)$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 5.2 ANN search

**Main idea.** Trade exactness for latency and memory.

Core relation:

$$\hat R_k(q)\approx R_k(q)$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 5.3 Recall at k

**Main idea.** Measure whether relevant docs appear in top k.

Core relation:

$$\mathrm{Recall@}k=|\mathrm{Rel}\cap R_k|/|\mathrm{Rel}|$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 5.4 Index compression

**Main idea.** Quantize or cluster vectors to reduce memory.

Core relation:

$$V\rightarrow \hat V$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 5.5 Latency recall tradeoff

**Main idea.** Faster search can miss relevant documents.

Core relation:

$$T\downarrow,\ \mathrm{recall}\downarrow$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
## 6. Chunking and Context Packing

This part studies chunking and context packing as retrieval math for LLM systems. Keep separate the embedding space, search algorithm, ranking metric, context budget, and generator behavior.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Chunk length](#6-chunk-length) | split documents into retrievable units | $d\rightarrow c_1,\ldots,c_m$ |
| [Overlap](#6-overlap) | repeat boundary tokens to avoid cutting facts | $c_i=[t_a,\ldots,t_b],\ c_{i+1}=[t_{b-o},\ldots]$ |
| [Packing budget](#6-packing-budget) | retrieved chunks must fit context | $\sum_i |c_i|\le T_\mathrm{context}$ |
| [Diversity](#6-diversity) | avoid filling context with near-duplicates | $\mathrm{MMR}=\lambda s(q,d)-(1-\lambda)\max_{d'\in S}s(d,d')$ |
| [Lost-in-context risk](#6-lostincontext-risk) | retrieved text must be ordered and summarized so the generator can use it | $p(y\mid q,c_{1:k})$ depends on packing |

### 6.1 Chunk length

**Main idea.** Split documents into retrievable units.

Core relation:

$$d\rightarrow c_1,\ldots,c_m$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 6.2 Overlap

**Main idea.** Repeat boundary tokens to avoid cutting facts.

Core relation:

$$c_i=[t_a,\ldots,t_b],\ c_{i+1}=[t_{b-o},\ldots]$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 6.3 Packing budget

**Main idea.** Retrieved chunks must fit context.

Core relation:

$$\sum_i |c_i|\le T_\mathrm{context}$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** Retrieval success still fails if the evidence is packed badly into context.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 6.4 Diversity

**Main idea.** Avoid filling context with near-duplicates.

Core relation:

$$\mathrm{MMR}=\lambda s(q,d)-(1-\lambda)\max_{d'\in S}s(d,d')$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 6.5 Lost-in-context risk

**Main idea.** Retrieved text must be ordered and summarized so the generator can use it.

Core relation:

$$p(y\mid q,c_{1:k})$ depends on packing$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
## 7. Reranking and Fusion

This part studies reranking and fusion as retrieval math for LLM systems. Keep separate the embedding space, search algorithm, ranking metric, context budget, and generator behavior.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [First-stage recall](#7-firststage-recall) | retrieve broad candidates cheaply | $K_\mathrm{first}\gg K_\mathrm{final}$ |
| [Reranker precision](#7-reranker-precision) | rerank candidates with a stronger model | $s_\mathrm{rerank}(q,d)$ |
| [Reciprocal rank fusion](#7-reciprocal-rank-fusion) | combine ranked lists robustly | $\mathrm{RRF}(d)=\sum_m 1/(k+r_m(d))$ |
| [Score calibration](#7-score-calibration) | dense, sparse, and reranker scores may live on different scales | $z=(s-\mu)/\sigma$ |
| [Citation selection](#7-citation-selection) | answer citations should correspond to evidence actually used | $d_i\rightarrow \mathrm{claim}_j$ |

### 7.1 First-stage recall

**Main idea.** Retrieve broad candidates cheaply.

Core relation:

$$K_\mathrm{first}\gg K_\mathrm{final}$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 7.2 Reranker precision

**Main idea.** Rerank candidates with a stronger model.

Core relation:

$$s_\mathrm{rerank}(q,d)$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 7.3 Reciprocal rank fusion

**Main idea.** Combine ranked lists robustly.

Core relation:

$$\mathrm{RRF}(d)=\sum_m 1/(k+r_m(d))$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 7.4 Score calibration

**Main idea.** Dense, sparse, and reranker scores may live on different scales.

Core relation:

$$z=(s-\mu)/\sigma$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 7.5 Citation selection

**Main idea.** Answer citations should correspond to evidence actually used.

Core relation:

$$d_i\rightarrow \mathrm{claim}_j$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
## 8. RAG Evaluation

This part studies rag evaluation as retrieval math for LLM systems. Keep separate the embedding space, search algorithm, ranking metric, context budget, and generator behavior.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Retrieval metrics](#8-retrieval-metrics) | measure search independently of generation | $\mathrm{Recall@}k,\ \mathrm{MRR},\ \mathrm{nDCG}$ |
| [Answer metrics](#8-answer-metrics) | measure final response correctness and faithfulness | $S_\mathrm{answer}$ |
| [Attribution](#8-attribution) | claims should be supported by retrieved evidence | $\mathrm{support}(\mathrm{claim},d_i)$ |
| [Ablations](#8-ablations) | compare no retrieval, sparse, dense, hybrid, and reranked variants | $\Delta S$ |
| [Dataset drift](#8-dataset-drift) | retrieval quality changes when corpus or query distribution changes | $p_\mathrm{query},D$ drift |

### 8.1 Retrieval metrics

**Main idea.** Measure search independently of generation.

Core relation:

$$\mathrm{Recall@}k,\ \mathrm{MRR},\ \mathrm{nDCG}$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 8.2 Answer metrics

**Main idea.** Measure final response correctness and faithfulness.

Core relation:

$$S_\mathrm{answer}$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 8.3 Attribution

**Main idea.** Claims should be supported by retrieved evidence.

Core relation:

$$\mathrm{support}(\mathrm{claim},d_i)$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 8.4 Ablations

**Main idea.** Compare no retrieval, sparse, dense, hybrid, and reranked variants.

Core relation:

$$\Delta S$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 8.5 Dataset drift

**Main idea.** Retrieval quality changes when corpus or query distribution changes.

Core relation:

$$p_\mathrm{query},D$ drift$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
## 9. Failure Modes

This part studies failure modes as retrieval math for LLM systems. Keep separate the embedding space, search algorithm, ranking metric, context budget, and generator behavior.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Missed retrieval](#9-missed-retrieval) | the answer document is not in top k | $d^+\notin R_k(q)$ |
| [Bad chunk](#9-bad-chunk) | the right document is retrieved but not the right span | $c^+\notin R_k(q)$ |
| [Distractor context](#9-distractor-context) | irrelevant high-scoring chunks pull generation away | $s(q,d^-)>s(q,d^+)$ |
| [Generator ignores evidence](#9-generator-ignores-evidence) | the answer is not grounded even with good retrieval | $p_\theta(y\mid q,R_k)$ uses parametric prior |
| [Citation mismatch](#9-citation-mismatch) | the cited chunk does not support the claim | $\mathrm{claim}\not\subset d_\mathrm{cited}$ |

### 9.1 Missed retrieval

**Main idea.** The answer document is not in top k.

Core relation:

$$d^+\notin R_k(q)$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 9.2 Bad chunk

**Main idea.** The right document is retrieved but not the right span.

Core relation:

$$c^+\notin R_k(q)$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 9.3 Distractor context

**Main idea.** Irrelevant high-scoring chunks pull generation away.

Core relation:

$$s(q,d^-)>s(q,d^+)$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 9.4 Generator ignores evidence

**Main idea.** The answer is not grounded even with good retrieval.

Core relation:

$$p_\theta(y\mid q,R_k)$ uses parametric prior$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 9.5 Citation mismatch

**Main idea.** The cited chunk does not support the claim.

Core relation:

$$\mathrm{claim}\not\subset d_\mathrm{cited}$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
## 10. Implementation Checklist

This part studies implementation checklist as retrieval math for LLM systems. Keep separate the embedding space, search algorithm, ranking metric, context budget, and generator behavior.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Embedding normalization](#10-embedding-normalization) | know whether the index expects normalized vectors | $v\leftarrow v/\Vert v\Vert$ |
| [Chunk audit](#10-chunk-audit) | inspect chunks before blaming the retriever | $c_i$ |
| [Gold retrieval set](#10-gold-retrieval-set) | keep queries with known supporting documents | $D_\mathrm{gold}$ |
| [Context budget tests](#10-context-budget-tests) | evaluate different k, chunk length, and overlap | $k,o,|c|$ |
| [End-to-end traces](#10-endtoend-traces) | log query, retrieved docs, reranker scores, prompt, answer, and citations | $\mathrm{trace}$ |

### 10.1 Embedding normalization

**Main idea.** Know whether the index expects normalized vectors.

Core relation:

$$v\leftarrow v/\Vert v\Vert$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 10.2 Chunk audit

**Main idea.** Inspect chunks before blaming the retriever.

Core relation:

$$c_i$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 10.3 Gold retrieval set

**Main idea.** Keep queries with known supporting documents.

Core relation:

$$D_\mathrm{gold}$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 10.4 Context budget tests

**Main idea.** Evaluate different k, chunk length, and overlap.

Core relation:

$$k,o,|c|$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 10.5 End-to-end traces

**Main idea.** Log query, retrieved docs, reranker scores, prompt, answer, and citations.

Core relation:

$$\mathrm{trace}$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** A RAG trace is the fastest way to locate whether failure came from search, ranking, packing, or generation.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.

---

## Practice Exercises

1. Normalize vectors and compute cosine similarities.
2. Retrieve top-k documents by dot product.
3. Compute a toy BM25-style lexical score.
4. Compute dense contrastive loss with one positive and negatives.
5. Compute recall@k and MRR.
6. Use MMR to select diverse chunks.
7. Pack chunks into a context budget.
8. Combine rankings with reciprocal rank fusion.
9. Decompose an end-to-end RAG failure.
10. Write a RAG trace checklist.

## Why This Matters for AI

RAG is often the cheapest way to update knowledge, cite sources, and ground answers. But RAG is not magic. Retrieval can miss the answer, rank distractors above evidence, split the useful span across chunks, or feed the generator context it ignores. Good RAG work is measurement-heavy.

## Bridge to Serving and Systems Tradeoffs

The final LLM math section studies the system-level tradeoffs around serving: batching, latency, throughput, memory, routing, caching, and cost. RAG adds another system layer because retrieval latency and context length feed directly into serving latency.

## References

- Patrick Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", 2020: https://arxiv.org/abs/2005.11401
- Vladimir Karpukhin et al., "Dense Passage Retrieval for Open-Domain Question Answering", 2020: https://arxiv.org/abs/2004.04906
- Omar Khattab and Matei Zaharia, "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT", 2020: https://arxiv.org/abs/2004.12832
- Jeff Johnson, Matthijs Douze, and Herve Jegou, "Billion-scale similarity search with GPUs", 2017: https://arxiv.org/abs/1702.08734
- Stephen Robertson and Hugo Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond", 2009: https://www.nowpublishers.com/article/Details/INR-019
