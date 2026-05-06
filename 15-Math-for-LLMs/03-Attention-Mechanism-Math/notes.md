[Previous: Embedding Space Math](../02-Embedding-Space-Math/notes.md) | [Back to Math for LLMs](../README.md) | [Next: Positional Encodings](../04-Positional-Encodings/notes.md)

---

# Attention Mechanism Math

> _"Attention is differentiable retrieval over the current context."_

## Overview

Attention is the mechanism that lets each token representation read from other token representations. It creates query, key, and value vectors, computes compatibility scores, masks illegal positions, normalizes with softmax, and mixes value vectors into context-aware outputs.

For LLMs, attention is the central sequence-mixing operation. It powers in-context learning, copying, retrieval use, chat-template boundaries, and long-context behavior. It is also one of the main systems bottlenecks because score matrices grow quadratically in sequence length.

This section uses LaTeX Markdown with `$...$` and `$$...$$`. The companion notebooks implement stable softmax attention, causal masks, multi-head shapes, entropy diagnostics, KV-cache sizing, and efficient-attention intuition in small NumPy examples.

## Prerequisites

- [Embedding Space Math](../02-Embedding-Space-Math/notes.md)
- [Matrix Operations](../../02-Linear-Algebra-Basics/02-Matrix-Operations/notes.md)
- [Softmax and Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md)
- [Positional Encodings](../04-Positional-Encodings/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Executable demonstrations of Q/K/V attention, masks, entropy, heads, KV cache, ALiBi-style bias, and FlashAttention intuition. |
| [exercises.ipynb](exercises.ipynb) | Ten checked exercises covering attention mechanics and LLM serving consequences. |

## Learning Objectives

After completing this section, you will be able to:

- Define queries, keys, values, attention scores, masks, weights, and outputs.
- Compute scaled dot-product attention by hand for a small sequence.
- Apply causal and padding masks before softmax.
- Explain why attention rows are probability-like but not full explanations.
- Track multi-head tensor shapes through split, attention, concat, and output projection.
- Compute attention entropy and interpret sharp or diffuse rows.
- Explain KV cache memory and the prefill/decode distinction.
- Describe why FlashAttention is exact attention with a more efficient memory algorithm.
- Diagnose mask leakage and padding bugs.
- Connect attention math to in-context learning, RAG, long context, and safety boundaries.

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 Attention as soft retrieval](#11-attention-as-soft-retrieval)
  - [1.2 Queries keys and values as roles](#12-queries-keys-and-values-as-roles)
  - [1.3 Why scaling is needed](#13-why-scaling-is-needed)
  - [1.4 Why masks are needed](#14-why-masks-are-needed)
  - [1.5 Why attention replaced recurrence in LLMs](#15-why-attention-replaced-recurrence-in-llms)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Input hidden-state matrix](#21-input-hiddenstate-matrix)
  - [2.2 Linear Q K V projections](#22-linear-q-k-v-projections)
  - [2.3 Scaled dot-product attention](#23-scaled-dotproduct-attention)
  - [2.4 Attention weights](#24-attention-weights)
  - [2.5 Causal and padding masks](#25-causal-and-padding-masks)
- [3. Core Mechanics](#3-core-mechanics)
  - [3.1 Softmax normalization](#31-softmax-normalization)
  - [3.2 Weighted value aggregation](#32-weighted-value-aggregation)
  - [3.3 Attention entropy](#33-attention-entropy)
  - [3.4 Temperature and score scale](#34-temperature-and-score-scale)
  - [3.5 Numerical stability](#35-numerical-stability)
- [4. Multi-Head Attention](#4-multihead-attention)
  - [4.1 Head dimensions](#41-head-dimensions)
  - [4.2 Parallel heads](#42-parallel-heads)
  - [4.3 Concatenation and output projection](#43-concatenation-and-output-projection)
  - [4.4 Head specialization and redundancy](#44-head-specialization-and-redundancy)
  - [4.5 Grouped query and multi-query attention](#45-grouped-query-and-multiquery-attention)
- [5. Decoder Attention in LLMs](#5-decoder-attention-in-llms)
  - [5.1 Autoregressive causal attention](#51-autoregressive-causal-attention)
  - [5.2 KV cache](#52-kv-cache)
  - [5.3 Prefill versus decode](#53-prefill-versus-decode)
  - [5.4 Attention with positional encodings](#54-attention-with-positional-encodings)
  - [5.5 Cross-attention preview](#55-crossattention-preview)
- [6. Complexity and Efficient Attention](#6-complexity-and-efficient-attention)
  - [6.1 Quadratic token cost](#61-quadratic-token-cost)
  - [6.2 Memory layout and IO](#62-memory-layout-and-io)
  - [6.3 FlashAttention intuition](#63-flashattention-intuition)
  - [6.4 Sparse and local attention preview](#64-sparse-and-local-attention-preview)
  - [6.5 Long-context diagnostics](#65-longcontext-diagnostics)
- [7. Interpretation and Diagnostics](#7-interpretation-and-diagnostics)
  - [7.1 Attention maps](#71-attention-maps)
  - [7.2 Mask tests](#72-mask-tests)
  - [7.3 Attention entropy dashboards](#73-attention-entropy-dashboards)
  - [7.4 Head ablation](#74-head-ablation)
  - [7.5 Attribution caveats](#75-attribution-caveats)
- [8. AI Applications](#8-ai-applications)
  - [8.1 In-context learning](#81-incontext-learning)
  - [8.2 Retrieval augmented generation](#82-retrieval-augmented-generation)
  - [8.3 Tool and chat delimiters](#83-tool-and-chat-delimiters)
  - [8.4 Long document modeling](#84-long-document-modeling)
  - [8.5 Safety and leakage](#85-safety-and-leakage)
- [9. Common Mistakes](#9-common-mistakes)
- [10. Exercises](#10-exercises)
- [11. Why This Matters for AI](#11-why-this-matters-for-ai)
- [12. Conceptual Bridge](#12-conceptual-bridge)
- [References](#references)

---

## 1. Intuition

Intuition explains how transformer layers route information across sequence positions using differentiable, mask-aware retrieval.

### 1.1 Attention as soft retrieval

**Purpose.** Attention as soft retrieval focuses on why each token reads from other token states. This is a core part of how transformer layers turn a sequence of embeddings into context-aware hidden states.

$$Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.$$

**Operational definition.**

Attention lets each token form a query, compare it against key vectors, and read a weighted mixture of value vectors.

**Worked reading.**

A token representing `it` can assign high weight to an earlier noun phrase, causing the next hidden state to mix in information from that earlier position.

| Object | Shape | Meaning |
| --- | --- | --- |
| $X$ | $T\times d_{\mathrm{model}}$ | hidden states entering the layer |
| $Q,K$ | $T\times d_k$ | query and key address vectors |
| $V$ | $T\times d_v$ | value payload vectors |
| $S=QK^\top/\sqrt{d_k}$ | $T\times T$ | compatibility scores |
| $A=\operatorname{softmax}(S+M)$ | $T\times T$ | attention weights |
| $Y=AV$ | $T\times d_v$ | mixed output values |

Examples:

1. pronoun resolution.
2. copying from context.
3. retrieved document use.

Non-examples:

1. fixed convolution window only.
2. one hidden state with no content-based mixing.

**Derivation habit.**

1. Write the shapes of $X,Q,K,V,S,A,Y$.
2. Add masks before softmax, not after.
3. Check every attention row sums to one over visible keys.
4. Separate mathematical attention from kernel implementation details.
5. For LLM serving, distinguish prefill attention from decode attention with a KV cache.

**Implementation lens.**

A correct attention implementation is mostly a shape and masking discipline. The bug that hurts language modeling most is often not the matrix multiplication; it is allowing a token to see future positions or padding tokens.

For efficient inference, the formula stays the same but the workload changes. During prefill, the model processes a full prompt. During decode, it adds one query at a time while reading cached keys and values from previous tokens.

For interpretation, attention weights are useful traces of information flow, but they are not the whole model explanation. Residual connections, MLPs, layer norms, and later layers can change or override what a single attention map appears to show.

### 1.2 Queries keys and values as roles

**Purpose.** Queries keys and values as roles focuses on search vector, address vector, payload vector. This is a core part of how transformer layers turn a sequence of embeddings into context-aware hidden states.

$$S=\frac{QK^\top}{\sqrt{d_k}}.$$

**Operational definition.**

This concept is part of the attention mechanism that mixes token representations according to learned compatibility scores.

**Worked reading.**

The implementation habit is to write shapes, scores, masks, softmax, and value aggregation explicitly.

| Object | Shape | Meaning |
| --- | --- | --- |
| $X$ | $T\times d_{\mathrm{model}}$ | hidden states entering the layer |
| $Q,K$ | $T\times d_k$ | query and key address vectors |
| $V$ | $T\times d_v$ | value payload vectors |
| $S=QK^\top/\sqrt{d_k}$ | $T\times T$ | compatibility scores |
| $A=\operatorname{softmax}(S+M)$ | $T\times T$ | attention weights |
| $Y=AV$ | $T\times d_v$ | mixed output values |

Examples:

1. self-attention.
2. decoder attention.
3. attention over retrieved context.

Non-examples:

1. independent token processing.
2. fixed averaging with no learned scores.

**Derivation habit.**

1. Write the shapes of $X,Q,K,V,S,A,Y$.
2. Add masks before softmax, not after.
3. Check every attention row sums to one over visible keys.
4. Separate mathematical attention from kernel implementation details.
5. For LLM serving, distinguish prefill attention from decode attention with a KV cache.

**Implementation lens.**

A correct attention implementation is mostly a shape and masking discipline. The bug that hurts language modeling most is often not the matrix multiplication; it is allowing a token to see future positions or padding tokens.

For efficient inference, the formula stays the same but the workload changes. During prefill, the model processes a full prompt. During decode, it adds one query at a time while reading cached keys and values from previous tokens.

For interpretation, attention weights are useful traces of information flow, but they are not the whole model explanation. Residual connections, MLPs, layer norms, and later layers can change or override what a single attention map appears to show.

### 1.3 Why scaling is needed

**Purpose.** Why scaling is needed focuses on variance control for dot products. This is a core part of how transformer layers turn a sequence of embeddings into context-aware hidden states.

$$A=\operatorname{softmax}(S+M),\qquad \sum_j A_{ij}=1.$$

**Operational definition.**

This concept is part of the attention mechanism that mixes token representations according to learned compatibility scores.

**Worked reading.**

The implementation habit is to write shapes, scores, masks, softmax, and value aggregation explicitly.

| Object | Shape | Meaning |
| --- | --- | --- |
| $X$ | $T\times d_{\mathrm{model}}$ | hidden states entering the layer |
| $Q,K$ | $T\times d_k$ | query and key address vectors |
| $V$ | $T\times d_v$ | value payload vectors |
| $S=QK^\top/\sqrt{d_k}$ | $T\times T$ | compatibility scores |
| $A=\operatorname{softmax}(S+M)$ | $T\times T$ | attention weights |
| $Y=AV$ | $T\times d_v$ | mixed output values |

Examples:

1. self-attention.
2. decoder attention.
3. attention over retrieved context.

Non-examples:

1. independent token processing.
2. fixed averaging with no learned scores.

**Derivation habit.**

1. Write the shapes of $X,Q,K,V,S,A,Y$.
2. Add masks before softmax, not after.
3. Check every attention row sums to one over visible keys.
4. Separate mathematical attention from kernel implementation details.
5. For LLM serving, distinguish prefill attention from decode attention with a KV cache.

**Implementation lens.**

A correct attention implementation is mostly a shape and masking discipline. The bug that hurts language modeling most is often not the matrix multiplication; it is allowing a token to see future positions or padding tokens.

For efficient inference, the formula stays the same but the workload changes. During prefill, the model processes a full prompt. During decode, it adds one query at a time while reading cached keys and values from previous tokens.

For interpretation, attention weights are useful traces of information flow, but they are not the whole model explanation. Residual connections, MLPs, layer norms, and later layers can change or override what a single attention map appears to show.

### 1.4 Why masks are needed

**Purpose.** Why masks are needed focuses on causality padding and visibility constraints. This is a core part of how transformer layers turn a sequence of embeddings into context-aware hidden states.

$$Y=AV.$$

**Operational definition.**

A mask changes which key positions a query is allowed to see by adding large negative values to forbidden logits before softmax.

**Worked reading.**

In decoder-only language modeling, token $i$ may attend to positions $j\le i$ but not to future positions $j>i$.

| Object | Shape | Meaning |
| --- | --- | --- |
| $X$ | $T\times d_{\mathrm{model}}$ | hidden states entering the layer |
| $Q,K$ | $T\times d_k$ | query and key address vectors |
| $V$ | $T\times d_v$ | value payload vectors |
| $S=QK^\top/\sqrt{d_k}$ | $T\times T$ | compatibility scores |
| $A=\operatorname{softmax}(S+M)$ | $T\times T$ | attention weights |
| $Y=AV$ | $T\times d_v$ | mixed output values |

Examples:

1. causal masks.
2. padding masks.
3. structured prompt masks.

Non-examples:

1. zeroing output after softmax.
2. trusting data order without a mask.

**Derivation habit.**

1. Write the shapes of $X,Q,K,V,S,A,Y$.
2. Add masks before softmax, not after.
3. Check every attention row sums to one over visible keys.
4. Separate mathematical attention from kernel implementation details.
5. For LLM serving, distinguish prefill attention from decode attention with a KV cache.

**Implementation lens.**

A correct attention implementation is mostly a shape and masking discipline. The bug that hurts language modeling most is often not the matrix multiplication; it is allowing a token to see future positions or padding tokens.

For efficient inference, the formula stays the same but the workload changes. During prefill, the model processes a full prompt. During decode, it adds one query at a time while reading cached keys and values from previous tokens.

For interpretation, attention weights are useful traces of information flow, but they are not the whole model explanation. Residual connections, MLPs, layer norms, and later layers can change or override what a single attention map appears to show.

### 1.5 Why attention replaced recurrence in LLMs

**Purpose.** Why attention replaced recurrence in LLMs focuses on parallel sequence mixing. This is a core part of how transformer layers turn a sequence of embeddings into context-aware hidden states.

$$\operatorname{MHA}(X)=\operatorname{Concat}(H_1,\ldots,H_h)W_O.$$

**Operational definition.**

This concept is part of the attention mechanism that mixes token representations according to learned compatibility scores.

**Worked reading.**

The implementation habit is to write shapes, scores, masks, softmax, and value aggregation explicitly.

| Object | Shape | Meaning |
| --- | --- | --- |
| $X$ | $T\times d_{\mathrm{model}}$ | hidden states entering the layer |
| $Q,K$ | $T\times d_k$ | query and key address vectors |
| $V$ | $T\times d_v$ | value payload vectors |
| $S=QK^\top/\sqrt{d_k}$ | $T\times T$ | compatibility scores |
| $A=\operatorname{softmax}(S+M)$ | $T\times T$ | attention weights |
| $Y=AV$ | $T\times d_v$ | mixed output values |

Examples:

1. self-attention.
2. decoder attention.
3. attention over retrieved context.

Non-examples:

1. independent token processing.
2. fixed averaging with no learned scores.

**Derivation habit.**

1. Write the shapes of $X,Q,K,V,S,A,Y$.
2. Add masks before softmax, not after.
3. Check every attention row sums to one over visible keys.
4. Separate mathematical attention from kernel implementation details.
5. For LLM serving, distinguish prefill attention from decode attention with a KV cache.

**Implementation lens.**

A correct attention implementation is mostly a shape and masking discipline. The bug that hurts language modeling most is often not the matrix multiplication; it is allowing a token to see future positions or padding tokens.

For efficient inference, the formula stays the same but the workload changes. During prefill, the model processes a full prompt. During decode, it adds one query at a time while reading cached keys and values from previous tokens.

For interpretation, attention weights are useful traces of information flow, but they are not the whole model explanation. Residual connections, MLPs, layer norms, and later layers can change or override what a single attention map appears to show.

## 2. Formal Definitions

Formal Definitions explains how transformer layers route information across sequence positions using differentiable, mask-aware retrieval.

### 2.1 Input hidden-state matrix

**Purpose.** Input hidden-state matrix focuses on the $T\times d_{\mathrm{model}}$ sequence matrix. This is a core part of how transformer layers turn a sequence of embeddings into context-aware hidden states.

$$S=\frac{QK^\top}{\sqrt{d_k}}.$$

**Operational definition.**

This concept is part of the attention mechanism that mixes token representations according to learned compatibility scores.

**Worked reading.**

The implementation habit is to write shapes, scores, masks, softmax, and value aggregation explicitly.

| Object | Shape | Meaning |
| --- | --- | --- |
| $X$ | $T\times d_{\mathrm{model}}$ | hidden states entering the layer |
| $Q,K$ | $T\times d_k$ | query and key address vectors |
| $V$ | $T\times d_v$ | value payload vectors |
| $S=QK^\top/\sqrt{d_k}$ | $T\times T$ | compatibility scores |
| $A=\operatorname{softmax}(S+M)$ | $T\times T$ | attention weights |
| $Y=AV$ | $T\times d_v$ | mixed output values |

Examples:

1. self-attention.
2. decoder attention.
3. attention over retrieved context.

Non-examples:

1. independent token processing.
2. fixed averaging with no learned scores.

**Derivation habit.**

1. Write the shapes of $X,Q,K,V,S,A,Y$.
2. Add masks before softmax, not after.
3. Check every attention row sums to one over visible keys.
4. Separate mathematical attention from kernel implementation details.
5. For LLM serving, distinguish prefill attention from decode attention with a KV cache.

**Implementation lens.**

A correct attention implementation is mostly a shape and masking discipline. The bug that hurts language modeling most is often not the matrix multiplication; it is allowing a token to see future positions or padding tokens.

For efficient inference, the formula stays the same but the workload changes. During prefill, the model processes a full prompt. During decode, it adds one query at a time while reading cached keys and values from previous tokens.

For interpretation, attention weights are useful traces of information flow, but they are not the whole model explanation. Residual connections, MLPs, layer norms, and later layers can change or override what a single attention map appears to show.

### 2.2 Linear Q K V projections

**Purpose.** Linear Q K V projections focuses on $Q=XW_Q$, $K=XW_K$, $V=XW_V$. This is a core part of how transformer layers turn a sequence of embeddings into context-aware hidden states.

$$A=\operatorname{softmax}(S+M),\qquad \sum_j A_{ij}=1.$$

**Operational definition.**

This concept is part of the attention mechanism that mixes token representations according to learned compatibility scores.

**Worked reading.**

The implementation habit is to write shapes, scores, masks, softmax, and value aggregation explicitly.

| Object | Shape | Meaning |
| --- | --- | --- |
| $X$ | $T\times d_{\mathrm{model}}$ | hidden states entering the layer |
| $Q,K$ | $T\times d_k$ | query and key address vectors |
| $V$ | $T\times d_v$ | value payload vectors |
| $S=QK^\top/\sqrt{d_k}$ | $T\times T$ | compatibility scores |
| $A=\operatorname{softmax}(S+M)$ | $T\times T$ | attention weights |
| $Y=AV$ | $T\times d_v$ | mixed output values |

Examples:

1. self-attention.
2. decoder attention.
3. attention over retrieved context.

Non-examples:

1. independent token processing.
2. fixed averaging with no learned scores.

**Derivation habit.**

1. Write the shapes of $X,Q,K,V,S,A,Y$.
2. Add masks before softmax, not after.
3. Check every attention row sums to one over visible keys.
4. Separate mathematical attention from kernel implementation details.
5. For LLM serving, distinguish prefill attention from decode attention with a KV cache.

**Implementation lens.**

A correct attention implementation is mostly a shape and masking discipline. The bug that hurts language modeling most is often not the matrix multiplication; it is allowing a token to see future positions or padding tokens.

For efficient inference, the formula stays the same but the workload changes. During prefill, the model processes a full prompt. During decode, it adds one query at a time while reading cached keys and values from previous tokens.

For interpretation, attention weights are useful traces of information flow, but they are not the whole model explanation. Residual connections, MLPs, layer norms, and later layers can change or override what a single attention map appears to show.

### 2.3 Scaled dot-product attention

**Purpose.** Scaled dot-product attention focuses on $\operatorname{softmax}(QK^\top/\sqrt{d_k})V$. This is a core part of how transformer layers turn a sequence of embeddings into context-aware hidden states.

$$Y=AV.$$

**Operational definition.**

Scaled dot-product attention computes pairwise query-key scores, normalizes each row with softmax, and uses the resulting weights to average values.

**Worked reading.**

The factor $\sqrt{d_k}$ keeps random dot products from growing too large as head dimension increases.

| Object | Shape | Meaning |
| --- | --- | --- |
| $X$ | $T\times d_{\mathrm{model}}$ | hidden states entering the layer |
| $Q,K$ | $T\times d_k$ | query and key address vectors |
| $V$ | $T\times d_v$ | value payload vectors |
| $S=QK^\top/\sqrt{d_k}$ | $T\times T$ | compatibility scores |
| $A=\operatorname{softmax}(S+M)$ | $T\times T$ | attention weights |
| $Y=AV$ | $T\times d_v$ | mixed output values |

Examples:

1. transformer self-attention.
2. cross-attention.
3. decoder attention.

Non-examples:

1. nearest neighbor with hard argmax only.
2. unscaled scores with unstable softmax.

**Derivation habit.**

1. Write the shapes of $X,Q,K,V,S,A,Y$.
2. Add masks before softmax, not after.
3. Check every attention row sums to one over visible keys.
4. Separate mathematical attention from kernel implementation details.
5. For LLM serving, distinguish prefill attention from decode attention with a KV cache.

**Implementation lens.**

A correct attention implementation is mostly a shape and masking discipline. The bug that hurts language modeling most is often not the matrix multiplication; it is allowing a token to see future positions or padding tokens.

For efficient inference, the formula stays the same but the workload changes. During prefill, the model processes a full prompt. During decode, it adds one query at a time while reading cached keys and values from previous tokens.

For interpretation, attention weights are useful traces of information flow, but they are not the whole model explanation. Residual connections, MLPs, layer norms, and later layers can change or override what a single attention map appears to show.

### 2.4 Attention weights

**Purpose.** Attention weights focuses on row-stochastic probability-like matrices. This is a core part of how transformer layers turn a sequence of embeddings into context-aware hidden states.

$$\operatorname{MHA}(X)=\operatorname{Concat}(H_1,\ldots,H_h)W_O.$$

**Operational definition.**

This concept is part of the attention mechanism that mixes token representations according to learned compatibility scores.

**Worked reading.**

The implementation habit is to write shapes, scores, masks, softmax, and value aggregation explicitly.

| Object | Shape | Meaning |
| --- | --- | --- |
| $X$ | $T\times d_{\mathrm{model}}$ | hidden states entering the layer |
| $Q,K$ | $T\times d_k$ | query and key address vectors |
| $V$ | $T\times d_v$ | value payload vectors |
| $S=QK^\top/\sqrt{d_k}$ | $T\times T$ | compatibility scores |
| $A=\operatorname{softmax}(S+M)$ | $T\times T$ | attention weights |
| $Y=AV$ | $T\times d_v$ | mixed output values |

Examples:

1. self-attention.
2. decoder attention.
3. attention over retrieved context.

Non-examples:

1. independent token processing.
2. fixed averaging with no learned scores.

**Derivation habit.**

1. Write the shapes of $X,Q,K,V,S,A,Y$.
2. Add masks before softmax, not after.
3. Check every attention row sums to one over visible keys.
4. Separate mathematical attention from kernel implementation details.
5. For LLM serving, distinguish prefill attention from decode attention with a KV cache.

**Implementation lens.**

A correct attention implementation is mostly a shape and masking discipline. The bug that hurts language modeling most is often not the matrix multiplication; it is allowing a token to see future positions or padding tokens.

For efficient inference, the formula stays the same but the workload changes. During prefill, the model processes a full prompt. During decode, it adds one query at a time while reading cached keys and values from previous tokens.

For interpretation, attention weights are useful traces of information flow, but they are not the whole model explanation. Residual connections, MLPs, layer norms, and later layers can change or override what a single attention map appears to show.

### 2.5 Causal and padding masks

**Purpose.** Causal and padding masks focuses on additive masks before softmax. This is a core part of how transformer layers turn a sequence of embeddings into context-aware hidden states.

$$\operatorname{cost}_{\mathrm{scores}}\in O(T^2d_k),\qquad \operatorname{memory}_{\mathrm{scores}}\in O(T^2).$$

**Operational definition.**

A mask changes which key positions a query is allowed to see by adding large negative values to forbidden logits before softmax.

**Worked reading.**

In decoder-only language modeling, token $i$ may attend to positions $j\le i$ but not to future positions $j>i$.

| Object | Shape | Meaning |
| --- | --- | --- |
| $X$ | $T\times d_{\mathrm{model}}$ | hidden states entering the layer |
| $Q,K$ | $T\times d_k$ | query and key address vectors |
| $V$ | $T\times d_v$ | value payload vectors |
| $S=QK^\top/\sqrt{d_k}$ | $T\times T$ | compatibility scores |
| $A=\operatorname{softmax}(S+M)$ | $T\times T$ | attention weights |
| $Y=AV$ | $T\times d_v$ | mixed output values |

Examples:

1. causal masks.
2. padding masks.
3. structured prompt masks.

Non-examples:

1. zeroing output after softmax.
2. trusting data order without a mask.

**Derivation habit.**

1. Write the shapes of $X,Q,K,V,S,A,Y$.
2. Add masks before softmax, not after.
3. Check every attention row sums to one over visible keys.
4. Separate mathematical attention from kernel implementation details.
5. For LLM serving, distinguish prefill attention from decode attention with a KV cache.

**Implementation lens.**

A correct attention implementation is mostly a shape and masking discipline. The bug that hurts language modeling most is often not the matrix multiplication; it is allowing a token to see future positions or padding tokens.

For efficient inference, the formula stays the same but the workload changes. During prefill, the model processes a full prompt. During decode, it adds one query at a time while reading cached keys and values from previous tokens.

For interpretation, attention weights are useful traces of information flow, but they are not the whole model explanation. Residual connections, MLPs, layer norms, and later layers can change or override what a single attention map appears to show.

## 3. Core Mechanics

Core Mechanics explains how transformer layers route information across sequence positions using differentiable, mask-aware retrieval.

### 3.1 Softmax normalization

**Purpose.** Softmax normalization focuses on turning scores into weights. This is a core part of how transformer layers turn a sequence of embeddings into context-aware hidden states.

$$A=\operatorname{softmax}(S+M),\qquad \sum_j A_{ij}=1.$$

**Operational definition.**

This concept is part of the attention mechanism that mixes token representations according to learned compatibility scores.

**Worked reading.**

The implementation habit is to write shapes, scores, masks, softmax, and value aggregation explicitly.

| Object | Shape | Meaning |
| --- | --- | --- |
| $X$ | $T\times d_{\mathrm{model}}$ | hidden states entering the layer |
| $Q,K$ | $T\times d_k$ | query and key address vectors |
| $V$ | $T\times d_v$ | value payload vectors |
| $S=QK^\top/\sqrt{d_k}$ | $T\times T$ | compatibility scores |
| $A=\operatorname{softmax}(S+M)$ | $T\times T$ | attention weights |
| $Y=AV$ | $T\times d_v$ | mixed output values |

Examples:

1. self-attention.
2. decoder attention.
3. attention over retrieved context.

Non-examples:

1. independent token processing.
2. fixed averaging with no learned scores.

**Derivation habit.**

1. Write the shapes of $X,Q,K,V,S,A,Y$.
2. Add masks before softmax, not after.
3. Check every attention row sums to one over visible keys.
4. Separate mathematical attention from kernel implementation details.
5. For LLM serving, distinguish prefill attention from decode attention with a KV cache.

**Implementation lens.**

A correct attention implementation is mostly a shape and masking discipline. The bug that hurts language modeling most is often not the matrix multiplication; it is allowing a token to see future positions or padding tokens.

For efficient inference, the formula stays the same but the workload changes. During prefill, the model processes a full prompt. During decode, it adds one query at a time while reading cached keys and values from previous tokens.

For interpretation, attention weights are useful traces of information flow, but they are not the whole model explanation. Residual connections, MLPs, layer norms, and later layers can change or override what a single attention map appears to show.

### 3.2 Weighted value aggregation

**Purpose.** Weighted value aggregation focuses on convex combinations of value vectors. This is a core part of how transformer layers turn a sequence of embeddings into context-aware hidden states.

$$Y=AV.$$

**Operational definition.**

This concept is part of the attention mechanism that mixes token representations according to learned compatibility scores.

**Worked reading.**

The implementation habit is to write shapes, scores, masks, softmax, and value aggregation explicitly.

| Object | Shape | Meaning |
| --- | --- | --- |
| $X$ | $T\times d_{\mathrm{model}}$ | hidden states entering the layer |
| $Q,K$ | $T\times d_k$ | query and key address vectors |
| $V$ | $T\times d_v$ | value payload vectors |
| $S=QK^\top/\sqrt{d_k}$ | $T\times T$ | compatibility scores |
| $A=\operatorname{softmax}(S+M)$ | $T\times T$ | attention weights |
| $Y=AV$ | $T\times d_v$ | mixed output values |

Examples:

1. self-attention.
2. decoder attention.
3. attention over retrieved context.

Non-examples:

1. independent token processing.
2. fixed averaging with no learned scores.

**Derivation habit.**

1. Write the shapes of $X,Q,K,V,S,A,Y$.
2. Add masks before softmax, not after.
3. Check every attention row sums to one over visible keys.
4. Separate mathematical attention from kernel implementation details.
5. For LLM serving, distinguish prefill attention from decode attention with a KV cache.

**Implementation lens.**

A correct attention implementation is mostly a shape and masking discipline. The bug that hurts language modeling most is often not the matrix multiplication; it is allowing a token to see future positions or padding tokens.

For efficient inference, the formula stays the same but the workload changes. During prefill, the model processes a full prompt. During decode, it adds one query at a time while reading cached keys and values from previous tokens.

For interpretation, attention weights are useful traces of information flow, but they are not the whole model explanation. Residual connections, MLPs, layer norms, and later layers can change or override what a single attention map appears to show.

### 3.3 Attention entropy

**Purpose.** Attention entropy focuses on sharp versus diffuse attention. This is a core part of how transformer layers turn a sequence of embeddings into context-aware hidden states.

$$\operatorname{MHA}(X)=\operatorname{Concat}(H_1,\ldots,H_h)W_O.$$

**Operational definition.**

Attention diagnostics inspect weights, entropy, masks, and head importance, but they do not by themselves prove causal explanations.

**Worked reading.**

A low-entropy row means one or a few keys dominate; a high-entropy row means information is mixed broadly.

| Object | Shape | Meaning |
| --- | --- | --- |
| $X$ | $T\times d_{\mathrm{model}}$ | hidden states entering the layer |
| $Q,K$ | $T\times d_k$ | query and key address vectors |
| $V$ | $T\times d_v$ | value payload vectors |
| $S=QK^\top/\sqrt{d_k}$ | $T\times T$ | compatibility scores |
| $A=\operatorname{softmax}(S+M)$ | $T\times T$ | attention weights |
| $Y=AV$ | $T\times d_v$ | mixed output values |

Examples:

1. attention heatmaps.
2. head ablations.
3. entropy dashboards.

Non-examples:

1. claiming attention weight equals explanation.
2. inspecting only one prompt.

**Derivation habit.**

1. Write the shapes of $X,Q,K,V,S,A,Y$.
2. Add masks before softmax, not after.
3. Check every attention row sums to one over visible keys.
4. Separate mathematical attention from kernel implementation details.
5. For LLM serving, distinguish prefill attention from decode attention with a KV cache.

**Implementation lens.**

A correct attention implementation is mostly a shape and masking discipline. The bug that hurts language modeling most is often not the matrix multiplication; it is allowing a token to see future positions or padding tokens.

For efficient inference, the formula stays the same but the workload changes. During prefill, the model processes a full prompt. During decode, it adds one query at a time while reading cached keys and values from previous tokens.

For interpretation, attention weights are useful traces of information flow, but they are not the whole model explanation. Residual connections, MLPs, layer norms, and later layers can change or override what a single attention map appears to show.

### 3.4 Temperature and score scale

**Purpose.** Temperature and score scale focuses on how scaling changes concentration. This is a core part of how transformer layers turn a sequence of embeddings into context-aware hidden states.

$$\operatorname{cost}_{\mathrm{scores}}\in O(T^2d_k),\qquad \operatorname{memory}_{\mathrm{scores}}\in O(T^2).$$

**Operational definition.**

This concept is part of the attention mechanism that mixes token representations according to learned compatibility scores.

**Worked reading.**

The implementation habit is to write shapes, scores, masks, softmax, and value aggregation explicitly.

| Object | Shape | Meaning |
| --- | --- | --- |
| $X$ | $T\times d_{\mathrm{model}}$ | hidden states entering the layer |
| $Q,K$ | $T\times d_k$ | query and key address vectors |
| $V$ | $T\times d_v$ | value payload vectors |
| $S=QK^\top/\sqrt{d_k}$ | $T\times T$ | compatibility scores |
| $A=\operatorname{softmax}(S+M)$ | $T\times T$ | attention weights |
| $Y=AV$ | $T\times d_v$ | mixed output values |

Examples:

1. self-attention.
2. decoder attention.
3. attention over retrieved context.

Non-examples:

1. independent token processing.
2. fixed averaging with no learned scores.

**Derivation habit.**

1. Write the shapes of $X,Q,K,V,S,A,Y$.
2. Add masks before softmax, not after.
3. Check every attention row sums to one over visible keys.
4. Separate mathematical attention from kernel implementation details.
5. For LLM serving, distinguish prefill attention from decode attention with a KV cache.

**Implementation lens.**

A correct attention implementation is mostly a shape and masking discipline. The bug that hurts language modeling most is often not the matrix multiplication; it is allowing a token to see future positions or padding tokens.

For efficient inference, the formula stays the same but the workload changes. During prefill, the model processes a full prompt. During decode, it adds one query at a time while reading cached keys and values from previous tokens.

For interpretation, attention weights are useful traces of information flow, but they are not the whole model explanation. Residual connections, MLPs, layer norms, and later layers can change or override what a single attention map appears to show.

### 3.5 Numerical stability

**Purpose.** Numerical stability focuses on subtracting row maxima before exponentiation. This is a core part of how transformer layers turn a sequence of embeddings into context-aware hidden states.

$$H(A_i)=-\sum_j A_{ij}\log A_{ij}.$$

**Operational definition.**

This concept is part of the attention mechanism that mixes token representations according to learned compatibility scores.

**Worked reading.**

The implementation habit is to write shapes, scores, masks, softmax, and value aggregation explicitly.

| Object | Shape | Meaning |
| --- | --- | --- |
| $X$ | $T\times d_{\mathrm{model}}$ | hidden states entering the layer |
| $Q,K$ | $T\times d_k$ | query and key address vectors |
| $V$ | $T\times d_v$ | value payload vectors |
| $S=QK^\top/\sqrt{d_k}$ | $T\times T$ | compatibility scores |
| $A=\operatorname{softmax}(S+M)$ | $T\times T$ | attention weights |
| $Y=AV$ | $T\times d_v$ | mixed output values |

Examples:

1. self-attention.
2. decoder attention.
3. attention over retrieved context.

Non-examples:

1. independent token processing.
2. fixed averaging with no learned scores.

**Derivation habit.**

1. Write the shapes of $X,Q,K,V,S,A,Y$.
2. Add masks before softmax, not after.
3. Check every attention row sums to one over visible keys.
4. Separate mathematical attention from kernel implementation details.
5. For LLM serving, distinguish prefill attention from decode attention with a KV cache.

**Implementation lens.**

A correct attention implementation is mostly a shape and masking discipline. The bug that hurts language modeling most is often not the matrix multiplication; it is allowing a token to see future positions or padding tokens.

For efficient inference, the formula stays the same but the workload changes. During prefill, the model processes a full prompt. During decode, it adds one query at a time while reading cached keys and values from previous tokens.

For interpretation, attention weights are useful traces of information flow, but they are not the whole model explanation. Residual connections, MLPs, layer norms, and later layers can change or override what a single attention map appears to show.

## 4. Multi-Head Attention

Multi-Head Attention explains how transformer layers route information across sequence positions using differentiable, mask-aware retrieval.

### 4.1 Head dimensions

**Purpose.** Head dimensions focuses on splitting representation into multiple subspaces. This is a core part of how transformer layers turn a sequence of embeddings into context-aware hidden states.

$$Y=AV.$$

**Operational definition.**

Multi-head attention runs several attention mechanisms in parallel using different learned projections.

**Worked reading.**

Each head has width $d_k=d_{\mathrm{model}}/h$ in the standard design, then head outputs are concatenated and projected.

| Object | Shape | Meaning |
| --- | --- | --- |
| $X$ | $T\times d_{\mathrm{model}}$ | hidden states entering the layer |
| $Q,K$ | $T\times d_k$ | query and key address vectors |
| $V$ | $T\times d_v$ | value payload vectors |
| $S=QK^\top/\sqrt{d_k}$ | $T\times T$ | compatibility scores |
| $A=\operatorname{softmax}(S+M)$ | $T\times T$ | attention weights |
| $Y=AV$ | $T\times d_v$ | mixed output values |

Examples:

1. syntax-like heads.
2. copy heads.
3. multi-query attention.

Non-examples:

1. one monolithic attention map only.
2. duplicating the same head without learned projections.

**Derivation habit.**

1. Write the shapes of $X,Q,K,V,S,A,Y$.
2. Add masks before softmax, not after.
3. Check every attention row sums to one over visible keys.
4. Separate mathematical attention from kernel implementation details.
5. For LLM serving, distinguish prefill attention from decode attention with a KV cache.

**Implementation lens.**

A correct attention implementation is mostly a shape and masking discipline. The bug that hurts language modeling most is often not the matrix multiplication; it is allowing a token to see future positions or padding tokens.

For efficient inference, the formula stays the same but the workload changes. During prefill, the model processes a full prompt. During decode, it adds one query at a time while reading cached keys and values from previous tokens.

For interpretation, attention weights are useful traces of information flow, but they are not the whole model explanation. Residual connections, MLPs, layer norms, and later layers can change or override what a single attention map appears to show.

### 4.2 Parallel heads

**Purpose.** Parallel heads focuses on different learned projections over the same sequence. This is a core part of how transformer layers turn a sequence of embeddings into context-aware hidden states.

$$\operatorname{MHA}(X)=\operatorname{Concat}(H_1,\ldots,H_h)W_O.$$

**Operational definition.**

Multi-head attention runs several attention mechanisms in parallel using different learned projections.

**Worked reading.**

Each head has width $d_k=d_{\mathrm{model}}/h$ in the standard design, then head outputs are concatenated and projected.

| Object | Shape | Meaning |
| --- | --- | --- |
| $X$ | $T\times d_{\mathrm{model}}$ | hidden states entering the layer |
| $Q,K$ | $T\times d_k$ | query and key address vectors |
| $V$ | $T\times d_v$ | value payload vectors |
| $S=QK^\top/\sqrt{d_k}$ | $T\times T$ | compatibility scores |
| $A=\operatorname{softmax}(S+M)$ | $T\times T$ | attention weights |
| $Y=AV$ | $T\times d_v$ | mixed output values |

Examples:

1. syntax-like heads.
2. copy heads.
3. multi-query attention.

Non-examples:

1. one monolithic attention map only.
2. duplicating the same head without learned projections.

**Derivation habit.**

1. Write the shapes of $X,Q,K,V,S,A,Y$.
2. Add masks before softmax, not after.
3. Check every attention row sums to one over visible keys.
4. Separate mathematical attention from kernel implementation details.
5. For LLM serving, distinguish prefill attention from decode attention with a KV cache.

**Implementation lens.**

A correct attention implementation is mostly a shape and masking discipline. The bug that hurts language modeling most is often not the matrix multiplication; it is allowing a token to see future positions or padding tokens.

For efficient inference, the formula stays the same but the workload changes. During prefill, the model processes a full prompt. During decode, it adds one query at a time while reading cached keys and values from previous tokens.

For interpretation, attention weights are useful traces of information flow, but they are not the whole model explanation. Residual connections, MLPs, layer norms, and later layers can change or override what a single attention map appears to show.

### 4.3 Concatenation and output projection

**Purpose.** Concatenation and output projection focuses on returning to model width. This is a core part of how transformer layers turn a sequence of embeddings into context-aware hidden states.

$$\operatorname{cost}_{\mathrm{scores}}\in O(T^2d_k),\qquad \operatorname{memory}_{\mathrm{scores}}\in O(T^2).$$

**Operational definition.**

This concept is part of the attention mechanism that mixes token representations according to learned compatibility scores.

**Worked reading.**

The implementation habit is to write shapes, scores, masks, softmax, and value aggregation explicitly.

| Object | Shape | Meaning |
| --- | --- | --- |
| $X$ | $T\times d_{\mathrm{model}}$ | hidden states entering the layer |
| $Q,K$ | $T\times d_k$ | query and key address vectors |
| $V$ | $T\times d_v$ | value payload vectors |
| $S=QK^\top/\sqrt{d_k}$ | $T\times T$ | compatibility scores |
| $A=\operatorname{softmax}(S+M)$ | $T\times T$ | attention weights |
| $Y=AV$ | $T\times d_v$ | mixed output values |

Examples:

1. self-attention.
2. decoder attention.
3. attention over retrieved context.

Non-examples:

1. independent token processing.
2. fixed averaging with no learned scores.

**Derivation habit.**

1. Write the shapes of $X,Q,K,V,S,A,Y$.
2. Add masks before softmax, not after.
3. Check every attention row sums to one over visible keys.
4. Separate mathematical attention from kernel implementation details.
5. For LLM serving, distinguish prefill attention from decode attention with a KV cache.

**Implementation lens.**

A correct attention implementation is mostly a shape and masking discipline. The bug that hurts language modeling most is often not the matrix multiplication; it is allowing a token to see future positions or padding tokens.

For efficient inference, the formula stays the same but the workload changes. During prefill, the model processes a full prompt. During decode, it adds one query at a time while reading cached keys and values from previous tokens.

For interpretation, attention weights are useful traces of information flow, but they are not the whole model explanation. Residual connections, MLPs, layer norms, and later layers can change or override what a single attention map appears to show.

### 4.4 Head specialization and redundancy

**Purpose.** Head specialization and redundancy focuses on why heads can be interpretable or unused. This is a core part of how transformer layers turn a sequence of embeddings into context-aware hidden states.

$$H(A_i)=-\sum_j A_{ij}\log A_{ij}.$$

**Operational definition.**

Multi-head attention runs several attention mechanisms in parallel using different learned projections.

**Worked reading.**

Each head has width $d_k=d_{\mathrm{model}}/h$ in the standard design, then head outputs are concatenated and projected.

| Object | Shape | Meaning |
| --- | --- | --- |
| $X$ | $T\times d_{\mathrm{model}}$ | hidden states entering the layer |
| $Q,K$ | $T\times d_k$ | query and key address vectors |
| $V$ | $T\times d_v$ | value payload vectors |
| $S=QK^\top/\sqrt{d_k}$ | $T\times T$ | compatibility scores |
| $A=\operatorname{softmax}(S+M)$ | $T\times T$ | attention weights |
| $Y=AV$ | $T\times d_v$ | mixed output values |

Examples:

1. syntax-like heads.
2. copy heads.
3. multi-query attention.

Non-examples:

1. one monolithic attention map only.
2. duplicating the same head without learned projections.

**Derivation habit.**

1. Write the shapes of $X,Q,K,V,S,A,Y$.
2. Add masks before softmax, not after.
3. Check every attention row sums to one over visible keys.
4. Separate mathematical attention from kernel implementation details.
5. For LLM serving, distinguish prefill attention from decode attention with a KV cache.

**Implementation lens.**

A correct attention implementation is mostly a shape and masking discipline. The bug that hurts language modeling most is often not the matrix multiplication; it is allowing a token to see future positions or padding tokens.

For efficient inference, the formula stays the same but the workload changes. During prefill, the model processes a full prompt. During decode, it adds one query at a time while reading cached keys and values from previous tokens.

For interpretation, attention weights are useful traces of information flow, but they are not the whole model explanation. Residual connections, MLPs, layer norms, and later layers can change or override what a single attention map appears to show.

### 4.5 Grouped query and multi-query attention

**Purpose.** Grouped query and multi-query attention focuses on sharing K/V for inference efficiency. This is a core part of how transformer layers turn a sequence of embeddings into context-aware hidden states.

$$\operatorname{causal\ mask}_{ij}=0\text{ if }j\le i,\quad -\infty\text{ otherwise}.$$

**Operational definition.**

Multi-head attention runs several attention mechanisms in parallel using different learned projections.

**Worked reading.**

Each head has width $d_k=d_{\mathrm{model}}/h$ in the standard design, then head outputs are concatenated and projected.

| Object | Shape | Meaning |
| --- | --- | --- |
| $X$ | $T\times d_{\mathrm{model}}$ | hidden states entering the layer |
| $Q,K$ | $T\times d_k$ | query and key address vectors |
| $V$ | $T\times d_v$ | value payload vectors |
| $S=QK^\top/\sqrt{d_k}$ | $T\times T$ | compatibility scores |
| $A=\operatorname{softmax}(S+M)$ | $T\times T$ | attention weights |
| $Y=AV$ | $T\times d_v$ | mixed output values |

Examples:

1. syntax-like heads.
2. copy heads.
3. multi-query attention.

Non-examples:

1. one monolithic attention map only.
2. duplicating the same head without learned projections.

**Derivation habit.**

1. Write the shapes of $X,Q,K,V,S,A,Y$.
2. Add masks before softmax, not after.
3. Check every attention row sums to one over visible keys.
4. Separate mathematical attention from kernel implementation details.
5. For LLM serving, distinguish prefill attention from decode attention with a KV cache.

**Implementation lens.**

A correct attention implementation is mostly a shape and masking discipline. The bug that hurts language modeling most is often not the matrix multiplication; it is allowing a token to see future positions or padding tokens.

For efficient inference, the formula stays the same but the workload changes. During prefill, the model processes a full prompt. During decode, it adds one query at a time while reading cached keys and values from previous tokens.

For interpretation, attention weights are useful traces of information flow, but they are not the whole model explanation. Residual connections, MLPs, layer norms, and later layers can change or override what a single attention map appears to show.

## 5. Decoder Attention in LLMs

Decoder Attention in LLMs explains how transformer layers route information across sequence positions using differentiable, mask-aware retrieval.

### 5.1 Autoregressive causal attention

**Purpose.** Autoregressive causal attention focuses on preventing future-token leakage. This is a core part of how transformer layers turn a sequence of embeddings into context-aware hidden states.

$$\operatorname{MHA}(X)=\operatorname{Concat}(H_1,\ldots,H_h)W_O.$$

**Operational definition.**

A mask changes which key positions a query is allowed to see by adding large negative values to forbidden logits before softmax.

**Worked reading.**

In decoder-only language modeling, token $i$ may attend to positions $j\le i$ but not to future positions $j>i$.

| Object | Shape | Meaning |
| --- | --- | --- |
| $X$ | $T\times d_{\mathrm{model}}$ | hidden states entering the layer |
| $Q,K$ | $T\times d_k$ | query and key address vectors |
| $V$ | $T\times d_v$ | value payload vectors |
| $S=QK^\top/\sqrt{d_k}$ | $T\times T$ | compatibility scores |
| $A=\operatorname{softmax}(S+M)$ | $T\times T$ | attention weights |
| $Y=AV$ | $T\times d_v$ | mixed output values |

Examples:

1. causal masks.
2. padding masks.
3. structured prompt masks.

Non-examples:

1. zeroing output after softmax.
2. trusting data order without a mask.

**Derivation habit.**

1. Write the shapes of $X,Q,K,V,S,A,Y$.
2. Add masks before softmax, not after.
3. Check every attention row sums to one over visible keys.
4. Separate mathematical attention from kernel implementation details.
5. For LLM serving, distinguish prefill attention from decode attention with a KV cache.

**Implementation lens.**

A correct attention implementation is mostly a shape and masking discipline. The bug that hurts language modeling most is often not the matrix multiplication; it is allowing a token to see future positions or padding tokens.

For efficient inference, the formula stays the same but the workload changes. During prefill, the model processes a full prompt. During decode, it adds one query at a time while reading cached keys and values from previous tokens.

For interpretation, attention weights are useful traces of information flow, but they are not the whole model explanation. Residual connections, MLPs, layer norms, and later layers can change or override what a single attention map appears to show.

### 5.2 KV cache

**Purpose.** KV cache focuses on reusing past keys and values during generation. This is a core part of how transformer layers turn a sequence of embeddings into context-aware hidden states.

$$\operatorname{cost}_{\mathrm{scores}}\in O(T^2d_k),\qquad \operatorname{memory}_{\mathrm{scores}}\in O(T^2).$$

**Operational definition.**

During autoregressive generation, past keys and values can be cached so each new token computes attention against old K/V instead of recomputing the entire prefix.

**Worked reading.**

Prefill processes the whole prompt; decode appends one token at a time and reuses cached K/V tensors.

| Object | Shape | Meaning |
| --- | --- | --- |
| $X$ | $T\times d_{\mathrm{model}}$ | hidden states entering the layer |
| $Q,K$ | $T\times d_k$ | query and key address vectors |
| $V$ | $T\times d_v$ | value payload vectors |
| $S=QK^\top/\sqrt{d_k}$ | $T\times T$ | compatibility scores |
| $A=\operatorname{softmax}(S+M)$ | $T\times T$ | attention weights |
| $Y=AV$ | $T\times d_v$ | mixed output values |

Examples:

1. LLM serving.
2. streaming decode.
3. multi-query attention.

Non-examples:

1. training with full parallel sequence processing.
2. recomputing all previous keys every token.

**Derivation habit.**

1. Write the shapes of $X,Q,K,V,S,A,Y$.
2. Add masks before softmax, not after.
3. Check every attention row sums to one over visible keys.
4. Separate mathematical attention from kernel implementation details.
5. For LLM serving, distinguish prefill attention from decode attention with a KV cache.

**Implementation lens.**

A correct attention implementation is mostly a shape and masking discipline. The bug that hurts language modeling most is often not the matrix multiplication; it is allowing a token to see future positions or padding tokens.

For efficient inference, the formula stays the same but the workload changes. During prefill, the model processes a full prompt. During decode, it adds one query at a time while reading cached keys and values from previous tokens.

For interpretation, attention weights are useful traces of information flow, but they are not the whole model explanation. Residual connections, MLPs, layer norms, and later layers can change or override what a single attention map appears to show.

### 5.3 Prefill versus decode

**Purpose.** Prefill versus decode focuses on two different attention workloads. This is a core part of how transformer layers turn a sequence of embeddings into context-aware hidden states.

$$H(A_i)=-\sum_j A_{ij}\log A_{ij}.$$

**Operational definition.**

This concept is part of the attention mechanism that mixes token representations according to learned compatibility scores.

**Worked reading.**

The implementation habit is to write shapes, scores, masks, softmax, and value aggregation explicitly.

| Object | Shape | Meaning |
| --- | --- | --- |
| $X$ | $T\times d_{\mathrm{model}}$ | hidden states entering the layer |
| $Q,K$ | $T\times d_k$ | query and key address vectors |
| $V$ | $T\times d_v$ | value payload vectors |
| $S=QK^\top/\sqrt{d_k}$ | $T\times T$ | compatibility scores |
| $A=\operatorname{softmax}(S+M)$ | $T\times T$ | attention weights |
| $Y=AV$ | $T\times d_v$ | mixed output values |

Examples:

1. self-attention.
2. decoder attention.
3. attention over retrieved context.

Non-examples:

1. independent token processing.
2. fixed averaging with no learned scores.

**Derivation habit.**

1. Write the shapes of $X,Q,K,V,S,A,Y$.
2. Add masks before softmax, not after.
3. Check every attention row sums to one over visible keys.
4. Separate mathematical attention from kernel implementation details.
5. For LLM serving, distinguish prefill attention from decode attention with a KV cache.

**Implementation lens.**

A correct attention implementation is mostly a shape and masking discipline. The bug that hurts language modeling most is often not the matrix multiplication; it is allowing a token to see future positions or padding tokens.

For efficient inference, the formula stays the same but the workload changes. During prefill, the model processes a full prompt. During decode, it adds one query at a time while reading cached keys and values from previous tokens.

For interpretation, attention weights are useful traces of information flow, but they are not the whole model explanation. Residual connections, MLPs, layer norms, and later layers can change or override what a single attention map appears to show.

### 5.4 Attention with positional encodings

**Purpose.** Attention with positional encodings focuses on how RoPE and ALiBi modify scores. This is a core part of how transformer layers turn a sequence of embeddings into context-aware hidden states.

$$\operatorname{causal\ mask}_{ij}=0\text{ if }j\le i,\quad -\infty\text{ otherwise}.$$

**Operational definition.**

This concept is part of the attention mechanism that mixes token representations according to learned compatibility scores.

**Worked reading.**

The implementation habit is to write shapes, scores, masks, softmax, and value aggregation explicitly.

| Object | Shape | Meaning |
| --- | --- | --- |
| $X$ | $T\times d_{\mathrm{model}}$ | hidden states entering the layer |
| $Q,K$ | $T\times d_k$ | query and key address vectors |
| $V$ | $T\times d_v$ | value payload vectors |
| $S=QK^\top/\sqrt{d_k}$ | $T\times T$ | compatibility scores |
| $A=\operatorname{softmax}(S+M)$ | $T\times T$ | attention weights |
| $Y=AV$ | $T\times d_v$ | mixed output values |

Examples:

1. self-attention.
2. decoder attention.
3. attention over retrieved context.

Non-examples:

1. independent token processing.
2. fixed averaging with no learned scores.

**Derivation habit.**

1. Write the shapes of $X,Q,K,V,S,A,Y$.
2. Add masks before softmax, not after.
3. Check every attention row sums to one over visible keys.
4. Separate mathematical attention from kernel implementation details.
5. For LLM serving, distinguish prefill attention from decode attention with a KV cache.

**Implementation lens.**

A correct attention implementation is mostly a shape and masking discipline. The bug that hurts language modeling most is often not the matrix multiplication; it is allowing a token to see future positions or padding tokens.

For efficient inference, the formula stays the same but the workload changes. During prefill, the model processes a full prompt. During decode, it adds one query at a time while reading cached keys and values from previous tokens.

For interpretation, attention weights are useful traces of information flow, but they are not the whole model explanation. Residual connections, MLPs, layer norms, and later layers can change or override what a single attention map appears to show.

### 5.5 Cross-attention preview

**Purpose.** Cross-attention preview focuses on encoder-decoder and retrieval-conditioned variants. This is a core part of how transformer layers turn a sequence of embeddings into context-aware hidden states.

$$Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.$$

**Operational definition.**

This concept is part of the attention mechanism that mixes token representations according to learned compatibility scores.

**Worked reading.**

The implementation habit is to write shapes, scores, masks, softmax, and value aggregation explicitly.

| Object | Shape | Meaning |
| --- | --- | --- |
| $X$ | $T\times d_{\mathrm{model}}$ | hidden states entering the layer |
| $Q,K$ | $T\times d_k$ | query and key address vectors |
| $V$ | $T\times d_v$ | value payload vectors |
| $S=QK^\top/\sqrt{d_k}$ | $T\times T$ | compatibility scores |
| $A=\operatorname{softmax}(S+M)$ | $T\times T$ | attention weights |
| $Y=AV$ | $T\times d_v$ | mixed output values |

Examples:

1. self-attention.
2. decoder attention.
3. attention over retrieved context.

Non-examples:

1. independent token processing.
2. fixed averaging with no learned scores.

**Derivation habit.**

1. Write the shapes of $X,Q,K,V,S,A,Y$.
2. Add masks before softmax, not after.
3. Check every attention row sums to one over visible keys.
4. Separate mathematical attention from kernel implementation details.
5. For LLM serving, distinguish prefill attention from decode attention with a KV cache.

**Implementation lens.**

A correct attention implementation is mostly a shape and masking discipline. The bug that hurts language modeling most is often not the matrix multiplication; it is allowing a token to see future positions or padding tokens.

For efficient inference, the formula stays the same but the workload changes. During prefill, the model processes a full prompt. During decode, it adds one query at a time while reading cached keys and values from previous tokens.

For interpretation, attention weights are useful traces of information flow, but they are not the whole model explanation. Residual connections, MLPs, layer norms, and later layers can change or override what a single attention map appears to show.

## 6. Complexity and Efficient Attention

Complexity and Efficient Attention explains how transformer layers route information across sequence positions using differentiable, mask-aware retrieval.

### 6.1 Quadratic token cost

**Purpose.** Quadratic token cost focuses on why $T^2$ score matrices dominate long context. This is a core part of how transformer layers turn a sequence of embeddings into context-aware hidden states.

$$\operatorname{cost}_{\mathrm{scores}}\in O(T^2d_k),\qquad \operatorname{memory}_{\mathrm{scores}}\in O(T^2).$$

**Operational definition.**

Standard attention forms all pairwise query-key scores, so score memory grows quadratically with sequence length.

**Worked reading.**

Doubling context length roughly quadruples the score matrix size, even before considering layer count and batch size.

| Object | Shape | Meaning |
| --- | --- | --- |
| $X$ | $T\times d_{\mathrm{model}}$ | hidden states entering the layer |
| $Q,K$ | $T\times d_k$ | query and key address vectors |
| $V$ | $T\times d_v$ | value payload vectors |
| $S=QK^\top/\sqrt{d_k}$ | $T\times T$ | compatibility scores |
| $A=\operatorname{softmax}(S+M)$ | $T\times T$ | attention weights |
| $Y=AV$ | $T\times d_v$ | mixed output values |

Examples:

1. long-context training.
2. KV-cache sizing.
3. FlashAttention kernels.

Non-examples:

1. linear cost assumptions.
2. ignoring memory traffic.

**Derivation habit.**

1. Write the shapes of $X,Q,K,V,S,A,Y$.
2. Add masks before softmax, not after.
3. Check every attention row sums to one over visible keys.
4. Separate mathematical attention from kernel implementation details.
5. For LLM serving, distinguish prefill attention from decode attention with a KV cache.

**Implementation lens.**

A correct attention implementation is mostly a shape and masking discipline. The bug that hurts language modeling most is often not the matrix multiplication; it is allowing a token to see future positions or padding tokens.

For efficient inference, the formula stays the same but the workload changes. During prefill, the model processes a full prompt. During decode, it adds one query at a time while reading cached keys and values from previous tokens.

For interpretation, attention weights are useful traces of information flow, but they are not the whole model explanation. Residual connections, MLPs, layer norms, and later layers can change or override what a single attention map appears to show.

### 6.2 Memory layout and IO

**Purpose.** Memory layout and IO focuses on why exact attention can be slow despite simple formulas. This is a core part of how transformer layers turn a sequence of embeddings into context-aware hidden states.

$$H(A_i)=-\sum_j A_{ij}\log A_{ij}.$$

**Operational definition.**

Standard attention forms all pairwise query-key scores, so score memory grows quadratically with sequence length.

**Worked reading.**

Doubling context length roughly quadruples the score matrix size, even before considering layer count and batch size.

| Object | Shape | Meaning |
| --- | --- | --- |
| $X$ | $T\times d_{\mathrm{model}}$ | hidden states entering the layer |
| $Q,K$ | $T\times d_k$ | query and key address vectors |
| $V$ | $T\times d_v$ | value payload vectors |
| $S=QK^\top/\sqrt{d_k}$ | $T\times T$ | compatibility scores |
| $A=\operatorname{softmax}(S+M)$ | $T\times T$ | attention weights |
| $Y=AV$ | $T\times d_v$ | mixed output values |

Examples:

1. long-context training.
2. KV-cache sizing.
3. FlashAttention kernels.

Non-examples:

1. linear cost assumptions.
2. ignoring memory traffic.

**Derivation habit.**

1. Write the shapes of $X,Q,K,V,S,A,Y$.
2. Add masks before softmax, not after.
3. Check every attention row sums to one over visible keys.
4. Separate mathematical attention from kernel implementation details.
5. For LLM serving, distinguish prefill attention from decode attention with a KV cache.

**Implementation lens.**

A correct attention implementation is mostly a shape and masking discipline. The bug that hurts language modeling most is often not the matrix multiplication; it is allowing a token to see future positions or padding tokens.

For efficient inference, the formula stays the same but the workload changes. During prefill, the model processes a full prompt. During decode, it adds one query at a time while reading cached keys and values from previous tokens.

For interpretation, attention weights are useful traces of information flow, but they are not the whole model explanation. Residual connections, MLPs, layer norms, and later layers can change or override what a single attention map appears to show.

### 6.3 FlashAttention intuition

**Purpose.** FlashAttention intuition focuses on tiling exact attention to reduce memory traffic. This is a core part of how transformer layers turn a sequence of embeddings into context-aware hidden states.

$$\operatorname{causal\ mask}_{ij}=0\text{ if }j\le i,\quad -\infty\text{ otherwise}.$$

**Operational definition.**

FlashAttention computes exact attention while avoiding materializing the full attention matrix in high-bandwidth memory.

**Worked reading.**

It tiles Q, K, and V blocks and maintains online softmax statistics so memory traffic is lower even though the mathematical result is exact.

| Object | Shape | Meaning |
| --- | --- | --- |
| $X$ | $T\times d_{\mathrm{model}}$ | hidden states entering the layer |
| $Q,K$ | $T\times d_k$ | query and key address vectors |
| $V$ | $T\times d_v$ | value payload vectors |
| $S=QK^\top/\sqrt{d_k}$ | $T\times T$ | compatibility scores |
| $A=\operatorname{softmax}(S+M)$ | $T\times T$ | attention weights |
| $Y=AV$ | $T\times d_v$ | mixed output values |

Examples:

1. long-context training.
2. GPU attention kernels.
3. memory-efficient exact attention.

Non-examples:

1. approximate sparse attention.
2. changing the attention formula.

**Derivation habit.**

1. Write the shapes of $X,Q,K,V,S,A,Y$.
2. Add masks before softmax, not after.
3. Check every attention row sums to one over visible keys.
4. Separate mathematical attention from kernel implementation details.
5. For LLM serving, distinguish prefill attention from decode attention with a KV cache.

**Implementation lens.**

A correct attention implementation is mostly a shape and masking discipline. The bug that hurts language modeling most is often not the matrix multiplication; it is allowing a token to see future positions or padding tokens.

For efficient inference, the formula stays the same but the workload changes. During prefill, the model processes a full prompt. During decode, it adds one query at a time while reading cached keys and values from previous tokens.

For interpretation, attention weights are useful traces of information flow, but they are not the whole model explanation. Residual connections, MLPs, layer norms, and later layers can change or override what a single attention map appears to show.

### 6.4 Sparse and local attention preview

**Purpose.** Sparse and local attention preview focuses on approximating visibility patterns. This is a core part of how transformer layers turn a sequence of embeddings into context-aware hidden states.

$$Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.$$

**Operational definition.**

This concept is part of the attention mechanism that mixes token representations according to learned compatibility scores.

**Worked reading.**

The implementation habit is to write shapes, scores, masks, softmax, and value aggregation explicitly.

| Object | Shape | Meaning |
| --- | --- | --- |
| $X$ | $T\times d_{\mathrm{model}}$ | hidden states entering the layer |
| $Q,K$ | $T\times d_k$ | query and key address vectors |
| $V$ | $T\times d_v$ | value payload vectors |
| $S=QK^\top/\sqrt{d_k}$ | $T\times T$ | compatibility scores |
| $A=\operatorname{softmax}(S+M)$ | $T\times T$ | attention weights |
| $Y=AV$ | $T\times d_v$ | mixed output values |

Examples:

1. self-attention.
2. decoder attention.
3. attention over retrieved context.

Non-examples:

1. independent token processing.
2. fixed averaging with no learned scores.

**Derivation habit.**

1. Write the shapes of $X,Q,K,V,S,A,Y$.
2. Add masks before softmax, not after.
3. Check every attention row sums to one over visible keys.
4. Separate mathematical attention from kernel implementation details.
5. For LLM serving, distinguish prefill attention from decode attention with a KV cache.

**Implementation lens.**

A correct attention implementation is mostly a shape and masking discipline. The bug that hurts language modeling most is often not the matrix multiplication; it is allowing a token to see future positions or padding tokens.

For efficient inference, the formula stays the same but the workload changes. During prefill, the model processes a full prompt. During decode, it adds one query at a time while reading cached keys and values from previous tokens.

For interpretation, attention weights are useful traces of information flow, but they are not the whole model explanation. Residual connections, MLPs, layer norms, and later layers can change or override what a single attention map appears to show.

### 6.5 Long-context diagnostics

**Purpose.** Long-context diagnostics focuses on checking quality cost and position behavior together. This is a core part of how transformer layers turn a sequence of embeddings into context-aware hidden states.

$$S=\frac{QK^\top}{\sqrt{d_k}}.$$

**Operational definition.**

Attention diagnostics inspect weights, entropy, masks, and head importance, but they do not by themselves prove causal explanations.

**Worked reading.**

A low-entropy row means one or a few keys dominate; a high-entropy row means information is mixed broadly.

| Object | Shape | Meaning |
| --- | --- | --- |
| $X$ | $T\times d_{\mathrm{model}}$ | hidden states entering the layer |
| $Q,K$ | $T\times d_k$ | query and key address vectors |
| $V$ | $T\times d_v$ | value payload vectors |
| $S=QK^\top/\sqrt{d_k}$ | $T\times T$ | compatibility scores |
| $A=\operatorname{softmax}(S+M)$ | $T\times T$ | attention weights |
| $Y=AV$ | $T\times d_v$ | mixed output values |

Examples:

1. attention heatmaps.
2. head ablations.
3. entropy dashboards.

Non-examples:

1. claiming attention weight equals explanation.
2. inspecting only one prompt.

**Derivation habit.**

1. Write the shapes of $X,Q,K,V,S,A,Y$.
2. Add masks before softmax, not after.
3. Check every attention row sums to one over visible keys.
4. Separate mathematical attention from kernel implementation details.
5. For LLM serving, distinguish prefill attention from decode attention with a KV cache.

**Implementation lens.**

A correct attention implementation is mostly a shape and masking discipline. The bug that hurts language modeling most is often not the matrix multiplication; it is allowing a token to see future positions or padding tokens.

For efficient inference, the formula stays the same but the workload changes. During prefill, the model processes a full prompt. During decode, it adds one query at a time while reading cached keys and values from previous tokens.

For interpretation, attention weights are useful traces of information flow, but they are not the whole model explanation. Residual connections, MLPs, layer norms, and later layers can change or override what a single attention map appears to show.

## 7. Interpretation and Diagnostics

Interpretation and Diagnostics explains how transformer layers route information across sequence positions using differentiable, mask-aware retrieval.

### 7.1 Attention maps

**Purpose.** Attention maps focuses on what weights show and what they do not prove. This is a core part of how transformer layers turn a sequence of embeddings into context-aware hidden states.

$$H(A_i)=-\sum_j A_{ij}\log A_{ij}.$$

**Operational definition.**

Attention diagnostics inspect weights, entropy, masks, and head importance, but they do not by themselves prove causal explanations.

**Worked reading.**

A low-entropy row means one or a few keys dominate; a high-entropy row means information is mixed broadly.

| Object | Shape | Meaning |
| --- | --- | --- |
| $X$ | $T\times d_{\mathrm{model}}$ | hidden states entering the layer |
| $Q,K$ | $T\times d_k$ | query and key address vectors |
| $V$ | $T\times d_v$ | value payload vectors |
| $S=QK^\top/\sqrt{d_k}$ | $T\times T$ | compatibility scores |
| $A=\operatorname{softmax}(S+M)$ | $T\times T$ | attention weights |
| $Y=AV$ | $T\times d_v$ | mixed output values |

Examples:

1. attention heatmaps.
2. head ablations.
3. entropy dashboards.

Non-examples:

1. claiming attention weight equals explanation.
2. inspecting only one prompt.

**Derivation habit.**

1. Write the shapes of $X,Q,K,V,S,A,Y$.
2. Add masks before softmax, not after.
3. Check every attention row sums to one over visible keys.
4. Separate mathematical attention from kernel implementation details.
5. For LLM serving, distinguish prefill attention from decode attention with a KV cache.

**Implementation lens.**

A correct attention implementation is mostly a shape and masking discipline. The bug that hurts language modeling most is often not the matrix multiplication; it is allowing a token to see future positions or padding tokens.

For efficient inference, the formula stays the same but the workload changes. During prefill, the model processes a full prompt. During decode, it adds one query at a time while reading cached keys and values from previous tokens.

For interpretation, attention weights are useful traces of information flow, but they are not the whole model explanation. Residual connections, MLPs, layer norms, and later layers can change or override what a single attention map appears to show.

### 7.2 Mask tests

**Purpose.** Mask tests focuses on catching leakage and padding bugs. This is a core part of how transformer layers turn a sequence of embeddings into context-aware hidden states.

$$\operatorname{causal\ mask}_{ij}=0\text{ if }j\le i,\quad -\infty\text{ otherwise}.$$

**Operational definition.**

A mask changes which key positions a query is allowed to see by adding large negative values to forbidden logits before softmax.

**Worked reading.**

In decoder-only language modeling, token $i$ may attend to positions $j\le i$ but not to future positions $j>i$.

| Object | Shape | Meaning |
| --- | --- | --- |
| $X$ | $T\times d_{\mathrm{model}}$ | hidden states entering the layer |
| $Q,K$ | $T\times d_k$ | query and key address vectors |
| $V$ | $T\times d_v$ | value payload vectors |
| $S=QK^\top/\sqrt{d_k}$ | $T\times T$ | compatibility scores |
| $A=\operatorname{softmax}(S+M)$ | $T\times T$ | attention weights |
| $Y=AV$ | $T\times d_v$ | mixed output values |

Examples:

1. causal masks.
2. padding masks.
3. structured prompt masks.

Non-examples:

1. zeroing output after softmax.
2. trusting data order without a mask.

**Derivation habit.**

1. Write the shapes of $X,Q,K,V,S,A,Y$.
2. Add masks before softmax, not after.
3. Check every attention row sums to one over visible keys.
4. Separate mathematical attention from kernel implementation details.
5. For LLM serving, distinguish prefill attention from decode attention with a KV cache.

**Implementation lens.**

A correct attention implementation is mostly a shape and masking discipline. The bug that hurts language modeling most is often not the matrix multiplication; it is allowing a token to see future positions or padding tokens.

For efficient inference, the formula stays the same but the workload changes. During prefill, the model processes a full prompt. During decode, it adds one query at a time while reading cached keys and values from previous tokens.

For interpretation, attention weights are useful traces of information flow, but they are not the whole model explanation. Residual connections, MLPs, layer norms, and later layers can change or override what a single attention map appears to show.

### 7.3 Attention entropy dashboards

**Purpose.** Attention entropy dashboards focuses on detecting collapse or over-diffusion. This is a core part of how transformer layers turn a sequence of embeddings into context-aware hidden states.

$$Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.$$

**Operational definition.**

Attention diagnostics inspect weights, entropy, masks, and head importance, but they do not by themselves prove causal explanations.

**Worked reading.**

A low-entropy row means one or a few keys dominate; a high-entropy row means information is mixed broadly.

| Object | Shape | Meaning |
| --- | --- | --- |
| $X$ | $T\times d_{\mathrm{model}}$ | hidden states entering the layer |
| $Q,K$ | $T\times d_k$ | query and key address vectors |
| $V$ | $T\times d_v$ | value payload vectors |
| $S=QK^\top/\sqrt{d_k}$ | $T\times T$ | compatibility scores |
| $A=\operatorname{softmax}(S+M)$ | $T\times T$ | attention weights |
| $Y=AV$ | $T\times d_v$ | mixed output values |

Examples:

1. attention heatmaps.
2. head ablations.
3. entropy dashboards.

Non-examples:

1. claiming attention weight equals explanation.
2. inspecting only one prompt.

**Derivation habit.**

1. Write the shapes of $X,Q,K,V,S,A,Y$.
2. Add masks before softmax, not after.
3. Check every attention row sums to one over visible keys.
4. Separate mathematical attention from kernel implementation details.
5. For LLM serving, distinguish prefill attention from decode attention with a KV cache.

**Implementation lens.**

A correct attention implementation is mostly a shape and masking discipline. The bug that hurts language modeling most is often not the matrix multiplication; it is allowing a token to see future positions or padding tokens.

For efficient inference, the formula stays the same but the workload changes. During prefill, the model processes a full prompt. During decode, it adds one query at a time while reading cached keys and values from previous tokens.

For interpretation, attention weights are useful traces of information flow, but they are not the whole model explanation. Residual connections, MLPs, layer norms, and later layers can change or override what a single attention map appears to show.

### 7.4 Head ablation

**Purpose.** Head ablation focuses on testing whether a head matters. This is a core part of how transformer layers turn a sequence of embeddings into context-aware hidden states.

$$S=\frac{QK^\top}{\sqrt{d_k}}.$$

**Operational definition.**

Multi-head attention runs several attention mechanisms in parallel using different learned projections.

**Worked reading.**

Each head has width $d_k=d_{\mathrm{model}}/h$ in the standard design, then head outputs are concatenated and projected.

| Object | Shape | Meaning |
| --- | --- | --- |
| $X$ | $T\times d_{\mathrm{model}}$ | hidden states entering the layer |
| $Q,K$ | $T\times d_k$ | query and key address vectors |
| $V$ | $T\times d_v$ | value payload vectors |
| $S=QK^\top/\sqrt{d_k}$ | $T\times T$ | compatibility scores |
| $A=\operatorname{softmax}(S+M)$ | $T\times T$ | attention weights |
| $Y=AV$ | $T\times d_v$ | mixed output values |

Examples:

1. syntax-like heads.
2. copy heads.
3. multi-query attention.

Non-examples:

1. one monolithic attention map only.
2. duplicating the same head without learned projections.

**Derivation habit.**

1. Write the shapes of $X,Q,K,V,S,A,Y$.
2. Add masks before softmax, not after.
3. Check every attention row sums to one over visible keys.
4. Separate mathematical attention from kernel implementation details.
5. For LLM serving, distinguish prefill attention from decode attention with a KV cache.

**Implementation lens.**

A correct attention implementation is mostly a shape and masking discipline. The bug that hurts language modeling most is often not the matrix multiplication; it is allowing a token to see future positions or padding tokens.

For efficient inference, the formula stays the same but the workload changes. During prefill, the model processes a full prompt. During decode, it adds one query at a time while reading cached keys and values from previous tokens.

For interpretation, attention weights are useful traces of information flow, but they are not the whole model explanation. Residual connections, MLPs, layer norms, and later layers can change or override what a single attention map appears to show.

### 7.5 Attribution caveats

**Purpose.** Attribution caveats focuses on attention is a mechanism not a full explanation. This is a core part of how transformer layers turn a sequence of embeddings into context-aware hidden states.

$$A=\operatorname{softmax}(S+M),\qquad \sum_j A_{ij}=1.$$

**Operational definition.**

Attention diagnostics inspect weights, entropy, masks, and head importance, but they do not by themselves prove causal explanations.

**Worked reading.**

A low-entropy row means one or a few keys dominate; a high-entropy row means information is mixed broadly.

| Object | Shape | Meaning |
| --- | --- | --- |
| $X$ | $T\times d_{\mathrm{model}}$ | hidden states entering the layer |
| $Q,K$ | $T\times d_k$ | query and key address vectors |
| $V$ | $T\times d_v$ | value payload vectors |
| $S=QK^\top/\sqrt{d_k}$ | $T\times T$ | compatibility scores |
| $A=\operatorname{softmax}(S+M)$ | $T\times T$ | attention weights |
| $Y=AV$ | $T\times d_v$ | mixed output values |

Examples:

1. attention heatmaps.
2. head ablations.
3. entropy dashboards.

Non-examples:

1. claiming attention weight equals explanation.
2. inspecting only one prompt.

**Derivation habit.**

1. Write the shapes of $X,Q,K,V,S,A,Y$.
2. Add masks before softmax, not after.
3. Check every attention row sums to one over visible keys.
4. Separate mathematical attention from kernel implementation details.
5. For LLM serving, distinguish prefill attention from decode attention with a KV cache.

**Implementation lens.**

A correct attention implementation is mostly a shape and masking discipline. The bug that hurts language modeling most is often not the matrix multiplication; it is allowing a token to see future positions or padding tokens.

For efficient inference, the formula stays the same but the workload changes. During prefill, the model processes a full prompt. During decode, it adds one query at a time while reading cached keys and values from previous tokens.

For interpretation, attention weights are useful traces of information flow, but they are not the whole model explanation. Residual connections, MLPs, layer norms, and later layers can change or override what a single attention map appears to show.

## 8. AI Applications

AI Applications explains how transformer layers route information across sequence positions using differentiable, mask-aware retrieval.

### 8.1 In-context learning

**Purpose.** In-context learning focuses on mixing examples and instructions inside a prompt. This is a core part of how transformer layers turn a sequence of embeddings into context-aware hidden states.

$$\operatorname{causal\ mask}_{ij}=0\text{ if }j\le i,\quad -\infty\text{ otherwise}.$$

**Operational definition.**

Attention is how prompt tokens, retrieved chunks, examples, tools, and instructions exchange information inside the model.

**Worked reading.**

A retrieved passage only helps if its tokens remain visible and the model learns to assign useful weights to them.

| Object | Shape | Meaning |
| --- | --- | --- |
| $X$ | $T\times d_{\mathrm{model}}$ | hidden states entering the layer |
| $Q,K$ | $T\times d_k$ | query and key address vectors |
| $V$ | $T\times d_v$ | value payload vectors |
| $S=QK^\top/\sqrt{d_k}$ | $T\times T$ | compatibility scores |
| $A=\operatorname{softmax}(S+M)$ | $T\times T$ | attention weights |
| $Y=AV$ | $T\times d_v$ | mixed output values |

Examples:

1. in-context learning.
2. retrieval augmented generation.
3. structured chat prompts.

Non-examples:

1. external memory never placed in context.
2. masked evidence tokens.

**Derivation habit.**

1. Write the shapes of $X,Q,K,V,S,A,Y$.
2. Add masks before softmax, not after.
3. Check every attention row sums to one over visible keys.
4. Separate mathematical attention from kernel implementation details.
5. For LLM serving, distinguish prefill attention from decode attention with a KV cache.

**Implementation lens.**

A correct attention implementation is mostly a shape and masking discipline. The bug that hurts language modeling most is often not the matrix multiplication; it is allowing a token to see future positions or padding tokens.

For efficient inference, the formula stays the same but the workload changes. During prefill, the model processes a full prompt. During decode, it adds one query at a time while reading cached keys and values from previous tokens.

For interpretation, attention weights are useful traces of information flow, but they are not the whole model explanation. Residual connections, MLPs, layer norms, and later layers can change or override what a single attention map appears to show.

### 8.2 Retrieval augmented generation

**Purpose.** Retrieval augmented generation focuses on attending over retrieved chunks. This is a core part of how transformer layers turn a sequence of embeddings into context-aware hidden states.

$$Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.$$

**Operational definition.**

This concept is part of the attention mechanism that mixes token representations according to learned compatibility scores.

**Worked reading.**

The implementation habit is to write shapes, scores, masks, softmax, and value aggregation explicitly.

| Object | Shape | Meaning |
| --- | --- | --- |
| $X$ | $T\times d_{\mathrm{model}}$ | hidden states entering the layer |
| $Q,K$ | $T\times d_k$ | query and key address vectors |
| $V$ | $T\times d_v$ | value payload vectors |
| $S=QK^\top/\sqrt{d_k}$ | $T\times T$ | compatibility scores |
| $A=\operatorname{softmax}(S+M)$ | $T\times T$ | attention weights |
| $Y=AV$ | $T\times d_v$ | mixed output values |

Examples:

1. self-attention.
2. decoder attention.
3. attention over retrieved context.

Non-examples:

1. independent token processing.
2. fixed averaging with no learned scores.

**Derivation habit.**

1. Write the shapes of $X,Q,K,V,S,A,Y$.
2. Add masks before softmax, not after.
3. Check every attention row sums to one over visible keys.
4. Separate mathematical attention from kernel implementation details.
5. For LLM serving, distinguish prefill attention from decode attention with a KV cache.

**Implementation lens.**

A correct attention implementation is mostly a shape and masking discipline. The bug that hurts language modeling most is often not the matrix multiplication; it is allowing a token to see future positions or padding tokens.

For efficient inference, the formula stays the same but the workload changes. During prefill, the model processes a full prompt. During decode, it adds one query at a time while reading cached keys and values from previous tokens.

For interpretation, attention weights are useful traces of information flow, but they are not the whole model explanation. Residual connections, MLPs, layer norms, and later layers can change or override what a single attention map appears to show.

### 8.3 Tool and chat delimiters

**Purpose.** Tool and chat delimiters focuses on attention across structured prompt boundaries. This is a core part of how transformer layers turn a sequence of embeddings into context-aware hidden states.

$$S=\frac{QK^\top}{\sqrt{d_k}}.$$

**Operational definition.**

Attention is how prompt tokens, retrieved chunks, examples, tools, and instructions exchange information inside the model.

**Worked reading.**

A retrieved passage only helps if its tokens remain visible and the model learns to assign useful weights to them.

| Object | Shape | Meaning |
| --- | --- | --- |
| $X$ | $T\times d_{\mathrm{model}}$ | hidden states entering the layer |
| $Q,K$ | $T\times d_k$ | query and key address vectors |
| $V$ | $T\times d_v$ | value payload vectors |
| $S=QK^\top/\sqrt{d_k}$ | $T\times T$ | compatibility scores |
| $A=\operatorname{softmax}(S+M)$ | $T\times T$ | attention weights |
| $Y=AV$ | $T\times d_v$ | mixed output values |

Examples:

1. in-context learning.
2. retrieval augmented generation.
3. structured chat prompts.

Non-examples:

1. external memory never placed in context.
2. masked evidence tokens.

**Derivation habit.**

1. Write the shapes of $X,Q,K,V,S,A,Y$.
2. Add masks before softmax, not after.
3. Check every attention row sums to one over visible keys.
4. Separate mathematical attention from kernel implementation details.
5. For LLM serving, distinguish prefill attention from decode attention with a KV cache.

**Implementation lens.**

A correct attention implementation is mostly a shape and masking discipline. The bug that hurts language modeling most is often not the matrix multiplication; it is allowing a token to see future positions or padding tokens.

For efficient inference, the formula stays the same but the workload changes. During prefill, the model processes a full prompt. During decode, it adds one query at a time while reading cached keys and values from previous tokens.

For interpretation, attention weights are useful traces of information flow, but they are not the whole model explanation. Residual connections, MLPs, layer norms, and later layers can change or override what a single attention map appears to show.

### 8.4 Long document modeling

**Purpose.** Long document modeling focuses on where attention cost and memory dominate. This is a core part of how transformer layers turn a sequence of embeddings into context-aware hidden states.

$$A=\operatorname{softmax}(S+M),\qquad \sum_j A_{ij}=1.$$

**Operational definition.**

Standard attention forms all pairwise query-key scores, so score memory grows quadratically with sequence length.

**Worked reading.**

Doubling context length roughly quadruples the score matrix size, even before considering layer count and batch size.

| Object | Shape | Meaning |
| --- | --- | --- |
| $X$ | $T\times d_{\mathrm{model}}$ | hidden states entering the layer |
| $Q,K$ | $T\times d_k$ | query and key address vectors |
| $V$ | $T\times d_v$ | value payload vectors |
| $S=QK^\top/\sqrt{d_k}$ | $T\times T$ | compatibility scores |
| $A=\operatorname{softmax}(S+M)$ | $T\times T$ | attention weights |
| $Y=AV$ | $T\times d_v$ | mixed output values |

Examples:

1. long-context training.
2. KV-cache sizing.
3. FlashAttention kernels.

Non-examples:

1. linear cost assumptions.
2. ignoring memory traffic.

**Derivation habit.**

1. Write the shapes of $X,Q,K,V,S,A,Y$.
2. Add masks before softmax, not after.
3. Check every attention row sums to one over visible keys.
4. Separate mathematical attention from kernel implementation details.
5. For LLM serving, distinguish prefill attention from decode attention with a KV cache.

**Implementation lens.**

A correct attention implementation is mostly a shape and masking discipline. The bug that hurts language modeling most is often not the matrix multiplication; it is allowing a token to see future positions or padding tokens.

For efficient inference, the formula stays the same but the workload changes. During prefill, the model processes a full prompt. During decode, it adds one query at a time while reading cached keys and values from previous tokens.

For interpretation, attention weights are useful traces of information flow, but they are not the whole model explanation. Residual connections, MLPs, layer norms, and later layers can change or override what a single attention map appears to show.

### 8.5 Safety and leakage

**Purpose.** Safety and leakage focuses on why masks and prompt boundaries are security-relevant. This is a core part of how transformer layers turn a sequence of embeddings into context-aware hidden states.

$$Y=AV.$$

**Operational definition.**

A mask changes which key positions a query is allowed to see by adding large negative values to forbidden logits before softmax.

**Worked reading.**

In decoder-only language modeling, token $i$ may attend to positions $j\le i$ but not to future positions $j>i$.

| Object | Shape | Meaning |
| --- | --- | --- |
| $X$ | $T\times d_{\mathrm{model}}$ | hidden states entering the layer |
| $Q,K$ | $T\times d_k$ | query and key address vectors |
| $V$ | $T\times d_v$ | value payload vectors |
| $S=QK^\top/\sqrt{d_k}$ | $T\times T$ | compatibility scores |
| $A=\operatorname{softmax}(S+M)$ | $T\times T$ | attention weights |
| $Y=AV$ | $T\times d_v$ | mixed output values |

Examples:

1. causal masks.
2. padding masks.
3. structured prompt masks.

Non-examples:

1. zeroing output after softmax.
2. trusting data order without a mask.

**Derivation habit.**

1. Write the shapes of $X,Q,K,V,S,A,Y$.
2. Add masks before softmax, not after.
3. Check every attention row sums to one over visible keys.
4. Separate mathematical attention from kernel implementation details.
5. For LLM serving, distinguish prefill attention from decode attention with a KV cache.

**Implementation lens.**

A correct attention implementation is mostly a shape and masking discipline. The bug that hurts language modeling most is often not the matrix multiplication; it is allowing a token to see future positions or padding tokens.

For efficient inference, the formula stays the same but the workload changes. During prefill, the model processes a full prompt. During decode, it adds one query at a time while reading cached keys and values from previous tokens.

For interpretation, attention weights are useful traces of information flow, but they are not the whole model explanation. Residual connections, MLPs, layer norms, and later layers can change or override what a single attention map appears to show.

## 9. Common Mistakes

| # | Mistake | Why it is wrong | Fix |
| --- | --- | --- | --- |
| 1 | Forgetting the scaling factor | Unscaled dot products can make softmax too sharp. | Divide by $\sqrt{d_k}$. |
| 2 | Applying the mask after softmax | Forbidden positions can still receive probability mass. | Add the mask to logits before softmax. |
| 3 | Confusing values with weights | Attention weights choose how values are mixed; values carry payloads. | Name Q, K, V roles separately. |
| 4 | Treating attention maps as full explanations | Weights are one mechanism among residual paths and MLPs. | Use ablations and causal interventions too. |
| 5 | Ignoring padding masks | Padding tokens can leak into real-token states. | Mask pads in every attention layer. |
| 6 | Breaking causal masking | Future-token leakage invalidates language-model training. | Unit-test upper-triangular masked weights. |
| 7 | Assuming all heads are useful | Some heads can be redundant or inactive. | Inspect entropy, ablations, and output norms. |
| 8 | Underestimating KV-cache memory | Long contexts store K/V for every layer and head. | Compute cache bytes before serving claims. |
| 9 | Calling FlashAttention approximate | FlashAttention is exact attention with a different IO-aware algorithm. | Separate kernel implementation from mathematical approximation. |
| 10 | Ignoring numerical stability | Large logits can overflow exponentials. | Subtract row maxima before softmax. |

## 10. Exercises

1. (*) Compute scaled dot-product attention for a two-token sequence.
   - (a) State tensor shapes.
   - (b) Compute the numeric result.
   - (c) Explain the LLM consequence.

2. (*) Apply a causal mask and verify future weights are zero.
   - (a) State tensor shapes.
   - (b) Compute the numeric result.
   - (c) Explain the LLM consequence.

3. (*) Compute attention entropy for sharp and diffuse rows.
   - (a) State tensor shapes.
   - (b) Compute the numeric result.
   - (c) Explain the LLM consequence.

4. (**) Track multi-head split and concat shapes.
   - (a) State tensor shapes.
   - (b) Compute the numeric result.
   - (c) Explain the LLM consequence.

5. (**) Build a padding mask and explain its effect.
   - (a) State tensor shapes.
   - (b) Compute the numeric result.
   - (c) Explain the LLM consequence.

6. (**) Compute KV-cache bytes for a small model.
   - (a) State tensor shapes.
   - (b) Compute the numeric result.
   - (c) Explain the LLM consequence.

7. (**) Add an ALiBi-style distance bias to scores.
   - (a) State tensor shapes.
   - (b) Compute the numeric result.
   - (c) Explain the LLM consequence.

8. (***) Compare prefill and decode attention cost.
   - (a) State tensor shapes.
   - (b) Compute the numeric result.
   - (c) Explain the LLM consequence.

9. (***) Explain why FlashAttention is exact but more memory efficient.
   - (a) State tensor shapes.
   - (b) Compute the numeric result.
   - (c) Explain the LLM consequence.

10. (***) Design a mask test for prompt-boundary safety.
   - (a) State tensor shapes.
   - (b) Compute the numeric result.
   - (c) Explain the LLM consequence.

## 11. Why This Matters for AI

| Concept | AI impact |
| --- | --- |
| Q/K/V projections | Let each token search, address, and retrieve information from the context. |
| Causal masks | Make next-token training valid by blocking future-token leakage. |
| Multi-head attention | Allows several learned relation patterns to operate in parallel. |
| KV cache | Makes autoregressive serving practical for long prompts. |
| Attention entropy | Diagnoses whether tokens attend narrowly, diffusely, or pathologically. |
| FlashAttention | Improves exact attention performance by reducing memory movement. |
| RAG attention | Lets generated tokens condition on retrieved evidence chunks. |
| Mask safety | Protects padding, chat boundaries, and tool-control delimiters from leakage bugs. |

## 12. Conceptual Bridge

The backward bridge is embedding space. Attention does not work on raw text; it works on vectors. Query, key, and value projections are learned linear maps applied to hidden states that began as token embeddings.

The forward bridge is positional encodings and transformer architecture. Position mechanisms modify what attention can know about order, while residual blocks and MLPs determine how attention outputs become deeper contextual representations.

```text
+-------------+      +-------------+      +-------------+      +-------------+
| embeddings  | ---> | Q K V       | ---> | softmax AV  | ---> | residual    |
| B x T x d   |      | projections |      | context mix |      | stream      |
+-------------+      +-------------+      +-------------+      +-------------+
```

The durable habit is to test masks. A model can have the right architecture and still learn invalid shortcuts if future tokens, padding, or protected boundaries are visible by mistake.

## References

- Vaswani et al.. Attention Is All You Need. https://arxiv.org/abs/1706.03762
- Bahdanau, Cho, Bengio. Neural Machine Translation by Jointly Learning to Align and Translate. https://arxiv.org/abs/1409.0473
- Dao et al.. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. https://arxiv.org/abs/2205.14135
- Dao. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. https://arxiv.org/abs/2307.08691
- Vaswani et al.. Transformer architecture reference. https://papers.nips.cc/paper/7181-attention-is-all-you-need
