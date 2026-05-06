[Previous: Attention Mechanism Math](../03-Attention-Mechanism-Math/notes.md) | [Back to Math for LLMs](../README.md) | [Next: Language Model Probability](../05-Language-Model-Probability/notes.md)

---

# Positional Encodings

> _"Attention can compare tokens; positional encoding tells it where those tokens live in the sequence."_

## Overview

A transformer without positional information is largely order-agnostic: it can compare token content, but it has no built-in reason to know which token came first. Positional encodings inject order through added vectors, relative biases, rotations, or score penalties.

Modern LLMs use several families of position methods. Sinusoidal and learned absolute encodings add position information to hidden states. Relative position methods modify attention scores. RoPE rotates queries and keys. ALiBi adds linear distance biases. These choices affect extrapolation, long-context behavior, KV-cache decoding, and attention diagnostics.

This section uses LaTeX Markdown with `$...$` and `$$...$$`. The notebooks implement each scheme in small NumPy examples and validate the invariants learners should remember.

## Prerequisites

- [Embedding Space Math](../02-Embedding-Space-Math/notes.md)
- [Attention Mechanism Math](../03-Attention-Mechanism-Math/notes.md)
- [Matrix Operations](../../02-Linear-Algebra-Basics/02-Matrix-Operations/notes.md)
- [Trigonometry and periodic functions](../../04-Calculus-Fundamentals/04-Series-and-Sequences/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Executable sinusoidal, learned, relative-bias, RoPE, ALiBi, scaling, and decode-position demonstrations. |
| [exercises.ipynb](exercises.ipynb) | Ten checked exercises for positional encoding mechanics and long-context diagnostics. |

## Learning Objectives

After completing this section, you will be able to:

- Explain why self-attention needs explicit position information.
- Distinguish absolute, relative, additive, rotary, and bias-based position schemes.
- Compute sinusoidal positional encodings for small positions and dimensions.
- Explain the advantages and limits of learned absolute position rows.
- Build relative attention bias matrices from offsets.
- Apply RoPE rotations and verify norm preservation.
- Explain the relative dot-product property of RoPE.
- Build ALiBi distance-bias matrices with head-specific slopes.
- Diagnose long-context and KV-cache position-id bugs.
- Choose a position scheme based on extrapolation, cost, and architecture constraints.

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 Why attention needs position](#11-why-attention-needs-position)
  - [1.2 Absolute versus relative position](#12-absolute-versus-relative-position)
  - [1.3 Additive versus score-based position](#13-additive-versus-scorebased-position)
  - [1.4 Length extrapolation](#14-length-extrapolation)
  - [1.5 Position in decoder-only LLMs](#15-position-in-decoderonly-llms)
- [2. Formal Setup](#2-formal-setup)
  - [2.1 Position indices](#21-position-indices)
  - [2.2 Token plus position representation](#22-token-plus-position-representation)
  - [2.3 Attention score modification](#23-attention-score-modification)
  - [2.4 Relative offset notation](#24-relative-offset-notation)
  - [2.5 Position interpolation and scaling](#25-position-interpolation-and-scaling)
- [3. Sinusoidal Encodings](#3-sinusoidal-encodings)
  - [3.1 Frequency ladder](#31-frequency-ladder)
  - [3.2 Sine cosine pairs](#32-sine-cosine-pairs)
  - [3.3 Linear relative-offset intuition](#33-linear-relativeoffset-intuition)
  - [3.4 Visualization and aliasing](#34-visualization-and-aliasing)
  - [3.5 Limitations](#35-limitations)
- [4. Learned Absolute Positions](#4-learned-absolute-positions)
  - [4.1 Learned position table](#41-learned-position-table)
  - [4.2 Training length limit](#42-training-length-limit)
  - [4.3 Interpolation resizing](#43-interpolation-resizing)
  - [4.4 BERT GPT-style usage](#44-bert-gptstyle-usage)
  - [4.5 Failure modes](#45-failure-modes)
- [5. Relative Position Representations](#5-relative-position-representations)
  - [5.1 Relative bias matrices](#51-relative-bias-matrices)
  - [5.2 Shaw-style relative keys](#52-shawstyle-relative-keys)
  - [5.3 Transformer-XL intuition](#53-transformerxl-intuition)
  - [5.4 Bucketed distances](#54-bucketed-distances)
  - [5.5 When relative position helps](#55-when-relative-position-helps)
- [6. Rotary Positional Embeddings](#6-rotary-positional-embeddings)
  - [6.1 RoPE rotations](#61-rope-rotations)
  - [6.2 Relative dot-product property](#62-relative-dotproduct-property)
  - [6.3 Frequency base and dimensions](#63-frequency-base-and-dimensions)
  - [6.4 Long-context RoPE scaling](#64-longcontext-rope-scaling)
  - [6.5 Implementation checks](#65-implementation-checks)
- [7. ALiBi and Bias Methods](#7-alibi-and-bias-methods)
  - [7.1 Linear attention biases](#71-linear-attention-biases)
  - [7.2 Head-specific slopes](#72-headspecific-slopes)
  - [7.3 Extrapolation behavior](#73-extrapolation-behavior)
  - [7.4 Bias versus embedding](#74-bias-versus-embedding)
  - [7.5 Mask interaction](#75-mask-interaction)
- [8. Diagnostics and LLM Practice](#8-diagnostics-and-llm-practice)
  - [8.1 Position-sensitivity tests](#81-positionsensitivity-tests)
  - [8.2 Long-context degradation](#82-longcontext-degradation)
  - [8.3 Attention pattern inspection](#83-attention-pattern-inspection)
  - [8.4 Serving and KV cache](#84-serving-and-kv-cache)
  - [8.5 Choosing a scheme](#85-choosing-a-scheme)
- [9. Common Mistakes](#9-common-mistakes)
- [10. Exercises](#10-exercises)
- [11. Why This Matters for AI](#11-why-this-matters-for-ai)
- [12. Conceptual Bridge](#12-conceptual-bridge)
- [References](#references)

---

## 1. Intuition

Intuition explains how transformer sequence order is represented in hidden states or attention scores.

### 1.1 Why attention needs position

**Purpose.** Why attention needs position focuses on self-attention is permutation-equivariant without order signals. This is the part of transformer math that tells attention where each token is and how far apart tokens are.

$$\operatorname{PE}_{p,2k}=\sin\left(p/10000^{2k/d}\right),\qquad \operatorname{PE}_{p,2k+1}=\cos\left(p/10000^{2k/d}\right).$$

**Operational definition.**

Position encoding injects order into transformer computation so identical tokens at different positions can have different roles.

**Worked reading.**

Without a position signal, self-attention can mix content but cannot know which token came first.

| Scheme | Where position enters | Typical strength | Typical risk |
| --- | --- | --- | --- |
| sinusoidal | added to hidden states | fixed and simple | limited flexibility |
| learned absolute | added learned row | strong in-range fit | weak extrapolation |
| relative bias | attention score | offset-aware | implementation complexity |
| RoPE | rotates Q/K | relative dot products | scaling choices matter |
| ALiBi | attention score bias | simple extrapolation | distance prior may be too rigid |

Examples:

1. sinusoidal features.
2. position rows.
3. relative offsets.

Non-examples:

1. bag-of-words attention.
2. token ids used as positions.

**Derivation habit.**

1. State whether the scheme is absolute or relative.
2. State whether the signal is added to hidden states, applied to Q/K, or added to scores.
3. Check shape compatibility with heads, sequence length, and cached decoding.
4. Test length behavior beyond the training context if the model will be served there.
5. Keep masks separate from position signals.

**Implementation lens.**

Position bugs are subtle because tensor shapes can still look correct. A decode loop can run while assigning the wrong position id to every generated token. A long-context model can accept a large input while quality collapses in the middle of the context.

The practical defense is small tests: compare attention with shifted inputs, verify RoPE norm preservation, check ALiBi bias values by distance, and run retrieval probes at several positions in the context window.

### 1.2 Absolute versus relative position

**Purpose.** Absolute versus relative position focuses on index identity versus pairwise offset. This is the part of transformer math that tells attention where each token is and how far apart tokens are.

$$\mathbf{h}_p=\mathbf{x}_p+\mathbf{p}_p.$$

**Operational definition.**

Relative position methods make attention depend on pairwise distance instead of only absolute index identity.

**Worked reading.**

The same offset can share parameters across many positions, which is useful when patterns repeat across the sequence.

| Scheme | Where position enters | Typical strength | Typical risk |
| --- | --- | --- | --- |
| sinusoidal | added to hidden states | fixed and simple | limited flexibility |
| learned absolute | added learned row | strong in-range fit | weak extrapolation |
| relative bias | attention score | offset-aware | implementation complexity |
| RoPE | rotates Q/K | relative dot products | scaling choices matter |
| ALiBi | attention score bias | simple extrapolation | distance prior may be too rigid |

Examples:

1. relative bias.
2. relative keys.
3. bucketed distance bins.

Non-examples:

1. one independent vector per absolute index.
2. no order signal at all.

**Derivation habit.**

1. State whether the scheme is absolute or relative.
2. State whether the signal is added to hidden states, applied to Q/K, or added to scores.
3. Check shape compatibility with heads, sequence length, and cached decoding.
4. Test length behavior beyond the training context if the model will be served there.
5. Keep masks separate from position signals.

**Implementation lens.**

Position bugs are subtle because tensor shapes can still look correct. A decode loop can run while assigning the wrong position id to every generated token. A long-context model can accept a large input while quality collapses in the middle of the context.

The practical defense is small tests: compare attention with shifted inputs, verify RoPE norm preservation, check ALiBi bias values by distance, and run retrieval probes at several positions in the context window.

### 1.3 Additive versus score-based position

**Purpose.** Additive versus score-based position focuses on where the signal enters the computation. This is the part of transformer math that tells attention where each token is and how far apart tokens are.

$$S_{ij}=\frac{\mathbf{q}_i^\top\mathbf{k}_j}{\sqrt{d_k}}+b_{i-j}.$$

**Operational definition.**

Position encoding injects order into transformer computation so identical tokens at different positions can have different roles.

**Worked reading.**

Without a position signal, self-attention can mix content but cannot know which token came first.

| Scheme | Where position enters | Typical strength | Typical risk |
| --- | --- | --- | --- |
| sinusoidal | added to hidden states | fixed and simple | limited flexibility |
| learned absolute | added learned row | strong in-range fit | weak extrapolation |
| relative bias | attention score | offset-aware | implementation complexity |
| RoPE | rotates Q/K | relative dot products | scaling choices matter |
| ALiBi | attention score bias | simple extrapolation | distance prior may be too rigid |

Examples:

1. sinusoidal features.
2. position rows.
3. relative offsets.

Non-examples:

1. bag-of-words attention.
2. token ids used as positions.

**Derivation habit.**

1. State whether the scheme is absolute or relative.
2. State whether the signal is added to hidden states, applied to Q/K, or added to scores.
3. Check shape compatibility with heads, sequence length, and cached decoding.
4. Test length behavior beyond the training context if the model will be served there.
5. Keep masks separate from position signals.

**Implementation lens.**

Position bugs are subtle because tensor shapes can still look correct. A decode loop can run while assigning the wrong position id to every generated token. A long-context model can accept a large input while quality collapses in the middle of the context.

The practical defense is small tests: compare attention with shifted inputs, verify RoPE norm preservation, check ALiBi bias values by distance, and run retrieval probes at several positions in the context window.

### 1.4 Length extrapolation

**Purpose.** Length extrapolation focuses on why training length and inference length can diverge. This is the part of transformer math that tells attention where each token is and how far apart tokens are.

$$R_p^\top R_j=R_{j-p}.$$

**Operational definition.**

Position encoding injects order into transformer computation so identical tokens at different positions can have different roles.

**Worked reading.**

Without a position signal, self-attention can mix content but cannot know which token came first.

| Scheme | Where position enters | Typical strength | Typical risk |
| --- | --- | --- | --- |
| sinusoidal | added to hidden states | fixed and simple | limited flexibility |
| learned absolute | added learned row | strong in-range fit | weak extrapolation |
| relative bias | attention score | offset-aware | implementation complexity |
| RoPE | rotates Q/K | relative dot products | scaling choices matter |
| ALiBi | attention score bias | simple extrapolation | distance prior may be too rigid |

Examples:

1. sinusoidal features.
2. position rows.
3. relative offsets.

Non-examples:

1. bag-of-words attention.
2. token ids used as positions.

**Derivation habit.**

1. State whether the scheme is absolute or relative.
2. State whether the signal is added to hidden states, applied to Q/K, or added to scores.
3. Check shape compatibility with heads, sequence length, and cached decoding.
4. Test length behavior beyond the training context if the model will be served there.
5. Keep masks separate from position signals.

**Implementation lens.**

Position bugs are subtle because tensor shapes can still look correct. A decode loop can run while assigning the wrong position id to every generated token. A long-context model can accept a large input while quality collapses in the middle of the context.

The practical defense is small tests: compare attention with shifted inputs, verify RoPE norm preservation, check ALiBi bias values by distance, and run retrieval probes at several positions in the context window.

### 1.5 Position in decoder-only LLMs

**Purpose.** Position in decoder-only LLMs focuses on causal order and generated prefixes. This is the part of transformer math that tells attention where each token is and how far apart tokens are.

$$\operatorname{RoPE}(\mathbf{q}_p)=R_p\mathbf{q}_p.$$

**Operational definition.**

Position encoding injects order into transformer computation so identical tokens at different positions can have different roles.

**Worked reading.**

Without a position signal, self-attention can mix content but cannot know which token came first.

| Scheme | Where position enters | Typical strength | Typical risk |
| --- | --- | --- | --- |
| sinusoidal | added to hidden states | fixed and simple | limited flexibility |
| learned absolute | added learned row | strong in-range fit | weak extrapolation |
| relative bias | attention score | offset-aware | implementation complexity |
| RoPE | rotates Q/K | relative dot products | scaling choices matter |
| ALiBi | attention score bias | simple extrapolation | distance prior may be too rigid |

Examples:

1. sinusoidal features.
2. position rows.
3. relative offsets.

Non-examples:

1. bag-of-words attention.
2. token ids used as positions.

**Derivation habit.**

1. State whether the scheme is absolute or relative.
2. State whether the signal is added to hidden states, applied to Q/K, or added to scores.
3. Check shape compatibility with heads, sequence length, and cached decoding.
4. Test length behavior beyond the training context if the model will be served there.
5. Keep masks separate from position signals.

**Implementation lens.**

Position bugs are subtle because tensor shapes can still look correct. A decode loop can run while assigning the wrong position id to every generated token. A long-context model can accept a large input while quality collapses in the middle of the context.

The practical defense is small tests: compare attention with shifted inputs, verify RoPE norm preservation, check ALiBi bias values by distance, and run retrieval probes at several positions in the context window.

## 2. Formal Setup

Formal Setup explains how transformer sequence order is represented in hidden states or attention scores.

### 2.1 Position indices

**Purpose.** Position indices focuses on $p\in\{0,\ldots,T-1\}$. This is the part of transformer math that tells attention where each token is and how far apart tokens are.

$$\mathbf{h}_p=\mathbf{x}_p+\mathbf{p}_p.$$

**Operational definition.**

Position encoding injects order into transformer computation so identical tokens at different positions can have different roles.

**Worked reading.**

Without a position signal, self-attention can mix content but cannot know which token came first.

| Scheme | Where position enters | Typical strength | Typical risk |
| --- | --- | --- | --- |
| sinusoidal | added to hidden states | fixed and simple | limited flexibility |
| learned absolute | added learned row | strong in-range fit | weak extrapolation |
| relative bias | attention score | offset-aware | implementation complexity |
| RoPE | rotates Q/K | relative dot products | scaling choices matter |
| ALiBi | attention score bias | simple extrapolation | distance prior may be too rigid |

Examples:

1. sinusoidal features.
2. position rows.
3. relative offsets.

Non-examples:

1. bag-of-words attention.
2. token ids used as positions.

**Derivation habit.**

1. State whether the scheme is absolute or relative.
2. State whether the signal is added to hidden states, applied to Q/K, or added to scores.
3. Check shape compatibility with heads, sequence length, and cached decoding.
4. Test length behavior beyond the training context if the model will be served there.
5. Keep masks separate from position signals.

**Implementation lens.**

Position bugs are subtle because tensor shapes can still look correct. A decode loop can run while assigning the wrong position id to every generated token. A long-context model can accept a large input while quality collapses in the middle of the context.

The practical defense is small tests: compare attention with shifted inputs, verify RoPE norm preservation, check ALiBi bias values by distance, and run retrieval probes at several positions in the context window.

### 2.2 Token plus position representation

**Purpose.** Token plus position representation focuses on $\mathbf{h}_p=\mathbf{x}_p+\mathbf{p}_p$. This is the part of transformer math that tells attention where each token is and how far apart tokens are.

$$S_{ij}=\frac{\mathbf{q}_i^\top\mathbf{k}_j}{\sqrt{d_k}}+b_{i-j}.$$

**Operational definition.**

Position encoding injects order into transformer computation so identical tokens at different positions can have different roles.

**Worked reading.**

Without a position signal, self-attention can mix content but cannot know which token came first.

| Scheme | Where position enters | Typical strength | Typical risk |
| --- | --- | --- | --- |
| sinusoidal | added to hidden states | fixed and simple | limited flexibility |
| learned absolute | added learned row | strong in-range fit | weak extrapolation |
| relative bias | attention score | offset-aware | implementation complexity |
| RoPE | rotates Q/K | relative dot products | scaling choices matter |
| ALiBi | attention score bias | simple extrapolation | distance prior may be too rigid |

Examples:

1. sinusoidal features.
2. position rows.
3. relative offsets.

Non-examples:

1. bag-of-words attention.
2. token ids used as positions.

**Derivation habit.**

1. State whether the scheme is absolute or relative.
2. State whether the signal is added to hidden states, applied to Q/K, or added to scores.
3. Check shape compatibility with heads, sequence length, and cached decoding.
4. Test length behavior beyond the training context if the model will be served there.
5. Keep masks separate from position signals.

**Implementation lens.**

Position bugs are subtle because tensor shapes can still look correct. A decode loop can run while assigning the wrong position id to every generated token. A long-context model can accept a large input while quality collapses in the middle of the context.

The practical defense is small tests: compare attention with shifted inputs, verify RoPE norm preservation, check ALiBi bias values by distance, and run retrieval probes at several positions in the context window.

### 2.3 Attention score modification

**Purpose.** Attention score modification focuses on biases and rotations before softmax. This is the part of transformer math that tells attention where each token is and how far apart tokens are.

$$R_p^\top R_j=R_{j-p}.$$

**Operational definition.**

Position encoding injects order into transformer computation so identical tokens at different positions can have different roles.

**Worked reading.**

Without a position signal, self-attention can mix content but cannot know which token came first.

| Scheme | Where position enters | Typical strength | Typical risk |
| --- | --- | --- | --- |
| sinusoidal | added to hidden states | fixed and simple | limited flexibility |
| learned absolute | added learned row | strong in-range fit | weak extrapolation |
| relative bias | attention score | offset-aware | implementation complexity |
| RoPE | rotates Q/K | relative dot products | scaling choices matter |
| ALiBi | attention score bias | simple extrapolation | distance prior may be too rigid |

Examples:

1. sinusoidal features.
2. position rows.
3. relative offsets.

Non-examples:

1. bag-of-words attention.
2. token ids used as positions.

**Derivation habit.**

1. State whether the scheme is absolute or relative.
2. State whether the signal is added to hidden states, applied to Q/K, or added to scores.
3. Check shape compatibility with heads, sequence length, and cached decoding.
4. Test length behavior beyond the training context if the model will be served there.
5. Keep masks separate from position signals.

**Implementation lens.**

Position bugs are subtle because tensor shapes can still look correct. A decode loop can run while assigning the wrong position id to every generated token. A long-context model can accept a large input while quality collapses in the middle of the context.

The practical defense is small tests: compare attention with shifted inputs, verify RoPE norm preservation, check ALiBi bias values by distance, and run retrieval probes at several positions in the context window.

### 2.4 Relative offset notation

**Purpose.** Relative offset notation focuses on $r=i-j$. This is the part of transformer math that tells attention where each token is and how far apart tokens are.

$$\operatorname{RoPE}(\mathbf{q}_p)=R_p\mathbf{q}_p.$$

**Operational definition.**

Relative position methods make attention depend on pairwise distance instead of only absolute index identity.

**Worked reading.**

The same offset can share parameters across many positions, which is useful when patterns repeat across the sequence.

| Scheme | Where position enters | Typical strength | Typical risk |
| --- | --- | --- | --- |
| sinusoidal | added to hidden states | fixed and simple | limited flexibility |
| learned absolute | added learned row | strong in-range fit | weak extrapolation |
| relative bias | attention score | offset-aware | implementation complexity |
| RoPE | rotates Q/K | relative dot products | scaling choices matter |
| ALiBi | attention score bias | simple extrapolation | distance prior may be too rigid |

Examples:

1. relative bias.
2. relative keys.
3. bucketed distance bins.

Non-examples:

1. one independent vector per absolute index.
2. no order signal at all.

**Derivation habit.**

1. State whether the scheme is absolute or relative.
2. State whether the signal is added to hidden states, applied to Q/K, or added to scores.
3. Check shape compatibility with heads, sequence length, and cached decoding.
4. Test length behavior beyond the training context if the model will be served there.
5. Keep masks separate from position signals.

**Implementation lens.**

Position bugs are subtle because tensor shapes can still look correct. A decode loop can run while assigning the wrong position id to every generated token. A long-context model can accept a large input while quality collapses in the middle of the context.

The practical defense is small tests: compare attention with shifted inputs, verify RoPE norm preservation, check ALiBi bias values by distance, and run retrieval probes at several positions in the context window.

### 2.5 Position interpolation and scaling

**Purpose.** Position interpolation and scaling focuses on changing effective positions for long context. This is the part of transformer math that tells attention where each token is and how far apart tokens are.

$$\operatorname{ALiBi}_{ij}=m_h(i-j)\quad\text{for }j\le i.$$

**Operational definition.**

Position encoding injects order into transformer computation so identical tokens at different positions can have different roles.

**Worked reading.**

Without a position signal, self-attention can mix content but cannot know which token came first.

| Scheme | Where position enters | Typical strength | Typical risk |
| --- | --- | --- | --- |
| sinusoidal | added to hidden states | fixed and simple | limited flexibility |
| learned absolute | added learned row | strong in-range fit | weak extrapolation |
| relative bias | attention score | offset-aware | implementation complexity |
| RoPE | rotates Q/K | relative dot products | scaling choices matter |
| ALiBi | attention score bias | simple extrapolation | distance prior may be too rigid |

Examples:

1. sinusoidal features.
2. position rows.
3. relative offsets.

Non-examples:

1. bag-of-words attention.
2. token ids used as positions.

**Derivation habit.**

1. State whether the scheme is absolute or relative.
2. State whether the signal is added to hidden states, applied to Q/K, or added to scores.
3. Check shape compatibility with heads, sequence length, and cached decoding.
4. Test length behavior beyond the training context if the model will be served there.
5. Keep masks separate from position signals.

**Implementation lens.**

Position bugs are subtle because tensor shapes can still look correct. A decode loop can run while assigning the wrong position id to every generated token. A long-context model can accept a large input while quality collapses in the middle of the context.

The practical defense is small tests: compare attention with shifted inputs, verify RoPE norm preservation, check ALiBi bias values by distance, and run retrieval probes at several positions in the context window.

## 3. Sinusoidal Encodings

Sinusoidal Encodings explains how transformer sequence order is represented in hidden states or attention scores.

### 3.1 Frequency ladder

**Purpose.** Frequency ladder focuses on geometric wavelengths across dimensions. This is the part of transformer math that tells attention where each token is and how far apart tokens are.

$$S_{ij}=\frac{\mathbf{q}_i^\top\mathbf{k}_j}{\sqrt{d_k}}+b_{i-j}.$$

**Operational definition.**

Position encoding injects order into transformer computation so identical tokens at different positions can have different roles.

**Worked reading.**

Without a position signal, self-attention can mix content but cannot know which token came first.

| Scheme | Where position enters | Typical strength | Typical risk |
| --- | --- | --- | --- |
| sinusoidal | added to hidden states | fixed and simple | limited flexibility |
| learned absolute | added learned row | strong in-range fit | weak extrapolation |
| relative bias | attention score | offset-aware | implementation complexity |
| RoPE | rotates Q/K | relative dot products | scaling choices matter |
| ALiBi | attention score bias | simple extrapolation | distance prior may be too rigid |

Examples:

1. sinusoidal features.
2. position rows.
3. relative offsets.

Non-examples:

1. bag-of-words attention.
2. token ids used as positions.

**Derivation habit.**

1. State whether the scheme is absolute or relative.
2. State whether the signal is added to hidden states, applied to Q/K, or added to scores.
3. Check shape compatibility with heads, sequence length, and cached decoding.
4. Test length behavior beyond the training context if the model will be served there.
5. Keep masks separate from position signals.

**Implementation lens.**

Position bugs are subtle because tensor shapes can still look correct. A decode loop can run while assigning the wrong position id to every generated token. A long-context model can accept a large input while quality collapses in the middle of the context.

The practical defense is small tests: compare attention with shifted inputs, verify RoPE norm preservation, check ALiBi bias values by distance, and run retrieval probes at several positions in the context window.

### 3.2 Sine cosine pairs

**Purpose.** Sine cosine pairs focuses on phase representation by coordinate pairs. This is the part of transformer math that tells attention where each token is and how far apart tokens are.

$$R_p^\top R_j=R_{j-p}.$$

**Operational definition.**

Position encoding injects order into transformer computation so identical tokens at different positions can have different roles.

**Worked reading.**

Without a position signal, self-attention can mix content but cannot know which token came first.

| Scheme | Where position enters | Typical strength | Typical risk |
| --- | --- | --- | --- |
| sinusoidal | added to hidden states | fixed and simple | limited flexibility |
| learned absolute | added learned row | strong in-range fit | weak extrapolation |
| relative bias | attention score | offset-aware | implementation complexity |
| RoPE | rotates Q/K | relative dot products | scaling choices matter |
| ALiBi | attention score bias | simple extrapolation | distance prior may be too rigid |

Examples:

1. sinusoidal features.
2. position rows.
3. relative offsets.

Non-examples:

1. bag-of-words attention.
2. token ids used as positions.

**Derivation habit.**

1. State whether the scheme is absolute or relative.
2. State whether the signal is added to hidden states, applied to Q/K, or added to scores.
3. Check shape compatibility with heads, sequence length, and cached decoding.
4. Test length behavior beyond the training context if the model will be served there.
5. Keep masks separate from position signals.

**Implementation lens.**

Position bugs are subtle because tensor shapes can still look correct. A decode loop can run while assigning the wrong position id to every generated token. A long-context model can accept a large input while quality collapses in the middle of the context.

The practical defense is small tests: compare attention with shifted inputs, verify RoPE norm preservation, check ALiBi bias values by distance, and run retrieval probes at several positions in the context window.

### 3.3 Linear relative-offset intuition

**Purpose.** Linear relative-offset intuition focuses on why fixed sinusoids can expose offsets. This is the part of transformer math that tells attention where each token is and how far apart tokens are.

$$\operatorname{RoPE}(\mathbf{q}_p)=R_p\mathbf{q}_p.$$

**Operational definition.**

Relative position methods make attention depend on pairwise distance instead of only absolute index identity.

**Worked reading.**

The same offset can share parameters across many positions, which is useful when patterns repeat across the sequence.

| Scheme | Where position enters | Typical strength | Typical risk |
| --- | --- | --- | --- |
| sinusoidal | added to hidden states | fixed and simple | limited flexibility |
| learned absolute | added learned row | strong in-range fit | weak extrapolation |
| relative bias | attention score | offset-aware | implementation complexity |
| RoPE | rotates Q/K | relative dot products | scaling choices matter |
| ALiBi | attention score bias | simple extrapolation | distance prior may be too rigid |

Examples:

1. relative bias.
2. relative keys.
3. bucketed distance bins.

Non-examples:

1. one independent vector per absolute index.
2. no order signal at all.

**Derivation habit.**

1. State whether the scheme is absolute or relative.
2. State whether the signal is added to hidden states, applied to Q/K, or added to scores.
3. Check shape compatibility with heads, sequence length, and cached decoding.
4. Test length behavior beyond the training context if the model will be served there.
5. Keep masks separate from position signals.

**Implementation lens.**

Position bugs are subtle because tensor shapes can still look correct. A decode loop can run while assigning the wrong position id to every generated token. A long-context model can accept a large input while quality collapses in the middle of the context.

The practical defense is small tests: compare attention with shifted inputs, verify RoPE norm preservation, check ALiBi bias values by distance, and run retrieval probes at several positions in the context window.

### 3.4 Visualization and aliasing

**Purpose.** Visualization and aliasing focuses on what high and low frequencies show. This is the part of transformer math that tells attention where each token is and how far apart tokens are.

$$\operatorname{ALiBi}_{ij}=m_h(i-j)\quad\text{for }j\le i.$$

**Operational definition.**

Position encoding injects order into transformer computation so identical tokens at different positions can have different roles.

**Worked reading.**

Without a position signal, self-attention can mix content but cannot know which token came first.

| Scheme | Where position enters | Typical strength | Typical risk |
| --- | --- | --- | --- |
| sinusoidal | added to hidden states | fixed and simple | limited flexibility |
| learned absolute | added learned row | strong in-range fit | weak extrapolation |
| relative bias | attention score | offset-aware | implementation complexity |
| RoPE | rotates Q/K | relative dot products | scaling choices matter |
| ALiBi | attention score bias | simple extrapolation | distance prior may be too rigid |

Examples:

1. sinusoidal features.
2. position rows.
3. relative offsets.

Non-examples:

1. bag-of-words attention.
2. token ids used as positions.

**Derivation habit.**

1. State whether the scheme is absolute or relative.
2. State whether the signal is added to hidden states, applied to Q/K, or added to scores.
3. Check shape compatibility with heads, sequence length, and cached decoding.
4. Test length behavior beyond the training context if the model will be served there.
5. Keep masks separate from position signals.

**Implementation lens.**

Position bugs are subtle because tensor shapes can still look correct. A decode loop can run while assigning the wrong position id to every generated token. A long-context model can accept a large input while quality collapses in the middle of the context.

The practical defense is small tests: compare attention with shifted inputs, verify RoPE norm preservation, check ALiBi bias values by distance, and run retrieval probes at several positions in the context window.

### 3.5 Limitations

**Purpose.** Limitations focuses on absolute addition and finite precision. This is the part of transformer math that tells attention where each token is and how far apart tokens are.

$$\lambda_k=10000^{2k/d}.$$

**Operational definition.**

Position encoding injects order into transformer computation so identical tokens at different positions can have different roles.

**Worked reading.**

Without a position signal, self-attention can mix content but cannot know which token came first.

| Scheme | Where position enters | Typical strength | Typical risk |
| --- | --- | --- | --- |
| sinusoidal | added to hidden states | fixed and simple | limited flexibility |
| learned absolute | added learned row | strong in-range fit | weak extrapolation |
| relative bias | attention score | offset-aware | implementation complexity |
| RoPE | rotates Q/K | relative dot products | scaling choices matter |
| ALiBi | attention score bias | simple extrapolation | distance prior may be too rigid |

Examples:

1. sinusoidal features.
2. position rows.
3. relative offsets.

Non-examples:

1. bag-of-words attention.
2. token ids used as positions.

**Derivation habit.**

1. State whether the scheme is absolute or relative.
2. State whether the signal is added to hidden states, applied to Q/K, or added to scores.
3. Check shape compatibility with heads, sequence length, and cached decoding.
4. Test length behavior beyond the training context if the model will be served there.
5. Keep masks separate from position signals.

**Implementation lens.**

Position bugs are subtle because tensor shapes can still look correct. A decode loop can run while assigning the wrong position id to every generated token. A long-context model can accept a large input while quality collapses in the middle of the context.

The practical defense is small tests: compare attention with shifted inputs, verify RoPE norm preservation, check ALiBi bias values by distance, and run retrieval probes at several positions in the context window.

## 4. Learned Absolute Positions

Learned Absolute Positions explains how transformer sequence order is represented in hidden states or attention scores.

### 4.1 Learned position table

**Purpose.** Learned position table focuses on $P\in\mathbb{R}^{T_{\max}\times d}$. This is the part of transformer math that tells attention where each token is and how far apart tokens are.

$$R_p^\top R_j=R_{j-p}.$$

**Operational definition.**

Learned absolute position embeddings assign a trainable vector to each position index up to a maximum length.

**Worked reading.**

They are simple and effective in-range, but positions beyond the trained table require resizing or another strategy.

| Scheme | Where position enters | Typical strength | Typical risk |
| --- | --- | --- | --- |
| sinusoidal | added to hidden states | fixed and simple | limited flexibility |
| learned absolute | added learned row | strong in-range fit | weak extrapolation |
| relative bias | attention score | offset-aware | implementation complexity |
| RoPE | rotates Q/K | relative dot products | scaling choices matter |
| ALiBi | attention score bias | simple extrapolation | distance prior may be too rigid |

Examples:

1. BERT-style position rows.
2. GPT-style learned absolute positions.
3. interpolation for longer windows.

Non-examples:

1. relative distance bias.
2. rotation of Q/K coordinates.

**Derivation habit.**

1. State whether the scheme is absolute or relative.
2. State whether the signal is added to hidden states, applied to Q/K, or added to scores.
3. Check shape compatibility with heads, sequence length, and cached decoding.
4. Test length behavior beyond the training context if the model will be served there.
5. Keep masks separate from position signals.

**Implementation lens.**

Position bugs are subtle because tensor shapes can still look correct. A decode loop can run while assigning the wrong position id to every generated token. A long-context model can accept a large input while quality collapses in the middle of the context.

The practical defense is small tests: compare attention with shifted inputs, verify RoPE norm preservation, check ALiBi bias values by distance, and run retrieval probes at several positions in the context window.

### 4.2 Training length limit

**Purpose.** Training length limit focuses on why rows beyond training are unavailable. This is the part of transformer math that tells attention where each token is and how far apart tokens are.

$$\operatorname{RoPE}(\mathbf{q}_p)=R_p\mathbf{q}_p.$$

**Operational definition.**

Learned absolute position embeddings assign a trainable vector to each position index up to a maximum length.

**Worked reading.**

They are simple and effective in-range, but positions beyond the trained table require resizing or another strategy.

| Scheme | Where position enters | Typical strength | Typical risk |
| --- | --- | --- | --- |
| sinusoidal | added to hidden states | fixed and simple | limited flexibility |
| learned absolute | added learned row | strong in-range fit | weak extrapolation |
| relative bias | attention score | offset-aware | implementation complexity |
| RoPE | rotates Q/K | relative dot products | scaling choices matter |
| ALiBi | attention score bias | simple extrapolation | distance prior may be too rigid |

Examples:

1. BERT-style position rows.
2. GPT-style learned absolute positions.
3. interpolation for longer windows.

Non-examples:

1. relative distance bias.
2. rotation of Q/K coordinates.

**Derivation habit.**

1. State whether the scheme is absolute or relative.
2. State whether the signal is added to hidden states, applied to Q/K, or added to scores.
3. Check shape compatibility with heads, sequence length, and cached decoding.
4. Test length behavior beyond the training context if the model will be served there.
5. Keep masks separate from position signals.

**Implementation lens.**

Position bugs are subtle because tensor shapes can still look correct. A decode loop can run while assigning the wrong position id to every generated token. A long-context model can accept a large input while quality collapses in the middle of the context.

The practical defense is small tests: compare attention with shifted inputs, verify RoPE norm preservation, check ALiBi bias values by distance, and run retrieval probes at several positions in the context window.

### 4.3 Interpolation resizing

**Purpose.** Interpolation resizing focuses on adapting rows to longer contexts. This is the part of transformer math that tells attention where each token is and how far apart tokens are.

$$\operatorname{ALiBi}_{ij}=m_h(i-j)\quad\text{for }j\le i.$$

**Operational definition.**

Position encoding injects order into transformer computation so identical tokens at different positions can have different roles.

**Worked reading.**

Without a position signal, self-attention can mix content but cannot know which token came first.

| Scheme | Where position enters | Typical strength | Typical risk |
| --- | --- | --- | --- |
| sinusoidal | added to hidden states | fixed and simple | limited flexibility |
| learned absolute | added learned row | strong in-range fit | weak extrapolation |
| relative bias | attention score | offset-aware | implementation complexity |
| RoPE | rotates Q/K | relative dot products | scaling choices matter |
| ALiBi | attention score bias | simple extrapolation | distance prior may be too rigid |

Examples:

1. sinusoidal features.
2. position rows.
3. relative offsets.

Non-examples:

1. bag-of-words attention.
2. token ids used as positions.

**Derivation habit.**

1. State whether the scheme is absolute or relative.
2. State whether the signal is added to hidden states, applied to Q/K, or added to scores.
3. Check shape compatibility with heads, sequence length, and cached decoding.
4. Test length behavior beyond the training context if the model will be served there.
5. Keep masks separate from position signals.

**Implementation lens.**

Position bugs are subtle because tensor shapes can still look correct. A decode loop can run while assigning the wrong position id to every generated token. A long-context model can accept a large input while quality collapses in the middle of the context.

The practical defense is small tests: compare attention with shifted inputs, verify RoPE norm preservation, check ALiBi bias values by distance, and run retrieval probes at several positions in the context window.

### 4.4 BERT GPT-style usage

**Purpose.** BERT GPT-style usage focuses on where learned rows appear. This is the part of transformer math that tells attention where each token is and how far apart tokens are.

$$\lambda_k=10000^{2k/d}.$$

**Operational definition.**

Position encoding injects order into transformer computation so identical tokens at different positions can have different roles.

**Worked reading.**

Without a position signal, self-attention can mix content but cannot know which token came first.

| Scheme | Where position enters | Typical strength | Typical risk |
| --- | --- | --- | --- |
| sinusoidal | added to hidden states | fixed and simple | limited flexibility |
| learned absolute | added learned row | strong in-range fit | weak extrapolation |
| relative bias | attention score | offset-aware | implementation complexity |
| RoPE | rotates Q/K | relative dot products | scaling choices matter |
| ALiBi | attention score bias | simple extrapolation | distance prior may be too rigid |

Examples:

1. sinusoidal features.
2. position rows.
3. relative offsets.

Non-examples:

1. bag-of-words attention.
2. token ids used as positions.

**Derivation habit.**

1. State whether the scheme is absolute or relative.
2. State whether the signal is added to hidden states, applied to Q/K, or added to scores.
3. Check shape compatibility with heads, sequence length, and cached decoding.
4. Test length behavior beyond the training context if the model will be served there.
5. Keep masks separate from position signals.

**Implementation lens.**

Position bugs are subtle because tensor shapes can still look correct. A decode loop can run while assigning the wrong position id to every generated token. A long-context model can accept a large input while quality collapses in the middle of the context.

The practical defense is small tests: compare attention with shifted inputs, verify RoPE norm preservation, check ALiBi bias values by distance, and run retrieval probes at several positions in the context window.

### 4.5 Failure modes

**Purpose.** Failure modes focuses on length extrapolation and out-of-range ids. This is the part of transformer math that tells attention where each token is and how far apart tokens are.

$$\operatorname{position\ id}_{\mathrm{decode}}=T_{\mathrm{prefix}}+t.$$

**Operational definition.**

Position encoding injects order into transformer computation so identical tokens at different positions can have different roles.

**Worked reading.**

Without a position signal, self-attention can mix content but cannot know which token came first.

| Scheme | Where position enters | Typical strength | Typical risk |
| --- | --- | --- | --- |
| sinusoidal | added to hidden states | fixed and simple | limited flexibility |
| learned absolute | added learned row | strong in-range fit | weak extrapolation |
| relative bias | attention score | offset-aware | implementation complexity |
| RoPE | rotates Q/K | relative dot products | scaling choices matter |
| ALiBi | attention score bias | simple extrapolation | distance prior may be too rigid |

Examples:

1. sinusoidal features.
2. position rows.
3. relative offsets.

Non-examples:

1. bag-of-words attention.
2. token ids used as positions.

**Derivation habit.**

1. State whether the scheme is absolute or relative.
2. State whether the signal is added to hidden states, applied to Q/K, or added to scores.
3. Check shape compatibility with heads, sequence length, and cached decoding.
4. Test length behavior beyond the training context if the model will be served there.
5. Keep masks separate from position signals.

**Implementation lens.**

Position bugs are subtle because tensor shapes can still look correct. A decode loop can run while assigning the wrong position id to every generated token. A long-context model can accept a large input while quality collapses in the middle of the context.

The practical defense is small tests: compare attention with shifted inputs, verify RoPE norm preservation, check ALiBi bias values by distance, and run retrieval probes at several positions in the context window.

## 5. Relative Position Representations

Relative Position Representations explains how transformer sequence order is represented in hidden states or attention scores.

### 5.1 Relative bias matrices

**Purpose.** Relative bias matrices focuses on score additions based on distance. This is the part of transformer math that tells attention where each token is and how far apart tokens are.

$$\operatorname{RoPE}(\mathbf{q}_p)=R_p\mathbf{q}_p.$$

**Operational definition.**

ALiBi-style methods add a distance-dependent bias directly to attention scores before softmax.

**Worked reading.**

Older keys can receive a linear penalty while causal masking still controls which keys are visible.

| Scheme | Where position enters | Typical strength | Typical risk |
| --- | --- | --- | --- |
| sinusoidal | added to hidden states | fixed and simple | limited flexibility |
| learned absolute | added learned row | strong in-range fit | weak extrapolation |
| relative bias | attention score | offset-aware | implementation complexity |
| RoPE | rotates Q/K | relative dot products | scaling choices matter |
| ALiBi | attention score bias | simple extrapolation | distance prior may be too rigid |

Examples:

1. long-context extrapolation.
2. head-specific slopes.
3. score-space position signals.

Non-examples:

1. residual-stream position embeddings.
2. bias added after softmax.

**Derivation habit.**

1. State whether the scheme is absolute or relative.
2. State whether the signal is added to hidden states, applied to Q/K, or added to scores.
3. Check shape compatibility with heads, sequence length, and cached decoding.
4. Test length behavior beyond the training context if the model will be served there.
5. Keep masks separate from position signals.

**Implementation lens.**

Position bugs are subtle because tensor shapes can still look correct. A decode loop can run while assigning the wrong position id to every generated token. A long-context model can accept a large input while quality collapses in the middle of the context.

The practical defense is small tests: compare attention with shifted inputs, verify RoPE norm preservation, check ALiBi bias values by distance, and run retrieval probes at several positions in the context window.

### 5.2 Shaw-style relative keys

**Purpose.** Shaw-style relative keys focuses on relative vectors inside attention compatibility. This is the part of transformer math that tells attention where each token is and how far apart tokens are.

$$\operatorname{ALiBi}_{ij}=m_h(i-j)\quad\text{for }j\le i.$$

**Operational definition.**

Relative position methods make attention depend on pairwise distance instead of only absolute index identity.

**Worked reading.**

The same offset can share parameters across many positions, which is useful when patterns repeat across the sequence.

| Scheme | Where position enters | Typical strength | Typical risk |
| --- | --- | --- | --- |
| sinusoidal | added to hidden states | fixed and simple | limited flexibility |
| learned absolute | added learned row | strong in-range fit | weak extrapolation |
| relative bias | attention score | offset-aware | implementation complexity |
| RoPE | rotates Q/K | relative dot products | scaling choices matter |
| ALiBi | attention score bias | simple extrapolation | distance prior may be too rigid |

Examples:

1. relative bias.
2. relative keys.
3. bucketed distance bins.

Non-examples:

1. one independent vector per absolute index.
2. no order signal at all.

**Derivation habit.**

1. State whether the scheme is absolute or relative.
2. State whether the signal is added to hidden states, applied to Q/K, or added to scores.
3. Check shape compatibility with heads, sequence length, and cached decoding.
4. Test length behavior beyond the training context if the model will be served there.
5. Keep masks separate from position signals.

**Implementation lens.**

Position bugs are subtle because tensor shapes can still look correct. A decode loop can run while assigning the wrong position id to every generated token. A long-context model can accept a large input while quality collapses in the middle of the context.

The practical defense is small tests: compare attention with shifted inputs, verify RoPE norm preservation, check ALiBi bias values by distance, and run retrieval probes at several positions in the context window.

### 5.3 Transformer-XL intuition

**Purpose.** Transformer-XL intuition focuses on segment recurrence and relative positions. This is the part of transformer math that tells attention where each token is and how far apart tokens are.

$$\lambda_k=10000^{2k/d}.$$

**Operational definition.**

Position encoding injects order into transformer computation so identical tokens at different positions can have different roles.

**Worked reading.**

Without a position signal, self-attention can mix content but cannot know which token came first.

| Scheme | Where position enters | Typical strength | Typical risk |
| --- | --- | --- | --- |
| sinusoidal | added to hidden states | fixed and simple | limited flexibility |
| learned absolute | added learned row | strong in-range fit | weak extrapolation |
| relative bias | attention score | offset-aware | implementation complexity |
| RoPE | rotates Q/K | relative dot products | scaling choices matter |
| ALiBi | attention score bias | simple extrapolation | distance prior may be too rigid |

Examples:

1. sinusoidal features.
2. position rows.
3. relative offsets.

Non-examples:

1. bag-of-words attention.
2. token ids used as positions.

**Derivation habit.**

1. State whether the scheme is absolute or relative.
2. State whether the signal is added to hidden states, applied to Q/K, or added to scores.
3. Check shape compatibility with heads, sequence length, and cached decoding.
4. Test length behavior beyond the training context if the model will be served there.
5. Keep masks separate from position signals.

**Implementation lens.**

Position bugs are subtle because tensor shapes can still look correct. A decode loop can run while assigning the wrong position id to every generated token. A long-context model can accept a large input while quality collapses in the middle of the context.

The practical defense is small tests: compare attention with shifted inputs, verify RoPE norm preservation, check ALiBi bias values by distance, and run retrieval probes at several positions in the context window.

### 5.4 Bucketed distances

**Purpose.** Bucketed distances focuses on sharing parameters across far offsets. This is the part of transformer math that tells attention where each token is and how far apart tokens are.

$$\operatorname{position\ id}_{\mathrm{decode}}=T_{\mathrm{prefix}}+t.$$

**Operational definition.**

Relative position methods make attention depend on pairwise distance instead of only absolute index identity.

**Worked reading.**

The same offset can share parameters across many positions, which is useful when patterns repeat across the sequence.

| Scheme | Where position enters | Typical strength | Typical risk |
| --- | --- | --- | --- |
| sinusoidal | added to hidden states | fixed and simple | limited flexibility |
| learned absolute | added learned row | strong in-range fit | weak extrapolation |
| relative bias | attention score | offset-aware | implementation complexity |
| RoPE | rotates Q/K | relative dot products | scaling choices matter |
| ALiBi | attention score bias | simple extrapolation | distance prior may be too rigid |

Examples:

1. relative bias.
2. relative keys.
3. bucketed distance bins.

Non-examples:

1. one independent vector per absolute index.
2. no order signal at all.

**Derivation habit.**

1. State whether the scheme is absolute or relative.
2. State whether the signal is added to hidden states, applied to Q/K, or added to scores.
3. Check shape compatibility with heads, sequence length, and cached decoding.
4. Test length behavior beyond the training context if the model will be served there.
5. Keep masks separate from position signals.

**Implementation lens.**

Position bugs are subtle because tensor shapes can still look correct. A decode loop can run while assigning the wrong position id to every generated token. A long-context model can accept a large input while quality collapses in the middle of the context.

The practical defense is small tests: compare attention with shifted inputs, verify RoPE norm preservation, check ALiBi bias values by distance, and run retrieval probes at several positions in the context window.

### 5.5 When relative position helps

**Purpose.** When relative position helps focuses on translation invariance and long sequences. This is the part of transformer math that tells attention where each token is and how far apart tokens are.

$$\operatorname{PE}_{p,2k}=\sin\left(p/10000^{2k/d}\right),\qquad \operatorname{PE}_{p,2k+1}=\cos\left(p/10000^{2k/d}\right).$$

**Operational definition.**

Relative position methods make attention depend on pairwise distance instead of only absolute index identity.

**Worked reading.**

The same offset can share parameters across many positions, which is useful when patterns repeat across the sequence.

| Scheme | Where position enters | Typical strength | Typical risk |
| --- | --- | --- | --- |
| sinusoidal | added to hidden states | fixed and simple | limited flexibility |
| learned absolute | added learned row | strong in-range fit | weak extrapolation |
| relative bias | attention score | offset-aware | implementation complexity |
| RoPE | rotates Q/K | relative dot products | scaling choices matter |
| ALiBi | attention score bias | simple extrapolation | distance prior may be too rigid |

Examples:

1. relative bias.
2. relative keys.
3. bucketed distance bins.

Non-examples:

1. one independent vector per absolute index.
2. no order signal at all.

**Derivation habit.**

1. State whether the scheme is absolute or relative.
2. State whether the signal is added to hidden states, applied to Q/K, or added to scores.
3. Check shape compatibility with heads, sequence length, and cached decoding.
4. Test length behavior beyond the training context if the model will be served there.
5. Keep masks separate from position signals.

**Implementation lens.**

Position bugs are subtle because tensor shapes can still look correct. A decode loop can run while assigning the wrong position id to every generated token. A long-context model can accept a large input while quality collapses in the middle of the context.

The practical defense is small tests: compare attention with shifted inputs, verify RoPE norm preservation, check ALiBi bias values by distance, and run retrieval probes at several positions in the context window.

## 6. Rotary Positional Embeddings

Rotary Positional Embeddings explains how transformer sequence order is represented in hidden states or attention scores.

### 6.1 RoPE rotations

**Purpose.** RoPE rotations focuses on rotating query and key coordinate pairs. This is the part of transformer math that tells attention where each token is and how far apart tokens are.

$$\operatorname{ALiBi}_{ij}=m_h(i-j)\quad\text{for }j\le i.$$

**Operational definition.**

RoPE encodes position by rotating query and key coordinate pairs with position-dependent angles.

**Worked reading.**

Because rotations compose by angle differences, a query at position $i$ and key at position $j$ can expose the relative offset $j-i$ through their dot product.

| Scheme | Where position enters | Typical strength | Typical risk |
| --- | --- | --- | --- |
| sinusoidal | added to hidden states | fixed and simple | limited flexibility |
| learned absolute | added learned row | strong in-range fit | weak extrapolation |
| relative bias | attention score | offset-aware | implementation complexity |
| RoPE | rotates Q/K | relative dot products | scaling choices matter |
| ALiBi | attention score bias | simple extrapolation | distance prior may be too rigid |

Examples:

1. decoder-only LLMs.
2. relative dot-product behavior.
3. long-context scaling variants.

Non-examples:

1. adding a position vector to $X$.
2. learned absolute position rows.

**Derivation habit.**

1. State whether the scheme is absolute or relative.
2. State whether the signal is added to hidden states, applied to Q/K, or added to scores.
3. Check shape compatibility with heads, sequence length, and cached decoding.
4. Test length behavior beyond the training context if the model will be served there.
5. Keep masks separate from position signals.

**Implementation lens.**

Position bugs are subtle because tensor shapes can still look correct. A decode loop can run while assigning the wrong position id to every generated token. A long-context model can accept a large input while quality collapses in the middle of the context.

The practical defense is small tests: compare attention with shifted inputs, verify RoPE norm preservation, check ALiBi bias values by distance, and run retrieval probes at several positions in the context window.

### 6.2 Relative dot-product property

**Purpose.** Relative dot-product property focuses on dependence on position difference. This is the part of transformer math that tells attention where each token is and how far apart tokens are.

$$\lambda_k=10000^{2k/d}.$$

**Operational definition.**

RoPE encodes position by rotating query and key coordinate pairs with position-dependent angles.

**Worked reading.**

Because rotations compose by angle differences, a query at position $i$ and key at position $j$ can expose the relative offset $j-i$ through their dot product.

| Scheme | Where position enters | Typical strength | Typical risk |
| --- | --- | --- | --- |
| sinusoidal | added to hidden states | fixed and simple | limited flexibility |
| learned absolute | added learned row | strong in-range fit | weak extrapolation |
| relative bias | attention score | offset-aware | implementation complexity |
| RoPE | rotates Q/K | relative dot products | scaling choices matter |
| ALiBi | attention score bias | simple extrapolation | distance prior may be too rigid |

Examples:

1. decoder-only LLMs.
2. relative dot-product behavior.
3. long-context scaling variants.

Non-examples:

1. adding a position vector to $X$.
2. learned absolute position rows.

**Derivation habit.**

1. State whether the scheme is absolute or relative.
2. State whether the signal is added to hidden states, applied to Q/K, or added to scores.
3. Check shape compatibility with heads, sequence length, and cached decoding.
4. Test length behavior beyond the training context if the model will be served there.
5. Keep masks separate from position signals.

**Implementation lens.**

Position bugs are subtle because tensor shapes can still look correct. A decode loop can run while assigning the wrong position id to every generated token. A long-context model can accept a large input while quality collapses in the middle of the context.

The practical defense is small tests: compare attention with shifted inputs, verify RoPE norm preservation, check ALiBi bias values by distance, and run retrieval probes at several positions in the context window.

### 6.3 Frequency base and dimensions

**Purpose.** Frequency base and dimensions focuses on how angular speeds are assigned. This is the part of transformer math that tells attention where each token is and how far apart tokens are.

$$\operatorname{position\ id}_{\mathrm{decode}}=T_{\mathrm{prefix}}+t.$$

**Operational definition.**

Position encoding injects order into transformer computation so identical tokens at different positions can have different roles.

**Worked reading.**

Without a position signal, self-attention can mix content but cannot know which token came first.

| Scheme | Where position enters | Typical strength | Typical risk |
| --- | --- | --- | --- |
| sinusoidal | added to hidden states | fixed and simple | limited flexibility |
| learned absolute | added learned row | strong in-range fit | weak extrapolation |
| relative bias | attention score | offset-aware | implementation complexity |
| RoPE | rotates Q/K | relative dot products | scaling choices matter |
| ALiBi | attention score bias | simple extrapolation | distance prior may be too rigid |

Examples:

1. sinusoidal features.
2. position rows.
3. relative offsets.

Non-examples:

1. bag-of-words attention.
2. token ids used as positions.

**Derivation habit.**

1. State whether the scheme is absolute or relative.
2. State whether the signal is added to hidden states, applied to Q/K, or added to scores.
3. Check shape compatibility with heads, sequence length, and cached decoding.
4. Test length behavior beyond the training context if the model will be served there.
5. Keep masks separate from position signals.

**Implementation lens.**

Position bugs are subtle because tensor shapes can still look correct. A decode loop can run while assigning the wrong position id to every generated token. A long-context model can accept a large input while quality collapses in the middle of the context.

The practical defense is small tests: compare attention with shifted inputs, verify RoPE norm preservation, check ALiBi bias values by distance, and run retrieval probes at several positions in the context window.

### 6.4 Long-context RoPE scaling

**Purpose.** Long-context RoPE scaling focuses on why interpolation and base changes are used. This is the part of transformer math that tells attention where each token is and how far apart tokens are.

$$\operatorname{PE}_{p,2k}=\sin\left(p/10000^{2k/d}\right),\qquad \operatorname{PE}_{p,2k+1}=\cos\left(p/10000^{2k/d}\right).$$

**Operational definition.**

RoPE encodes position by rotating query and key coordinate pairs with position-dependent angles.

**Worked reading.**

Because rotations compose by angle differences, a query at position $i$ and key at position $j$ can expose the relative offset $j-i$ through their dot product.

| Scheme | Where position enters | Typical strength | Typical risk |
| --- | --- | --- | --- |
| sinusoidal | added to hidden states | fixed and simple | limited flexibility |
| learned absolute | added learned row | strong in-range fit | weak extrapolation |
| relative bias | attention score | offset-aware | implementation complexity |
| RoPE | rotates Q/K | relative dot products | scaling choices matter |
| ALiBi | attention score bias | simple extrapolation | distance prior may be too rigid |

Examples:

1. decoder-only LLMs.
2. relative dot-product behavior.
3. long-context scaling variants.

Non-examples:

1. adding a position vector to $X$.
2. learned absolute position rows.

**Derivation habit.**

1. State whether the scheme is absolute or relative.
2. State whether the signal is added to hidden states, applied to Q/K, or added to scores.
3. Check shape compatibility with heads, sequence length, and cached decoding.
4. Test length behavior beyond the training context if the model will be served there.
5. Keep masks separate from position signals.

**Implementation lens.**

Position bugs are subtle because tensor shapes can still look correct. A decode loop can run while assigning the wrong position id to every generated token. A long-context model can accept a large input while quality collapses in the middle of the context.

The practical defense is small tests: compare attention with shifted inputs, verify RoPE norm preservation, check ALiBi bias values by distance, and run retrieval probes at several positions in the context window.

### 6.5 Implementation checks

**Purpose.** Implementation checks focuses on norm preservation and pairwise rotations. This is the part of transformer math that tells attention where each token is and how far apart tokens are.

$$\mathbf{h}_p=\mathbf{x}_p+\mathbf{p}_p.$$

**Operational definition.**

Position encoding injects order into transformer computation so identical tokens at different positions can have different roles.

**Worked reading.**

Without a position signal, self-attention can mix content but cannot know which token came first.

| Scheme | Where position enters | Typical strength | Typical risk |
| --- | --- | --- | --- |
| sinusoidal | added to hidden states | fixed and simple | limited flexibility |
| learned absolute | added learned row | strong in-range fit | weak extrapolation |
| relative bias | attention score | offset-aware | implementation complexity |
| RoPE | rotates Q/K | relative dot products | scaling choices matter |
| ALiBi | attention score bias | simple extrapolation | distance prior may be too rigid |

Examples:

1. sinusoidal features.
2. position rows.
3. relative offsets.

Non-examples:

1. bag-of-words attention.
2. token ids used as positions.

**Derivation habit.**

1. State whether the scheme is absolute or relative.
2. State whether the signal is added to hidden states, applied to Q/K, or added to scores.
3. Check shape compatibility with heads, sequence length, and cached decoding.
4. Test length behavior beyond the training context if the model will be served there.
5. Keep masks separate from position signals.

**Implementation lens.**

Position bugs are subtle because tensor shapes can still look correct. A decode loop can run while assigning the wrong position id to every generated token. A long-context model can accept a large input while quality collapses in the middle of the context.

The practical defense is small tests: compare attention with shifted inputs, verify RoPE norm preservation, check ALiBi bias values by distance, and run retrieval probes at several positions in the context window.

## 7. ALiBi and Bias Methods

ALiBi and Bias Methods explains how transformer sequence order is represented in hidden states or attention scores.

### 7.1 Linear attention biases

**Purpose.** Linear attention biases focuses on distance penalty added to scores. This is the part of transformer math that tells attention where each token is and how far apart tokens are.

$$\lambda_k=10000^{2k/d}.$$

**Operational definition.**

ALiBi-style methods add a distance-dependent bias directly to attention scores before softmax.

**Worked reading.**

Older keys can receive a linear penalty while causal masking still controls which keys are visible.

| Scheme | Where position enters | Typical strength | Typical risk |
| --- | --- | --- | --- |
| sinusoidal | added to hidden states | fixed and simple | limited flexibility |
| learned absolute | added learned row | strong in-range fit | weak extrapolation |
| relative bias | attention score | offset-aware | implementation complexity |
| RoPE | rotates Q/K | relative dot products | scaling choices matter |
| ALiBi | attention score bias | simple extrapolation | distance prior may be too rigid |

Examples:

1. long-context extrapolation.
2. head-specific slopes.
3. score-space position signals.

Non-examples:

1. residual-stream position embeddings.
2. bias added after softmax.

**Derivation habit.**

1. State whether the scheme is absolute or relative.
2. State whether the signal is added to hidden states, applied to Q/K, or added to scores.
3. Check shape compatibility with heads, sequence length, and cached decoding.
4. Test length behavior beyond the training context if the model will be served there.
5. Keep masks separate from position signals.

**Implementation lens.**

Position bugs are subtle because tensor shapes can still look correct. A decode loop can run while assigning the wrong position id to every generated token. A long-context model can accept a large input while quality collapses in the middle of the context.

The practical defense is small tests: compare attention with shifted inputs, verify RoPE norm preservation, check ALiBi bias values by distance, and run retrieval probes at several positions in the context window.

### 7.2 Head-specific slopes

**Purpose.** Head-specific slopes focuses on multiple distance scales across heads. This is the part of transformer math that tells attention where each token is and how far apart tokens are.

$$\operatorname{position\ id}_{\mathrm{decode}}=T_{\mathrm{prefix}}+t.$$

**Operational definition.**

Position encoding injects order into transformer computation so identical tokens at different positions can have different roles.

**Worked reading.**

Without a position signal, self-attention can mix content but cannot know which token came first.

| Scheme | Where position enters | Typical strength | Typical risk |
| --- | --- | --- | --- |
| sinusoidal | added to hidden states | fixed and simple | limited flexibility |
| learned absolute | added learned row | strong in-range fit | weak extrapolation |
| relative bias | attention score | offset-aware | implementation complexity |
| RoPE | rotates Q/K | relative dot products | scaling choices matter |
| ALiBi | attention score bias | simple extrapolation | distance prior may be too rigid |

Examples:

1. sinusoidal features.
2. position rows.
3. relative offsets.

Non-examples:

1. bag-of-words attention.
2. token ids used as positions.

**Derivation habit.**

1. State whether the scheme is absolute or relative.
2. State whether the signal is added to hidden states, applied to Q/K, or added to scores.
3. Check shape compatibility with heads, sequence length, and cached decoding.
4. Test length behavior beyond the training context if the model will be served there.
5. Keep masks separate from position signals.

**Implementation lens.**

Position bugs are subtle because tensor shapes can still look correct. A decode loop can run while assigning the wrong position id to every generated token. A long-context model can accept a large input while quality collapses in the middle of the context.

The practical defense is small tests: compare attention with shifted inputs, verify RoPE norm preservation, check ALiBi bias values by distance, and run retrieval probes at several positions in the context window.

### 7.3 Extrapolation behavior

**Purpose.** Extrapolation behavior focuses on no learned position table required. This is the part of transformer math that tells attention where each token is and how far apart tokens are.

$$\operatorname{PE}_{p,2k}=\sin\left(p/10000^{2k/d}\right),\qquad \operatorname{PE}_{p,2k+1}=\cos\left(p/10000^{2k/d}\right).$$

**Operational definition.**

Position encoding injects order into transformer computation so identical tokens at different positions can have different roles.

**Worked reading.**

Without a position signal, self-attention can mix content but cannot know which token came first.

| Scheme | Where position enters | Typical strength | Typical risk |
| --- | --- | --- | --- |
| sinusoidal | added to hidden states | fixed and simple | limited flexibility |
| learned absolute | added learned row | strong in-range fit | weak extrapolation |
| relative bias | attention score | offset-aware | implementation complexity |
| RoPE | rotates Q/K | relative dot products | scaling choices matter |
| ALiBi | attention score bias | simple extrapolation | distance prior may be too rigid |

Examples:

1. sinusoidal features.
2. position rows.
3. relative offsets.

Non-examples:

1. bag-of-words attention.
2. token ids used as positions.

**Derivation habit.**

1. State whether the scheme is absolute or relative.
2. State whether the signal is added to hidden states, applied to Q/K, or added to scores.
3. Check shape compatibility with heads, sequence length, and cached decoding.
4. Test length behavior beyond the training context if the model will be served there.
5. Keep masks separate from position signals.

**Implementation lens.**

Position bugs are subtle because tensor shapes can still look correct. A decode loop can run while assigning the wrong position id to every generated token. A long-context model can accept a large input while quality collapses in the middle of the context.

The practical defense is small tests: compare attention with shifted inputs, verify RoPE norm preservation, check ALiBi bias values by distance, and run retrieval probes at several positions in the context window.

### 7.4 Bias versus embedding

**Purpose.** Bias versus embedding focuses on score-space signal not residual-stream signal. This is the part of transformer math that tells attention where each token is and how far apart tokens are.

$$\mathbf{h}_p=\mathbf{x}_p+\mathbf{p}_p.$$

**Operational definition.**

ALiBi-style methods add a distance-dependent bias directly to attention scores before softmax.

**Worked reading.**

Older keys can receive a linear penalty while causal masking still controls which keys are visible.

| Scheme | Where position enters | Typical strength | Typical risk |
| --- | --- | --- | --- |
| sinusoidal | added to hidden states | fixed and simple | limited flexibility |
| learned absolute | added learned row | strong in-range fit | weak extrapolation |
| relative bias | attention score | offset-aware | implementation complexity |
| RoPE | rotates Q/K | relative dot products | scaling choices matter |
| ALiBi | attention score bias | simple extrapolation | distance prior may be too rigid |

Examples:

1. long-context extrapolation.
2. head-specific slopes.
3. score-space position signals.

Non-examples:

1. residual-stream position embeddings.
2. bias added after softmax.

**Derivation habit.**

1. State whether the scheme is absolute or relative.
2. State whether the signal is added to hidden states, applied to Q/K, or added to scores.
3. Check shape compatibility with heads, sequence length, and cached decoding.
4. Test length behavior beyond the training context if the model will be served there.
5. Keep masks separate from position signals.

**Implementation lens.**

Position bugs are subtle because tensor shapes can still look correct. A decode loop can run while assigning the wrong position id to every generated token. A long-context model can accept a large input while quality collapses in the middle of the context.

The practical defense is small tests: compare attention with shifted inputs, verify RoPE norm preservation, check ALiBi bias values by distance, and run retrieval probes at several positions in the context window.

### 7.5 Mask interaction

**Purpose.** Mask interaction focuses on biases still respect causal visibility. This is the part of transformer math that tells attention where each token is and how far apart tokens are.

$$S_{ij}=\frac{\mathbf{q}_i^\top\mathbf{k}_j}{\sqrt{d_k}}+b_{i-j}.$$

**Operational definition.**

Position encoding injects order into transformer computation so identical tokens at different positions can have different roles.

**Worked reading.**

Without a position signal, self-attention can mix content but cannot know which token came first.

| Scheme | Where position enters | Typical strength | Typical risk |
| --- | --- | --- | --- |
| sinusoidal | added to hidden states | fixed and simple | limited flexibility |
| learned absolute | added learned row | strong in-range fit | weak extrapolation |
| relative bias | attention score | offset-aware | implementation complexity |
| RoPE | rotates Q/K | relative dot products | scaling choices matter |
| ALiBi | attention score bias | simple extrapolation | distance prior may be too rigid |

Examples:

1. sinusoidal features.
2. position rows.
3. relative offsets.

Non-examples:

1. bag-of-words attention.
2. token ids used as positions.

**Derivation habit.**

1. State whether the scheme is absolute or relative.
2. State whether the signal is added to hidden states, applied to Q/K, or added to scores.
3. Check shape compatibility with heads, sequence length, and cached decoding.
4. Test length behavior beyond the training context if the model will be served there.
5. Keep masks separate from position signals.

**Implementation lens.**

Position bugs are subtle because tensor shapes can still look correct. A decode loop can run while assigning the wrong position id to every generated token. A long-context model can accept a large input while quality collapses in the middle of the context.

The practical defense is small tests: compare attention with shifted inputs, verify RoPE norm preservation, check ALiBi bias values by distance, and run retrieval probes at several positions in the context window.

## 8. Diagnostics and LLM Practice

Diagnostics and LLM Practice explains how transformer sequence order is represented in hidden states or attention scores.

### 8.1 Position-sensitivity tests

**Purpose.** Position-sensitivity tests focuses on permutation and shift tests. This is the part of transformer math that tells attention where each token is and how far apart tokens are.

$$\operatorname{position\ id}_{\mathrm{decode}}=T_{\mathrm{prefix}}+t.$$

**Operational definition.**

Position diagnostics test whether a model uses order correctly and remains reliable at target context lengths.

**Worked reading.**

A correct serving stack must assign decode position ids consistently with the KV cache and any RoPE or bias scheme.

| Scheme | Where position enters | Typical strength | Typical risk |
| --- | --- | --- | --- |
| sinusoidal | added to hidden states | fixed and simple | limited flexibility |
| learned absolute | added learned row | strong in-range fit | weak extrapolation |
| relative bias | attention score | offset-aware | implementation complexity |
| RoPE | rotates Q/K | relative dot products | scaling choices matter |
| ALiBi | attention score bias | simple extrapolation | distance prior may be too rigid |

Examples:

1. shift tests.
2. needle retrieval by position.
3. KV-cache position id checks.

Non-examples:

1. only testing short prompts.
2. assuming max context length implies uniform quality.

**Derivation habit.**

1. State whether the scheme is absolute or relative.
2. State whether the signal is added to hidden states, applied to Q/K, or added to scores.
3. Check shape compatibility with heads, sequence length, and cached decoding.
4. Test length behavior beyond the training context if the model will be served there.
5. Keep masks separate from position signals.

**Implementation lens.**

Position bugs are subtle because tensor shapes can still look correct. A decode loop can run while assigning the wrong position id to every generated token. A long-context model can accept a large input while quality collapses in the middle of the context.

The practical defense is small tests: compare attention with shifted inputs, verify RoPE norm preservation, check ALiBi bias values by distance, and run retrieval probes at several positions in the context window.

### 8.2 Long-context degradation

**Purpose.** Long-context degradation focuses on lost-in-the-middle and recency effects. This is the part of transformer math that tells attention where each token is and how far apart tokens are.

$$\operatorname{PE}_{p,2k}=\sin\left(p/10000^{2k/d}\right),\qquad \operatorname{PE}_{p,2k+1}=\cos\left(p/10000^{2k/d}\right).$$

**Operational definition.**

Position diagnostics test whether a model uses order correctly and remains reliable at target context lengths.

**Worked reading.**

A correct serving stack must assign decode position ids consistently with the KV cache and any RoPE or bias scheme.

| Scheme | Where position enters | Typical strength | Typical risk |
| --- | --- | --- | --- |
| sinusoidal | added to hidden states | fixed and simple | limited flexibility |
| learned absolute | added learned row | strong in-range fit | weak extrapolation |
| relative bias | attention score | offset-aware | implementation complexity |
| RoPE | rotates Q/K | relative dot products | scaling choices matter |
| ALiBi | attention score bias | simple extrapolation | distance prior may be too rigid |

Examples:

1. shift tests.
2. needle retrieval by position.
3. KV-cache position id checks.

Non-examples:

1. only testing short prompts.
2. assuming max context length implies uniform quality.

**Derivation habit.**

1. State whether the scheme is absolute or relative.
2. State whether the signal is added to hidden states, applied to Q/K, or added to scores.
3. Check shape compatibility with heads, sequence length, and cached decoding.
4. Test length behavior beyond the training context if the model will be served there.
5. Keep masks separate from position signals.

**Implementation lens.**

Position bugs are subtle because tensor shapes can still look correct. A decode loop can run while assigning the wrong position id to every generated token. A long-context model can accept a large input while quality collapses in the middle of the context.

The practical defense is small tests: compare attention with shifted inputs, verify RoPE norm preservation, check ALiBi bias values by distance, and run retrieval probes at several positions in the context window.

### 8.3 Attention pattern inspection

**Purpose.** Attention pattern inspection focuses on distance distributions and entropy. This is the part of transformer math that tells attention where each token is and how far apart tokens are.

$$\mathbf{h}_p=\mathbf{x}_p+\mathbf{p}_p.$$

**Operational definition.**

Position encoding injects order into transformer computation so identical tokens at different positions can have different roles.

**Worked reading.**

Without a position signal, self-attention can mix content but cannot know which token came first.

| Scheme | Where position enters | Typical strength | Typical risk |
| --- | --- | --- | --- |
| sinusoidal | added to hidden states | fixed and simple | limited flexibility |
| learned absolute | added learned row | strong in-range fit | weak extrapolation |
| relative bias | attention score | offset-aware | implementation complexity |
| RoPE | rotates Q/K | relative dot products | scaling choices matter |
| ALiBi | attention score bias | simple extrapolation | distance prior may be too rigid |

Examples:

1. sinusoidal features.
2. position rows.
3. relative offsets.

Non-examples:

1. bag-of-words attention.
2. token ids used as positions.

**Derivation habit.**

1. State whether the scheme is absolute or relative.
2. State whether the signal is added to hidden states, applied to Q/K, or added to scores.
3. Check shape compatibility with heads, sequence length, and cached decoding.
4. Test length behavior beyond the training context if the model will be served there.
5. Keep masks separate from position signals.

**Implementation lens.**

Position bugs are subtle because tensor shapes can still look correct. A decode loop can run while assigning the wrong position id to every generated token. A long-context model can accept a large input while quality collapses in the middle of the context.

The practical defense is small tests: compare attention with shifted inputs, verify RoPE norm preservation, check ALiBi bias values by distance, and run retrieval probes at several positions in the context window.

### 8.4 Serving and KV cache

**Purpose.** Serving and KV cache focuses on position ids during decode. This is the part of transformer math that tells attention where each token is and how far apart tokens are.

$$S_{ij}=\frac{\mathbf{q}_i^\top\mathbf{k}_j}{\sqrt{d_k}}+b_{i-j}.$$

**Operational definition.**

Position diagnostics test whether a model uses order correctly and remains reliable at target context lengths.

**Worked reading.**

A correct serving stack must assign decode position ids consistently with the KV cache and any RoPE or bias scheme.

| Scheme | Where position enters | Typical strength | Typical risk |
| --- | --- | --- | --- |
| sinusoidal | added to hidden states | fixed and simple | limited flexibility |
| learned absolute | added learned row | strong in-range fit | weak extrapolation |
| relative bias | attention score | offset-aware | implementation complexity |
| RoPE | rotates Q/K | relative dot products | scaling choices matter |
| ALiBi | attention score bias | simple extrapolation | distance prior may be too rigid |

Examples:

1. shift tests.
2. needle retrieval by position.
3. KV-cache position id checks.

Non-examples:

1. only testing short prompts.
2. assuming max context length implies uniform quality.

**Derivation habit.**

1. State whether the scheme is absolute or relative.
2. State whether the signal is added to hidden states, applied to Q/K, or added to scores.
3. Check shape compatibility with heads, sequence length, and cached decoding.
4. Test length behavior beyond the training context if the model will be served there.
5. Keep masks separate from position signals.

**Implementation lens.**

Position bugs are subtle because tensor shapes can still look correct. A decode loop can run while assigning the wrong position id to every generated token. A long-context model can accept a large input while quality collapses in the middle of the context.

The practical defense is small tests: compare attention with shifted inputs, verify RoPE norm preservation, check ALiBi bias values by distance, and run retrieval probes at several positions in the context window.

### 8.5 Choosing a scheme

**Purpose.** Choosing a scheme focuses on tradeoffs among learned sinusoidal RoPE ALiBi and relative bias. This is the part of transformer math that tells attention where each token is and how far apart tokens are.

$$R_p^\top R_j=R_{j-p}.$$

**Operational definition.**

Position diagnostics test whether a model uses order correctly and remains reliable at target context lengths.

**Worked reading.**

A correct serving stack must assign decode position ids consistently with the KV cache and any RoPE or bias scheme.

| Scheme | Where position enters | Typical strength | Typical risk |
| --- | --- | --- | --- |
| sinusoidal | added to hidden states | fixed and simple | limited flexibility |
| learned absolute | added learned row | strong in-range fit | weak extrapolation |
| relative bias | attention score | offset-aware | implementation complexity |
| RoPE | rotates Q/K | relative dot products | scaling choices matter |
| ALiBi | attention score bias | simple extrapolation | distance prior may be too rigid |

Examples:

1. shift tests.
2. needle retrieval by position.
3. KV-cache position id checks.

Non-examples:

1. only testing short prompts.
2. assuming max context length implies uniform quality.

**Derivation habit.**

1. State whether the scheme is absolute or relative.
2. State whether the signal is added to hidden states, applied to Q/K, or added to scores.
3. Check shape compatibility with heads, sequence length, and cached decoding.
4. Test length behavior beyond the training context if the model will be served there.
5. Keep masks separate from position signals.

**Implementation lens.**

Position bugs are subtle because tensor shapes can still look correct. A decode loop can run while assigning the wrong position id to every generated token. A long-context model can accept a large input while quality collapses in the middle of the context.

The practical defense is small tests: compare attention with shifted inputs, verify RoPE norm preservation, check ALiBi bias values by distance, and run retrieval probes at several positions in the context window.

## 9. Common Mistakes

| # | Mistake | Why it is wrong | Fix |
| --- | --- | --- | --- |
| 1 | Assuming attention knows order automatically | Dot-product attention over a set is order-agnostic without position information. | Add or inject a positional signal. |
| 2 | Mixing absolute and relative claims | Absolute ids and relative offsets behave differently. | State where the position enters the formula. |
| 3 | Adding RoPE like a vector | RoPE rotates Q/K pairs rather than adding rows to hidden states. | Implement pairwise rotations. |
| 4 | Forgetting norm preservation | RoPE should preserve pair norms. | Unit-test norms after rotation. |
| 5 | Using learned rows beyond their range | A learned table has finite trained positions. | Resize/interpolate or choose an extrapolating scheme. |
| 6 | Applying ALiBi after softmax | Bias must affect logits before normalization. | Add bias to scores before softmax. |
| 7 | Ignoring decode position ids | KV cache requires consistent positions for new tokens. | Track prefix length and generated step. |
| 8 | Judging long context by max length only | Quality may degrade inside the window. | Evaluate retrieval by position and distance. |
| 9 | Confusing padding masks with position encodings | Masks control visibility; encodings provide order. | Use both when needed. |
| 10 | Treating extrapolation as guaranteed | All schemes can fail outside training distribution. | Test at target lengths. |

## 10. Exercises

1. (*) Compute a sinusoidal position row.
   - (a) State the scheme.
   - (b) Compute the numeric example.
   - (c) Explain the LLM consequence.

2. (*) Show token plus position addition.
   - (a) State the scheme.
   - (b) Compute the numeric example.
   - (c) Explain the LLM consequence.

3. (*) Build a relative distance matrix.
   - (a) State the scheme.
   - (b) Compute the numeric example.
   - (c) Explain the LLM consequence.

4. (**) Create a relative attention bias matrix.
   - (a) State the scheme.
   - (b) Compute the numeric example.
   - (c) Explain the LLM consequence.

5. (**) Apply a RoPE rotation and check norm preservation.
   - (a) State the scheme.
   - (b) Compute the numeric example.
   - (c) Explain the LLM consequence.

6. (**) Verify a RoPE relative-offset dot-product identity in two dimensions.
   - (a) State the scheme.
   - (b) Compute the numeric example.
   - (c) Explain the LLM consequence.

7. (**) Build an ALiBi matrix.
   - (a) State the scheme.
   - (b) Compute the numeric example.
   - (c) Explain the LLM consequence.

8. (***) Compute learned position table parameters.
   - (a) State the scheme.
   - (b) Compute the numeric example.
   - (c) Explain the LLM consequence.

9. (***) Check decode position ids for a KV cache.
   - (a) State the scheme.
   - (b) Compute the numeric example.
   - (c) Explain the LLM consequence.

10. (***) Design a long-context position diagnostic.
   - (a) State the scheme.
   - (b) Compute the numeric example.
   - (c) Explain the LLM consequence.

## 11. Why This Matters for AI

| Concept | AI impact |
| --- | --- |
| Sinusoidal encodings | Provide fixed absolute order features without learned position rows. |
| Learned position tables | Work well in-range but tie the model to trained maximum positions. |
| Relative biases | Let attention reason about pairwise distances. |
| RoPE | Supports relative offset behavior through rotations and is common in modern decoder LLMs. |
| ALiBi | Adds simple distance penalties that extrapolate without a learned position table. |
| Position ids | Matter for KV-cache decoding and long-context serving correctness. |
| Long-context diagnostics | Expose lost-in-the-middle, recency bias, and extrapolation failures. |
| Mask interaction | Ensures order signals do not override causal or padding visibility. |

## 12. Conceptual Bridge

The backward bridge is attention. Attention computes content-based interactions, but position mechanisms determine whether those interactions know sequence order and distance.

The forward bridge is language-model probability. In next-token prediction, position affects which prefix states are visible, how generated tokens receive ids, and whether the model can use long contexts reliably.

```text
+-------------+      +----------------+      +----------------------+
| attention   | ---> | position signal | ---> | ordered next-token    |
| content mix |      | absolute/rel    |      | prediction            |
+-------------+      +----------------+      +----------------------+
```

The practical habit is to test length behavior, not only maximum accepted length. Position encodings can be mathematically valid and still behave poorly outside the training regime.

## References

- Vaswani et al.. Attention Is All You Need. https://arxiv.org/abs/1706.03762
- Shaw, Uszkoreit, Vaswani. Self-Attention with Relative Position Representations. https://arxiv.org/abs/1803.02155
- Dai et al.. Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context. https://arxiv.org/abs/1901.02860
- Su et al.. RoFormer: Enhanced Transformer with Rotary Position Embedding. https://arxiv.org/abs/2104.09864
- Press et al.. Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation. https://arxiv.org/abs/2108.12409
