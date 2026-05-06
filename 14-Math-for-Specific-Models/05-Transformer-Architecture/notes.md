[<- RNN and LSTM Math](../04-RNN-and-LSTM-Math/notes.md) | [Home](../../README.md) | [Reinforcement Learning ->](../06-Reinforcement-Learning/notes.md)

---

# Transformer Architecture

The transformer is a stack of attention, feed-forward, residual, normalization, and position mechanisms. Its core advantage is that training can process all token positions in parallel while attention lets each token condition on other tokens.

## Overview

The central attention operation is:

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

This formula turns token representations into queries, keys, and values. Queries ask, keys match, values carry content. Multi-head attention repeats this operation in several subspaces. Residual blocks and normalization make the stack trainable. Positional information restores order. Masks define which tokens are allowed to interact.

## Prerequisites

- Matrix multiplication and softmax
- Embeddings and positional encodings
- Sequence probability and autoregressive masking
- RNN section for the recurrence-to-attention comparison

## Companion Notebooks

| Notebook | Purpose |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Demonstrates attention shapes, causal masks, multi-head splitting, MLP blocks, Pre-LN/Post-LN, positional signals, KV cache size, and diagnostics. |
| [exercises.ipynb](exercises.ipynb) | Ten practice problems for attention arithmetic, masks, shapes, parameter counts, and transformer debugging. |

## Learning Objectives

After this section, you should be able to:

- Compute scaled dot-product attention from Q, K, and V.
- Explain multi-head attention and head dimension arithmetic.
- Distinguish attention token mixing from position-wise MLP channel mixing.
- Explain residual streams, LayerNorm, Pre-LN, Post-LN, and RMSNorm.
- Compare encoder, decoder, encoder-decoder, encoder-only, and decoder-only architectures.
- Explain positional signals, attention masks, and KV cache memory.
- Estimate attention and MLP complexity.
- Build shape, mask, attention-entropy, and residual-norm diagnostics.

## Table of Contents

1. [Transformer Design Goal](#1-transformer-design-goal)
   - 1.1 [Sequence transduction](#11-sequence-transduction)
   - 1.2 [Parallel token processing](#12-parallel-token-processing)
   - 1.3 [Context mixing](#13-context-mixing)
   - 1.4 [Position information](#14-position-information)
   - 1.5 [Stacked blocks](#15-stacked-blocks)
2. [Scaled Dot-Product Attention](#2-scaled-dotproduct-attention)
   - 2.1 [Queries keys values](#21-queries-keys-values)
   - 2.2 [Score matrix](#22-score-matrix)
   - 2.3 [Masking](#23-masking)
   - 2.4 [Attention weights](#24-attention-weights)
   - 2.5 [Weighted values](#25-weighted-values)
3. [Multi-Head Attention](#3-multihead-attention)
   - 3.1 [Head splitting](#31-head-splitting)
   - 3.2 [Per-head dimension](#32-perhead-dimension)
   - 3.3 [Concatenation](#33-concatenation)
   - 3.4 [Specialization](#34-specialization)
   - 3.5 [GQA and MQA bridge](#35-gqa-and-mqa-bridge)
4. [Feed-Forward Network](#4-feedforward-network)
   - 4.1 [Position-wise MLP](#41-positionwise-mlp)
   - 4.2 [Expansion ratio](#42-expansion-ratio)
   - 4.3 [Activation](#43-activation)
   - 4.4 [Parameter count](#44-parameter-count)
   - 4.5 [Token independence](#45-token-independence)
5. [Residuals and Normalization](#5-residuals-and-normalization)
   - 5.1 [Residual stream](#51-residual-stream)
   - 5.2 [Layer normalization](#52-layer-normalization)
   - 5.3 [Pre-LN block](#53-preln-block)
   - 5.4 [Post-LN block](#54-postln-block)
   - 5.5 [RMSNorm](#55-rmsnorm)
6. [Encoder Decoder and Decoder-Only Forms](#6-encoder-decoder-and-decoderonly-forms)
   - 6.1 [Encoder self-attention](#61-encoder-selfattention)
   - 6.2 [Decoder causal self-attention](#62-decoder-causal-selfattention)
   - 6.3 [Cross-attention](#63-crossattention)
   - 6.4 [Encoder-only](#64-encoderonly)
   - 6.5 [Decoder-only](#65-decoderonly)
7. [Positional Information](#7-positional-information)
   - 7.1 [Absolute positions](#71-absolute-positions)
   - 7.2 [Sinusoidal positions](#72-sinusoidal-positions)
   - 7.3 [Relative bias](#73-relative-bias)
   - 7.4 [RoPE](#74-rope)
   - 7.5 [Length behavior](#75-length-behavior)
8. [Complexity and Memory](#8-complexity-and-memory)
   - 8.1 [Attention FLOPs](#81-attention-flops)
   - 8.2 [Attention score memory](#82-attention-score-memory)
   - 8.3 [MLP FLOPs](#83-mlp-flops)
   - 8.4 [KV cache](#84-kv-cache)
   - 8.5 [FlashAttention bridge](#85-flashattention-bridge)
9. [Training Signals](#9-training-signals)
   - 9.1 [Language-model loss](#91-languagemodel-loss)
   - 9.2 [Masked LM loss](#92-masked-lm-loss)
   - 9.3 [Teacher forcing](#93-teacher-forcing)
   - 9.4 [Attention masks](#94-attention-masks)
   - 9.5 [Weight tying](#95-weight-tying)
10. [Diagnostics](#10-diagnostics)
   - 10.1 [Shape checks](#101-shape-checks)
   - 10.2 [Mask tests](#102-mask-tests)
   - 10.3 [Attention entropy](#103-attention-entropy)
   - 10.4 [Residual norms](#104-residual-norms)
   - 10.5 [Ablations](#105-ablations)

---

## Block Shape Map

```text
hidden states:       H       shape (B, T, d_model)
queries:             Q       shape (B, heads, T, d_head)
keys:                K       shape (B, heads, T, d_head)
values:              V       shape (B, heads, T, d_head)
attention scores:    S       shape (B, heads, T, T)
attention output:    O       shape (B, T, d_model)
```

## 1. Transformer Design Goal

This part studies transformer design goal as transformer block math. Keep track of token axes, head axes, masks, residual updates, and where cross-token mixing happens.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Sequence transduction](#1-sequence-transduction) | map one sequence representation to another | $x_{1:T}\rightarrow y_{1:M}$ |
| [Parallel token processing](#1-parallel-token-processing) | avoid recurrence during training | $H^{(0)}\in\mathbb{R}^{B\times T\times d}$ |
| [Context mixing](#1-context-mixing) | let every token read other tokens through attention | $\mathrm{Attention}(Q,K,V)$ |
| [Position information](#1-position-information) | inject order because attention alone is permutation equivariant | $h_i=x_i+p_i$ |
| [Stacked blocks](#1-stacked-blocks) | repeat attention and MLP transformations | $H^{(\ell+1)}=\mathrm{Block}_\ell(H^{(\ell)})$ |

### 1.1 Sequence transduction

**Main idea.** Map one sequence representation to another.

Core relation:

$$x_{1:T}\rightarrow y_{1:M}$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 1.2 Parallel token processing

**Main idea.** Avoid recurrence during training.

Core relation:

$$H^{(0)}\in\mathbb{R}^{B\times T\times d}$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 1.3 Context mixing

**Main idea.** Let every token read other tokens through attention.

Core relation:

$$\mathrm{Attention}(Q,K,V)$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 1.4 Position information

**Main idea.** Inject order because attention alone is permutation equivariant.

Core relation:

$$h_i=x_i+p_i$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 1.5 Stacked blocks

**Main idea.** Repeat attention and mlp transformations.

Core relation:

$$H^{(\ell+1)}=\mathrm{Block}_\ell(H^{(\ell)})$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
## 2. Scaled Dot-Product Attention

This part studies scaled dot-product attention as transformer block math. Keep track of token axes, head axes, masks, residual updates, and where cross-token mixing happens.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Queries keys values](#2-queries-keys-values) | project hidden states into matching and content spaces | $Q=HW_Q,\ K=HW_K,\ V=HW_V$ |
| [Score matrix](#2-score-matrix) | compare every query with every key | $S=QK^\top/\sqrt{d_k}$ |
| [Masking](#2-masking) | block padding or future positions before softmax | $S_{ij}\leftarrow-\infty$ when masked |
| [Attention weights](#2-attention-weights) | softmax gives a distribution over source positions | $A=\mathrm{softmax}(S)$ |
| [Weighted values](#2-weighted-values) | mix value vectors by attention weights | $O=AV$ |

### 2.1 Queries keys values

**Main idea.** Project hidden states into matching and content spaces.

Core relation:

$$Q=HW_Q,\ K=HW_K,\ V=HW_V$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 2.2 Score matrix

**Main idea.** Compare every query with every key.

Core relation:

$$S=QK^\top/\sqrt{d_k}$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is the computation that lets each token ask which other tokens matter.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 2.3 Masking

**Main idea.** Block padding or future positions before softmax.

Core relation:

$$S_{ij}\leftarrow-\infty$ when masked$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 2.4 Attention weights

**Main idea.** Softmax gives a distribution over source positions.

Core relation:

$$A=\mathrm{softmax}(S)$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 2.5 Weighted values

**Main idea.** Mix value vectors by attention weights.

Core relation:

$$O=AV$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
## 3. Multi-Head Attention

This part studies multi-head attention as transformer block math. Keep track of token axes, head axes, masks, residual updates, and where cross-token mixing happens.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Head splitting](#3-head-splitting) | several smaller attention heads run in parallel | $h=1,\ldots,H$ |
| [Per-head dimension](#3-perhead-dimension) | model dimension is split across heads | $d_h=d_\mathrm{model}/H$ |
| [Concatenation](#3-concatenation) | head outputs are concatenated and projected | $\mathrm{MHA}(H)=\mathrm{Concat}(O_1,\ldots,O_H)W_O$ |
| [Specialization](#3-specialization) | different heads can learn different relation patterns | $A_h$ varies by head |
| [GQA and MQA bridge](#3-gqa-and-mqa-bridge) | serving variants share key-value heads | $H_{kv}\le H_q$ |

### 3.1 Head splitting

**Main idea.** Several smaller attention heads run in parallel.

Core relation:

$$h=1,\ldots,H$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 3.2 Per-head dimension

**Main idea.** Model dimension is split across heads.

Core relation:

$$d_h=d_\mathrm{model}/H$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 3.3 Concatenation

**Main idea.** Head outputs are concatenated and projected.

Core relation:

$$\mathrm{MHA}(H)=\mathrm{Concat}(O_1,\ldots,O_H)W_O$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 3.4 Specialization

**Main idea.** Different heads can learn different relation patterns.

Core relation:

$$A_h$ varies by head$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 3.5 GQA and MQA bridge

**Main idea.** Serving variants share key-value heads.

Core relation:

$$H_{kv}\le H_q$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
## 4. Feed-Forward Network

This part studies feed-forward network as transformer block math. Keep track of token axes, head axes, masks, residual updates, and where cross-token mixing happens.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Position-wise MLP](#4-positionwise-mlp) | apply the same MLP to each token independently | $\mathrm{FFN}(x)=W_2\phi(W_1x+b_1)+b_2$ |
| [Expansion ratio](#4-expansion-ratio) | hidden width is often larger than model width | $d_\mathrm{ff}\approx4d_\mathrm{model}$ |
| [Activation](#4-activation) | GELU or SwiGLU controls nonlinear feature mixing | $\phi$ |
| [Parameter count](#4-parameter-count) | MLP parameters often dominate each block | $2d_\mathrm{model}d_\mathrm{ff}$ |
| [Token independence](#4-token-independence) | cross-token mixing happens in attention, not the MLP | $x_t$ processed separately |

### 4.1 Position-wise MLP

**Main idea.** Apply the same mlp to each token independently.

Core relation:

$$\mathrm{FFN}(x)=W_2\phi(W_1x+b_1)+b_2$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 4.2 Expansion ratio

**Main idea.** Hidden width is often larger than model width.

Core relation:

$$d_\mathrm{ff}\approx4d_\mathrm{model}$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 4.3 Activation

**Main idea.** Gelu or swiglu controls nonlinear feature mixing.

Core relation:

$$\phi$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 4.4 Parameter count

**Main idea.** Mlp parameters often dominate each block.

Core relation:

$$2d_\mathrm{model}d_\mathrm{ff}$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 4.5 Token independence

**Main idea.** Cross-token mixing happens in attention, not the mlp.

Core relation:

$$x_t$ processed separately$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
## 5. Residuals and Normalization

This part studies residuals and normalization as transformer block math. Keep track of token axes, head axes, masks, residual updates, and where cross-token mixing happens.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Residual stream](#5-residual-stream) | blocks add updates to a persistent representation | $x\leftarrow x+F(x)$ |
| [Layer normalization](#5-layer-normalization) | normalize features within each token | $\hat x=(x-\mu)/\sqrt{\sigma^2+\epsilon}$ |
| [Pre-LN block](#5-preln-block) | normalize before sublayer for more stable deep training | $x\leftarrow x+F(\mathrm{LN}(x))$ |
| [Post-LN block](#5-postln-block) | original transformer normalized after residual addition | $x\leftarrow\mathrm{LN}(x+F(x))$ |
| [RMSNorm](#5-rmsnorm) | normalize by root mean square without mean subtraction | $x/\sqrt{\mathrm{mean}(x^2)+\epsilon}$ |

### 5.1 Residual stream

**Main idea.** Blocks add updates to a persistent representation.

Core relation:

$$x\leftarrow x+F(x)$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 5.2 Layer normalization

**Main idea.** Normalize features within each token.

Core relation:

$$\hat x=(x-\mu)/\sqrt{\sigma^2+\epsilon}$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 5.3 Pre-LN block

**Main idea.** Normalize before sublayer for more stable deep training.

Core relation:

$$x\leftarrow x+F(\mathrm{LN}(x))$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** Pre-LN is a key reason very deep transformer stacks are easier to optimize.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 5.4 Post-LN block

**Main idea.** Original transformer normalized after residual addition.

Core relation:

$$x\leftarrow\mathrm{LN}(x+F(x))$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 5.5 RMSNorm

**Main idea.** Normalize by root mean square without mean subtraction.

Core relation:

$$x/\sqrt{\mathrm{mean}(x^2)+\epsilon}$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
## 6. Encoder Decoder and Decoder-Only Forms

This part studies encoder decoder and decoder-only forms as transformer block math. Keep track of token axes, head axes, masks, residual updates, and where cross-token mixing happens.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Encoder self-attention](#6-encoder-selfattention) | source tokens attend bidirectionally | $A_{ij}$ allowed for all source positions |
| [Decoder causal self-attention](#6-decoder-causal-selfattention) | target tokens cannot see future target tokens | $j\le i$ |
| [Cross-attention](#6-crossattention) | decoder queries attend to encoder keys and values | $Q=H_\mathrm{dec}W_Q,\ K,V=H_\mathrm{enc}W_{K,V}$ |
| [Encoder-only](#6-encoderonly) | BERT-style models produce contextual representations | $p(\mathrm{masked}\ token\mid x)$ |
| [Decoder-only](#6-decoderonly) | GPT-style models predict next tokens autoregressively | $p(x_t\mid x_{<t})$ |

### 6.1 Encoder self-attention

**Main idea.** Source tokens attend bidirectionally.

Core relation:

$$A_{ij}$ allowed for all source positions$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 6.2 Decoder causal self-attention

**Main idea.** Target tokens cannot see future target tokens.

Core relation:

$$j\le i$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 6.3 Cross-attention

**Main idea.** Decoder queries attend to encoder keys and values.

Core relation:

$$Q=H_\mathrm{dec}W_Q,\ K,V=H_\mathrm{enc}W_{K,V}$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 6.4 Encoder-only

**Main idea.** Bert-style models produce contextual representations.

Core relation:

$$p(\mathrm{masked}\ token\mid x)$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 6.5 Decoder-only

**Main idea.** Gpt-style models predict next tokens autoregressively.

Core relation:

$$p(x_t\mid x_{<t})$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is the architecture behind most modern autoregressive LLMs.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
## 7. Positional Information

This part studies positional information as transformer block math. Keep track of token axes, head axes, masks, residual updates, and where cross-token mixing happens.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Absolute positions](#7-absolute-positions) | add learned or fixed position vectors | $h_i=x_i+p_i$ |
| [Sinusoidal positions](#7-sinusoidal-positions) | fixed frequencies encode index information | $\sin(i/\omega_k),\cos(i/\omega_k)$ |
| [Relative bias](#7-relative-bias) | attention scores depend on distance | $S_{ij}\leftarrow S_{ij}+b_{i-j}$ |
| [RoPE](#7-rope) | rotate query and key pairs by position-dependent angles | $q_i^\top k_j$ depends on $i-j$ |
| [Length behavior](#7-length-behavior) | position design affects long-context extrapolation | $T_\mathrm{test}>T_\mathrm{train}$ |

### 7.1 Absolute positions

**Main idea.** Add learned or fixed position vectors.

Core relation:

$$h_i=x_i+p_i$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 7.2 Sinusoidal positions

**Main idea.** Fixed frequencies encode index information.

Core relation:

$$\sin(i/\omega_k),\cos(i/\omega_k)$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 7.3 Relative bias

**Main idea.** Attention scores depend on distance.

Core relation:

$$S_{ij}\leftarrow S_{ij}+b_{i-j}$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 7.4 RoPE

**Main idea.** Rotate query and key pairs by position-dependent angles.

Core relation:

$$q_i^\top k_j$ depends on $i-j$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 7.5 Length behavior

**Main idea.** Position design affects long-context extrapolation.

Core relation:

$$T_\mathrm{test}>T_\mathrm{train}$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
## 8. Complexity and Memory

This part studies complexity and memory as transformer block math. Keep track of token axes, head axes, masks, residual updates, and where cross-token mixing happens.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Attention FLOPs](#8-attention-flops) | dense attention is quadratic in sequence length | $O(BHT^2d_h)$ |
| [Attention score memory](#8-attention-score-memory) | naive attention materializes T by T scores | $O(BHT^2)$ |
| [MLP FLOPs](#8-mlp-flops) | MLP cost is linear in sequence length but large in width | $O(BTd_\mathrm{model}d_\mathrm{ff})$ |
| [KV cache](#8-kv-cache) | decoder inference stores keys and values per layer | $M_\mathrm{KV}=2BLTH_{kv}d_hb$ |
| [FlashAttention bridge](#8-flashattention-bridge) | IO-aware attention computes exact attention without storing all scores | $\mathrm{softmax}(QK^\top)V$ tiled |

### 8.1 Attention FLOPs

**Main idea.** Dense attention is quadratic in sequence length.

Core relation:

$$O(BHT^2d_h)$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 8.2 Attention score memory

**Main idea.** Naive attention materializes t by t scores.

Core relation:

$$O(BHT^2)$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 8.3 MLP FLOPs

**Main idea.** Mlp cost is linear in sequence length but large in width.

Core relation:

$$O(BTd_\mathrm{model}d_\mathrm{ff})$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 8.4 KV cache

**Main idea.** Decoder inference stores keys and values per layer.

Core relation:

$$M_\mathrm{KV}=2BLTH_{kv}d_hb$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is the central memory object for fast decoder inference.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 8.5 FlashAttention bridge

**Main idea.** Io-aware attention computes exact attention without storing all scores.

Core relation:

$$\mathrm{softmax}(QK^\top)V$ tiled$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
## 9. Training Signals

This part studies training signals as transformer block math. Keep track of token axes, head axes, masks, residual updates, and where cross-token mixing happens.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Language-model loss](#9-languagemodel-loss) | decoder-only models train by next-token prediction | $L=-\sum_t\log p_\theta(x_t\mid x_{<t})$ |
| [Masked LM loss](#9-masked-lm-loss) | encoder-only models predict hidden masked tokens | $L=-\sum_{i\in M}\log p_\theta(x_i\mid x_{\setminus M})$ |
| [Teacher forcing](#9-teacher-forcing) | decoder training conditions on gold previous tokens | $y_{<t}^\star$ |
| [Attention masks](#9-attention-masks) | loss and attention masks must agree with the task | $M_\mathrm{attn},M_\mathrm{loss}$ |
| [Weight tying](#9-weight-tying) | input embedding and output projection can share weights | $W_\mathrm{out}=E^\top$ |

### 9.1 Language-model loss

**Main idea.** Decoder-only models train by next-token prediction.

Core relation:

$$L=-\sum_t\log p_\theta(x_t\mid x_{<t})$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 9.2 Masked LM loss

**Main idea.** Encoder-only models predict hidden masked tokens.

Core relation:

$$L=-\sum_{i\in M}\log p_\theta(x_i\mid x_{\setminus M})$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 9.3 Teacher forcing

**Main idea.** Decoder training conditions on gold previous tokens.

Core relation:

$$y_{<t}^\star$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 9.4 Attention masks

**Main idea.** Loss and attention masks must agree with the task.

Core relation:

$$M_\mathrm{attn},M_\mathrm{loss}$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 9.5 Weight tying

**Main idea.** Input embedding and output projection can share weights.

Core relation:

$$W_\mathrm{out}=E^\top$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
## 10. Diagnostics

This part studies diagnostics as transformer block math. Keep track of token axes, head axes, masks, residual updates, and where cross-token mixing happens.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Shape checks](#10-shape-checks) | track batch, time, heads, head dimension, and model dimension | $(B,H,T,d_h)$ |
| [Mask tests](#10-mask-tests) | future leakage breaks autoregressive training | $A_{ij}=0$ for $j>i$ |
| [Attention entropy](#10-attention-entropy) | overly sharp or diffuse heads can indicate issues | $H(A_i)$ |
| [Residual norms](#10-residual-norms) | monitor update size relative to residual stream | $\Vert F(x)\Vert/\Vert x\Vert$ |
| [Ablations](#10-ablations) | compare head count, MLP width, norm placement, and position method | $\Delta L,\Delta T$ |

### 10.1 Shape checks

**Main idea.** Track batch, time, heads, head dimension, and model dimension.

Core relation:

$$(B,H,T,d_h)$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 10.2 Mask tests

**Main idea.** Future leakage breaks autoregressive training.

Core relation:

$$A_{ij}=0$ for $j>i$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** A single mask bug can make a model appear to learn while leaking future answers.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 10.3 Attention entropy

**Main idea.** Overly sharp or diffuse heads can indicate issues.

Core relation:

$$H(A_i)$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 10.4 Residual norms

**Main idea.** Monitor update size relative to residual stream.

Core relation:

$$\Vert F(x)\Vert/\Vert x\Vert$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 10.5 Ablations

**Main idea.** Compare head count, mlp width, norm placement, and position method.

Core relation:

$$\Delta L,\Delta T$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.

---

## Practice Exercises

1. Compute scaled dot-product attention for tiny matrices.
2. Build a causal mask and apply it to attention scores.
3. Split model dimension into heads.
4. Count parameters in Q, K, V, and output projections.
5. Count MLP parameters and compare to attention parameters.
6. Compute LayerNorm and RMSNorm for one vector.
7. Compare Pre-LN and Post-LN block equations.
8. Compute KV cache memory for a decoder.
9. Identify encoder-only, decoder-only, and encoder-decoder masks.
10. Write a transformer debugging checklist.

## Why This Matters for AI

The transformer is the architectural base of modern LLMs and many multimodal models. Understanding it at the level of shapes, masks, residuals, and costs prevents vague explanations and catches real implementation mistakes.

## Bridge to Reinforcement Learning

The next model-specific section studies reinforcement learning. In modern LLM systems, transformer policies are often optimized further with preference or reward-based objectives, so architecture and optimization meet there.

## References

- Ashish Vaswani et al., "Attention Is All You Need", 2017: https://arxiv.org/abs/1706.03762
- Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey Hinton, "Layer Normalization", 2016: https://arxiv.org/abs/1607.06450
- Ruibin Xiong et al., "On Layer Normalization in the Transformer Architecture", 2020: https://arxiv.org/abs/2002.04745
- Biao Zhang and Rico Sennrich, "Root Mean Square Layer Normalization", 2019: https://arxiv.org/abs/1910.07467
