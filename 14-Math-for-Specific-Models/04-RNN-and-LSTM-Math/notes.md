[<- Probabilistic Models](../03-Probabilistic-Models/notes.md) | [Home](../../README.md) | [Transformer Architecture ->](../05-Transformer-Architecture/notes.md)

---

# RNN and LSTM Math

Recurrent neural networks model sequences by reusing the same transition at every time step. Their power is memory; their difficulty is training through long chains of time.

## Overview

The basic recurrent equation is:

$$
h_t=f_\theta(h_{t-1},x_t).
$$

This state update lets a model process variable-length sequences. But training the model means backpropagating through the unrolled computation graph. Long products of recurrent Jacobians can vanish or explode. LSTM and GRU cells add gates and additive memory paths to make sequence learning more stable.

## Prerequisites

- Matrix multiplication and nonlinear activations
- Chain rule and backpropagation
- Cross-entropy for sequence prediction
- Basic transformer attention intuition for the bridge section

## Companion Notebooks

| Notebook | Purpose |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Demonstrates vanilla recurrence, BPTT gradient products, clipping, LSTM/GRU gates, masks, teacher forcing, attention weights, and diagnostics. |
| [exercises.ipynb](exercises.ipynb) | Ten practice problems for recurrent equations, BPTT, gates, masking, and sequence-task shapes. |

## Learning Objectives

After this section, you should be able to:

- Write vanilla RNN, LSTM, and GRU update equations.
- Explain recurrence as a parameter-shared deep network along time.
- Derive why gradients vanish or explode through repeated Jacobian products.
- Apply gradient clipping and truncated BPTT.
- Interpret LSTM forget/input/output gates and GRU update/reset gates.
- Distinguish many-to-one, many-to-many, seq2seq, and bidirectional tasks.
- Explain why attention was introduced to reduce the fixed-context bottleneck.
- Build diagnostics for masks, gate saturation, length generalization, and gradient norms.

## Table of Contents

1. [Sequence Modeling View](#1-sequence-modeling-view)
   - 1.1 [Sequential data](#11-sequential-data)
   - 1.2 [Hidden state](#12-hidden-state)
   - 1.3 [Output distribution](#13-output-distribution)
   - 1.4 [Autoregressive generation](#14-autoregressive-generation)
   - 1.5 [RNN versus transformer](#15-rnn-versus-transformer)
2. [Vanilla RNN Equations](#2-vanilla-rnn-equations)
   - 2.1 [State update](#21-state-update)
   - 2.2 [Output head](#22-output-head)
   - 2.3 [Parameter sharing](#23-parameter-sharing)
   - 2.4 [Unrolled graph](#24-unrolled-graph)
   - 2.5 [Initial state](#25-initial-state)
3. [Backpropagation Through Time](#3-backpropagation-through-time)
   - 3.1 [Sequence loss](#31-sequence-loss)
   - 3.2 [Temporal chain rule](#32-temporal-chain-rule)
   - 3.3 [Weight gradient](#33-weight-gradient)
   - 3.4 [Truncated BPTT](#34-truncated-bptt)
   - 3.5 [State detachment](#35-state-detachment)
4. [Vanishing and Exploding Gradients](#4-vanishing-and-exploding-gradients)
   - 4.1 [Jacobian product](#41-jacobian-product)
   - 4.2 [Spectral radius](#42-spectral-radius)
   - 4.3 [Vanishing](#43-vanishing)
   - 4.4 [Exploding](#44-exploding)
   - 4.5 [Gradient clipping](#45-gradient-clipping)
5. [LSTM Cell](#5-lstm-cell)
   - 5.1 [Forget gate](#51-forget-gate)
   - 5.2 [Input gate](#52-input-gate)
   - 5.3 [Candidate memory](#53-candidate-memory)
   - 5.4 [Cell update](#54-cell-update)
   - 5.5 [Output gate](#55-output-gate)
6. [GRU Cell](#6-gru-cell)
   - 6.1 [Update gate](#61-update-gate)
   - 6.2 [Reset gate](#62-reset-gate)
   - 6.3 [Candidate hidden](#63-candidate-hidden)
   - 6.4 [Hidden update](#64-hidden-update)
   - 6.5 [GRU versus LSTM](#65-gru-versus-lstm)
7. [Sequence Tasks](#7-sequence-tasks)
   - 7.1 [Many-to-one](#71-manytoone)
   - 7.2 [Many-to-many](#72-manytomany)
   - 7.3 [Seq2seq](#73-seq2seq)
   - 7.4 [Bidirectional RNN](#74-bidirectional-rnn)
   - 7.5 [Teacher forcing](#75-teacher-forcing)
8. [Attention Bridge](#8-attention-bridge)
   - 8.1 [Fixed context bottleneck](#81-fixed-context-bottleneck)
   - 8.2 [Alignment scores](#82-alignment-scores)
   - 8.3 [Attention weights](#83-attention-weights)
   - 8.4 [Context vector](#84-context-vector)
   - 8.5 [Transformer bridge](#85-transformer-bridge)
9. [Training Practice](#9-training-practice)
   - 9.1 [Padding masks](#91-padding-masks)
   - 9.2 [Packed sequences](#92-packed-sequences)
   - 9.3 [Stateful streaming](#93-stateful-streaming)
   - 9.4 [Initialization](#94-initialization)
   - 9.5 [Regularization](#95-regularization)
10. [Diagnostics](#10-diagnostics)
   - 10.1 [Shape checks](#101-shape-checks)
   - 10.2 [Gradient norms](#102-gradient-norms)
   - 10.3 [Gate statistics](#103-gate-statistics)
   - 10.4 [Length tests](#104-length-tests)
   - 10.5 [Ablations](#105-ablations)

---

## Shape Map

```text
input sequence:       X      shape (B, T, d_x)
hidden state:         h_t    shape (B, d_h)
hidden sequence:      H      shape (B, T, d_h)
logits per token:     O      shape (B, T, |V|)
padding mask:         M      shape (B, T)
```

## 1. Sequence Modeling View

This part studies sequence modeling view through the lens of sequence learning. The central question is how information and gradients move through time.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Sequential data](#1-sequential-data) | observations arrive with order and history | $x_{1:T}=(x_1,\ldots,x_T)$ |
| [Hidden state](#1-hidden-state) | the model carries a summary of the past | $h_t=f_\theta(h_{t-1},x_t)$ |
| [Output distribution](#1-output-distribution) | each state can produce a prediction | $p(y_t\mid x_{\le t})=g_\theta(h_t)$ |
| [Autoregressive generation](#1-autoregressive-generation) | the previous output can become the next input | $p(x_{1:T})=\prod_t p(x_t\mid x_{<t})$ |
| [RNN versus transformer](#1-rnn-versus-transformer) | RNNs compress history into a state, transformers keep token states visible | $h_t$ versus $H_{1:t}$ |

### 1.1 Sequential data

**Main idea.** Observations arrive with order and history.

Core relation:

$$x_{1:T}=(x_1,\ldots,x_T)$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 1.2 Hidden state

**Main idea.** The model carries a summary of the past.

Core relation:

$$h_t=f_\theta(h_{t-1},x_t)$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 1.3 Output distribution

**Main idea.** Each state can produce a prediction.

Core relation:

$$p(y_t\mid x_{\le t})=g_\theta(h_t)$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 1.4 Autoregressive generation

**Main idea.** The previous output can become the next input.

Core relation:

$$p(x_{1:T})=\prod_t p(x_t\mid x_{<t})$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 1.5 RNN versus transformer

**Main idea.** Rnns compress history into a state, transformers keep token states visible.

Core relation:

$$h_t$ versus $H_{1:t}$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
## 2. Vanilla RNN Equations

This part studies vanilla rnn equations through the lens of sequence learning. The central question is how information and gradients move through time.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [State update](#2-state-update) | combine current input and previous hidden state | $h_t=\phi(W_{xh}x_t+W_{hh}h_{t-1}+b_h)$ |
| [Output head](#2-output-head) | map hidden state to logits or regression output | $o_t=W_{hy}h_t+b_y$ |
| [Parameter sharing](#2-parameter-sharing) | the same weights are reused at every time step | $\theta_t=\theta$ |
| [Unrolled graph](#2-unrolled-graph) | an RNN is a deep network along time | $h_1\rightarrow h_2\rightarrow\cdots\rightarrow h_T$ |
| [Initial state](#2-initial-state) | start from zeros or a learned state | $h_0=0$ or trainable |

### 2.1 State update

**Main idea.** Combine current input and previous hidden state.

Core relation:

$$h_t=\phi(W_{xh}x_t+W_{hh}h_{t-1}+b_h)$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 2.2 Output head

**Main idea.** Map hidden state to logits or regression output.

Core relation:

$$o_t=W_{hy}h_t+b_y$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 2.3 Parameter sharing

**Main idea.** The same weights are reused at every time step.

Core relation:

$$\theta_t=\theta$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 2.4 Unrolled graph

**Main idea.** An rnn is a deep network along time.

Core relation:

$$h_1\rightarrow h_2\rightarrow\cdots\rightarrow h_T$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 2.5 Initial state

**Main idea.** Start from zeros or a learned state.

Core relation:

$$h_0=0$ or trainable$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
## 3. Backpropagation Through Time

This part studies backpropagation through time through the lens of sequence learning. The central question is how information and gradients move through time.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Sequence loss](#3-sequence-loss) | sum or average losses over time | $L=\sum_{t=1}^{T}\ell_t$ |
| [Temporal chain rule](#3-temporal-chain-rule) | future losses depend on earlier states through recurrence | $\partial L/\partial h_t=\partial \ell_t/\partial h_t+(\partial h_{t+1}/\partial h_t)^\top\partial L/\partial h_{t+1}$ |
| [Weight gradient](#3-weight-gradient) | shared weights collect gradient from every time step | $\partial L/\partial W=\sum_t \partial L_t/\partial W$ |
| [Truncated BPTT](#3-truncated-bptt) | backpropagate through a limited window for efficiency | $t-k,\ldots,t$ |
| [State detachment](#3-state-detachment) | detach hidden state between chunks to control graph length | $h_t\leftarrow\mathrm{stopgrad}(h_t)$ |

### 3.1 Sequence loss

**Main idea.** Sum or average losses over time.

Core relation:

$$L=\sum_{t=1}^{T}\ell_t$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 3.2 Temporal chain rule

**Main idea.** Future losses depend on earlier states through recurrence.

Core relation:

$$\partial L/\partial h_t=\partial \ell_t/\partial h_t+(\partial h_{t+1}/\partial h_t)^\top\partial L/\partial h_{t+1}$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is why RNN training is training a very deep network along time.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 3.3 Weight gradient

**Main idea.** Shared weights collect gradient from every time step.

Core relation:

$$\partial L/\partial W=\sum_t \partial L_t/\partial W$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 3.4 Truncated BPTT

**Main idea.** Backpropagate through a limited window for efficiency.

Core relation:

$$t-k,\ldots,t$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 3.5 State detachment

**Main idea.** Detach hidden state between chunks to control graph length.

Core relation:

$$h_t\leftarrow\mathrm{stopgrad}(h_t)$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
## 4. Vanishing and Exploding Gradients

This part studies vanishing and exploding gradients through the lens of sequence learning. The central question is how information and gradients move through time.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Jacobian product](#4-jacobian-product) | long-range gradients multiply many recurrent Jacobians | $\prod_{i=s+1}^{t}\partial h_i/\partial h_{i-1}$ |
| [Spectral radius](#4-spectral-radius) | gradient scale depends on recurrent dynamics | $\rho(W_{hh})$ |
| [Vanishing](#4-vanishing) | singular values below one shrink long-range gradients | $\Vert g_s\Vert\rightarrow 0$ |
| [Exploding](#4-exploding) | singular values above one can blow up gradients | $\Vert g_s\Vert\rightarrow\infty$ |
| [Gradient clipping](#4-gradient-clipping) | cap gradient norm before optimizer update | $g\leftarrow g\min(1,c/\Vert g\Vert)$ |

### 4.1 Jacobian product

**Main idea.** Long-range gradients multiply many recurrent jacobians.

Core relation:

$$\prod_{i=s+1}^{t}\partial h_i/\partial h_{i-1}$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 4.2 Spectral radius

**Main idea.** Gradient scale depends on recurrent dynamics.

Core relation:

$$\rho(W_{hh})$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 4.3 Vanishing

**Main idea.** Singular values below one shrink long-range gradients.

Core relation:

$$\Vert g_s\Vert\rightarrow 0$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 4.4 Exploding

**Main idea.** Singular values above one can blow up gradients.

Core relation:

$$\Vert g_s\Vert\rightarrow\infty$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 4.5 Gradient clipping

**Main idea.** Cap gradient norm before optimizer update.

Core relation:

$$g\leftarrow g\min(1,c/\Vert g\Vert)$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** Clipping became a standard tool because recurrent gradient products can explode suddenly.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
## 5. LSTM Cell

This part studies lstm cell through the lens of sequence learning. The central question is how information and gradients move through time.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Forget gate](#5-forget-gate) | decide what memory to keep | $f_t=\sigma(W_f[x_t,h_{t-1}]+b_f)$ |
| [Input gate](#5-input-gate) | decide what new content to write | $i_t=\sigma(W_i[x_t,h_{t-1}]+b_i)$ |
| [Candidate memory](#5-candidate-memory) | propose new cell content | $\tilde c_t=\tanh(W_c[x_t,h_{t-1}]+b_c)$ |
| [Cell update](#5-cell-update) | additive memory path improves gradient flow | $c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t$ |
| [Output gate](#5-output-gate) | expose part of cell memory as hidden state | $h_t=o_t\odot\tanh(c_t)$ |

### 5.1 Forget gate

**Main idea.** Decide what memory to keep.

Core relation:

$$f_t=\sigma(W_f[x_t,h_{t-1}]+b_f)$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 5.2 Input gate

**Main idea.** Decide what new content to write.

Core relation:

$$i_t=\sigma(W_i[x_t,h_{t-1}]+b_i)$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 5.3 Candidate memory

**Main idea.** Propose new cell content.

Core relation:

$$\tilde c_t=\tanh(W_c[x_t,h_{t-1}]+b_c)$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 5.4 Cell update

**Main idea.** Additive memory path improves gradient flow.

Core relation:

$$c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** The additive cell path is the mathematical reason LSTMs can carry information longer than a plain tanh RNN.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 5.5 Output gate

**Main idea.** Expose part of cell memory as hidden state.

Core relation:

$$h_t=o_t\odot\tanh(c_t)$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
## 6. GRU Cell

This part studies gru cell through the lens of sequence learning. The central question is how information and gradients move through time.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Update gate](#6-update-gate) | interpolate old and candidate states | $z_t=\sigma(W_z[x_t,h_{t-1}])$ |
| [Reset gate](#6-reset-gate) | control how much past enters the candidate | $r_t=\sigma(W_r[x_t,h_{t-1}])$ |
| [Candidate hidden](#6-candidate-hidden) | build proposed new hidden state | $\tilde h_t=\tanh(W_h[x_t,r_t\odot h_{t-1}])$ |
| [Hidden update](#6-hidden-update) | blend old state and candidate | $h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t$ |
| [GRU versus LSTM](#6-gru-versus-lstm) | GRU is smaller; LSTM has separate cell and hidden state | $h_t$ only versus $(c_t,h_t)$ |

### 6.1 Update gate

**Main idea.** Interpolate old and candidate states.

Core relation:

$$z_t=\sigma(W_z[x_t,h_{t-1}])$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 6.2 Reset gate

**Main idea.** Control how much past enters the candidate.

Core relation:

$$r_t=\sigma(W_r[x_t,h_{t-1}])$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 6.3 Candidate hidden

**Main idea.** Build proposed new hidden state.

Core relation:

$$\tilde h_t=\tanh(W_h[x_t,r_t\odot h_{t-1}])$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 6.4 Hidden update

**Main idea.** Blend old state and candidate.

Core relation:

$$h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 6.5 GRU versus LSTM

**Main idea.** Gru is smaller; lstm has separate cell and hidden state.

Core relation:

$$h_t$ only versus $(c_t,h_t)$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
## 7. Sequence Tasks

This part studies sequence tasks through the lens of sequence learning. The central question is how information and gradients move through time.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Many-to-one](#7-manytoone) | classify a whole sequence from final or pooled state | $\hat y=g(h_T)$ |
| [Many-to-many](#7-manytomany) | predict at every time step | $\hat y_t=g(h_t)$ |
| [Seq2seq](#7-seq2seq) | encode one sequence and decode another | $p(y_{1:M}\mid x_{1:T})=\prod_jp(y_j\mid y_{<j},c)$ |
| [Bidirectional RNN](#7-bidirectional-rnn) | use past and future context for non-causal tasks | $h_t=[\overrightarrow h_t;\overleftarrow h_t]$ |
| [Teacher forcing](#7-teacher-forcing) | decoder conditions on gold previous outputs during training | $p(y_t\mid y_{<t}^\star,c)$ |

### 7.1 Many-to-one

**Main idea.** Classify a whole sequence from final or pooled state.

Core relation:

$$\hat y=g(h_T)$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 7.2 Many-to-many

**Main idea.** Predict at every time step.

Core relation:

$$\hat y_t=g(h_t)$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 7.3 Seq2seq

**Main idea.** Encode one sequence and decode another.

Core relation:

$$p(y_{1:M}\mid x_{1:T})=\prod_jp(y_j\mid y_{<j},c)$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 7.4 Bidirectional RNN

**Main idea.** Use past and future context for non-causal tasks.

Core relation:

$$h_t=[\overrightarrow h_t;\overleftarrow h_t]$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 7.5 Teacher forcing

**Main idea.** Decoder conditions on gold previous outputs during training.

Core relation:

$$p(y_t\mid y_{<t}^\star,c)$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
## 8. Attention Bridge

This part studies attention bridge through the lens of sequence learning. The central question is how information and gradients move through time.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Fixed context bottleneck](#8-fixed-context-bottleneck) | a single encoder vector struggles with long inputs | $c=h_T$ |
| [Alignment scores](#8-alignment-scores) | decoder state scores every encoder state | $e_{tj}=a(s_{t-1},h_j)$ |
| [Attention weights](#8-attention-weights) | softmax turns scores into a distribution over positions | $\alpha_{tj}=\mathrm{softmax}_j(e_{tj})$ |
| [Context vector](#8-context-vector) | weighted sum exposes relevant encoder states | $c_t=\sum_j\alpha_{tj}h_j$ |
| [Transformer bridge](#8-transformer-bridge) | self-attention removes recurrence and exposes all token states directly | $\mathrm{Attention}(Q,K,V)$ |

### 8.1 Fixed context bottleneck

**Main idea.** A single encoder vector struggles with long inputs.

Core relation:

$$c=h_T$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 8.2 Alignment scores

**Main idea.** Decoder state scores every encoder state.

Core relation:

$$e_{tj}=a(s_{t-1},h_j)$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 8.3 Attention weights

**Main idea.** Softmax turns scores into a distribution over positions.

Core relation:

$$\alpha_{tj}=\mathrm{softmax}_j(e_{tj})$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is the historical bridge from seq2seq RNNs to transformer attention.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 8.4 Context vector

**Main idea.** Weighted sum exposes relevant encoder states.

Core relation:

$$c_t=\sum_j\alpha_{tj}h_j$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 8.5 Transformer bridge

**Main idea.** Self-attention removes recurrence and exposes all token states directly.

Core relation:

$$\mathrm{Attention}(Q,K,V)$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
## 9. Training Practice

This part studies training practice through the lens of sequence learning. The central question is how information and gradients move through time.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Padding masks](#9-padding-masks) | sequence batches have variable lengths | $m_t\in\{0,1\}$ |
| [Packed sequences](#9-packed-sequences) | avoid computing loss on padding | $L=\sum_tm_t\ell_t/\sum_tm_t$ |
| [Stateful streaming](#9-stateful-streaming) | carry state across chunks for long streams | $h_\mathrm{next}=h_T$ |
| [Initialization](#9-initialization) | orthogonal recurrent matrices can stabilize early dynamics | $W_{hh}^\top W_{hh}=I$ |
| [Regularization](#9-regularization) | dropout, weight decay, and clipping fight overfit and instability | $L+\lambda\Vert\theta\Vert^2$ |

### 9.1 Padding masks

**Main idea.** Sequence batches have variable lengths.

Core relation:

$$m_t\in\{0,1\}$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 9.2 Packed sequences

**Main idea.** Avoid computing loss on padding.

Core relation:

$$L=\sum_tm_t\ell_t/\sum_tm_t$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 9.3 Stateful streaming

**Main idea.** Carry state across chunks for long streams.

Core relation:

$$h_\mathrm{next}=h_T$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 9.4 Initialization

**Main idea.** Orthogonal recurrent matrices can stabilize early dynamics.

Core relation:

$$W_{hh}^\top W_{hh}=I$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 9.5 Regularization

**Main idea.** Dropout, weight decay, and clipping fight overfit and instability.

Core relation:

$$L+\lambda\Vert\theta\Vert^2$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
## 10. Diagnostics

This part studies diagnostics through the lens of sequence learning. The central question is how information and gradients move through time.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Shape checks](#10-shape-checks) | inputs, hidden states, gates, and outputs have distinct axes | $(B,T,d)$ and $(B,h)$ |
| [Gradient norms](#10-gradient-norms) | track exploding or vanishing gradients through time | $\Vert g_t\Vert$ |
| [Gate statistics](#10-gate-statistics) | saturated gates indicate stuck memory behavior | $f_t,i_t,z_t$ near 0 or 1 |
| [Length tests](#10-length-tests) | evaluate short and long sequences separately | $S(T)$ |
| [Ablations](#10-ablations) | compare vanilla RNN, GRU, LSTM, and attention bridge | $\Delta L,\Delta S$ |

### 10.1 Shape checks

**Main idea.** Inputs, hidden states, gates, and outputs have distinct axes.

Core relation:

$$(B,T,d)$ and $(B,h)$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 10.2 Gradient norms

**Main idea.** Track exploding or vanishing gradients through time.

Core relation:

$$\Vert g_t\Vert$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 10.3 Gate statistics

**Main idea.** Saturated gates indicate stuck memory behavior.

Core relation:

$$f_t,i_t,z_t$ near 0 or 1$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** Gate histograms quickly reveal whether an LSTM or GRU is actually using its memory.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 10.4 Length tests

**Main idea.** Evaluate short and long sequences separately.

Core relation:

$$S(T)$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 10.5 Ablations

**Main idea.** Compare vanilla rnn, gru, lstm, and attention bridge.

Core relation:

$$\Delta L,\Delta S$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.

---

## Practice Exercises

1. Compute one vanilla RNN hidden update.
2. Compute a sequence log probability from conditional probabilities.
3. Show a scalar gradient product that vanishes or explodes.
4. Clip a gradient vector by norm.
5. Compute one LSTM cell update.
6. Compute one GRU hidden update.
7. Apply a padding mask to sequence losses.
8. Identify shapes for many-to-one and many-to-many tasks.
9. Compute attention weights and context over encoder states.
10. Write an RNN debugging checklist.

## Why This Matters for AI

Transformers dominate current LLMs, but RNNs still teach the core sequence-learning problems: hidden state, recurrence, long-range credit assignment, gradient stability, teacher forcing, and attention as a solution to fixed-context bottlenecks. Understanding RNNs makes transformer design feel less arbitrary.

## Bridge to Transformer Architecture

Transformers replace recurrent state updates with attention over token states. The next section studies how self-attention, residual streams, normalization, and feed-forward blocks solve many RNN bottlenecks while introducing their own memory and compute tradeoffs.

## References

- Sepp Hochreiter and Jurgen Schmidhuber, "Long Short-Term Memory", Neural Computation, 1997: https://doi.org/10.1162/neco.1997.9.8.1735
- Kyunghyun Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation", 2014: https://arxiv.org/abs/1406.1078
- Alex Graves, "Generating Sequences With Recurrent Neural Networks", 2013: https://arxiv.org/abs/1308.0850
- Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio, "On the difficulty of training Recurrent Neural Networks", 2013: https://arxiv.org/abs/1211.5063
- Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio, "Neural Machine Translation by Jointly Learning to Align and Translate", 2014: https://arxiv.org/abs/1409.0473
