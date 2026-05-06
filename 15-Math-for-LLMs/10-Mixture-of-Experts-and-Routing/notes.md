[<- Efficient Attention and Inference](../09-Efficient-Attention-and-Inference/notes.md) | [Home](../../README.md) | [Quantization and Distillation ->](../11-Quantization-and-Distillation/notes.md)

---

# Mixture of Experts and Routing

Mixture-of-experts models separate total capacity from active per-token computation. A router chooses a small number of expert networks for each token, giving the model many parameters without using all of them on every token.

## Overview

A dense transformer FFN applies the same network to every token. An MoE FFN replaces that one network with many experts:

$$
y_t=\sum_{i\in S_t} g_{t,i}E_i(x_t),
$$

where $S_t$ is the selected expert set for token $t$, $g_{t,i}$ is the router weight, and $E_i$ is expert $i$. The central math is not only the weighted sum. It is the accounting around it: active parameters, total parameters, router probabilities, expert capacity, load balancing, all-to-all dispatch, drop rate, and latency.

## Prerequisites

- Transformer FFN shapes
- Softmax and top-k selection
- Training-at-scale memory and parallelism
- Efficient inference and serving latency vocabulary

## Companion Notebooks

| Notebook | Purpose |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Demonstrates router softmax, top-k dispatch, capacity overflow, auxiliary balance loss, expert histograms, all-to-all traffic, active parameter counts, and router-collapse diagnostics. |
| [exercises.ipynb](exercises.ipynb) | Ten practice problems for routing probabilities, capacity, drop rate, load balancing, expert counts, and MoE debugging. |

## Learning Objectives

After this section, you should be able to:

- Explain total parameters versus active parameters.
- Compute router probabilities and top-k selected experts.
- Count MoE expert parameters and active expert compute.
- Compute expert capacity and token overflow.
- Define importance, load, auxiliary load-balancing loss, entropy, and z-loss.
- Explain why MoE training often needs all-to-all communication.
- Diagnose expert collapse with histograms, drop rate, entropy, and gradient norms.
- Explain the inference tradeoff: lower active compute but higher memory and routing complexity.

## Table of Contents

1. [Dense versus Sparse Computation](#1-dense-versus-sparse-computation)
   - 1.1 [Dense FFN](#11-dense-ffn)
   - 1.2 [Expert bank](#12-expert-bank)
   - 1.3 [Sparse activation](#13-sparse-activation)
   - 1.4 [Total versus active parameters](#14-total-versus-active-parameters)
   - 1.5 [Memory caveat](#15-memory-caveat)
2. [Router Mathematics](#2-router-mathematics)
   - 2.1 [Router logits](#21-router-logits)
   - 2.2 [Routing probabilities](#22-routing-probabilities)
   - 2.3 [Top-k selection](#23-topk-selection)
   - 2.4 [Gated combination](#24-gated-combination)
   - 2.5 [Top-1 Switch routing](#25-top1-switch-routing)
3. [Parameter and FLOP Accounting](#3-parameter-and-flop-accounting)
   - 3.1 [FFN parameter count](#31-ffn-parameter-count)
   - 3.2 [MoE total parameters](#32-moe-total-parameters)
   - 3.3 [MoE active parameters](#33-moe-active-parameters)
   - 3.4 [Router overhead](#34-router-overhead)
   - 3.5 [Compute ratio](#35-compute-ratio)
4. [Capacity and Token Dropping](#4-capacity-and-token-dropping)
   - 4.1 [Expected tokens per expert](#41-expected-tokens-per-expert)
   - 4.2 [Capacity factor](#42-capacity-factor)
   - 4.3 [Overflow](#43-overflow)
   - 4.4 [Batch sensitivity](#44-batch-sensitivity)
   - 4.5 [Expert collapse](#45-expert-collapse)
5. [Load Balancing Losses](#5-load-balancing-losses)
   - 5.1 [Importance](#51-importance)
   - 5.2 [Load](#52-load)
   - 5.3 [Auxiliary loss](#53-auxiliary-loss)
   - 5.4 [Entropy encouragement](#54-entropy-encouragement)
   - 5.5 [Z-loss](#55-zloss)
6. [Expert Parallelism](#6-expert-parallelism)
   - 6.1 [Expert placement](#61-expert-placement)
   - 6.2 [All-to-all dispatch](#62-alltoall-dispatch)
   - 6.3 [Combine step](#63-combine-step)
   - 6.4 [Communication bottleneck](#64-communication-bottleneck)
   - 6.5 [Locality](#65-locality)
7. [Training Dynamics](#7-training-dynamics)
   - 7.1 [Specialization](#71-specialization)
   - 7.2 [Cold experts](#72-cold-experts)
   - 7.3 [Router noise](#73-router-noise)
   - 7.4 [Top-2 gradients](#74-top2-gradients)
   - 7.5 [Stability tradeoff](#75-stability-tradeoff)
8. [Inference Behavior](#8-inference-behavior)
   - 8.1 [Active compute](#81-active-compute)
   - 8.2 [Weight memory](#82-weight-memory)
   - 8.3 [Batch routing variance](#83-batch-routing-variance)
   - 8.4 [Cache interaction](#84-cache-interaction)
   - 8.5 [Latency tails](#85-latency-tails)
9. [MoE Design Variants](#9-moe-design-variants)
   - 9.1 [Sparsely gated MoE](#91-sparsely-gated-moe)
   - 9.2 [GShard](#92-gshard)
   - 9.3 [Switch Transformer](#93-switch-transformer)
   - 9.4 [Top-2 MoE](#94-top2-moe)
   - 9.5 [Shared experts](#95-shared-experts)
10. [Diagnostics](#10-diagnostics)
   - 10.1 [Expert histogram](#101-expert-histogram)
   - 10.2 [Drop rate](#102-drop-rate)
   - 10.3 [Router entropy](#103-router-entropy)
   - 10.4 [Per-expert gradients](#104-perexpert-gradients)
   - 10.5 [Ablations](#105-ablations)

---

## One-Layer MoE Shape Flow

```text
tokens x_t
   |
router logits r_t = W_r x_t
   |
top-k experts S_t
   |
dispatch tokens to experts
   |
expert FFNs E_i(x_t)
   |
weighted combine and restore token order
```

## 1. Dense versus Sparse Computation

This part studies dense versus sparse computation in mixture-of-experts LLMs. The useful habit is to separate probability, capacity, compute, memory, and communication.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Dense FFN](#1-dense-ffn) | every token uses the same feed-forward network | $y=\mathrm{FFN}(x)$ |
| [Expert bank](#1-expert-bank) | replace one FFN with many candidate FFNs | $E_1,\ldots,E_M$ |
| [Sparse activation](#1-sparse-activation) | each token uses only k experts | $k\ll M$ |
| [Total versus active parameters](#1-total-versus-active-parameters) | MoE increases capacity without proportional per-token compute | $P_\mathrm{active}\ll P_\mathrm{total}$ |
| [Memory caveat](#1-memory-caveat) | inactive experts still occupy memory | $M_\mathrm{weights}\propto P_\mathrm{total}$ |

### 1.1 Dense FFN

**Main idea.** Every token uses the same feed-forward network.

Core relation:

$$y=\mathrm{FFN}(x)$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 1.2 Expert bank

**Main idea.** Replace one ffn with many candidate ffns.

Core relation:

$$E_1,\ldots,E_M$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 1.3 Sparse activation

**Main idea.** Each token uses only k experts.

Core relation:

$$k\ll M$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 1.4 Total versus active parameters

**Main idea.** Moe increases capacity without proportional per-token compute.

Core relation:

$$P_\mathrm{active}\ll P_\mathrm{total}$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This distinction is the reason MoE models can have large total capacity with smaller per-token compute.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 1.5 Memory caveat

**Main idea.** Inactive experts still occupy memory.

Core relation:

$$M_\mathrm{weights}\propto P_\mathrm{total}$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
## 2. Router Mathematics

This part studies router mathematics in mixture-of-experts LLMs. The useful habit is to separate probability, capacity, compute, memory, and communication.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Router logits](#2-router-logits) | a small projection scores experts for each token | $r=W_rx$ |
| [Routing probabilities](#2-routing-probabilities) | softmax turns router logits into expert probabilities | $p_i=\exp(r_i)/\sum_j\exp(r_j)$ |
| [Top-k selection](#2-topk-selection) | only the highest scoring experts receive the token | $S=\mathrm{TopK}(p,k)$ |
| [Gated combination](#2-gated-combination) | selected expert outputs are weighted by router probabilities | $y=\sum_{i\in S} \tilde p_i E_i(x)$ |
| [Top-1 Switch routing](#2-top1-switch-routing) | route each token to one expert for simplicity | $y=E_{\arg\max_i p_i}(x)$ |

### 2.1 Router logits

**Main idea.** A small projection scores experts for each token.

Core relation:

$$r=W_rx$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 2.2 Routing probabilities

**Main idea.** Softmax turns router logits into expert probabilities.

Core relation:

$$p_i=\exp(r_i)/\sum_j\exp(r_j)$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 2.3 Top-k selection

**Main idea.** Only the highest scoring experts receive the token.

Core relation:

$$S=\mathrm{TopK}(p,k)$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 2.4 Gated combination

**Main idea.** Selected expert outputs are weighted by router probabilities.

Core relation:

$$y=\sum_{i\in S} \tilde p_i E_i(x)$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 2.5 Top-1 Switch routing

**Main idea.** Route each token to one expert for simplicity.

Core relation:

$$y=E_{\arg\max_i p_i}(x)$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
## 3. Parameter and FLOP Accounting

This part studies parameter and flop accounting in mixture-of-experts LLMs. The useful habit is to separate probability, capacity, compute, memory, and communication.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [FFN parameter count](#3-ffn-parameter-count) | a transformer FFN has two large projections | $P_\mathrm{ffn}\approx 2dd_\mathrm{ff}$ |
| [MoE total parameters](#3-moe-total-parameters) | experts multiply the FFN parameter count | $P_\mathrm{experts}\approx M\cdot 2dd_\mathrm{ff}$ |
| [MoE active parameters](#3-moe-active-parameters) | only selected experts are used per token | $P_\mathrm{active}\approx k\cdot 2dd_\mathrm{ff}$ |
| [Router overhead](#3-router-overhead) | router cost is usually small compared with expert FFNs | $P_\mathrm{router}=dM$ |
| [Compute ratio](#3-compute-ratio) | sparse compute scales with k, not M | $\mathrm{FLOPs}_\mathrm{MoE}/\mathrm{FLOPs}_\mathrm{dense}\approx k$ if expert size matches dense FFN |

### 3.1 FFN parameter count

**Main idea.** A transformer ffn has two large projections.

Core relation:

$$P_\mathrm{ffn}\approx 2dd_\mathrm{ff}$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 3.2 MoE total parameters

**Main idea.** Experts multiply the ffn parameter count.

Core relation:

$$P_\mathrm{experts}\approx M\cdot 2dd_\mathrm{ff}$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 3.3 MoE active parameters

**Main idea.** Only selected experts are used per token.

Core relation:

$$P_\mathrm{active}\approx k\cdot 2dd_\mathrm{ff}$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 3.4 Router overhead

**Main idea.** Router cost is usually small compared with expert ffns.

Core relation:

$$P_\mathrm{router}=dM$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 3.5 Compute ratio

**Main idea.** Sparse compute scales with k, not m.

Core relation:

$$\mathrm{FLOPs}_\mathrm{MoE}/\mathrm{FLOPs}_\mathrm{dense}\approx k$ if expert size matches dense FFN$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
## 4. Capacity and Token Dropping

This part studies capacity and token dropping in mixture-of-experts LLMs. The useful habit is to separate probability, capacity, compute, memory, and communication.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Expected tokens per expert](#4-expected-tokens-per-expert) | balanced routing sends roughly T/M tokens to each expert | $E[n_i]=T/M$ |
| [Capacity factor](#4-capacity-factor) | reserve extra slots beyond the expected load | $C_i=\lceil \mathrm{capacity\ factor}\cdot T/M\rceil$ |
| [Overflow](#4-overflow) | tokens above capacity are dropped or rerouted | $\max(0,n_i-C_i)$ |
| [Batch sensitivity](#4-batch-sensitivity) | small batches have noisier expert loads | $\mathrm{Var}(n_i)=Tp_i(1-p_i)$ |
| [Expert collapse](#4-expert-collapse) | if the router favors a few experts, capacity and learning both suffer | $p_i\approx 0$ for many experts |

### 4.1 Expected tokens per expert

**Main idea.** Balanced routing sends roughly t/m tokens to each expert.

Core relation:

$$E[n_i]=T/M$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 4.2 Capacity factor

**Main idea.** Reserve extra slots beyond the expected load.

Core relation:

$$C_i=\lceil \mathrm{capacity\ factor}\cdot T/M\rceil$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** Capacity is the serving and training contract between the router and the expert bank.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 4.3 Overflow

**Main idea.** Tokens above capacity are dropped or rerouted.

Core relation:

$$\max(0,n_i-C_i)$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 4.4 Batch sensitivity

**Main idea.** Small batches have noisier expert loads.

Core relation:

$$\mathrm{Var}(n_i)=Tp_i(1-p_i)$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 4.5 Expert collapse

**Main idea.** If the router favors a few experts, capacity and learning both suffer.

Core relation:

$$p_i\approx 0$ for many experts$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
## 5. Load Balancing Losses

This part studies load balancing losses in mixture-of-experts LLMs. The useful habit is to separate probability, capacity, compute, memory, and communication.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Importance](#5-importance) | sum of router probabilities assigned to each expert | $I_i=\sum_t p_{t,i}$ |
| [Load](#5-load) | number of tokens actually routed to each expert | $L_i=\sum_t \mathbf{1}\{i\in S_t\}$ |
| [Auxiliary loss](#5-auxiliary-loss) | penalize uneven routing | $L_\mathrm{aux}\propto M\sum_i f_iP_i$ |
| [Entropy encouragement](#5-entropy-encouragement) | router entropy can discourage overconfident early routing | $H(p_t)=-\sum_i p_{t,i}\log p_{t,i}$ |
| [Z-loss](#5-zloss) | penalize large router logits for stability | $L_z=(\log\sum_i e^{r_i})^2$ |

### 5.1 Importance

**Main idea.** Sum of router probabilities assigned to each expert.

Core relation:

$$I_i=\sum_t p_{t,i}$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 5.2 Load

**Main idea.** Number of tokens actually routed to each expert.

Core relation:

$$L_i=\sum_t \mathbf{1}\{i\in S_t\}$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 5.3 Auxiliary loss

**Main idea.** Penalize uneven routing.

Core relation:

$$L_\mathrm{aux}\propto M\sum_i f_iP_i$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** Without balancing, a router can discover a few experts and ignore the rest.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 5.4 Entropy encouragement

**Main idea.** Router entropy can discourage overconfident early routing.

Core relation:

$$H(p_t)=-\sum_i p_{t,i}\log p_{t,i}$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 5.5 Z-loss

**Main idea.** Penalize large router logits for stability.

Core relation:

$$L_z=(\log\sum_i e^{r_i})^2$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
## 6. Expert Parallelism

This part studies expert parallelism in mixture-of-experts LLMs. The useful habit is to separate probability, capacity, compute, memory, and communication.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Expert placement](#6-expert-placement) | different devices own different experts | $E_i\rightarrow \mathrm{rank}(i)$ |
| [All-to-all dispatch](#6-alltoall-dispatch) | tokens move to the devices that own their experts | $\mathrm{tokens}\rightarrow\mathrm{experts}$ |
| [Combine step](#6-combine-step) | expert outputs return to original token order | $y_t=\sum_i g_{t,i}E_i(x_t)$ |
| [Communication bottleneck](#6-communication-bottleneck) | MoE speed depends on token traffic, not only FLOPs | $T_\mathrm{step}\approx\max(T_\mathrm{expert},T_\mathrm{alltoall})$ |
| [Locality](#6-locality) | routing and placement choices can reduce cross-device movement | $\mathrm{traffic}\downarrow$ |

### 6.1 Expert placement

**Main idea.** Different devices own different experts.

Core relation:

$$E_i\rightarrow \mathrm{rank}(i)$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 6.2 All-to-all dispatch

**Main idea.** Tokens move to the devices that own their experts.

Core relation:

$$\mathrm{tokens}\rightarrow\mathrm{experts}$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** MoE turns part of the model into a distributed routing problem.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 6.3 Combine step

**Main idea.** Expert outputs return to original token order.

Core relation:

$$y_t=\sum_i g_{t,i}E_i(x_t)$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 6.4 Communication bottleneck

**Main idea.** Moe speed depends on token traffic, not only flops.

Core relation:

$$T_\mathrm{step}\approx\max(T_\mathrm{expert},T_\mathrm{alltoall})$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 6.5 Locality

**Main idea.** Routing and placement choices can reduce cross-device movement.

Core relation:

$$\mathrm{traffic}\downarrow$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
## 7. Training Dynamics

This part studies training dynamics in mixture-of-experts LLMs. The useful habit is to separate probability, capacity, compute, memory, and communication.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Specialization](#7-specialization) | experts differentiate because they receive different token subsets | $\nabla_{\theta_i}L$ only from routed tokens |
| [Cold experts](#7-cold-experts) | rarely selected experts learn slowly | $n_i\approx 0\Rightarrow \nabla_{\theta_i}\approx 0$ |
| [Router noise](#7-router-noise) | noise can encourage exploration early in training | $r'=r+\epsilon$ |
| [Top-2 gradients](#7-top2-gradients) | top-2 routing gives more experts gradient signal than top-1 | $|S|=2$ |
| [Stability tradeoff](#7-stability-tradeoff) | strong balancing can fight useful specialization | $L=L_\mathrm{task}+\lambda L_\mathrm{aux}$ |

### 7.1 Specialization

**Main idea.** Experts differentiate because they receive different token subsets.

Core relation:

$$\nabla_{\theta_i}L$ only from routed tokens$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 7.2 Cold experts

**Main idea.** Rarely selected experts learn slowly.

Core relation:

$$n_i\approx 0\Rightarrow \nabla_{\theta_i}\approx 0$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 7.3 Router noise

**Main idea.** Noise can encourage exploration early in training.

Core relation:

$$r'=r+\epsilon$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 7.4 Top-2 gradients

**Main idea.** Top-2 routing gives more experts gradient signal than top-1.

Core relation:

$$|S|=2$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 7.5 Stability tradeoff

**Main idea.** Strong balancing can fight useful specialization.

Core relation:

$$L=L_\mathrm{task}+\lambda L_\mathrm{aux}$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
## 8. Inference Behavior

This part studies inference behavior in mixture-of-experts LLMs. The useful habit is to separate probability, capacity, compute, memory, and communication.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Active compute](#8-active-compute) | per-token expert compute depends on k | $k$ selected experts |
| [Weight memory](#8-weight-memory) | serving must store or page all experts that may be routed | $P_\mathrm{total}$ resident or streamed |
| [Batch routing variance](#8-batch-routing-variance) | different requests can activate different experts | $S_t$ varies by token |
| [Cache interaction](#8-cache-interaction) | MoE changes FFN compute but not attention KV cache math directly | $M_\mathrm{KV}$ unchanged by experts |
| [Latency tails](#8-latency-tails) | hot experts and cross-device traffic can increase p95 latency | $Q_{0.95}(T)$ |

### 8.1 Active compute

**Main idea.** Per-token expert compute depends on k.

Core relation:

$$k$ selected experts$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 8.2 Weight memory

**Main idea.** Serving must store or page all experts that may be routed.

Core relation:

$$P_\mathrm{total}$ resident or streamed$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 8.3 Batch routing variance

**Main idea.** Different requests can activate different experts.

Core relation:

$$S_t$ varies by token$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 8.4 Cache interaction

**Main idea.** Moe changes ffn compute but not attention kv cache math directly.

Core relation:

$$M_\mathrm{KV}$ unchanged by experts$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 8.5 Latency tails

**Main idea.** Hot experts and cross-device traffic can increase p95 latency.

Core relation:

$$Q_{0.95}(T)$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
## 9. MoE Design Variants

This part studies moe design variants in mixture-of-experts LLMs. The useful habit is to separate probability, capacity, compute, memory, and communication.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Sparsely gated MoE](#9-sparsely-gated-moe) | learned gates select a sparse expert subset | $y=\sum_i g_iE_i(x)$ |
| [GShard](#9-gshard) | scaled conditional computation with automatic sharding | $\mathrm{expert\ parallelism}$ |
| [Switch Transformer](#9-switch-transformer) | top-1 routing simplifies dispatch | $k=1$ |
| [Top-2 MoE](#9-top2-moe) | two experts can improve quality at higher compute | $k=2$ |
| [Shared experts](#9-shared-experts) | some designs combine routed experts with always-on shared experts | $y=E_\mathrm{shared}(x)+E_\mathrm{routed}(x)$ |

### 9.1 Sparsely gated MoE

**Main idea.** Learned gates select a sparse expert subset.

Core relation:

$$y=\sum_i g_iE_i(x)$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 9.2 GShard

**Main idea.** Scaled conditional computation with automatic sharding.

Core relation:

$$\mathrm{expert\ parallelism}$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 9.3 Switch Transformer

**Main idea.** Top-1 routing simplifies dispatch.

Core relation:

$$k=1$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 9.4 Top-2 MoE

**Main idea.** Two experts can improve quality at higher compute.

Core relation:

$$k=2$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 9.5 Shared experts

**Main idea.** Some designs combine routed experts with always-on shared experts.

Core relation:

$$y=E_\mathrm{shared}(x)+E_\mathrm{routed}(x)$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
## 10. Diagnostics

This part studies diagnostics in mixture-of-experts LLMs. The useful habit is to separate probability, capacity, compute, memory, and communication.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Expert histogram](#10-expert-histogram) | plot token counts per expert | $n_i$ |
| [Drop rate](#10-drop-rate) | measure overflowed tokens | $\mathrm{drop}=\sum_i\max(0,n_i-C_i)/T$ |
| [Router entropy](#10-router-entropy) | track whether routing is collapsing or too diffuse | $H(p)$ |
| [Per-expert gradients](#10-perexpert-gradients) | cold experts have small or zero gradient norms | $\Vert g_i\Vert$ |
| [Ablations](#10-ablations) | compare dense, top-1, top-2, and capacity factors | $\Delta L,\Delta T,\Delta M$ |

### 10.1 Expert histogram

**Main idea.** Plot token counts per expert.

Core relation:

$$n_i$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** The histogram is the first picture to look at when an MoE model behaves strangely.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 10.2 Drop rate

**Main idea.** Measure overflowed tokens.

Core relation:

$$\mathrm{drop}=\sum_i\max(0,n_i-C_i)/T$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 10.3 Router entropy

**Main idea.** Track whether routing is collapsing or too diffuse.

Core relation:

$$H(p)$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 10.4 Per-expert gradients

**Main idea.** Cold experts have small or zero gradient norms.

Core relation:

$$\Vert g_i\Vert$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 10.5 Ablations

**Main idea.** Compare dense, top-1, top-2, and capacity factors.

Core relation:

$$\Delta L,\Delta T,\Delta M$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.

---

## Practice Exercises

1. Compute top-k experts from router probabilities.
2. Count dense FFN and MoE expert parameters.
3. Compute active versus total expert parameters.
4. Compute expert capacity from tokens, experts, and capacity factor.
5. Compute drop rate from expert loads.
6. Compute a Switch-style auxiliary balancing term.
7. Compute router entropy for a token.
8. Estimate all-to-all token traffic by expert placement.
9. Compare top-1 and top-2 active compute.
10. Write an MoE debugging checklist.

## Why This Matters for AI

MoE models are attractive because they can increase capacity without a proportional increase in active compute. But they are not free. They create routing, balancing, communication, memory, and serving problems. Learning MoE math means learning to ask precise questions: which experts were active, how balanced were they, how many tokens were dropped, how much traffic moved, and how much quality came from sparsity rather than raw parameter count?

## Bridge to Quantization and Distillation

Quantization and distillation also change the relationship between quality, memory, and compute. The next section studies how precision reduction and teacher-student training compress models while trying to preserve behavior.

## References

- Noam Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer", 2017: https://arxiv.org/abs/1701.06538
- Dmitry Lepikhin et al., "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding", 2020: https://arxiv.org/abs/2006.16668
- William Fedus, Barret Zoph, and Noam Shazeer, "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", 2021: https://arxiv.org/abs/2101.03961
- Mistral AI, "Mixtral of Experts", 2024: https://arxiv.org/abs/2401.04088
