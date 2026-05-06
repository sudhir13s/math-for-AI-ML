[<- Scaling Laws](../08-Scaling-Laws/notes.md) | [Home](../../README.md) | [Mixture of Experts and Routing ->](../10-Mixture-of-Experts-and-Routing/notes.md)

---

# Efficient Attention and Inference

Efficient inference is the mathematics of turning a trained LLM into a responsive service. The main objects are attention cost, KV cache memory, prefill and decode latency, batching, scheduling, and probability-preserving acceleration.

## Overview

Autoregressive inference has two phases:

```text
prompt tokens -> prefill -> KV cache -> decode one token -> append -> decode one token -> ...
```

Prefill processes the prompt in parallel and is often compute-heavy. Decode emits one token at a time and is often memory-bandwidth-heavy because each step reads weights and KV cache. Efficient serving is therefore not just "make attention faster." It is a coordinated memory, kernel, cache, batching, and scheduling problem.

The central cache formula is:

$$
M_\mathrm{KV}=2BLTH_{kv}d_hb.
$$

The factor 2 is for keys and values. $B$ is batch size, $L$ is layers, $T$ is cached context length, $H_{kv}$ is the number of key-value heads, $d_h$ is head dimension, and $b$ is bytes per cache element.

## Prerequisites

- Attention mechanism math and causal masking
- Positional encodings and autoregressive decoding
- Scaling-law and training-at-scale cost vocabulary
- Basic memory units and tensor shapes

## Companion Notebooks

| Notebook | Purpose |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Computes prefill/decode costs, KV cache memory, MHA/MQA/GQA savings, FlashAttention memory intuition, paged-cache waste, speculative speedup, and latency budgets. |
| [exercises.ipynb](exercises.ipynb) | Ten practice problems for KV cache sizing, GQA savings, pipeline latency, page waste, speculative decoding, and serving diagnostics. |

## Learning Objectives

After this section, you should be able to:

- Distinguish prefill latency, decode latency, TTFT, TPOT, throughput, and tail latency.
- Estimate dense attention and decode attention cost.
- Compute KV cache memory for MHA, MQA, and GQA.
- Explain why FlashAttention is exact but IO-aware.
- Explain why PagedAttention-style cache management improves serving memory utilization.
- Reason about continuous batching, chunked prefill, and head-of-line blocking.
- Estimate speculative decoding speedup from acceptance rate and draft latency.
- Build an inference debugging checklist that checks both speed and correctness.

## Table of Contents

1. [Inference Phases](#1-inference-phases)
   - 1.1 [Prefill](#11-prefill)
   - 1.2 [Decode](#12-decode)
   - 1.3 [TTFT](#13-ttft)
   - 1.4 [TPOT](#14-tpot)
   - 1.5 [Throughput](#15-throughput)
2. [Attention Cost](#2-attention-cost)
   - 2.1 [Training attention](#21-training-attention)
   - 2.2 [Decode attention](#22-decode-attention)
   - 2.3 [Causal mask](#23-causal-mask)
   - 2.4 [Memory traffic](#24-memory-traffic)
   - 2.5 [Arithmetic intensity](#25-arithmetic-intensity)
3. [KV Cache Math](#3-kv-cache-math)
   - 3.1 [Cache contents](#31-cache-contents)
   - 3.2 [Cache memory](#32-cache-memory)
   - 3.3 [MHA](#33-mha)
   - 3.4 [MQA](#34-mqa)
   - 3.5 [GQA](#35-gqa)
4. [FlashAttention and IO Awareness](#4-flashattention-and-io-awareness)
   - 4.1 [Naive attention memory](#41-naive-attention-memory)
   - 4.2 [Tiling](#42-tiling)
   - 4.3 [Online softmax](#43-online-softmax)
   - 4.4 [Exactness](#44-exactness)
   - 4.5 [Long context](#45-long-context)
5. [Serving Memory Management](#5-serving-memory-management)
   - 5.1 [Request lengths vary](#51-request-lengths-vary)
   - 5.2 [Paged KV cache](#52-paged-kv-cache)
   - 5.3 [Continuous batching](#53-continuous-batching)
   - 5.4 [Prefix sharing](#54-prefix-sharing)
   - 5.5 [Eviction and recompute](#55-eviction-and-recompute)
6. [Decode Acceleration](#6-decode-acceleration)
   - 6.1 [Greedy serial bottleneck](#61-greedy-serial-bottleneck)
   - 6.2 [Speculative decoding](#62-speculative-decoding)
   - 6.3 [Acceptance rate](#63-acceptance-rate)
   - 6.4 [Parallel heads](#64-parallel-heads)
   - 6.5 [Distribution preservation](#65-distribution-preservation)
7. [Batching and Scheduling](#7-batching-and-scheduling)
   - 7.1 [Static batching](#71-static-batching)
   - 7.2 [Dynamic batching](#72-dynamic-batching)
   - 7.3 [Head-of-line blocking](#73-headofline-blocking)
   - 7.4 [Chunked prefill](#74-chunked-prefill)
   - 7.5 [SLA tradeoff](#75-sla-tradeoff)
8. [Quantization and Bandwidth](#8-quantization-and-bandwidth)
   - 8.1 [Weight bandwidth](#81-weight-bandwidth)
   - 8.2 [KV quantization](#82-kv-quantization)
   - 8.3 [Accuracy tradeoff](#83-accuracy-tradeoff)
   - 8.4 [Kernel support](#84-kernel-support)
   - 8.5 [Mixed systems](#85-mixed-systems)
9. [Latency Metrics](#9-latency-metrics)
   - 9.1 [Queue time](#91-queue-time)
   - 9.2 [First-token latency](#92-firsttoken-latency)
   - 9.3 [Inter-token latency](#93-intertoken-latency)
   - 9.4 [Tail latency](#94-tail-latency)
   - 9.5 [Cost per token](#95-cost-per-token)
10. [Debugging Efficient Inference](#10-debugging-efficient-inference)
   - 10.1 [Shape checks](#101-shape-checks)
   - 10.2 [Cache correctness](#102-cache-correctness)
   - 10.3 [Memory accounting](#103-memory-accounting)
   - 10.4 [Bottleneck attribution](#104-bottleneck-attribution)
   - 10.5 [Quality regression](#105-quality-regression)

---

## Mental Model

```text
quality target
   |
model weights -- kernels -- memory bandwidth -- KV cache -- scheduler -- user latency
                   |              |             |             |
              FlashAttention   quantization   paging      batching
```

Every speedup lives somewhere in this chain. The math tells you which bottleneck it can actually improve.

## 1. Inference Phases

This part studies inference phases as serving math. The goal is to connect latency and memory behavior to concrete tensor shapes, not to memorize system names.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Prefill](#1-prefill) | process the prompt in parallel and create the KV cache | $T_\mathrm{prefill}\propto T_\mathrm{prompt}^2$ for dense attention |
| [Decode](#1-decode) | generate one token at a time using cached keys and values | $T_\mathrm{decode}\propto T_\mathrm{context}$ per output token |
| [TTFT](#1-ttft) | time to first token includes scheduling plus prefill | $\mathrm{TTFT}=T_\mathrm{queue}+T_\mathrm{prefill}+T_\mathrm{sample}$ |
| [TPOT](#1-tpot) | time per output token measures decode speed | $\mathrm{TPOT}=T_\mathrm{decode\ step}$ |
| [Throughput](#1-throughput) | serving systems balance tokens per second against latency | $\mathrm{tokens/sec}=\mathrm{completed\ tokens}/T$ |

### 1.1 Prefill

**Main idea.** Process the prompt in parallel and create the kv cache.

Core relation:

$$T_\mathrm{prefill}\propto T_\mathrm{prompt}^2$ for dense attention$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 1.2 Decode

**Main idea.** Generate one token at a time using cached keys and values.

Core relation:

$$T_\mathrm{decode}\propto T_\mathrm{context}$ per output token$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 1.3 TTFT

**Main idea.** Time to first token includes scheduling plus prefill.

Core relation:

$$\mathrm{TTFT}=T_\mathrm{queue}+T_\mathrm{prefill}+T_\mathrm{sample}$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 1.4 TPOT

**Main idea.** Time per output token measures decode speed.

Core relation:

$$\mathrm{TPOT}=T_\mathrm{decode\ step}$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 1.5 Throughput

**Main idea.** Serving systems balance tokens per second against latency.

Core relation:

$$\mathrm{tokens/sec}=\mathrm{completed\ tokens}/T$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
## 2. Attention Cost

This part studies attention cost as serving math. The goal is to connect latency and memory behavior to concrete tensor shapes, not to memorize system names.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Training attention](#2-training-attention) | dense attention over a full sequence forms a T by T score matrix | $O(T^2d)$ |
| [Decode attention](#2-decode-attention) | one query attends to all cached keys | $O(Td)$ per generated token |
| [Causal mask](#2-causal-mask) | future tokens are invisible but the triangular work is still large in prefill | $j\le i$ |
| [Memory traffic](#2-memory-traffic) | attention can be limited by reads and writes, not only FLOPs | $T\approx\max(T_\mathrm{compute},T_\mathrm{memory})$ |
| [Arithmetic intensity](#2-arithmetic-intensity) | roofline reasoning compares FLOPs to bytes moved | $I=\mathrm{FLOPs}/\mathrm{bytes}$ |

### 2.1 Training attention

**Main idea.** Dense attention over a full sequence forms a t by t score matrix.

Core relation:

$$O(T^2d)$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 2.2 Decode attention

**Main idea.** One query attends to all cached keys.

Core relation:

$$O(Td)$ per generated token$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 2.3 Causal mask

**Main idea.** Future tokens are invisible but the triangular work is still large in prefill.

Core relation:

$$j\le i$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 2.4 Memory traffic

**Main idea.** Attention can be limited by reads and writes, not only flops.

Core relation:

$$T\approx\max(T_\mathrm{compute},T_\mathrm{memory})$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 2.5 Arithmetic intensity

**Main idea.** Roofline reasoning compares flops to bytes moved.

Core relation:

$$I=\mathrm{FLOPs}/\mathrm{bytes}$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
## 3. KV Cache Math

This part studies kv cache math as serving math. The goal is to connect latency and memory behavior to concrete tensor shapes, not to memorize system names.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Cache contents](#3-cache-contents) | store keys and values for every layer, token, and KV head | $K,V\in\mathbb{R}^{L\times T\times H_{kv}\times d_h}$ |
| [Cache memory](#3-cache-memory) | KV memory grows linearly with context and batch | $M_\mathrm{KV}=2BLTH_{kv}d_hb$ |
| [MHA](#3-mha) | standard multi-head attention has one KV head per query head | $H_{kv}=H_q$ |
| [MQA](#3-mqa) | multi-query attention shares one KV head across query heads | $H_{kv}=1$ |
| [GQA](#3-gqa) | grouped-query attention interpolates between MHA and MQA | $1\le H_{kv}\le H_q$ |

### 3.1 Cache contents

**Main idea.** Store keys and values for every layer, token, and kv head.

Core relation:

$$K,V\in\mathbb{R}^{L\times T\times H_{kv}\times d_h}$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 3.2 Cache memory

**Main idea.** Kv memory grows linearly with context and batch.

Core relation:

$$M_\mathrm{KV}=2BLTH_{kv}d_hb$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This one formula often decides how many requests fit on a serving GPU.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 3.3 MHA

**Main idea.** Standard multi-head attention has one kv head per query head.

Core relation:

$$H_{kv}=H_q$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 3.4 MQA

**Main idea.** Multi-query attention shares one kv head across query heads.

Core relation:

$$H_{kv}=1$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 3.5 GQA

**Main idea.** Grouped-query attention interpolates between mha and mqa.

Core relation:

$$1\le H_{kv}\le H_q$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
## 4. FlashAttention and IO Awareness

This part studies flashattention and io awareness as serving math. The goal is to connect latency and memory behavior to concrete tensor shapes, not to memorize system names.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Naive attention memory](#4-naive-attention-memory) | materializing attention scores costs T squared memory | $M_\mathrm{scores}=BT^2H$ |
| [Tiling](#4-tiling) | FlashAttention computes blocks while keeping partial statistics | $\mathrm{softmax}(QK^\top)V$ without storing all scores |
| [Online softmax](#4-online-softmax) | blockwise max and denominator keep softmax exact | $m=\max_i s_i,\quad l=\sum_i e^{s_i-m}$ |
| [Exactness](#4-exactness) | FlashAttention changes memory access, not the attention definition | $\mathrm{output}_\mathrm{flash}=\mathrm{output}_\mathrm{dense}$ |
| [Long context](#4-long-context) | IO savings matter more as T grows | $T^2$ score storage becomes the wall |

### 4.1 Naive attention memory

**Main idea.** Materializing attention scores costs t squared memory.

Core relation:

$$M_\mathrm{scores}=BT^2H$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 4.2 Tiling

**Main idea.** Flashattention computes blocks while keeping partial statistics.

Core relation:

$$\mathrm{softmax}(QK^\top)V$ without storing all scores$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** FlashAttention is fast because it respects the memory hierarchy, not because it approximates attention.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 4.3 Online softmax

**Main idea.** Blockwise max and denominator keep softmax exact.

Core relation:

$$m=\max_i s_i,\quad l=\sum_i e^{s_i-m}$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 4.4 Exactness

**Main idea.** Flashattention changes memory access, not the attention definition.

Core relation:

$$\mathrm{output}_\mathrm{flash}=\mathrm{output}_\mathrm{dense}$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 4.5 Long context

**Main idea.** Io savings matter more as t grows.

Core relation:

$$T^2$ score storage becomes the wall$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
## 5. Serving Memory Management

This part studies serving memory management as serving math. The goal is to connect latency and memory behavior to concrete tensor shapes, not to memorize system names.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Request lengths vary](#5-request-lengths-vary) | static allocation wastes KV memory for short or finished requests | $\mathrm{waste}=M_\mathrm{reserved}-M_\mathrm{used}$ |
| [Paged KV cache](#5-paged-kv-cache) | allocate KV cache in blocks and map logical tokens to physical blocks | $\mathrm{token}\rightarrow\mathrm{block\ id},\mathrm{offset}$ |
| [Continuous batching](#5-continuous-batching) | add and remove requests between decode steps | $B_t$ changes over time |
| [Prefix sharing](#5-prefix-sharing) | shared prompts can reuse KV cache | $KV(x,y)=KV(x)+KV(y\mid x)$ |
| [Eviction and recompute](#5-eviction-and-recompute) | memory pressure can force swapping or rebuilding cache | $T_\mathrm{recompute}$ trades against $M_\mathrm{free}$ |

### 5.1 Request lengths vary

**Main idea.** Static allocation wastes kv memory for short or finished requests.

Core relation:

$$\mathrm{waste}=M_\mathrm{reserved}-M_\mathrm{used}$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 5.2 Paged KV cache

**Main idea.** Allocate kv cache in blocks and map logical tokens to physical blocks.

Core relation:

$$\mathrm{token}\rightarrow\mathrm{block\ id},\mathrm{offset}$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** Paged allocation makes serving look more like an operating-system memory problem.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 5.3 Continuous batching

**Main idea.** Add and remove requests between decode steps.

Core relation:

$$B_t$ changes over time$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 5.4 Prefix sharing

**Main idea.** Shared prompts can reuse kv cache.

Core relation:

$$KV(x,y)=KV(x)+KV(y\mid x)$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 5.5 Eviction and recompute

**Main idea.** Memory pressure can force swapping or rebuilding cache.

Core relation:

$$T_\mathrm{recompute}$ trades against $M_\mathrm{free}$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
## 6. Decode Acceleration

This part studies decode acceleration as serving math. The goal is to connect latency and memory behavior to concrete tensor shapes, not to memorize system names.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Greedy serial bottleneck](#6-greedy-serial-bottleneck) | autoregressive decoding normally emits one token per target-model step | $K$ tokens require $K$ target passes |
| [Speculative decoding](#6-speculative-decoding) | a draft model proposes tokens and the target verifies them | $\mathrm{speedup}\approx \mathrm{accepted\ tokens}/\mathrm{target\ pass}$ |
| [Acceptance rate](#6-acceptance-rate) | speedup depends on draft quality and draft latency | $a=P(\mathrm{accept})$ |
| [Parallel heads](#6-parallel-heads) | methods such as Medusa predict multiple future tokens with extra heads | $y_{t+1:t+k}$ candidates |
| [Distribution preservation](#6-distribution-preservation) | exact speculative schemes preserve the target distribution when acceptance rules are correct | $p_\mathrm{output}=p_\mathrm{target}$ |

### 6.1 Greedy serial bottleneck

**Main idea.** Autoregressive decoding normally emits one token per target-model step.

Core relation:

$$K$ tokens require $K$ target passes$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 6.2 Speculative decoding

**Main idea.** A draft model proposes tokens and the target verifies them.

Core relation:

$$\mathrm{speedup}\approx \mathrm{accepted\ tokens}/\mathrm{target\ pass}$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** The target model can verify several proposed tokens in one pass, reducing serial decode steps.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 6.3 Acceptance rate

**Main idea.** Speedup depends on draft quality and draft latency.

Core relation:

$$a=P(\mathrm{accept})$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 6.4 Parallel heads

**Main idea.** Methods such as medusa predict multiple future tokens with extra heads.

Core relation:

$$y_{t+1:t+k}$ candidates$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 6.5 Distribution preservation

**Main idea.** Exact speculative schemes preserve the target distribution when acceptance rules are correct.

Core relation:

$$p_\mathrm{output}=p_\mathrm{target}$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
## 7. Batching and Scheduling

This part studies batching and scheduling as serving math. The goal is to connect latency and memory behavior to concrete tensor shapes, not to memorize system names.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Static batching](#7-static-batching) | wait for a batch, run together, then finish together | $B$ fixed |
| [Dynamic batching](#7-dynamic-batching) | merge compatible requests at runtime | $B_t$ variable |
| [Head-of-line blocking](#7-headofline-blocking) | one long request can delay short ones | $T_\mathrm{latency}$ depends on schedule |
| [Chunked prefill](#7-chunked-prefill) | split long prompts so decode traffic is not starved | $T_\mathrm{prompt}$ chunks |
| [SLA tradeoff](#7-sla-tradeoff) | higher throughput can increase tail latency | $p95\ \mathrm{latency}$ versus tokens/sec |

### 7.1 Static batching

**Main idea.** Wait for a batch, run together, then finish together.

Core relation:

$$B$ fixed$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 7.2 Dynamic batching

**Main idea.** Merge compatible requests at runtime.

Core relation:

$$B_t$ variable$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 7.3 Head-of-line blocking

**Main idea.** One long request can delay short ones.

Core relation:

$$T_\mathrm{latency}$ depends on schedule$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 7.4 Chunked prefill

**Main idea.** Split long prompts so decode traffic is not starved.

Core relation:

$$T_\mathrm{prompt}$ chunks$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 7.5 SLA tradeoff

**Main idea.** Higher throughput can increase tail latency.

Core relation:

$$p95\ \mathrm{latency}$ versus tokens/sec$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
## 8. Quantization and Bandwidth

This part studies quantization and bandwidth as serving math. The goal is to connect latency and memory behavior to concrete tensor shapes, not to memorize system names.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Weight bandwidth](#8-weight-bandwidth) | decode often reloads weights for each generated token | $T_\mathrm{weights}\approx M_\mathrm{weights}/\mathrm{bandwidth}$ |
| [KV quantization](#8-kv-quantization) | compressing KV cache can increase batch or context capacity | $M_\mathrm{KV}\downarrow$ |
| [Accuracy tradeoff](#8-accuracy-tradeoff) | lower precision can change probabilities | $D_\mathrm{KL}(p_\mathrm{fp},p_\mathrm{quant})$ |
| [Kernel support](#8-kernel-support) | a quantized format helps only if kernels are fast | $T_\mathrm{kernel}$ matters |
| [Mixed systems](#8-mixed-systems) | weights, activations, and KV cache may use different precision | $b_\mathrm{weight}\ne b_\mathrm{KV}$ |

### 8.1 Weight bandwidth

**Main idea.** Decode often reloads weights for each generated token.

Core relation:

$$T_\mathrm{weights}\approx M_\mathrm{weights}/\mathrm{bandwidth}$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 8.2 KV quantization

**Main idea.** Compressing kv cache can increase batch or context capacity.

Core relation:

$$M_\mathrm{KV}\downarrow$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 8.3 Accuracy tradeoff

**Main idea.** Lower precision can change probabilities.

Core relation:

$$D_\mathrm{KL}(p_\mathrm{fp},p_\mathrm{quant})$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 8.4 Kernel support

**Main idea.** A quantized format helps only if kernels are fast.

Core relation:

$$T_\mathrm{kernel}$ matters$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 8.5 Mixed systems

**Main idea.** Weights, activations, and kv cache may use different precision.

Core relation:

$$b_\mathrm{weight}\ne b_\mathrm{KV}$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
## 9. Latency Metrics

This part studies latency metrics as serving math. The goal is to connect latency and memory behavior to concrete tensor shapes, not to memorize system names.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Queue time](#9-queue-time) | user latency includes time before GPU work starts | $T_\mathrm{queue}$ |
| [First-token latency](#9-firsttoken-latency) | long prompts mainly hurt TTFT | $\mathrm{TTFT}$ |
| [Inter-token latency](#9-intertoken-latency) | long outputs mainly accumulate TPOT | $T_\mathrm{output}\approx n_\mathrm{out}\mathrm{TPOT}$ |
| [Tail latency](#9-tail-latency) | p95 and p99 matter more than average for products | $Q_{0.95}(T)$ |
| [Cost per token](#9-cost-per-token) | serving cost combines hardware time and utilization | $\mathrm{cost/token}=T_\mathrm{gpu}\mathrm{price}/\mathrm{tokens}$ |

### 9.1 Queue time

**Main idea.** User latency includes time before gpu work starts.

Core relation:

$$T_\mathrm{queue}$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 9.2 First-token latency

**Main idea.** Long prompts mainly hurt ttft.

Core relation:

$$\mathrm{TTFT}$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 9.3 Inter-token latency

**Main idea.** Long outputs mainly accumulate tpot.

Core relation:

$$T_\mathrm{output}\approx n_\mathrm{out}\mathrm{TPOT}$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 9.4 Tail latency

**Main idea.** P95 and p99 matter more than average for products.

Core relation:

$$Q_{0.95}(T)$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 9.5 Cost per token

**Main idea.** Serving cost combines hardware time and utilization.

Core relation:

$$\mathrm{cost/token}=T_\mathrm{gpu}\mathrm{price}/\mathrm{tokens}$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
## 10. Debugging Efficient Inference

This part studies debugging efficient inference as serving math. The goal is to connect latency and memory behavior to concrete tensor shapes, not to memorize system names.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Shape checks](#10-shape-checks) | KV heads, query heads, and head dimension must align | $H_q/H_{kv}$ integer for GQA |
| [Cache correctness](#10-cache-correctness) | cached decode must match full recomputation | $\max|y_\mathrm{cache}-y_\mathrm{full}|$ small |
| [Memory accounting](#10-memory-accounting) | estimate weights plus KV cache plus workspace | $M=M_\mathrm{weights}+M_\mathrm{KV}+M_\mathrm{workspace}$ |
| [Bottleneck attribution](#10-bottleneck-attribution) | separate compute, memory, scheduler, and network time | $T=\sum_j T_j$ |
| [Quality regression](#10-quality-regression) | speedups must preserve target quality unless explicitly approximate | $\Delta L,\Delta S$ tracked |

### 10.1 Shape checks

**Main idea.** Kv heads, query heads, and head dimension must align.

Core relation:

$$H_q/H_{kv}$ integer for GQA$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 10.2 Cache correctness

**Main idea.** Cached decode must match full recomputation.

Core relation:

$$\max|y_\mathrm{cache}-y_\mathrm{full}|$ small$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** A fast cached path is worthless if it disagrees with full recomputation.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 10.3 Memory accounting

**Main idea.** Estimate weights plus kv cache plus workspace.

Core relation:

$$M=M_\mathrm{weights}+M_\mathrm{KV}+M_\mathrm{workspace}$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 10.4 Bottleneck attribution

**Main idea.** Separate compute, memory, scheduler, and network time.

Core relation:

$$T=\sum_j T_j$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 10.5 Quality regression

**Main idea.** Speedups must preserve target quality unless explicitly approximate.

Core relation:

$$\Delta L,\Delta S$ tracked$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.

---

## Practice Exercises

1. Compute KV cache memory for a model configuration.
2. Compare MHA, GQA, and MQA cache sizes.
3. Estimate prefill versus decode attention operations.
4. Compute naive attention-score memory and FlashAttention savings intuition.
5. Estimate roofline bandwidth-limited runtime.
6. Compute page/block waste for variable request lengths.
7. Estimate continuous batching utilization.
8. Compute speculative decoding expected target passes.
9. Build a latency budget from queue, prefill, decode, and sampling.
10. Write a cached-decode correctness checklist.

## Why This Matters for AI

Training makes a model capable. Inference makes it usable. A model that is excellent but too slow, too expensive, or too memory-hungry cannot serve real users well. Efficient inference math helps you decide whether to change attention kernels, KV head count, cache layout, quantization, batching, or decoding strategy.

## Bridge to Mixture of Experts and Routing

Mixture-of-experts models change the inference problem by activating only some parameters per token while keeping many parameters in memory. The next section studies routing, expert capacity, load balancing, and the difference between total parameters and active parameters.

## References

- Ashish Vaswani et al., "Attention Is All You Need", 2017: https://arxiv.org/abs/1706.03762
- Noam Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need", 2019: https://arxiv.org/abs/1911.02150
- Tri Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", 2022: https://arxiv.org/abs/2205.14135
- Joshua Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", 2023: https://arxiv.org/abs/2305.13245
- Woosuk Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", 2023: https://arxiv.org/abs/2309.06180
- Yaniv Leviathan, Matan Kalman, and Yossi Matias, "Fast Inference from Transformers via Speculative Decoding", 2023: https://proceedings.mlr.press/v202/leviathan23a.html
- Tianle Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads", 2024: https://arxiv.org/abs/2401.10774
