[<- RAG Math and Retrieval](../12-RAG-Math-and-Retrieval/notes.md) | [Home](../../README.md)

---

# Serving and Systems Tradeoffs

Serving is where LLM math becomes a live system. A deployed model must manage stochastic demand, queueing, prefill, decode, KV cache memory, batching, scheduling, cost, observability, and reliability.

## Overview

The serving equation is not one equation. It is a budget:

$$
T_\mathrm{user}=T_q+T_p+n_\mathrm{out}\mathrm{TPOT}+T_o,
$$

plus memory:

$$
M_w+M_\mathrm{KV}+M_\mathrm{work}\le M_\mathrm{GPU},
$$

plus cost:

$$
\mathrm{CPM}=10^6C_\mathrm{hour}/(3600Q).
$$

Serving tradeoffs are practical: low latency, high throughput, low cost, high quality, and high reliability cannot all be maximized independently.

## Prerequisites

- Efficient attention and inference metrics
- KV cache memory math
- RAG retrieval latency and context budget
- Basic probability for percentiles and queueing

## Companion Notebooks

| Notebook | Purpose |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Demonstrates Little's law, utilization curves, latency decomposition, batch tradeoffs, memory concurrency, cost per million tokens, autoscaling, and SLO budgets. |
| [exercises.ipynb](exercises.ipynb) | Ten practice problems for queueing, latency, memory, cost, scheduling, and observability. |

## Learning Objectives

After this section, you should be able to:

- Compute TTFT, TPOT, total latency, throughput, and cost per million tokens.
- Use Little's law for serving capacity planning.
- Explain why high utilization increases queueing delay.
- Estimate max concurrency from weight memory and KV cache memory.
- Compare static batching, continuous batching, and chunked prefill.
- Explain serving parallelism choices and phase splitting.
- Build a cost model for tokens per dollar.
- Define SLOs, error budgets, and observability traces.
- Choose operational fallbacks under overload.

## Table of Contents

1. [Serving Objectives](#1-serving-objectives)
   - 1.1 [User latency](#11-user-latency)
   - 1.2 [Throughput](#12-throughput)
   - 1.3 [Cost](#13-cost)
   - 1.4 [Quality](#14-quality)
   - 1.5 [Reliability](#15-reliability)
2. [Queueing Basics](#2-queueing-basics)
   - 2.1 [Arrival rate](#21-arrival-rate)
   - 2.2 [Service rate](#22-service-rate)
   - 2.3 [Utilization](#23-utilization)
   - 2.4 [Little's law](#24-littles-law)
   - 2.5 [Tail latency](#25-tail-latency)
3. [Latency Decomposition](#3-latency-decomposition)
   - 3.1 [Queue time](#31-queue-time)
   - 3.2 [Prefill time](#32-prefill-time)
   - 3.3 [Decode time](#33-decode-time)
   - 3.4 [Postprocess time](#34-postprocess-time)
   - 3.5 [End-to-end latency](#35-endtoend-latency)
4. [Batching Tradeoffs](#4-batching-tradeoffs)
   - 4.1 [Static batching](#41-static-batching)
   - 4.2 [Continuous batching](#42-continuous-batching)
   - 4.3 [Batch efficiency](#43-batch-efficiency)
   - 4.4 [Head-of-line blocking](#44-headofline-blocking)
   - 4.5 [Chunked prefill](#45-chunked-prefill)
5. [Memory and Concurrency](#5-memory-and-concurrency)
   - 5.1 [Weight memory](#51-weight-memory)
   - 5.2 [KV cache memory](#52-kv-cache-memory)
   - 5.3 [Workspace memory](#53-workspace-memory)
   - 5.4 [Max concurrency](#54-max-concurrency)
   - 5.5 [Fragmentation](#55-fragmentation)
6. [Parallelism for Serving](#6-parallelism-for-serving)
   - 6.1 [Tensor parallelism](#61-tensor-parallelism)
   - 6.2 [Pipeline parallelism](#62-pipeline-parallelism)
   - 6.3 [Data parallel replicas](#63-data-parallel-replicas)
   - 6.4 [Phase splitting](#64-phase-splitting)
   - 6.5 [Network cost](#65-network-cost)
7. [Cost Modeling](#7-cost-modeling)
   - 7.1 [GPU-hour cost](#71-gpuhour-cost)
   - 7.2 [Tokens per dollar](#72-tokens-per-dollar)
   - 7.3 [Cost per million tokens](#73-cost-per-million-tokens)
   - 7.4 [Utilization](#74-utilization)
   - 7.5 [Quality-adjusted cost](#75-qualityadjusted-cost)
8. [Scheduling Policies](#8-scheduling-policies)
   - 8.1 [FIFO](#81-fifo)
   - 8.2 [Shortest remaining work](#82-shortest-remaining-work)
   - 8.3 [Priority queues](#83-priority-queues)
   - 8.4 [Admission control](#84-admission-control)
   - 8.5 [Autoscaling](#85-autoscaling)
9. [Observability and SLOs](#9-observability-and-slos)
   - 9.1 [Metrics](#91-metrics)
   - 9.2 [Percentiles](#92-percentiles)
   - 9.3 [Error budget](#93-error-budget)
   - 9.4 [Tracing](#94-tracing)
   - 9.5 [Canarying](#95-canarying)
10. [Operational Tradeoffs](#10-operational-tradeoffs)
   - 10.1 [Fallback models](#101-fallback-models)
   - 10.2 [Caching](#102-caching)
   - 10.3 [Rate limits](#103-rate-limits)
   - 10.4 [Graceful degradation](#104-graceful-degradation)
   - 10.5 [Rollback](#105-rollback)

---

## Serving Control Loop

```text
traffic -> admission -> queue -> scheduler -> prefill/decode workers -> postprocess -> response
             |            |          |                 |                  |
          rate limit    metrics   batching          memory             traces
```

## 1. Serving Objectives

This part studies serving objectives as systems math for LLM deployment. The useful habit is to connect every serving choice to latency, throughput, memory, cost, quality, or reliability.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [User latency](#1-user-latency) | interactive systems care about time to first token and total response time | $T_\mathrm{user}=\mathrm{TTFT}+n_\mathrm{out}\mathrm{TPOT}$ |
| [Throughput](#1-throughput) | operators care about tokens completed per second | $Q=\mathrm{tokens}/\mathrm{sec}$ |
| [Cost](#1-cost) | product decisions depend on cost per useful token | $\mathrm{cost/token}=T_\mathrm{gpu}\cdot \mathrm{price}/\mathrm{tokens}$ |
| [Quality](#1-quality) | systems changes must preserve model behavior | $\Delta S$ bounded |
| [Reliability](#1-reliability) | serving must meet SLOs under variable load | $P(T\le T_\mathrm{SLO})\ge 0.95$ |

### 1.1 User latency

**Main idea.** Interactive systems care about time to first token and total response time.

Core relation:

$$T_\mathrm{user}=\mathrm{TTFT}+n_\mathrm{out}\mathrm{TPOT}$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 1.2 Throughput

**Main idea.** Operators care about tokens completed per second.

Core relation:

$$Q=\mathrm{tokens}/\mathrm{sec}$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 1.3 Cost

**Main idea.** Product decisions depend on cost per useful token.

Core relation:

$$\mathrm{cost/token}=T_\mathrm{gpu}\cdot \mathrm{price}/\mathrm{tokens}$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 1.4 Quality

**Main idea.** Systems changes must preserve model behavior.

Core relation:

$$\Delta S$ bounded$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 1.5 Reliability

**Main idea.** Serving must meet slos under variable load.

Core relation:

$$P(T\le T_\mathrm{SLO})\ge 0.95$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
## 2. Queueing Basics

This part studies queueing basics as systems math for LLM deployment. The useful habit is to connect every serving choice to latency, throughput, memory, cost, quality, or reliability.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Arrival rate](#2-arrival-rate) | requests arrive stochastically | $\lambda$ requests/sec |
| [Service rate](#2-service-rate) | the system completes work at rate mu | $\mu$ requests/sec |
| [Utilization](#2-utilization) | high utilization increases queueing delay | $\rho=\lambda/\mu$ |
| [Little's law](#2-littles-law) | average concurrency equals arrival rate times latency | $L=\lambda W$ |
| [Tail latency](#2-tail-latency) | p95 and p99 grow before averages look alarming | $Q_{0.95}(T)$ |

### 2.1 Arrival rate

**Main idea.** Requests arrive stochastically.

Core relation:

$$\lambda$ requests/sec$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 2.2 Service rate

**Main idea.** The system completes work at rate mu.

Core relation:

$$\mu$ requests/sec$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 2.3 Utilization

**Main idea.** High utilization increases queueing delay.

Core relation:

$$\rho=\lambda/\mu$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 2.4 Little's law

**Main idea.** Average concurrency equals arrival rate times latency.

Core relation:

$$L=\lambda W$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is the smallest useful equation for capacity planning.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 2.5 Tail latency

**Main idea.** P95 and p99 grow before averages look alarming.

Core relation:

$$Q_{0.95}(T)$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
## 3. Latency Decomposition

This part studies latency decomposition as systems math for LLM deployment. The useful habit is to connect every serving choice to latency, throughput, memory, cost, quality, or reliability.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Queue time](#3-queue-time) | waiting before GPU work starts | $T_q$ |
| [Prefill time](#3-prefill-time) | process prompt and build KV cache | $T_p$ |
| [Decode time](#3-decode-time) | generate output tokens serially | $T_d=n_\mathrm{out}\mathrm{TPOT}$ |
| [Postprocess time](#3-postprocess-time) | sampling, detokenization, filters, and transport add overhead | $T_o$ |
| [End-to-end latency](#3-endtoend-latency) | measure the whole path users experience | $T=T_q+T_p+T_d+T_o$ |

### 3.1 Queue time

**Main idea.** Waiting before gpu work starts.

Core relation:

$$T_q$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 3.2 Prefill time

**Main idea.** Process prompt and build kv cache.

Core relation:

$$T_p$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 3.3 Decode time

**Main idea.** Generate output tokens serially.

Core relation:

$$T_d=n_\mathrm{out}\mathrm{TPOT}$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 3.4 Postprocess time

**Main idea.** Sampling, detokenization, filters, and transport add overhead.

Core relation:

$$T_o$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 3.5 End-to-end latency

**Main idea.** Measure the whole path users experience.

Core relation:

$$T=T_q+T_p+T_d+T_o$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
## 4. Batching Tradeoffs

This part studies batching tradeoffs as systems math for LLM deployment. The useful habit is to connect every serving choice to latency, throughput, memory, cost, quality, or reliability.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Static batching](#4-static-batching) | wait to form a batch and run it together | $B$ fixed |
| [Continuous batching](#4-continuous-batching) | insert and remove requests between decode steps | $B_t$ changes |
| [Batch efficiency](#4-batch-efficiency) | larger batches increase utilization until memory or latency limits | $Q(B)$ |
| [Head-of-line blocking](#4-headofline-blocking) | long requests can delay short requests | $T_\mathrm{short}$ increases |
| [Chunked prefill](#4-chunked-prefill) | split long prompts to avoid starving decode | $T_p=\sum_c T_{p,c}$ |

### 4.1 Static batching

**Main idea.** Wait to form a batch and run it together.

Core relation:

$$B$ fixed$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 4.2 Continuous batching

**Main idea.** Insert and remove requests between decode steps.

Core relation:

$$B_t$ changes$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is why modern LLM serving does not behave like ordinary fixed-batch inference.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 4.3 Batch efficiency

**Main idea.** Larger batches increase utilization until memory or latency limits.

Core relation:

$$Q(B)$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 4.4 Head-of-line blocking

**Main idea.** Long requests can delay short requests.

Core relation:

$$T_\mathrm{short}$ increases$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 4.5 Chunked prefill

**Main idea.** Split long prompts to avoid starving decode.

Core relation:

$$T_p=\sum_c T_{p,c}$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
## 5. Memory and Concurrency

This part studies memory and concurrency as systems math for LLM deployment. The useful habit is to connect every serving choice to latency, throughput, memory, cost, quality, or reliability.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Weight memory](#5-weight-memory) | model weights consume a fixed resident footprint | $M_w$ |
| [KV cache memory](#5-kv-cache-memory) | active requests consume context-dependent memory | $M_\mathrm{KV}=2BLTH_{kv}d_hb$ |
| [Workspace memory](#5-workspace-memory) | kernels and temporary buffers also need headroom | $M_\mathrm{work}$ |
| [Max concurrency](#5-max-concurrency) | available memory bounds active tokens | $M_w+M_\mathrm{KV}+M_\mathrm{work}\le M_\mathrm{GPU}$ |
| [Fragmentation](#5-fragmentation) | variable request lengths waste reserved cache blocks | $M_\mathrm{reserved}-M_\mathrm{used}$ |

### 5.1 Weight memory

**Main idea.** Model weights consume a fixed resident footprint.

Core relation:

$$M_w$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 5.2 KV cache memory

**Main idea.** Active requests consume context-dependent memory.

Core relation:

$$M_\mathrm{KV}=2BLTH_{kv}d_hb$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 5.3 Workspace memory

**Main idea.** Kernels and temporary buffers also need headroom.

Core relation:

$$M_\mathrm{work}$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 5.4 Max concurrency

**Main idea.** Available memory bounds active tokens.

Core relation:

$$M_w+M_\mathrm{KV}+M_\mathrm{work}\le M_\mathrm{GPU}$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** Most practical serving limits are memory limits before they are pure compute limits.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 5.5 Fragmentation

**Main idea.** Variable request lengths waste reserved cache blocks.

Core relation:

$$M_\mathrm{reserved}-M_\mathrm{used}$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
## 6. Parallelism for Serving

This part studies parallelism for serving as systems math for LLM deployment. The useful habit is to connect every serving choice to latency, throughput, memory, cost, quality, or reliability.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Tensor parallelism](#6-tensor-parallelism) | split matrix operations across devices | $N_\mathrm{tp}$ |
| [Pipeline parallelism](#6-pipeline-parallelism) | place layers on different devices | $N_\mathrm{pp}$ |
| [Data parallel replicas](#6-data-parallel-replicas) | replicate the full serving stack for more throughput | $N_\mathrm{replica}$ |
| [Phase splitting](#6-phase-splitting) | prefill and decode may run on different pools | $\mathrm{prefill\ pool},\mathrm{decode\ pool}$ |
| [Network cost](#6-network-cost) | multi-node serving pays communication latency | $T_\mathrm{net}$ |

### 6.1 Tensor parallelism

**Main idea.** Split matrix operations across devices.

Core relation:

$$N_\mathrm{tp}$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 6.2 Pipeline parallelism

**Main idea.** Place layers on different devices.

Core relation:

$$N_\mathrm{pp}$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 6.3 Data parallel replicas

**Main idea.** Replicate the full serving stack for more throughput.

Core relation:

$$N_\mathrm{replica}$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 6.4 Phase splitting

**Main idea.** Prefill and decode may run on different pools.

Core relation:

$$\mathrm{prefill\ pool},\mathrm{decode\ pool}$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 6.5 Network cost

**Main idea.** Multi-node serving pays communication latency.

Core relation:

$$T_\mathrm{net}$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
## 7. Cost Modeling

This part studies cost modeling as systems math for LLM deployment. The useful habit is to connect every serving choice to latency, throughput, memory, cost, quality, or reliability.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [GPU-hour cost](#7-gpuhour-cost) | hardware price turns time into dollars | $C_\mathrm{hour}$ |
| [Tokens per dollar](#7-tokens-per-dollar) | throughput divided by hourly cost | $\mathrm{tokens}/\$=3600Q/C_\mathrm{hour}$ |
| [Cost per million tokens](#7-cost-per-million-tokens) | standardize cost reporting | $\mathrm{CPM}=10^6C_\mathrm{hour}/(3600Q)$ |
| [Utilization](#7-utilization) | idle capacity increases effective cost | $C_\mathrm{effective}=C_\mathrm{nominal}/u$ |
| [Quality-adjusted cost](#7-qualityadjusted-cost) | cheaper systems can be worse if quality falls | $J=\mathrm{cost}+\lambda(1-S)$ |

### 7.1 GPU-hour cost

**Main idea.** Hardware price turns time into dollars.

Core relation:

$$C_\mathrm{hour}$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 7.2 Tokens per dollar

**Main idea.** Throughput divided by hourly cost.

Core relation:

$$\mathrm{tokens}/\$=3600Q/C_\mathrm{hour}$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 7.3 Cost per million tokens

**Main idea.** Standardize cost reporting.

Core relation:

$$\mathrm{CPM}=10^6C_\mathrm{hour}/(3600Q)$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is the number that connects kernel work to product economics.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 7.4 Utilization

**Main idea.** Idle capacity increases effective cost.

Core relation:

$$C_\mathrm{effective}=C_\mathrm{nominal}/u$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 7.5 Quality-adjusted cost

**Main idea.** Cheaper systems can be worse if quality falls.

Core relation:

$$J=\mathrm{cost}+\lambda(1-S)$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
## 8. Scheduling Policies

This part studies scheduling policies as systems math for LLM deployment. The useful habit is to connect every serving choice to latency, throughput, memory, cost, quality, or reliability.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [FIFO](#8-fifo) | serve requests in arrival order | $\mathrm{order}=t_\mathrm{arrival}$ |
| [Shortest remaining work](#8-shortest-remaining-work) | favor small jobs to reduce mean latency | $\min \hat T_\mathrm{remaining}$ |
| [Priority queues](#8-priority-queues) | separate interactive and batch traffic | $p_i$ priority |
| [Admission control](#8-admission-control) | reject or defer work when queues exceed budget | $L_q>L_\mathrm{max}$ |
| [Autoscaling](#8-autoscaling) | add replicas when load exceeds target utilization | $n\ge \lambda/(\rho_\mathrm{target}\mu)$ |

### 8.1 FIFO

**Main idea.** Serve requests in arrival order.

Core relation:

$$\mathrm{order}=t_\mathrm{arrival}$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 8.2 Shortest remaining work

**Main idea.** Favor small jobs to reduce mean latency.

Core relation:

$$\min \hat T_\mathrm{remaining}$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 8.3 Priority queues

**Main idea.** Separate interactive and batch traffic.

Core relation:

$$p_i$ priority$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 8.4 Admission control

**Main idea.** Reject or defer work when queues exceed budget.

Core relation:

$$L_q>L_\mathrm{max}$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 8.5 Autoscaling

**Main idea.** Add replicas when load exceeds target utilization.

Core relation:

$$n\ge \lambda/(\rho_\mathrm{target}\mu)$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
## 9. Observability and SLOs

This part studies observability and slos as systems math for LLM deployment. The useful habit is to connect every serving choice to latency, throughput, memory, cost, quality, or reliability.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Metrics](#9-metrics) | log TTFT, TPOT, total latency, queue time, throughput, memory, and errors | $m_t$ |
| [Percentiles](#9-percentiles) | track p50, p95, and p99 separately | $Q_p(T)$ |
| [Error budget](#9-error-budget) | allowed failures over a window | $B=(1-\mathrm{SLO})N$ |
| [Tracing](#9-tracing) | attach per-request spans for queue, prefill, decode, retrieval, and postprocess | $\mathrm{trace}$ |
| [Canarying](#9-canarying) | roll out changes to a small fraction and compare metrics | $\Delta m$ |

### 9.1 Metrics

**Main idea.** Log ttft, tpot, total latency, queue time, throughput, memory, and errors.

Core relation:

$$m_t$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 9.2 Percentiles

**Main idea.** Track p50, p95, and p99 separately.

Core relation:

$$Q_p(T)$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 9.3 Error budget

**Main idea.** Allowed failures over a window.

Core relation:

$$B=(1-\mathrm{SLO})N$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 9.4 Tracing

**Main idea.** Attach per-request spans for queue, prefill, decode, retrieval, and postprocess.

Core relation:

$$\mathrm{trace}$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** Without per-request traces, serving optimization becomes guesswork.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 9.5 Canarying

**Main idea.** Roll out changes to a small fraction and compare metrics.

Core relation:

$$\Delta m$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
## 10. Operational Tradeoffs

This part studies operational tradeoffs as systems math for LLM deployment. The useful habit is to connect every serving choice to latency, throughput, memory, cost, quality, or reliability.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Fallback models](#10-fallback-models) | route to smaller models under overload | $M_\mathrm{large}\rightarrow M_\mathrm{small}$ |
| [Caching](#10-caching) | reuse repeated prompts or retrieved context when safe | $y=f(x)$ cache hit |
| [Rate limits](#10-rate-limits) | protect service health by limiting demand | $\lambda\le\lambda_\mathrm{max}$ |
| [Graceful degradation](#10-graceful-degradation) | shorter outputs, lower k retrieval, or smaller model can preserve responsiveness | $T\downarrow$ with bounded $\Delta S$ |
| [Rollback](#10-rollback) | keep a fast path back to the previous stable serving configuration | $v_{new}\rightarrow v_{old}$ |

### 10.1 Fallback models

**Main idea.** Route to smaller models under overload.

Core relation:

$$M_\mathrm{large}\rightarrow M_\mathrm{small}$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 10.2 Caching

**Main idea.** Reuse repeated prompts or retrieved context when safe.

Core relation:

$$y=f(x)$ cache hit$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 10.3 Rate limits

**Main idea.** Protect service health by limiting demand.

Core relation:

$$\lambda\le\lambda_\mathrm{max}$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 10.4 Graceful degradation

**Main idea.** Shorter outputs, lower k retrieval, or smaller model can preserve responsiveness.

Core relation:

$$T\downarrow$ with bounded $\Delta S$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 10.5 Rollback

**Main idea.** Keep a fast path back to the previous stable serving configuration.

Core relation:

$$v_{new}\rightarrow v_{old}$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.

---

## Practice Exercises

1. Use Little's law to compute average concurrency.
2. Compute utilization from arrival and service rates.
3. Build an end-to-end latency budget.
4. Compute max concurrent requests from KV cache memory.
5. Estimate cost per million tokens.
6. Compare batch choices under a latency budget.
7. Compute autoscaling replica count.
8. Compute an SLO error budget.
9. Choose a graceful degradation action under overload.
10. Write a serving trace checklist.

## Why This Matters for AI

LLMs are not useful only because they are trained. They become useful when they can answer real requests within latency, cost, and reliability limits. Serving math keeps deployment decisions honest: every model, context length, retrieval choice, quantization format, and batching policy has a measurable tradeoff.

## References

- Gyeong-In Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models", 2022: https://www.usenix.org/conference/osdi22/presentation/yu
- Woosuk Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", 2023: https://arxiv.org/abs/2309.06180
- Lequn Chen et al., "Efficiently Scaling Transformer Inference", 2023: https://arxiv.org/abs/2211.05102
- Ying Sheng et al., "FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU", 2023: https://arxiv.org/abs/2303.06865
- Amey Agrawal et al., "Sarathi-Serve: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills", 2024: https://arxiv.org/abs/2403.02310
