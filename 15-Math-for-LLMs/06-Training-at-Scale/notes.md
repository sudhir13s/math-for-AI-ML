[<- Language Model Probability](../05-Language-Model-Probability/notes.md) | [Home](../../README.md) | [Fine-Tuning Math ->](../07-Fine-Tuning-Math/notes.md)

---

# Training at Scale

Training at scale is the mathematics of keeping the same language-model objective useful when the model, data, batch size, memory footprint, communication pattern, and wall-clock budget are all large enough to break naive code.

## Overview

The previous section derived the next-token probability objective. This section asks what changes when that objective is optimized over billions of parameters and trillions of token updates. The answer is not a new loss. The answer is a stack of constraints:

```text
loss math -> optimizer state -> batch schedule -> memory budget -> parallelism -> communication -> checkpoint/restart -> validation signal
```

The central skill is quantitative accounting. You should be able to estimate memory before launching, compute effective batch size from the training configuration, reason about why ZeRO/FSDP helps, explain the pipeline bubble, compute approximate training FLOPs, and debug a run whose loss or throughput looks wrong.

## Prerequisites

- Cross-entropy and next-token probability from [05-Language-Model-Probability](../05-Language-Model-Probability/notes.md)
- Gradients, vector norms, and stochastic optimization
- Matrix multiplication shapes from attention and MLP blocks
- Basic probability for mini-batch gradient variance
- Familiarity with GPU memory and distributed workers is useful, but not required

## Companion Notebooks

| Notebook | Purpose |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Executable demos for AdamW, clipping, schedules, memory accounting, ZeRO stages, pipeline bubbles, FLOPs, MFU, and checkpoint overhead. |
| [exercises.ipynb](exercises.ipynb) | Ten practice problems that make you compute the quantities used in real training-plan reviews. |

## Learning Objectives

After this section, you should be able to:

- Explain why training memory is larger than inference memory.
- Derive Adam and AdamW updates with bias correction.
- Compute effective batch size under data parallelism and gradient accumulation.
- Build warmup and cosine learning-rate schedules.
- Estimate parameter, gradient, optimizer, and activation memory.
- Compare data, tensor, pipeline, sequence, and sharded parallelism.
- Compute pipeline bubble fraction and approximate all-reduce cost.
- Estimate training FLOPs with $C\approx 6ND$ and interpret MFU.
- Explain the practical meaning of Kaplan-style and Chinchilla-style scaling laws.
- Build a debugging checklist for loss spikes, bad resumes, low throughput, and masking errors.

## Table of Contents

1. [Scale as a Constraint Problem](#1-scale-as-a-constraint-problem)
   - 1.1 [The same loss at larger cost](#11-the-same-loss-at-larger-cost)
   - 1.2 [Four limiting resources](#12-four-limiting-resources)
   - 1.3 [Parameter, optimizer, and activation memory](#13-parameter-optimizer-and-activation-memory)
   - 1.4 [Throughput versus convergence](#14-throughput-versus-convergence)
   - 1.5 [Failure modes](#15-failure-modes)
2. [Optimization Core](#2-optimization-core)
   - 2.1 [Mini-batch gradient](#21-minibatch-gradient)
   - 2.2 [Adam moments](#22-adam-moments)
   - 2.3 [Bias correction](#23-bias-correction)
   - 2.4 [AdamW](#24-adamw)
   - 2.5 [Gradient clipping](#25-gradient-clipping)
3. [Batching and Schedules](#3-batching-and-schedules)
   - 3.1 [Effective batch size](#31-effective-batch-size)
   - 3.2 [Gradient accumulation](#32-gradient-accumulation)
   - 3.3 [Linear warmup](#33-linear-warmup)
   - 3.4 [Cosine decay](#34-cosine-decay)
   - 3.5 [Critical batch intuition](#35-critical-batch-intuition)
4. [Memory Accounting](#4-memory-accounting)
   - 4.1 [Bytes per parameter](#41-bytes-per-parameter)
   - 4.2 [Activation memory](#42-activation-memory)
   - 4.3 [Activation checkpointing](#43-activation-checkpointing)
   - 4.4 [Optimizer state sharding](#44-optimizer-state-sharding)
   - 4.5 [Offload boundary](#45-offload-boundary)
5. [Parallelism Strategies](#5-parallelism-strategies)
   - 5.1 [Data parallelism](#51-data-parallelism)
   - 5.2 [Tensor parallelism](#52-tensor-parallelism)
   - 5.3 [Pipeline parallelism](#53-pipeline-parallelism)
   - 5.4 [Sequence parallelism](#54-sequence-parallelism)
   - 5.5 [Parallelism product](#55-parallelism-product)
6. [Communication Math](#6-communication-math)
   - 6.1 [All-reduce cost](#61-allreduce-cost)
   - 6.2 [Reduce-scatter and all-gather](#62-reducescatter-and-allgather)
   - 6.3 [Overlap](#63-overlap)
   - 6.4 [Bandwidth hierarchy](#64-bandwidth-hierarchy)
   - 6.5 [Straggler sensitivity](#65-straggler-sensitivity)
7. [Compute and Scaling Laws](#7-compute-and-scaling-laws)
   - 7.1 [Training FLOPs estimate](#71-training-flops-estimate)
   - 7.2 [Kaplan-style power laws](#72-kaplanstyle-power-laws)
   - 7.3 [Compute-optimal tradeoff](#73-computeoptimal-tradeoff)
   - 7.4 [MFU](#74-mfu)
   - 7.5 [Inference-aware training](#75-inferenceaware-training)
8. [Numerical Stability](#8-numerical-stability)
   - 8.1 [Mixed precision](#81-mixed-precision)
   - 8.2 [Loss scaling](#82-loss-scaling)
   - 8.3 [Attention stability](#83-attention-stability)
   - 8.4 [Loss spikes](#84-loss-spikes)
   - 8.5 [Resume correctness](#85-resume-correctness)
9. [Data and Checkpoint Operations](#9-data-and-checkpoint-operations)
   - 9.1 [Token budget](#91-token-budget)
   - 9.2 [Packing](#92-packing)
   - 9.3 [Deduplication and filtering](#93-deduplication-and-filtering)
   - 9.4 [Checkpoint frequency](#94-checkpoint-frequency)
   - 9.5 [Validation cadence](#95-validation-cadence)
10. [Operational Debugging](#10-operational-debugging)
   - 10.1 [Shape and mask checks](#101-shape-and-mask-checks)
   - 10.2 [Gradient norm traces](#102-gradient-norm-traces)
   - 10.3 [Learning-rate traces](#103-learningrate-traces)
   - 10.4 [Throughput decomposition](#104-throughput-decomposition)
   - 10.5 [Reproducible small run](#105-reproducible-small-run)

---

## A Quick Scale Budget

Before a large training run, write a one-page budget:

| Quantity | Example question |
| --- | --- |
| Parameters | How many trainable parameters are updated? |
| Tokens | How many training tokens are consumed? |
| Precision | Which tensors are bf16, fp16, fp32, or quantized? |
| Batch | What is the effective global batch in tokens? |
| Parallelism | Which axes are data, tensor, pipeline, sequence, and sharded? |
| Optimizer | Which states are stored, sharded, or offloaded? |
| Checkpoint | What must be saved to resume exactly? |
| Validation | Which held-out loss confirms learning? |

The point is not paperwork. The point is to make hidden assumptions explicit before they become expensive failures.

## 1. Scale as a Constraint Problem

This part focuses on scale as a constraint problem as a practical mathematical constraint in LLM training. The goal is not to memorize infrastructure names, but to understand the formulas that determine whether a run fits, learns, communicates, and resumes.

| Subtopic | Operational question | Formula |
| --- | --- | --- |
| [The same loss at larger cost](#1-the-same-loss-at-larger-cost) | training at scale still minimizes next-token cross-entropy | $L(\theta)=-\sum_i \log p_\theta(t_i\mid t_{<i})$ |
| [Four limiting resources](#1-four-limiting-resources) | memory, compute, bandwidth, and data quality each become a bottleneck | $\mathrm{time}\approx\max(T_\mathrm{compute},T_\mathrm{comm},T_\mathrm{input})$ |
| [Parameter, optimizer, and activation memory](#1-parameter-optimizer-and-activation-memory) | weights are only one part of training memory | $M=M_\mathrm{params}+M_\mathrm{grads}+M_\mathrm{opt}+M_\mathrm{act}$ |
| [Throughput versus convergence](#1-throughput-versus-convergence) | fast tokens per second are useful only if loss improves | $\mathrm{tokens/sec}$ must be read with $L(\mathrm{tokens})$ |
| [Failure modes](#1-failure-modes) | large training fails by divergence, stalls, bad data, communication bottlenecks, or checkpoint loss | $\Delta L>0$ for many steps is a symptom, not a diagnosis |

### 1.1 The same loss at larger cost

**Main idea.** Training at scale still minimizes next-token cross-entropy.

Core relation:

$$L(\theta)=-\sum_i \log p_\theta(t_i\mid t_{<i})$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 1.2 Four limiting resources

**Main idea.** Memory, compute, bandwidth, and data quality each become a bottleneck.

Core relation:

$$\mathrm{time}\approx\max(T_\mathrm{compute},T_\mathrm{comm},T_\mathrm{input})$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 1.3 Parameter, optimizer, and activation memory

**Main idea.** Weights are only one part of training memory.

Core relation:

$$M=M_\mathrm{params}+M_\mathrm{grads}+M_\mathrm{opt}+M_\mathrm{act}$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 1.4 Throughput versus convergence

**Main idea.** Fast tokens per second are useful only if loss improves.

Core relation:

$$\mathrm{tokens/sec}$ must be read with $L(\mathrm{tokens})$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 1.5 Failure modes

**Main idea.** Large training fails by divergence, stalls, bad data, communication bottlenecks, or checkpoint loss.

Core relation:

$$\Delta L>0$ for many steps is a symptom, not a diagnosis$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
## 2. Optimization Core

This part focuses on optimization core as a practical mathematical constraint in LLM training. The goal is not to memorize infrastructure names, but to understand the formulas that determine whether a run fits, learns, communicates, and resumes.

| Subtopic | Operational question | Formula |
| --- | --- | --- |
| [Mini-batch gradient](#2-minibatch-gradient) | distributed workers estimate the same gradient with different data shards | $g_t=\frac{1}{B}\sum_{i=1}^{B}\nabla_\theta \ell_i$ |
| [Adam moments](#2-adam-moments) | first and second moment estimates adapt update scale | $m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$ |
| [Bias correction](#2-bias-correction) | early moments are corrected because they start at zero | $\hat m_t=m_t/(1-\beta_1^t),\quad \hat v_t=v_t/(1-\beta_2^t)$ |
| [AdamW](#2-adamw) | weight decay is applied outside the adaptive gradient ratio | $\theta_{t+1}=\theta_t-\eta\hat m_t/(\sqrt{\hat v_t}+\epsilon)-\eta\lambda\theta_t$ |
| [Gradient clipping](#2-gradient-clipping) | cap update norm when rare batches produce spikes | $g\leftarrow g\min(1,c/\Vert g\Vert_2)$ |

### 2.1 Mini-batch gradient

**Main idea.** Distributed workers estimate the same gradient with different data shards.

Core relation:

$$g_t=\frac{1}{B}\sum_{i=1}^{B}\nabla_\theta \ell_i$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 2.2 Adam moments

**Main idea.** First and second moment estimates adapt update scale.

Core relation:

$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 2.3 Bias correction

**Main idea.** Early moments are corrected because they start at zero.

Core relation:

$$\hat m_t=m_t/(1-\beta_1^t),\quad \hat v_t=v_t/(1-\beta_2^t)$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 2.4 AdamW

**Main idea.** Weight decay is applied outside the adaptive gradient ratio.

Core relation:

$$\theta_{t+1}=\theta_t-\eta\hat m_t/(\sqrt{\hat v_t}+\epsilon)-\eta\lambda\theta_t$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 2.5 Gradient clipping

**Main idea.** Cap update norm when rare batches produce spikes.

Core relation:

$$g\leftarrow g\min(1,c/\Vert g\Vert_2)$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** Clipping is not a cure for a bad run, but it can prevent one rare batch from destroying useful optimizer state.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
## 3. Batching and Schedules

This part focuses on batching and schedules as a practical mathematical constraint in LLM training. The goal is not to memorize infrastructure names, but to understand the formulas that determine whether a run fits, learns, communicates, and resumes.

| Subtopic | Operational question | Formula |
| --- | --- | --- |
| [Effective batch size](#3-effective-batch-size) | global batch combines devices and accumulation steps | $B_\mathrm{eff}=B_\mathrm{device}G_\mathrm{accum}N_\mathrm{dp}$ |
| [Gradient accumulation](#3-gradient-accumulation) | several micro-batches approximate one larger batch | $g=\frac{1}{K}\sum_{k=1}^{K}g_k$ |
| [Linear warmup](#3-linear-warmup) | the learning rate starts small to avoid early instability | $\eta_t=\eta_\max t/T_\mathrm{warmup}$ |
| [Cosine decay](#3-cosine-decay) | the learning rate anneals smoothly after warmup | $\eta_t=\eta_\min+\frac{1}{2}(\eta_\max-\eta_\min)(1+\cos(\pi s))$ |
| [Critical batch intuition](#3-critical-batch-intuition) | past a point, larger batches waste compute rather than reducing noise usefully | $\mathrm{noise}\propto 1/B$ only in the useful regime |

### 3.1 Effective batch size

**Main idea.** Global batch combines devices and accumulation steps.

Core relation:

$$B_\mathrm{eff}=B_\mathrm{device}G_\mathrm{accum}N_\mathrm{dp}$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 3.2 Gradient accumulation

**Main idea.** Several micro-batches approximate one larger batch.

Core relation:

$$g=\frac{1}{K}\sum_{k=1}^{K}g_k$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 3.3 Linear warmup

**Main idea.** The learning rate starts small to avoid early instability.

Core relation:

$$\eta_t=\eta_\max t/T_\mathrm{warmup}$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 3.4 Cosine decay

**Main idea.** The learning rate anneals smoothly after warmup.

Core relation:

$$\eta_t=\eta_\min+\frac{1}{2}(\eta_\max-\eta_\min)(1+\cos(\pi s))$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 3.5 Critical batch intuition

**Main idea.** Past a point, larger batches waste compute rather than reducing noise usefully.

Core relation:

$$\mathrm{noise}\propto 1/B$ only in the useful regime$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
## 4. Memory Accounting

This part focuses on memory accounting as a practical mathematical constraint in LLM training. The goal is not to memorize infrastructure names, but to understand the formulas that determine whether a run fits, learns, communicates, and resumes.

| Subtopic | Operational question | Formula |
| --- | --- | --- |
| [Bytes per parameter](#4-bytes-per-parameter) | bf16 weights use 2 bytes but Adam states are often fp32 | $M_\mathrm{Adam}\approx 2P + 2P + 8P$ bytes |
| [Activation memory](#4-activation-memory) | stored forward activations can dominate at long context | $M_\mathrm{act}\propto BTLd$ |
| [Activation checkpointing](#4-activation-checkpointing) | save memory by recomputing intermediate activations | $M_\mathrm{act}\downarrow,\quad T_\mathrm{compute}\uparrow$ |
| [Optimizer state sharding](#4-optimizer-state-sharding) | ZeRO/FSDP shard model states across data-parallel ranks | $M_\mathrm{per\ rank}\approx M/N$ for fully sharded states |
| [Offload boundary](#4-offload-boundary) | CPU or NVMe offload trades memory for bandwidth and latency | $T_\mathrm{step}$ can become transfer-bound |

### 4.1 Bytes per parameter

**Main idea.** Bf16 weights use 2 bytes but adam states are often fp32.

Core relation:

$$M_\mathrm{Adam}\approx 2P + 2P + 8P$ bytes$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 4.2 Activation memory

**Main idea.** Stored forward activations can dominate at long context.

Core relation:

$$M_\mathrm{act}\propto BTLd$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 4.3 Activation checkpointing

**Main idea.** Save memory by recomputing intermediate activations.

Core relation:

$$M_\mathrm{act}\downarrow,\quad T_\mathrm{compute}\uparrow$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 4.4 Optimizer state sharding

**Main idea.** Zero/fsdp shard model states across data-parallel ranks.

Core relation:

$$M_\mathrm{per\ rank}\approx M/N$ for fully sharded states$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This is why a model that cannot fit on one accelerator can still be trained across many accelerators.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 4.5 Offload boundary

**Main idea.** Cpu or nvme offload trades memory for bandwidth and latency.

Core relation:

$$T_\mathrm{step}$ can become transfer-bound$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
## 5. Parallelism Strategies

This part focuses on parallelism strategies as a practical mathematical constraint in LLM training. The goal is not to memorize infrastructure names, but to understand the formulas that determine whether a run fits, learns, communicates, and resumes.

| Subtopic | Operational question | Formula |
| --- | --- | --- |
| [Data parallelism](#5-data-parallelism) | replicate the model and all-reduce gradients | $g=\frac{1}{N}\sum_{r=1}^{N}g_r$ |
| [Tensor parallelism](#5-tensor-parallelism) | split matrix multiplications across devices | $Y=X[W_1\ W_2]$ or $Y=XW_1+XW_2$ depending on layout |
| [Pipeline parallelism](#5-pipeline-parallelism) | place layers on stages and stream micro-batches | $\mathrm{bubble}\approx(P-1)/(M+P-1)$ |
| [Sequence parallelism](#5-sequence-parallelism) | split sequence-length work when activations are too large | $T$ is partitioned across ranks |
| [Parallelism product](#5-parallelism-product) | large jobs combine data, tensor, pipeline, and sometimes sequence parallelism | $N_\mathrm{total}=N_\mathrm{dp}N_\mathrm{tp}N_\mathrm{pp}N_\mathrm{sp}$ |

### 5.1 Data parallelism

**Main idea.** Replicate the model and all-reduce gradients.

Core relation:

$$g=\frac{1}{N}\sum_{r=1}^{N}g_r$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 5.2 Tensor parallelism

**Main idea.** Split matrix multiplications across devices.

Core relation:

$$Y=X[W_1\ W_2]$ or $Y=XW_1+XW_2$ depending on layout$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 5.3 Pipeline parallelism

**Main idea.** Place layers on stages and stream micro-batches.

Core relation:

$$\mathrm{bubble}\approx(P-1)/(M+P-1)$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 5.4 Sequence parallelism

**Main idea.** Split sequence-length work when activations are too large.

Core relation:

$$T$ is partitioned across ranks$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 5.5 Parallelism product

**Main idea.** Large jobs combine data, tensor, pipeline, and sometimes sequence parallelism.

Core relation:

$$N_\mathrm{total}=N_\mathrm{dp}N_\mathrm{tp}N_\mathrm{pp}N_\mathrm{sp}$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
## 6. Communication Math

This part focuses on communication math as a practical mathematical constraint in LLM training. The goal is not to memorize infrastructure names, but to understand the formulas that determine whether a run fits, learns, communicates, and resumes.

| Subtopic | Operational question | Formula |
| --- | --- | --- |
| [All-reduce cost](#6-allreduce-cost) | gradient synchronization costs latency plus bandwidth | $T\approx \alpha\log N+\beta S$ |
| [Reduce-scatter and all-gather](#6-reducescatter-and-allgather) | sharded training replaces one all-reduce with state movement primitives | $\mathrm{allreduce}=\mathrm{reduce\ scatter}+\mathrm{all\ gather}$ |
| [Overlap](#6-overlap) | hide communication under backward computation when dependencies allow it | $T_\mathrm{step}\approx\max(T_\mathrm{compute},T_\mathrm{comm})$ |
| [Bandwidth hierarchy](#6-bandwidth-hierarchy) | intra-node links are much faster than inter-node links | $T_\mathrm{inter}>T_\mathrm{intra}$ for the same payload |
| [Straggler sensitivity](#6-straggler-sensitivity) | synchronous steps wait for the slowest rank | $T_\mathrm{step}=\max_r T_r$ |

### 6.1 All-reduce cost

**Main idea.** Gradient synchronization costs latency plus bandwidth.

Core relation:

$$T\approx \alpha\log N+\beta S$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 6.2 Reduce-scatter and all-gather

**Main idea.** Sharded training replaces one all-reduce with state movement primitives.

Core relation:

$$\mathrm{allreduce}=\mathrm{reduce\ scatter}+\mathrm{all\ gather}$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 6.3 Overlap

**Main idea.** Hide communication under backward computation when dependencies allow it.

Core relation:

$$T_\mathrm{step}\approx\max(T_\mathrm{compute},T_\mathrm{comm})$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 6.4 Bandwidth hierarchy

**Main idea.** Intra-node links are much faster than inter-node links.

Core relation:

$$T_\mathrm{inter}>T_\mathrm{intra}$ for the same payload$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 6.5 Straggler sensitivity

**Main idea.** Synchronous steps wait for the slowest rank.

Core relation:

$$T_\mathrm{step}=\max_r T_r$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
## 7. Compute and Scaling Laws

This part focuses on compute and scaling laws as a practical mathematical constraint in LLM training. The goal is not to memorize infrastructure names, but to understand the formulas that determine whether a run fits, learns, communicates, and resumes.

| Subtopic | Operational question | Formula |
| --- | --- | --- |
| [Training FLOPs estimate](#7-training-flops-estimate) | dense transformer training is often approximated by six times parameters times tokens | $C\approx 6ND$ |
| [Kaplan-style power laws](#7-kaplanstyle-power-laws) | loss follows predictable power trends over model, data, and compute in a range | $L(C)=L_\infty+aC^{-\alpha}$ |
| [Compute-optimal tradeoff](#7-computeoptimal-tradeoff) | for a fixed budget, model size and token count must be balanced | $C\approx 6ND$ with both $N$ and $D$ chosen |
| [MFU](#7-mfu) | model FLOPs utilization compares achieved useful FLOPs to hardware peak | $\mathrm{MFU}=\mathrm{model\ FLOPs/sec}/\mathrm{peak\ FLOPs/sec}$ |
| [Inference-aware training](#7-inferenceaware-training) | overtraining a smaller model can reduce serving cost even if it is not pure compute-optimal pretraining | $\mathrm{train\ cost}+\mathrm{serve\ cost}$ matters |

### 7.1 Training FLOPs estimate

**Main idea.** Dense transformer training is often approximated by six times parameters times tokens.

Core relation:

$$C\approx 6ND$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This simple estimate is often the first line in a training-budget spreadsheet.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 7.2 Kaplan-style power laws

**Main idea.** Loss follows predictable power trends over model, data, and compute in a range.

Core relation:

$$L(C)=L_\infty+aC^{-\alpha}$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 7.3 Compute-optimal tradeoff

**Main idea.** For a fixed budget, model size and token count must be balanced.

Core relation:

$$C\approx 6ND$ with both $N$ and $D$ chosen$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 7.4 MFU

**Main idea.** Model flops utilization compares achieved useful flops to hardware peak.

Core relation:

$$\mathrm{MFU}=\mathrm{model\ FLOPs/sec}/\mathrm{peak\ FLOPs/sec}$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This separates a slow model because it is mathematically large from a slow run because the system is wasting hardware.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 7.5 Inference-aware training

**Main idea.** Overtraining a smaller model can reduce serving cost even if it is not pure compute-optimal pretraining.

Core relation:

$$\mathrm{train\ cost}+\mathrm{serve\ cost}$ matters$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
## 8. Numerical Stability

This part focuses on numerical stability as a practical mathematical constraint in LLM training. The goal is not to memorize infrastructure names, but to understand the formulas that determine whether a run fits, learns, communicates, and resumes.

| Subtopic | Operational question | Formula |
| --- | --- | --- |
| [Mixed precision](#8-mixed-precision) | bf16/fp16 reduce memory and increase throughput but require stable reductions | $\theta$ may be bf16 while optimizer states stay fp32 |
| [Loss scaling](#8-loss-scaling) | fp16 may need scaling to avoid underflow | $\tilde L=sL,\quad \tilde g=sg$ |
| [Attention stability](#8-attention-stability) | score scaling and stable softmax matter more at long sequence lengths | $QK^\top/\sqrt d$ |
| [Loss spikes](#8-loss-spikes) | spikes can come from data, optimizer state, numerical overflow, or synchronization problems | $L_t\gg\mathrm{median}(L_{t-k:t})$ |
| [Resume correctness](#8-resume-correctness) | checkpoint reload must restore model, optimizer, scheduler, RNG, and dataloader state | $\theta,m,v,t,\mathrm{rng}$ all matter |

### 8.1 Mixed precision

**Main idea.** Bf16/fp16 reduce memory and increase throughput but require stable reductions.

Core relation:

$$\theta$ may be bf16 while optimizer states stay fp32$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 8.2 Loss scaling

**Main idea.** Fp16 may need scaling to avoid underflow.

Core relation:

$$\tilde L=sL,\quad \tilde g=sg$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 8.3 Attention stability

**Main idea.** Score scaling and stable softmax matter more at long sequence lengths.

Core relation:

$$QK^\top/\sqrt d$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 8.4 Loss spikes

**Main idea.** Spikes can come from data, optimizer state, numerical overflow, or synchronization problems.

Core relation:

$$L_t\gg\mathrm{median}(L_{t-k:t})$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 8.5 Resume correctness

**Main idea.** Checkpoint reload must restore model, optimizer, scheduler, rng, and dataloader state.

Core relation:

$$\theta,m,v,t,\mathrm{rng}$ all matter$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** A bad resume can silently fork the training trajectory even when the checkpoint file loads.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
## 9. Data and Checkpoint Operations

This part focuses on data and checkpoint operations as a practical mathematical constraint in LLM training. The goal is not to memorize infrastructure names, but to understand the formulas that determine whether a run fits, learns, communicates, and resumes.

| Subtopic | Operational question | Formula |
| --- | --- | --- |
| [Token budget](#9-token-budget) | data is counted in tokens, not documents | $D=\sum_\mathrm{docs}\mathrm{tokens(doc)}$ |
| [Packing](#9-packing) | short examples are packed to reduce padding waste | $\mathrm{utilization}=\mathrm{real\ tokens}/\mathrm{allocated\ tokens}$ |
| [Deduplication and filtering](#9-deduplication-and-filtering) | bad repeated data can improve train loss while hurting generalization | $p_\mathrm{train}$ can drift from desired $p_\mathrm{deploy}$ |
| [Checkpoint frequency](#9-checkpoint-frequency) | the optimal interval balances lost work and checkpoint overhead | $\mathrm{overhead}\approx T_\mathrm{ckpt}/K+\mathrm{failure\ loss}(K)$ |
| [Validation cadence](#9-validation-cadence) | held-out loss catches overfitting, data bugs, and regression after resume | $L_\mathrm{val}$ is the early warning signal |

### 9.1 Token budget

**Main idea.** Data is counted in tokens, not documents.

Core relation:

$$D=\sum_\mathrm{docs}\mathrm{tokens(doc)}$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 9.2 Packing

**Main idea.** Short examples are packed to reduce padding waste.

Core relation:

$$\mathrm{utilization}=\mathrm{real\ tokens}/\mathrm{allocated\ tokens}$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 9.3 Deduplication and filtering

**Main idea.** Bad repeated data can improve train loss while hurting generalization.

Core relation:

$$p_\mathrm{train}$ can drift from desired $p_\mathrm{deploy}$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 9.4 Checkpoint frequency

**Main idea.** The optimal interval balances lost work and checkpoint overhead.

Core relation:

$$\mathrm{overhead}\approx T_\mathrm{ckpt}/K+\mathrm{failure\ loss}(K)$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 9.5 Validation cadence

**Main idea.** Held-out loss catches overfitting, data bugs, and regression after resume.

Core relation:

$$L_\mathrm{val}$ is the early warning signal$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
## 10. Operational Debugging

This part focuses on operational debugging as a practical mathematical constraint in LLM training. The goal is not to memorize infrastructure names, but to understand the formulas that determine whether a run fits, learns, communicates, and resumes.

| Subtopic | Operational question | Formula |
| --- | --- | --- |
| [Shape and mask checks](#10-shape-and-mask-checks) | wrong labels or masks can produce plausible but meaningless loss | $\mathrm{target}_i=\mathrm{input}_{i+1}$ |
| [Gradient norm traces](#10-gradient-norm-traces) | track global norms before and after clipping | $\Vert g\Vert_2$ |
| [Learning-rate traces](#10-learningrate-traces) | optimizer behavior must match the intended schedule | $\eta_t$ |
| [Throughput decomposition](#10-throughput-decomposition) | separate dataloader, forward, backward, communication, optimizer, and checkpoint time | $T_\mathrm{step}=\sum_j T_j$ |
| [Reproducible small run](#10-reproducible-small-run) | scale only after a small deterministic run learns and resumes correctly | $L_{100}<L_0$ is a smoke test |

### 10.1 Shape and mask checks

**Main idea.** Wrong labels or masks can produce plausible but meaningless loss.

Core relation:

$$\mathrm{target}_i=\mathrm{input}_{i+1}$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 10.2 Gradient norm traces

**Main idea.** Track global norms before and after clipping.

Core relation:

$$\Vert g\Vert_2$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 10.3 Learning-rate traces

**Main idea.** Optimizer behavior must match the intended schedule.

Core relation:

$$\eta_t$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 10.4 Throughput decomposition

**Main idea.** Separate dataloader, forward, backward, communication, optimizer, and checkpoint time.

Core relation:

$$T_\mathrm{step}=\sum_j T_j$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 10.5 Reproducible small run

**Main idea.** Scale only after a small deterministic run learns and resumes correctly.

Core relation:

$$L_{100}<L_0$ is a smoke test$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.

---

## Practice Exercises

1. Compute one AdamW update by hand for a scalar parameter.
2. Clip a gradient vector to a target norm.
3. Build a warmup plus cosine learning-rate schedule.
4. Compute effective batch size in tokens.
5. Estimate memory for Adam training with and without sharding.
6. Compute a pipeline bubble fraction.
7. Determine tensor-parallel shard shapes for a linear layer.
8. Estimate training FLOPs from parameter and token counts.
9. Compute model FLOPs utilization from achieved throughput.
10. Create a launch checklist for a small reproducible training run.

## Why This Matters for AI

Good LLM training is not only about choosing a model architecture. The optimizer can diverge, the memory plan can be impossible, the communication plan can waste the cluster, the data stream can repeat contaminated text, and the checkpoint can fail to restore optimizer state. The mathematics in this section lets you reason about those failures before the run burns budget.

## Bridge to Fine-Tuning Math

Fine-tuning keeps many of the same scale constraints, but usually changes the parameter-update surface. The next section studies full fine-tuning, adapters, LoRA-style low-rank updates, prompt tuning, and preference-oriented objectives. The training-at-scale accounting here remains useful because every fine-tuning method still has memory, optimizer, throughput, and evaluation budgets.

## References

- Jared Kaplan et al., "Scaling Laws for Neural Language Models", 2020: https://arxiv.org/abs/2001.08361
- Jordan Hoffmann et al., "Training Compute-Optimal Large Language Models", 2022: https://arxiv.org/abs/2203.15556
- Mohammad Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism", 2019: https://arxiv.org/abs/1909.08053
- Samyam Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models", 2019: https://arxiv.org/abs/1910.02054
- Priya Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour", 2017: https://arxiv.org/abs/1706.02677
- Noam Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer", 2017: https://arxiv.org/abs/1701.06538
- OpenAI, "AI and Compute", 2018: https://openai.com/research/ai-and-compute
