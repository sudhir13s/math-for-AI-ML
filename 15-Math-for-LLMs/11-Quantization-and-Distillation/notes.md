[<- Mixture of Experts and Routing](../10-Mixture-of-Experts-and-Routing/notes.md) | [Home](../../README.md) | [RAG Math and Retrieval ->](../12-RAG-Math-and-Retrieval/notes.md)

---

# Quantization and Distillation

Quantization compresses tensors by changing their numeric representation. Distillation compresses behavior by training a smaller or cheaper model to imitate a stronger teacher. Both are central to making LLMs cheaper to store, serve, and deploy.

## Overview

The simplest uniform quantizer is:

$$
q=\mathrm{round}(x/s)+z,\qquad \hat x=s(q-z).
$$

Here $s$ is the scale and $z$ is the zero point. Quantization asks how much error this approximation introduces and whether the hardware can exploit the smaller representation.

Distillation uses a teacher distribution:

$$
L_\mathrm{KD}=\tau^2D_\mathrm{KL}(p_T^{(\tau)}\Vert p_S^{(\tau)}).
$$

The student learns from teacher probabilities, generated sequences, features, or preferences. Quantization and distillation can be combined: distill to a smaller model, then quantize it for deployment.

## Prerequisites

- Logits, softmax, KL divergence, and cross-entropy
- Matrix multiplication and tensor shapes
- Inference memory and bandwidth constraints
- Fine-tuning and LoRA basics

## Companion Notebooks

| Notebook | Purpose |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Demonstrates uniform quantization, clipping, group-wise scales, error versus bits, SmoothQuant intuition, distillation temperature, KL loss, and memory savings. |
| [exercises.ipynb](exercises.ipynb) | Ten practice problems for quantizer formulas, memory savings, clipping, KL distillation, and deployment checks. |

## Learning Objectives

After this section, you should be able to:

- Implement affine quantization and dequantization.
- Explain scale, zero point, clipping, and quantization error.
- Compare per-tensor, per-channel, and group-wise quantization.
- Explain PTQ, QAT, GPTQ intuition, AWQ intuition, and QLoRA.
- Compute memory savings from lower bit widths.
- Compute distillation KL loss with temperature.
- Explain logit, sequence, feature, and preference distillation.
- Build a quality and latency evaluation checklist for compressed LLMs.

## Table of Contents

1. [Compression Goals](#1-compression-goals)
   - 1.1 [Memory reduction](#11-memory-reduction)
   - 1.2 [Bandwidth reduction](#12-bandwidth-reduction)
   - 1.3 [Compute support](#13-compute-support)
   - 1.4 [Quality preservation](#14-quality-preservation)
   - 1.5 [Pareto frontier](#15-pareto-frontier)
2. [Uniform Quantization](#2-uniform-quantization)
   - 2.1 [Affine quantizer](#21-affine-quantizer)
   - 2.2 [Dequantization](#22-dequantization)
   - 2.3 [Scale](#23-scale)
   - 2.4 [Zero point](#24-zero-point)
   - 2.5 [Quantization error](#25-quantization-error)
3. [Granularity](#3-granularity)
   - 3.1 [Per-tensor](#31-pertensor)
   - 3.2 [Per-channel](#32-perchannel)
   - 3.3 [Group-wise](#33-groupwise)
   - 3.4 [Activation quantization](#34-activation-quantization)
   - 3.5 [KV-cache quantization](#35-kvcache-quantization)
4. [Post-Training Quantization](#4-posttraining-quantization)
   - 4.1 [Calibration data](#41-calibration-data)
   - 4.2 [Clipping](#42-clipping)
   - 4.3 [Weighted error](#43-weighted-error)
   - 4.4 [GPTQ intuition](#44-gptq-intuition)
   - 4.5 [AWQ intuition](#45-awq-intuition)
5. [Quantization-Aware Training and QLoRA](#5-quantizationaware-training-and-qlora)
   - 5.1 [Fake quantization](#51-fake-quantization)
   - 5.2 [Straight-through estimator](#52-straightthrough-estimator)
   - 5.3 [QLoRA pattern](#53-qlora-pattern)
   - 5.4 [Optimizer memory](#54-optimizer-memory)
   - 5.5 [Dequantization path](#55-dequantization-path)
6. [Distillation Basics](#6-distillation-basics)
   - 6.1 [Teacher and student](#61-teacher-and-student)
   - 6.2 [Soft targets](#62-soft-targets)
   - 6.3 [Temperature](#63-temperature)
   - 6.4 [KL distillation loss](#64-kl-distillation-loss)
   - 6.5 [Hard-label mixture](#65-hardlabel-mixture)
7. [LLM Distillation Types](#7-llm-distillation-types)
   - 7.1 [Logit distillation](#71-logit-distillation)
   - 7.2 [Sequence distillation](#72-sequence-distillation)
   - 7.3 [Feature distillation](#73-feature-distillation)
   - 7.4 [Preference distillation](#74-preference-distillation)
   - 7.5 [Reasoning trace distillation](#75-reasoning-trace-distillation)
8. [Error and Evaluation](#8-error-and-evaluation)
   - 8.1 [Perplexity shift](#81-perplexity-shift)
   - 8.2 [Task score shift](#82-task-score-shift)
   - 8.3 [Calibration shift](#83-calibration-shift)
   - 8.4 [Layer sensitivity](#84-layer-sensitivity)
   - 8.5 [Outlier channels](#85-outlier-channels)
9. [Deployment Choices](#9-deployment-choices)
   - 9.1 [Weight-only quantization](#91-weightonly-quantization)
   - 9.2 [Weight-activation quantization](#92-weightactivation-quantization)
   - 9.3 [KV-cache quantization](#93-kvcache-quantization)
   - 9.4 [Distill then quantize](#94-distill-then-quantize)
   - 9.5 [Hardware format](#95-hardware-format)
10. [Debugging Checklist](#10-debugging-checklist)
   - 10.1 [Calibration representativeness](#101-calibration-representativeness)
   - 10.2 [Layer-by-layer error](#102-layerbylayer-error)
   - 10.3 [Reference comparisons](#103-reference-comparisons)
   - 10.4 [Latency measurement](#104-latency-measurement)
   - 10.5 [Quality gates](#105-quality-gates)

---

## Compression Map

| Method | Changes | Needs data? | Main benefit | Main risk |
| --- | --- | --- | --- | --- |
| PTQ | Numeric format after training | Calibration data | Fast compression | Calibration mismatch |
| QAT | Training simulates quantization | Training data | Better low-bit robustness | More compute |
| QLoRA | Quantized frozen base plus LoRA | Fine-tune data | Cheap adaptation | Activation memory remains |
| Logit distillation | Student matches teacher probabilities | Teacher outputs | Smaller model behavior transfer | Teacher errors transfer |
| Sequence distillation | Student trains on teacher completions | Teacher generations | Simple data pipeline | Diversity loss |

## 1. Compression Goals

This part studies compression goals as compression math. The useful habit is to separate storage format, dequantized computation, approximation error, and evaluation.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Memory reduction](#1-memory-reduction) | store fewer bytes per parameter or cache entry | $M=P\cdot b/8$ |
| [Bandwidth reduction](#1-bandwidth-reduction) | read fewer bytes per generated token | $T_\mathrm{read}\approx M/\mathrm{bandwidth}$ |
| [Compute support](#1-compute-support) | low precision helps only when kernels and hardware support it | $T_\mathrm{kernel}$ |
| [Quality preservation](#1-quality-preservation) | compressed outputs should stay close to reference outputs | $D_\mathrm{KL}(p_\mathrm{ref}\Vert p_\mathrm{comp})$ |
| [Pareto frontier](#1-pareto-frontier) | compression is a quality-cost tradeoff | $\mathrm{quality}$ versus $\mathrm{memory},\mathrm{latency},\mathrm{cost}$ |

### 1.1 Memory reduction

**Main idea.** Store fewer bytes per parameter or cache entry.

Core relation:

$$M=P\cdot b/8$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 1.2 Bandwidth reduction

**Main idea.** Read fewer bytes per generated token.

Core relation:

$$T_\mathrm{read}\approx M/\mathrm{bandwidth}$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 1.3 Compute support

**Main idea.** Low precision helps only when kernels and hardware support it.

Core relation:

$$T_\mathrm{kernel}$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 1.4 Quality preservation

**Main idea.** Compressed outputs should stay close to reference outputs.

Core relation:

$$D_\mathrm{KL}(p_\mathrm{ref}\Vert p_\mathrm{comp})$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 1.5 Pareto frontier

**Main idea.** Compression is a quality-cost tradeoff.

Core relation:

$$\mathrm{quality}$ versus $\mathrm{memory},\mathrm{latency},\mathrm{cost}$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
## 2. Uniform Quantization

This part studies uniform quantization as compression math. The useful habit is to separate storage format, dequantized computation, approximation error, and evaluation.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Affine quantizer](#2-affine-quantizer) | map real values to integer grid points | $q=\mathrm{round}(x/s)+z$ |
| [Dequantization](#2-dequantization) | recover an approximate real value | $\hat x=s(q-z)$ |
| [Scale](#2-scale) | the step size controls resolution | $s=(x_\max-x_\min)/(q_\max-q_\min)$ |
| [Zero point](#2-zero-point) | asymmetric quantization uses an integer offset | $z=q_\min-\mathrm{round}(x_\min/s)$ |
| [Quantization error](#2-quantization-error) | rounding error is bounded by half a step before clipping | $|x-\hat x|\le s/2$ |

### 2.1 Affine quantizer

**Main idea.** Map real values to integer grid points.

Core relation:

$$q=\mathrm{round}(x/s)+z$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This tiny formula is the bridge between real model weights and integer storage.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 2.2 Dequantization

**Main idea.** Recover an approximate real value.

Core relation:

$$\hat x=s(q-z)$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 2.3 Scale

**Main idea.** The step size controls resolution.

Core relation:

$$s=(x_\max-x_\min)/(q_\max-q_\min)$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 2.4 Zero point

**Main idea.** Asymmetric quantization uses an integer offset.

Core relation:

$$z=q_\min-\mathrm{round}(x_\min/s)$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 2.5 Quantization error

**Main idea.** Rounding error is bounded by half a step before clipping.

Core relation:

$$|x-\hat x|\le s/2$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
## 3. Granularity

This part studies granularity as compression math. The useful habit is to separate storage format, dequantized computation, approximation error, and evaluation.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Per-tensor](#3-pertensor) | one scale for the whole tensor | $s$ shared by all entries |
| [Per-channel](#3-perchannel) | one scale per output channel or column | $s_c$ |
| [Group-wise](#3-groupwise) | one scale per block of weights | $s_g$ |
| [Activation quantization](#3-activation-quantization) | activation ranges depend on input data | $x=x(\mathrm{batch})$ |
| [KV-cache quantization](#3-kvcache-quantization) | cache precision affects long-context memory and attention quality | $M_\mathrm{KV}\propto b_\mathrm{KV}$ |

### 3.1 Per-tensor

**Main idea.** One scale for the whole tensor.

Core relation:

$$s$ shared by all entries$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 3.2 Per-channel

**Main idea.** One scale per output channel or column.

Core relation:

$$s_c$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 3.3 Group-wise

**Main idea.** One scale per block of weights.

Core relation:

$$s_g$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** Group scales are one reason modern low-bit LLM quantization can work better than one global scale.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 3.4 Activation quantization

**Main idea.** Activation ranges depend on input data.

Core relation:

$$x=x(\mathrm{batch})$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 3.5 KV-cache quantization

**Main idea.** Cache precision affects long-context memory and attention quality.

Core relation:

$$M_\mathrm{KV}\propto b_\mathrm{KV}$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
## 4. Post-Training Quantization

This part studies post-training quantization as compression math. The useful habit is to separate storage format, dequantized computation, approximation error, and evaluation.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Calibration data](#4-calibration-data) | estimate activation or weight ranges from representative examples | $D_\mathrm{cal}$ |
| [Clipping](#4-clipping) | smaller range improves resolution but clips outliers | $x\leftarrow\mathrm{clip}(x,-c,c)$ |
| [Weighted error](#4-weighted-error) | some weights matter more because activations amplify them | $\Vert X(W-\hat W)\Vert^2$ |
| [GPTQ intuition](#4-gptq-intuition) | quantize weights while compensating error using approximate second-order information | $\Delta L\approx \frac12\Delta w^\top H\Delta w$ |
| [AWQ intuition](#4-awq-intuition) | protect activation-salient channels during weight quantization | $|x_j|$ indicates sensitivity |

### 4.1 Calibration data

**Main idea.** Estimate activation or weight ranges from representative examples.

Core relation:

$$D_\mathrm{cal}$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 4.2 Clipping

**Main idea.** Smaller range improves resolution but clips outliers.

Core relation:

$$x\leftarrow\mathrm{clip}(x,-c,c)$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 4.3 Weighted error

**Main idea.** Some weights matter more because activations amplify them.

Core relation:

$$\Vert X(W-\hat W)\Vert^2$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** Quantizing a weight is more harmful when common activations magnify that weight's error.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 4.4 GPTQ intuition

**Main idea.** Quantize weights while compensating error using approximate second-order information.

Core relation:

$$\Delta L\approx \frac12\Delta w^\top H\Delta w$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 4.5 AWQ intuition

**Main idea.** Protect activation-salient channels during weight quantization.

Core relation:

$$|x_j|$ indicates sensitivity$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
## 5. Quantization-Aware Training and QLoRA

This part studies quantization-aware training and qlora as compression math. The useful habit is to separate storage format, dequantized computation, approximation error, and evaluation.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Fake quantization](#5-fake-quantization) | simulate quantization during training while keeping gradients useful | $\hat W=Q(W)$ in forward |
| [Straight-through estimator](#5-straightthrough-estimator) | treat rounding as identity in backward | $\partial \mathrm{round}(x)/\partial x\approx 1$ |
| [QLoRA pattern](#5-qlora-pattern) | freeze a quantized base and train low-rank adapters | $W\approx Q(W_0)+(\alpha/r)BA$ |
| [Optimizer memory](#5-optimizer-memory) | optimizer states are needed for trainable adapters, not frozen base weights | $M_\mathrm{opt}\propto P_\mathrm{trainable}$ |
| [Dequantization path](#5-dequantization-path) | many kernels dequantize blocks on the fly for matmul | $q,s\rightarrow \hat W$ |

### 5.1 Fake quantization

**Main idea.** Simulate quantization during training while keeping gradients useful.

Core relation:

$$\hat W=Q(W)$ in forward$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 5.2 Straight-through estimator

**Main idea.** Treat rounding as identity in backward.

Core relation:

$$\partial \mathrm{round}(x)/\partial x\approx 1$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 5.3 QLoRA pattern

**Main idea.** Freeze a quantized base and train low-rank adapters.

Core relation:

$$W\approx Q(W_0)+(\alpha/r)BA$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 5.4 Optimizer memory

**Main idea.** Optimizer states are needed for trainable adapters, not frozen base weights.

Core relation:

$$M_\mathrm{opt}\propto P_\mathrm{trainable}$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 5.5 Dequantization path

**Main idea.** Many kernels dequantize blocks on the fly for matmul.

Core relation:

$$q,s\rightarrow \hat W$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
## 6. Distillation Basics

This part studies distillation basics as compression math. The useful habit is to separate storage format, dequantized computation, approximation error, and evaluation.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Teacher and student](#6-teacher-and-student) | train a smaller model to match a larger model | $p_T(y\mid x),p_S(y\mid x)$ |
| [Soft targets](#6-soft-targets) | teacher probabilities contain similarity information beyond one-hot labels | $p_T$ |
| [Temperature](#6-temperature) | soften probability distributions | $p_i^{(\tau)}=\mathrm{softmax}(z_i/\tau)$ |
| [KL distillation loss](#6-kl-distillation-loss) | minimize divergence from teacher to student | $L_\mathrm{KD}=\tau^2D_\mathrm{KL}(p_T^{(\tau)}\Vert p_S^{(\tau)})$ |
| [Hard-label mixture](#6-hardlabel-mixture) | combine task loss with distillation loss | $L=\alpha L_\mathrm{CE}+(1-\alpha)L_\mathrm{KD}$ |

### 6.1 Teacher and student

**Main idea.** Train a smaller model to match a larger model.

Core relation:

$$p_T(y\mid x),p_S(y\mid x)$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 6.2 Soft targets

**Main idea.** Teacher probabilities contain similarity information beyond one-hot labels.

Core relation:

$$p_T$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 6.3 Temperature

**Main idea.** Soften probability distributions.

Core relation:

$$p_i^{(\tau)}=\mathrm{softmax}(z_i/\tau)$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 6.4 KL distillation loss

**Main idea.** Minimize divergence from teacher to student.

Core relation:

$$L_\mathrm{KD}=\tau^2D_\mathrm{KL}(p_T^{(\tau)}\Vert p_S^{(\tau)})$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** Distillation trains the student on the teacher's distribution, not only the final answer.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 6.5 Hard-label mixture

**Main idea.** Combine task loss with distillation loss.

Core relation:

$$L=\alpha L_\mathrm{CE}+(1-\alpha)L_\mathrm{KD}$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
## 7. LLM Distillation Types

This part studies llm distillation types as compression math. The useful habit is to separate storage format, dequantized computation, approximation error, and evaluation.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Logit distillation](#7-logit-distillation) | match next-token distributions | $D_\mathrm{KL}(p_T(\cdot\mid x)\Vert p_S(\cdot\mid x))$ |
| [Sequence distillation](#7-sequence-distillation) | train on teacher-generated completions | $y\sim p_T(\cdot\mid x)$ |
| [Feature distillation](#7-feature-distillation) | match hidden states or attention maps | $\Vert h_T-h_S\Vert^2$ |
| [Preference distillation](#7-preference-distillation) | transfer teacher comparisons or judge preferences | $y^+\succ_T y^-$ |
| [Reasoning trace distillation](#7-reasoning-trace-distillation) | train on teacher-produced intermediate reasoning when appropriate | $p_S(r,y\mid x)$ |

### 7.1 Logit distillation

**Main idea.** Match next-token distributions.

Core relation:

$$D_\mathrm{KL}(p_T(\cdot\mid x)\Vert p_S(\cdot\mid x))$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 7.2 Sequence distillation

**Main idea.** Train on teacher-generated completions.

Core relation:

$$y\sim p_T(\cdot\mid x)$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 7.3 Feature distillation

**Main idea.** Match hidden states or attention maps.

Core relation:

$$\Vert h_T-h_S\Vert^2$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 7.4 Preference distillation

**Main idea.** Transfer teacher comparisons or judge preferences.

Core relation:

$$y^+\succ_T y^-$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 7.5 Reasoning trace distillation

**Main idea.** Train on teacher-produced intermediate reasoning when appropriate.

Core relation:

$$p_S(r,y\mid x)$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
## 8. Error and Evaluation

This part studies error and evaluation as compression math. The useful habit is to separate storage format, dequantized computation, approximation error, and evaluation.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Perplexity shift](#8-perplexity-shift) | quantization can be measured by held-out NLL change | $\Delta L=L_\mathrm{quant}-L_\mathrm{base}$ |
| [Task score shift](#8-task-score-shift) | compression should be checked on downstream tasks | $\Delta S=S_\mathrm{comp}-S_\mathrm{base}$ |
| [Calibration shift](#8-calibration-shift) | probabilities may become miscalibrated | $\mathrm{ECE}$ |
| [Layer sensitivity](#8-layer-sensitivity) | some layers or projections tolerate fewer bits poorly | $\Delta L_\ell$ |
| [Outlier channels](#8-outlier-channels) | activation outliers often dominate low-bit error | $\max |x_j|$ |

### 8.1 Perplexity shift

**Main idea.** Quantization can be measured by held-out nll change.

Core relation:

$$\Delta L=L_\mathrm{quant}-L_\mathrm{base}$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 8.2 Task score shift

**Main idea.** Compression should be checked on downstream tasks.

Core relation:

$$\Delta S=S_\mathrm{comp}-S_\mathrm{base}$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 8.3 Calibration shift

**Main idea.** Probabilities may become miscalibrated.

Core relation:

$$\mathrm{ECE}$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 8.4 Layer sensitivity

**Main idea.** Some layers or projections tolerate fewer bits poorly.

Core relation:

$$\Delta L_\ell$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 8.5 Outlier channels

**Main idea.** Activation outliers often dominate low-bit error.

Core relation:

$$\max |x_j|$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
## 9. Deployment Choices

This part studies deployment choices as compression math. The useful habit is to separate storage format, dequantized computation, approximation error, and evaluation.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Weight-only quantization](#9-weightonly-quantization) | reduce weight bandwidth while leaving activations higher precision | $W_q,\ x_\mathrm{fp}$ |
| [Weight-activation quantization](#9-weightactivation-quantization) | quantize both weights and activations for more kernel speed | $W_q,\ x_q$ |
| [KV-cache quantization](#9-kvcache-quantization) | increase context or batch capacity | $b_\mathrm{KV}\downarrow$ |
| [Distill then quantize](#9-distill-then-quantize) | a smaller student can also be quantized | $S\rightarrow Q(S)$ |
| [Hardware format](#9-hardware-format) | INT4, INT8, FP8, and NF4 need matching kernels | $\mathrm{format}\rightarrow\mathrm{kernel}$ |

### 9.1 Weight-only quantization

**Main idea.** Reduce weight bandwidth while leaving activations higher precision.

Core relation:

$$W_q,\ x_\mathrm{fp}$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 9.2 Weight-activation quantization

**Main idea.** Quantize both weights and activations for more kernel speed.

Core relation:

$$W_q,\ x_q$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 9.3 KV-cache quantization

**Main idea.** Increase context or batch capacity.

Core relation:

$$b_\mathrm{KV}\downarrow$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 9.4 Distill then quantize

**Main idea.** A smaller student can also be quantized.

Core relation:

$$S\rightarrow Q(S)$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 9.5 Hardware format

**Main idea.** Int4, int8, fp8, and nf4 need matching kernels.

Core relation:

$$\mathrm{format}\rightarrow\mathrm{kernel}$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
## 10. Debugging Checklist

This part studies debugging checklist as compression math. The useful habit is to separate storage format, dequantized computation, approximation error, and evaluation.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Calibration representativeness](#10-calibration-representativeness) | calibration data should match deployment prompts | $D_\mathrm{cal}\approx D_\mathrm{deploy}$ |
| [Layer-by-layer error](#10-layerbylayer-error) | inspect where compression hurts | $\Vert X(W-\hat W)\Vert$ |
| [Reference comparisons](#10-reference-comparisons) | compare logits before and after compression | $\max|z-\hat z|$ |
| [Latency measurement](#10-latency-measurement) | confirm the chosen format is actually faster | $T_\mathrm{comp}<T_\mathrm{base}$ |
| [Quality gates](#10-quality-gates) | do not ship a compressed model without task and safety checks | $\Delta S$ bounded |

### 10.1 Calibration representativeness

**Main idea.** Calibration data should match deployment prompts.

Core relation:

$$D_\mathrm{cal}\approx D_\mathrm{deploy}$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 10.2 Layer-by-layer error

**Main idea.** Inspect where compression hurts.

Core relation:

$$\Vert X(W-\hat W)\Vert$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 10.3 Reference comparisons

**Main idea.** Compare logits before and after compression.

Core relation:

$$\max|z-\hat z|$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 10.4 Latency measurement

**Main idea.** Confirm the chosen format is actually faster.

Core relation:

$$T_\mathrm{comp}<T_\mathrm{base}$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** A smaller file is not automatically a faster model if kernels do not support the format well.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 10.5 Quality gates

**Main idea.** Do not ship a compressed model without task and safety checks.

Core relation:

$$\Delta S$ bounded$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.

---

## Practice Exercises

1. Quantize and dequantize scalar values with an affine quantizer.
2. Compute symmetric INT4 scale and error.
3. Compare per-tensor and per-channel quantization error.
4. Compute memory reduction from bf16 to 4-bit weights.
5. Sweep clipping ranges and choose the lowest MSE.
6. Compute distillation probabilities at temperature.
7. Compute KL distillation loss for teacher and student distributions.
8. Combine hard-label CE and distillation loss.
9. Estimate QLoRA optimizer-state memory.
10. Write a compression deployment checklist.

## Why This Matters for AI

Compression determines who can run a model, how much serving costs, and which devices can host useful AI locally. The math matters because bad compression can keep a model small but silently damage probabilities, calibration, long-context behavior, or safety behavior.

## Bridge to RAG Math and Retrieval

Compression makes a model cheaper. Retrieval can make a model more informed without changing all its weights. The next section studies embedding retrieval, similarity search, ranking, context packing, and how retrieval changes the conditional distribution used by an LLM.

## References

- Geoffrey Hinton, Oriol Vinyals, and Jeff Dean, "Distilling the Knowledge in a Neural Network", 2015: https://arxiv.org/abs/1503.02531
- Benoit Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference", 2017: https://arxiv.org/abs/1712.05877
- Tim Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale", 2022: https://arxiv.org/abs/2208.07339
- Elias Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers", 2022: https://arxiv.org/abs/2210.17323
- Guangxuan Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models", 2022: https://arxiv.org/abs/2211.10438
- Ji Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration", 2023: https://arxiv.org/abs/2306.00978
- Tim Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs", 2023: https://arxiv.org/abs/2305.14314
