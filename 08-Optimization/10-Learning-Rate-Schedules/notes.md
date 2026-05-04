[Back to Optimization](../README.md) | [Previous: Hyperparameter Optimization](../09-Hyperparameter-Optimization/notes.md) | [Next Chapter: Information Theory](../../09-Information-Theory/README.md)

---

# Learning Rate Schedules

> _"The learning rate is not just a knob. It is the time signal that tells optimization when to explore, when to settle, and when to cash in the compute budget."_

## Overview

A learning rate schedule is a rule for changing the scalar learning rate during training. Vanilla gradient descent uses a constant $\eta$, but modern neural network training almost never does. The learning rate is usually warmed up, held high, decayed, cycled, restarted, or annealed according to a plan tied to steps, epochs, tokens, or compute budget.

This section is the canonical home for external, time-varying learning-rate schedules in the Optimization chapter. The optimizer's internal per-parameter adaptation, such as AdamW's moment-normalized update, belongs to [Adaptive Learning Rate](../07-Adaptive-Learning-Rate/notes.md). Hyperparameter search over the peak learning rate belongs to [Hyperparameter Optimization](../09-Hyperparameter-Optimization/notes.md). Here we focus on the scalar time policy $\eta_t$: how to choose it, why warmup helps, why cosine annealing became a default, why cyclic schedules can regularize, and why warmup-stable-decay is useful for large language model pretraining.

For AI systems, scheduling is operational mathematics. A transformer can diverge in the first few hundred steps without warmup. A cosine schedule can make a fixed-budget run simple to reproduce. A one-cycle policy can train small vision models quickly by using intentionally large learning rates. A warmup-stable-decay schedule can produce multiple useful LLM checkpoints from one continuing training branch. These are not merely engineering tricks; they are ways of controlling stability, gradient noise, implicit regularization, and final model selection.

**Scope note.** This section covers schedules for the scalar learning rate $\eta_t$. It references AdamW, batch-size scaling, weight decay, and optimizer state only when they affect the meaning of $\eta_t$. Full treatments live in their own sections.

## Prerequisites

- **Gradient descent and step-size stability** - $\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla \mathcal{L}(\boldsymbol{\theta}_t)$ - [Gradient Descent](../02-Gradient-Descent/notes.md)
- **Stochastic optimization** - mini-batch gradients, gradient noise, and batch-size effects - [Stochastic Optimization](../05-Stochastic-Optimization/notes.md)
- **Optimization landscape** - curvature, sharpness, flat minima, and training trajectories - [Optimization Landscape](../06-Optimization-Landscape/notes.md)
- **Adaptive optimizers** - Adam, AdamW, moment estimates, effective coordinatewise learning rates - [Adaptive Learning Rate](../07-Adaptive-Learning-Rate/notes.md)
- **Hyperparameter tuning** - choosing peak learning rates, search ranges, and validation protocols - [Hyperparameter Optimization](../09-Hyperparameter-Optimization/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Interactive demonstrations of classical decay, warmup, cosine, cyclic, one-cycle, WSD, implementation pitfalls, and training diagnostics |
| [exercises.ipynb](exercises.ipynb) | 8 graded exercises covering formulas, schedule implementation, resume state, WSD, and AI training diagnosis |

## Learning Objectives

After completing this section, you will:

1. Define a learning-rate schedule as a function $\eta_t = s(t)\eta_{\max}$ over steps, epochs, or tokens
2. Distinguish external scalar scheduling from optimizer-internal adaptation in Adam-like methods
3. Explain why fixed learning rates can diverge, oscillate, stagnate, or waste compute
4. Derive constant, step, exponential, polynomial, inverse-square-root, cosine, cyclic, one-cycle, and WSD schedules
5. Implement warmup and decay schedules with correct step indexing and resume behavior
6. Analyze how warmup interacts with optimizer moments, gradient clipping, and large-batch scaling
7. Compare cosine annealing, warm restarts, cyclic schedules, and one-cycle policies
8. Explain why warmup-stable-decay is attractive for long-running and budget-uncertain LLM pretraining
9. Diagnose scheduler mistakes using learning-rate logs, update norms, gradient norms, and validation curves
10. Choose defensible schedule defaults for transformer pretraining, fine-tuning, small-data training, and large-batch runs

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 Why Fixed Learning Rates Fail](#11-why-fixed-learning-rates-fail)
  - [1.2 Schedules as Training-Time Control Signals](#12-schedules-as-training-time-control-signals)
  - [1.3 Why Schedules Matter for AI and LLMs](#13-why-schedules-matter-for-ai-and-llms)
  - [1.4 Historical Timeline](#14-historical-timeline)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Schedule Function](#21-schedule-function)
  - [2.2 Step, Epoch, Token, and Resume Indexing](#22-step-epoch-token-and-resume-indexing)
  - [2.3 Warmup, Plateau, Decay, Restart, Floor, and Cooldown](#23-warmup-plateau-decay-restart-floor-and-cooldown)
  - [2.4 Effective Update Size](#24-effective-update-size)
  - [2.5 Stability Constraints](#25-stability-constraints)
- [3. Classical Decay Schedules](#3-classical-decay-schedules)
  - [3.1 Constant and Constant-With-Decay Baselines](#31-constant-and-constant-with-decay-baselines)
  - [3.2 Step Decay and Multi-Step Decay](#32-step-decay-and-multi-step-decay)
  - [3.3 Exponential and Polynomial Decay](#33-exponential-and-polynomial-decay)
  - [3.4 Inverse-Square-Root Decay and the Transformer Schedule](#34-inverse-square-root-decay-and-the-transformer-schedule)
  - [3.5 When Monotone Decay Is Justified](#35-when-monotone-decay-is-justified)
- [4. Warmup Theory and Practice](#4-warmup-theory-and-practice)
  - [4.1 Linear Warmup](#41-linear-warmup)
  - [4.2 Warmup, Optimizer Moments, and Gradient Clipping](#42-warmup-optimizer-moments-and-gradient-clipping)
  - [4.3 Large-Batch Scaling](#43-large-batch-scaling)
  - [4.4 Transformer and LLM Warmup Patterns](#44-transformer-and-llm-warmup-patterns)
- [5. Cosine Annealing and Restarts](#5-cosine-annealing-and-restarts)
  - [5.1 Cosine Schedule Formula](#51-cosine-schedule-formula)
  - [5.2 Minimum Learning-Rate Floors](#52-minimum-learning-rate-floors)
  - [5.3 SGDR and Warm Restarts](#53-sgdr-and-warm-restarts)
  - [5.4 Cosine With Warmup in Trainer APIs](#54-cosine-with-warmup-in-trainer-apis)
  - [5.5 Resume and Total-Step Pitfalls](#55-resume-and-total-step-pitfalls)
- [6. Cyclical and One-Cycle Policies](#6-cyclical-and-one-cycle-policies)
  - [6.1 Cyclical Learning Rates](#61-cyclical-learning-rates)
  - [6.2 Triangular, Triangular2, and Exponential-Range Cycles](#62-triangular-triangular2-and-exponential-range-cycles)
  - [6.3 One-Cycle Policy](#63-one-cycle-policy)
  - [6.4 Super-Convergence and Large-LR Regularization](#64-super-convergence-and-large-lr-regularization)
- [7. Warmup-Stable-Decay and Modern LLM Schedules](#7-warmup-stable-decay-and-modern-llm-schedules)
  - [7.1 WSD Definition](#71-wsd-definition)
  - [7.2 Budget Uncertainty and Branching](#72-budget-uncertainty-and-branching)
  - [7.3 River-Valley Loss Landscape Intuition](#73-river-valley-loss-landscape-intuition)
  - [7.4 WSD Versus Cosine](#74-wsd-versus-cosine)
  - [7.5 Practical LLM Schedule Selection](#75-practical-llm-schedule-selection)
- [8. Implementation Patterns](#8-implementation-patterns)
  - [8.1 PyTorch Scheduler Patterns](#81-pytorch-scheduler-patterns)
  - [8.2 Hugging Face Scheduler Patterns](#82-hugging-face-scheduler-patterns)
  - [8.3 Keras and TensorFlow Schedule Objects](#83-keras-and-tensorflow-schedule-objects)
  - [8.4 Per-Step Versus Per-Epoch Stepping](#84-per-step-versus-per-epoch-stepping)
  - [8.5 Logging and Diagnostics](#85-logging-and-diagnostics)
- [9. Common Mistakes](#9-common-mistakes)
- [10. Exercises](#10-exercises)
- [11. Why This Matters for AI (2026 Perspective)](#11-why-this-matters-for-ai-2026-perspective)
- [12. Conceptual Bridge](#12-conceptual-bridge)
- [References](#references)

---

## 1. Intuition

### 1.1 Why Fixed Learning Rates Fail

The simplest training loop uses one constant learning rate:

$$
\boldsymbol{\theta}_{t+1}
=
\boldsymbol{\theta}_t
-
\eta \mathbf{g}_t,
$$

where $\mathbf{g}_t$ is either the full gradient or a mini-batch estimate. This is clean enough for convergence proofs, but it is rarely the best policy across an entire neural network run. Early training has unstable activations, uncalibrated optimizer state, and large changes in representation. Middle training benefits from fast movement along broad valleys. Late training often needs smaller steps to reduce oscillation and improve validation loss.

A fixed $\eta$ must satisfy incompatible demands. If $\eta$ is large, it moves quickly in flat directions but can overshoot sharp directions. If $\eta$ is small, it is stable but may waste most of the compute budget. In stochastic training, a large $\eta$ may behave like useful noise in the middle of training and harmful noise at the end. The schedule resolves this conflict by letting $\eta_t$ change with time.

For a quadratic objective

$$
f(\boldsymbol{\theta}) =
\frac{1}{2}\boldsymbol{\theta}^\top H\boldsymbol{\theta},
$$

gradient descent is stable in the steepest direction only when

$$
0 < \eta < \frac{2}{\lambda_{\max}(H)}.
$$

But the learning rate that is stable near a sharp region can be far smaller than the rate needed to move quickly along a flat region. This is the first mathematical reason schedules matter.

```text
FIXED LEARNING RATE TENSION
========================================================================

  early training:     need caution          -> warmup helps
  middle training:    need fast exploration -> high LR helps
  late training:      need settling         -> decay helps

  one constant eta tries to satisfy all three phases at once

========================================================================
```

**For AI:** Transformers often start with unstable gradient statistics. Large models use AdamW, mixed precision, gradient clipping, and massive batches. A constant peak learning rate from step 1 can cause a loss spike before the optimizer's moment estimates become meaningful. Warmup is a practical way to enter the training trajectory gently.

### 1.2 Schedules as Training-Time Control Signals

A schedule is best viewed as a control signal. The optimizer determines the update direction and coordinate scaling. The schedule determines how much global movement is allowed at each time. For AdamW, the simplified update is

$$
\boldsymbol{\theta}_{t+1}
=
\boldsymbol{\theta}_t
-
\eta_t
\frac{\widehat{\mathbf{m}}_t}{\sqrt{\widehat{\mathbf{v}}_t}+\epsilon}
-
\eta_t\lambda\boldsymbol{\theta}_t.
$$

The same scalar $\eta_t$ multiplies both the Adam-style adaptive step and, in decoupled AdamW, the weight-decay shrinkage. That means scheduling affects optimization speed and regularization strength.

The schedule signal usually has a few phases:

```text
PHASE VIEW OF A TRAINING SCHEDULE
========================================================================

  eta
   ^
   |        plateau or slow anneal
   |       ______________________
   |      /                      \
   |     / warmup                 \ cooldown / decay
   |____/                          \___________________> step

========================================================================
```

The important point is not that every run needs all phases. The point is that each phase has a job. Warmup controls startup stability. Plateau or high learning rate controls fast representation learning. Decay controls final oscillation and checkpoint quality.

### 1.3 Why Schedules Matter for AI and LLMs

In small convex problems, the schedule can be a technical detail. In deep learning, it is often decisive. The same architecture, data, optimizer, and seed can produce different validation curves when the schedule changes. The schedule determines whether training survives the first steps, how much implicit regularization comes from stochastic movement, and where the final iterate lands.

For language models, the schedule also interacts with fixed token budgets. A cosine schedule requires knowing the total number of training steps because the curve is defined over a complete run. If the training budget changes, the schedule changes. Warmup-stable-decay was introduced partly to address this rigidity: keep a main branch at a high stable rate, then decay from chosen checkpoints when a final model is needed.

Specific examples:

- The original Transformer used Adam with linear warmup followed by inverse-square-root decay.
- GPT-3 used Adam, gradient clipping, linear warmup, and cosine decay to a learning-rate floor.
- Many Hugging Face fine-tuning recipes use linear or cosine schedules with warmup.
- PyTorch exposes cosine, cyclic, and one-cycle schedulers as first-class objects.
- Recent LLM pretraining studies analyze warmup-stable-decay as a way to obtain multiple checkpoints from one long run.

**For AI:** The schedule is a reproducibility artifact. A paper that reports `AdamW, lr=3e-4` but omits warmup, decay, total steps, and floor is underspecified. In large-scale training, the schedule is part of the method.

### 1.4 Historical Timeline

```text
LEARNING RATE SCHEDULE TIMELINE
========================================================================

  1950s-1980s  Step-size rules for iterative numerical optimization
  1990s        Robbins-Monro style stochastic approximation conditions
  2012         Deep CNN practice popularizes step decay by validation plateaus
  2015         Leslie Smith proposes cyclical learning rates
  2017         SGDR proposes cosine annealing with warm restarts
  2017         Transformer uses warmup plus inverse-square-root decay
  2017-2018    One-cycle and super-convergence popularize large LR cycles
  2018-2020    BERT/GPT-style training normalizes warmup plus linear/cosine decay
  2020         GPT-3 uses linear warmup and cosine decay to a floor
  2024-2025    WSD schedules gain attention for LLM pretraining budgets
  2026         Schedule choice is treated as part of scaling-law and checkpoint design

========================================================================
```

The history matters because schedule families encode different assumptions. Step decay assumes you know useful drop times. Cosine assumes a fixed horizon. Cyclic schedules assume temporary increases in $\eta_t$ can help exploration or regularization. WSD assumes that a long stable branch can be useful when the final compute budget is uncertain.

---

## 2. Formal Definitions

### 2.1 Schedule Function

A learning-rate schedule is a function

$$
s: \{0,1,\dots,T\} \to \mathbb{R}_{\ge 0}
$$

that produces

$$
\eta_t = \eta_{\max} s(t).
$$

Here $\eta_{\max}$ is the peak or base learning rate, and $s(t)$ is a dimensionless multiplier. Some APIs store $\eta_{\max}$ inside the optimizer's parameter group and only compute $s(t)$. Others compute the full $\eta_t$ directly.

Examples:

| Schedule | Multiplier $s(t)$ | Typical use |
| --- | --- | --- |
| Constant | $1$ | debugging, short fine-tunes |
| Step decay | $\gamma^{\lfloor t/K \rfloor}$ | classical CNN training |
| Exponential | $\gamma^t$ | smooth monotone decay |
| Polynomial | $(1 - t/T)^p$ | BERT-style linear decay when $p=1$ |
| Inverse square root | $t^{-1/2}$ after warmup | original Transformer |
| Cosine | $\frac{1}{2}(1+\cos(\pi t/T))$ | fixed-budget training |
| Cyclic | triangular or sinusoidal between bounds | LR range tests, small models |
| WSD | warmup, stable plateau, final decay | long LLM pretraining |

A schedule is not an optimizer. SGD with cosine decay and AdamW with cosine decay use the same external $\eta_t$ shape but different update directions.

### 2.2 Step, Epoch, Token, and Resume Indexing

The index $t$ must be defined precisely. In small datasets, schedules are often indexed by epoch. In modern deep learning, they are usually indexed by optimizer step. In language model training, it may be more natural to specify phase lengths in tokens and convert to optimizer steps.

If the global batch contains $B_{\text{tok}}$ tokens per optimizer step, then after $t$ optimizer steps,

$$
N_{\text{tok}}(t) = tB_{\text{tok}}.
$$

A warmup of $375$ million tokens with $B_{\text{tok}} = 1$ million tokens per step corresponds to $375$ optimizer steps. With gradient accumulation, the step count increments only when `optimizer.step()` is called, not when each microbatch is processed.

Resume state matters. A scheduler checkpoint must store the current step, base learning rates, phase counters, restart counters, and any API-specific state. If training resumes from step $t=100000$ but the scheduler restarts at $t=0$, the model receives the wrong learning rate and the run is no longer a continuation.

**For AI:** In distributed LLM training, the schedule is tied to global optimizer steps, not device-local microbatches. A common bug is to warm up over microbatches, making the real warmup too short by the gradient accumulation factor.

### 2.3 Warmup, Plateau, Decay, Restart, Floor, and Cooldown

A schedule can be described by named phases:

| Term | Meaning | Mathematical role |
| --- | --- | --- |
| Warmup | Increase from small $\eta_{\text{init}}$ to $\eta_{\max}$ | Reduce startup instability |
| Plateau | Hold $\eta_t \approx \eta_{\max}$ | Move quickly while optimizer is stable |
| Decay | Reduce $\eta_t$ over time | Reduce oscillation and settle |
| Restart | Raise $\eta_t$ after decay | Explore a new basin or improve anytime behavior |
| Floor | Lower bound $\eta_{\min}$ | Avoid completely stopping adaptation |
| Cooldown | Final decay near run end | Improve final checkpoint |

A warmup-cosine schedule with floor can be written as

$$
\eta_t =
\begin{cases}
\eta_{\max}\frac{t}{T_{\mathrm{warm}}}, & 0 \le t < T_{\mathrm{warm}}, \\\\
\eta_{\min}
+
\frac{1}{2}(\eta_{\max}-\eta_{\min})
\left(1+\cos\left(\pi\frac{t-T_{\mathrm{warm}}}{T-T_{\mathrm{warm}}}\right)\right),
& T_{\mathrm{warm}} \le t \le T.
\end{cases}
$$

The phase boundaries are hyperparameters. For a fixed budget, they can be tuned. For budget-uncertain pretraining, a schedule that depends heavily on $T$ can be inconvenient.

### 2.4 Effective Update Size

The scalar learning rate is not the same as the actual parameter displacement. Define the update vector

$$
\Delta \boldsymbol{\theta}_t =
\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}_t.
$$

For SGD,

$$
\lVert \Delta \boldsymbol{\theta}_t \rVert_2
=
\eta_t \lVert \mathbf{g}_t \rVert_2.
$$

For AdamW, the displacement depends on moment-normalized gradients and decoupled weight decay:

$$
\Delta \boldsymbol{\theta}_t
=
-
\eta_t
\frac{\widehat{\mathbf{m}}_t}{\sqrt{\widehat{\mathbf{v}}_t}+\epsilon}
-
\eta_t\lambda\boldsymbol{\theta}_t.
$$

Useful diagnostics include:

$$
\text{update ratio}_t =
\frac{\lVert \Delta \boldsymbol{\theta}_t \rVert_2}
{\lVert \boldsymbol{\theta}_t \rVert_2 + \epsilon},
$$

and

$$
\text{gradient noise scale} \approx
\frac{\operatorname{tr}(\operatorname{Cov}[\mathbf{g}_{\mathcal{B}}])}
{\lVert \mathbb{E}[\mathbf{g}_{\mathcal{B}}] \rVert_2^2}.
$$

The schedule should be interpreted through these diagnostics. A nominal $\eta_t$ can be harmless when gradients are small and dangerous when update ratios spike.

**Worked example: SGD update scale.** Suppose a parameter vector has $\lVert \boldsymbol{\theta}_t\rVert_2 = 100$, gradient norm $\lVert \mathbf{g}_t\rVert_2 = 20$, and learning rate $\eta_t = 10^{-3}$. The nominal SGD update norm is

$$
\lVert \Delta\boldsymbol{\theta}_t\rVert_2
=
10^{-3}\cdot 20
=
0.02.
$$

The update ratio is

$$
\frac{0.02}{100}=2\times 10^{-4}.
$$

If a schedule bug increases $\eta_t$ to $10^{-2}$, the update ratio becomes $2\times 10^{-3}$. That is still a small number in absolute terms, but it is a $10\times$ increase in parameter movement. For a large model, such jumps can be enough to cause loss spikes, clipping, or irreversible movement away from a good region.

**Worked example: AdamW update scale.** Suppose a coordinate has $\widehat{m}_{t,i}=0.01$, $\widehat{v}_{t,i}=10^{-4}$, and $\epsilon=10^{-8}$. The normalized Adam part is approximately

$$
\frac{0.01}{\sqrt{10^{-4}}+10^{-8}} \approx 1.
$$

With $\eta_t=3\times 10^{-4}$, the coordinate update is about $3\times 10^{-4}$ before decoupled weight decay. If the schedule is resumed incorrectly and jumps to $\eta_t=3\times 10^{-3}$, the same optimizer state now produces a $10\times$ larger coordinate movement.

**Non-example: reading $\eta_t$ alone.** A learning rate of $10^{-3}$ may be too large for one model and too small for another. The number only becomes meaningful with optimizer type, gradient statistics, parameter scale, batch size, and precision regime. This is why schedule logs should be read with update-ratio logs, not in isolation.

### 2.5 Stability Constraints

For a convex $L$-smooth function, gradient descent with constant $\eta$ has the standard descent condition when

$$
0 < \eta \le \frac{1}{L}.
$$

For the quadratic case, stability along eigen-direction $i$ requires

$$
\lvert 1-\eta\lambda_i \rvert < 1.
$$

This gives $0 < \eta < 2/\lambda_i$, and the steepest direction imposes $0 < \eta < 2/\lambda_{\max}$. Deep networks are non-convex, stochastic, adaptive, and time-varying, so this bound is not a full training rule. It is still a useful mental model: high curvature and high gradient noise reduce the safe learning-rate region.

Schedules exploit the fact that the safe region can change during training. Warmup assumes the initial safe region is smaller than the later one. Decay assumes the late-stage useful region is smaller than the mid-training exploration region. Cycles assume temporary increases can help escape or regularize without permanently destabilizing the run.

---

## 3. Classical Decay Schedules

### 3.1 Constant and Constant-With-Decay Baselines

The constant schedule is

$$
\eta_t = \eta_0.
$$

It is the best baseline for understanding a new optimizer or objective. If constant $\eta_0$ diverges immediately, the peak learning rate is probably too high or the model has a numerical bug. If constant $\eta_0$ trains but plateaus early, a decay schedule may improve final loss.

A constant schedule with a late manual drop is a primitive form of scheduling:

$$
\eta_t =
\begin{cases}
\eta_0, & t < \tau, \\\\
\gamma \eta_0, & t \ge \tau,
\end{cases}
$$

where $0 < \gamma < 1$. This idea generalizes to step decay.

**Examples.**

- Short fine-tuning run: constant $\eta_t$ can be acceptable if only a few hundred steps are used.
- Debugging a training loop: constant $\eta_t$ makes scheduler mistakes impossible.
- Late manual cooldown: reduce $\eta_t$ by $10\times$ after validation loss stalls.

**Non-examples.**

- Large transformer pretraining from scratch: constant peak LR from step 1 is usually too aggressive.
- Fixed-budget benchmark where final validation loss matters: a constant schedule often leaves noisy late-stage updates.

### 3.2 Step Decay and Multi-Step Decay

Step decay reduces the learning rate by a factor $\gamma$ at fixed intervals:

$$
\eta_t = \eta_0 \gamma^{\lfloor t/K \rfloor},
$$

where $K$ is the step interval and $0 < \gamma < 1$. Multi-step decay uses a list of milestones $\tau_1,\dots,\tau_m$:

$$
\eta_t = \eta_0 \gamma^{\sum_{j=1}^m \mathbf{1}[t \ge \tau_j]}.
$$

Step decay was common in classical CNN training because it is simple and easy to reason about. The discontinuity is also its weakness. The loss curve can show sharp changes at milestones, and the schedule depends heavily on choosing the right drop times.

```text
STEP DECAY
========================================================================

  eta
   |
   |████████
   |        ████████
   |                ████████
   |                        ████████
   +-------------------------------------------------> step
            tau1    tau2    tau3

========================================================================
```

**For AI:** Step decay is still useful for controlled experiments because the phase changes are explicit. It is less common for LLM pretraining because cosine and WSD-style schedules provide smoother or more flexible budget behavior.

### 3.3 Exponential and Polynomial Decay

Exponential decay uses

$$
\eta_t = \eta_0 \gamma^t,
$$

or, with a half-life $h$,

$$
\eta_t = \eta_0 2^{-t/h}.
$$

It is smooth and memoryless: every step multiplies the previous learning rate by the same factor. This can be convenient, but it can decay too early if $h$ is not chosen carefully.

Polynomial decay over a fixed horizon $T$ is

$$
\eta_t =
\eta_{\min}
+
(\eta_0-\eta_{\min})
\left(1-\frac{t}{T}\right)^p,
\quad 0 \le t \le T.
$$

The case $p=1$ is linear decay. Linear warmup plus linear decay became common in BERT-style fine-tuning. Larger $p$ preserves a higher learning rate longer and decays more sharply near the end.

**Examples.**

- Linear decay to zero: common in transformer fine-tuning APIs.
- Polynomial decay with a floor: useful when the final steps should still adapt.
- Exponential decay: useful when schedule design is specified by a half-life.

**Non-examples.**

- Budget-uncertain continuing pretraining: polynomial decay depends on a fixed $T$.
- Long runs where late-stage learning should not vanish: decay-to-zero can stop useful adaptation.

### 3.4 Inverse-Square-Root Decay and the Transformer Schedule

The original Transformer schedule is

$$
\eta_t =
d_{\mathrm{model}}^{-1/2}
\min(t^{-1/2}, tT_{\mathrm{warm}}^{-3/2}).
$$

For $t < T_{\mathrm{warm}}$, the term $tT_{\mathrm{warm}}^{-3/2}$ dominates, so the learning rate increases linearly. For $t > T_{\mathrm{warm}}$, the term $t^{-1/2}$ dominates, so the learning rate decays as an inverse square root.

The peak occurs at

$$
t = T_{\mathrm{warm}},
$$

with

$$
\eta_{\max} =
d_{\mathrm{model}}^{-1/2}T_{\mathrm{warm}}^{-1/2}.
$$

This schedule is historically important because it tied learning-rate scale to model width and included warmup as part of the transformer training recipe. Modern implementations often decouple the peak learning rate from $d_{\mathrm{model}}$ and tune it directly, but the warmup-plus-decay shape remains influential.

**For AI:** Inverse-square-root decay decays quickly after warmup, then slowly. This can be useful for sequence models where long training should continue adapting without rapidly driving $\eta_t$ to zero.

### 3.5 When Monotone Decay Is Justified

Monotone decay is justified when the optimization process should become increasingly local. In stochastic approximation, classical convergence conditions often require

$$
\sum_{t=1}^{\infty} \eta_t = \infty
\quad\text{and}\quad
\sum_{t=1}^{\infty} \eta_t^2 < \infty.
$$

These conditions say that the optimizer keeps moving forever but the accumulated noise remains controlled. The schedule $\eta_t = c/t$ satisfies the second condition but may decay too quickly for deep learning practice. The schedule $\eta_t = c/\sqrt{t}$ decays more slowly and is common in online convex optimization, but it does not satisfy $\sum \eta_t^2 < \infty$.

Deep learning schedules are not pure stochastic approximation schedules. They are finite-horizon training policies. Still, the same intuition applies: high $\eta_t$ enables movement, low $\eta_t$ reduces noise.

---

## 4. Warmup Theory and Practice

### 4.1 Linear Warmup

Linear warmup increases the learning rate from $\eta_{\mathrm{init}}$ to $\eta_{\max}$ over $T_{\mathrm{warm}}$ steps:

$$
\eta_t =
\eta_{\mathrm{init}}
+
(\eta_{\max}-\eta_{\mathrm{init}})
\frac{t}{T_{\mathrm{warm}}},
\quad 0 \le t \le T_{\mathrm{warm}}.
$$

When $\eta_{\mathrm{init}} = 0$, this becomes

$$
\eta_t = \eta_{\max}\frac{t}{T_{\mathrm{warm}}}.
$$

Warmup has three practical interpretations.

First, it avoids immediate large steps before the network's activations and gradients have settled. Second, it gives Adam-like optimizers time to build moment estimates. Third, it is a guard against large-batch training instability, where a peak learning rate chosen for mid-training may be too large at initialization.

```text
LINEAR WARMUP
========================================================================

  eta_max |              ____________________
          |             /
          |            /
          |           /
          |__________/____________________________________ step
                     T_warm

========================================================================
```

**For AI:** Warmup is especially important in transformer training because residual streams, normalization, attention logits, and optimizer moments can produce early instability. Warmup does not fix bad initialization or broken loss scaling, but it gives the training system a smoother start.

### 4.2 Warmup, Optimizer Moments, and Gradient Clipping

AdamW maintains first and second moments:

$$
\mathbf{m}_t = \beta_1\mathbf{m}_{t-1} + (1-\beta_1)\mathbf{g}_t,
$$

$$
\mathbf{v}_t = \beta_2\mathbf{v}_{t-1} + (1-\beta_2)\mathbf{g}_t\odot\mathbf{g}_t.
$$

The bias-corrected estimates are

$$
\widehat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t},
\quad
\widehat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t}.
$$

Even with bias correction, the early update statistics are fragile because the model is changing quickly and the moment averages have little history. Warmup reduces the scalar multiplier on these early updates.

Gradient clipping often appears in the same training recipe:

$$
\mathbf{g}_t^{\mathrm{clip}} =
\mathbf{g}_t
\min\left(1, \frac{c}{\lVert \mathbf{g}_t\rVert_2}\right).
$$

Warmup and clipping solve different problems. Warmup controls scheduled update scale. Clipping controls rare gradient spikes. A model that only trains when clipping activates every step is probably using too large a learning rate, too short a warmup, or an unstable architecture.

### 4.3 Large-Batch Scaling

Large batches reduce gradient noise. A common heuristic is the linear scaling rule:

$$
\eta(B) \approx \eta(B_0)\frac{B}{B_0},
$$

where $B$ is batch size. Another common rule is square-root scaling:

$$
\eta(B) \approx \eta(B_0)\sqrt{\frac{B}{B_0}}.
$$

The right rule depends on optimizer, architecture, loss, batch regime, and training objective. Linear scaling is aggressive; square-root scaling is more conservative. Warmup becomes more important when the peak rate is increased for a larger batch.

The signal-to-noise view is helpful. If batch size increases, the stochastic gradient variance decreases:

$$
\operatorname{Var}[\mathbf{g}_{\mathcal{B}}]
\approx
\frac{1}{B}\Sigma_g.
$$

Lower noise can support a larger stable learning rate, but only up to a critical batch size. Beyond that point, increasing batch size gives diminishing returns.

**For AI:** Large language model training often chooses batch size, learning rate, warmup, gradient clipping, and weight decay together. The schedule is not independent from the parallel training strategy.

### 4.4 Transformer and LLM Warmup Patterns

Transformer schedules commonly use a short warmup relative to total training. A typical pattern is

$$
T_{\mathrm{warm}} \in [0.001T, 0.05T],
$$

but the exact value depends on model size, optimizer, batch, initialization, and data. Fine-tuning runs may use warmup ratios such as $0.03$ or $0.1$. Pretraining runs often specify warmup in tokens or optimizer steps.

The original Transformer used $T_{\mathrm{warm}}=4000$ steps. GPT-3 reported a linear learning-rate warmup over the first $375$ million tokens, then cosine decay to a floor. These examples should not be copied blindly. They are evidence for a general principle: the peak learning rate should not be applied before the training dynamics are ready for it.

Common warmup failures:

- Warmup too short: early loss spikes or NaNs.
- Warmup too long: undertraining because too many tokens are spent at low $\eta_t$.
- Warmup indexed by microbatch: actual optimizer warmup is shorter than intended.
- Warmup restarted after resume: training receives a second unintended ramp.

---

## 5. Cosine Annealing and Restarts

### 5.1 Cosine Schedule Formula

Cosine annealing decays smoothly from $\eta_{\max}$ to $\eta_{\min}$:

$$
\eta_t =
\eta_{\min}
+
\frac{1}{2}(\eta_{\max}-\eta_{\min})
\left(1+\cos\left(\pi\frac{t}{T}\right)\right),
\quad 0 \le t \le T.
$$

The curve starts flat, decays faster in the middle, and ends flat. This avoids abrupt step-decay discontinuities while giving a strong final cooldown.

```text
COSINE ANNEALING
========================================================================

  eta
   |███████████__
   |             \__
   |                \__
   |                   \___
   |                       \____
   +-------------------------------------------------> step
                         T

========================================================================
```

Cosine is convenient for fixed-budget training. If the total number of steps $T$ is known, the schedule is a simple one-line function. Its weakness is exactly that dependency: if $T$ changes, the schedule changes.

### 5.2 Minimum Learning-Rate Floors

Many runs do not decay all the way to zero. A floor $\eta_{\min}$ preserves some adaptation:

$$
\eta_{\min} = \alpha \eta_{\max},
$$

where $\alpha$ might be $0$, $0.01$, or $0.1$ depending on the run. GPT-3 reported cosine decay down to $10\%$ of the original learning rate, then training continued at that floor.

A floor is useful when:

- training continues after the nominal decay window,
- the final model should not freeze completely,
- the run is part of a continual pretraining workflow,
- the model may branch into fine-tuning or extra data.

A floor can hurt when final convergence needs true cooldown. Decaying to zero can produce a strong final checkpoint in a fixed-budget benchmark, but it is less flexible.

### 5.3 SGDR and Warm Restarts

SGDR uses cosine annealing inside cycles, then restarts the learning rate upward. In cycle $i$, with cycle length $T_i$ and local cycle index $T_{\mathrm{cur}}$, the schedule is

$$
\eta_t =
\eta_{\min}^{(i)}
+
\frac{1}{2}(\eta_{\max}^{(i)}-\eta_{\min}^{(i)})
\left(1+\cos\left(\pi\frac{T_{\mathrm{cur}}}{T_i}\right)\right).
$$

At a restart, $T_{\mathrm{cur}}$ resets to $0$, and $\eta_t$ jumps upward. The parameters are not reset. The training state continues, but the learning-rate policy begins a new high-LR phase.

Restarts can improve anytime performance: if training is stopped after different budgets, several points after restarts may be useful. They can also help explore new regions or escape sharp basins. However, restarts are another hyperparameter source: cycle lengths, cycle multipliers, peak decay, and floor all matter.

**For AI:** Warm restarts are more common in vision and small-to-medium experiments than in large LLM pretraining. In LLMs, restarting the learning rate can cause forgetting or instability unless carefully paired with data replay and checkpoint strategy.

### 5.4 Cosine With Warmup in Trainer APIs

The practical schedule used in many libraries is warmup plus cosine:

$$
\eta_t =
\begin{cases}
\eta_{\max}\frac{t}{T_{\mathrm{warm}}}, & t < T_{\mathrm{warm}}, \\\\
\eta_{\min}
+
\frac{1}{2}(\eta_{\max}-\eta_{\min})
\left(1+\cos\left(\pi\frac{t-T_{\mathrm{warm}}}{T-T_{\mathrm{warm}}}\right)\right),
& T_{\mathrm{warm}} \le t \le T.
\end{cases}
$$

Hugging Face exposes variants such as linear, cosine, cosine with restarts, polynomial, constant with warmup, inverse square root, and cosine with minimum learning rate. PyTorch exposes schedulers such as `CosineAnnealingLR`, `CosineAnnealingWarmRestarts`, `CyclicLR`, and `OneCycleLR`. TensorFlow/Keras exposes schedule objects such as `CosineDecay` with optional warmup.

The API name is less important than the exact curve. Always log the actual learning rate values.

### 5.5 Resume and Total-Step Pitfalls

Cosine schedules depend on $T$. Suppose a run is planned for $T=100000$ steps and interrupted at $t=60000$. If it resumes with a new planned horizon $T=200000$, there are two different interpretations:

1. Continue the original schedule and extend at the floor.
2. Recompute the schedule as if the original plan had always been $200000$.

These are not equivalent. The first preserves training history. The second changes the learning rate at step $60000$ relative to the previous run.

For reproducibility, checkpoint the scheduler state. For scientific comparison, report:

- peak learning rate $\eta_{\max}$,
- warmup steps or ratio,
- total decay steps,
- minimum learning rate or floor ratio,
- whether the scheduler steps per batch or per epoch,
- resume behavior,
- optimizer and weight decay.

---

## 6. Cyclical and One-Cycle Policies

### 6.1 Cyclical Learning Rates

Cyclical learning rates vary between lower and upper bounds:

$$
\eta_t \in [\eta_{\min}, \eta_{\max}].
$$

The simplest triangular cycle increases linearly, then decreases linearly. If the cycle length is $2K$, define

$$
r_t = \frac{t \bmod 2K}{K}.
$$

Then a triangular multiplier can be written as

$$
s(t) =
\begin{cases}
r_t, & 0 \le r_t \le 1, \\\\
2-r_t, & 1 < r_t < 2.
\end{cases}
$$

and

$$
\eta_t = \eta_{\min} + s(t)(\eta_{\max}-\eta_{\min}).
$$

Cyclical learning rates were proposed as a practical alternative to manually tuned monotone decay. A learning-rate range test increases $\eta_t$ over a short run and observes where the loss starts decreasing and where it diverges. The chosen cycle bounds come from that diagnostic.

### 6.2 Triangular, Triangular2, and Exponential-Range Cycles

Common CLR variants differ in how cycle amplitude changes:

| Policy | Amplitude behavior | Use |
| --- | --- | --- |
| Triangular | constant amplitude | simple exploration |
| Triangular2 | amplitude halves each cycle | exploration followed by settling |
| Exp-range | amplitude decays exponentially | smooth reduction across cycles |

For triangular2, the amplitude in cycle $c$ is

$$
A_c = \frac{A_0}{2^{c-1}}.
$$

For exp-range,

$$
A_t = A_0 \gamma^t.
$$

These schedules blur the line between cycling and annealing. They repeatedly increase the learning rate, but each increase can become smaller over time.

**For AI:** Cyclic schedules can be helpful when training smaller models quickly or running LR range tests. They are less standard for full LLM pretraining, where the cost of a bad high-LR phase is enormous.

### 6.3 One-Cycle Policy

The one-cycle policy uses one main cycle. It increases the learning rate from a low initial value to a high maximum, then decays to a very low final value:

$$
\eta_{\mathrm{init}}
<
\eta_{\max},
\quad
\eta_{\mathrm{final}}
\ll
\eta_{\mathrm{init}}.
$$

Momentum is often cycled inversely:

$$
\mu_t \uparrow \quad \text{when} \quad \eta_t \downarrow,
$$

and

$$
\mu_t \downarrow \quad \text{when} \quad \eta_t \uparrow.
$$

The intuition is that high learning rates inject strong regularization and encourage broad-basin exploration, while the final low learning rate settles the model.

```text
ONE-CYCLE POLICY
========================================================================

  eta:       low  -> high -> very low
  momentum: high -> low  -> high

  eta
   |          /\
   |         /  \
   |        /    \__________
   |_______/                \________________
   +-------------------------------------------------> step

========================================================================
```

### 6.4 Super-Convergence and Large-LR Regularization

Super-convergence is the empirical observation that some networks can train much faster with a one-cycle schedule and a very large maximum learning rate. The large learning rate acts as regularization: it prevents the model from settling too early and can reduce the need for other regularizers.

This is powerful but not universal. It is most associated with vision models and moderate-size supervised training. It can fail when:

- the maximum learning rate is too high,
- the model is already numerically fragile,
- the data are noisy or small in a way that high-LR regularization worsens,
- the optimizer's moment dynamics are incompatible with the cycle,
- the final cooldown is too short.

**For AI:** One-cycle is a tool to understand and sometimes accelerate training. It should not be treated as the default for every transformer fine-tune. For LLMs, warmup plus cosine or WSD-style schedules are more common starting points.

---

## 7. Warmup-Stable-Decay and Modern LLM Schedules

### 7.1 WSD Definition

Warmup-stable-decay has three phases:

1. Warm up from a small learning rate to $\eta_{\max}$.
2. Keep a stable high learning rate for a long main branch.
3. Decay rapidly near the end or from selected branch points.

A simple WSD schedule is

$$
\eta_t =
\begin{cases}
\eta_{\max}\frac{t}{T_{\mathrm{warm}}}, & 0 \le t < T_{\mathrm{warm}}, \\\\
\eta_{\max}, & T_{\mathrm{warm}} \le t < T_{\mathrm{decay-start}}, \\\\
\eta_{\min}
+
\frac{1}{2}(\eta_{\max}-\eta_{\min})
\left(1+\cos\left(\pi
\frac{t-T_{\mathrm{decay-start}}}{T-T_{\mathrm{decay-start}}}
\right)\right),
& T_{\mathrm{decay-start}} \le t \le T.
\end{cases}
$$

The key difference from ordinary warmup-cosine is the long stable phase. In a standard cosine run, the decay begins immediately after warmup and depends on the total planned horizon. In WSD, the main branch can continue at $\eta_{\max}$ until a final model is needed.

### 7.2 Budget Uncertainty and Branching

Large pretraining runs often face uncertain budgets. More data may arrive. Compute may be extended. Scaling-law experiments may need multiple intermediate checkpoints. A schedule that requires a fixed $T$ from the beginning is inconvenient in this setting.

WSD addresses this by separating the main training branch from decay branches. Let $\boldsymbol{\theta}^{\mathrm{main}}_t$ be checkpoints on the stable branch. For chosen branch steps $D_1,\dots,D_k$, run separate decay phases:

$$
\boldsymbol{\theta}^{(j)}_{D_j} =
\boldsymbol{\theta}^{\mathrm{main}}_{D_j},
$$

then decay $\eta_t$ for a short horizon to produce final checkpoint $\boldsymbol{\theta}^{(j)}_{\mathrm{final}}$.

This lets one long run supply multiple final models:

```text
WSD BRANCHING
========================================================================

  main branch:  warmup -> stable -> stable -> stable -> stable ...
                              |          |          |
                              v          v          v
                           decay      decay      decay
                              |          |          |
                         model A    model B    model C

========================================================================
```

**For AI:** This is a systems-friendly schedule. It supports checkpoint production without restarting pretraining for every budget.

### 7.3 River-Valley Loss Landscape Intuition

Recent WSD analysis uses a river-valley metaphor. Imagine the loss landscape has a long valley with steep sides and a river running along the bottom. A high learning rate makes the iterate oscillate across the valley walls, so measured loss can remain high. But the same high learning rate can move quickly along the valley direction. During the final decay, oscillation shrinks and the iterate moves closer to the river bottom, revealing improved loss.

Mathematically, separate directions into a sharp direction and a flat progress direction:

$$
\boldsymbol{\theta}
=
\begin{bmatrix}
u \\\\
v
\end{bmatrix},
$$

where $u$ is across the valley and $v$ is along the valley. A high $\eta_t$ can cause large oscillations in $u$ while still making progress in $v$. Decay reduces the $u$ oscillation.

This is an interpretation, not a universal theorem about every neural network. Its value is practical: it explains why stable-phase loss may look worse than cosine-phase loss while the final decayed checkpoint can be strong.

### 7.4 WSD Versus Cosine

| Question | Warmup + Cosine | Warmup-Stable-Decay |
| --- | --- | --- |
| Needs fixed total step count? | Yes | Not for main branch |
| LR after warmup | Immediately decays | Holds near peak |
| Good for one fixed-budget run? | Yes | Yes, if decay branch chosen well |
| Good for multiple budget checkpoints? | Requires multiple schedules or retraining | Natural branching |
| Loss during middle phase | Often lower earlier | May stay higher |
| Final checkpoint quality | Strong default | Strong when stable and decay phases are tuned |

Cosine is a strong default when the total budget is known. WSD is attractive when budgets are uncertain, when multiple checkpoints are needed, or when continual pretraining is expected.

### 7.5 Practical LLM Schedule Selection

Reasonable starting points:

| Setting | Schedule starting point | Notes |
| --- | --- | --- |
| Small transformer fine-tune | linear warmup + linear decay | Simple, common in APIs |
| Medium pretraining run | warmup + cosine to floor | Strong fixed-budget default |
| Long LLM pretraining | warmup + cosine or WSD | Choose based on budget certainty |
| Continual pretraining | WSD or re-warm/re-decay with replay | Avoid forgetting from abrupt schedule resets |
| Small vision model | one-cycle or cosine | Try LR range test |
| Debugging | constant LR | Remove scheduler as a variable |

Checklist:

1. Choose optimizer and batch size first.
2. Run a small LR range test if the setting is cheap enough.
3. Pick $\eta_{\max}$ conservatively for the first serious run.
4. Add warmup if training is unstable or uses AdamW with large batches.
5. Use cosine for fixed budget; consider WSD for budget uncertainty.
6. Log $\eta_t$, gradient norm, update norm, and loss at every checkpoint.

**Worked schedule choices.**

| Scenario | Reasonable starting schedule | Why |
| --- | --- | --- |
| New transformer from scratch, known token budget | warmup plus cosine to a small floor | Stable startup and strong fixed-budget cooldown |
| New transformer from scratch, uncertain budget | WSD with branch decays | Main branch can continue while final checkpoints branch off |
| Instruction fine-tuning for a few epochs | warmup plus linear decay | Simple and widely supported by trainer APIs |
| Adapter or LoRA fine-tuning | short warmup plus cosine or constant-with-decay | Fewer trainable parameters often need less elaborate scheduling |
| Small CNN on image classification | one-cycle after LR range test | Large LR can regularize and accelerate |
| Regression model with noisy labels | conservative warmup plus cosine floor | Avoid overfitting from late aggressive movement |
| Continual pretraining on new corpus | WSD or cautious re-warm plus replay | Avoid forgetting and preserve extendability |
| Debugging a new loss function | constant LR | Remove schedule as a moving part |

**Transformer pretraining sketch.** Suppose a model will train for $T=200000$ optimizer steps with AdamW. A conservative first schedule might be:

$$
T_{\mathrm{warm}} = 2000,
\quad
\eta_{\max}=3\times 10^{-4},
\quad
\eta_{\min}=3\times 10^{-5}.
$$

The warmup-cosine schedule is then

$$
\eta_t =
\begin{cases}
3\times 10^{-4}\frac{t}{2000}, & t<2000, \\\\
3\times 10^{-5}
+
\frac{1}{2}(3\times 10^{-4}-3\times 10^{-5})
\left(1+\cos\left(\pi\frac{t-2000}{198000}\right)\right),
& 2000 \le t \le 200000.
\end{cases}
$$

This is a fixed-budget policy. If the run later extends to $400000$ steps, the team must decide whether to keep the old cosine, recompute a new cosine, continue at the floor, or switch to a new branch. That decision changes the training method.

**WSD pretraining sketch.** For a budget-uncertain run, choose

$$
T_{\mathrm{warm}} = 2000,
\quad
\eta_{\max}=3\times 10^{-4},
\quad
\eta_{\min}=3\times 10^{-5},
$$

but keep the main branch at $\eta_{\max}$ after warmup. If final checkpoints are needed at steps $100000$, $200000$, and $300000$, branch a decay of length $20000$ from each checkpoint:

$$
\eta_{D_j+k}
=
\eta_{\min}
+
\frac{1}{2}(\eta_{\max}-\eta_{\min})
\left(1+\cos\left(\pi\frac{k}{20000}\right)\right),
\quad
0 \le k \le 20000.
$$

The main branch is not consumed by those decays. It can keep training while the branch checkpoints cool down.

**Fine-tuning sketch.** For a small supervised fine-tune of $T=1000$ optimizer steps, a common starting point is

$$
T_{\mathrm{warm}} = 100,
\quad
\eta_{\max}=2\times 10^{-5},
\quad
\eta_T = 0.
$$

The model gets a short warmup and then a simple linear cooldown. This is often enough because the base model already has useful representations and the run is short.

**Adapter sketch.** For LoRA or adapter tuning, the trainable parameter count is small and the effective loss landscape can differ from full fine-tuning. A slightly larger $\eta_{\max}$ may be safe, but this is not guaranteed. Compare update ratios on adapter parameters, not only full model norms.

**RLHF or preference optimization sketch.** Schedules interact with KL penalties and reward-model noise. If the learning rate remains high too long, policy updates can move aggressively away from the reference model. A conservative warmup and cooldown are often safer than a high-LR cycle.

**Serving-aware checkpoint sketch.** If the goal is to produce several candidate checkpoints for evaluation, choose a schedule that supports evaluation points. Cosine gives a natural final checkpoint. WSD gives multiple branch-decayed checkpoints. Step decay gives milestone checkpoints but may be less smooth.

**Ablation rule.** When comparing schedules, tune the peak learning rate for each schedule. A one-cycle run with a carefully chosen $\eta_{\max}$ should not be compared to a cosine run with an arbitrary $\eta_{\max}$. Otherwise the experiment confounds schedule shape with learning-rate scale.

**Reporting rule.** A schedule is fully specified only when the units are explicit. Write `2000 optimizer steps`, `375 million tokens`, or `3 epochs`; do not write `warmup=2000` without units.
This one habit prevents most schedule comparisons from becoming ambiguous.

---

## 8. Implementation Patterns

### 8.1 PyTorch Scheduler Patterns

PyTorch schedulers usually wrap an optimizer and mutate parameter-group learning rates. The important choices are:

- whether `scheduler.step()` is called per batch or per epoch,
- whether `optimizer.step()` happens before `scheduler.step()`,
- whether the scheduler state is saved and restored,
- whether the scheduler knows total steps.

For per-step schedules, the training loop should conceptually be

```python
loss.backward()
optimizer.step()
scheduler.step()
optimizer.zero_grad()
```

`CosineAnnealingLR` computes a cosine decay over a maximum number of iterations. `CyclicLR` cycles between two bounds and should be stepped after each batch. `OneCycleLR` also changes after every batch and requires either explicit total steps or epochs and steps per epoch.

**For AI:** In a gradient accumulation loop, call `scheduler.step()` only when `optimizer.step()` is called. Otherwise the schedule advances too quickly.

### 8.2 Hugging Face Scheduler Patterns

Hugging Face exposes schedule names such as:

- `linear`,
- `cosine`,
- `cosine_with_restarts`,
- `polynomial`,
- `constant`,
- `constant_with_warmup`,
- `inverse_sqrt`,
- `reduce_lr_on_plateau`,
- `cosine_with_min_lr`.

Most warmup schedules require

$$
T_{\mathrm{warm}}
$$

and

$$
T_{\mathrm{train}}.
$$

If `max_steps` changes, the decay curve changes. If a run is resumed, the trainer must restore scheduler state and global step correctly.

### 8.3 Keras and TensorFlow Schedule Objects

Keras schedule objects are callables:

$$
\eta_t = \texttt{schedule}(t).
$$

They can be passed directly as the optimizer's learning rate. `CosineDecay` supports optional warmup through a warmup target and warmup steps. This object-oriented style makes serialization explicit, but the same risks remain: wrong step count, wrong warmup length, and missing restore state.

The schedule should be tested independently before training. Plot $\eta_t$ over all planned steps. Check the first step, warmup end, midpoint, final step, and any restart boundaries.

**Minimal scheduler unit tests.** Before running a serious model, create a small table of expected values:

| Checkpoint | What to verify | Common failure |
| --- | --- | --- |
| $t=0$ | LR is $\eta_{\mathrm{init}}$, often $0$ or a small value | Off-by-one starts at peak |
| $t=T_{\mathrm{warm}}$ | LR reaches $\eta_{\max}$ | Warmup denominator wrong |
| $t=T/2$ | LR has the expected middle value | Total step count wrong |
| $t=T$ | LR reaches $\eta_{\min}$ or $0$ | Decay horizon wrong |
| $t>T$ | LR clamps instead of going negative | Missing clipping on progress |
| Resume step | LR equals the uninterrupted run | Scheduler state not restored |
| Restart boundary | LR jumps only when intended | Cycle counter wrong |

These tests look trivial, but they catch the highest-frequency production bugs. A schedule is just a function; test it like one.

### 8.4 Per-Step Versus Per-Epoch Stepping

Per-epoch stepping is common in older image classification recipes:

$$
t = \text{epoch}.
$$

Per-step stepping is common in transformers:

$$
t = \text{optimizer step}.
$$

The difference is not cosmetic. Suppose a dataset has $1000$ optimizer steps per epoch. A warmup of $10$ units means:

- $10$ steps if stepping per batch,
- $10000$ steps if stepping per epoch and interpreting units incorrectly.

Always write the units:

```text
warmup_steps = 2000 optimizer steps
decay_steps = 100000 optimizer steps
not: warmup = 2000
```

### 8.5 Logging and Diagnostics

At minimum, log:

- current learning rate $\eta_t$ for each parameter group,
- training loss,
- validation loss,
- gradient norm $\lVert \mathbf{g}_t\rVert_2$,
- update norm $\lVert \Delta \boldsymbol{\theta}_t\rVert_2$,
- update ratio $\lVert \Delta \boldsymbol{\theta}_t\rVert_2 / \lVert \boldsymbol{\theta}_t\rVert_2$,
- clipping frequency,
- skipped steps from mixed-precision overflow,
- scheduler phase and global step.

Useful interpretations:

| Symptom | Possible schedule issue |
| --- | --- |
| NaNs during warmup | peak LR too high, warmup too short, bad loss scaling |
| Smooth train loss but poor validation | insufficient decay or too little regularization |
| Validation improves only after final decay | high-LR phase is useful but noisy |
| LR reaches zero early | total steps or scheduler stepping is wrong |
| Resume loss spike | scheduler state was not restored |
| Clipping every step | effective update scale is too high |

**Diagnostic protocol.** When a run behaves badly, inspect the schedule in this order:

1. Plot the configured schedule before training. This catches wrong phase lengths without spending compute.
2. Log the actual learning rate from the optimizer, not a separately computed copy. This catches API and parameter-group mismatch.
3. Check whether the scheduler steps once per optimizer update. This catches gradient accumulation bugs.
4. Check resume behavior by comparing $\eta_t$ before and after checkpoint load. This catches accidental restarts.
5. Overlay learning rate with train loss, validation loss, gradient norm, update norm, and clipping frequency. This separates schedule failures from data or architecture failures.

**Case: early loss explosion.** If loss spikes during the first few hundred steps, inspect warmup and peak LR. The likely fixes are:

- lower $\eta_{\max}$,
- increase $T_{\mathrm{warm}}$,
- add or lower gradient clipping threshold,
- inspect mixed-precision overflow,
- check that the scheduler is not stepping per microbatch.

**Case: validation improves only after decay.** This is common with cosine and WSD schedules. It means the high-LR phase may be exploring or moving along a broad valley while the final decay reduces oscillation. Do not judge the schedule only by mid-run validation loss.

**Case: train loss improves but validation gets worse.** A schedule may be too aggressive late in training, or it may lack a sufficient cooldown. Try a lower floor, longer final decay, stronger regularization, or earlier stopping.

**Case: the final LR is zero too early.** This usually means `num_training_steps` was too small or the scheduler was stepped too often. In an LLM run, the usual culprit is confusing microbatch count with optimizer-step count.

**Case: resumed training gets a sudden loss spike.** Compare the learning rate just before saving and just after loading. If the value jumps upward, the scheduler state did not resume correctly. If the value is correct, inspect optimizer moments, loss scaler state, data-loader state, and random seed continuity.

**Case: different parameter groups show different learning rates.** This may be intended, for example a lower LR for embeddings or a higher LR for adapters. It may also be an API mistake. Log every parameter group by name so the schedule can be audited.

**Case: clipping is active for most steps.** Gradient clipping is a safety device, not the main optimizer. If clipping activates constantly, the actual update is determined by the clip threshold as much as by the schedule. Reduce $\eta_{\max}$ or lengthen warmup before trusting the result.

**Case: one-cycle underperforms.** Large learning rates act as regularization. If one-cycle hurts, reduce other regularizers such as dropout or weight decay before abandoning the schedule. If it still hurts, the setting may simply not be one where super-convergence applies.

**Case: WSD stable-phase validation looks worse than cosine.** This can be expected. The stable phase may keep the iterate oscillating at a high apparent loss. Compare final decayed checkpoints, not only main-branch checkpoints.

**Case: a low learning-rate floor prevents final convergence.** Lower the floor or add a short final cooldown. A floor is useful for continued adaptation, but fixed-budget benchmarks often benefit from a more decisive final decay.

**Case: a decay-to-zero schedule blocks continued training.** If more data or compute appears after the schedule reaches zero, the model may need re-warmup or a new schedule. WSD avoids this by preserving a high-LR main branch.

**Case: peak LR transfers poorly between batch sizes.** Revisit batch-size scaling. Linear scaling can overshoot at large batch sizes; square-root scaling is safer but may underuse compute. Treat the peak LR and batch size as a coupled pair.

**Case: peak LR transfers poorly between optimizers.** SGD, AdamW, Adafactor, and Lion-style optimizers do not interpret the same scalar $\eta_t$ identically. Retune $\eta_{\max}$ when the optimizer changes.

---

## 9. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
| --- | --- | --- | --- |
| 1 | Confusing Adam's adaptive update with a schedule | Adam changes coordinatewise scaling, but $\eta_t$ still controls global update scale | Report both optimizer and schedule |
| 2 | Calling `scheduler.step()` every microbatch during gradient accumulation | The schedule advances faster than optimizer updates | Step the scheduler only after `optimizer.step()` |
| 3 | Forgetting to checkpoint scheduler state | Resume changes $\eta_t$ and breaks reproducibility | Save and restore scheduler state with optimizer state |
| 4 | Choosing cosine decay without knowing total steps | Cosine shape depends on $T$ | Use a fixed horizon or choose WSD/constant-with-decay |
| 5 | Warmup too short | Startup updates can diverge or produce loss spikes | Increase $T_{\mathrm{warm}}$ or reduce $\eta_{\max}$ |
| 6 | Warmup too long | Too much compute is spent at low learning rates | Track loss and update ratio during warmup |
| 7 | Decaying to zero unintentionally | Training effectively stops before the run ends | Use a floor $\eta_{\min}$ when continued adaptation matters |
| 8 | Comparing schedules with different peak LRs | The peak LR can dominate the result | Tune or control $\eta_{\max}$ for each schedule |
| 9 | Ignoring weight decay coupling | In AdamW, scheduled $\eta_t$ scales decoupled decay | Treat schedule and weight decay as linked |
| 10 | Using one-cycle with too much other regularization | Large LR already regularizes | Reduce dropout/weight decay when testing super-convergence |
| 11 | Restarting LR during continual pretraining without data replay | High LR can overwrite previous capabilities | Use replay, cautious re-warm, or WSD-style branches |
| 12 | Logging only loss, not $\eta_t$ | Scheduler bugs are invisible | Plot LR, gradient norm, and update norm |

---

## 10. Exercises

1. **Constant versus decayed SGD (*)**  
   For $f(\theta)=\frac{1}{2}\lambda\theta^2$, derive the stability range for constant $\eta$ and compare it with $\eta_t=\eta_0/(1+t)$.

2. **Implement warmup plus cosine (*)**  
   Write a function for
   $$
   \eta_t =
   \begin{cases}
   \eta_{\max}t/T_{\mathrm{warm}}, & t<T_{\mathrm{warm}}, \\\\
   \eta_{\min}+\frac{1}{2}(\eta_{\max}-\eta_{\min})(1+\cos(\pi u)), & t\ge T_{\mathrm{warm}},
   \end{cases}
   $$
   where $u=(t-T_{\mathrm{warm}})/(T-T_{\mathrm{warm}})$.

3. **Transformer inverse-square-root schedule (*)**  
   Compute the peak learning rate for $d_{\mathrm{model}}=512$ and $T_{\mathrm{warm}}=4000$.

4. **Step-indexing bug (**)**
   A run uses gradient accumulation of $8$. The schedule is stepped every microbatch. Derive the actual warmup length in optimizer steps.

5. **Cyclic learning-rate range test (**)**
   Simulate loss values during a linear LR sweep and choose lower/upper cycle bounds from the stable decreasing region.

6. **One-cycle and momentum coupling (**)**
   Implement a two-phase one-cycle schedule and an inverse momentum schedule. Explain why high learning rate pairs with low momentum.

7. **WSD branch design (***)**
   Given a main branch of $500000$ steps, choose three decay branch points and write the schedule for each branch.

8. **LLM training diagnosis (***)**
   Given logs of $\eta_t$, gradient norm, update ratio, clipping frequency, train loss, and validation loss, identify whether the failure is warmup, peak LR, decay, or resume-state related.

---

## 11. Why This Matters for AI (2026 Perspective)

| Concept | AI Impact |
| --- | --- |
| Warmup | Prevents early transformer instability and gives AdamW moments time to stabilize |
| Cosine decay | Strong fixed-budget default for pretraining and fine-tuning |
| LR floor | Allows continued learning after nominal decay, useful in long pretraining |
| Linear decay | Simple default for many BERT-style fine-tuning workflows |
| Inverse-square-root decay | Historical transformer schedule and useful slow-decay pattern |
| Cyclical LR | Practical LR range testing and fast training for smaller models |
| One-cycle | Can create super-convergence by using large LR regularization |
| WSD | Supports budget-uncertain LLM pretraining and multiple checkpoint branches |
| Scheduler state | Essential for reproducible distributed training |
| Update ratio logging | Converts abstract LR values into actual training movement |
| Batch-size coupling | Peak LR must be interpreted together with global batch size |
| Weight decay coupling | AdamW decay strength changes with scheduled $\eta_t$ |

---

## 12. Conceptual Bridge

This section completes the Optimization chapter's training-control arc. Earlier sections built the update direction: convexity tells us when optimization is well behaved, gradient descent gives the first-order method, stochastic optimization explains noisy mini-batches, adaptive learning-rate methods rescale coordinates, and hyperparameter optimization chooses good settings. Learning-rate schedules answer the final time-dependent question: how should the global step scale change over the run?

The next chapter, Information Theory, studies quantities such as entropy, KL divergence, and cross-entropy. Those quantities often define the loss being optimized. For example, LLM pretraining minimizes empirical cross-entropy, while RLHF-style training may include KL penalties. The schedule does not change the objective; it changes how aggressively the optimizer moves through that objective's landscape over time.

```text
CURRICULUM POSITION
========================================================================

  Gradient descent      -> update direction
  Stochastic training   -> noisy gradient estimates
  Adaptive optimizers   -> coordinatewise scaling
  Hyperparameter search -> choose peak settings
  LR schedules          -> time policy for eta_t
        |
        v
  Information theory    -> cross-entropy, KL, perplexity objectives

========================================================================
```

The practical bridge is this: when training a model, do not say "I used AdamW." Say "I used AdamW with peak learning rate $\eta_{\max}$, warmup $T_{\mathrm{warm}}$, schedule shape, decay horizon, floor, batch size, weight decay, clipping, and resume policy." That is the reproducible mathematical description.

## References

1. Robbins, H. and Monro, S. (1951). _A Stochastic Approximation Method_.
2. Vaswani, A. et al. (2017). _Attention Is All You Need_.
3. Loshchilov, I. and Hutter, F. (2017). _SGDR: Stochastic Gradient Descent with Warm Restarts_.
4. Smith, L. N. (2015). _Cyclical Learning Rates for Training Neural Networks_.
5. Smith, L. N. and Topin, N. (2017). _Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates_.
6. Devlin, J. et al. (2019). _BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding_.
7. Brown, T. B. et al. (2020). _Language Models are Few-Shot Learners_.
8. Loshchilov, I. and Hutter, F. (2019). _Decoupled Weight Decay Regularization_.
9. Hu, S. et al. (2024). _Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective_.
10. PyTorch documentation. `torch.optim.lr_scheduler`.
11. Hugging Face Transformers documentation. Optimizers and schedules.
12. TensorFlow/Keras documentation. `tf.keras.optimizers.schedules.CosineDecay`.
