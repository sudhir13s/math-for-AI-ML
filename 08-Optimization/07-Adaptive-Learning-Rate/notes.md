[← Back to Optimization](../README.md) | [Previous: Optimization Landscape ←](../06-Optimization-Landscape/notes.md) | [Next: Regularization Methods →](../08-Regularization-Methods/notes.md)

---

# Adaptive Learning Rate

> _"A good optimizer does not merely follow the gradient; it decides how much to trust each coordinate of it."_

## Overview

Adaptive learning-rate methods modify gradient descent by giving different coordinates, parameters, layers, or blocks different effective step sizes. Instead of applying one scalar learning rate $\eta$ to every coordinate, they maintain running statistics of past gradients and use those statistics to rescale future updates. This is the mathematical idea behind AdaGrad, RMSProp, Adam, AdamW, Adafactor, LARS, and LAMB.

This section is the canonical home for per-parameter adaptation in the Optimization chapter. The stochastic gradients themselves are covered in [Stochastic Optimization](../05-Stochastic-Optimization/notes.md), the geometry of the loss surface is covered in [Optimization Landscape](../06-Optimization-Landscape/notes.md), and external time-varying schedules such as warmup and cosine decay belong to [Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md). Here the focus is the optimizer's internal adaptation: how gradient history changes the update direction and scale.

For modern AI, this topic is not optional. AdamW is the default optimizer for transformers and LLM fine-tuning; Adafactor reduces optimizer-state memory for very large models; LAMB and LARS were designed for large-batch training; and new optimizer proposals are usually judged against AdamW. To use these tools well, you need more than a recipe. You need to understand effective learning rates, moment estimates, bias correction, decoupled weight decay, numerical stabilizers, and optimizer diagnostics.

## Prerequisites

- **Gradient descent and momentum** — update rules, step-size stability, Polyak momentum — [Gradient Descent](../02-Gradient-Descent/notes.md)
- **Second-order and preconditioned methods** — Newton directions, diagonal approximations, natural-gradient intuition — [Second-Order Methods](../03-Second-Order-Methods/notes.md)
- **Stochastic gradients** — mini-batch estimates, gradient noise, batch size scaling — [Stochastic Optimization](../05-Stochastic-Optimization/notes.md)
- **Optimization landscapes** — curvature, sharpness, flatness, and instability diagnostics — [Optimization Landscape](../06-Optimization-Landscape/notes.md)
- **Vector and matrix notation** — elementwise operations, diagonal matrices, norms — [Linear Algebra Basics](../../02-Linear-Algebra-Basics/README.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Interactive demonstrations of AdaGrad, RMSProp, Adam, AdamW, LAMB, Adafactor memory, effective learning rates, and optimizer diagnostics |
| [exercises.ipynb](exercises.ipynb) | 8 graded exercises covering adaptive updates, bias correction, decoupled weight decay, trust ratios, memory cost, and practical diagnosis |

## Learning Objectives

After completing this section, you will:

1. Explain why one global learning rate is inefficient for ill-conditioned and sparse-gradient problems
2. Write adaptive optimizers as diagonal preconditioned gradient methods
3. Derive AdaGrad, RMSProp, Adam, AdamW, AMSGrad, Adafactor, LARS, and LAMB update rules
4. Compute effective per-coordinate learning rates from optimizer state
5. Explain Adam's first-moment, second-moment, and bias-correction terms
6. Distinguish coupled $L_2$ regularization from decoupled weight decay in AdamW
7. Analyze optimizer-state memory costs for billion-parameter models
8. Interpret trust ratios in layerwise adaptive methods such as LARS and LAMB
9. Diagnose training instability using gradient norms, update norms, parameter norms, and effective learning rates
10. Choose reasonable optimizer defaults for transformers, sparse models, reinforcement learning, and large-batch training
11. Separate stable optimizer theory from empirical 2026-era optimizer fashion

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 Why One Global Learning Rate Fails](#11-why-one-global-learning-rate-fails)
  - [1.2 Per-Parameter Learning Rates as Diagonal Preconditioning](#12-per-parameter-learning-rates-as-diagonal-preconditioning)
  - [1.3 Sparse Gradients, Noisy Gradients, and Ill-Conditioning](#13-sparse-gradients-noisy-gradients-and-ill-conditioning)
  - [1.4 Optimizer Lineage](#14-optimizer-lineage)
  - [1.5 Why Adaptive Optimizers Matter for Transformers and LLMs](#15-why-adaptive-optimizers-matter-for-transformers-and-llms)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Base Update Rule and Effective Learning Rate](#21-base-update-rule-and-effective-learning-rate)
  - [2.2 Gradient Accumulators and Second-Moment Estimates](#22-gradient-accumulators-and-second-moment-estimates)
  - [2.3 Diagonal Preconditioners](#23-diagonal-preconditioners)
  - [2.4 Bias Correction and Initialization Effects](#24-bias-correction-and-initialization-effects)
  - [2.5 Optimizer State, Memory Cost, and Numerical Stabilizers](#25-optimizer-state-memory-cost-and-numerical-stabilizers)
- [3. Core Theory I: AdaGrad and RMSProp](#3-core-theory-i-adagrad-and-rmsprop)
  - [3.1 AdaGrad Derivation and Sparse-Feature Behavior](#31-adagrad-derivation-and-sparse-feature-behavior)
  - [3.2 AdaGrad Regret Intuition and Diminishing Steps](#32-adagrad-regret-intuition-and-diminishing-steps)
  - [3.3 RMSProp as Exponential Forgetting](#33-rmsprop-as-exponential-forgetting)
  - [3.4 Choosing Epsilon and Accumulator Decay](#34-choosing-epsilon-and-accumulator-decay)
  - [3.5 AdaDelta and Practical Historical Variants](#35-adadelta-and-practical-historical-variants)
- [4. Core Theory II: Adam Family](#4-core-theory-ii-adam-family)
  - [4.1 Adam as Momentum plus RMSProp](#41-adam-as-momentum-plus-rmsprop)
  - [4.2 First Moment, Second Moment, and Bias Correction](#42-first-moment-second-moment-and-bias-correction)
  - [4.3 Effective Step Size and Update Normalization](#43-effective-step-size-and-update-normalization)
  - [4.4 Adam Convergence Caveats and AMSGrad](#44-adam-convergence-caveats-and-amsgrad)
  - [4.5 AdamW and Decoupled Weight Decay](#45-adamw-and-decoupled-weight-decay)
- [5. Core Theory III: Layerwise and Memory-Efficient Adaptation](#5-core-theory-iii-layerwise-and-memory-efficient-adaptation)
  - [5.1 LARS and LAMB for Large-Batch Training](#51-lars-and-lamb-for-large-batch-training)
  - [5.2 Trust Ratio and Layerwise Update Scaling](#52-trust-ratio-and-layerwise-update-scaling)
  - [5.3 Adafactor and Factored Second-Moment State](#53-adafactor-and-factored-second-moment-state)
  - [5.4 Optimizer State Memory in Large Models](#54-optimizer-state-memory-in-large-models)
  - [5.5 When Layerwise Adaptation Helps or Hurts](#55-when-layerwise-adaptation-helps-or-hurts)
- [6. Advanced Topics](#6-advanced-topics)
  - [6.1 Adaptive Methods as Diagonal Natural-Gradient Approximations](#61-adaptive-methods-as-diagonal-natural-gradient-approximations)
  - [6.2 Relationship to Curvature and Preconditioning](#62-relationship-to-curvature-and-preconditioning)
  - [6.3 Sign-Based and Low-Memory Optimizers](#63-sign-based-and-low-memory-optimizers)
  - [6.4 Structured Preconditioners](#64-structured-preconditioners)
  - [6.5 Stability Diagnostics](#65-stability-diagnostics)
- [7. Applications in Machine Learning](#7-applications-in-machine-learning)
  - [7.1 AdamW for Transformer Pretraining and Fine-Tuning](#71-adamw-for-transformer-pretraining-and-fine-tuning)
  - [7.2 Sparse Embeddings and Recommendation Models](#72-sparse-embeddings-and-recommendation-models)
  - [7.3 Reinforcement Learning and Non-Stationary Objectives](#73-reinforcement-learning-and-non-stationary-objectives)
  - [7.4 Large-Batch Training with LAMB](#74-large-batch-training-with-lamb)
  - [7.5 Practical Optimizer Selection Checklist](#75-practical-optimizer-selection-checklist)
- [8. Common Mistakes](#8-common-mistakes)
- [9. Exercises](#9-exercises)
- [10. Why This Matters for AI (2026 Perspective)](#10-why-this-matters-for-ai-2026-perspective)
- [11. Conceptual Bridge](#11-conceptual-bridge)
- [References](#references)

---

## 1. Intuition

### 1.1 Why One Global Learning Rate Fails

Vanilla gradient descent uses one scalar learning rate:

$$
\boldsymbol{\theta}_{t+1}
= \boldsymbol{\theta}_t - \eta \mathbf{g}_t,
$$

where $\mathbf{g}_t = \nabla \mathcal{L}(\boldsymbol{\theta}_t)$ or a mini-batch estimate of it. This treats every coordinate of $\boldsymbol{\theta}$ as if it had the same scale, curvature, noise level, and frequency of update. Real neural networks do not behave that way.

In an embedding table, some token vectors may receive gradients constantly while rare-token vectors receive gradients only occasionally. In a transformer, attention projections, MLP matrices, normalization parameters, and output heads can have very different gradient statistics. In an ill-conditioned quadratic, one coordinate may be steep and another shallow. A single $\eta$ must be small enough not to explode in the steep coordinate, but that makes progress painfully slow in the shallow coordinate.

The core motivation for adaptive learning rates is therefore simple:

$$
\text{different coordinates need different effective step sizes.}
$$

```text
ONE ETA VS MANY EFFECTIVE ETAS
════════════════════════════════════════════════════════════════════════

  Single global step:
      theta_{t+1} = theta_t - eta * g_t

  Adaptive diagonal step:
      theta_{t+1} = theta_t - eta * D_t^{-1/2} * g_t

  coordinate with large historical gradients  -> smaller effective LR
  coordinate with small historical gradients  -> larger effective LR
  sparse coordinate updated rarely            -> not drowned by common features

════════════════════════════════════════════════════════════════════════
```

**For AI:** This is why AdamW often trains transformers more easily than SGD with momentum. It automatically rescales parameters whose gradients live on different numerical scales. That does not make it magic, but it removes a large amount of manual learning-rate tuning.

### 1.2 Per-Parameter Learning Rates as Diagonal Preconditioning

Adaptive methods can be understood as diagonal preconditioned gradient descent. Instead of moving in direction $-\mathbf{g}_t$, they move in direction

$$
-P_t \mathbf{g}_t,
$$

where $P_t$ is usually a positive diagonal matrix:

$$
P_t = \operatorname{diag}(p_{t,1},\dots,p_{t,d}).
$$

For Adam-like methods, $p_{t,i}$ is roughly the inverse square root of a running average of squared gradients:

$$
p_{t,i} \approx \frac{1}{\sqrt{v_{t,i}}+\epsilon}.
$$

The update becomes

$$
\theta_{t+1,i}
= \theta_{t,i} - \eta \frac{m_{t,i}}{\sqrt{v_{t,i}}+\epsilon}.
$$

Here $m_t$ is a first-moment estimate, $v_t$ is a second-moment estimate, and $\epsilon$ is a numerical stabilizer. The term

$$
\eta_{t,i}^{\mathrm{eff}} = \frac{\eta}{\sqrt{v_{t,i}}+\epsilon}
$$

is the effective learning rate for coordinate $i$ before momentum and bias correction.

This diagonal view also explains the connection to second-order methods. Newton's method uses $H^{-1}$, the inverse Hessian. Natural gradient uses a Fisher inverse. Adaptive methods use something much cheaper: a diagonal statistic based on gradient history. They are not exact second-order methods, but they borrow the same idea: rescale the gradient by local geometry.

### 1.3 Sparse Gradients, Noisy Gradients, and Ill-Conditioning

Adaptive methods are especially useful in three regimes.

First, **sparse gradients**. In language models and recommendation systems, some parameters are updated rarely. AdaGrad was designed to give rare but informative coordinates larger relative updates because their accumulated squared-gradient history grows slowly.

Second, **noisy gradients**. In reinforcement learning, generative modeling, and small-batch training, gradient estimates can be highly variable. Running moment estimates smooth the update and reduce sensitivity to any single noisy mini-batch.

Third, **ill-conditioning**. When different directions have different curvature, vanilla gradient descent zigzags. A diagonal preconditioner cannot fix all curvature, because it ignores correlations between coordinates, but it can reduce scale mismatch when the main problem is coordinatewise.

Examples:

- Sparse word embeddings: AdaGrad or Adam can help rare tokens move.
- Transformer blocks: AdamW stabilizes heterogeneous parameter groups.
- Large-batch BERT-style training: LAMB scales layer updates by a trust ratio.
- Memory-constrained training: Adafactor approximates the second moment using factored state.

Non-examples:

- If all coordinates have identical scale and curvature, adaptivity may add little.
- If the main curvature problem is strong off-diagonal correlation, diagonal methods may be insufficient.

### 1.4 Optimizer Lineage

The family tree is easier to understand if each method answers one specific complaint about the previous method.

```text
FROM SGD TO ADAMW
════════════════════════════════════════════════════════════════════════

  SGD
    problem: one global step size
       |
       v
  AdaGrad
    accumulates squared gradients forever
    good for sparsity, but steps can decay too much
       |
       v
  RMSProp
    uses exponential moving average of squared gradients
    fixes AdaGrad's never-forget behavior
       |
       v
  Adam
    RMSProp-style second moment + momentum first moment + bias correction
       |
       v
  AdamW
    Adam with decoupled weight decay
    default for many transformer and LLM workloads
       |
       v
  LAMB / Adafactor / newer variants
    layerwise scaling, memory efficiency, and workload-specific refinements

════════════════════════════════════════════════════════════════════════
```

Historically, AdaGrad came from online convex optimization, RMSProp from neural-network training practice, Adam from stochastic optimization with moment estimates, AdamW from the observation that $L_2$ regularization and weight decay are not equivalent for adaptive methods, and LAMB from large-batch BERT training.

### 1.5 Why Adaptive Optimizers Matter for Transformers and LLMs

Transformers are optimizer-sensitive for several reasons:

- their parameter groups have very different gradient scales
- token embeddings create sparse and frequency-skewed updates
- attention and MLP blocks can have different curvature statistics
- normalization parameters and biases usually should not receive the same weight decay as large weight matrices
- mixed precision makes numerical stability more important
- large-batch training changes gradient noise and update norms

AdamW works well because it combines three useful behaviors:

1. momentum through the first moment $m_t$
2. coordinatewise scaling through the second moment $v_t$
3. decoupled weight decay that avoids mixing regularization with adaptive preconditioning

The practical lesson is not "always use AdamW." The practical lesson is: AdamW is the baseline you must understand before evaluating alternatives.

---

## 2. Formal Definitions

### 2.1 Base Update Rule and Effective Learning Rate

Let $\boldsymbol{\theta}_t \in \mathbb{R}^d$ be parameters and $\mathbf{g}_t$ be a stochastic gradient at step $t$. A coordinatewise adaptive method often has the form

$$
\boldsymbol{\theta}_{t+1}
= \boldsymbol{\theta}_t - \eta \, \mathbf{s}_t,
$$

where the update direction $\mathbf{s}_t$ is not simply $\mathbf{g}_t$. For diagonal adaptive methods,

$$
\mathbf{s}_t = A_t^{-1/2}\mathbf{u}_t,
$$

where $A_t$ is diagonal and nonnegative, and $\mathbf{u}_t$ is usually a raw gradient or a momentum-smoothed gradient.

The **effective learning rate** for coordinate $i$ is the scalar multiplying the update signal:

$$
\eta_{t,i}^{\mathrm{eff}}
= \frac{\eta}{\sqrt{a_{t,i}}+\epsilon}.
$$

This quantity is diagnostic gold. If $\eta_{t,i}^{\mathrm{eff}}$ collapses to nearly zero, the parameter may stop learning. If it becomes too large because $a_{t,i}$ is tiny, updates can explode.

### 2.2 Gradient Accumulators and Second-Moment Estimates

Adaptive optimizers rely on gradient history. The two most common state variables are:

$$
m_t \approx \mathbb{E}[\mathbf{g}_t],
\qquad
v_t \approx \mathbb{E}[\mathbf{g}_t \odot \mathbf{g}_t].
$$

Here $\odot$ denotes elementwise multiplication. $m_t$ tracks direction; $v_t$ tracks magnitude. For Adam,

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)\mathbf{g}_t,
$$

and

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)(\mathbf{g}_t \odot \mathbf{g}_t).
$$

The hyperparameters $\beta_1$ and $\beta_2$ are decay rates. Large values mean long memory. Common defaults are $\beta_1=0.9$ and $\beta_2=0.999$, though LLM training often uses $\beta_2$ values such as $0.95$ or $0.98$ depending on scale, batch, and schedule.

### 2.3 Diagonal Preconditioners

A **preconditioner** changes the geometry of gradient descent. In ordinary gradient descent,

$$
\Delta \boldsymbol{\theta}_t = -\eta \mathbf{g}_t.
$$

With a preconditioner $P_t$,

$$
\Delta \boldsymbol{\theta}_t = -\eta P_t \mathbf{g}_t.
$$

If $P_t$ is diagonal, this is coordinatewise rescaling:

$$
\Delta \theta_{t,i}
= -\eta p_{t,i} g_{t,i}.
$$

AdaGrad uses a cumulative diagonal preconditioner. RMSProp and Adam use exponential moving averages. Adafactor uses a factored approximation to avoid storing a full $v_t$ tensor. Shampoo uses richer matrix preconditioners, but that belongs mostly to the structured preconditioning bridge rather than the core diagonal family.

### 2.4 Bias Correction and Initialization Effects

Moment estimates initialized at zero are biased toward zero in early steps. If

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)\mathbf{g}_t,
\qquad
m_0=\mathbf{0},
$$

and the gradient is constant $\mathbf{g}$, then

$$
m_t = (1-\beta_1^t)\mathbf{g}.
$$

So Adam uses

$$
\widehat{m}_t = \frac{m_t}{1-\beta_1^t},
\qquad
\widehat{v}_t = \frac{v_t}{1-\beta_2^t}.
$$

The corrected Adam update is

$$
\boldsymbol{\theta}_{t+1}
= \boldsymbol{\theta}_t
- \eta \frac{\widehat{m}_t}{\sqrt{\widehat{v}_t}+\epsilon}.
$$

Without bias correction, the early steps can be mis-scaled. This is especially relevant during warmup, where optimizer state and schedule both change quickly.

### 2.5 Optimizer State, Memory Cost, and Numerical Stabilizers

Adaptive methods are not free. For a parameter vector with $d$ scalar parameters:

| Optimizer | State per parameter | Main state |
| --- | ---: | --- |
| SGD | $0$ | none |
| SGD + momentum | $1$ | velocity |
| AdaGrad | $1$ | cumulative squared gradients |
| RMSProp | $1$ | EMA squared gradients |
| Adam / AdamW | $2$ | first and second moments |
| Adafactor | sublinear for matrices | factored second moment |

For a billion parameters in FP32, one state tensor costs about $4$ GB. AdamW with two FP32 state tensors costs about $8$ GB just for optimizer state, before parameters, gradients, activations, and distributed optimizer sharding.

The stabilizer $\epsilon$ also matters:

$$
\frac{1}{\sqrt{v_{t,i}}+\epsilon}.
$$

If $\epsilon$ is too small, coordinates with tiny $v_{t,i}$ can take huge steps. If it is too large, adaptivity is damped. Framework defaults are useful starting points, not laws.

---

## 3. Core Theory I: AdaGrad and RMSProp

### 3.1 AdaGrad Derivation and Sparse-Feature Behavior

AdaGrad accumulates squared gradients:

$$
\mathbf{r}_t = \mathbf{r}_{t-1} + \mathbf{g}_t \odot \mathbf{g}_t,
\qquad
\mathbf{r}_0 = \mathbf{0}.
$$

Then it updates

$$
\boldsymbol{\theta}_{t+1}
= \boldsymbol{\theta}_t
- \eta \frac{\mathbf{g}_t}{\sqrt{\mathbf{r}_t}+\epsilon}.
$$

For coordinate $i$,

$$
\eta_{t,i}^{\mathrm{eff}}
= \frac{\eta}{\sqrt{\sum_{\tau=1}^t g_{\tau,i}^2}+\epsilon}.
$$

Frequent coordinates accumulate large denominators and receive smaller future steps. Rare coordinates accumulate slowly and keep larger effective learning rates. This is exactly what you want for sparse features.

**Examples.**

1. Text classification with rare words: rare features should not be ignored merely because they update less often.
2. Recommendation embeddings: infrequent users or items need meaningful updates when observed.
3. Online learning: the feature distribution may be long-tailed and nonstationary.

**Non-examples.**

1. Dense homogeneous gradients: AdaGrad's sparsity advantage may not matter.
2. Long deep-network training: the denominator can grow forever, making effective steps too small.

### 3.2 AdaGrad Regret Intuition and Diminishing Steps

AdaGrad has strong roots in online convex optimization. The regret analysis shows that coordinatewise adaptation can improve bounds when gradients are sparse or unevenly distributed. The informal idea is:

$$
\sum_{t=1}^T g_{t,i}^2
$$

measures how much coordinate $i$ has mattered historically. Coordinates that matter often get cautious updates; coordinates that matter rarely retain larger steps.

The same property is also AdaGrad's weakness. Because $\mathbf{r}_t$ only grows,

$$
\eta_{t,i}^{\mathrm{eff}} \to 0
$$

if coordinate $i$ receives persistent nonzero gradients. This can prematurely stop learning in long nonconvex training runs.

### 3.3 RMSProp as Exponential Forgetting

RMSProp replaces AdaGrad's cumulative sum with exponential forgetting:

$$
\mathbf{r}_t
= \rho \mathbf{r}_{t-1}
+ (1-\rho)(\mathbf{g}_t \odot \mathbf{g}_t).
$$

The update is

$$
\boldsymbol{\theta}_{t+1}
= \boldsymbol{\theta}_t
- \eta \frac{\mathbf{g}_t}{\sqrt{\mathbf{r}_t}+\epsilon}.
$$

Now the optimizer remembers recent gradient scale but eventually forgets old scale. This is more suitable for nonstationary objectives and long training runs.

RMSProp is historically important because Adam is essentially RMSProp plus momentum plus bias correction.

### 3.4 Choosing Epsilon and Accumulator Decay

The decay $\rho$ controls the time horizon. The effective memory length is roughly

$$
\frac{1}{1-\rho}.
$$

So $\rho=0.9$ remembers roughly $10$ steps, while $\rho=0.99$ remembers roughly $100$ steps. Adam's $\beta_2=0.999$ remembers even longer.

The stabilizer $\epsilon$ has two roles:

- prevent division by zero
- limit the maximum effective learning rate

For AI systems, $\epsilon$ can interact with mixed precision. A value that is harmless in FP32 may behave differently with BF16 or FP16 state if implementation details change. Good frameworks usually keep optimizer state in FP32 for this reason.

### 3.5 AdaDelta and Practical Historical Variants

AdaDelta was designed to reduce manual learning-rate sensitivity by tracking both squared gradients and squared updates. Its practical influence is historical: it helped establish the idea that adaptive update magnitudes could be controlled by running statistics.

The modern lesson is not that AdaDelta should be your default optimizer. The lesson is that every adaptive method chooses:

- what statistic to track
- how long to remember it
- whether to track gradients, updates, or both
- how to stabilize division
- how much state memory to pay

Once you see optimizers through this lens, new variants become less mysterious.

---

## 4. Core Theory II: Adam Family

### 4.1 Adam as Momentum plus RMSProp

Adam tracks two exponential moving averages:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)\mathbf{g}_t,
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)(\mathbf{g}_t \odot \mathbf{g}_t).
$$

After bias correction,

$$
\widehat{m}_t = \frac{m_t}{1-\beta_1^t},
\qquad
\widehat{v}_t = \frac{v_t}{1-\beta_2^t}.
$$

The update is

$$
\boldsymbol{\theta}_{t+1}
= \boldsymbol{\theta}_t
- \eta \frac{\widehat{m}_t}{\sqrt{\widehat{v}_t}+\epsilon}.
$$

Adam's first moment behaves like momentum. Its second moment behaves like RMSProp. Bias correction fixes the fact that both estimates start at zero.

### 4.2 First Moment, Second Moment, and Bias Correction

The first moment $m_t$ smooths direction. It reduces update jitter and accumulates persistent descent directions. The second moment $v_t$ smooths scale. It prevents coordinates with large historical gradients from dominating the update.

For a constant scalar gradient $g$, the uncorrected moments are:

$$
m_t = (1-\beta_1^t)g,
\qquad
v_t = (1-\beta_2^t)g^2.
$$

Bias correction recovers $g$ and $g^2$ exactly in this idealized case:

$$
\widehat{m}_t=g,
\qquad
\widehat{v}_t=g^2.
$$

That explains why bias correction matters most early in training.

### 4.3 Effective Step Size and Update Normalization

Ignoring $\epsilon$, if the gradient is constant and bias correction is applied, Adam's scalar update approximately becomes

$$
-\eta \frac{g}{\sqrt{g^2}}
= -\eta \operatorname{sign}(g).
$$

This sign-like behavior is one reason Adam is robust to gradient scale, but it is also a caution. Adam does not simply take a smaller step when gradients are small. It normalizes by estimated magnitude.

In vector form, Adam can produce update directions that differ substantially from SGD. This is useful when scale mismatch is real, but it can be harmful when the second-moment estimate is noisy or when normalization amplifies unimportant coordinates.

### 4.4 Adam Convergence Caveats and AMSGrad

Adam works extremely well in practice, but its original convergence story is subtler than the headline suggests. Reddi, Kale, and Kumar showed examples where Adam can fail to converge under certain online convex settings because the adaptive denominator can change in problematic ways.

AMSGrad modifies Adam by keeping a nondecreasing second-moment maximum:

$$
\widehat{v}_t^{\max}
= \max(\widehat{v}_{t-1}^{\max}, \widehat{v}_t),
$$

and updates with $\widehat{v}_t^{\max}$ in the denominator. This prevents the effective learning rate from increasing in a way that breaks the proof.

In deep learning practice, AMSGrad is not universally better than Adam. Its main value in this curriculum is conceptual: adaptive methods need stability conditions, not just appealing formulas.

### 4.5 AdamW and Decoupled Weight Decay

Standard $L_2$ regularization adds a penalty to the loss:

$$
\mathcal{L}_{\lambda}(\boldsymbol{\theta})
= \mathcal{L}(\boldsymbol{\theta})
+ \frac{\lambda}{2}\lVert \boldsymbol{\theta} \rVert_2^2.
$$

The gradient becomes

$$
\nabla \mathcal{L}_{\lambda}(\boldsymbol{\theta})
= \nabla \mathcal{L}(\boldsymbol{\theta}) + \lambda \boldsymbol{\theta}.
$$

For vanilla SGD, this is equivalent to weight decay under common conventions. For Adam, it is not equivalent because $\lambda \boldsymbol{\theta}$ enters the adaptive preconditioner. Large and small coordinates get decayed through the same nonlinear denominator used for gradients.

AdamW decouples decay from the adaptive gradient step:

$$
\boldsymbol{\theta}_{t+1}
= (1-\eta\lambda)\boldsymbol{\theta}_t
- \eta \frac{\widehat{m}_t}{\sqrt{\widehat{v}_t}+\epsilon}.
$$

This separation is one of the most important optimizer details in modern transformer training.

> **Forward link:** The broader regularization interpretation of weight decay belongs to [Regularization Methods](../08-Regularization-Methods/notes.md). Here the focus is the optimizer-level distinction between coupled and decoupled decay.

---

## 5. Core Theory III: Layerwise and Memory-Efficient Adaptation

### 5.1 LARS and LAMB for Large-Batch Training

Large-batch training changes the scale of optimization. Gradient noise decreases, throughput improves, and learning-rate scaling becomes tempting, but very large global updates can destabilize some layers.

LARS and LAMB introduce layerwise trust ratios. For a layer parameter vector $\boldsymbol{\theta}^{[\ell]}$ and update vector $\mathbf{u}^{[\ell]}$, the trust ratio is roughly

$$
r^{[\ell]}
= \frac{\lVert \boldsymbol{\theta}^{[\ell]} \rVert_2}
{\lVert \mathbf{u}^{[\ell]} \rVert_2 + \epsilon}.
$$

The update becomes

$$
\boldsymbol{\theta}_{t+1}^{[\ell]}
= \boldsymbol{\theta}_t^{[\ell]}
- \eta r^{[\ell]}\mathbf{u}^{[\ell]}.
$$

LAMB combines this layerwise scaling with Adam-style moments. It was introduced for large-batch BERT training, where layerwise update normalization helped maintain stable progress at huge batch sizes.

### 5.2 Trust Ratio and Layerwise Update Scaling

The trust ratio compares update norm to parameter norm. The dimensionless ratio

$$
\frac{\lVert \Delta \boldsymbol{\theta}^{[\ell]} \rVert_2}
{\lVert \boldsymbol{\theta}^{[\ell]} \rVert_2}
$$

is often more informative than raw update norm. If this ratio is too high, the layer is being rewritten too aggressively. If it is too low, the layer is barely moving.

**For AI:** Monitoring this ratio is useful in large-scale training. Many training failures look mysterious in loss curves but obvious in update-to-parameter norm diagnostics.

### 5.3 Adafactor and Factored Second-Moment State

Adam stores a second-moment tensor $v_t$ with the same shape as the parameter. For a matrix $W \in \mathbb{R}^{m \times n}$, that costs $mn$ state entries.

Adafactor approximates the second-moment matrix using row and column statistics. Instead of storing all of $V \in \mathbb{R}^{m \times n}$, it stores

$$
\mathbf{r} \in \mathbb{R}^m,
\qquad
\mathbf{c} \in \mathbb{R}^n.
$$

A simple factored reconstruction has the flavor

$$
\widehat{V}_{ij}
\approx \frac{r_i c_j}{\frac{1}{m}\sum_{k=1}^m r_k}.
$$

This reduces memory from $O(mn)$ to $O(m+n)$ for matrix-shaped parameters. That is a major difference for large models.

### 5.4 Optimizer State Memory in Large Models

For $N$ parameters, FP32 AdamW state costs approximately:

$$
2 \times 4N \text{ bytes}
$$

for $m_t$ and $v_t$. With FP32 master weights and gradients, the full training memory picture can be much larger. Distributed training systems therefore use sharding, offloading, lower-precision state, or memory-efficient optimizers.

| Model scale | AdamW state only, FP32 | Practical implication |
| ---: | ---: | --- |
| $10^8$ params | about $0.8$ GB | easy on a modern accelerator |
| $10^9$ params | about $8$ GB | optimizer state is a major budget item |
| $10^{10}$ params | about $80$ GB | requires sharding/offload/factoring |
| $10^{11}$ params | about $800$ GB | optimizer engineering dominates |

### 5.5 When Layerwise Adaptation Helps or Hurts

Layerwise adaptation helps when:

- layers have very different parameter norms
- large-batch training reduces gradient noise enough to permit bigger global steps
- update-to-parameter ratios are unstable across layers
- the model is large enough that layerwise scale mismatch matters

It can hurt when:

- parameter norms are tiny or not meaningful
- normalization parameters or biases are treated like large matrices
- the trust ratio masks bad gradients
- hyperparameters are copied blindly from another workload

The right lesson is not "LAMB is better than AdamW." It is "layerwise scale is another axis of adaptation, useful when scale mismatch is layer-level rather than coordinate-level."

---

## 6. Advanced Topics

### 6.1 Adaptive Methods as Diagonal Natural-Gradient Approximations

Natural gradient rescales updates by the inverse Fisher information matrix:

$$
\Delta \boldsymbol{\theta}
= -\eta F(\boldsymbol{\theta})^{-1}\nabla \mathcal{L}(\boldsymbol{\theta}).
$$

If $F$ is approximated by a diagonal matrix whose entries are estimated by squared gradients, the update begins to resemble AdaGrad, RMSProp, or Adam without momentum. This is not an exact identity, but it is a useful mental bridge.

The distinction matters. Natural gradient is invariant to certain reparameterizations because it uses the geometry of distributions. Adam uses coordinatewise gradient statistics. It is cheaper and easier, but less geometrically principled.

### 6.2 Relationship to Curvature and Preconditioning

Adaptive methods are sometimes described as curvature methods. That is partly true and partly misleading.

True:

- they rescale gradients like preconditioned methods
- squared gradients can estimate diagonal Fisher-like quantities
- they reduce some coordinatewise ill-conditioning

Misleading:

- they do not estimate the full Hessian
- they ignore off-diagonal curvature
- they can amplify noise if moment estimates are poor

So Adam is best understood as a practical diagonal preconditioner with momentum, not as a cheap Newton method.

### 6.3 Sign-Based and Low-Memory Optimizers

Sign-based methods update using the sign of a gradient or momentum vector:

$$
\boldsymbol{\theta}_{t+1}
= \boldsymbol{\theta}_t - \eta \operatorname{sign}(\mathbf{u}_t).
$$

Because Adam's normalized update can become sign-like in simple regimes, sign-based optimizers such as Lion are natural relatives. They may reduce state memory or improve certain workloads, but they should be treated as empirical optimizer proposals, not universal replacements.

The 2026 practical stance is conservative: compare against strong AdamW baselines with the same schedule, compute budget, parameter groups, and regularization.

### 6.4 Structured Preconditioners

Diagonal preconditioners are cheap but limited. Structured methods such as Shampoo use matrix preconditioners for tensor dimensions. For a weight matrix $W$ and gradient $G$, a Shampoo-style update may use

$$
L^{-1/4} G R^{-1/4},
$$

where $L$ and $R$ approximate curvature along output and input dimensions.

This can capture correlations that diagonal methods miss, but it costs more memory, computation, and systems complexity. The full treatment of curvature methods belongs to [Second-Order Methods](../03-Second-Order-Methods/notes.md); here Shampoo is a bridge showing how far optimizer design can move beyond coordinatewise adaptation.

### 6.5 Stability Diagnostics

A serious training run should log more than loss. Useful optimizer diagnostics include:

$$
\lVert \mathbf{g}_t \rVert_2,
\qquad
\lVert \Delta \boldsymbol{\theta}_t \rVert_2,
\qquad
\frac{\lVert \Delta \boldsymbol{\theta}_t \rVert_2}
{\lVert \boldsymbol{\theta}_t \rVert_2},
\qquad
\eta_{t,i}^{\mathrm{eff}}.
$$

These reveal different failures:

- exploding gradient norm: backward instability or loss scaling issue
- tiny update norm: learning rate too small or denominators too large
- huge update-to-parameter ratio: optimizer is rewriting weights
- effective LR spikes: second-moment estimate too small or $\epsilon$ too small
- layerwise imbalance: some modules learn while others stagnate

---

## 7. Applications in Machine Learning

### 7.1 AdamW for Transformer Pretraining and Fine-Tuning

AdamW is the default starting point for transformer training because it handles scale mismatch, sparse embeddings, noisy gradients, and decoupled weight decay. Typical parameter grouping excludes biases and normalization parameters from weight decay:

```text
decay:
  large weight matrices

no_decay:
  bias terms
  LayerNorm / RMSNorm scale parameters
  sometimes embeddings depending on recipe
```

Fine-tuning usually uses much smaller learning rates than pretraining because pretrained weights already encode useful structure. The optimizer's job is to adapt without destroying.

### 7.2 Sparse Embeddings and Recommendation Models

Sparse embeddings are one of the cleanest use cases for adaptive methods. If a rare item appears once every million examples, it should not be forced to use the same historical step scale as a common item. AdaGrad's original motivation fits this setting well.

For recommender systems, optimizer choice interacts with:

- embedding frequency
- delayed updates
- distributed parameter servers
- per-row accumulators
- memory constraints

### 7.3 Reinforcement Learning and Non-Stationary Objectives

Reinforcement learning gradients are noisy and nonstationary. The data distribution changes as the policy changes, and gradient estimates often have high variance. Adam and RMSProp are common because their running statistics smooth unstable update directions.

However, adaptivity can also hide instability. If the objective changes rapidly, old moment estimates can become stale. In RL, logging KL divergence, entropy, gradient norm, and update norm is often as important as optimizer choice.

### 7.4 Large-Batch Training with LAMB

Large-batch training tries to improve throughput by using more examples per optimizer step. But very large batches can destabilize the relationship between global learning rate, gradient noise, and layerwise update scale.

LAMB addresses this by combining Adam-style updates with layerwise trust ratios. It was designed for BERT-scale large-batch training, where batch sizes can be much larger than ordinary single-node training.

Use LAMB when large-batch scaling is the bottleneck and AdamW with careful scheduling is not enough. Do not use it just because it sounds more advanced.

### 7.5 Practical Optimizer Selection Checklist

| Workload | Strong starting point | Watch |
| --- | --- | --- |
| Transformer pretraining | AdamW | update norm, weight decay groups, schedule |
| LLM fine-tuning | AdamW | catastrophic forgetting, LR too high |
| Sparse embeddings | AdaGrad or Adam | rare-feature learning, state memory |
| Vision classification | SGD + momentum or AdamW | generalization gap |
| RL | Adam or RMSProp | nonstationarity and stale moments |
| Large-batch BERT-style training | AdamW, then LAMB | trust ratio and schedule |
| Memory-constrained large model | Adafactor or sharded AdamW | quality-memory trade-off |

---

## 8. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
| --- | --- | --- | --- |
| 1 | "Adam chooses the learning rate for me." | Adam rescales coordinates but still has a global $\eta$ that strongly controls training. | Tune $\eta$ and use diagnostics for effective learning rates. |
| 2 | "Adaptive optimizer means learning-rate schedule is unnecessary." | Internal adaptivity and external schedules solve different problems. | Use AdamW with warmup/decay when the workload calls for it. |
| 3 | "Adam + $L_2$ is the same as AdamW." | Coupled $L_2$ enters the adaptive denominator; AdamW decouples decay. | Use AdamW for decoupled weight decay. |
| 4 | "Weight decay should apply to every parameter." | Bias and normalization parameters often should not be decayed. | Use explicit parameter groups. |
| 5 | "A smaller $\epsilon$ is always more accurate." | Tiny $\epsilon$ can create huge effective learning rates when $v_t$ is small. | Treat $\epsilon$ as a stability hyperparameter. |
| 6 | "AMSGrad is always better because it has a proof." | Proof conditions do not guarantee better empirical deep-learning performance. | Use AMSGrad when stability demands it, not by default. |
| 7 | "LAMB is a drop-in AdamW replacement." | LAMB changes layerwise update scaling and is mainly motivated by large batches. | Use it when trust ratios solve a real scaling issue. |
| 8 | "Optimizer state memory is secondary." | AdamW state can exceed parameter memory at large scale. | Budget optimizer state early; consider sharding or Adafactor. |
| 9 | "New optimizer beats AdamW in one paper, so switch." | Optimizer comparisons are highly schedule- and implementation-dependent. | Compare under equal compute, schedule, and tuning. |
| 10 | "Diagonal adaptivity handles curvature." | It handles coordinatewise scale, not off-diagonal curvature. | Use structured preconditioners only when justified. |
| 11 | "Effective learning rate is the same for all parameters." | The whole point of adaptive methods is coordinatewise effective LR. | Inspect distributions of effective LR and update norms. |
| 12 | "Bias correction is a small implementation detail." | Early-step moments are biased toward zero without correction. | Keep bias correction unless reproducing a specific variant. |

---

## 9. Exercises

1. **Exercise 1 (★): AdaGrad by Hand**
   Given a sequence of scalar gradients, compute AdaGrad's accumulator, effective learning rate, and parameter updates.

2. **Exercise 2 (★): RMSProp Forgetting**
   Compare cumulative AdaGrad and exponentially weighted RMSProp on a gradient sequence whose scale changes halfway through.

3. **Exercise 3 (★): Adam Bias Correction**
   For a constant gradient, derive uncorrected and bias-corrected first and second moments.

4. **Exercise 4 (★★): Effective Learning Rate Diagnostics**
   Simulate multi-coordinate gradients and plot the effective learning rates under AdaGrad, RMSProp, and Adam.

5. **Exercise 5 (★★): AdamW vs Coupled $L_2$**
   Show numerically that coupled $L_2$ regularization and decoupled weight decay differ for Adam.

6. **Exercise 6 (★★): LAMB Trust Ratio**
   Compute layerwise trust ratios and explain how they change update magnitudes.

7. **Exercise 7 (★★★): Adafactor Memory Savings**
   Compare full Adam second-moment memory to factored Adafactor memory for matrix-shaped parameters.

8. **Exercise 8 (★★★): Optimizer Diagnosis for a Failed Run**
   Given synthetic logs of gradients, updates, parameters, and effective LRs, identify the likely failure mode and propose a fix.

---

## 10. Why This Matters for AI (2026 Perspective)

| Concept | AI Impact |
| --- | --- |
| AdaGrad | Strong for sparse features, embeddings, and online learning |
| RMSProp | Historical bridge from AdaGrad to Adam; useful in noisy/nonstationary settings |
| Adam | Robust adaptive optimizer for stochastic objectives |
| AdamW | Default baseline for transformers and LLM fine-tuning |
| Bias correction | Stabilizes early optimizer steps |
| Effective learning rate | Explains why adaptive methods still need global LR tuning |
| Decoupled weight decay | Separates optimization step from regularization effect |
| LAMB / LARS | Helps large-batch training by controlling layerwise update scale |
| Adafactor | Reduces optimizer memory for large matrix-shaped parameters |
| Optimizer diagnostics | Turns training failures into measurable signals |
| Structured preconditioning | Active research direction beyond diagonal adaptation |

The key 2026 lesson is that optimizer choice is now a systems decision as much as a mathematical one. AdamW may be mathematically simple compared with full curvature methods, but its memory footprint, mixed-precision behavior, parameter grouping, distributed state sharding, and interaction with schedules determine whether large model training is stable.

At the same time, the field should be skeptical of optimizer hype. Many new optimizers look strong under one schedule, one model scale, or one benchmark. A serious comparison must control for learning-rate schedule, weight decay, parameter grouping, batch size, precision, and compute.

---

## 11. Conceptual Bridge

Adaptive learning-rate methods sit at the center of the practical optimization stack. They inherit stochastic-gradient noise from [Stochastic Optimization](../05-Stochastic-Optimization/notes.md), respond to geometry studied in [Optimization Landscape](../06-Optimization-Landscape/notes.md), and borrow the preconditioning idea from [Second-Order Methods](../03-Second-Order-Methods/notes.md).

They also point forward. [Regularization Methods](../08-Regularization-Methods/notes.md) explains why weight decay and other penalties affect generalization. [Hyperparameter Optimization](../09-Hyperparameter-Optimization/notes.md) studies how to tune optimizer hyperparameters systematically. [Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md) studies external time-varying schedules such as warmup, cosine decay, and WSD.

The most important mental model is this: adaptive optimizers are not replacements for understanding learning rates. They are mechanisms for distributing one global learning-rate budget across coordinates, layers, and time.

```text
ADAPTIVE LEARNING RATE IN THE OPTIMIZATION CHAPTER
════════════════════════════════════════════════════════════════════════

  05 Stochastic Optimization
       noisy gradient estimates
              |
              v
  06 Optimization Landscape
       geometry, sharpness, curvature
              |
              v
  07 Adaptive Learning Rate
       per-coordinate and layerwise update scaling
              |
      +-------+--------+
      |                |
      v                v
  08 Regularization   10 Learning Rate Schedules
      weight decay        warmup, cosine, WSD

════════════════════════════════════════════════════════════════════════
```

## References

1. Duchi, J., Hazan, E., and Singer, Y. (2011). "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization." *Journal of Machine Learning Research*. https://jmlr.org/papers/v12/duchi11a.html
2. Tieleman, T., and Hinton, G. (2012). "Lecture 6.5 - RMSProp." *Neural Networks for Machine Learning*.
3. Kingma, D. P., and Ba, J. (2014). "Adam: A Method for Stochastic Optimization." https://arxiv.org/abs/1412.6980
4. Reddi, S. J., Kale, S., and Kumar, S. (2018). "On the Convergence of Adam and Beyond." https://arxiv.org/abs/1904.09237
5. Loshchilov, I., and Hutter, F. (2017). "Decoupled Weight Decay Regularization." https://arxiv.org/abs/1711.05101
6. You, Y. et al. (2019). "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes." https://arxiv.org/abs/1904.00962
7. Shazeer, N., and Stern, M. (2018). "Adafactor: Adaptive Learning Rates with Sublinear Memory Cost." https://arxiv.org/abs/1804.04235
8. Gupta, V., Koren, T., and Singer, Y. (2018). "Shampoo: Preconditioned Stochastic Tensor Optimization." https://arxiv.org/abs/1802.09568
9. PyTorch Documentation. `torch.optim.AdamW`. https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
10. Optax Documentation. Optimizers API. https://optax.readthedocs.io/en/stable/api/optimizers.html
