[Back to Curriculum](../../README.md) | [Previous: Activation Functions](../02-Activation-Functions/notes.md) | [Next: Sampling Methods](../04-Sampling-Methods/notes.md)

---

# Normalization Techniques

> _"Deep networks are easier to train when their internal scales are made visible, controlled, and learnable."_

## Overview

Normalization techniques control the statistics of activations, weights, or
features so optimization sees a better-conditioned problem. They are not just
preprocessing tricks. Inside neural networks, normalization changes forward
signal scale, backward gradient flow, effective parameterization, train/eval
behavior, and numerical stability.

This section is the canonical home for normalization math as a reusable ML
primitive. Chapter 14 discusses how BatchNorm, LayerNorm, RMSNorm, and
SpectralNorm appear in specific model families. Chapter 15 discusses LLM-scale
block design and fused kernels. Here we focus on axes, moments, affine
reparameterization, train-time statistics, inference statistics, epsilon,
RMS-only scaling, group statistics, weight statistics, residual placement, and
the implementation mistakes that make normalization layers silently wrong.

The core habit is to always ask: "What axis is normalized? Which statistics are
used? Are they batch-dependent? Are they learned, frozen, or recomputed? What
gradient path does this create?"

## Prerequisites

- **Mean, variance, and standard deviation** - [Descriptive Statistics](../../07-Statistics/01-Descriptive-Statistics/notes.md)
- **Expectation and moments** - [Expectation and Moments](../../06-Probability-Theory/04-Expectation-and-Moments/notes.md)
- **Activation functions and gradient flow** - [Activation Functions](../02-Activation-Functions/notes.md)
- **Numerical stability and floating point** - [Floating Point Arithmetic](../../10-Numerical-Methods/01-Floating-Point-Arithmetic/notes.md)
- **Regularization and spectral constraints** - [Regularization Methods](../../08-Optimization/08-Regularization-Methods/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Axis-aware BatchNorm, LayerNorm, RMSNorm, GroupNorm, WeightNorm, SpectralNorm, and residual-placement demos |
| [exercises.ipynb](exercises.ipynb) | 10 graded exercises on moments, broadcasting, train/eval behavior, and normalization diagnostics |

## Learning Objectives

After completing this section, you will be able to:

- Identify which axes BatchNorm, LayerNorm, RMSNorm, GroupNorm, InstanceNorm, WeightNorm, and SpectralNorm normalize
- Derive the normalized affine transform $\gamma\hat{x}+\beta$
- Explain why epsilon is a numerical stabilizer, not a regularization constant
- Distinguish batch statistics, running statistics, and per-example statistics
- Implement BatchNorm, LayerNorm, RMSNorm, and GroupNorm in NumPy
- Explain why BatchNorm has different train and eval behavior
- Compare LayerNorm and RMSNorm for sequence-style hidden states
- Diagnose broadcasting, axis, small-batch, and mixed-precision failures
- Explain pre-norm versus post-norm as a gradient-flow design choice

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 Why Activations Drift](#11-why-activations-drift)
  - [1.2 Normalization as Re-Centering](#12-normalization-as-re-centering)
  - [1.3 Normalization as Conditioning](#13-normalization-as-conditioning)
  - [1.4 Signal Scale in Deep Nets](#14-signal-scale-in-deep-nets)
  - [1.5 Historical Path](#15-historical-path)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Normalization Axes](#21-normalization-axes)
  - [2.2 Mean and Variance Statistics](#22-mean-and-variance-statistics)
  - [2.3 Epsilon](#23-epsilon)
  - [2.4 Learnable Gain and Bias](#24-learnable-gain-and-bias)
  - [2.5 Train-Time Versus Inference-Time Behavior](#25-train-time-versus-inference-time-behavior)
- [3. Batch Normalization](#3-batch-normalization)
  - [3.1 Batch Statistics](#31-batch-statistics)
  - [3.2 Running Averages](#32-running-averages)
  - [3.3 Affine Transform](#33-affine-transform)
  - [3.4 Train/Eval Gap](#34-traineval-gap)
  - [3.5 Batch-Size Sensitivity](#35-batch-size-sensitivity)
- [4. Layer Normalization](#4-layer-normalization)
  - [4.1 Per-Example Statistics](#41-per-example-statistics)
  - [4.2 Feature-Axis Normalization](#42-feature-axis-normalization)
  - [4.3 Sequence Models](#43-sequence-models)
  - [4.4 Invariance Properties](#44-invariance-properties)
  - [4.5 Comparison to BatchNorm](#45-comparison-to-batchnorm)
- [5. RMSNorm](#5-rmsnorm)
  - [5.1 RMS Statistic](#51-rms-statistic)
  - [5.2 Scale-Only Normalization](#52-scale-only-normalization)
  - [5.3 No Mean Subtraction](#53-no-mean-subtraction)
  - [5.4 Efficiency](#54-efficiency)
  - [5.5 LLM Usage Preview](#55-llm-usage-preview)
- [6. Other Normalizations](#6-other-normalizations)
  - [6.1 GroupNorm](#61-groupnorm)
  - [6.2 InstanceNorm](#62-instancenorm)
  - [6.3 WeightNorm](#63-weightnorm)
  - [6.4 SpectralNorm](#64-spectralnorm)
  - [6.5 ScaleNorm Preview](#65-scalenorm-preview)
- [7. Residual Blocks and Placement](#7-residual-blocks-and-placement)
  - [7.1 Pre-Norm](#71-pre-norm)
  - [7.2 Post-Norm](#72-post-norm)
  - [7.3 Sandwich Norm Preview](#73-sandwich-norm-preview)
  - [7.4 Gradient Flow](#74-gradient-flow)
  - [7.5 Deep-Stack Stability](#75-deep-stack-stability)
- [8. Numerical and Implementation Details](#8-numerical-and-implementation-details)
  - [8.1 Broadcasting Axes](#81-broadcasting-axes)
  - [8.2 Epsilon Choice](#82-epsilon-choice)
  - [8.3 Mixed Precision](#83-mixed-precision)
  - [8.4 Small-Batch Failure](#84-small-batch-failure)
  - [8.5 Fused Kernels Preview](#85-fused-kernels-preview)
- [9. Applications in Machine Learning](#9-applications-in-machine-learning)
  - [9.1 CNNs](#91-cnns)
  - [9.2 RNNs](#92-rnns)
  - [9.3 Transformers](#93-transformers)
  - [9.4 GANs](#94-gans)
  - [9.5 Large-Scale Training](#95-large-scale-training)
- [10. Common Mistakes](#10-common-mistakes)
- [11. Exercises](#11-exercises)
- [12. Why This Matters for AI](#12-why-this-matters-for-ai)
- [13. Conceptual Bridge](#13-conceptual-bridge)
- [References](#references)

---

## 1. Intuition

### 1.1 Why Activations Drift

During training, every layer receives inputs produced by previous layers whose
parameters are changing. Even if the raw data distribution is fixed, internal
activation distributions move. Means shift, variances change, outliers appear,
and gradient scales become inconsistent across depth.

Normalization addresses this by computing statistics over a chosen axis and
reparameterizing activations into a controlled scale. A generic normalized
activation has the form

$$
\hat{x}=\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}.
$$

### 1.2 Normalization as Re-Centering

Mean subtraction re-centers values:

$$
x \mapsto x-\mu.
$$

This can make optimization easier because downstream affine layers do not need
to constantly compensate for drifting offsets. But not every normalization
subtracts the mean. RMSNorm normalizes by root mean square and preserves the
mean direction.

### 1.3 Normalization as Conditioning

Optimization is sensitive to scale. If one feature dimension has variance
$10^6$ and another has variance $10^{-4}$, a single learning rate is poorly
matched to both. Normalization reduces such scale mismatch. It does not make
all optimization problems convex, but it often improves conditioning.

### 1.4 Signal Scale in Deep Nets

Deep networks multiply many local transformations. If activation scale grows
slightly at each layer, it can explode. If it shrinks slightly, it can vanish.
Normalization layers repeatedly reset or constrain scale so residual blocks,
attention blocks, convolution blocks, or recurrent transitions remain
trainable.

### 1.5 Historical Path

BatchNorm made normalization a core deep-learning layer by using mini-batch
statistics. LayerNorm removed batch dependence and became natural for sequence
models. GroupNorm and InstanceNorm addressed small-batch and image-style
settings. RMSNorm simplified LayerNorm by removing mean subtraction.
SpectralNorm constrained weight operators rather than activations.

## 2. Formal Definitions

### 2.1 Normalization Axes

Let $\mathcal{X}\in\mathbb{R}^{B\times T\times D}$ represent a batch of
sequence activations. Here $B$ is batch size, $T$ is sequence length, and $D$
is feature width. Different normalization methods choose different axes:

| Method | Typical axes for statistics | Batch-dependent? |
| --- | --- | --- |
| BatchNorm | batch and sometimes spatial axes | Yes |
| LayerNorm | feature axis per example/token | No |
| RMSNorm | feature axis per example/token | No |
| GroupNorm | groups of channels per example | No |
| InstanceNorm | spatial axes per example/channel | No |
| WeightNorm | weight-vector direction/scale | No |
| SpectralNorm | largest singular value of weight matrix | No batch statistics |

The axis choice is the most important part of the definition. Two formulas can
look identical while normalizing completely different objects.

### 2.2 Mean and Variance Statistics

For a set of values $\{x_i\}_{i=1}^{m}$, the mean and variance are

$$
\mu=\frac{1}{m}\sum_{i=1}^{m}x_i,
\qquad
\sigma^2=\frac{1}{m}\sum_{i=1}^{m}(x_i-\mu)^2.
$$

The normalized value is

$$
\hat{x}_i=\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}.
$$

In deep-learning libraries, variance is often the biased batch variance, not
the unbiased sample variance. The choice is part of the layer definition.

### 2.3 Epsilon

Epsilon prevents division by zero:

$$
\sqrt{\sigma^2+\epsilon}.
$$

It is not a regularizer in the usual statistical sense. It is a numerical
stabilizer. Too small an epsilon can fail in low precision. Too large an
epsilon changes the effective normalization by preventing true unit variance.

### 2.4 Learnable Gain and Bias

After normalization, most methods apply learnable affine parameters:

$$
y_i=\gamma_i\hat{x}_i+\beta_i.
$$

The gain $\gamma$ lets the network restore useful scale. The bias $\beta$ lets
it restore useful offset. Without these parameters, normalization could remove
representations the model needs.

### 2.5 Train-Time Versus Inference-Time Behavior

BatchNorm uses mini-batch statistics during training and running estimates
during inference. LayerNorm and RMSNorm compute per-example statistics at both
train and inference time. This distinction affects reproducibility,
deployment, and small-batch behavior.

## 3. Batch Normalization

### 3.1 Batch Statistics

For a feature $j$ over a batch,

$$
\mu_j=\frac{1}{B}\sum_{i=1}^{B}x_{ij},
\qquad
\sigma_j^2=\frac{1}{B}\sum_{i=1}^{B}(x_{ij}-\mu_j)^2.
$$

Then

$$
\hat{x}_{ij}=\frac{x_{ij}-\mu_j}{\sqrt{\sigma_j^2+\epsilon}}.
$$

For convolutional tensors, statistics are often computed across batch and
spatial positions for each channel.

### 3.2 Running Averages

BatchNorm maintains running estimates:

$$
\mu_{\mathrm{run}}\leftarrow
(1-\alpha)\mu_{\mathrm{run}}+\alpha\mu_{\mathrm{batch}},
$$

$$
\sigma^2_{\mathrm{run}}\leftarrow
(1-\alpha)\sigma^2_{\mathrm{run}}+\alpha\sigma^2_{\mathrm{batch}}.
$$

These estimates are used in evaluation mode. If the running estimates are bad,
validation and deployment behavior can be bad even when training behavior
looked fine.

### 3.3 Affine Transform

BatchNorm output is

$$
y_{ij}=\gamma_j\hat{x}_{ij}+\beta_j.
$$

The parameters are feature-specific. In CNNs they are channel-specific and
broadcast over spatial dimensions.

### 3.4 Train/Eval Gap

During training, a sample is normalized using statistics from its current
mini-batch. During inference, it is normalized using stored running statistics.
This creates a train/eval gap. The gap is small when batches are large and
representative. It can be large for tiny batches, distribution shift, or
nonstationary data.

### 3.5 Batch-Size Sensitivity

BatchNorm is noisy with small batches because mean and variance estimates have
high variance. Distributed training can also change effective batch statistics
depending on whether BatchNorm is synchronized across devices.

## 4. Layer Normalization

### 4.1 Per-Example Statistics

LayerNorm computes statistics over features for each example:

$$
\mu_i=\frac{1}{D}\sum_{j=1}^{D}x_{ij},
\qquad
\sigma_i^2=\frac{1}{D}\sum_{j=1}^{D}(x_{ij}-\mu_i)^2.
$$

For sequence tensors, it is usually applied separately to each token position.

### 4.2 Feature-Axis Normalization

LayerNorm normalizes the feature vector:

$$
\hat{\mathbf{x}}_i
=\frac{\mathbf{x}_i-\mu_i\mathbf{1}}
{\sqrt{\sigma_i^2+\epsilon}}.
$$

Its gain and bias are feature-wise:

$$
\mathbf{y}_i=\boldsymbol{\gamma}\odot\hat{\mathbf{x}}_i+\boldsymbol{\beta}.
$$

### 4.3 Sequence Models

LayerNorm is natural for sequence models because it does not depend on other
examples in the batch. A token can be normalized using only its own hidden
state. This makes behavior consistent between training and autoregressive
inference.

### 4.4 Invariance Properties

LayerNorm is invariant to adding a constant offset to all features in the
normalized vector and to multiplying all features by a positive scalar, up to
epsilon and learned affine parameters.

### 4.5 Comparison to BatchNorm

BatchNorm normalizes each feature using batch statistics. LayerNorm normalizes
each example using feature statistics. BatchNorm couples examples in a batch.
LayerNorm couples features inside an example.

## 5. RMSNorm

### 5.1 RMS Statistic

RMSNorm uses the root mean square:

$$
\operatorname{RMS}(\mathbf{x})
=\sqrt{\frac{1}{D}\sum_{j=1}^{D}x_j^2+\epsilon}.
$$

### 5.2 Scale-Only Normalization

The normalized vector is

$$
\hat{\mathbf{x}}=
\frac{\mathbf{x}}{\operatorname{RMS}(\mathbf{x})}.
$$

Then

$$
\mathbf{y}=\boldsymbol{\gamma}\odot\hat{\mathbf{x}}.
$$

### 5.3 No Mean Subtraction

RMSNorm does not subtract the mean and usually does not include a bias
parameter. It controls scale but preserves mean direction. This makes it
cheaper and can work well in residual-stream architectures.

### 5.4 Efficiency

LayerNorm requires mean and variance. RMSNorm requires only the mean square.
That saves operations and can simplify fused kernels. At large scale, small
per-token savings matter.

### 5.5 LLM Usage Preview

Many modern LLM-style architectures use RMSNorm or RMSNorm-like variants.
Chapter 15 covers LLM-specific block details. Here the reusable math is simply
scale-only feature normalization.

## 6. Other Normalizations

### 6.1 GroupNorm

GroupNorm divides channels into groups and normalizes within each group for
each example. It avoids batch dependence while retaining channel-group
structure.

### 6.2 InstanceNorm

InstanceNorm normalizes each sample and channel over spatial dimensions. It is
common in style-transfer and image-generation contexts where instance-specific
contrast should be controlled.

### 6.3 WeightNorm

WeightNorm reparameterizes a weight vector:

$$
\mathbf{w}=g\frac{\mathbf{v}}{\lVert\mathbf{v}\rVert_2}.
$$

It separates direction $\mathbf{v}/\lVert\mathbf{v}\rVert_2$ from magnitude
$g$.

### 6.4 SpectralNorm

Spectral normalization constrains a weight matrix by its largest singular
value:

$$
\bar{W}=\frac{W}{\sigma_{\max}(W)}.
$$

This controls the operator norm and therefore the layer's Lipschitz constant.
The full singular-value theory belongs in
[Singular Value Decomposition](../../03-Advanced-Linear-Algebra/02-Singular-Value-Decomposition/notes.md).

### 6.5 ScaleNorm Preview

ScaleNorm normalizes a vector by its norm and multiplies by a learned scalar.
It is conceptually close to RMSNorm but uses explicit vector norm scaling.

## 7. Residual Blocks and Placement

### 7.1 Pre-Norm

Pre-norm residual blocks apply normalization before the sublayer:

$$
\mathbf{h}_{l+1}
=\mathbf{h}_l+F(\operatorname{Norm}(\mathbf{h}_l)).
$$

This gives the residual path a direct identity route for gradients.

### 7.2 Post-Norm

Post-norm applies normalization after the residual addition:

$$
\mathbf{h}_{l+1}
=\operatorname{Norm}(\mathbf{h}_l+F(\mathbf{h}_l)).
$$

It can produce cleaner normalized outputs per block but may make very deep
stacks harder to optimize.

### 7.3 Sandwich Norm Preview

Some architectures add normalization both before and after certain sublayers.
This is a model-design detail, so the full treatment belongs in model-specific
chapters. The general math is that each norm changes both forward scale and
backward Jacobian.

### 7.4 Gradient Flow

Normalization layers have Jacobians. They do not simply "standardize and
disappear." Their backward pass subtracts mean-like components and rescales
gradients. This can stabilize training but can also introduce coupling across
the normalized axis.

### 7.5 Deep-Stack Stability

Deep residual stacks rely on controlling activation and residual-stream scale.
Normalization placement, residual scaling, initialization, and optimizer
choice interact. A stable block is a system, not a single layer.

## 8. Numerical and Implementation Details

### 8.1 Broadcasting Axes

Most normalization bugs are axis bugs. A parameter $\gamma\in\mathbb{R}^D$
must broadcast over batch and sequence axes but align with feature axes. If
the shape is wrong, code may run while normalizing the wrong dimension.

### 8.2 Epsilon Choice

Epsilon must be large enough to protect low-variance vectors and low-precision
formats. It must be small enough not to dominate real variance. Common values
include $10^{-5}$ and $10^{-6}$, but the right value is implementation and
dtype dependent.

### 8.3 Mixed Precision

Mixed-precision normalization often accumulates statistics in higher precision
even when inputs are lower precision. This reduces underflow, overflow, and
rounding errors in variance computation.

### 8.4 Small-Batch Failure

BatchNorm with batch size one is usually ill-posed for dense features because
the batch variance can be zero. CNN spatial axes may still provide enough
samples, but the effective sample count should be checked.

### 8.5 Fused Kernels Preview

At large scale, normalization is often fused with neighboring operations to
reduce memory traffic. The math is unchanged, but implementation details
affect speed and numerical behavior.

## 9. Applications in Machine Learning

### 9.1 CNNs

BatchNorm is common in CNNs because batch and spatial dimensions provide many
samples per channel. It also acts as a mild source of stochasticity during
training.

### 9.2 RNNs

LayerNorm is often easier than BatchNorm for recurrent models because sequence
lengths and time dependencies make batch statistics awkward.

### 9.3 Transformers

Transformers use LayerNorm, RMSNorm, and placement variants to stabilize deep
residual blocks. The full Transformer block treatment belongs in
[Transformer Architecture](../../14-Math-for-Specific-Models/05-Transformer-Architecture/notes.md).

### 9.4 GANs

SpectralNorm controls discriminator Lipschitz behavior and helps stabilize
adversarial training. InstanceNorm and normalization variants also appear in
image generation pipelines.

### 9.5 Large-Scale Training

At large scale, normalization affects stability, throughput, memory bandwidth,
mixed-precision behavior, and distributed reproducibility. Seemingly small
choices become system-level choices.

## 10. Common Mistakes

| # | Mistake | Why It Is Wrong | Fix |
| --- | --- | --- | --- |
| 1 | Normalizing the wrong axis | Statistics describe the wrong object | Write tensor shapes and axes explicitly |
| 2 | Forgetting BatchNorm train/eval mode | Batch and running statistics differ | Switch modes deliberately and test both |
| 3 | Treating epsilon as harmless | Large epsilon changes scale | Tune epsilon with dtype and variance in mind |
| 4 | Using BatchNorm with tiny batches blindly | Statistics are noisy or degenerate | Use LayerNorm, GroupNorm, or synchronized stats |
| 5 | Broadcasting gain over the wrong dimension | Code may run but learn wrong parameters | Assert shapes before training |
| 6 | Comparing LayerNorm and RMSNorm as identical | RMSNorm does not center | Track mean and RMS separately |
| 7 | Ignoring mixed-precision accumulation | Variance can underflow or overflow | Accumulate statistics in higher precision |
| 8 | Calling SpectralNorm an activation norm | It normalizes weights, not activations | Separate activation and operator normalization |
| 9 | Assuming normalization replaces initialization | Bad initialization can still break training | Use compatible initialization and norm placement |
| 10 | Removing affine gain/bias casually | The model may need restored scale/offset | Remove only with a clear architectural reason |

## 11. Exercises

1. (*) Compute mean, variance, and normalized values for a vector.
2. (*) Implement BatchNorm for a matrix with shape $B\times D$.
3. (*) Implement LayerNorm for a matrix with shape $B\times D$.
4. (**) Show that BatchNorm changes when the batch composition changes.
5. (**) Show that LayerNorm is independent of other examples in the batch.
6. (**) Implement RMSNorm and compare its output mean to LayerNorm.
7. (**) Implement GroupNorm for a tensor with grouped channels.
8. (***) Compute a WeightNorm reparameterization and verify its norm.
9. (***) Estimate a spectral norm by power iteration.
10. (***) Diagnose a shape bug in a fake normalization layer.

## 12. Why This Matters for AI

| Concept | AI impact |
| --- | --- |
| Axis choice | Determines whether examples, features, channels, or weights are coupled |
| BatchNorm | Stabilizes CNN training but introduces batch dependence |
| LayerNorm | Makes sequence and token-wise models batch-independent |
| RMSNorm | Controls scale with lower overhead and no centering |
| GroupNorm | Works when batch statistics are unreliable |
| SpectralNorm | Controls operator scale and Lipschitz behavior |
| Epsilon | Prevents numerical failure in low-variance and low-precision regimes |
| Pre-norm | Improves gradient flow in deep residual stacks |
| Train/eval mode | Affects reproducibility and deployment correctness |
| Fused normalization | Matters for throughput at large scale |

## 13. Conceptual Bridge

Activation functions create nonlinear hidden states. Normalization techniques
control the statistics of those hidden states.

```text
activation values
    -> mean / variance / RMS statistics
    -> normalized hidden state
    -> learnable scale and bias
    -> next layer or residual block
```

Next, [Sampling Methods](../04-Sampling-Methods/notes.md) moves from
deterministic transformations to randomized estimators, proposal distributions,
and generation procedures.

## References

- Ioffe, S., and Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.
- Ba, J. L., Kiros, J. R., and Hinton, G. E. (2016). Layer Normalization.
- Salimans, T., and Kingma, D. P. (2016). Weight Normalization.
- Ulyanov, D., Vedaldi, A., and Lempitsky, V. (2016). Instance Normalization.
- Miyato, T., Kataoka, T., Koyama, M., and Yoshida, Y. (2018). Spectral Normalization for Generative Adversarial Networks.
- Wu, Y., and He, K. (2018). Group Normalization.
- Zhang, B., and Sennrich, R. (2019). Root Mean Square Layer Normalization.
