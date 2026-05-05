[Back to Curriculum](../../README.md) | [Previous: Loss Functions](../01-Loss-Functions/notes.md) | [Next: Normalization Techniques](../03-Normalization-Techniques/notes.md)

---

# Activation Functions

> _"A neural network without nonlinear activation is only a disguised linear map."_

## Overview

Activation functions are the local nonlinear maps that turn stacks of affine
layers into expressive models. They decide which signals pass forward, which
gradients flow backward, which neurons saturate, which features become sparse,
and how numerical scale changes with depth. Their formulas look small, but
their training consequences are large.

This section is the canonical home for activation-function math in the
curriculum. Chapter 14 discusses how neural networks, RNNs, CNNs, and
Transformers use these functions inside model architectures. Chapter 15
discusses LLM-specific block design. Here we focus on the reusable mathematical
objects themselves: scalar activations, vector activations, derivatives,
Jacobians, saturation, smoothness, Lipschitz behavior, gating, softmax
temperature, and gradient-flow consequences.

The central question is not "Which activation is popular?" but "What map does
this activation apply, what derivative does it create, and what does that
derivative do to learning?"

## Prerequisites

- **Single-variable derivatives and chain rule** - [Derivatives](../../04-Calculus-Fundamentals/02-Derivatives/notes.md)
- **Jacobians and vector-valued functions** - [Jacobian Matrix](../../05-Multivariate-Calculus/03-Jacobian-Matrix/notes.md)
- **Gradient flow through optimization** - [Gradient Descent](../../08-Optimization/02-Gradient-Descent/notes.md)
- **Loss gradients at the output** - [Loss Functions](../01-Loss-Functions/notes.md)
- **Numerical stability** - [Floating Point Arithmetic](../../10-Numerical-Methods/01-Floating-Point-Arithmetic/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Curves, derivatives, Jacobians, saturation, gated activations, and softmax temperature |
| [exercises.ipynb](exercises.ipynb) | 10 graded exercises on activation values, derivatives, gradients, and stable implementations |

## Learning Objectives

After completing this section, you will be able to:

- Define scalar, elementwise vector, and coupled vector activations
- Compute derivatives for sigmoid, tanh, ReLU, Leaky ReLU, GELU, SiLU, and softmax
- Explain saturation and why it causes vanishing gradients
- Compare piecewise-linear and smooth activations by gradient behavior
- Derive the softmax Jacobian $J_{ij}=s_i(\delta_{ij}-s_j)$
- Explain why gated activations such as GLU, GEGLU, and SwiGLU are multiplicative feature selectors
- Connect activation variance to Xavier and He initialization
- Diagnose dead neurons, exploding activations, saturated gates, and unstable softmax
- Choose activations by mathematical role rather than popularity

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 Why Nonlinearities Matter](#11-why-nonlinearities-matter)
  - [1.2 Activations as Gates](#12-activations-as-gates)
  - [1.3 Activations as Gradient Controllers](#13-activations-as-gradient-controllers)
  - [1.4 Shape of Activation Curves](#14-shape-of-activation-curves)
  - [1.5 Historical Path](#15-historical-path)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Scalar Activation](#21-scalar-activation)
  - [2.2 Elementwise Vector Activation](#22-elementwise-vector-activation)
  - [2.3 Coupled Vector Activation](#23-coupled-vector-activation)
  - [2.4 Jacobian](#24-jacobian)
  - [2.5 Smoothness, Lipschitz Constants, and Monotonicity](#25-smoothness-lipschitz-constants-and-monotonicity)
- [3. Classical Activations](#3-classical-activations)
  - [3.1 Sigmoid](#31-sigmoid)
  - [3.2 Tanh](#32-tanh)
  - [3.3 Softplus](#33-softplus)
  - [3.4 Saturation](#34-saturation)
  - [3.5 Gate Interpretation](#35-gate-interpretation)
- [4. ReLU Family](#4-relu-family)
  - [4.1 ReLU](#41-relu)
  - [4.2 Leaky ReLU](#42-leaky-relu)
  - [4.3 PReLU](#43-prelu)
  - [4.4 ELU and SELU Preview](#44-elu-and-selu-preview)
  - [4.5 Dead Neurons and Sparse Activations](#45-dead-neurons-and-sparse-activations)
- [5. Smooth Modern Activations](#5-smooth-modern-activations)
  - [5.1 GELU](#51-gelu)
  - [5.2 SiLU and Swish](#52-silu-and-swish)
  - [5.3 Mish](#53-mish)
  - [5.4 Curvature Comparison](#54-curvature-comparison)
  - [5.5 Gradient-Flow Effects](#55-gradient-flow-effects)
- [6. Gated Activations](#6-gated-activations)
  - [6.1 GLU](#61-glu)
  - [6.2 GEGLU](#62-geglu)
  - [6.3 SwiGLU](#63-swiglu)
  - [6.4 Bilinear Gating](#64-bilinear-gating)
  - [6.5 Why Gated MLPs Help](#65-why-gated-mlps-help)
- [7. Vector Activations](#7-vector-activations)
  - [7.1 Softmax](#71-softmax)
  - [7.2 Temperature Softmax](#72-temperature-softmax)
  - [7.3 Softmax Jacobian](#73-softmax-jacobian)
  - [7.4 Sparsemax and Entmax Preview](#74-sparsemax-and-entmax-preview)
  - [7.5 Attention-Output Preview](#75-attention-output-preview)
- [8. Initialization and Gradient Flow](#8-initialization-and-gradient-flow)
  - [8.1 Activation Variance](#81-activation-variance)
  - [8.2 Xavier and He Initialization](#82-xavier-and-he-initialization)
  - [8.3 Vanishing Gradients](#83-vanishing-gradients)
  - [8.4 Exploding Gradients](#84-exploding-gradients)
  - [8.5 Residual Connection Preview](#85-residual-connection-preview)
- [9. Applications in Machine Learning](#9-applications-in-machine-learning)
  - [9.1 CNNs](#91-cnns)
  - [9.2 RNN Gates](#92-rnn-gates)
  - [9.3 Transformer Feedforward Blocks](#93-transformer-feedforward-blocks)
  - [9.4 Binary Outputs](#94-binary-outputs)
  - [9.5 Probability Heads](#95-probability-heads)
- [10. Common Mistakes](#10-common-mistakes)
- [11. Exercises](#11-exercises)
- [12. Why This Matters for AI](#12-why-this-matters-for-ai)
- [13. Conceptual Bridge](#13-conceptual-bridge)
- [References](#references)

---

## 1. Intuition

### 1.1 Why Nonlinearities Matter

An affine layer has the form

$$
\mathbf{h}=W\mathbf{x}+\mathbf{b}.
$$

If we stack affine layers without nonlinearities, the result is still affine:

$$
W_3(W_2(W_1\mathbf{x}+\mathbf{b}_1)+\mathbf{b}_2)+\mathbf{b}_3
=A\mathbf{x}+\mathbf{c}.
$$

Depth alone does not create nonlinear expressivity. Activation functions insert
nonlinear maps between affine transforms:

$$
\mathbf{h}^{[l]}=\phi(W^{[l]}\mathbf{h}^{[l-1]}+\mathbf{b}^{[l]}).
$$

The activation $\phi$ is what lets the network bend decision boundaries,
compose features, gate information, and approximate complex functions.

### 1.2 Activations as Gates

Many activations behave like gates. ReLU passes positive values and blocks
negative values:

$$
\operatorname{ReLU}(x)=\max(0,x).
$$

Sigmoid maps values into $(0,1)$ and is often interpreted as a soft gate:

$$
\sigma(x)=\frac{1}{1+\exp(-x)}.
$$

Gated activations multiply a content stream by a learned gate:

$$
\operatorname{GLU}(\mathbf{a},\mathbf{b})
=\mathbf{a}\odot\sigma(\mathbf{b}).
$$

This gate perspective is especially useful in LSTMs, GRUs, gated MLPs, and
Transformer feedforward variants.

### 1.3 Activations as Gradient Controllers

Backpropagation multiplies by activation derivatives. In a depth-$L$ network,
early-layer gradients contain products of terms like

$$
W^{[l]\top}\operatorname{diag}(\phi'(\mathbf{z}^{[l]})).
$$

If $\phi'$ is often near zero, gradients vanish. If the product of weight norms
and activation slopes is large, gradients can explode. This is why sigmoid can
saturate deep networks, why ReLU helped train deeper models, and why
normalization and residual connections matter.

### 1.4 Shape of Activation Curves

The curve shape determines both forward statistics and backward gradients:

| Activation | Range | Saturates? | Derivative behavior |
| --- | --- | --- | --- |
| Sigmoid | $(0,1)$ | Both tails | Max derivative $1/4$ |
| Tanh | $(-1,1)$ | Both tails | Max derivative $1$ |
| ReLU | $[0,\infty)$ | Negative side | Derivative $0$ or $1$ |
| GELU | $\mathbb{R}$ | Soft negative gate | Smooth nonmonotone region |
| SiLU | $\mathbb{R}$ | Soft negative gate | Smooth self-gating |
| Softmax | Simplex | Probability saturation | Coupled Jacobian |

### 1.5 Historical Path

Sigmoid and tanh dominated early neural networks because they looked like
smooth biological firing rates and probabilistic gates. ReLU became important
because it reduced saturation and made sparse activations easy. Smooth
variants such as GELU and SiLU became common in large architectures because
they combine gating-like behavior with differentiability. Gated activations
such as GLU, GEGLU, and SwiGLU made multiplicative feature selection a standard
MLP ingredient.

The historical lesson is that activation choice follows training dynamics.
Expressivity, gradient propagation, initialization, normalization, and hardware
all influence the best choice.

## 2. Formal Definitions

### 2.1 Scalar Activation

A scalar activation is a function

$$
\phi:\mathbb{R}\to\mathbb{R}.
$$

Examples:

- $\sigma(x)=1/(1+\exp(-x))$
- $\tanh(x)$
- $\operatorname{ReLU}(x)=\max(0,x)$

Non-examples:

- A loss function $\ell(\hat{y},y)$ is not an activation, because it compares prediction and target.
- A full neural-network layer $W\mathbf{x}+\mathbf{b}$ is not an activation, because it is affine and parameterized.

### 2.2 Elementwise Vector Activation

Given $\mathbf{x}\in\mathbb{R}^d$, an elementwise activation applies the same
scalar map to each coordinate:

$$
\phi(\mathbf{x})=
(\phi(x_1),\ldots,\phi(x_d))^\top.
$$

The Jacobian is diagonal:

$$
J_{\phi}(\mathbf{x})=
\operatorname{diag}(\phi'(x_1),\ldots,\phi'(x_d)).
$$

Examples include sigmoid, tanh, ReLU, GELU, and SiLU when applied to vectors.

### 2.3 Coupled Vector Activation

A coupled vector activation maps $\mathbb{R}^d$ to $\mathbb{R}^d$ but each
output can depend on multiple input coordinates. Softmax is the central
example:

$$
\operatorname{softmax}(\mathbf{z})_i
=\frac{\exp(z_i)}{\sum_{j=1}^{d}\exp(z_j)}.
$$

Changing one logit changes all softmax probabilities because the denominator
couples every coordinate.

### 2.4 Jacobian

For $\mathbf{s}=\operatorname{softmax}(\mathbf{z})$, the Jacobian entries are

$$
\frac{\partial s_i}{\partial z_j}
=s_i(\delta_{ij}-s_j).
$$

In matrix form,

$$
J_{\operatorname{softmax}}(\mathbf{z})
=\operatorname{diag}(\mathbf{s})-\mathbf{s}\mathbf{s}^\top.
$$

This matrix is positive semidefinite and has row sums zero. The row-sum
property reflects shift invariance:

$$
\operatorname{softmax}(\mathbf{z}+c\mathbf{1})
=\operatorname{softmax}(\mathbf{z}).
$$

### 2.5 Smoothness, Lipschitz Constants, and Monotonicity

An activation is smooth if it has continuous derivatives of the needed order.
ReLU is continuous but not differentiable at zero. GELU and SiLU are smooth.
Sigmoid, tanh, softplus, and softmax are smooth on their domains.

An activation is Lipschitz if there exists $L$ such that

$$
\lvert \phi(x)-\phi(y)\rvert\le L\lvert x-y\rvert.
$$

For differentiable scalar activations, a bounded derivative gives a Lipschitz
constant. ReLU is 1-Lipschitz. Sigmoid is $1/4$-Lipschitz. Tanh is
1-Lipschitz.

## 3. Classical Activations

### 3.1 Sigmoid

The sigmoid is

$$
\sigma(x)=\frac{1}{1+\exp(-x)}.
$$

Its derivative is

$$
\sigma'(x)=\sigma(x)(1-\sigma(x)).
$$

The derivative is largest at $x=0$ and approaches zero in both tails. This is
why sigmoid works well as a gate but poorly as a default hidden activation in
deep feedforward networks.

### 3.2 Tanh

The hyperbolic tangent is

$$
\tanh(x)=\frac{\exp(x)-\exp(-x)}{\exp(x)+\exp(-x)}.
$$

Its derivative is

$$
\frac{d}{dx}\tanh(x)=1-\tanh^2(x).
$$

Tanh is zero-centered, unlike sigmoid, which often makes optimization easier.
But it still saturates for large $\lvert x\rvert$.

### 3.3 Softplus

Softplus is a smooth approximation to ReLU:

$$
\operatorname{softplus}(x)=\log(1+\exp(x)).
$$

Its derivative is sigmoid:

$$
\frac{d}{dx}\operatorname{softplus}(x)=\sigma(x).
$$

Softplus is useful when a strictly positive output is needed, for example a
variance or rate parameter.

### 3.4 Saturation

An activation saturates when its derivative becomes very small over a large
input region. Sigmoid and tanh saturate in both tails:

$$
\lim_{x\to\infty}\sigma'(x)=0,
\qquad
\lim_{x\to-\infty}\sigma'(x)=0.
$$

Saturation creates vanishing gradients. Once a unit enters a saturated region,
large upstream gradients may be multiplied by near-zero local derivatives.

### 3.5 Gate Interpretation

Sigmoid values can be interpreted as soft gates because they lie between zero
and one. If

$$
\mathbf{h}_{\mathrm{out}}
=\mathbf{g}\odot\mathbf{h}_{\mathrm{in}},
\qquad
\mathbf{g}=\sigma(\mathbf{a}),
$$

then each coordinate of $\mathbf{g}$ decides how much of the input passes.
This pattern is foundational for LSTM input, forget, and output gates, and it
reappears in modern gated feedforward blocks.

## 4. ReLU Family

### 4.1 ReLU

ReLU is

$$
\operatorname{ReLU}(x)=\max(0,x).
$$

A common derivative convention is

$$
\operatorname{ReLU}'(x)=
\begin{cases}
1, & x>0,\\
0, & x\le 0.
\end{cases}
$$

ReLU avoids positive-side saturation and creates sparse activations. Its main
failure mode is dead neurons: units whose preactivations remain negative so
their gradients stay zero.

### 4.2 Leaky ReLU

Leaky ReLU uses a small negative slope:

$$
\operatorname{LeakyReLU}_{\alpha}(x)=
\begin{cases}
x, & x>0,\\
\alpha x, & x\le 0.
\end{cases}
$$

Its derivative is $\alpha$ on the negative side and $1$ on the positive side.
This reduces the dead-neuron problem while preserving much of ReLU's behavior.

### 4.3 PReLU

Parametric ReLU learns the negative slope:

$$
\operatorname{PReLU}_{a}(x)=
\begin{cases}
x, & x>0,\\
ax, & x\le 0.
\end{cases}
$$

The slope $a$ becomes a parameter. This can improve flexibility, but it adds a
small risk: if slopes grow uncontrolled, activation scale can drift.

### 4.4 ELU and SELU Preview

ELU uses an exponential negative branch:

$$
\operatorname{ELU}_{\alpha}(x)=
\begin{cases}
x, & x>0,\\
\alpha(\exp(x)-1), & x\le 0.
\end{cases}
$$

SELU scales ELU to encourage self-normalizing behavior under assumptions about
initialization and network structure. These activations are useful historically
and conceptually, though modern large Transformer-style models more commonly
use GELU, SiLU, or gated variants.

### 4.5 Dead Neurons and Sparse Activations

A ReLU neuron is dead when its preactivation is negative for almost all
examples and updates do not move it back into the active region. Causes
include too-large learning rates, biased initialization, distribution shift,
and poor normalization.

Sparse activation is not always bad. ReLU sparsity can improve feature
selectivity and computation. The problem is irreversible inactivity, not
zeros themselves.

## 5. Smooth Modern Activations

### 5.1 GELU

GELU is

$$
\operatorname{GELU}(x)=x\Phi(x),
$$

where $\Phi$ is the standard normal CDF. A common approximation is

$$
\operatorname{GELU}(x)
\approx
0.5x\left(1+\tanh\left(\sqrt{\frac{2}{\pi}}(x+0.044715x^3)\right)\right).
$$

GELU can be read as stochastic-looking gating: larger positive values pass
almost unchanged, large negative values are suppressed, and values near zero
are softly weighted.

### 5.2 SiLU and Swish

SiLU, also called Swish when parameterized, is

$$
\operatorname{SiLU}(x)=x\sigma(x).
$$

Its derivative is

$$
\operatorname{SiLU}'(x)
=\sigma(x)+x\sigma(x)(1-\sigma(x)).
$$

SiLU is smooth and self-gated. It allows small negative outputs, which can
improve gradient flow compared with hard zeroing.

### 5.3 Mish

Mish is

$$
\operatorname{Mish}(x)=x\tanh(\operatorname{softplus}(x)).
$$

Like GELU and SiLU, Mish is smooth and allows a negative tail. It is less
standard in large LLM stacks but useful for understanding the family of smooth
self-gated activations.

### 5.4 Curvature Comparison

Smooth activations have nonzero second derivatives over wider regions. That
changes curvature seen by optimizers. ReLU is piecewise linear, so its second
derivative is zero away from the kink. GELU and SiLU have smooth curvature near
zero, which can give more gradual gradient transitions.

Curvature is not automatically good. It can improve optimization smoothness,
but it also changes gradient scale and may interact with initialization.

### 5.5 Gradient-Flow Effects

The main gradient-flow question is:

$$
\text{typical slope} \times \text{typical weight scale}.
$$

Sigmoid has small slopes in the tails. ReLU has slope one on active units and
zero on inactive units. GELU and SiLU have soft slopes that vary smoothly.
Gated activations multiply by learned factors, so their gradients include both
content and gate paths.

## 6. Gated Activations

### 6.1 GLU

Given two vectors $\mathbf{a},\mathbf{b}\in\mathbb{R}^d$, the gated linear
unit is

$$
\operatorname{GLU}(\mathbf{a},\mathbf{b})
=\mathbf{a}\odot\sigma(\mathbf{b}).
$$

The output is linear content modulated by a sigmoid gate. The derivative has
two paths: one through $\mathbf{a}$ and one through $\mathbf{b}$.

### 6.2 GEGLU

GEGLU replaces the sigmoid gate with GELU:

$$
\operatorname{GEGLU}(\mathbf{a},\mathbf{b})
=\mathbf{a}\odot\operatorname{GELU}(\mathbf{b}).
$$

This gives a smooth non-probability gate. Unlike sigmoid gates, GELU gates are
not constrained to $(0,1)$.

### 6.3 SwiGLU

SwiGLU uses SiLU as the gate:

$$
\operatorname{SwiGLU}(\mathbf{a},\mathbf{b})
=\mathbf{a}\odot\operatorname{SiLU}(\mathbf{b}).
$$

SwiGLU is common in modern Transformer feedforward blocks. The architecture
details belong to Chapter 14 and Chapter 15; here the reusable math is the
bilinear interaction between content and gate streams.

### 6.4 Bilinear Gating

Gated activations introduce multiplicative interactions:

$$
y_i=a_i g_i.
$$

The local derivatives are

$$
\frac{\partial y_i}{\partial a_i}=g_i,
\qquad
\frac{\partial y_i}{\partial g_i}=a_i.
$$

Thus the gate controls content gradients, and the content controls gate
gradients. This is more expressive than an elementwise activation applied to a
single stream.

### 6.5 Why Gated MLPs Help

Gated MLPs can select features conditionally. A standard activation transforms
each coordinate independently. A gated activation lets one projection decide
which coordinates of another projection matter. This adds a simple form of
feature interaction without attention.

## 7. Vector Activations

### 7.1 Softmax

Softmax maps logits to probabilities:

$$
s_i=\frac{\exp(z_i)}{\sum_{j=1}^{C}\exp(z_j)}.
$$

It is shift-invariant:

$$
\operatorname{softmax}(\mathbf{z}+c\mathbf{1})
=\operatorname{softmax}(\mathbf{z}).
$$

Softmax is used in multiclass classification, attention weights, categorical
sampling, and contrastive objectives.

### 7.2 Temperature Softmax

Temperature softmax is

$$
s_i(\tau)
=\frac{\exp(z_i/\tau)}{\sum_j\exp(z_j/\tau)}.
$$

Small $\tau$ sharpens the distribution. Large $\tau$ flattens it. Temperature
changes both probabilities and gradient scale, because derivatives include the
factor $1/\tau$.

### 7.3 Softmax Jacobian

The softmax Jacobian is

$$
J=\operatorname{diag}(\mathbf{s})-\mathbf{s}\mathbf{s}^\top.
$$

The diagonal entries are $s_i(1-s_i)$. Off-diagonal entries are $-s_is_j$.
This coupling is why increasing one logit decreases other probabilities.

### 7.4 Sparsemax and Entmax Preview

Softmax assigns positive probability to every class. Sparse alternatives such
as sparsemax and entmax can assign exact zeros. These are useful when sparse
probability distributions are desirable, but their full treatment belongs to
specialized sequence and attention contexts.

### 7.5 Attention-Output Preview

In attention, softmax converts scores into weights:

$$
A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right).
$$

The full attention mechanism belongs in
[Attention Mechanism Math](../../15-Math-for-LLMs/03-Attention-Mechanism-Math/notes.md).
Here the important activation fact is that softmax can saturate when logits
have large variance, which motivates scaling by $\sqrt{d_k}$.

## 8. Initialization and Gradient Flow

### 8.1 Activation Variance

If activations grow layer by layer, later layers may explode. If activations
shrink layer by layer, signals may vanish. Initialization chooses weight
variance to keep preactivation and activation variance controlled.

### 8.2 Xavier and He Initialization

Xavier initialization is designed for roughly symmetric activations such as
tanh:

$$
\operatorname{Var}(W_{ij})\approx\frac{2}{n_{\mathrm{in}}+n_{\mathrm{out}}}.
$$

He initialization is designed for ReLU-like activations:

$$
\operatorname{Var}(W_{ij})\approx\frac{2}{n_{\mathrm{in}}}.
$$

The factor of two appears because ReLU zeroes roughly half of a symmetric
input distribution.

### 8.3 Vanishing Gradients

If many derivatives satisfy $\lvert \phi'(z)\rvert<1$, products of derivatives
can shrink exponentially with depth. This was a major reason sigmoid and tanh
hidden activations were hard to train in deep networks without careful
initialization, normalization, or residual paths.

### 8.4 Exploding Gradients

Exploding gradients occur when Jacobian products have large singular values.
Activations contribute through their slopes. ReLU slopes are bounded by one,
but weights can still produce exploding products. Smooth activations can have
slopes slightly above one in some regions, so scale control still matters.

### 8.5 Residual Connection Preview

Residual connections create paths where gradients can flow through identity
maps:

$$
\mathbf{h}^{[l+1]}=\mathbf{h}^{[l]}+F(\mathbf{h}^{[l]}).
$$

This reduces dependence on long products of activation derivatives. The full
model-specific treatment belongs in Chapter 14.

## 9. Applications in Machine Learning

### 9.1 CNNs

CNNs historically use ReLU-family activations because they are cheap,
piecewise-linear, and sparse. Leaky variants can help when dead filters appear.

### 9.2 RNN Gates

RNNs use sigmoid gates and tanh candidate states. The bounded range controls
state updates, while gates regulate memory. The full recurrent architecture
math belongs in [RNN and LSTM Math](../../14-Math-for-Specific-Models/04-RNN-and-LSTM-Math/notes.md).

### 9.3 Transformer Feedforward Blocks

Transformers commonly use GELU, SiLU, or gated variants inside feedforward
blocks. The activation controls token-wise nonlinear transformation between
attention layers. Chapter 14 and Chapter 15 cover the full block design.

### 9.4 Binary Outputs

Sigmoid maps logits to Bernoulli probabilities for binary classification.
Training should usually use a logits-based BCE loss rather than manually
applying sigmoid and then taking logs.

### 9.5 Probability Heads

Softmax maps class logits to categorical probabilities. Its coupling and
shift-invariance make it the standard final activation for multiclass
probability heads and many contrastive objectives.

## 10. Common Mistakes

| # | Mistake | Why It Is Wrong | Fix |
| --- | --- | --- | --- |
| 1 | Stacking affine layers without activations | The composition is still affine | Insert nonlinear activations between affine maps |
| 2 | Using sigmoid hidden units in a deep plain MLP | Saturation causes vanishing gradients | Prefer ReLU/GELU/SiLU plus normalization when appropriate |
| 3 | Calling softmax elementwise | Softmax couples coordinates through the denominator | Use the full Jacobian when deriving gradients |
| 4 | Forgetting softmax shift-invariance | Large logits may overflow | Subtract the max before exponentiating |
| 5 | Treating ReLU derivative at zero as important | The convention rarely changes training | State the chosen convention and move on |
| 6 | Ignoring activation scale in initialization | Variance can explode or vanish | Match initialization to activation family |
| 7 | Assuming smooth activations are always better | Smoothness changes scale and cost | Compare gradients, not only curves |
| 8 | Applying sigmoid before a logits BCE loss | The loss applies sigmoid internally | Pass raw logits to fused logits losses |
| 9 | Confusing GLU gates with probabilities | GELU/SwiGLU gates are not constrained to $(0,1)$ | Interpret them as multiplicative feature selectors |
| 10 | Using temperature without considering gradients | Temperature rescales derivatives | Retune or monitor gradient norms |

## 11. Exercises

1. (*) Derive $\sigma'(x)=\sigma(x)(1-\sigma(x))$.
2. (*) Derive $\frac{d}{dx}\tanh(x)=1-\tanh^2(x)$.
3. (*) Compute ReLU, Leaky ReLU, and their derivatives for a vector.
4. (**) Show that stacked affine layers without activations collapse to one affine map.
5. (**) Implement stable softmax and verify shift-invariance.
6. (**) Derive the softmax Jacobian for a three-class vector.
7. (**) Compare GELU and SiLU curves and derivatives numerically.
8. (***) Compute gradients for a GLU and explain the two gradient paths.
9. (***) Estimate activation variance after Xavier and He initialization.
10. (***) Diagnose a dead-ReLU layer from activation and gradient statistics.

## 12. Why This Matters for AI

| Activation concept | AI impact |
| --- | --- |
| Nonlinearity | Gives networks expressive function classes |
| Sigmoid gates | Controls memory and binary probabilities |
| Tanh | Bounded hidden states and centered gates |
| ReLU | Sparse, cheap, deep-friendly hidden activations |
| GELU | Smooth stochastic-style gating in Transformer blocks |
| SiLU/SwiGLU | Self-gated and multiplicative feedforward transformations |
| Softmax | Converts scores into probabilities and attention weights |
| Softmax temperature | Controls sharpness in classification, sampling, and contrastive learning |
| Activation derivatives | Determine gradient propagation and trainability |
| Initialization coupling | Keeps activation scale stable across depth |

## 13. Conceptual Bridge

Loss functions define the gradient at the output. Activation functions decide
how that gradient passes through each hidden layer.

```text
Loss gradient
    -> output head
    -> activation derivatives
    -> layer Jacobians
    -> earlier parameters
```

Next, [Normalization Techniques](../03-Normalization-Techniques/notes.md)
studies how to control the statistics of those activations so deep networks
remain trainable.

## References

- Rumelhart, D. E., Hinton, G. E., and Williams, R. J. (1986). Learning representations by back-propagating errors.
- Nair, V., and Hinton, G. E. (2010). Rectified Linear Units Improve Restricted Boltzmann Machines.
- Glorot, X., Bordes, A., and Bengio, Y. (2011). Deep Sparse Rectifier Neural Networks.
- He, K., Zhang, X., Ren, S., and Sun, J. (2015). Delving Deep into Rectifiers.
- Hendrycks, D., and Gimpel, K. (2016). Gaussian Error Linear Units.
- Elfwing, S., Uchibe, E., and Doya, K. (2018). Sigmoid-Weighted Linear Units.
- Shazeer, N. (2020). GLU Variants Improve Transformer.
