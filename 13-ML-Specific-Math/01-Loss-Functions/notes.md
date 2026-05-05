[Back to Curriculum](../../README.md) | [Previous: Kernel Methods](../../12-Functional-Analysis/03-Kernel-Methods/notes.md) | [Next: Activation Functions](../02-Activation-Functions/notes.md)

---

# Loss Functions

> _"A loss is the contract between the task we wish we had and the gradient signal we can actually optimize."_

## Overview

Loss functions are the mathematical interface between data, models, and
optimization. A model does not directly optimize "accuracy", "helpfulness",
"semantic similarity", or "good generation". It optimizes a scalar objective
whose gradients tell parameters how to move. Choosing a loss is therefore not a
minor implementation detail. It encodes assumptions about noise, uncertainty,
robustness, ranking, calibration, geometry, and what kinds of mistakes should
matter most.

This section treats loss functions as reusable ML primitives. The canonical
home for entropy, KL divergence, and cross-entropy is Chapter 9; the canonical
home for neural-network architectures is Chapter 14; the canonical home for
LLM probability and decoding is Chapter 15. Here we focus on what belongs
specifically to the ML loss layer: empirical risk, reductions, masking,
regression losses, classification losses, probabilistic losses, contrastive
losses, ranking losses, preference-loss previews, and the numerical details
that decide whether a formula becomes a stable training objective.

The practical goal is simple: after this section, you should be able to look at
a task and ask, "What does this loss assume? What gradients does it produce?
What failure modes does it hide? What code path computes it safely?" That is a
different skill from memorizing a formula table. It is the skill needed to
debug model training.

## Prerequisites

- **Expectation and empirical averages** - [Expectation and Moments](../../06-Probability-Theory/04-Expectation-and-Moments/notes.md)
- **Maximum likelihood estimation** - [Estimation Theory](../../07-Statistics/02-Estimation-Theory/notes.md)
- **Convexity and smoothness** - [Convex Optimization](../../08-Optimization/01-Convex-Optimization/notes.md)
- **Gradient descent and stochastic optimization** - [Gradient Descent](../../08-Optimization/02-Gradient-Descent/notes.md), [Stochastic Optimization](../../08-Optimization/05-Stochastic-Optimization/notes.md)
- **Entropy, KL, and cross-entropy** - [Entropy](../../09-Information-Theory/01-Entropy/notes.md), [KL Divergence](../../09-Information-Theory/02-KL-Divergence/notes.md), [Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md)
- **Numerical stability basics** - [Floating Point Arithmetic](../../10-Numerical-Methods/01-Floating-Point-Arithmetic/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Executable loss geometry, gradient, masking, contrastive, and stability demonstrations |
| [exercises.ipynb](exercises.ipynb) | 10 graded exercises on regression, classification, probabilistic, contrastive, ranking, and masked losses |

## Learning Objectives

After completing this section, you will be able to:

- Define pointwise loss, empirical risk, population risk, and reduced batch loss
- Explain how a loss encodes a statistical noise model and a geometric penalty
- Compare MSE, MAE, Huber, quantile, and log-cosh losses for regression
- Derive binary cross-entropy from a Bernoulli likelihood without confusing logits and probabilities
- Explain hinge, focal, and label-smoothed losses as surrogate or reweighted objectives
- Connect negative log-likelihood, KL objectives, ELBO previews, and proper scoring rules
- Implement contrastive, triplet, InfoNCE, margin ranking, and pairwise preference losses
- Diagnose gradient-scale, masking, class-imbalance, and mixed-precision loss failures
- Use log-sum-exp and fused logit-space formulas for stable classification losses
- Decide which loss belongs to a task and which neighboring chapter owns deeper theory

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 Loss as Training Signal](#11-loss-as-training-signal)
  - [1.2 Loss as Statistical Assumption](#12-loss-as-statistical-assumption)
  - [1.3 Loss as Geometry](#13-loss-as-geometry)
  - [1.4 Loss as Metric Proxy](#14-loss-as-metric-proxy)
  - [1.5 Historical Path](#15-historical-path)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Dataset, Model, and Prediction](#21-dataset-model-and-prediction)
  - [2.2 Pointwise Loss](#22-pointwise-loss)
  - [2.3 Population Risk and Empirical Risk](#23-population-risk-and-empirical-risk)
  - [2.4 Reductions: Sum, Mean, and None](#24-reductions-sum-mean-and-none)
  - [2.5 Masked and Weighted Losses](#25-masked-and-weighted-losses)
  - [2.6 Surrogate Losses](#26-surrogate-losses)
- [3. Regression Losses](#3-regression-losses)
  - [3.1 Mean Squared Error](#31-mean-squared-error)
  - [3.2 Mean Absolute Error](#32-mean-absolute-error)
  - [3.3 Huber Loss](#33-huber-loss)
  - [3.4 Quantile Loss](#34-quantile-loss)
  - [3.5 Log-Cosh and Smooth Robust Losses](#35-log-cosh-and-smooth-robust-losses)
  - [3.6 Robustness and Outliers](#36-robustness-and-outliers)
- [4. Classification Losses](#4-classification-losses)
  - [4.1 Binary Cross-Entropy](#41-binary-cross-entropy)
  - [4.2 Categorical Cross-Entropy Preview](#42-categorical-cross-entropy-preview)
  - [4.3 Hinge and Margin Losses](#43-hinge-and-margin-losses)
  - [4.4 Focal Loss](#44-focal-loss)
  - [4.5 Label Smoothing and Class Weighting](#45-label-smoothing-and-class-weighting)
- [5. Probabilistic Losses](#5-probabilistic-losses)
  - [5.1 Negative Log-Likelihood](#51-negative-log-likelihood)
  - [5.2 Gaussian, Bernoulli, and Categorical Likelihoods](#52-gaussian-bernoulli-and-categorical-likelihoods)
  - [5.3 KL-Based Objectives](#53-kl-based-objectives)
  - [5.4 ELBO Preview](#54-elbo-preview)
  - [5.5 Calibration and Proper Scoring Rules](#55-calibration-and-proper-scoring-rules)
- [6. Contrastive and Ranking Losses](#6-contrastive-and-ranking-losses)
  - [6.1 Contrastive Pair Loss](#61-contrastive-pair-loss)
  - [6.2 Triplet Loss](#62-triplet-loss)
  - [6.3 InfoNCE](#63-infonce)
  - [6.4 Margin Ranking](#64-margin-ranking)
  - [6.5 Preference-Loss Preview](#65-preference-loss-preview)
- [7. Loss Geometry and Optimization](#7-loss-geometry-and-optimization)
  - [7.1 Convex and Nonconvex Losses](#71-convex-and-nonconvex-losses)
  - [7.2 Smoothness and Subgradients](#72-smoothness-and-subgradients)
  - [7.3 Gradient Scale](#73-gradient-scale)
  - [7.4 Hessian Intuition](#74-hessian-intuition)
  - [7.5 Loss Balancing](#75-loss-balancing)
- [8. Implementation Stability](#8-implementation-stability)
  - [8.1 Logits Versus Probabilities](#81-logits-versus-probabilities)
  - [8.2 Log-Sum-Exp](#82-log-sum-exp)
  - [8.3 Masking Sequence Losses](#83-masking-sequence-losses)
  - [8.4 Ignore Index](#84-ignore-index)
  - [8.5 Mixed Precision and Loss Scaling](#85-mixed-precision-and-loss-scaling)
- [9. Applications in Machine Learning](#9-applications-in-machine-learning)
  - [9.1 Regression](#91-regression)
  - [9.2 Classification](#92-classification)
  - [9.3 Metric Learning](#93-metric-learning)
  - [9.4 Detection and Imbalanced Data](#94-detection-and-imbalanced-data)
  - [9.5 Alignment and Preference Tuning](#95-alignment-and-preference-tuning)
- [10. Common Mistakes](#10-common-mistakes)
- [11. Exercises](#11-exercises)
- [12. Why This Matters for AI](#12-why-this-matters-for-ai)
- [13. Conceptual Bridge](#13-conceptual-bridge)
- [References](#references)

---

## 1. Intuition

### 1.1 Loss as Training Signal

A supervised learning dataset is usually written as

$$
\mathcal{D}=\{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^{n}.
$$

A model with parameters $\boldsymbol{\theta}$ produces predictions
$f_{\boldsymbol{\theta}}(\mathbf{x})$. The loss converts each prediction and
target into a scalar penalty:

$$
\ell(f_{\boldsymbol{\theta}}(\mathbf{x}^{(i)}), y^{(i)}) \in \mathbb{R}_{\ge 0}.
$$

The optimizer never sees the task directly. It sees this scalar and its
gradient. If the loss gives large gradients to unimportant errors, training
will chase those errors. If the loss gives small gradients to important errors,
training may look stable while the model learns the wrong behavior.

The loss is therefore a feedback channel:

```text
data + target -> model prediction -> loss -> gradient -> parameter update
```

The shape of that channel matters. MSE gives errors a gradient proportional to
their size. MAE gives all nonzero errors roughly equal magnitude. Cross-entropy
gives especially strong pressure when the model assigns tiny probability to the
target. Focal loss reduces pressure from easy examples. Contrastive loss
compares positives against negatives, so its signal depends on the batch.

### 1.2 Loss as Statistical Assumption

Many common losses are negative log-likelihoods under a noise model. If

$$
y = f_{\boldsymbol{\theta}}(\mathbf{x}) + \epsilon,
$$

then a Gaussian noise assumption leads to squared error. A Laplace noise
assumption leads to absolute error. A Bernoulli output model leads to binary
cross-entropy. A categorical output model leads to categorical
cross-entropy.

This does not mean every loss must be probabilistic. Hinge loss, triplet loss,
and ranking losses are often better understood as surrogate geometric
objectives. But the statistical reading is useful because it reveals hidden
assumptions. MSE assumes large residuals are extremely unlikely. MAE assumes
heavier tails. Focal loss assumes easy examples should contribute less. A
preference loss assumes pairwise comparisons are more reliable than absolute
scores.

### 1.3 Loss as Geometry

A loss carves a geometry on prediction space. For scalar residual
$r=\hat{y}-y$, MSE is quadratic:

$$
\ell_{\mathrm{MSE}}(r)=r^2.
$$

Its level sets in two residual dimensions are circles. MAE has diamond-shaped
level sets:

$$
\ell_{\mathrm{MAE}}(\mathbf{r})=\lVert \mathbf{r} \rVert_1.
$$

Huber loss behaves like a quadratic near zero and like an absolute value far
from zero. This geometry matters because optimization follows gradients of
that surface. The same model, same data, and same optimizer can behave very
differently if the loss reshapes the surface.

### 1.4 Loss as Metric Proxy

Most task metrics are not directly optimized. Accuracy is piecewise constant
with respect to logits. F1 score depends on discrete thresholding and dataset
level counts. BLEU, ROUGE, exact match, ranking metrics, and human preference
judgments are usually not smooth scalar functions of model parameters.

Losses are often surrogate objectives: differentiable substitutes that are
easier to optimize than the final metric. A good surrogate should align with
the desired task behavior. Cross-entropy is a good surrogate for probabilistic
classification. Hinge loss is a margin surrogate. InfoNCE is a contrastive
surrogate for representation alignment. DPO-style objectives turn pairwise
preference data into a classification-like objective.

The danger is proxy mismatch. A model can reduce loss while the true metric
stagnates or worsens. In real training, loss curves must be read together with
task metrics, calibration, and qualitative errors.

### 1.5 Historical Path

Least squares appeared long before modern machine learning because it gives a
clean analytic solution and a Gaussian-noise interpretation. Logistic and
softmax losses became central when probabilistic classification became the
default framing. Hinge loss powered large-margin methods such as support vector
machines. Deep learning made cross-entropy and MSE everyday objectives. Modern
self-supervised learning popularized contrastive losses such as InfoNCE.
Detection systems introduced focal loss to handle extreme imbalance. Recent
alignment systems use pairwise and preference losses to steer language models
without requiring a scalar reward at every token.

This history matters because no single loss is "the ML loss." Losses evolved
to match data regimes, model families, and engineering constraints.

## 2. Formal Definitions

### 2.1 Dataset, Model, and Prediction

Let

$$
\mathcal{D}=\{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^{n}
$$

be a supervised dataset with inputs $\mathbf{x}^{(i)} \in \mathcal{X}$ and
targets $y^{(i)} \in \mathcal{Y}$. A model maps inputs to predictions:

$$
f_{\boldsymbol{\theta}}:\mathcal{X}\to\mathcal{A}.
$$

The prediction space $\mathcal{A}$ depends on the task. For regression,
$\mathcal{A}=\mathbb{R}^d$. For binary classification, a model may output a
logit $z\in\mathbb{R}$ or a probability $\hat{p}\in(0,1)$. For multiclass
classification, it may output logits $\mathbf{z}\in\mathbb{R}^C$ or a
probability vector $\hat{\mathbf{p}}\in\Delta^{C-1}$.

Examples:

- Scalar regression: $f_{\boldsymbol{\theta}}(\mathbf{x})=\hat{y}\in\mathbb{R}$.
- Binary classification from logits: $f_{\boldsymbol{\theta}}(\mathbf{x})=z$ and $\hat{p}=\sigma(z)$.
- Multiclass classification from logits: $f_{\boldsymbol{\theta}}(\mathbf{x})=\mathbf{z}$ and $\hat{\mathbf{p}}=\operatorname{softmax}(\mathbf{z})$.

Non-examples:

- The optimizer update rule is not itself the prediction.
- A metric such as validation accuracy is not the model output.

### 2.2 Pointwise Loss

A pointwise loss is a function

$$
\ell:\mathcal{A}\times\mathcal{Y}\to\mathbb{R}.
$$

It assigns a scalar penalty to one prediction-target pair. Usually we want
$\ell(\hat{a},y)\ge 0$ and lower values to mean better predictions, but some
objectives are defined up to constants or signs. For example, maximizing
log-likelihood is equivalent to minimizing negative log-likelihood.

Examples:

- Squared error: $\ell(\hat{y},y)=(\hat{y}-y)^2$.
- Absolute error: $\ell(\hat{y},y)=\lvert \hat{y}-y \rvert$.
- Binary cross-entropy: $\ell(\hat{p},y)=-y\log\hat{p}-(1-y)\log(1-\hat{p})$.

Non-examples:

- A dataset split is not a loss.
- A model architecture is not a loss, even when it contains an output head.

### 2.3 Population Risk and Empirical Risk

The population risk is the expected loss under the true data distribution:

$$
R(\boldsymbol{\theta})=
\mathbb{E}_{(\mathbf{x},y)\sim p_{\mathrm{data}}}
\left[\ell(f_{\boldsymbol{\theta}}(\mathbf{x}),y)\right].
$$

The empirical risk replaces the unknown distribution with the observed sample:

$$
\hat{R}_n(\boldsymbol{\theta})=
\frac{1}{n}\sum_{i=1}^{n}
\ell(f_{\boldsymbol{\theta}}(\mathbf{x}^{(i)}),y^{(i)}).
$$

Training usually minimizes a regularized empirical objective:

$$
J(\boldsymbol{\theta})=
\hat{R}_n(\boldsymbol{\theta})+\lambda\Omega(\boldsymbol{\theta}).
$$

The loss $\ell$ defines the data-fit term. The regularizer $\Omega$ is covered
in [Regularization Methods](../../08-Optimization/08-Regularization-Methods/notes.md).

### 2.4 Reductions: Sum, Mean, and None

Deep learning libraries often compute a vector of per-example losses and then
apply a reduction:

$$
\boldsymbol{\ell}=(\ell_1,\ldots,\ell_B).
$$

Common reductions are:

$$
\operatorname{sum}(\boldsymbol{\ell})=\sum_{i=1}^{B}\ell_i,
$$

$$
\operatorname{mean}(\boldsymbol{\ell})=\frac{1}{B}\sum_{i=1}^{B}\ell_i,
$$

and `none`, which returns the unreduced vector. The reduction changes gradient
scale. If a batch mean is used, doubling the batch size does not double the
expected gradient magnitude. If a batch sum is used, it does.

This matters for distributed training, gradient accumulation, sequence
masking, and loss balancing. Two codebases can implement the same formula but
train differently because one averages over tokens and the other averages over
sequences.

### 2.5 Masked and Weighted Losses

A masked loss ignores selected examples or tokens. Let $m_i\in\{0,1\}$ be a
mask. The masked mean is

$$
\mathcal{L}_{\mathrm{masked}}
=\frac{\sum_{i=1}^{B}m_i\ell_i}{\sum_{i=1}^{B}m_i+\epsilon}.
$$

A weighted loss uses weights $w_i\ge 0$:

$$
\mathcal{L}_{\mathrm{weighted}}
=\frac{\sum_{i=1}^{B}w_i\ell_i}{\sum_{i=1}^{B}w_i+\epsilon}.
$$

Examples:

- Ignore padding tokens in a sequence loss.
- Upweight rare classes in imbalanced classification.
- Weight recent samples more heavily in nonstationary data.

Non-examples:

- Setting a target to zero is not the same as masking it.
- Dropping hard examples without recording the rule is not a principled mask.

### 2.6 Surrogate Losses

A surrogate loss is optimized because the true target metric is nondifferentiable
or inconvenient. For binary classification, the zero-one loss is

$$
\ell_{0/1}(\hat{y},y)=\mathbb{1}[\hat{y}\ne y].
$$

It is not useful for gradient-based training because it is flat almost
everywhere. Logistic loss and hinge loss are smooth or subdifferentiable
surrogates. A good surrogate upper-bounds, calibrates, or consistently aligns
with the desired metric under appropriate assumptions.

## 3. Regression Losses

### 3.1 Mean Squared Error

For scalar regression, the squared error is

$$
\ell_{\mathrm{MSE}}(\hat{y},y)=(\hat{y}-y)^2.
$$

For vector predictions,

$$
\ell_{\mathrm{MSE}}(\hat{\mathbf{y}},\mathbf{y})
=\lVert \hat{\mathbf{y}}-\mathbf{y}\rVert_2^2.
$$

Its derivative with respect to $\hat{y}$ is

$$
\frac{\partial \ell_{\mathrm{MSE}}}{\partial \hat{y}}
=2(\hat{y}-y).
$$

The gradient grows linearly with residual size. That is useful when large
errors should dominate, but harmful when labels contain outliers. Under a
Gaussian likelihood with fixed variance, minimizing MSE is equivalent to
maximizing likelihood up to constants:

$$
y\mid \mathbf{x}\sim \mathcal{N}(f_{\boldsymbol{\theta}}(\mathbf{x}),\sigma^2).
$$

Examples:

- Predicting a clean physical measurement with roughly Gaussian noise.
- Training an autoencoder with continuous normalized pixel targets.
- Denoising under a simple isotropic Gaussian residual model.

Non-examples:

- Heavy-tailed target noise where a few labels are corrupted.
- Classification labels represented as integers, such as class `7`.

### 3.2 Mean Absolute Error

The absolute error is

$$
\ell_{\mathrm{MAE}}(\hat{y},y)=\lvert \hat{y}-y\rvert.
$$

For nonzero residual $r=\hat{y}-y$,

$$
\frac{\partial \ell_{\mathrm{MAE}}}{\partial \hat{y}}
=\operatorname{sign}(r).
$$

MAE is less sensitive to outliers than MSE because the gradient magnitude does
not grow with residual size. The price is a nondifferentiable corner at zero
and less aggressive correction of large errors. Under a Laplace noise model,
MAE is the negative log-likelihood up to constants.

### 3.3 Huber Loss

Huber loss blends MSE near zero with MAE in the tails:

$$
\ell_{\delta}(r)=
\begin{cases}
\frac{1}{2}r^2, & \lvert r\rvert\le \delta,\\
\delta(\lvert r\rvert-\frac{1}{2}\delta), & \lvert r\rvert>\delta.
\end{cases}
$$

Its derivative is

$$
\ell_{\delta}'(r)=
\begin{cases}
r, & \lvert r\rvert\le \delta,\\
\delta\operatorname{sign}(r), & \lvert r\rvert>\delta.
\end{cases}
$$

The parameter $\delta$ defines the residual scale at which the loss stops
being quadratic. Small $\delta$ behaves more like MAE. Large $\delta$ behaves
more like MSE.

### 3.4 Quantile Loss

Quantile loss estimates conditional quantiles rather than conditional means.
For quantile level $\tau\in(0,1)$ and residual $r=y-\hat{q}_{\tau}$:

$$
\ell_{\tau}(r)=
\max(\tau r,(\tau-1)r).
$$

When $\tau=0.5$, this is proportional to MAE and estimates the median. When
$\tau=0.9$, over-prediction and under-prediction are penalized asymmetrically
so the model learns a high conditional quantile.

This is useful when uncertainty intervals matter. A model can output
$\hat{q}_{0.1}$, $\hat{q}_{0.5}$, and $\hat{q}_{0.9}$ to form a predictive
band without assuming Gaussian residuals.

### 3.5 Log-Cosh and Smooth Robust Losses

The log-cosh loss is

$$
\ell(r)=\log(\cosh r).
$$

For small $r$, $\log(\cosh r)\approx \frac{1}{2}r^2$. For large $\lvert r\rvert$,
it grows roughly like $\lvert r\rvert-\log 2$. Thus it behaves like MSE near
zero and like MAE in the tails, while remaining smooth everywhere.

Smooth robust losses are useful when second-order approximations or stable
automatic differentiation matter. They avoid the sharp kink of MAE and Huber,
but they may be less interpretable than Huber's explicit threshold.

### 3.6 Robustness and Outliers

The core tradeoff is gradient growth:

| Loss | Tail growth | Gradient tail | Robustness |
| --- | --- | --- | --- |
| MSE | Quadratic | Linear | Low |
| MAE | Linear | Constant | High |
| Huber | Linear after $\delta$ | Clipped | Medium-high |
| Quantile | Linear asymmetric | Constant asymmetric | High |
| Log-cosh | Linear asymptotic | Bounded by tanh | Medium-high |

In ML systems, outliers come from sensor failures, annotation errors,
distribution shift, rare edge cases, and preprocessing bugs. A robust loss can
make training less fragile, but it can also hide important rare failures. The
right choice depends on whether large residuals are noise or signal.

## 4. Classification Losses

### 4.1 Binary Cross-Entropy

For $y\in\{0,1\}$ and predicted probability $\hat{p}\in(0,1)$,

$$
\ell_{\mathrm{BCE}}(\hat{p},y)
=-y\log\hat{p}-(1-y)\log(1-\hat{p}).
$$

If the model outputs a logit $z$, then $\hat{p}=\sigma(z)$. In practice, stable
implementations compute BCE directly from logits:

$$
\ell_{\mathrm{BCELogits}}(z,y)
=\max(z,0)-zy+\log(1+\exp(-\lvert z\rvert)).
$$

This avoids overflow when $\lvert z\rvert$ is large. The derivative with
respect to the logit is

$$
\frac{\partial \ell}{\partial z}=\sigma(z)-y.
$$

That simple form explains why BCE from logits is the standard binary
classification objective.

### 4.2 Categorical Cross-Entropy Preview

For a one-hot target $\mathbf{y}\in\Delta^{C-1}$ and predicted probability
$\hat{\mathbf{p}}\in\Delta^{C-1}$,

$$
\ell_{\mathrm{CE}}(\hat{\mathbf{p}},\mathbf{y})
=-\sum_{c=1}^{C}y_c\log \hat{p}_c.
$$

The full information-theoretic treatment belongs in
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md). Here
the ML-specific point is implementation: most models output logits
$\mathbf{z}$, not probabilities. Stable CE computes

$$
\ell(\mathbf{z},c)
=-\mathbf{z}_c+\log\sum_{j=1}^{C}\exp(z_j)
$$

using log-sum-exp. The logit gradient is

$$
\nabla_{\mathbf{z}}\ell=\operatorname{softmax}(\mathbf{z})-\mathbf{y}.
$$

### 4.3 Hinge and Margin Losses

For binary labels $y\in\{-1,+1\}$ and score $s$, hinge loss is

$$
\ell_{\mathrm{hinge}}(s,y)=\max(0,1-ys).
$$

It penalizes examples inside the margin and gives zero loss to examples that
are correctly classified with margin at least one. Multiclass margin losses
compare the correct class score against competing class scores.

Hinge loss is not probabilistic. It does not ask for calibrated probabilities.
It asks for separation. That makes it natural for large-margin classifiers and
ranking-style systems.

### 4.4 Focal Loss

Focal loss was introduced for dense object detection, where easy background
examples overwhelm rare foreground examples. For binary classification:

$$
\ell_{\mathrm{focal}}
=-\alpha_t(1-p_t)^{\gamma}\log p_t,
$$

where $p_t=\hat{p}$ if $y=1$ and $p_t=1-\hat{p}$ if $y=0$. The factor
$(1-p_t)^\gamma$ downweights easy examples. When $\gamma=0$, focal loss reduces
to weighted BCE.

The key tuning question is whether class imbalance is the real bottleneck. If
labels are noisy, focal loss may overemphasize mislabeled hard examples.

### 4.5 Label Smoothing and Class Weighting

Label smoothing replaces a hard one-hot target with a softened target:

$$
\mathbf{y}^{\mathrm{smooth}}
=(1-\epsilon)\mathbf{y}
+\epsilon\frac{\mathbf{1}}{C}.
$$

It discourages overconfident predictions and can improve calibration. Class
weighting changes the contribution of classes:

$$
\ell_{\mathrm{weighted}}(\mathbf{z},c)
=w_c\,\ell_{\mathrm{CE}}(\mathbf{z},c).
$$

These tools solve different problems. Label smoothing changes the target
distribution. Class weighting changes the dataset objective. Combining them
without thinking can produce unintuitive gradients.

## 5. Probabilistic Losses

### 5.1 Negative Log-Likelihood

Negative log-likelihood is the general template:

$$
\ell_{\mathrm{NLL}}(\boldsymbol{\theta})
=-\log p_{\boldsymbol{\theta}}(y\mid\mathbf{x}).
$$

Over a dataset,

$$
\mathcal{L}_{\mathrm{NLL}}
=-\frac{1}{n}\sum_{i=1}^{n}
\log p_{\boldsymbol{\theta}}(y^{(i)}\mid\mathbf{x}^{(i)}).
$$

MSE, BCE, and CE are special cases when the likelihood family is Gaussian,
Bernoulli, or Categorical. NLL is the right abstraction when the model outputs
parameters of a distribution rather than a single point prediction.

### 5.2 Gaussian, Bernoulli, and Categorical Likelihoods

Gaussian regression with fixed variance:

$$
-\log p(y\mid\mu)
=\frac{1}{2\sigma^2}(y-\mu)^2+\text{constant}.
$$

Bernoulli classification:

$$
-\log p(y\mid p)
=-y\log p-(1-y)\log(1-p).
$$

Categorical classification:

$$
-\log p(y=c\mid\mathbf{p})
=-\log p_c.
$$

These three formulas explain much of supervised learning. But the likelihood
must match the target. If the target is ordinal, censored, multimodal, or
heavy-tailed, a simple Gaussian or Categorical model may be the wrong contract.

### 5.3 KL-Based Objectives

KL divergence often appears inside losses:

$$
D_{\mathrm{KL}}(p\Vert q)
=\mathbb{E}_{p}\left[\log\frac{p(X)}{q(X)}\right].
$$

In supervised learning, minimizing cross-entropy with fixed target
distribution $p$ is equivalent to minimizing $D_{\mathrm{KL}}(p\Vert q)$,
because $H(p)$ does not depend on the model. In distillation, KL can compare
teacher and student distributions. In RLHF-style training, a KL penalty can
discourage a policy from drifting too far from a reference model.

Full KL theory belongs in
[KL Divergence](../../09-Information-Theory/02-KL-Divergence/notes.md). Here
the practical point is that KL direction matters. $D_{\mathrm{KL}}(p\Vert q)$
and $D_{\mathrm{KL}}(q\Vert p)$ produce different behavior.

### 5.4 ELBO Preview

Variational models often maximize an evidence lower bound:

$$
\mathcal{L}_{\mathrm{ELBO}}
=
\mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})}
\left[\log p_{\boldsymbol{\theta}}(\mathbf{x}\mid\mathbf{z})\right]
-D_{\mathrm{KL}}
\left(q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})
\Vert p(\mathbf{z})\right).
$$

As a loss to minimize, one usually takes the negative ELBO. The reconstruction
term is a likelihood loss. The KL term regularizes the approximate posterior.
The full generative-model development belongs in
[Probabilistic Models](../../14-Math-for-Specific-Models/03-Probabilistic-Models/notes.md)
and [Generative Models](../../14-Math-for-Specific-Models/07-Generative-Models/notes.md).

### 5.5 Calibration and Proper Scoring Rules

A scoring rule rewards probabilistic forecasts. A strictly proper scoring rule
is minimized in expectation by reporting the true distribution. Log loss is
strictly proper, which is one reason cross-entropy is central to probabilistic
classification.

Calibration asks whether predicted probabilities match observed frequencies.
If a classifier says "0.8" on many examples, about 80 percent should be
positive. A low loss often helps calibration, but low loss and good
calibration are not identical. Temperature scaling, label smoothing, and
post-hoc calibration methods modify this behavior.

## 6. Contrastive and Ranking Losses

### 6.1 Contrastive Pair Loss

Metric learning often starts with pairs. Let
$\mathbf{u}=g_{\boldsymbol{\theta}}(\mathbf{x}_a)$ and
$\mathbf{v}=g_{\boldsymbol{\theta}}(\mathbf{x}_b)$ be embeddings. For label
$s\in\{0,1\}$ indicating similar pairs, a simple contrastive loss is

$$
\ell
=s\,\lVert \mathbf{u}-\mathbf{v}\rVert_2^2
+(1-s)\max(0,m-\lVert \mathbf{u}-\mathbf{v}\rVert_2)^2.
$$

Positive pairs are pulled together. Negative pairs are pushed apart until they
reach margin $m$. This loss depends on distance geometry, so normalization and
embedding scale matter.

### 6.2 Triplet Loss

Triplet loss uses an anchor $\mathbf{a}$, positive $\mathbf{p}$, and negative
$\mathbf{n}$:

$$
\ell_{\mathrm{triplet}}
=\max\left(
0,
\lVert \mathbf{a}-\mathbf{p}\rVert_2^2
-\lVert \mathbf{a}-\mathbf{n}\rVert_2^2
+m
\right).
$$

The loss is zero when the positive is closer than the negative by margin $m$.
Its effectiveness depends heavily on mining useful negatives. Random negatives
may be too easy; extremely hard negatives may be mislabeled or destabilizing.

### 6.3 InfoNCE

InfoNCE is a softmax-style contrastive objective. For query $\mathbf{q}$,
positive key $\mathbf{k}^{+}$, and negative keys
$\{\mathbf{k}^{-}_j\}$:

$$
\ell_{\mathrm{InfoNCE}}
=-\log
\frac{\exp(\operatorname{sim}(\mathbf{q},\mathbf{k}^{+})/\tau)}
{\exp(\operatorname{sim}(\mathbf{q},\mathbf{k}^{+})/\tau)
+\sum_j \exp(\operatorname{sim}(\mathbf{q},\mathbf{k}^{-}_j)/\tau)}.
$$

The temperature $\tau$ controls sharpness. Smaller $\tau$ makes the softmax
more selective and gradients more concentrated. InfoNCE underlies many
self-supervised and multimodal alignment methods.

### 6.4 Margin Ranking

For scores $s_a$ and $s_b$ with target preference $y\in\{-1,+1\}$, margin
ranking loss can be written as

$$
\ell_{\mathrm{rank}}
=\max(0,m-y(s_a-s_b)).
$$

It asks the preferred item to score higher by at least margin $m$. This is a
surrogate for ranking metrics that depend on pair order rather than absolute
score values.

### 6.5 Preference-Loss Preview

Preference tuning in modern language models often uses pairs:
$(x,y_w,y_l)$, where $y_w$ is preferred over $y_l$. A DPO-style objective can
be written in terms of log-probability ratios between a trainable policy and a
reference policy:

$$
\ell_{\mathrm{DPO}}
=-\log\sigma\left(
\beta
\left[
\log\frac{\pi_{\boldsymbol{\theta}}(y_w\mid x)}
{\pi_{\mathrm{ref}}(y_w\mid x)}
-
\log\frac{\pi_{\boldsymbol{\theta}}(y_l\mid x)}
{\pi_{\mathrm{ref}}(y_l\mid x)}
\right]\right).
$$

Full alignment training belongs later. The key loss-function idea is that a
pairwise preference can become a differentiable classification-style loss over
relative log probabilities.

## 7. Loss Geometry and Optimization

### 7.1 Convex and Nonconvex Losses

A loss can be convex in predictions but not convex in parameters. Squared
error is convex in $\hat{y}$. Cross-entropy is convex in logits for linear
softmax regression. But once $\hat{y}=f_{\boldsymbol{\theta}}(\mathbf{x})$ is
a deep network output, the objective is usually nonconvex in
$\boldsymbol{\theta}$.

This distinction prevents a common mistake: calling MSE "convex" does not make
deep regression training convex. Convexity must be stated with respect to the
variable being optimized.

### 7.2 Smoothness and Subgradients

MSE and log-cosh are smooth. MAE, hinge, and Huber are nondifferentiable at a
small set of points. Gradient-based systems handle such losses through
subgradients or implementation choices.

The subgradient of $\lvert r\rvert$ at $r=0$ is the interval $[-1,1]$. A
library may choose zero. That choice rarely matters in isolation, but it is a
reminder that a formula and its implementation are not always identical.

### 7.3 Gradient Scale

Different losses produce different gradient magnitudes:

$$
\nabla_{\hat{y}}(\hat{y}-y)^2=2(\hat{y}-y),
$$

$$
\nabla_{\hat{y}}\lvert \hat{y}-y\rvert=\operatorname{sign}(\hat{y}-y),
$$

$$
\nabla_{\mathbf{z}}\ell_{\mathrm{CE}}
=\operatorname{softmax}(\mathbf{z})-\mathbf{y}.
$$

Gradient scale interacts with learning rate. If changing the loss changes the
typical gradient norm by a factor of 100, the old learning rate may no longer
make sense.

### 7.4 Hessian Intuition

Curvature affects step stability. MSE has constant second derivative with
respect to scalar prediction. MAE has zero curvature away from the kink.
Cross-entropy with softmax has Hessian

$$
H=\operatorname{diag}(\hat{\mathbf{p}})
-\hat{\mathbf{p}}\hat{\mathbf{p}}^\top.
$$

This matrix is positive semidefinite and reflects probability uncertainty.
When the model is very confident, curvature concentrates in directions that
change the confident probabilities.

### 7.5 Loss Balancing

Many modern systems optimize a weighted sum:

$$
\mathcal{L}
=\lambda_1\mathcal{L}_1
+\lambda_2\mathcal{L}_2
+\cdots
+\lambda_k\mathcal{L}_k.
$$

Examples include detection models with classification and box regression
terms, VAEs with reconstruction and KL terms, RL systems with reward and KL
penalty terms, and multimodal systems with contrastive and generative losses.

The weights $\lambda_j$ are not cosmetic. They set relative gradient pressure.
If one term has much larger scale, it can dominate training even when its
coefficient looks small.

## 8. Implementation Stability

### 8.1 Logits Versus Probabilities

A logit is an unconstrained real number. A probability is constrained to
$(0,1)$ or the simplex. Many loss APIs expect logits because logits allow
stable fused computation. Passing probabilities to a "from logits" loss applies
the sigmoid or softmax twice. Passing logits to a probability-space loss can
take $\log$ of invalid values.

Rule of thumb:

- Use `BCEWithLogitsLoss` for binary logits.
- Use cross-entropy from logits for multiclass logits.
- Use NLL loss only after a stable `log_softmax`.
- Avoid computing `softmax` and then `log` separately.

### 8.2 Log-Sum-Exp

The stable identity is

$$
\log\sum_j \exp(z_j)
=m+\log\sum_j\exp(z_j-m),
\qquad m=\max_j z_j.
$$

Subtracting $m$ prevents overflow without changing the result. This identity
is the heart of stable softmax cross-entropy, InfoNCE, and many energy-based
losses.

### 8.3 Masking Sequence Losses

For token sequences, let $\ell_{b,t}$ be a token loss and $m_{b,t}$ a mask.
The masked token mean is

$$
\mathcal{L}
=
\frac{\sum_{b,t}m_{b,t}\ell_{b,t}}
{\sum_{b,t}m_{b,t}+\epsilon}.
$$

The denominator should count valid tokens, not batch size, unless the desired
objective is a per-sequence average. This distinction changes how long and
short examples influence training.

### 8.4 Ignore Index

Classification losses often support an `ignore_index`. This is a convenience
for masking labels such as padding. It should not be used as an extra class.
The ignored value should not appear in the softmax vocabulary; it is a label
sentinel telling the loss not to include that position.

### 8.5 Mixed Precision and Loss Scaling

In mixed precision, small gradients can underflow and large logits can
overflow. Stable fused losses reduce the danger, but training may still require
dynamic loss scaling. A loss that returns `nan` is often not mathematically
wrong; it is numerically unprotected.

Debug checklist:

- Check for logits with extreme magnitude.
- Check whether probabilities are exactly 0 or 1 before `log`.
- Check masks and denominators.
- Check whether the loss is averaged twice.
- Check whether class weights contain zeros or huge values.

## 9. Applications in Machine Learning

### 9.1 Regression

Regression losses choose what "center" means. MSE targets conditional means.
MAE targets conditional medians. Quantile loss targets conditional quantiles.
Huber and log-cosh trade efficiency for robustness. For safety-critical
systems, this choice affects uncertainty and tail behavior.

### 9.2 Classification

Classification losses choose whether the model should separate, rank, or
calibrate. Cross-entropy asks for probability matching. Hinge asks for margin.
Focal loss asks for less attention to easy examples. A calibrated medical
classifier and a high-recall detector may need different loss choices.

### 9.3 Metric Learning

Metric learning losses define embedding geometry. Contrastive and triplet
losses shape distances directly. InfoNCE shapes similarity scores relative to
other samples in the batch. This is why batch composition and negative
sampling are part of the loss design, not only data loading.

### 9.4 Detection and Imbalanced Data

Dense detection has many easy negatives. Focal loss became important because
standard cross-entropy can be overwhelmed by those examples. Class weighting,
resampling, and hard-example mining are alternative interventions. They change
the effective training distribution in different ways.

### 9.5 Alignment and Preference Tuning

Preference losses let models learn from comparisons instead of absolute target
labels. Pairwise ranking, Bradley-Terry-style models, and DPO-style objectives
all convert preference data into differentiable signals. The loss is not just
"make answer A likely"; it is "make preferred answer A relatively more likely
than rejected answer B under a controlled reference."

## 10. Common Mistakes

| # | Mistake | Why It Is Wrong | Fix |
| --- | --- | --- | --- |
| 1 | Passing probabilities into a logits loss | The sigmoid or softmax is applied twice | Keep model outputs as logits for fused losses |
| 2 | Taking `log(softmax(z))` manually | This can overflow or underflow | Use stable `log_softmax` or fused CE |
| 3 | Averaging over padded tokens | Padding changes the objective | Use a mask and divide by valid-token count |
| 4 | Treating class IDs as regression targets | Class labels are nominal, not ordered numbers | Use CE or an ordinal loss when order matters |
| 5 | Using MSE for heavy-tailed noisy labels | Outliers dominate gradients | Use MAE, Huber, quantile, or a heavy-tailed likelihood |
| 6 | Comparing losses across different reductions | Sum and mean have different scales | Record reduction and denominator |
| 7 | Forgetting class-weight normalization | Weights can silently change gradient scale | Normalize or retune learning rate |
| 8 | Using focal loss for label noise without care | It emphasizes hard examples, including bad labels | Audit hard examples and tune $\gamma$ |
| 9 | Assuming low loss means good metric | Surrogate mismatch is common | Track task metrics and calibration |
| 10 | Combining loss terms by arbitrary coefficients | The largest gradient term dominates | Monitor per-term values and gradient norms |
| 11 | Ignoring negative-sample construction | Contrastive losses depend on negatives | Treat batch and sampler design as part of the objective |
| 12 | Calling a loss convex without naming the variable | Convex in prediction may be nonconvex in parameters | State convexity with respect to logits, scores, or parameters |

## 11. Exercises

1. (*) Derive the MSE gradient with respect to scalar prediction $\hat{y}$ and explain why outliers dominate.
2. (*) Show that MAE has constant gradient magnitude away from zero and discuss the subgradient at zero.
3. (*) Derive the stable binary cross-entropy-from-logits formula from the probability-space BCE.
4. (**) Implement masked mean loss and show how the denominator changes gradients.
5. (**) Compare MSE, MAE, and Huber gradients for residuals from $-5$ to $5$.
6. (**) Derive the hinge-loss subgradient for $ys<1$, $ys=1$, and $ys>1$.
7. (**) Implement InfoNCE for a similarity matrix and verify that lowering temperature sharpens probabilities.
8. (***) Build a small pairwise preference loss and inspect how the reference model changes the gradient signal.
9. (***) Design a loss for an imbalanced binary detection problem and justify whether BCE weighting or focal loss is more appropriate.
10. (***) Given a multi-task objective, propose a diagnostic for whether one loss term dominates training.

## 12. Why This Matters for AI

| Concept | AI impact |
| --- | --- |
| Empirical risk | Defines what training actually minimizes |
| Reduction choice | Controls gradient scale in batching and distributed training |
| MSE vs MAE vs Huber | Determines robustness to outliers and label noise |
| Cross-entropy from logits | Powers stable probabilistic classification and language modeling |
| Focal loss | Handles dense imbalance in detection and rare-event settings |
| InfoNCE | Drives contrastive representation learning and multimodal alignment |
| Preference loss | Turns human or AI comparisons into trainable objectives |
| Masked loss | Makes sequence and padding-aware training mathematically correct |
| Loss balancing | Controls tradeoffs in multi-objective systems |
| Proper scoring | Connects probabilistic training to calibrated uncertainty |

In modern AI, loss functions are not just the final line of the training
script. They determine what data counts, what examples dominate, how stable the
gradients are, and whether the model is being asked to predict, rank, align,
reconstruct, calibrate, or explore.

## 13. Conceptual Bridge

This section sits after probability, statistics, information theory, and
optimization because loss functions combine all four.

```text
Probability       -> likelihood losses
Statistics        -> empirical risk and estimation
Information theory -> CE, KL, InfoNCE, calibration
Optimization      -> gradients, curvature, stability
ML-specific math   -> losses as reusable training contracts
```

Next, [Activation Functions](../02-Activation-Functions/notes.md) studies the
nonlinear maps that shape gradients inside the model. Losses define the signal
at the output; activations determine how that signal travels backward through
hidden layers.

## References

- Legendre, A. M. (1805). _Nouvelles methodes pour la determination des orbites des cometes_.
- Huber, P. J. (1964). Robust estimation of a location parameter.
- Vapnik, V. (1995). _The Nature of Statistical Learning Theory_.
- Bishop, C. M. (2006). _Pattern Recognition and Machine Learning_.
- Goodfellow, I., Bengio, Y., and Courville, A. (2016). _Deep Learning_.
- Lin, T.-Y., Goyal, P., Girshick, R., He, K., and Dollar, P. (2017). Focal Loss for Dense Object Detection.
- van den Oord, A., Li, Y., and Vinyals, O. (2018). Representation Learning with Contrastive Predictive Coding.
- Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., and Finn, C. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model.
