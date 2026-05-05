[Back to Curriculum](../../README.md) | [Previous: Normalization Techniques](../03-Normalization-Techniques/notes.md) | [Next: Linear Models](../../14-Math-for-Specific-Models/01-Linear-Models/notes.md)

---

# Sampling Methods

> _"Sampling is how machine learning turns distributions it cannot enumerate into estimates it can compute."_

## Overview

Sampling methods let ML systems approximate expectations, generate candidates,
choose mini-batches, estimate uncertainty, train latent-variable models, draw
negative examples, and explore action spaces. A deterministic formula may say
"take an expectation", but a real model often only has samples. Understanding
sampling means understanding estimator bias, variance, proposal distributions,
effective sample size, Markov chains, differentiable random variables, and the
difference between sampling for training and sampling for generation.

This section is the canonical home for general ML sampling math. Chapter 6 owns
foundational probability and distributions. Chapter 14 owns full generative
model families such as VAEs, GANs, diffusion, and flows. Chapter 15 owns full
LLM decoding. Here we focus on reusable sampling techniques: inverse transform,
categorical sampling, rejection sampling, Monte Carlo estimation, importance
sampling, MCMC, reparameterization, Gumbel tricks, minibatch sampling, negative
sampling, and previews of temperature, top-k, top-p, and diffusion sampling.

The main habit is to ask: "What distribution do I want, what distribution am I
actually sampling from, and what estimator does that produce?"

## Prerequisites

- **Random variables and distributions** - [Introduction and Random Variables](../../06-Probability-Theory/01-Introduction-and-Random-Variables/notes.md), [Common Distributions](../../06-Probability-Theory/02-Common-Distributions/notes.md)
- **Expectation and variance** - [Expectation and Moments](../../06-Probability-Theory/04-Expectation-and-Moments/notes.md)
- **Markov chains** - [Markov Chains](../../06-Probability-Theory/07-Markov-Chains/notes.md)
- **Estimation theory** - [Estimation Theory](../../07-Statistics/02-Estimation-Theory/notes.md)
- **Loss functions and contrastive objectives** - [Loss Functions](../01-Loss-Functions/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Executable sampling demos for inverse CDF, rejection, Monte Carlo, importance sampling, MCMC, Gumbel, and top-k/top-p previews |
| [exercises.ipynb](exercises.ipynb) | 10 graded exercises on estimators, proposal mismatch, differentiable sampling, and ML sampling diagnostics |

## Learning Objectives

After completing this section, you will be able to:

- Define a sample, estimator, proposal distribution, bias, variance, and effective sample size
- Implement inverse-transform, categorical, rejection, and stratified sampling
- Estimate expectations with Monte Carlo and quantify standard error
- Use importance weights and diagnose weight degeneracy
- Implement simple Metropolis-Hastings, Gibbs-style, and Langevin-style samplers
- Explain the reparameterization trick for Gaussian latent variables
- Implement Gumbel-Max and Gumbel-Softmax sampling
- Explain how minibatch and negative sampling change training objectives
- Treat temperature, top-k, top-p, and diffusion sampling as previews without duplicating later chapters

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 Sampling as Approximation](#11-sampling-as-approximation)
  - [1.2 Sampling as Generation](#12-sampling-as-generation)
  - [1.3 Sampling as Exploration](#13-sampling-as-exploration)
  - [1.4 Randomness Versus Determinism](#14-randomness-versus-determinism)
  - [1.5 Historical Path](#15-historical-path)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Random Sample](#21-random-sample)
  - [2.2 Estimator](#22-estimator)
  - [2.3 Bias and Variance](#23-bias-and-variance)
  - [2.4 Proposal Distribution](#24-proposal-distribution)
  - [2.5 Effective Sample Size](#25-effective-sample-size)
- [3. Direct Sampling](#3-direct-sampling)
  - [3.1 Inverse-Transform Sampling](#31-inverse-transform-sampling)
  - [3.2 Categorical Sampling](#32-categorical-sampling)
  - [3.3 Rejection Sampling](#33-rejection-sampling)
  - [3.4 Stratified Sampling](#34-stratified-sampling)
  - [3.5 Practical Limitations](#35-practical-limitations)
- [4. Monte Carlo Estimation](#4-monte-carlo-estimation)
  - [4.1 Sample Mean Estimator](#41-sample-mean-estimator)
  - [4.2 Law of Large Numbers](#42-law-of-large-numbers)
  - [4.3 Confidence Intervals](#43-confidence-intervals)
  - [4.4 Variance Reduction](#44-variance-reduction)
  - [4.5 ML Expectation Estimates](#45-ml-expectation-estimates)
- [5. Importance Sampling](#5-importance-sampling)
  - [5.1 Proposal Distribution](#51-proposal-distribution)
  - [5.2 Importance Weights](#52-importance-weights)
  - [5.3 Self-Normalized Estimator](#53-self-normalized-estimator)
  - [5.4 Weight Degeneracy](#54-weight-degeneracy)
  - [5.5 Off-Policy Preview](#55-off-policy-preview)
- [6. MCMC Methods](#6-mcmc-methods)
  - [6.1 Markov-Chain Idea](#61-markov-chain-idea)
  - [6.2 Metropolis-Hastings](#62-metropolis-hastings)
  - [6.3 Gibbs Sampling](#63-gibbs-sampling)
  - [6.4 Langevin Dynamics](#64-langevin-dynamics)
  - [6.5 HMC Preview](#65-hmc-preview)
- [7. Differentiable Sampling](#7-differentiable-sampling)
  - [7.1 Reparameterization Trick](#71-reparameterization-trick)
  - [7.2 Gaussian Latent Variables](#72-gaussian-latent-variables)
  - [7.3 Gumbel-Max](#73-gumbel-max)
  - [7.4 Gumbel-Softmax](#74-gumbel-softmax)
  - [7.5 Straight-Through Estimator](#75-straight-through-estimator)
- [8. ML-Specific Sampling](#8-ml-specific-sampling)
  - [8.1 Minibatch Sampling](#81-minibatch-sampling)
  - [8.2 Negative Sampling](#82-negative-sampling)
  - [8.3 Temperature Sampling Preview](#83-temperature-sampling-preview)
  - [8.4 Top-k and Top-p Preview](#84-top-k-and-top-p-preview)
  - [8.5 Diffusion Sampling Preview](#85-diffusion-sampling-preview)
- [9. Applications in Machine Learning](#9-applications-in-machine-learning)
  - [9.1 Bayesian Inference](#91-bayesian-inference)
  - [9.2 VAEs](#92-vaes)
  - [9.3 Contrastive Learning](#93-contrastive-learning)
  - [9.4 Generative Models](#94-generative-models)
  - [9.5 LLM Decoding Preview](#95-llm-decoding-preview)
- [10. Common Mistakes](#10-common-mistakes)
- [11. Exercises](#11-exercises)
- [12. Why This Matters for AI](#12-why-this-matters-for-ai)
- [13. Conceptual Bridge](#13-conceptual-bridge)
- [References](#references)

---

## 1. Intuition

### 1.1 Sampling as Approximation

Many ML objectives contain expectations:

$$
\mathbb{E}_{X\sim p}[f(X)].
$$

When the expectation cannot be computed exactly, we draw samples
$X_1,\ldots,X_n\sim p$ and approximate:

$$
\hat{\mu}_n=\frac{1}{n}\sum_{i=1}^{n}f(X_i).
$$

Sampling turns an integral into an average. The price is randomness.

### 1.2 Sampling as Generation

Generative systems sample outputs from learned distributions. A language model
samples tokens from a categorical distribution. A diffusion model samples
noise and progressively transforms it. A VAE samples latent variables and
decodes them. The full models belong later; the shared math is drawing random
variables from specified distributions.

### 1.3 Sampling as Exploration

In reinforcement learning and bandits, sampling actions supports exploration.
In contrastive learning, sampling negatives defines what distinctions a model
learns. In minibatch SGD, sampling data controls gradient noise. Randomness is
not merely inconvenience; it is often part of the learning algorithm.

### 1.4 Randomness Versus Determinism

Sampling methods can be stochastic or deterministic approximations to
stochastic goals. Greedy decoding is deterministic. Temperature sampling is
random. Quasi-Monte Carlo uses deterministic low-discrepancy sequences to
reduce variance. The right method depends on whether diversity, unbiasedness,
reproducibility, or coverage matters most.

### 1.5 Historical Path

Monte Carlo methods grew from numerical integration and physics. MCMC made
Bayesian inference feasible in high-dimensional models. Importance sampling
became central when direct sampling from the target distribution was hard.
Reparameterization made latent-variable models trainable with low-variance
gradients. Gumbel tricks made categorical sampling compatible with neural
network training. Modern generative AI combines all of these ideas.

## 2. Formal Definitions

### 2.1 Random Sample

A random sample from $p$ is a collection

$$
X_1,\ldots,X_n\sim p.
$$

If the samples are independent and identically distributed, we write

$$
X_i \overset{\mathrm{iid}}{\sim} p.
$$

In ML, samples may also be dependent, as in Markov chains or correlated data
augmentations.

### 2.2 Estimator

An estimator is a function of samples used to approximate an unknown quantity.
For $\mu=\mathbb{E}[X]$, the sample mean is

$$
\hat{\mu}_n=\frac{1}{n}\sum_{i=1}^{n}X_i.
$$

The estimator is random because the sample is random.

### 2.3 Bias and Variance

Bias is

$$
\operatorname{Bias}(\hat{\theta})
=\mathbb{E}[\hat{\theta}]-\theta.
$$

Variance is

$$
\operatorname{Var}(\hat{\theta})
=\mathbb{E}\left[(\hat{\theta}-\mathbb{E}[\hat{\theta}])^2\right].
$$

An unbiased estimator can still be useless if variance is huge.

### 2.4 Proposal Distribution

When sampling directly from target $p$ is hard, we may sample from a proposal
$q$ and correct with weights. The proposal must cover the target support:

$$
p(x)>0 \implies q(x)>0.
$$

If the proposal misses important target regions, estimates can be biased or
have infinite variance.

### 2.5 Effective Sample Size

For normalized importance weights $\tilde{w}_i$, a common effective sample
size is

$$
\operatorname{ESS}
=\frac{1}{\sum_{i=1}^{n}\tilde{w}_i^2}.
$$

If one weight dominates, ESS is near one even if many samples were drawn.

## 3. Direct Sampling

### 3.1 Inverse-Transform Sampling

If $U\sim\mathcal{U}(0,1)$ and $F$ is a CDF, then

$$
X=F^{-1}(U)
$$

has distribution $F$. For an exponential distribution with rate $\lambda$:

$$
X=-\frac{1}{\lambda}\log(1-U).
$$

### 3.2 Categorical Sampling

For probabilities $\mathbf{p}\in\Delta^{C-1}$, categorical sampling chooses
class $k$ with probability $p_k$. One implementation draws $u\sim\mathcal{U}(0,1)$
and returns the first index where cumulative probability exceeds $u$.

### 3.3 Rejection Sampling

Suppose we want samples from $p$ but can sample from $q$ and know a constant
$M$ such that

$$
p(x)\le Mq(x).
$$

Draw $x\sim q$ and $u\sim\mathcal{U}(0,1)$. Accept $x$ if

$$
u\le \frac{p(x)}{Mq(x)}.
$$

High rejection rates mean the proposal is inefficient.

### 3.4 Stratified Sampling

Stratified sampling divides the domain into strata and samples within each.
It can reduce variance when strata are more homogeneous than the full
population.

### 3.5 Practical Limitations

Direct methods are elegant when CDFs, bounds, or probabilities are accessible.
In high-dimensional ML, exact CDF inversion and tight rejection bounds are
rare. This motivates Monte Carlo, importance sampling, and MCMC.

## 4. Monte Carlo Estimation

### 4.1 Sample Mean Estimator

For iid samples $X_i\sim p$,

$$
\hat{\mu}_n=\frac{1}{n}\sum_{i=1}^{n}f(X_i)
$$

estimates $\mathbb{E}_p[f(X)]$.

### 4.2 Law of Large Numbers

Under standard conditions,

$$
\hat{\mu}_n\to \mathbb{E}[f(X)]
$$

as $n\to\infty$. The convergence rate is stochastic and depends on variance.

### 4.3 Confidence Intervals

If $\sigma_f^2=\operatorname{Var}(f(X))$, then the standard error is

$$
\operatorname{SE}(\hat{\mu}_n)=\frac{\sigma_f}{\sqrt{n}}.
$$

In practice, $\sigma_f$ is estimated from samples.

### 4.4 Variance Reduction

Variance reduction methods include antithetic sampling, stratification,
control variates, and better proposal distributions. The goal is not always
more samples; it is more useful samples.

### 4.5 ML Expectation Estimates

SGD estimates a full-data gradient using a minibatch. Dropout estimates an
ensemble-like objective with random masks. Contrastive learning estimates a
normalization term using sampled negatives. These are all sampling-based
approximations.

## 5. Importance Sampling

### 5.1 Proposal Distribution

To estimate

$$
\mathbb{E}_{p}[f(X)]
=\int f(x)p(x)\,dx,
$$

sample from $q$ and rewrite:

$$
\mathbb{E}_{p}[f(X)]
=\mathbb{E}_{q}\left[f(X)\frac{p(X)}{q(X)}\right].
$$

### 5.2 Importance Weights

The weight is

$$
w(x)=\frac{p(x)}{q(x)}.
$$

The estimator is

$$
\hat{\mu}=\frac{1}{n}\sum_{i=1}^{n}w(X_i)f(X_i),
\qquad X_i\sim q.
$$

### 5.3 Self-Normalized Estimator

If the target density is known only up to a constant, use

$$
\hat{\mu}_{\mathrm{SN}}
=\frac{\sum_i w_i f(X_i)}{\sum_i w_i}.
$$

This estimator is biased for finite $n$ but often practical.

### 5.4 Weight Degeneracy

If most weights are near zero and one weight is huge, the estimator has high
variance. Effective sample size detects this collapse:

$$
\operatorname{ESS}=\frac{(\sum_i w_i)^2}{\sum_i w_i^2}.
$$

### 5.5 Off-Policy Preview

In off-policy reinforcement learning, data may come from a behavior policy
while the target policy differs. Importance ratios correct for this mismatch,
but high-variance ratios are a central difficulty.

## 6. MCMC Methods

### 6.1 Markov-Chain Idea

MCMC constructs a Markov chain whose stationary distribution is the target
$p$. Samples are dependent, but after burn-in they can approximate target
expectations.

### 6.2 Metropolis-Hastings

Given current state $x$, propose $x'\sim q(x'\mid x)$ and accept with
probability

$$
\alpha=\min\left(1,
\frac{p(x')q(x\mid x')}{p(x)q(x'\mid x)}
\right).
$$

For symmetric proposals, the proposal terms cancel.

### 6.3 Gibbs Sampling

Gibbs sampling updates one variable at a time from its conditional
distribution. It is useful when conditionals are easy even though the joint
distribution is hard.

### 6.4 Langevin Dynamics

Langevin methods use gradients of log density:

$$
x_{t+1}=x_t+\frac{\eta}{2}\nabla_x\log p(x_t)+\sqrt{\eta}\epsilon_t.
$$

They combine drift toward high-density regions with random exploration.

### 6.5 HMC Preview

Hamiltonian Monte Carlo augments position variables with momentum and simulates
Hamiltonian dynamics. The full method is beyond this section, but the key idea
is gradient-informed proposals with better long-distance movement than a
random walk.

## 7. Differentiable Sampling

### 7.1 Reparameterization Trick

If

$$
Z\sim\mathcal{N}(\mu,\sigma^2),
$$

write

$$
Z=\mu+\sigma\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,1).
$$

The randomness is isolated in $\epsilon$, so gradients can flow through
$\mu$ and $\sigma$.

### 7.2 Gaussian Latent Variables

VAEs use Gaussian latent variables and the reparameterization trick to train
encoders with low-variance pathwise gradients. Full VAE details belong in
Chapter 14.

### 7.3 Gumbel-Max

If $G_i$ are iid Gumbel random variables, then

$$
\arg\max_i(\log p_i+G_i)
$$

is a categorical sample from $\mathbf{p}$.

### 7.4 Gumbel-Softmax

The Gumbel-Softmax relaxation is

$$
Y_i=
\frac{\exp((\log p_i+G_i)/\tau)}
{\sum_j\exp((\log p_j+G_j)/\tau)}.
$$

As $\tau\to 0$, samples become nearly one-hot. For positive $\tau$, the output
is differentiable with respect to logits.

### 7.5 Straight-Through Estimator

A straight-through estimator uses a hard discrete sample in the forward pass
and a soft or surrogate gradient in the backward pass. It is biased but often
useful when discrete choices must appear inside a differentiable model.

## 8. ML-Specific Sampling

### 8.1 Minibatch Sampling

SGD samples a minibatch and uses its gradient as an estimator of the full
gradient. Shuffle order, replacement, stratification, and distributed sampling
all affect gradient noise.

### 8.2 Negative Sampling

Negative sampling chooses contrastive alternatives. The negative distribution
defines what distinctions the representation learns. Easy negatives create
weak gradients. Hard negatives can be informative or mislabeled.

### 8.3 Temperature Sampling Preview

Temperature modifies categorical probabilities:

$$
p_i(\tau)=
\frac{\exp(z_i/\tau)}{\sum_j\exp(z_j/\tau)}.
$$

Low temperature sharpens. High temperature flattens. Full LLM decoding belongs
in [Language Model Probability Math](../../15-Math-for-LLMs/05-Language-Model-Probability/notes.md).

### 8.4 Top-k and Top-p Preview

Top-k restricts sampling to the $k$ largest probabilities. Top-p, or nucleus
sampling, restricts sampling to the smallest set whose cumulative probability
exceeds $p$. These are generation-time truncation rules.

### 8.5 Diffusion Sampling Preview

Diffusion sampling starts from noise and iteratively denoises. The full
diffusion treatment belongs in
[Generative Models](../../14-Math-for-Specific-Models/07-Generative-Models/notes.md).
Here it is enough to see diffusion as repeated stochastic or deterministic
sampling from learned transition rules.

## 9. Applications in Machine Learning

### 9.1 Bayesian Inference

Bayesian inference often requires expectations under a posterior distribution
that cannot be normalized exactly. MCMC and importance sampling approximate
posterior means, credible intervals, and predictive distributions.

### 9.2 VAEs

VAEs use differentiable Gaussian sampling to train latent-variable models. The
reparameterization trick is what keeps encoder gradients low variance.

### 9.3 Contrastive Learning

Contrastive learning depends on how positives and negatives are sampled. The
loss and sampler are inseparable.

### 9.4 Generative Models

Generative models use sampling to produce outputs. Sampling quality affects
diversity, fidelity, mode coverage, and reproducibility.

### 9.5 LLM Decoding Preview

LLM decoding samples tokens from model logits. Temperature, top-k, top-p,
typical sampling, and beam search are treated fully in Chapter 15. This
section only provides the general categorical sampling machinery.

## 10. Common Mistakes

| # | Mistake | Why It Is Wrong | Fix |
| --- | --- | --- | --- |
| 1 | Confusing target and proposal distributions | Estimates are for the wrong distribution | Write $p$ and $q$ explicitly |
| 2 | Ignoring support mismatch | Missing target regions cannot be recovered | Ensure $q(x)>0$ wherever $p(x)>0$ |
| 3 | Reporting sample count instead of ESS | Many samples can have one effective weight | Monitor effective sample size |
| 4 | Treating MCMC samples as iid | Markov samples are correlated | Use burn-in, thinning cautiously, and autocorrelation diagnostics |
| 5 | Using too small a proposal step | Chains mix slowly | Tune acceptance and movement |
| 6 | Using too large a proposal step | Acceptance collapses | Tune proposal scale |
| 7 | Backpropagating through hard categorical samples naively | Argmax is nondifferentiable | Use reparameterization, relaxation, or score-function estimators |
| 8 | Sampling negatives without auditing them | False negatives corrupt contrastive training | Design and inspect the sampler |
| 9 | Comparing decoding methods without fixed seed or metric | Randomness hides differences | Report seeds, entropy, and samples |
| 10 | Forgetting temperature changes entropy | It changes distribution shape | Track entropy and probability mass |

## 11. Exercises

1. (*) Implement inverse-transform sampling for an exponential distribution.
2. (*) Implement categorical sampling from cumulative probabilities.
3. (*) Estimate $\mathbb{E}[X^2]$ for $X\sim\mathcal{N}(0,1)$ by Monte Carlo.
4. (**) Compute a Monte Carlo confidence interval.
5. (**) Implement rejection sampling for a simple target.
6. (**) Compute importance weights and effective sample size.
7. (***) Implement a random-walk Metropolis sampler.
8. (***) Implement the Gaussian reparameterization trick and verify sample mean.
9. (***) Implement Gumbel-Max and compare frequencies to target probabilities.
10. (***) Implement top-k and top-p filtering for categorical logits.

## 12. Why This Matters for AI

| Concept | AI impact |
| --- | --- |
| Monte Carlo | Approximates expectations and gradients |
| Importance sampling | Corrects proposal mismatch but can have high variance |
| MCMC | Enables posterior approximation and energy-based sampling |
| Reparameterization | Makes latent-variable models trainable by pathwise gradients |
| Gumbel tricks | Relax categorical choices for differentiable training |
| Minibatch sampling | Defines SGD noise and data coverage |
| Negative sampling | Shapes representation learning |
| Temperature | Controls entropy in categorical outputs |
| Top-k/top-p | Truncates generation distributions |
| ESS | Diagnoses whether weighted samples are actually useful |

## 13. Conceptual Bridge

Losses define training signals, activations transform hidden states, and
normalization controls scale. Sampling adds randomness: approximate
expectations, stochastic gradients, latent variables, negative examples, and
generated outputs.

```text
probability distribution
    -> sampler
    -> random samples
    -> estimator or generated output
    -> training / inference decision
```

Next, Chapter 14 uses these primitives inside concrete model families such as
linear models, neural networks, probabilistic models, reinforcement learning,
and generative models.

## References

- Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., and Teller, E. (1953). Equation of State Calculations by Fast Computing Machines.
- Hastings, W. K. (1970). Monte Carlo Sampling Methods Using Markov Chains and Their Applications.
- Robert, C. P., and Casella, G. (2004). _Monte Carlo Statistical Methods_.
- Bishop, C. M. (2006). _Pattern Recognition and Machine Learning_.
- Kingma, D. P., and Welling, M. (2014). Auto-Encoding Variational Bayes.
- Rezende, D. J., Mohamed, S., and Wierstra, D. (2014). Stochastic Backpropagation and Approximate Inference in Deep Generative Models.
- Jang, E., Gu, S., and Poole, B. (2016). Categorical Reparameterization with Gumbel-Softmax.
- Maddison, C. J., Mnih, A., and Teh, Y. W. (2016). The Concrete Distribution.
- Holtzman, A., Buys, J., Du, L., Forbes, M., and Choi, Y. (2019). The Curious Case of Neural Text Degeneration.
