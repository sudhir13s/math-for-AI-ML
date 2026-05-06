[<- Back to Curriculum](../../README.md) | [Previous: Cross-Entropy <-](../04-Cross-Entropy/notes.md)

---

# Fisher Information

> _"The amount of information carried by a sample is not how surprising the sample is in isolation, but how sharply it lets us distinguish one model from its neighbors."_

## Overview

Fisher information is one of the rare ideas that lives comfortably in several
mathematical worlds at once. In statistics, it measures how much an observation
reveals about an unknown parameter. In differential geometry, it becomes the
metric tensor of a statistical manifold. In information theory, it is the local
second-order curvature of KL divergence and a bridge between estimation and
entropy. In machine learning, it reappears as the geometry behind natural
gradient descent, curvature-aware optimization, continual-learning penalties,
uncertainty approximations, and score-based generative modeling.

The intuition is simple but deep. If two nearby parameter values
$\boldsymbol{\theta}$ and $\boldsymbol{\theta} + \mathrm{d}\boldsymbol{\theta}$
produce nearly indistinguishable distributions, then the data is weakly
informative about the parameter. If tiny parameter changes induce visibly
different likelihoods, then the parameter is statistically easy to estimate.
Fisher information is the quantitative object that measures this local
distinguishability.

This section gives Fisher information its own canonical home inside the
Information Theory chapter. The Statistics chapter already covers Fisher
information as part of estimation theory, especially through maximum likelihood
estimation, asymptotic normality, and the Cramer-Rao lower bound. The
Optimization chapter already covers natural gradient and practical second-order
methods. Here we focus on what is unique about Fisher information itself:
score-based definitions, structural properties, local KL geometry,
Fisher-Rao metrics, Jeffreys priors, Fisher divergence, de Bruijn-style
identities, and the modern AI systems that use these ideas.

## Prerequisites

- **Entropy, KL divergence, and cross-entropy** - [01-Entropy](../01-Entropy/notes.md), [02-KL-Divergence](../02-KL-Divergence/notes.md), [04-Cross-Entropy](../04-Cross-Entropy/notes.md)
- **Mutual information and statistical dependence** - [03-Mutual-Information](../03-Mutual-Information/notes.md)
- **Gradients, Hessians, and multivariate calculus** - [05-Multivariate-Calculus/README.md](../../05-Multivariate-Calculus/README.md)
- **Maximum likelihood estimation and asymptotic statistics** - [07-Statistics/02-Estimation-Theory](../../07-Statistics/02-Estimation-Theory/notes.md)
- **Natural gradient and curvature-aware optimization** - [08-Optimization/03-Second-Order-Methods](../../08-Optimization/03-Second-Order-Methods/notes.md)
- **Basic probability distributions and expectations** - [06-Probability-Theory/README.md](../../06-Probability-Theory/README.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Interactive derivations, closed-form Fisher calculations, KL-curvature checks, Jeffreys priors, and natural-gradient demos |
| [exercises.ipynb](exercises.ipynb) | 10 graded exercises covering score functions, matrix Fisher information, KL curvature, Jeffreys priors, empirical Fisher pitfalls, and ML applications |

## Learning Objectives

After completing this section, you will be able to:

- Define the score function and explain why Fisher information is its variance
- Compute scalar Fisher information and matrix Fisher information for standard models
- Distinguish expected Fisher, observed information, and empirical Fisher clearly
- Prove or explain the key structural properties: additivity, positive semidefiniteness, and reparameterization behavior
- Interpret Fisher information as local curvature of KL divergence
- Explain the Fisher-Rao metric and why it gives a geometry on parametric model families
- Derive Jeffreys priors from $\sqrt{\det I(\boldsymbol{\theta})}$
- Distinguish Fisher divergence from KL divergence and cross-entropy
- Understand the role of Fisher information in natural gradient methods without confusing it with the full optimization chapter
- Explain why empirical Fisher approximations can fail in modern deep learning
- Connect Fisher information to K-FAC, EWC, continual learning, and score-based generative modeling
- Relate Fisher information to entropy smoothing identities such as de Bruijn's identity

---

## Table of Contents

- [Fisher Information](#fisher-information)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
  - [Companion Notebooks](#companion-notebooks)
  - [Learning Objectives](#learning-objectives)
  - [Table of Contents](#table-of-contents)
  - [1. Intuition](#1-intuition)
    - [1.1 Local Distinguishability of Models](#11-local-distinguishability-of-models)
    - [1.2 From Curvature to Information](#12-from-curvature-to-information)
    - [1.3 Why Fisher Information Matters for AI](#13-why-fisher-information-matters-for-ai)
    - [1.4 Historical Timeline](#14-historical-timeline)
  - [2. Formal Definitions](#2-formal-definitions)
    - [2.1 Score Function](#21-score-function)
    - [2.2 Scalar Fisher Information](#22-scalar-fisher-information)
    - [2.3 Fisher Information Matrix](#23-fisher-information-matrix)
    - [2.4 Expected Hessian Form](#24-expected-hessian-form)
    - [2.5 Observed, Expected, and Empirical Fisher](#25-observed-expected-and-empirical-fisher)
  - [3. Core Theory I: Structural Properties](#3-core-theory-i-structural-properties)
    - [3.1 Nonnegativity and PSD Geometry](#31-nonnegativity-and-psd-geometry)
    - [3.2 Additivity over Independent Observations](#32-additivity-over-independent-observations)
    - [3.3 Reparameterization Law](#33-reparameterization-law)
    - [3.4 Sufficiency and Information Preservation](#34-sufficiency-and-information-preservation)
    - [3.5 Closed-Form Examples](#35-closed-form-examples)
  - [4. Core Theory II: Geometry and Local Divergence](#4-core-theory-ii-geometry-and-local-divergence)
    - [4.1 Fisher as Local Curvature of KL Divergence](#41-fisher-as-local-curvature-of-kl-divergence)
    - [4.2 Fisher-Rao Metric](#42-fisher-rao-metric)
    - [4.3 Jeffreys Prior](#43-jeffreys-prior)
    - [4.4 Natural Gradient as KL-Constrained Steepest Descent](#44-natural-gradient-as-kl-constrained-steepest-descent)
    - [4.5 Identifiability, Flat Directions, and Singular Models](#45-identifiability-flat-directions-and-singular-models)
  - [5. Core Theory III: Information-Theoretic Identities](#5-core-theory-iii-information-theoretic-identities)
    - [5.1 Fisher Divergence](#51-fisher-divergence)
    - [5.2 de Bruijn Identity](#52-de-bruijn-identity)
    - [5.3 Stam-Type Inequalities](#53-stam-type-inequalities)
    - [5.4 Heat Flow, Noise Injection, and Scores](#54-heat-flow-noise-injection-and-scores)
    - [5.5 Why These Identities Matter for Modern AI](#55-why-these-identities-matter-for-modern-ai)
  - [6. Advanced Topics](#6-advanced-topics)
    - [6.1 Fisher vs Hessian vs Gauss-Newton](#61-fisher-vs-hessian-vs-gauss-newton)
    - [6.2 True Fisher vs Empirical Fisher](#62-true-fisher-vs-empirical-fisher)
    - [6.3 Misspecification and Robust Information](#63-misspecification-and-robust-information)
    - [6.4 Large-Scale Approximations](#64-large-scale-approximations)
    - [6.5 Singular and Symmetric Neural Models](#65-singular-and-symmetric-neural-models)
  - [7. Applications in Machine Learning](#7-applications-in-machine-learning)
    - [7.1 Logistic and Softmax Models](#71-logistic-and-softmax-models)
    - [7.2 Natural Gradient and Curvature-Aware Training](#72-natural-gradient-and-curvature-aware-training)
    - [7.3 K-FAC and Structured Fisher Approximations](#73-k-fac-and-structured-fisher-approximations)
    - [7.4 Elastic Weight Consolidation](#74-elastic-weight-consolidation)
    - [7.5 Score Matching, Denoising, and Diffusion Connections](#75-score-matching-denoising-and-diffusion-connections)
  - [8. Common Mistakes](#8-common-mistakes)
  - [9. Exercises](#9-exercises)
  - [10. Why This Matters for AI (2026 Perspective)](#10-why-this-matters-for-ai-2026-perspective)
  - [11. Conceptual Bridge](#11-conceptual-bridge)
  - [References](#references)

---

## 1. Intuition

### 1.1 Local Distinguishability of Models

Suppose we have a family of probability distributions
$\{p(x \mid \theta) : \theta \in \Theta\}$. A parameter value by itself is not
the object we observe. What we actually observe is data sampled from the
distribution associated with that parameter. This means that "how much we know
about $\theta$" can only be interpreted through the distributions the parameter
generates.

That observation immediately suggests a local perspective. If two nearby
parameter values $\theta$ and $\theta + \delta$ produce nearly identical
distributions, then even a large amount of data may struggle to distinguish
them. If tiny perturbations in $\theta$ produce visibly different likelihoods,
then a small sample can already tell us a lot about the parameter.

Fisher information formalizes this local distinguishability. It does not ask:

- how much uncertainty the parameter has in some prior sense
- how surprising a specific observation is
- how different two arbitrary distributions are globally

Instead, it asks:

> How sensitive is the model distribution to infinitesimal changes in the
> parameter?

This is why Fisher information sits naturally between statistics and
information theory. It is not a global divergence like KL divergence, but it is
also not only an optimization curvature quantity. It is a *local information
metric* on the model family itself.

```text
LOCAL DISTINGUISHABILITY
===============================================================

parameter space:
  theta -------- theta + dtheta

induced model family:
  p(x|theta) ---- p(x|theta + dtheta)

If the two model distributions are almost the same:
  little statistical information about theta

If the two model distributions separate quickly:
  much statistical information about theta

Fisher information measures:
  sensitivity of the distribution, not just sensitivity of a number
===============================================================
```

There is a useful contrast here with entropy. Entropy measures uncertainty
inside a single distribution. Fisher information measures the sharpness with
which a *family* of distributions changes as we move in parameter space.

Three quick examples build the instinct:

1. **Bernoulli coin with $p$ near $1/2$**:
   the likelihood changes moderately with the parameter, and the Fisher
   information per sample is finite and symmetric around $1/2$.
2. **Bernoulli coin with $p$ near $0$ or $1$**:
   tiny changes in $p$ dramatically affect the likelihood of rare outcomes, so
   the Fisher information becomes large.
3. **Overparameterized neural network with redundant weights**:
   many different weight vectors can induce almost the same function, so there
   are directions in parameter space with very low local distinguishability.

The third example is the bridge to modern AI. Flat directions, symmetries, and
degeneracies in deep networks are not just optimization curiosities. They are
geometric statements about information in parameter space.

### 1.2 From Curvature to Information

There are several equivalent ways to think about Fisher information. Each is
useful in a different context.

The first is the **score-variance view**. The score is the derivative of the
log-likelihood:

$$ s_\theta(X) = \frac{\partial}{\partial \theta}\log p(X \mid \theta). $$

If the score fluctuates a lot under the model, then the likelihood is very
sensitive to the parameter. That means the sample carries strong information
about $\theta$.

The second is the **curvature view**. Near the truth, the log-likelihood has a
local quadratic shape. Sharp curvature means small parameter perturbations
rapidly reduce the likelihood, which again means the data localizes the
parameter well.

The third is the **distance view**. The KL divergence between two nearby model
distributions satisfies a second-order approximation:

$$ D_{\mathrm{KL}}\!\left(p_{\boldsymbol{\theta}} \,\middle\|\, p_{\boldsymbol{\theta}+\mathrm{d}\boldsymbol{\theta}}\right) \approx \frac{1}{2}\mathrm{d}\boldsymbol{\theta}^\top I(\boldsymbol{\theta})\,\mathrm{d}\boldsymbol{\theta}. $$

So Fisher information is literally the quadratic form that measures local KL
distance.

These three views are not separate facts to memorize. They are three readings
of the same object:

| View | Object | Interpretation |
| --- | --- | --- |
| Score view | $\mathbb{E}[s_\theta(X)^2]$ or $\mathbb{E}[\mathbf{s}\mathbf{s}^\top]$ | sensitivity of log-likelihood |
| Curvature view | $-\mathbb{E}[\nabla^2 \log p(X \mid \boldsymbol{\theta})]$ | expected local sharpness |
| Divergence view | local Hessian of KL | infinitesimal information geometry |

This is also why Fisher information shows up in so many chapters of the
curriculum:

- in statistics, because curvature controls estimation precision
- in optimization, because curvature affects update geometry
- in Bayesian inference, because local curvature shapes Laplace-style
  approximations and Jeffreys priors
- in information theory, because local KL structure is part of the geometry of
  model families

```text
THREE VIEWS OF FISHER INFORMATION
===============================================================

score variance
  Var_theta[ d/dtheta log p(X|theta) ]
        |
        v
likelihood curvature
  - E_theta[ d^2/dtheta^2 log p(X|theta) ]
        |
        v
local KL geometry
  D_KL(p_theta || p_{theta + dtheta})
    ~ (1/2) dtheta^T I(theta) dtheta

same object, three interpretations
===============================================================
```

One subtle but important point: Fisher information is *local*. It tells us
what happens in a tiny neighborhood around a parameter value. It does not
summarize the full global geometry of the model family. This local-global
distinction matters in deep learning, where local curvature can be informative
even when the global landscape is highly nonconvex.

### 1.3 Why Fisher Information Matters for AI

If you work on modern AI systems, Fisher information matters for at least five
practical reasons.

First, it is the geometric object behind **natural gradient descent**. Ordinary
gradient descent treats parameter space as Euclidean. Natural gradient treats
the space of model distributions as primary and uses Fisher information as the
metric tensor. This matters because two parameter vectors can be numerically
different yet functionally similar, and Fisher geometry is designed to respect
that difference.

Second, it appears in **structured second-order methods** such as K-FAC and
related preconditioners. These methods do not use the full Fisher matrix
exactly, but they are motivated by the idea that steepest descent should be
measured in distribution space, not naive coordinate space.

Third, Fisher information appears in **continual learning**, especially Elastic
Weight Consolidation (EWC). There, diagonal approximations to Fisher
information are used as importance weights: parameters that strongly affect the
old task's likelihood are penalized more heavily when learning a new task.

Fourth, Fisher ideas connect to **uncertainty and calibration**. The inverse of
the Fisher matrix appears in asymptotic covariance approximations, which is why
observed and expected information are tied to standard errors, Laplace
approximations, and local uncertainty summaries.

Fifth, Fisher-related identities connect to **score-based modeling** and
diffusion-style generative methods. There the relevant "score" is with respect
to the data variable rather than the parameter, but the conceptual bridge is
the same: derivatives of log densities are information-bearing objects.

| AI setting | Where Fisher enters | Why it matters |
| --- | --- | --- |
| Natural gradient | metric on model family | parameterization-aware updates |
| K-FAC / structured curvature | tractable Fisher approximation | faster curvature-aware optimization |
| Continual learning | diagonal Fisher importance weights | reduce catastrophic forgetting |
| Laplace / uncertainty | inverse local curvature | local posterior or confidence shape |
| Diffusion / score models | score-field viewpoint | local density geometry and denoising |

A common misconception is that Fisher information is "old classical stats" and
therefore not central to frontier ML. The opposite is closer to the truth.
Deep learning repeatedly rediscovers that geometry matters, and Fisher
information is one of the cleanest geometry-carrying objects we have for
probabilistic models.

### 1.4 Historical Timeline

```text
FISHER INFORMATION -- KEY MILESTONES
===============================================================

1920s-1930s  Ronald Fisher
             Introduces likelihood, score, and information as central
             objects in statistical inference.

1940s        Cramer and Rao
             Fisher information becomes the denominator of a fundamental
             lower bound on estimator variance.

1945-1946    Jeffreys and Cramer
             Jeffreys prior and asymptotic efficiency deepen the link
             between geometry, invariance, and estimation.

1940s-1970s  Information theory era
             Fisher information becomes connected to inequalities,
             de Bruijn identities, and entropy power arguments.

1945-1990s   Rao and information geometry
             Fisher information is recognized as a Riemannian metric
             on statistical model manifolds.

1998         Amari
             Natural gradient reinterprets learning through Fisher-Rao
             geometry.

2015         Martens and Grosse
             K-FAC makes approximate Fisher-based curvature methods more
             practical for neural networks.

2017         Kirkpatrick et al.
             EWC uses diagonal Fisher approximations for continual learning.

2019-2026    Modern AI
             Fisher remains central in natural-gradient methods, curvature
             approximations, continual learning, uncertainty, and
             information-geometric interpretations of model families.
===============================================================
```

Historically, Fisher information began as an inference object: a way of
quantifying how much a sample says about a parameter. It later became clear
that this quantity carries geometry. The parameter space of distributions is
not just a coordinate box; it is a curved manifold whose local metric is given
by Fisher information.

That shift from "variance formula" to "geometry of distributions" is exactly
why Fisher information belongs in this chapter. Its deepest meaning is not only
statistical efficiency. Its deepest meaning is local information geometry.

## 2. Formal Definitions

### 2.1 Score Function

Let $X$ be a random variable with density or mass function
$p(x \mid \theta)$ depending on a scalar parameter $\theta$. The
**score function** is

$$ s_\theta(X) = \frac{\partial}{\partial \theta}\log p(X \mid \theta). $$

For vector parameters $\boldsymbol{\theta} \in \mathbb{R}^d$, the score is the
gradient

$$ \mathbf{s}_{\boldsymbol{\theta}}(X) = \nabla_{\boldsymbol{\theta}} \log p(X \mid \boldsymbol{\theta}). $$

The score is not itself the amount of information. It is the *local
sensitivity* of the log-likelihood. If the likelihood responds strongly to
changes in the parameter, the score will be large in magnitude. If the model is
locally insensitive, the score will be small.

Under standard regularity conditions, the score has mean zero:

$$ \mathbb{E}_{X \sim p(\cdot \mid \theta)}[s_\theta(X)] = 0. $$

The proof is simple and worth remembering because it drives many later
identities:

$$ \mathbb{E}[s_\theta(X)] = \int \frac{\partial}{\partial \theta}\log p(x \mid \theta)\, p(x \mid \theta)\,dx = \int \frac{\partial}{\partial \theta} p(x \mid \theta)\,dx = \frac{\partial}{\partial \theta}\int p(x \mid \theta)\,dx = 0. $$

This identity says that the score is centered. Positive and negative local
perturbation signals cancel in expectation under the model itself.

Three examples make the score concrete.

**Example 1: Bernoulli($p$).**
For $X \in \{0,1\}$ with
$p(x \mid p) = p^x(1-p)^{1-x}$,

$$ \log p(X \mid p) = X \log p + (1-X)\log(1-p), $$

so

$$ s_p(X) = \frac{X}{p} - \frac{1-X}{1-p}. $$

**Example 2: Poisson($\lambda$).**
For $X \sim \operatorname{Poi}(\lambda)$,

$$ \log p(X \mid \lambda) = -\lambda + X\log \lambda - \log(X!), $$

so

$$ s_\lambda(X) = -1 + \frac{X}{\lambda}. $$

**Example 3: Gaussian mean with known variance.**
If $X \sim \mathcal{N}(\mu, \sigma^2)$ with fixed $\sigma^2$,

$$ \log p(X \mid \mu) = -\frac{1}{2}\log(2\pi \sigma^2) - \frac{(X-\mu)^2}{2\sigma^2}, $$

so

$$ s_\mu(X) = \frac{X-\mu}{\sigma^2}. $$

These examples already show the pattern: the score is a standardized residual.

Non-examples help too:

- the raw likelihood $p(X \mid \theta)$ is not the score
- the negative log-likelihood itself is not the score
- the gradient with respect to the *data* variable is not the parameter score

This third distinction becomes crucial later, because score matching and
diffusion models use derivatives with respect to $\mathbf{x}$, not with respect
to $\boldsymbol{\theta}$.

### 2.2 Scalar Fisher Information

For a scalar parameter $\theta$, the **Fisher information** is defined by

$$ I(\theta) = \mathbb{E}_{X \sim p(\cdot \mid \theta)}[s_\theta(X)^2]. $$

Because the score has mean zero, this is the variance of the score:

$$ I(\theta) = \operatorname{Var}_{\theta}(s_\theta(X)). $$

That single formula already explains a lot.

- If the score barely fluctuates, the log-likelihood is locally insensitive and
  the sample provides little information about the parameter.
- If the score fluctuates strongly, the likelihood sharply reacts to parameter
  changes and the sample is informative.

Since it is a variance, Fisher information is always nonnegative:

$$ I(\theta) \ge 0. $$

The units also make sense. If $\theta$ is measured in some parameter unit, then
the score has units of inverse parameter, and the Fisher information has units
of inverse parameter squared. This is why its inverse naturally behaves like a
variance scale.

For the earlier examples:

**Bernoulli($p$).**

$$ I(p) = \mathbb{E}\!\left[\left(\frac{X}{p} - \frac{1-X}{1-p}\right)^2\right] = \frac{1}{p(1-p)}. $$

**Poisson($\lambda$).**

$$ I(\lambda) = \mathbb{E}\!\left[\left(-1 + \frac{X}{\lambda}\right)^2\right] = \frac{1}{\lambda}. $$

**Gaussian mean with known variance.**

$$ I(\mu) = \mathbb{E}\!\left[\left(\frac{X-\mu}{\sigma^2}\right)^2\right] = \frac{1}{\sigma^2}. $$

These formulas are worth internalizing because they reveal qualitative
behavior:

- Bernoulli information explodes near $p=0$ and $p=1$
- Poisson information decreases as the mean grows
- Gaussian mean information is constant when variance is fixed

```text
SCALAR FISHER INFORMATION EXAMPLES
===============================================================

Bernoulli(p):        I(p)       = 1 / (p(1-p))
Poisson(lambda):     I(lambda)  = 1 / lambda
Gaussian mean mu:    I(mu)      = 1 / sigma^2

Interpretation:
  smaller noise  -> larger information
  larger variance -> smaller information
  more fragile likelihood -> sharper distinguishability
===============================================================
```

There is an important forward reference here:

> **Preview: Cramer-Rao lower bound**
>
> In [Estimation Theory](../../07-Statistics/02-Estimation-Theory/notes.md),
> Fisher information appears in the bound
> $\operatorname{Var}(\hat{\theta}) \ge 1/(nI(\theta))$ for unbiased
> estimators. That section is the canonical home for the full theorem and its
> proof. Here we keep the focus on Fisher information as an information object
> rather than fully redeveloping classical efficiency theory.

### 2.3 Fisher Information Matrix

For vector parameters $\boldsymbol{\theta} \in \mathbb{R}^d$, the scalar
definition generalizes to the **Fisher information matrix**

$$ I(\boldsymbol{\theta}) = \mathbb{E}_{X \sim p(\cdot \mid \boldsymbol{\theta})}\!\left[\mathbf{s}_{\boldsymbol{\theta}}(X)\mathbf{s}_{\boldsymbol{\theta}}(X)^\top\right]. $$

Because this is an outer product averaged over the model distribution, the
matrix is symmetric and positive semidefinite:

$$ I(\boldsymbol{\theta}) \succeq 0. $$

For any vector $\mathbf{v} \in \mathbb{R}^d$,

$$ \mathbf{v}^\top I(\boldsymbol{\theta}) \mathbf{v} = \mathbb{E}\!\left[(\mathbf{v}^\top \mathbf{s}_{\boldsymbol{\theta}}(X))^2\right] \ge 0. $$

This quadratic-form identity is one of the cleanest ways to understand the
matrix. The matrix does not merely record coordinate-wise sensitivity. It tells
us how informative the model is in *every direction* of parameter space.

Interpret the matrix entries carefully:

$$ I_{ij}(\boldsymbol{\theta}) = \mathbb{E}\!\left[\frac{\partial}{\partial \theta_i}\log p(X \mid \boldsymbol{\theta}) \frac{\partial}{\partial \theta_j}\log p(X \mid \boldsymbol{\theta})\right]. $$

- Diagonal entries measure sensitivity with respect to single coordinates.
- Off-diagonal entries measure coupling between coordinates.
- Small eigenvalues indicate flat or weakly identifiable directions.
- Large eigenvalues indicate sharply distinguishable directions.

Example: Gaussian mean with known covariance.
Let $X \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$ with fixed positive definite
$\Sigma$. Then

$$ \mathbf{s}_{\boldsymbol{\mu}}(X) = \Sigma^{-1}(X-\boldsymbol{\mu}), $$

so

$$ I(\boldsymbol{\mu}) = \Sigma^{-1}. $$

This example is foundational. In a Gaussian location model, covariance and
information are literal inverses. High observation noise means low information;
low observation noise means high information.

Another example: scalar Gaussian variance parameter with known mean. If
$X \sim \mathcal{N}(\mu, \sigma^2)$ and we parameterize by $\sigma^2$, then the
Fisher information depends strongly on the parameterization. This is our first
warning that Fisher information values are not invariant as plain numbers across
different coordinates.

### 2.4 Expected Hessian Form

Under regularity conditions that allow interchange of differentiation and
integration, Fisher information admits another representation:

$$ I(\boldsymbol{\theta}) = -\mathbb{E}_{X \sim p(\cdot \mid \boldsymbol{\theta})}\!\left[\nabla_{\boldsymbol{\theta}}^2 \log p(X \mid \boldsymbol{\theta})\right]. $$

In the scalar case,

$$ I(\theta) = -\mathbb{E}\!\left[\frac{\partial^2}{\partial \theta^2}\log p(X \mid \theta)\right]. $$

This is often called the expected-curvature form of Fisher information. It
matches the intuition that Fisher information measures how sharply the
log-likelihood bends on average.

Why the minus sign? Because log-likelihoods are often locally concave near the
truth, so their second derivatives are negative. Taking the negative expected
Hessian yields a nonnegative information quantity.

It is tempting to conclude that Fisher information is just "the Hessian." That
is too quick. The exact relationship depends on context:

- Fisher information is the **expected** negative Hessian under the model
- observed information is the **sample-specific** negative Hessian
- empirical Fisher is a different object again, based on gradients from
  observed labels or samples

These distinctions matter a lot in machine learning practice.

Regularity conditions are not decorative technicalities here. The identity can
fail when:

- the support of the distribution depends on the parameter
- derivatives cannot be exchanged with integrals safely
- the model is singular or not differentiable in the required sense

For most textbook parametric families used in ML and statistics, the identity
works cleanly. But it is still conceptually healthier to treat the score-outer-
product definition as primary and the expected-Hessian formula as a theorem.

### 2.5 Observed, Expected, and Empirical Fisher

One of the biggest sources of confusion in modern ML is that several related
curvature-like objects are all called "Fisher" in casual conversation.

We need to separate three of them carefully.

**Expected Fisher**

$$ I(\boldsymbol{\theta}) = \mathbb{E}_{X \sim p(\cdot \mid \boldsymbol{\theta})}\!\left[\mathbf{s}_{\boldsymbol{\theta}}(X)\mathbf{s}_{\boldsymbol{\theta}}(X)^\top\right]. $$

This is the population object. It depends on the model family and the parameter
value.

**Observed information**

$$ J(\boldsymbol{\theta}; \mathcal{D}) = -\nabla_{\boldsymbol{\theta}}^2 \log p(\mathcal{D} \mid \boldsymbol{\theta}). $$

This is the sample-specific negative Hessian of the log-likelihood.

**Empirical Fisher**

$$ \widehat{I}_{\mathrm{emp}}(\boldsymbol{\theta}) = \frac{1}{n}\sum_{i=1}^n \mathbf{g}_i \mathbf{g}_i^\top, \qquad \mathbf{g}_i = \nabla_{\boldsymbol{\theta}} \log p(y^{(i)} \mid \mathbf{x}^{(i)}; \boldsymbol{\theta}). $$

This is a gradient outer-product computed on observed data. In practice it is
often easier to estimate than the true Fisher, which is why it shows up in
diagonal approximations and heuristics.

These objects can be close in some settings, but they are not generally the
same.

| Object | Depends on | Interpretation | Same as Fisher? |
| --- | --- | --- | --- |
| Expected Fisher | model distribution | population local information geometry | yes |
| Observed information | specific dataset | sample-specific local curvature | not exactly |
| Empirical Fisher | data gradients | computational approximation | not generally |

This section will keep these distinctions explicit throughout.

> **Preview: Optimization chapter**
>
> The full optimizer-level discussion of Hessian, Gauss-Newton, natural
> gradient, K-FAC, and empirical Fisher approximations belongs in
> [08-Optimization/03-Second-Order-Methods](../../08-Optimization/03-Second-Order-Methods/notes.md).
> Here we focus on the mathematical identity of these objects, not only on
> their engineering tradeoffs.

## 3. Core Theory I: Structural Properties

### 3.1 Nonnegativity and PSD Geometry

For scalar parameters, Fisher information is the variance of the score:

$$ I(\theta) = \operatorname{Var}_{\theta}(s_\theta(X)). $$

That immediately implies

$$ I(\theta) \ge 0. $$

For vector parameters, the matrix form

$$ I(\boldsymbol{\theta}) = \mathbb{E}[\mathbf{s}\mathbf{s}^\top] $$

implies that $I(\boldsymbol{\theta})$ is positive semidefinite. The proof is a
single line:

$$ \mathbf{v}^\top I(\boldsymbol{\theta}) \mathbf{v} = \mathbb{E}\!\left[(\mathbf{v}^\top \mathbf{s}_{\boldsymbol{\theta}}(X))^2\right] \ge 0 \qquad \text{for all } \mathbf{v} \in \mathbb{R}^d. $$

This line is more than a proof. It is the geometric interpretation. Every
vector $\mathbf{v}$ specifies a direction in parameter space. The quantity

$$ \mathbf{v}^\top I(\boldsymbol{\theta}) \mathbf{v} $$

is the one-dimensional Fisher information in that direction. So the full matrix
packages directional information over the entire tangent space.

When is the Fisher information strictly positive definite rather than only
semidefinite?

- when nearby parameter perturbations in every direction change the
  distribution distinguishably
- when the parameterization is locally identifiable
- when there are no redundant or symmetry directions

It becomes singular when some directions fail to change the distribution to
first order.

Three recurring reasons for singularity are:

1. **Redundant parameterization**:
   different parameter vectors represent the same distribution
2. **Boundary phenomena**:
   the model loses smooth identifiability near parameter boundaries
3. **Symmetry**:
   permutations or rescalings leave the function unchanged

Neural networks exhibit all three. Hidden-unit permutations, scale symmetries
under normalization, and saturated activations all create low-information
directions.

```text
PSD GEOMETRY OF FISHER INFORMATION
===============================================================

direction in parameter space: v

directional information:
  v^T I(theta) v

if large:
  changing theta along v quickly changes the model

if near zero:
  changing theta along v barely changes the model

eigenvalues of I(theta):
  large  -> sharp / informative directions
  small  -> flat / weakly identifiable directions
  zero   -> exact local redundancy
===============================================================
```

This is the first place where Fisher differs from plain Euclidean geometry.
Euclidean length treats all coordinate directions equally. Fisher geometry
weights directions by their effect on the model distribution.

### 3.2 Additivity over Independent Observations

One of the most important structural facts about Fisher information is
additivity under independence.

Suppose $X_1,\dots,X_n$ are iid from $p(x \mid \theta)$. Then the log-likelihood
adds:

$$ \log p(x_{1:n} \mid \theta) = \sum_{i=1}^n \log p(x_i \mid \theta). $$

Therefore the score adds too:

$$ s_\theta(X_{1:n}) = \sum_{i=1}^n s_\theta(X_i). $$

Taking the variance under the model gives

$$ I_n(\theta) = \operatorname{Var}\!\left(\sum_{i=1}^n s_\theta(X_i)\right). $$

Because the scores are iid and mean zero,

$$ I_n(\theta) = \sum_{i=1}^n \operatorname{Var}(s_\theta(X_i)) = n I(\theta). $$

For vectors, exactly the same logic yields

$$ I_n(\boldsymbol{\theta}) = n I(\boldsymbol{\theta}). $$

This additivity is one reason Fisher information is so natural in asymptotic
statistics. Information grows linearly with sample size, so standard errors
often shrink like $1/\sqrt{n}$.

There is a deeper conceptual reading too:

- entropy does not generally add under dependence
- KL divergence adds over independent product models
- Fisher information also adds over independent observations because it is the
  local second-order form associated with that product structure

This makes Fisher information an "accumulating resource." Each independent
observation contributes a local information increment.

Example: Gaussian mean with known variance.
If $X_i \sim \mathcal{N}(\mu,\sigma^2)$ iid, then each observation contributes
$1/\sigma^2$, so

$$ I_n(\mu) = \frac{n}{\sigma^2}. $$

Example: Bernoulli($p$).
Each sample contributes $1/[p(1-p)]$, so

$$ I_n(p) = \frac{n}{p(1-p)}. $$

This is exactly why confidence intervals narrow with more samples and why
temperature-calibration or logistic-regression standard errors become more
stable on large datasets.

Two caveats matter:

1. If observations are dependent, simple linear additivity fails.
2. If the model is misspecified, the relevant variance and curvature objects can
   separate, leading to sandwich-type corrections.

Those caveats do not destroy the additivity idea. They just clarify that it
belongs to the clean iid parametric setting.

### 3.3 Reparameterization Law

Fisher information has a famous and subtle invariance structure.

Suppose we reparameterize a scalar model using a smooth bijection
$\phi = g(\theta)$. Then by the chain rule,

$$ \frac{\partial}{\partial \phi}\log p(X \mid \phi) = \frac{\partial \theta}{\partial \phi}\frac{\partial}{\partial \theta}\log p(X \mid \theta). $$

Squaring and taking expectation gives

$$ I_\phi(\phi) = I_\theta(\theta)\left(\frac{d\theta}{d\phi}\right)^2. $$

So scalar Fisher information is not invariant as a plain number. It transforms
like a metric coefficient.

For vector parameters with $\boldsymbol{\phi} = g(\boldsymbol{\theta})$ and
Jacobian $J = \partial \boldsymbol{\theta} / \partial \boldsymbol{\phi}$, the
matrix transforms as

$$ I_{\boldsymbol{\phi}}(\boldsymbol{\phi}) = J^\top I_{\boldsymbol{\theta}}(\boldsymbol{\theta}) J. $$

This is exactly how a Riemannian metric tensor transforms.

What remains invariant is not the matrix itself, but the quadratic differential
form

$$ \mathrm{d}\boldsymbol{\theta}^\top I_{\boldsymbol{\theta}}(\boldsymbol{\theta}) \,\mathrm{d}\boldsymbol{\theta}. $$

That is why Fisher information defines geometry. Coordinate descriptions change,
but the intrinsic local distance does not.

Simple scalar example:
Let $\theta = p$ for a Bernoulli model. Then

$$ I(p) = \frac{1}{p(1-p)}. $$

Now reparameterize with the logit

$$ \phi = \log \frac{p}{1-p}. $$

Since

$$ \frac{dp}{d\phi} = p(1-p), $$

the Fisher information in logit coordinates is

$$ I(\phi) = I(p)\left(\frac{dp}{d\phi}\right)^2 = p(1-p). $$

These look very different numerically, but they describe the same local
statistical geometry once expressed in their own coordinates.

This is one of the strongest reasons Fisher information matters in machine
learning. Raw gradients depend heavily on parameterization. Fisher geometry
tells us how to measure change in a coordinate-aware but intrinsically
meaningful way.

```text
REPARAMETERIZATION LAW
===============================================================

same model family
  p(x|theta) = p(x|phi(theta))

coordinate matrix/value changes
  I_theta  != I_phi

but local statistical distance stays the same
  dtheta^T I_theta dtheta
    =
  dphi^T I_phi dphi

Fisher is not "number invariant"
Fisher is "geometry invariant"
===============================================================
```

### 3.4 Sufficiency and Information Preservation

A statistic $T(X)$ is sufficient for $\theta$ if it compresses the sample
without losing information about the parameter. In classical statistics this is
usually stated via factorization theorems or conditional independence, but it
has a Fisher-information reading as well.

If $T$ is sufficient under regularity conditions, then

$$ I_T(\theta) = I_X(\theta). $$

That is, the statistic preserves Fisher information completely.

This is an information-theoretic compression statement. The raw observation $X$
may contain many bits of irrelevant randomness, but as long as a statistic $T$
retains the entire local sensitivity structure relevant to $\theta$, Fisher
information is unchanged.

Example: Gaussian mean with known variance.
If $X_1,\dots,X_n \sim \mathcal{N}(\mu,\sigma^2)$ iid with known $\sigma^2$,
then the sample mean $\bar{X}$ is sufficient for $\mu$. The full sample Fisher
information is $n/\sigma^2$. Since

$$ \bar{X} \sim \mathcal{N}\!\left(\mu, \frac{\sigma^2}{n}\right), $$

the Fisher information in $\bar{X}$ is also

$$ I_{\bar{X}}(\mu) = \frac{n}{\sigma^2}. $$

No local information is lost by replacing the full sample with the sample mean.

This is a beautiful bridge between statistics and information theory:

- sufficient statistics are data compressions
- but unlike generic compressions, they preserve all parameter-relevant local
  information

In deep learning language, a sufficient statistic is a representation that
retains everything needed about the target parameter while discarding nuisance
variation. The analogy is imperfect but instructive.

We should also state a limitation clearly. Equality of Fisher information is a
local statement about parameter sensitivity. It does not automatically imply
that all global inferential tasks are identical outside the regular parametric
setting. Still, in the classical setting it is the right local preservation
criterion.

### 3.5 Closed-Form Examples

Closed-form examples are where Fisher information stops being abstract.

#### Example 1: Bernoulli($p$)

For $X \in \{0,1\}$,

$$ p(x \mid p) = p^x(1-p)^{1-x}. $$

We already computed the score:

$$ s_p(X) = \frac{X}{p} - \frac{1-X}{1-p}. $$

Then

$$ I(p) = \mathbb{E}[s_p(X)^2] = \frac{1}{p(1-p)}. $$

Interpretation:

- information is lowest at $p=1/2$
- information grows near the boundaries
- rare outcomes are highly informative about whether the parameter is near an
  edge

#### Example 2: Poisson($\lambda$)

For $X \sim \operatorname{Poi}(\lambda)$,

$$ s_\lambda(X) = -1 + \frac{X}{\lambda}, \qquad I(\lambda) = \frac{1}{\lambda}. $$

Interpretation:

- larger count scales produce smaller per-sample information about the rate
  itself
- to estimate a large rate accurately, we usually need more samples

#### Example 3: Gaussian mean, known variance

For $X \sim \mathcal{N}(\mu,\sigma^2)$ with known $\sigma^2$,

$$ s_\mu(X) = \frac{X-\mu}{\sigma^2}, \qquad I(\mu) = \frac{1}{\sigma^2}. $$

Interpretation:

- noisier data means less information
- every observation contributes the same amount of information regardless of
  the mean itself

#### Example 4: Gaussian variance, known mean

Parameterize by $\sigma^2 = v$. Then

$$ \log p(X \mid v) = -\frac{1}{2}\log(2\pi v) - \frac{(X-\mu)^2}{2v}, $$

so

$$ s_v(X) = -\frac{1}{2v} + \frac{(X-\mu)^2}{2v^2}. $$

A straightforward calculation yields

$$ I(v) = \frac{1}{2v^2}. $$

This example is a good reminder that parameterization matters: using
$\sigma$ instead of $\sigma^2$ produces a different numerical Fisher quantity,
though the induced geometry is consistent after transformation.

#### Example 5: Multivariate Gaussian mean, known covariance

For $X \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$,

$$ \mathbf{s}_{\boldsymbol{\mu}}(X) = \Sigma^{-1}(X-\boldsymbol{\mu}), \qquad I(\boldsymbol{\mu}) = \Sigma^{-1}. $$

Interpretation:

- correlated noise creates coupled information directions
- principal directions of $\Sigma$ correspond to inverse principal directions
  of information

These examples are enough to establish the recurring theme:

- noise variance suppresses information
- repeated independent samples add information
- nonlinear parameterizations rescale information
- matrix Fisher captures anisotropy and coupling

## 4. Core Theory II: Geometry and Local Divergence

### 4.1 Fisher as Local Curvature of KL Divergence

Among all interpretations of Fisher information, the most important for this
chapter is the local KL-curvature view.

Consider a smooth parametric family $p_{\boldsymbol{\theta}}$. Compare the
distribution at $\boldsymbol{\theta}$ to the nearby distribution at
$\boldsymbol{\theta} + \mathrm{d}\boldsymbol{\theta}$. Then the KL divergence
satisfies the second-order expansion

$$ D_{\mathrm{KL}}\!\left(p_{\boldsymbol{\theta}} \,\middle\|\, p_{\boldsymbol{\theta}+\mathrm{d}\boldsymbol{\theta}}\right) = \frac{1}{2}\mathrm{d}\boldsymbol{\theta}^\top I(\boldsymbol{\theta})\,\mathrm{d}\boldsymbol{\theta} + O(\lVert \mathrm{d}\boldsymbol{\theta} \rVert^3). $$

This formula is a cornerstone of information geometry.

Why is there no first-order term? Because KL divergence is minimized at zero
when the two distributions coincide, so the linear term vanishes. The second
derivative is the first nontrivial local structure, and that derivative is the
Fisher information matrix.

This identity tells us something profound:

- KL divergence is a global asymmetric divergence
- Fisher information is its local symmetric quadratic approximation

So Fisher information is the infinitesimal geometry hidden inside KL
divergence.

Sketch of the derivation:

1. Expand $\log p_{\boldsymbol{\theta}+\mathrm{d}\boldsymbol{\theta}}(x)$ to
   second order around $\boldsymbol{\theta}$.
2. Insert the expansion into the KL formula
   $\mathbb{E}_{p_{\boldsymbol{\theta}}}[\log p_{\boldsymbol{\theta}} -
   \log p_{\boldsymbol{\theta}+\mathrm{d}\boldsymbol{\theta}}]$.
3. The first-order term disappears because the expected score is zero.
4. The remaining quadratic term becomes the Fisher matrix.

This local-KL identity is one of the cleanest examples of why information
theory and statistics belong together. Global divergence and local estimation
geometry are not separate topics. One is the infinitesimal limit of the other.

Three consequences are worth stating immediately:

1. Fisher information is symmetric even though KL divergence is not.
2. Fisher information captures only local behavior; two model families with the
   same local Fisher metric can still differ globally.
3. Small KL-trust regions naturally induce Fisher-metric balls.

That third point is what powers natural-gradient derivations.

### 4.2 Fisher-Rao Metric

Because the quadratic KL expansion defines an intrinsic positive semidefinite
form on the tangent space, Fisher information gives a Riemannian metric on a
smooth identifiable statistical model. This metric is called the
**Fisher-Rao metric**.

Infinitesimally, the line element is

$$ ds^2 = \mathrm{d}\boldsymbol{\theta}^\top I(\boldsymbol{\theta})\,\mathrm{d}\boldsymbol{\theta}. $$

This formula should be read the same way one reads a metric tensor in ordinary
differential geometry:

- coordinates are parameters
- tangent directions are parameter perturbations
- local lengths are measured by Fisher information, not Euclidean dot products

Why should this metric be privileged rather than any arbitrary positive
definite matrix field? The classical answer comes from invariance. Fisher-Rao is
the canonical metric compatible with probabilistic reparameterization structure.
It transforms correctly under smooth coordinate changes and is deeply tied to
KL divergence.

Simple Gaussian example:
For the one-dimensional Gaussian location family
$\mathcal{N}(\mu,\sigma^2)$ with known $\sigma^2$, the metric is

$$ ds^2 = \frac{1}{\sigma^2} d\mu^2. $$

So the manifold is flat in $\mu$, but distances are scaled by the inverse noise
variance. This makes intuitive sense: if noise is high, two means need to be
further apart numerically to be statistically distinguishable.

For richer families, the geometry becomes curved, and this curvature encodes
how parameter changes distort distributions.

```text
FISHER-RAO GEOMETRY
===============================================================

parameter coordinates:
  theta_1, theta_2, ..., theta_d

ordinary Euclidean view:
  distance depends on coordinate differences only

Fisher-Rao view:
  distance depends on how those differences change the distributions

local metric:
  ds^2 = dtheta^T I(theta) dtheta

so the model family is not just a parameter grid
it is a curved statistical manifold
===============================================================
```

This perspective matters for AI because neural-network parameter spaces are full
of anisotropy, redundancy, and badly scaled coordinates. A geometry based on
distributional effect is often much more meaningful than one based on raw
coordinate magnitudes.

### 4.3 Jeffreys Prior

Jeffreys proposed using Fisher information to define a parameterization-
invariant prior. For a scalar parameter,

$$ \pi_J(\theta) \propto \sqrt{I(\theta)}. $$

For vector parameters,

$$ \pi_J(\boldsymbol{\theta}) \propto \sqrt{\det I(\boldsymbol{\theta})}. $$

Why the square root of the determinant? Because the metric tensor defines a
volume form on the statistical manifold. Jeffreys prior is the density induced
by that intrinsic volume.

The invariance story is the key:

- ordinary "uniform priors" are not invariant under reparameterization
- Jeffreys prior transforms correctly with the metric volume element

This is one reason Jeffreys prior is often introduced as a default objective or
reference prior in Bayesian analysis.

Example: Bernoulli($p$).
We know

$$ I(p) = \frac{1}{p(1-p)}. $$

So

$$ \pi_J(p) \propto \frac{1}{\sqrt{p(1-p)}}. $$

This is the Beta$(1/2,1/2)$ prior, also known as the arcsine prior.

Example: Exponential rate parameter $\lambda$.
If $I(\lambda)=1/\lambda^2$, then

$$ \pi_J(\lambda) \propto \frac{1}{\lambda}. $$

This may be improper on $(0,\infty)$, which is an important practical caveat:
Jeffreys priors are geometrically natural, but they are not always proper
probability distributions without additional conditions or truncations.

Why Jeffreys prior belongs in this section:

- it depends directly on Fisher information
- it is about invariant local geometry
- it links estimation, Bayesian default priors, and information-theoretic
  redundancy arguments

Why it does **not** belong here in full generality:

- the whole prior/posterior machinery belongs in
  [07-Statistics/04-Bayesian-Inference](../../07-Statistics/04-Bayesian-Inference/notes.md)

So this section presents Jeffreys prior as a Fisher-geometric object and not as
a full Bayesian workflow.

### 4.4 Natural Gradient as KL-Constrained Steepest Descent

Natural gradient is often introduced in optimization classes, but its cleanest
derivation is actually information-theoretic.

Suppose we want to improve an objective $L(\boldsymbol{\theta})$, but we want
to measure the step size not in Euclidean norm
$\lVert \Delta \boldsymbol{\theta} \rVert_2$, but in the local KL change it
induces on the model distribution.

Using the quadratic KL approximation, we solve:

$$ \max_{\Delta \boldsymbol{\theta}} \nabla L(\boldsymbol{\theta})^\top \Delta \boldsymbol{\theta} \quad \text{s.t.} \quad \frac{1}{2}\Delta \boldsymbol{\theta}^\top I(\boldsymbol{\theta}) \Delta \boldsymbol{\theta} \le \varepsilon. $$

This is a constrained linear optimization problem in a quadratic metric. Using
Lagrange multipliers yields the solution direction

$$ \Delta \boldsymbol{\theta} \propto I(\boldsymbol{\theta})^{-1} \nabla L(\boldsymbol{\theta}). $$

That is the natural gradient.

So natural gradient is the steepest-ascent direction when distance is measured
by local KL divergence rather than coordinate norm.

This derivation matters because it shows what natural gradient is *for*. It is
not just "gradient preconditioning with some matrix." It is the geometry-aware
update induced by the local information metric.

Still, we should keep the chapter boundaries clear:

> **Forward reference: Optimization**
>
> The full algorithmic story of natural gradient, K-FAC, damping, inversion
> approximations, and large-scale training tradeoffs belongs in
> [08-Optimization/03-Second-Order-Methods](../../08-Optimization/03-Second-Order-Methods/notes.md).
> Here the goal is to understand *why* Fisher information defines the relevant
> geometry.

Two high-level consequences matter for AI:

1. reparameterization invariance in distribution space
2. step sizes that adapt to local model sensitivity

This is particularly attractive in large probabilistic models where coordinate
scales can be wildly heterogeneous.

### 4.5 Identifiability, Flat Directions, and Singular Models

Not every model family has a nicely invertible Fisher matrix.

If two distinct parameter values produce the same distribution, the model is
not identifiable. Then some directions in parameter space do not change the
distribution locally, and the Fisher matrix becomes singular.

This is not a pathological corner case. It is common in modern deep learning.

Examples:

- **Permutation symmetries**:
  permuting hidden units can leave the function unchanged
- **Scaling symmetries**:
  in some architectures, scaling one layer up and another down preserves the
  same input-output map
- **Inactive units or saturated gates**:
  local parameter directions cease to affect the output distribution

When Fisher has near-zero eigenvalues, several things happen:

- natural gradient updates become unstable without damping or pseudo-inverses
- local uncertainty approximations become broad or ill-conditioned
- empirical diagonals can mislead us about which directions matter

This is why "invert the Fisher matrix" is conceptually elegant but often
computationally delicate in deep networks.

There is a broader lesson here too. In singular models, local quadratic
approximations can fail to capture the true inferential geometry. The Fisher
matrix remains informative, but it is no longer a complete summary.

```text
WHEN FISHER LOSES RANK
===============================================================

parameter direction changes coordinates
but does not change the distribution

then:
  score in that direction is ~ 0
  local KL curvature is ~ 0
  Fisher eigenvalue is ~ 0

interpretation:
  the model cannot locally distinguish movement in that direction

deep-learning sources of rank loss:
  hidden-unit permutation
  scale symmetry
  dead/saturated units
  redundant parameterization
===============================================================
```

This section is a good place to be honest about a common misconception:

> Fisher information is not magic curvature that resolves every deep-learning
> geometry problem. It is the correct local metric for probabilistic model
> families, but local correctness does not remove singularity, redundancy, or
> large-scale computational barriers.

## 5. Core Theory III: Information-Theoretic Identities

### 5.1 Fisher Divergence

KL divergence compares probability distributions directly through log-density
ratios. Fisher divergence compares them through the difference of their score
functions.

For smooth densities $p$ and $q$ on $\mathbb{R}^d$, the **Fisher divergence**
is

$$ D_F(p \| q) = \mathbb{E}_{X \sim p}\!\left[\left\lVert \nabla_{\mathbf{x}} \log p(X) - \nabla_{\mathbf{x}} \log q(X) \right\rVert_2^2\right]. $$

Be careful about the object here: this uses derivatives with respect to the
**data variable** $\mathbf{x}$, not the parameter
$\boldsymbol{\theta}$. So despite the name, Fisher divergence is not a
parameter-space curvature matrix. It is a discrepancy between *score fields*.

This distinction matters enough to say twice:

- Fisher information uses $\nabla_{\boldsymbol{\theta}} \log p(x \mid \boldsymbol{\theta})$
- Fisher divergence uses $\nabla_{\mathbf{x}} \log p(\mathbf{x})$

Why keep Fisher divergence in this section at all?

Because it belongs to the same conceptual family:

- both are built from log-density derivatives
- both encode local distinguishability
- both become central when we study score-based objects

There is also a practical reason. Fisher divergence is the population quantity
underlying **score matching**, which provides a route to training unnormalized
density models without needing the partition function explicitly.

High-level contrast:

| Quantity | Variable differentiated | What it compares |
| --- | --- | --- |
| Fisher information | parameter $\boldsymbol{\theta}$ | local sensitivity of a model family |
| Fisher divergence | data $\mathbf{x}$ | mismatch of score fields between densities |
| KL divergence | none directly | global expected log-ratio mismatch |

Fisher divergence is local in a different sense from KL. It does not directly
care about global normalizing constants. This is why score-matching objectives
can be attractive when exact likelihoods are intractable.

### 5.2 de Bruijn Identity

One of the most beautiful links between entropy and Fisher information is the
de Bruijn identity.

Let $X$ be a random vector with smooth density, and let $Z \sim
\mathcal{N}(\mathbf{0}, I)$ be independent Gaussian noise. Define a Gaussian-
smoothed random variable

$$ Y_t = X + \sqrt{t}\,Z. $$

Then the differential entropy of $Y_t$ evolves according to

$$ \frac{d}{dt} h(Y_t) = \frac{1}{2} J(Y_t), $$

where $J(\cdot)$ denotes the Fisher information with respect to location.

This identity is extraordinary because it directly links:

- entropy, a global uncertainty quantity
- Gaussian smoothing, a heat-flow or noise-injection process
- Fisher information, a local derivative-based sharpness quantity

Intuitively:

- adding Gaussian noise smooths the distribution
- smoothing increases entropy
- the instantaneous rate of entropy increase is governed by Fisher information

So Fisher information measures how rapidly uncertainty expands under
infinitesimal Gaussian perturbation.

This identity is a key bridge between the "estimation world" and the
"information world." Entropy looks global and combinatorial. Fisher looks local
and differential. de Bruijn's identity says they are dynamically connected.

We do not need the full proof here, but the structure matters:

1. Gaussian convolution corresponds to heat-flow evolution.
2. Differentiating entropy along the heat flow produces an integral involving
   gradients of the density.
3. After algebraic rearrangement, that integral is exactly Fisher information.

This is not just a classical theorem with little modern relevance. It explains
why score fields become the natural infinitesimal objects when studying noisy
distribution evolution.

### 5.3 Stam-Type Inequalities

Another family of classical results connects Fisher information and entropy
through inequalities of convolution and smoothing. The exact technical forms
vary, but the conceptual theme is stable:

> Gaussian distributions optimize or extremize many joint entropy-Fisher
> relationships.

One representative example is **Stam's inequality**, which in one of its common
forms implies a reciprocal-information inequality for sums of independent
random variables:

$$ \frac{1}{J(X+Y)} \ge \frac{1}{J(X)} + \frac{1}{J(Y)} $$

for suitable smooth independent variables $X$ and $Y$.

This is structurally reminiscent of other Gaussian-extremal facts:

- entropy power inequalities
- central limit smoothing effects
- Gaussian maximality under variance constraints

What should you take away without diving into every proof?

1. Fisher information decreases under additive Gaussian smoothing.
2. Convolution tends to reduce local sharpness.
3. Gaussian laws occupy a distinguished extremal role.

This matters because many modern generative and denoising procedures
systematically inject noise and then learn to reverse or control that smoothing
process. Stam-type results tell us that Fisher is the right local quantity for
tracking how sharpness behaves under such transformations.

### 5.4 Heat Flow, Noise Injection, and Scores

The de Bruijn identity and Stam-type inequalities are easiest to remember if we
view them through heat flow.

Adding Gaussian noise to a random variable corresponds to evolving its density
under the heat equation. As this happens:

- sharp local structure gets blurred
- entropy rises
- Fisher information falls

This gives a clean dynamic picture:

```text
GAUSSIAN SMOOTHING FLOW
===============================================================

start with density p(x)
        |
        v
add small Gaussian noise
        |
        v
density becomes smoother
        |
        +--> entropy increases
        +--> Fisher information decreases
        +--> score field becomes less sharp

entropy tracks global spread
Fisher tracks local sharpness
===============================================================
```

In one dimension, Fisher information for a density $p$ can be written as

$$ J(p) = \int p(x)\left(\frac{d}{dx}\log p(x)\right)^2 dx. $$

So when the log-density has steep derivatives, Fisher information is large.
Heat flow blunts those derivatives, pushing Fisher downward.

This point is especially useful pedagogically because it cures a possible
misunderstanding:

> Fisher information is not "more uncertainty."

It is closer to the opposite. Large Fisher information corresponds to sharp
local structure and strong local distinguishability. When Gaussian noise blurs
the density, entropy increases but Fisher information tends to decrease.

This entropy-Fisher tradeoff is one of the most elegant dualities in the entire
subject.

### 5.5 Why These Identities Matter for Modern AI

At first glance, de Bruijn and Stam may look too classical for machine
learning. But they matter because they teach the right local language for noisy
generative modeling.

Three modern AI-facing lessons emerge.

**1. Score fields are natural objects under noise.**
Diffusion and score-based models are built around learning vector fields related
to gradients of log densities. The entropy-Fisher identities explain why
gradients of log densities are the right infinitesimal objects to study under
Gaussian corruption.

**2. Noise schedules are not arbitrary smoothing tricks.**
Adding noise changes entropy and Fisher in predictable directional ways. Even if
we do not use de Bruijn's identity explicitly in every training loop, the
theory explains why Gaussian perturbation creates increasingly smooth
intermediate distributions.

**3. Local geometry matters in generative modeling.**
Likelihood, score matching, denoising objectives, and reverse-time SDE/ODE
views are all easier to interpret once we understand that Fisher-type objects
measure local structural sharpness.

This section is not the canonical home for diffusion models or score matching
in full generality, but the identities here provide conceptual infrastructure
that those topics rely on.

## 6. Advanced Topics

### 6.1 Fisher vs Hessian vs Gauss-Newton

In machine learning, these three objects are often used as if they were
interchangeable:

- Fisher information
- Hessian of the loss
- Gauss-Newton matrix

They are related, but they are not the same object in general.

**Fisher information** is the expected outer product of score vectors under the
model distribution.

**Hessian** is the exact second derivative matrix of the objective being
optimized.

**Gauss-Newton** is a positive-semidefinite approximation that drops certain
second-derivative terms and is especially natural for least-squares and
likelihood-style problems.

When do they align most closely?

- near well-specified likelihood optima
- in generalized linear-model style settings
- when the objective is negative log-likelihood and model assumptions are well
  matched

When do they differ?

- for misspecified models
- for non-log-likelihood objectives
- far from optimum
- in large neural nets with strong nonlinearities and saturation

This is why the sentence "the Fisher is the Hessian" is both common and
dangerous. It can be a helpful local approximation, but it is not an identity
you should use blindly.

### 6.2 True Fisher vs Empirical Fisher

The empirical Fisher is popular because it is easy to compute: take gradients on
observed examples and average their outer products.

But the true Fisher is defined through expectation under the model distribution,
not just through the observed-label gradients on a finite sample.

Why practitioners like empirical Fisher:

- simple to estimate from minibatches
- automatically positive semidefinite
- cheap compared to exact curvature objects

Why the approximation can fail:

- it uses realized targets rather than the full model-induced distribution
- it need not capture the same second-order structure as the true Fisher
- it can behave very differently away from the optimum

This distinction became important enough in ML theory that there are now
dedicated analyses showing the empirical Fisher can have undesirable pathologies
when used as a stand-in for natural-gradient geometry.

The safe takeaway is:

> empirical Fisher can be a useful heuristic approximation, but it is not the
> theoretical Fisher information matrix by default.

### 6.3 Misspecification and Robust Information

In the ideal parametric story, the model family contains the truth. In reality,
models are often misspecified.

When misspecification occurs, two score-based matrices matter:

- the outer-product variance of the score
- the expected negative Hessian

Under correct specification, these coincide. Under misspecification, they can
separate. The resulting asymptotic covariance formulas use sandwich-type
corrections rather than a single clean Fisher inverse.

This section will not fully develop robust asymptotic theory, because that
belongs more naturally in advanced statistics. But it is important to know that
"the Fisher matrix" is cleanest in the well-specified setting.

For machine learning, misspecification is the norm rather than the exception.
That is one reason local Fisher geometry remains useful as a principled
reference object, even if practical uncertainty quantification often needs more
than naive Fisher inverses.

### 6.4 Large-Scale Approximations

The exact Fisher matrix for a modern deep model is too large to form or invert
directly. This forces approximations.

Common approximation families:

- **Diagonal Fisher** - cheapest, but loses coupling
- **Block-diagonal Fisher** - keeps some structure by layer or module
- **Kronecker-factored approximations** - exploit layer structure in dense or
  convolutional modules
- **Low-rank or sketching methods** - keep only dominant directions

Each approximation trades off:

- memory
- computation
- faithfulness to the true geometry

This section treats these approximations conceptually. The engineering and
optimizer-level details belong in Chapter 8.

Still, the high-level idea is worth emphasizing:

> Approximate Fisher methods are not trying to approximate "some arbitrary
> matrix." They are trying to preserve the local statistical geometry while
> making it computationally tractable.

### 6.5 Singular and Symmetric Neural Models

Deep networks often live in regimes where classical regularity assumptions are
strained:

- parameters can be redundant
- functions can be unchanged by permutations
- activations can become locally insensitive
- the same predictor can correspond to many parameter vectors

This means Fisher matrices can be poorly conditioned or singular even when
training is proceeding normally.

The consequence is not that Fisher information becomes useless. Rather, we
should interpret it more carefully:

- large eigenvalues identify highly sensitive parameter directions
- tiny eigenvalues often reveal symmetry or redundancy
- damping, pseudo-inverses, or structured approximations become essential

This is a major reason why practical Fisher-based optimization is an art as
well as a theorem.

## 7. Applications in Machine Learning

### 7.1 Logistic and Softmax Models

Some of the cleanest Fisher calculations in ML come from logistic and softmax
models.

For binary logistic regression with

$$ p_\theta(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x}), $$

the Fisher matrix has the form

$$ I(\mathbf{w}) = \mathbb{E}\!\left[\sigma(z)(1-\sigma(z)) \mathbf{x}\mathbf{x}^\top\right], \qquad z = \mathbf{w}^\top \mathbf{x}. $$

This matrix automatically weights examples by predictive uncertainty:

- examples with probability near $1/2$ contribute strongly
- saturated examples near $0$ or $1$ contribute less

That is a very intuitive and useful fact. Fisher information in logistic models
emphasizes regions where the model is locally most sensitive.

For multiclass softmax models, the Fisher matrix involves the softmax covariance
structure

$$ \operatorname{diag}(\hat{\mathbf{p}}) - \hat{\mathbf{p}}\hat{\mathbf{p}}^\top. $$

This is exactly the same local probability geometry that appears in cross-
entropy Hessians and generalized Gauss-Newton approximations.

So Fisher information is not an abstract add-on to classification. It is built
into the local curvature of the most common predictive models in ML.

### 7.2 Natural Gradient and Curvature-Aware Training

Natural gradient methods use Fisher information to define steepest descent in
distribution space rather than Euclidean parameter space.

Why is this attractive for ML?

- different parameter coordinates can be arbitrarily scaled
- raw gradients can be badly conditioned
- two large Euclidean steps can produce almost no change in the model, while a
  tiny Euclidean step in another direction can radically alter predictions

Fisher-aware updates attempt to normalize for this mismatch.

The conceptual update is

$$ \Delta \boldsymbol{\theta} \propto - I(\boldsymbol{\theta})^{-1}\nabla_{\boldsymbol{\theta}} L. $$

In practice, exact inversion is impossible for large networks, but the idea
still guides structured approximations and trust-region methods.

This is especially relevant in large probabilistic models, recurrent systems,
and settings where invariance to reparameterization is not just mathematically
nice, but operationally stabilizing.

### 7.3 K-FAC and Structured Fisher Approximations

K-FAC is one of the most influential large-scale approximations to natural-
gradient geometry.

Its core idea is that in layered neural networks, certain blocks of the Fisher
matrix can be approximated by Kronecker products of smaller factors. This makes
matrix inversion much cheaper while preserving more structure than a purely
diagonal approximation.

Why K-FAC matters conceptually:

- it treats Fisher geometry as worth preserving
- it exploits model structure rather than discarding it
- it demonstrates that local information geometry can be operationalized at
  useful scale

K-FAC is not the only such approximation, but it is a canonical example of how
Fisher information motivates algorithm design rather than only statistical
analysis.

The right mental model is:

> K-FAC is a tractability strategy for Fisher-aware optimization, not a claim
> that full Fisher inversion has become cheap.

### 7.4 Elastic Weight Consolidation

Elastic Weight Consolidation (EWC) uses Fisher information in a very different
way from natural gradient.

Instead of preconditioning updates, EWC uses diagonal Fisher information to
estimate how important each parameter was for a previously learned task. When
training on a new task, it penalizes movement along parameters deemed important
for old-task likelihood.

A typical penalty has the form

$$ L_{\mathrm{new}}(\boldsymbol{\theta}) + \frac{\lambda}{2}\sum_j F_j(\theta_j - \theta_j^\star)^2, $$

where $F_j$ is a diagonal Fisher estimate from the old task and
$\theta_j^\star$ is the old-task parameter value.

Interpretation:

- if changing a parameter would strongly alter the old task's likelihood, the
  Fisher entry is large
- if a parameter barely mattered for the old task, the Fisher entry is small

This is a powerful reuse of the local-distinguishability idea. Fisher
information identifies directions that were important to the model's previous
behavior, and EWC tries not to forget them.

The limitations are also instructive:

- diagonal approximations ignore parameter coupling
- empirical Fisher estimates can be noisy
- deep-network symmetries can complicate the notion of parameter importance

Still, EWC is one of the clearest examples of Fisher information leaving
classical statistics and becoming an ML systems tool.

### 7.5 Score Matching, Denoising, and Diffusion Connections

Score-based generative modeling uses derivatives of log densities with respect
to data, not parameters:

$$ \nabla_{\mathbf{x}} \log p(\mathbf{x}). $$

This is not the same object as parameter-space Fisher information, but the
conceptual bridge is direct:

- both are built from local log-density derivatives
- both encode local geometric structure
- both become especially meaningful under Gaussian smoothing

In score matching, one minimizes objectives related to Fisher divergence. In
diffusion and denoising models, Gaussian corruption and score recovery are the
central moves. The de Bruijn identity and related Fisher-entropy connections
help explain why these procedures are so natural mathematically.

So while the full diffusion chapter would live elsewhere, Fisher information
provides part of the local calculus that makes score-based generative modeling
coherent.

## 8. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
| --- | --- | --- | --- |
| 1 | Treating Fisher information as "just the Hessian." | Fisher is the expected score outer product and, under regularity, the negative expected Hessian. The observed Hessian is sample-specific and not identical in general. | Keep expected Fisher, observed information, and Hessian separate unless you have conditions that justify equating them. |
| 2 | Thinking Fisher information is globally invariant as a number. | Scalar and matrix entries change under reparameterization. What is invariant is the induced metric form. | Track the quadratic form or metric interpretation, not raw matrix entries alone. |
| 3 | Confusing parameter score and data score. | $\nabla_{\boldsymbol{\theta}}\log p(x \mid \boldsymbol{\theta})$ and $\nabla_{\mathbf{x}}\log p(\mathbf{x})$ play different roles. | Always say which variable you are differentiating with respect to. |
| 4 | Using empirical Fisher as if it were the true Fisher automatically. | Empirical Fisher is a convenient approximation, not the same object by definition. | Name the approximation explicitly and explain why you are using it. |
| 5 | Assuming high Fisher means "more uncertainty." | High Fisher means stronger local distinguishability and typically lower variance bounds. | Remember that inverse Fisher behaves like a local variance scale. |
| 6 | Forgetting regularity conditions. | Mean-zero score and expected-Hessian identities can fail if support depends on the parameter or smoothness breaks. | Use the score-outer-product definition as primary and mention assumptions when invoking equivalent forms. |
| 7 | Believing Fisher must always be invertible. | Redundancy, symmetry, and non-identifiability create zero or tiny eigenvalues. | Expect singularity in overparameterized or symmetric models and use damping or pseudo-inverses where appropriate. |
| 8 | Treating Jeffreys prior as always proper. | $\sqrt{\det I(\boldsymbol{\theta})}$ can fail to integrate over the parameter domain. | Check propriety separately from invariance arguments. |
| 9 | Thinking natural gradient solves all optimization problems. | It improves local geometry, but inversion, noise, singularity, and approximation quality still matter. | View Fisher-based updates as principled geometry-aware tools, not universal silver bullets. |
| 10 | Ignoring chapter boundaries and re-proving the full Cramer-Rao theory here. | The canonical home for CRB and asymptotic efficiency is Estimation Theory, not this Information Theory section. | Use forward and backward references instead of duplicating full proofs. |
| 11 | Assuming Fisher and Gauss-Newton coincide everywhere in neural nets. | They align only in specific likelihood-structured settings and approximations. | State the conditions under which the approximation is justified. |
| 12 | Missing the entropy connection. | Fisher can look like a purely statistical curvature object, but de Bruijn and related identities show it links directly to information theory. | Keep both the local-geometry and entropy-smoothing viewpoints in mind. |

## 9. Exercises

1. **Exercise 1 [*] - Bernoulli and Poisson Fisher information.**
   Derive the score and Fisher information for Bernoulli($p$) and
   Poisson($\lambda$), and explain the parameter regions where each model is
   most informative.

2. **Exercise 2 [*] - Additivity under iid sampling.**
   Starting from the score of an iid sample, prove that
   $I_n(\theta) = nI(\theta)$ for scalar parameters and explain the connection
   to $1/\sqrt{n}$ uncertainty scaling.

3. **Exercise 3 [**] - Reparameterization.**
   For the Bernoulli model, compute Fisher information both in probability
   coordinates $p$ and in logit coordinates
   $\phi = \log \frac{p}{1-p}$, and verify the transformation law explicitly.

4. **Exercise 4 [**] - Local KL curvature.**
   Perform a second-order Taylor expansion of
   $D_{\mathrm{KL}}(p_\theta \| p_{\theta+\delta})$ for a smooth scalar model
   and show that the quadratic coefficient is $\frac{1}{2}I(\theta)$.

5. **Exercise 5 [**] - Jeffreys prior.**
   Derive Jeffreys prior for Bernoulli($p$), Poisson($\lambda$), and the
   Gaussian mean model with known variance. Identify which priors are proper and
   which are not.

6. **Exercise 6 [***] - True Fisher, empirical Fisher, and Hessian.**
   On a toy logistic-regression problem, compute a minibatch empirical Fisher,
   an observed Hessian, and a Monte-Carlo approximation to the true Fisher. Compare
   them numerically and interpret the differences.

7. **Exercise 7 [***] - Natural gradient or K-FAC-style structure.**
   Derive the Fisher matrix for a logistic model and use it to construct a
   natural-gradient update. Then explain what would need to be approximated in a
   multilayer network for this idea to scale.

8. **Exercise 8 [***] - Fisher in a modern AI workflow.**
   Choose one of the following:
   (a) diagonal Fisher for continual-learning importance weights,
   (b) Fisher divergence in score matching,
   (c) Jeffreys prior in a one-parameter Bayesian model.
   Build a small worked example and explain the modeling tradeoffs.

## 10. Why This Matters for AI (2026 Perspective)

| Concept | AI impact |
| --- | --- |
| Score variance interpretation | Clarifies how local sensitivity of probabilistic models translates into learnability and uncertainty |
| Fisher information matrix | Provides a local geometry for model families and informs curvature-aware training |
| Expected vs empirical Fisher | Separates principled information geometry from convenient deep-learning approximations |
| Local KL curvature | Explains trust-region and natural-gradient ideas in probabilistic terms |
| Fisher-Rao metric | Gives an intrinsic notion of distance between nearby models, useful for invariant optimization and geometry |
| Jeffreys prior | Connects invariant Bayesian defaults to local information geometry |
| Fisher divergence | Bridges local score-field matching and likelihood-free or unnormalized modeling ideas |
| de Bruijn identity | Connects noise injection, entropy growth, and local sharpness in generative modeling |
| K-FAC-style approximations | Shows how Fisher-aware updates can become computationally plausible in large nets |
| EWC | Uses Fisher diagonals to protect old-task knowledge in continual learning |
| Singular Fisher directions | Explains why overparameterized neural models can have flat, redundant, or weakly identifiable directions |
| Diffusion / score connections | Provides conceptual grounding for why local density gradients matter so much in modern generative AI |

Fisher information matters in 2026 because AI systems are no longer judged only
by raw accuracy or loss curves. We care about stability, calibration,
catastrophic forgetting, parameter efficiency, and local uncertainty geometry.
Fisher information sits near the center of all those concerns.

It is also a concept that travels unusually well across abstraction levels. The
same matrix can appear as:

- a variance of scores in a statistics proof
- a metric tensor in information geometry
- a preconditioner target in natural-gradient optimization
- a parameter-importance proxy in continual learning
- a conceptual bridge to score-based generative modeling

That is rare. Most mathematical objects live primarily in one chapter of the
curriculum. Fisher information touches many.

There is also a strategic reason advanced practitioners keep returning to
Fisher-based ideas. Modern foundation-model training mixes at least four
pressures at once:

- models are huge, so exact curvature is impossible
- parameterizations are highly redundant
- calibration and uncertainty still matter
- continual or modular adaptation is increasingly common

Fisher information does not solve all four by itself, but it gives a common
language for all four. That is why it appears repeatedly in papers that look
superficially unrelated: optimizers, continual-learning penalties, local
uncertainty approximations, and score-based generative methods are all
borrowing from the same local geometry.

For LLM-era systems, the most realistic takeaway is not "compute the exact
Fisher matrix." It is:

1. understand what Fisher geometry says the *right* local metric should be
2. understand which approximations preserve that geometry faithfully enough for
   your use case
3. know when the approximation you are using is merely heuristic

That distinction between principle and approximation is a large part of mature
ML engineering in 2026.

## 11. Conceptual Bridge

Fisher information closes the Information Theory chapter by turning global
divergence ideas into local geometry. [Entropy](../01-Entropy/notes.md) gave us
intrinsic uncertainty. [KL Divergence](../02-KL-Divergence/notes.md) gave us a
global mismatch measure. [Mutual Information](../03-Mutual-Information/notes.md)
gave us uncertainty reduction between variables. [Cross-Entropy](../04-Cross-Entropy/notes.md)
gave us expected code length under a model. Fisher information now gives us the
local second-order structure of those model families.

Backward, it connects strongly to
[07-Statistics/02-Estimation-Theory](../../07-Statistics/02-Estimation-Theory/notes.md),
where Fisher information appears in the asymptotic theory of maximum likelihood,
standard errors, and the Cramer-Rao lower bound. That Statistics section is the
canonical home of full efficiency theory. This section reframes the same object
through the lens of local information geometry.

Forward, the next major connections are:

- to [08-Optimization/03-Second-Order-Methods](../../08-Optimization/03-Second-Order-Methods/notes.md)
  through natural gradient, structured curvature, and K-FAC
- to [07-Statistics/04-Bayesian-Inference](../../07-Statistics/04-Bayesian-Inference/notes.md)
  through Jeffreys priors and local curvature approximations
- to future generative-model sections through score matching, denoising, and
  diffusion-style local density geometry

```text
FISHER INFORMATION IN THE CURRICULUM
===============================================================

Entropy
  uncertainty of a distribution
        |
        v
KL divergence
  global mismatch between distributions
        |
        v
Cross-entropy
  expected code length under a model
        |
        v
Fisher information
  local curvature of model distinguishability
        |
        +--> Cramer-Rao and asymptotic variance
        +--> Jeffreys prior
        +--> natural gradient
        +--> K-FAC / curvature approximations
        +--> EWC
        +--> score matching and diffusion intuition
===============================================================
```

If there is one sentence to keep from this chapter, it is this:

> Fisher information measures how sharply nearby models become distinguishable.

That sentence unifies likelihood curvature, estimator precision, local KL
geometry, natural gradient, and the modern reuse of Fisher ideas in deep
learning.

There is an additional bridge worth carrying forward into later chapters:

- in estimation theory, Fisher tells us how hard it is to infer parameters
- in optimization, Fisher tells us how to move parameters in a geometry-aware
  way
- in Bayesian workflows, Fisher shapes invariant priors and local posterior
  approximations
- in generative modeling, Fisher-type score objects tell us how densities
  respond to infinitesimal smoothing and denoising

Seen this way, Fisher information is not "one more formula after KL." It is the
object that converts information theory from a language of global divergences
into a language of local geometry.

That local-geometry perspective is one of the main mathematical habits that
separates beginner probabilistic ML from advanced probabilistic ML. Once you
start asking not only "what is the loss?" but also "what metric is this update
implicitly using?" or "what local directions are statistically meaningful?"
you are already thinking in Fisher's language.

This is why the chapter order matters. Entropy, KL, mutual information, and
cross-entropy teach you how to measure uncertainty and mismatch globally.
Fisher information teaches you how those global quantities behave
infinitesimally. That is the natural endpoint of the Information Theory chapter
and a natural launch point into geometry-aware optimization and probabilistic
modeling.

## References

1. Fisher, R. A. (1925). *Theory of Statistical Estimation*. Proceedings of the Cambridge Philosophical Society.
2. Cramer, H. (1946). *Mathematical Methods of Statistics*. Princeton University Press.
3. Rao, C. R. (1945). "Information and the Accuracy Attainable in the Estimation of Statistical Parameters." *Bulletin of the Calcutta Mathematical Society*.
4. Jeffreys, H. (1946). "An Invariant Form for the Prior Probability in Estimation Problems." *Proceedings of the Royal Society A*.
5. Cover, T. M., and Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Wiley.
6. Duchi, J. C. *Lecture Notes on Statistics and Information Theory*. [PDF](https://web.stanford.edu/class/stats311/lecture-notes.pdf)
7. MIT OpenCourseWare. *Statistics for Applications - lecture3.pdf*. [Course resource](https://ocw.mit.edu/courses/18-443-statistics-for-applications-fall-2006/resources/lecture3/)
8. Ly, A., Marsman, M., Verhagen, J., Grasman, R., and Wagenmakers, E.-J. (2017). "A Tutorial on Fisher Information." [arXiv](https://arxiv.org/abs/1705.01064)
9. Amari, S. (1998). "Natural Gradient Works Efficiently in Learning." *Neural Computation*. [Publisher page](https://direct.mit.edu/neco/article/10/2/251/6143/Natural-Gradient-Works-Efficiently-in-Learning)
10. Martens, J., and Grosse, R. (2015). "Optimizing Neural Networks with Kronecker-factored Approximate Curvature." [arXiv](https://arxiv.org/abs/1503.05671)
11. Kirkpatrick, J., Pascanu, R., Rabinowitz, N., et al. (2017). "Overcoming catastrophic forgetting in neural networks." [arXiv](https://arxiv.org/abs/1612.00796)
12. Kunstner, F., Balles, L., and Hennig, P. (2020). "Limitations of the Empirical Fisher Approximation for Natural Gradient Descent." [arXiv](https://arxiv.org/abs/1905.12558)
13. Amari, S., Karakida, R., and Oizumi, M. (2019). "Fisher Information and Natural Gradient Learning in Random Deep Networks." [PMLR](https://proceedings.mlr.press/v89/amari19a.html)
14. Gupta, V., Koren, T., and Singer, Y. (2018). "Shampoo: Preconditioned Stochastic Tensor Optimization." Proceedings of ICML.
15. Hyvarinen, A. (2005). "Estimation of Non-Normalized Statistical Models by Score Matching." *Journal of Machine Learning Research*.
