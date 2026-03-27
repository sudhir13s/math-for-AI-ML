[← Back to Chapter 7: Statistics](../README.md) | [Next: Time Series →](../05-Time-Series/notes.md)

---

# Bayesian Inference

> _"Probability theory is nothing but common sense reduced to calculation."_
> - Pierre-Simon Laplace

## Overview

Bayesian inference treats uncertainty itself as an object of computation. Instead of saying "the parameter has one true but unknown value and our job is to estimate it as accurately as possible," the Bayesian view says: before seeing data we have a distribution of plausible parameter values, after seeing data we should update that distribution, and every decision should propagate the remaining uncertainty rather than pretending it disappeared. The result is a full probabilistic description of what we know, what we do not know, and how strongly the data have shifted our beliefs.

This perspective is not a philosophical ornament. In machine learning it is the mathematical language of uncertainty quantification, small-data learning, calibration, active learning, Bayesian optimization, approximate posterior inference in latent-variable models, and posterior-predictive reasoning for safety-critical decisions. Weight decay can be read as a Gaussian prior. Sparsity can be read as a Laplace prior. Variational autoencoders optimize an evidence lower bound because exact posterior computation is intractable. MC dropout, SGLD, Laplace approximations, and SWAG all exist because practitioners want some version of Bayesian uncertainty without paying the full computational price of exact inference.

This section develops Bayesian inference in a way that is rigorous, scoped, and useful for modern AI. We will not re-teach Bayes' theorem from probability theory, and we will not re-teach MLE, Fisher information, or classical confidence intervals from the previous statistics sections. Instead, we treat those as prerequisites and focus on what is new here: priors, posteriors, evidence, conjugacy, posterior prediction, credible intervals, Bayes factors, hierarchical models, and the approximation methods that make Bayesian machine learning computationally viable.

## Prerequisites

- **Bayes' theorem, conditional distributions, and multivariate Gaussian conditioning** - [Chapter 6, Joint Distributions](../../06-Probability-Theory/03-Joint-Distributions/notes.md)
- **Expectation, variance, covariance, and moments** - [Chapter 6, Expectation and Moments](../../06-Probability-Theory/04-Expectation-and-Moments/notes.md)
- **Markov chains and MCMC background** - [Chapter 6, Markov Chains](../../06-Probability-Theory/07-Markov-Chains/notes.md)
- **MLE, Fisher information, asymptotic normality, and confidence intervals** - [Section 02, Estimation Theory](../02-Estimation-Theory/notes.md)
- **p-values, hypothesis tests, and likelihood-ratio logic** - [Section 03, Hypothesis Testing](../03-Hypothesis-Testing/notes.md)
- **Matrix algebra and positive definite matrices** - [Chapter 3, Advanced Linear Algebra](../../03-Advanced-Linear-Algebra/README.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Interactive derivations and simulations for conjugate Bayes, MAP, posterior predictive inference, Bayes factors, MCMC, VI, and Bayesian ML examples |
| [exercises.ipynb](exercises.ipynb) | 8 graded exercises covering closed-form posterior updates, MAP estimation, posterior prediction, Bayes factors, sampling, ELBO derivations, and Bayesian uncertainty in ML |

## Learning Objectives

After completing this section, you will:

- Define prior, likelihood, posterior, and evidence precisely and explain the role each plays in Bayesian inference
- Distinguish posterior uncertainty from sampling-distribution uncertainty without confusing credible intervals and confidence intervals
- Derive conjugate posterior updates for Beta-Binomial, Gamma-Poisson, Dirichlet-Categorical, and Gaussian-Gaussian models
- Explain why MAP estimation is regularized MLE and identify the corresponding prior for L2 and L1 penalties
- Compute posterior predictive distributions and explain why they differ from plug-in predictions
- Interpret posterior covariance and shrinkage in small-data and noisy-data settings
- Use Bayes factors and marginal likelihoods for model comparison and explain the associated Occam penalty
- Explain partial pooling in hierarchical Bayes and why it improves estimation across related groups
- Describe the computational goals of MCMC, variational inference, Laplace approximation, SGLD, MC dropout, and SWAG
- Connect Bayesian reasoning to calibration, active learning, Bayesian optimization, Bayesian neural networks, and uncertainty-aware deployment

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 From Data to Belief](#11-from-data-to-belief)
  - [1.2 Frequentist vs Bayesian Uncertainty](#12-frequentist-vs-bayesian-uncertainty)
  - [1.3 Historical Timeline](#13-historical-timeline)
  - [1.4 Why Bayesian Inference Matters for AI](#14-why-bayesian-inference-matters-for-ai)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Prior, Likelihood, Posterior, Evidence](#21-prior-likelihood-posterior-evidence)
  - [2.2 Continuous Bayes and Normalisation](#22-continuous-bayes-and-normalisation)
  - [2.3 Posterior Summaries and Bayes Estimators](#23-posterior-summaries-and-bayes-estimators)
  - [2.4 Credible Intervals vs Confidence Intervals](#24-credible-intervals-vs-confidence-intervals)
  - [2.5 Prior Families and Elicitation](#25-prior-families-and-elicitation)
- [3. Core Theory I: Conjugacy and Closed-Form Updating](#3-core-theory-i-conjugacy-and-closed-form-updating)
  - [3.1 Conjugate Priors and Exponential Families](#31-conjugate-priors-and-exponential-families)
  - [3.2 Beta-Binomial Model](#32-beta-binomial-model)
  - [3.3 Gamma-Poisson and Dirichlet-Categorical Models](#33-gamma-poisson-and-dirichlet-categorical-models)
  - [3.4 Gaussian-Gaussian Updating](#34-gaussian-gaussian-updating)
  - [3.5 What Conjugacy Buys and What It Misses](#35-what-conjugacy-buys-and-what-it-misses)
- [4. Core Theory II: Estimation, Prediction, and Uncertainty](#4-core-theory-ii-estimation-prediction-and-uncertainty)
  - [4.1 MAP Estimation as Regularised MLE](#41-map-estimation-as-regularised-mle)
  - [4.2 Posterior Covariance and Shrinkage](#42-posterior-covariance-and-shrinkage)
  - [4.3 Posterior Predictive Distribution](#43-posterior-predictive-distribution)
  - [4.4 Posterior Predictive Checks](#44-posterior-predictive-checks)
  - [4.5 Bayesian Linear Regression as a Worked Example](#45-bayesian-linear-regression-as-a-worked-example)
- [5. Core Theory III: Model Comparison and Hierarchical Bayes](#5-core-theory-iii-model-comparison-and-hierarchical-bayes)
  - [5.1 Marginal Likelihood and Occam's Razor](#51-marginal-likelihood-and-occams-razor)
  - [5.2 Bayes Factors and Bayesian Model Comparison](#52-bayes-factors-and-bayesian-model-comparison)
  - [5.3 Empirical Bayes](#53-empirical-bayes)
  - [5.4 Hierarchical Models and Partial Pooling](#54-hierarchical-models-and-partial-pooling)
  - [5.5 Prior Sensitivity and Robustness](#55-prior-sensitivity-and-robustness)
- [6. Advanced Topics: Approximate Inference](#6-advanced-topics-approximate-inference)
  - [6.1 Why Exact Posteriors Become Intractable](#61-why-exact-posteriors-become-intractable)
  - [6.2 MCMC for Posterior Sampling](#62-mcmc-for-posterior-sampling)
  - [6.3 Variational Inference and the ELBO](#63-variational-inference-and-the-elbo)
  - [6.4 Amortized Inference and VAEs](#64-amortized-inference-and-vaes)
  - [6.5 Laplace, MC Dropout, SGLD, and SWAG](#65-laplace-mc-dropout-sgld-and-swag)
- [7. Applications in Machine Learning](#7-applications-in-machine-learning)
  - [7.1 Naive Bayes as Generative Classification](#71-naive-bayes-as-generative-classification)
  - [7.2 Bayesian Neural Networks](#72-bayesian-neural-networks)
  - [7.3 Uncertainty for Calibration, Active Learning, and OOD Detection](#73-uncertainty-for-calibration-active-learning-and-ood-detection)
  - [7.4 Bayesian Optimization and Thompson Sampling](#74-bayesian-optimization-and-thompson-sampling)
  - [7.5 Bayesian Views of Regularization and Fine-Tuning](#75-bayesian-views-of-regularization-and-fine-tuning)
- [8. Common Mistakes](#8-common-mistakes)
- [9. Exercises](#9-exercises)
- [10. Why This Matters for AI (2026 Perspective)](#10-why-this-matters-for-ai-2026-perspective)
- [11. Conceptual Bridge](#11-conceptual-bridge)

---

## 1. Intuition

### 1.1 From Data to Belief

Frequentist estimation begins with a fixed but unknown parameter and asks how a data-dependent procedure behaves over repeated samples. Bayesian inference begins one step earlier. Before seeing data, we already have a state of uncertainty about the parameter. That uncertainty may come from previous experiments, physical constraints, domain expertise, symmetry assumptions, or deliberately weak prior information. The parameter is not "random" because nature rolls a die after the data arrive; it is random because uncertainty is being modeled explicitly.

Suppose a startup deploys a new ranking model and wants to know the probability that the click-through rate exceeds the old model's rate by at least 1 percentage point. A point estimate does not answer that question. A p-value does not answer that question either. A posterior distribution does. Once we have $p(\theta \mid \mathcal{D})$, we can compute probabilities of events such as $P(\theta > 0.01 \mid \mathcal{D})$, expected utilities, posterior predictive risk, and the distribution of future outcomes.

This makes Bayesian inference especially natural for sequential decision-making. Every posterior after one round of data becomes the prior for the next round. The algebra says
$$
p(\theta \mid \mathcal{D}_{1:t}) \propto p(\mathcal{D}_t \mid \theta)\, p(\theta \mid \mathcal{D}_{1:t-1}),
$$
which means learning is literally iterative belief revision. Online advertising, scientific experimentation, preference learning, and Bayesian optimization all exploit this recursive structure.

There is also a structural reason Bayesian thinking feels natural in AI. Machine learning systems are rarely used only for retrospective explanation. They are used to act: rank, classify, recommend, optimize, allocate compute, or trigger alarms. To act sensibly, a system needs not just a best guess but a measure of confidence. Bayesian inference provides one principled route from data to uncertainty-aware action.

```text
ASCII VIEW OF BAYESIAN UPDATING
======================================================================

  prior belief about parameter theta
              +
  likelihood of observed data under theta
              |
              v
  posterior belief after seeing the data
              |
              v
  predictions, decisions, uncertainty intervals, model comparison

======================================================================
```

Three examples clarify the basic idea.

**Example 1: coin bias.** We observe $k$ heads in $n$ flips. Before seeing any flips, we place a prior on the bias $p$. After the flips, the posterior sharpens around values consistent with the data. If $n$ is small, the prior still matters. If $n$ is large, the likelihood dominates.

**Example 2: sparse model weights.** In linear prediction or logistic regression, a Laplace prior on coefficients expresses a belief that many coefficients should be near zero. The resulting MAP estimate becomes L1-regularized optimization. The full posterior adds uncertainty quantification around that sparse tendency.

**Example 3: exploration in recommender systems.** If we have high posterior uncertainty about a new recommendation policy, the Bayesian decision is not simply "ignore it until enough data accumulate." The posterior can justify exploration because uncertainty itself has value when it can be reduced.

Non-examples matter too, because Bayesian language is easy to misuse.

- A posterior is **not** just "the likelihood curve, relabeled." The prior changes the shape unless it is flat or asymptotically negligible.
- A prior is **not** necessarily subjective whim. Symmetry arguments, weakly informative constraints, hierarchical structure, and invariance principles often determine reasonable choices.

**For AI:** Frontier model evaluation increasingly cares about calibration, abstention, uncertainty-triggered fallback, and sample efficiency. Those are posterior questions. They ask what the model still does not know after training, not merely which point estimate optimizes a loss.

### 1.2 Frequentist vs Bayesian Uncertainty

The sharpest distinction between frequentist and Bayesian inference is not computational. It is semantic.

In frequentist statistics, the parameter $\theta$ is fixed. Randomness lives in the sample. Therefore a 95% confidence interval means: if we repeated the whole data-generating and interval-building procedure many times, 95% of those intervals would contain the true $\theta$. The probability statement attaches to the procedure.

In Bayesian statistics, the observed data are fixed once collected, and uncertainty lives in the posterior over $\theta$. Therefore a 95% credible interval means: under the posterior distribution, the probability that $\theta$ lies in the interval is 0.95. The probability statement attaches directly to the parameter because the parameter is being modeled as uncertain.

This distinction is not cosmetic. It changes what questions are meaningful.

- Frequentist question: "How often would this method cover the truth over repeated samples?"
- Bayesian question: "Given the data actually observed, how much posterior mass lies in this region?"

Neither framework dominates all others in every context. Frequentist guarantees are valuable when procedures must perform well across repeated deployment. Bayesian posterior statements are valuable when a single realized dataset must support a concrete decision right now. In practice, ML engineers often want both: calibrated uncertainty on the realized dataset plus reliable behavior under repeated use.

The previous section on [Estimation Theory](../02-Estimation-Theory/notes.md) already developed confidence intervals and asymptotic normality. We will not duplicate that material here. Instead, we build the Bayesian parallel and emphasize where the two frameworks coincide and where they do not.

There are important overlap points:

1. Under regularity conditions and large sample size, the posterior is often approximately Gaussian around the MLE.
2. Under the same conditions, a Bayesian credible interval can numerically resemble a frequentist confidence interval.
3. MAP estimation with a flat prior reduces to MLE.

But there are equally important divergences:

1. Bayesians can assign probabilities to hypotheses and parameters directly; frequentists do not.
2. Bayesian model comparison integrates over parameters; frequentist model comparison usually plugs in estimates or uses long-run testing logic.
3. Priors matter in finite data, and sometimes matter a lot.

```text
FREQUENTIST vs BAYESIAN UNCERTAINTY
======================================================================

  Framework        Parameter         Randomness lives in
  --------------------------------------------------------------------
  Frequentist      fixed             repeated samples
  Bayesian         uncertain         posterior over parameters

  Interval type    Meaning
  --------------------------------------------------------------------
  Confidence       long-run coverage of the method
  Credible         posterior probability for this dataset

  Canonical output
  --------------------------------------------------------------------
  Frequentist      estimator, standard error, CI, p-value
  Bayesian         posterior, posterior mean/mode, credible interval

======================================================================
```

Examples help prevent confusion.

**Example A: same numbers, different meaning.** Suppose both methods output the interval $[0.41, 0.58]$ for a Bernoulli success probability. The frequentist meaning is about repeated procedures. The Bayesian meaning is about posterior mass on this particular problem. Identical endpoints do not imply identical interpretation.

**Example B: no exact frequentist analog.** A product team asks, "what is the probability that model A is better than model B by at least 0.5%?" A posterior over effect size answers directly. A p-value does not.

**Example C: no exact Bayesian analog without choices.** A regulator asks for a decision rule with guaranteed 5% Type I error across repeated trials. Bayesian posterior thresholds can be tuned to behave this way, but the guarantee is not automatic; it depends on the prior and the loss function.

Common non-examples:

- "A 95% confidence interval means there is 95% probability the parameter lies in it." False.
- "A 95% credible interval is automatically better because it is easier to interpret." Not always. It can be more decision-relevant, but only relative to a prior and model that deserve trust.

**For AI:** Model deployment often needs statements such as "the posterior probability this policy violates the safety threshold is below 1%." That is a Bayesian statement. Benchmark reporting often needs statements such as "this evaluation protocol has 95% coverage under repeated re-sampling." That is a frequentist statement. Mature systems increasingly use both.

### 1.3 Historical Timeline

| Year | Contributor | Contribution |
| --- | --- | --- |
| 1763 | Thomas Bayes | Posthumous essay introduces inverse probability in a simple setting |
| 1774-1812 | Pierre-Simon Laplace | Generalizes Bayesian updating, develops posterior approximation ideas, and applies inverse probability broadly |
| 1810s-1900s | Gauss, Poisson, others | Use probabilistic inversion ideas in astronomy and measurement, often without modern Bayesian terminology |
| 1939 | Harold Jeffreys | Formalizes objective Bayes and Jeffreys priors; emphasizes invariance |
| 1954 | Leonard J. Savage | Modern subjective Bayesian decision theory |
| 1970s | Lindley, Bernardo, Berger | Develop reference prior theory, decision-theoretic Bayes, and formal objective-Bayes programs |
| 1990s | Gelfand, Smith, Neal, Gelman | MCMC makes Bayesian computation practical for complex models |
| 2000s | Bishop, Murphy | Bayesian ML becomes standard in machine learning curricula |
| 2014 | Kingma and Welling | Variational autoencoder turns approximate posterior inference into a scalable deep-learning primitive |
| 2015 | Blundell et al. | Bayes by Backprop popularizes variational Bayesian neural networks |
| 2016 | Gal and Ghahramani | MC dropout interpreted as approximate Bayesian inference |
| 2019 | Maddox et al. | SWAG provides a scalable uncertainty baseline from SGD iterates |
| 2020s | Wide ecosystem | Bayesian calibration, active learning, approximate posterior methods, and probabilistic decision systems become mainstream in AI practice |
Two themes run through this history.

First, Bayesian inference was conceptually elegant long before it was computationally convenient. For simple conjugate models, the algebra is beautiful. For realistic hierarchical or nonconjugate models, the posterior often contains high-dimensional integrals that were historically intractable.

Second, modern ML revived Bayesian thinking not because the philosophy changed, but because the computational landscape changed. Gradient methods, automatic differentiation, stochastic optimization, and scalable approximate inference let practitioners revisit posterior reasoning at scales that were once impossible.

### 1.4 Why Bayesian Inference Matters for AI

Bayesian inference matters in AI because uncertainty is operational. A frontier model can have excellent average loss and still be dangerously overconfident off-distribution. A medical triage model can have strong AUROC and still be unsafe if it cannot say "I am unsure." A hyperparameter search can waste thousands of GPU-hours if it treats every unexplored configuration as equally promising. Posterior reasoning gives tools for all three.

**Calibration and reliability.** Posterior predictive distributions provide a route to uncertainty-aware prediction. Exact Bayesian posterior predictive inference is often unavailable in deep learning, but approximate methods such as ensembles, Laplace approximations, MC dropout, SGLD, and SWAG are all motivated by the same goal: uncertainty that reacts to data scarcity, ambiguity, or shift.

**Small-data learning and shrinkage.** In low-data regimes, pure MLE is brittle. Priors stabilize estimation. The posterior mean in Gaussian-Gaussian updating is a precision-weighted average of prior knowledge and observed data. Hierarchical models let many related tasks share strength, which is especially important in personalization, recommendation, multilingual NLP, and scientific ML.

**Decision-making under uncertainty.** Thompson sampling, Bayesian optimization, and Bayesian active learning all choose actions by integrating over posterior uncertainty. This turns "exploration vs exploitation" from a heuristic tuning problem into a probabilistic decision problem.

**Interpretation of regularization.** Weight decay is naturally read as a Gaussian prior. L1 penalties reflect Laplace priors. Sparsity-inducing Bayesian models, low-rank priors, and hierarchical shrinkage priors all offer a language for understanding what optimization penalties are trying to express.

**Latent-variable deep learning.** Variational autoencoders, deep latent Gaussian models, and many probabilistic representation-learning methods are built around the problem of approximating posteriors that are unavailable in closed form. The ELBO is a Bayesian object before it is an optimization objective.

Three concrete AI examples show the range.

1. **Calibration for abstention:** a classifier used in content moderation can route high-uncertainty examples to human review instead of issuing brittle hard labels.
2. **Bayesian optimization for expensive training:** when fine-tuning a large model costs hours or days, posterior uncertainty over the response surface makes hyperparameter search far more sample-efficient than blind sweeps.
3. **Posterior predictive anomaly detection:** if the posterior predictive distribution says a new observation is extremely implausible under the learned model, that can trigger an alert for distribution shift or system misuse.

> **Backward reference:** Bayes' theorem itself was developed in [Chapter 6, Joint Distributions](../../06-Probability-Theory/03-Joint-Distributions/notes.md#34-bayes-theorem-for-random-variables). Here we use it as the organizing principle for a full inferential framework.

---

## 2. Formal Definitions

### 2.1 Prior, Likelihood, Posterior, Evidence

Bayesian inference starts from four objects:

$$
p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta)\, p(\theta)}{p(\mathcal{D})}.
$$

Each term has a distinct mathematical role.

- The **prior** $p(\theta)$ encodes uncertainty about the parameter before observing the current dataset.
- The **likelihood** $p(\mathcal{D} \mid \theta)$ measures how compatible the observed data are with a candidate parameter value.
- The **posterior** $p(\theta \mid \mathcal{D})$ is the updated distribution after combining prior information with data.
- The **evidence** or **marginal likelihood** $p(\mathcal{D})$ normalizes the posterior and is obtained by integrating out the parameter:
$$
p(\mathcal{D}) = \int p(\mathcal{D} \mid \theta)\, p(\theta)\, d\theta.
$$

Because the evidence does not depend on $\theta$, the posterior is often written in proportional form:
$$
p(\theta \mid \mathcal{D}) \propto p(\mathcal{D} \mid \theta)\, p(\theta).
$$

This compact formula hides two deep facts.

First, Bayesian inference is a fusion rule. It multiplies two sources of information: what we believed before and what the data say now. Second, the evidence is not a nuisance constant in every context. It is also the quantity that makes Bayesian model comparison possible, because it scores how well an entire model, not just one fitted parameter value, explains the data.

Three examples:

**Example 1: Bernoulli likelihood.** If $x_1, \ldots, x_n \in \{0,1\}$ are coin flips and $\theta$ is the success probability, then
$$
p(\mathcal{D} \mid \theta) = \theta^{\sum_i x_i}(1-\theta)^{n-\sum_i x_i}.
$$
Multiplying by a Beta prior gives a Beta posterior.

**Example 2: Gaussian mean with known variance.** If $x_i \sim \mathcal{N}(\mu, \sigma^2)$ and the prior is $\mu \sim \mathcal{N}(\mu_0, \tau_0^2)$, then the posterior is also Gaussian. The posterior mean is a precision-weighted blend of $\mu_0$ and $\bar{x}$.

**Example 3: Bayesian classifier.** If $y$ is the class label and $\mathbf{x}$ the features, then a posterior class probability
$$
P(y=k \mid \mathbf{x}) \propto p(\mathbf{x} \mid y=k)\, P(y=k)
$$
forms the basis of Naive Bayes and many other generative classifiers.

Non-examples clarify what these terms are not.

- The likelihood is **not** a probability distribution over $\theta$. It is a function of $\theta$ indexed by the observed data.
- The prior is **not** always a belief about observed data. It is a distribution on parameters, hypotheses, or latent variables.

**For AI:** In modern ML notation, the negative log-likelihood is the training loss for many probabilistic models. Bayesian inference adds $\log p(\theta)$ and then, if done fully rather than at the MAP level, integrates over $\theta$ rather than stopping at a single optimizer.

### 2.2 Continuous Bayes and Normalisation

For discrete parameter spaces, Bayes' rule involves finite sums. For continuous parameters, the denominator becomes an integral:
$$
p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta)p(\theta)}{\int p(\mathcal{D} \mid \tilde{\theta})p(\tilde{\theta})\, d\tilde{\theta}}.
$$

This denominator is essential. Without it, the right-hand side is only an unnormalized density. The posterior must integrate to 1:
$$
\int p(\theta \mid \mathcal{D})\, d\theta = 1.
$$

The integral is often easy in conjugate models and painfully hard elsewhere. That computational gap is one reason Bayesian statistics splits naturally into exact inference and approximate inference.

Consider the Gaussian-Gaussian example from [Chapter 6, Joint Distributions](../../06-Probability-Theory/03-Joint-Distributions/notes.md#34-bayes-theorem-for-random-variables). Suppose
$$
X \mid \mu \sim \mathcal{N}(\mu, \sigma^2), \qquad \mu \sim \mathcal{N}(\mu_0, \tau_0^2).
$$
Then
$$
p(\mu \mid x) \propto \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)\exp\left(-\frac{(\mu-\mu_0)^2}{2\tau_0^2}\right).
$$
Completing the square shows the posterior is Gaussian. The normalization constant exists in closed form because the product of two Gaussian kernels is another Gaussian kernel up to scale.

This example teaches a general lesson: Bayesian computation becomes manageable when multiplication by the prior preserves the family of the likelihood kernel.

There are several standard failure modes around normalization:

1. The integral may be analytically unavailable.
2. The prior may be improper, so one must verify the posterior is still proper.
3. The parameter space may be high-dimensional, making numerical integration exponentially hard.

Examples:

- **Proper posterior from improper prior:** for some Gaussian-location problems, a flat improper prior still yields a normalizable posterior.
- **Improper posterior from careless prior:** certain hierarchical models with diffuse priors can fail to normalize, so the posterior is not a valid probability distribution.
- **High-dimensional deep net:** a Bayesian neural network with millions of weights has a posterior integral far beyond exact quadrature.

Non-examples:

- Replacing the evidence with the likelihood at the MLE is not Bayesian inference; that turns the posterior into an unnormalized surrogate.
- Dropping the evidence is okay for MAP optimization, but not for posterior probabilities, marginal likelihoods, Bayes factors, or predictive integration.

**For AI:** The entire machinery of variational inference and MCMC exists because the denominator is usually the computational bottleneck. In deep latent-variable models, the hard part is rarely writing down the posterior proportionality. The hard part is normalizing or integrating with respect to it.

### 2.3 Posterior Summaries and Bayes Estimators

The posterior is a full distribution, but many tasks require a point summary. Bayesian decision theory says the "best" posterior summary depends on the loss function.

Let $a$ be an action or estimate. Under posterior loss
$$
\rho(a) = \mathbb{E}[L(a,\theta) \mid \mathcal{D}],
$$
the Bayes estimator is
$$
a^* = \arg\min_a \rho(a).
$$

Important special cases:

- Under squared loss $L(a,\theta) = (a-\theta)^2$, the Bayes estimator is the **posterior mean** $\mathbb{E}[\theta \mid \mathcal{D}]$.
- Under absolute loss $L(a,\theta) = |a-\theta|$, the Bayes estimator is a **posterior median**.
- Under 0-1 style mode-seeking loss on a discrete space, the Bayes estimator is the **posterior mode**, or MAP.

This already explains why MAP is not the unique Bayesian point estimator. It is one summary among several, and it is optimal only under a specific decision criterion.

Three examples:

**Example 1: asymmetric costs.** In content moderation, a false negative may cost more than a false positive. The optimal Bayesian decision threshold is then not 0.5 posterior probability. It depends on the posterior odds and the loss asymmetry.

**Example 2: posterior mean for forecast demand.** If overprediction and underprediction are penalized quadratically, the posterior mean is the optimal point forecast.

**Example 3: posterior median for robust estimation.** Under absolute loss, the posterior median is less sensitive to heavy tails than the posterior mean.

Non-examples:

- Choosing MAP because it "looks most probable" is not automatically justified if the downstream loss is not mode-seeking.
- Treating posterior mean and MAP as interchangeable is wrong in skewed, multimodal, or heavy-tailed posteriors.

**For AI:** Ranking, retrieval, moderation, and safe control all involve asymmetric losses. Bayesian inference becomes most useful when it is connected to the decision loss explicitly instead of being reduced immediately to one generic point estimate.

### 2.4 Credible Intervals vs Confidence Intervals

A $(1-\alpha)$ Bayesian credible interval $C(\mathcal{D})$ satisfies
$$
P(\theta \in C(\mathcal{D}) \mid \mathcal{D}) = 1-\alpha
$$
under the posterior.

This should be contrasted with the frequentist confidence-interval statement from [Estimation Theory](../02-Estimation-Theory/notes.md#71-the-frequentist-interpretation), where probability refers to the long-run behavior of the method, not to the realized interval.

Two common types of credible intervals are:

1. **Equal-tail interval:** cut $\alpha/2$ posterior mass from each tail.
2. **Highest posterior density (HPD) interval:** choose the smallest region containing posterior mass $1-\alpha$.

Examples:

- For a symmetric unimodal posterior, equal-tail and HPD intervals may be nearly identical.
- For a skewed posterior, the equal-tail interval can be noticeably wider on one side than the HPD set.
- For a multimodal posterior, the HPD region may be disconnected, which reminds us that "interval" language can hide complicated posterior geometry.

Non-examples:

- A credible interval is not guaranteed to have nominal frequentist coverage.
- A confidence interval does not permit the statement "there is 95% probability that $\theta$ lies here" after observing the data.

**For AI:** If a product owner asks for the probability that a model metric exceeds a launch threshold given observed test data, that is a posterior probability statement. Credible intervals and posterior tail probabilities answer it naturally.

### 2.5 Prior Families and Elicitation

Priors come in several broad families.

- **Informative priors:** encode strong domain knowledge.
- **Weakly informative priors:** rule out implausible extremes while leaving substantial room for the data.
- **Conjugate priors:** chosen for analytical convenience.
- **Improper priors:** not normalizable as distributions by themselves, but sometimes still produce proper posteriors.
- **Hierarchical priors:** place priors on hyperparameters so the strength of regularization is itself uncertain.

Prior elicitation is the process of translating domain knowledge into a distribution. In scientific problems, priors may come from physical limits, historical studies, or expert judgments. In ML, priors often come from structural beliefs: sparsity, smoothness, low rank, parameter sharing, or scale constraints.

Examples:

1. A Beta prior on a click-through probability encodes plausible conversion rates.
2. A Gaussian prior on linear coefficients encodes shrinkage toward zero.
3. A hierarchical prior on task-specific parameters says related tasks should be similar but not identical.

Non-examples:

1. "Use a completely flat prior because that is objective." Flatness depends on parameterization and is not invariant.
2. "Use a conjugate prior because it must be correct." Conjugacy buys tractability, not truth.

Jeffreys priors attempt to address reparameterization by using the Fisher information:
$$
p_J(\theta) \propto \sqrt{\mathcal{I}(\theta)}.
$$
This is a principled preview of objective Bayes, but not a universal solution. In modern practice, weakly informative priors are often preferred because they keep inference numerically stable and encode realistic scales.

**For AI:** Prior design is often regularizer design in disguise. If we believe weights should stay small, that is a Gaussian prior. If we believe only a few features matter, that is a sparsity prior. If we believe several tasks share structure, that is a hierarchical prior. Bayesian language makes those assumptions explicit rather than hiding them in penalties and initialization choices.

---

## 3. Core Theory I: Conjugacy and Closed-Form Updating

### 3.1 Conjugate Priors and Exponential Families

A prior family is **conjugate** to a likelihood family if the posterior belongs to the same family as the prior. Conjugacy matters because it converts posterior updating from a numerical integration problem into algebra.

The cleanest setting is the exponential family. If the likelihood can be written as
$$
p(x \mid \theta) = h(x)\exp\left(\eta(\theta)^\top T(x) - A(\theta)\right),
$$
then a conjugate prior often has the form
$$
p(\theta \mid \boldsymbol{\chi}, \nu) \propto \exp\left(\eta(\theta)^\top \boldsymbol{\chi} - \nu A(\theta)\right).
$$

After observing data $x_1, \ldots, x_n$, the posterior updates by simple addition:
$$
\boldsymbol{\chi}_{\text{post}} = \boldsymbol{\chi} + \sum_{i=1}^n T(x_i), \qquad
\nu_{\text{post}} = \nu + n.
$$

This is one of the most revealing formulas in Bayesian statistics. It says prior information and data information often combine by adding sufficient statistics. The prior acts as pseudo-observations or prior counts. The posterior simply aggregates.

Three examples:

1. Bernoulli likelihood + Beta prior: prior successes and failures become pseudo-counts.
2. Poisson likelihood + Gamma prior: prior count rate and prior exposure update additively.
3. Categorical likelihood + Dirichlet prior: prior class counts combine with observed class counts.

Why conjugacy is useful:

- closed-form posterior updates
- closed-form posterior predictive distributions
- clear interpretation of shrinkage and pseudo-counts
- efficient online updating
- clean pedagogical bridge from abstract Bayes to practical inference

Why conjugacy is limited:

- the prior may be chosen for algebra rather than realism
- many modern ML models are nonconjugate
- convenient families can underrepresent heavy tails, multimodality, or structural dependence

Examples and non-examples:

- **Example:** Beta prior for Bernoulli probability is conjugate because the powers of $\theta$ and $1-\theta$ remain Beta-shaped after multiplication by the likelihood.
- **Example:** Dirichlet prior for categorical class probabilities is conjugate because the multinomial likelihood simply increments component exponents.
- **Non-example:** a Gaussian prior on a Bernoulli probability constrained to $[0,1]$ is not even a valid prior on the parameter space.
- **Non-example:** a mixture prior may be valid and expressive, but it may no longer give a same-family posterior.

**For AI:** Conjugate models are not relics. Beta-Binomial click models, Dirichlet-Categorical smoothing, Gamma-Poisson count models, and Gaussian-Gaussian shrinkage all remain active tools in online experimentation, recommendation, Bayesian bandits, probabilistic monitoring, and NLP smoothing.

### 3.2 Beta-Binomial Model

The Beta-Binomial model is the canonical Bayesian update.

Let $X_1, \ldots, X_n \overset{\text{i.i.d.}}{\sim} \operatorname{Bern}(\theta)$, and let $k = \sum_i X_i$. The likelihood is
$$
p(\mathcal{D} \mid \theta) \propto \theta^k (1-\theta)^{n-k}.
$$
Take the prior
$$
\theta \sim \operatorname{Beta}(\alpha, \beta),
\qquad
p(\theta) \propto \theta^{\alpha-1}(1-\theta)^{\beta-1}.
$$
Then the posterior is
$$
p(\theta \mid \mathcal{D}) \propto \theta^{\alpha+k-1}(1-\theta)^{\beta+n-k-1},
$$
so
$$
\theta \mid \mathcal{D} \sim \operatorname{Beta}(\alpha+k, \beta+n-k).
$$

This single formula teaches most of the basic Bayesian intuitions.

**Pseudo-count interpretation.** The prior behaves as if we had already seen $\alpha-1$ successes and $\beta-1$ failures in a soft sense. More operationally, $\alpha$ and $\beta$ set the prior mass and strength.

**Shrinkage.** The posterior mean is
$$
\mathbb{E}[\theta \mid \mathcal{D}] = \frac{\alpha+k}{\alpha+\beta+n}.
$$
It lies between the sample proportion $k/n$ and the prior mean $\alpha/(\alpha+\beta)$. Small $n$ means stronger shrinkage to the prior; large $n$ lets the data dominate.

**MAP.** When $\alpha+k>1$ and $\beta+n-k>1$,
$$
\theta_{\text{MAP}} = \frac{\alpha+k-1}{\alpha+\beta+n-2}.
$$
This differs from both the posterior mean and the MLE $k/n$ unless special parameter choices collapse them together.

Examples:

1. **Uniform prior:** $\operatorname{Beta}(1,1)$ gives posterior $\operatorname{Beta}(1+k,1+n-k)$.
2. **Jeffreys prior:** $\operatorname{Beta}(1/2,1/2)$ is invariant and avoids overcommitting near the boundaries.
3. **Strong prior around 0.5:** choosing $\alpha=\beta=50$ makes the posterior resistant to small-sample fluctuations.

Non-examples:

1. Interpreting $\alpha$ and $\beta$ as literal historical counts in every application. They are analogous to counts, not always actual counts.
2. Forgetting that the Beta family lives on $[0,1]$. It cannot be used directly for unconstrained parameters.

The posterior predictive for a new Bernoulli trial is
$$
P(X_{n+1}=1 \mid \mathcal{D}) = \mathbb{E}[\theta \mid \mathcal{D}] = \frac{\alpha+k}{\alpha+\beta+n}.
$$
This is already a glimpse of a general principle: Bayesian prediction integrates out parameter uncertainty instead of plugging in a single estimate.

Why this matters conceptually:

- The posterior is a distribution over the click rate, conversion rate, toxicity rate, or failure probability.
- The posterior predictive gives the next-event probability directly.
- Sequential updating is trivial: every new success increments one shape parameter; every new failure increments the other.

```text
ASCII POSTERIOR UPDATE FOR BERNOULLI DATA
======================================================================

  prior:      Beta(alpha, beta)
  data:       k successes, n-k failures
  posterior:  Beta(alpha + k, beta + n - k)

  intuition:
  successes push mass to the right
  failures  push mass to the left
  larger alpha + beta means stronger prior pull

======================================================================
```

**For AI:** This model appears in A/B testing, Thompson sampling, spam filtering, quality-control monitoring, and online safety pipelines where outcomes are binary and decisions must update in real time.

It is worth seeing the algebra once in slow motion. Starting from
$$
p(\theta \mid \mathcal{D})
\propto
p(\mathcal{D} \mid \theta)p(\theta)
\propto
\theta^k(1-\theta)^{n-k}\theta^{\alpha-1}(1-\theta)^{\beta-1},
$$
we gather exponents:
$$
p(\theta \mid \mathcal{D})
\propto
\theta^{\alpha+k-1}(1-\theta)^{\beta+n-k-1}.
$$
That kernel is exactly Beta. There is no approximation, no asymptotic step, and no optimization. Exact Bayes is simply pattern recognition in the algebra.

The posterior variance is
$$
\operatorname{Var}(\theta \mid \mathcal{D})
=
\frac{(\alpha+k)(\beta+n-k)}
{(\alpha+\beta+n)^2(\alpha+\beta+n+1)}.
$$
This formula gives the second major Bayesian intuition after shrinkage: uncertainty contracts as evidence accumulates. The posterior does not only move. It tightens.

Consider three standard examples.

**Example 1: cold-start recommender system.** A new item has 2 clicks from 3 impressions. The MLE click rate is $2/3$, which is wildly unstable. A prior such as $\operatorname{Beta}(5,45)$ reflects an ecosystem in which most items receive around 10% CTR. The posterior becomes far more conservative, which is usually what a ranking system wants under scarce data.

**Example 2: safety monitoring.** If a moderation pipeline sees 0 failures in 20 inspected examples, the MLE failure rate is 0. A Bayesian posterior with a weak prior never collapses to exactly 0, which is much more realistic for risk management.

**Example 3: adaptive experimentation.** Thompson sampling draws $\theta^{(s)}$ from the posterior and chooses the arm with the highest sampled reward rate. Beta-Binomial conjugacy makes that update-and-sample loop extremely cheap.

Two non-examples are just as important.

**Non-example 1: prior pretending to be data.** Saying "my prior is equivalent to exactly 100 historical observations" can be useful pedagogically, but in serious applications the prior may encode structure, soft beliefs, or constraints that are not literally reducible to old samples.

**Non-example 2: posterior mean as universal choice.** In severe asymmetric-loss settings, the posterior mean of a probability can be the wrong decision summary. If false negatives are much more costly, the Bayes-optimal action may require a threshold on posterior tail probability rather than the mean.

There is also a useful boundary case. If $\alpha,\beta \to 0$ formally, the prior becomes highly concentrated near the edges and can induce unstable or aggressive boundary behavior. This is a reminder that "uninformative" does not mean "harmless." Prior geometry matters.

### 3.3 Gamma-Poisson and Dirichlet-Categorical Models

The same additive logic extends beyond Bernoulli data.

#### Gamma-Poisson

Suppose $X_1, \ldots, X_n \overset{\text{i.i.d.}}{\sim} \operatorname{Poisson}(\lambda)$. The likelihood is
$$
p(\mathcal{D} \mid \lambda) \propto \lambda^{\sum_i x_i} e^{-n\lambda}.
$$
With prior
$$
\lambda \sim \operatorname{Gamma}(\alpha, \beta),
\qquad
p(\lambda) \propto \lambda^{\alpha-1}e^{-\beta \lambda},
$$
the posterior becomes
$$
\lambda \mid \mathcal{D} \sim \operatorname{Gamma}\left(\alpha+\sum_i x_i,\; \beta+n\right).
$$

Interpretation:

- $\alpha$ acts like prior event mass
- $\beta$ acts like prior exposure
- posterior mean is
$$
\mathbb{E}[\lambda \mid \mathcal{D}] = \frac{\alpha+\sum_i x_i}{\beta+n}
$$
which blends prior rate and observed average count

Applications include event-frequency modeling, request-rate monitoring, failure counts, and token-count processes in simplified probabilistic language settings.

#### Dirichlet-Categorical

Suppose a categorical variable takes one of $K$ values with parameter vector $\boldsymbol{\pi} = (\pi_1,\ldots,\pi_K)$ satisfying $\pi_k \ge 0$ and $\sum_k \pi_k = 1$. A Dirichlet prior is
$$
\boldsymbol{\pi} \sim \operatorname{Dir}(\alpha_1,\ldots,\alpha_K),
\qquad
p(\boldsymbol{\pi}) \propto \prod_{k=1}^K \pi_k^{\alpha_k-1}.
$$
If the observed counts are $n_1,\ldots,n_K$, then
$$
\boldsymbol{\pi} \mid \mathcal{D} \sim \operatorname{Dir}(\alpha_1+n_1,\ldots,\alpha_K+n_K).
$$

Again the update is just count addition.

Examples:

1. **Language-model smoothing preview:** a Dirichlet prior prevents zero-probability estimates for rarely observed categories.
2. **Class-probability inference:** posterior class probabilities remain defined even when some classes are absent in a small sample.
3. **Mixture-model component weights:** Dirichlet priors regularize simplex-valued parameters.

Non-examples:

1. Using independent Beta priors for a multinomial probability vector without enforcing the simplex constraint.
2. Treating Dirichlet concentration as only a technical parameter. It governs both mean and certainty.

The Dirichlet concentration sum
$$
\alpha_0 = \sum_{k=1}^K \alpha_k
$$
controls prior strength. Small $\alpha_0$ yields spiky priors; large $\alpha_0$ yields concentrated priors around the prior mean $\alpha_k/\alpha_0$.

**For AI:** Dirichlet and Gamma conjugacy remain useful in topic models, count data, categorical smoothing, bandit algorithms, and probabilistic routing systems.

### 3.4 Gaussian-Gaussian Updating

Gaussian updating is the continuous conjugate model that most clearly illustrates Bayesian shrinkage.

Assume
$$
X_i \mid \mu \overset{\text{i.i.d.}}{\sim} \mathcal{N}(\mu, \sigma^2),
\qquad
\mu \sim \mathcal{N}(\mu_0, \tau_0^2),
$$
with known observation variance $\sigma^2$ and prior variance $\tau_0^2$.

The likelihood contribution from the sample mean is
$$
\bar{X} \mid \mu \sim \mathcal{N}\left(\mu, \frac{\sigma^2}{n}\right).
$$
Combining prior and likelihood gives
$$
\mu \mid \mathcal{D} \sim \mathcal{N}(\mu_n, \tau_n^2)
$$
where
$$
\tau_n^2 = \left(\frac{1}{\tau_0^2} + \frac{n}{\sigma^2}\right)^{-1},
\qquad
\mu_n = \tau_n^2 \left(\frac{\mu_0}{\tau_0^2} + \frac{n\bar{x}}{\sigma^2}\right).
$$

Writing precision as inverse variance makes the structure even clearer:
$$
\text{posterior precision} = \text{prior precision} + \text{data precision}.
$$

The posterior mean is a precision-weighted average:
$$
\mu_n
=
\frac{\tau_0^{-2}}{\tau_0^{-2}+n\sigma^{-2}}\,\mu_0
+
\frac{n\sigma^{-2}}{\tau_0^{-2}+n\sigma^{-2}}\,\bar{x}.
$$

This formula deserves to be internalized. Bayesian learning is not "prior versus data." It is weighted evidence combination. More reliable information gets more weight.

Examples:

1. If $\tau_0^2$ is large, the prior is weak and $\mu_n \approx \bar{x}$.
2. If $\sigma^2$ is large, data are noisy and the posterior remains close to $\mu_0$.
3. If $n$ grows, the data precision overwhelms the prior precision, so the posterior concentrates near the MLE.

Non-examples:

1. Saying the prior "biases" the estimate in a pathological sense. In small samples it often stabilizes estimation rather than distorting it.
2. Assuming the posterior variance is the same as the sampling variance of $\bar{X}$. It is smaller whenever the prior carries information.

This model also gives an immediate bridge to ridge-style regularization and Kalman-style updating, though the latter belongs in the time-series section.

> **Forward reference:** The Kalman filter is repeated Gaussian Bayesian updating in a linear dynamical system. Its full treatment belongs in [Time Series](../05-Time-Series/notes.md), not here.

**For AI:** Gaussian shrinkage appears in Bayesian linear regression, Laplace approximations, Gaussian process intuition, and uncertainty-aware fine-tuning methods that treat parameter updates as noisy evidence.

The derivation also reveals something geometric. Starting from
$$
\log p(\mu \mid \mathcal{D})
=
-\frac{1}{2\sigma^2}\sum_{i=1}^n (x_i-\mu)^2
-\frac{1}{2\tau_0^2}(\mu-\mu_0)^2
+ C,
$$
expand the quadratic terms in $\mu$ and collect coefficients:
$$
\log p(\mu \mid \mathcal{D})
=
-\frac{1}{2}
\left(
\frac{n}{\sigma^2} + \frac{1}{\tau_0^2}
\right)\mu^2
+
\left(
\frac{n\bar{x}}{\sigma^2} + \frac{\mu_0}{\tau_0^2}
\right)\mu
+ C'.
$$
Completing the square gives the Gaussian posterior. The coefficient in front of $\mu^2$ is the posterior precision, and the linear coefficient determines the posterior center.

This derivation suggests a useful general heuristic:

- the likelihood contributes curvature proportional to data precision
- the prior contributes curvature proportional to prior precision
- the posterior combines them additively

That same heuristic reappears in Laplace approximations, natural-gradient intuition, Gaussian processes, Kalman filtering, and second-order uncertainty approximations in deep learning.

Three extended examples:

**Example 1: prior as trusted baseline.** Suppose a system has historical evidence that latency is near 80 ms, but the current rollout has only a few noisy measurements. The posterior mean stays near the historical value unless the new sample mean is both different and precise enough to overcome the prior.

**Example 2: prior-data conflict.** If the prior mean is far from the observed sample mean and the sample size is large, the posterior resolves the conflict decisively in favor of the data. Bayesian inference does not "force" prior belief forever. It merely weighs evidence.

**Example 3: one noisy observation.** With $n=1$ and large $\sigma^2$, the posterior mean barely moves. This is not stubbornness. It is rational resistance to noisy evidence.

Two non-examples:

**Non-example 1:** calling the posterior mean "biased toward the prior" as if any departure from $\bar{x}$ were automatically a flaw. In finite samples, this bias can lower total posterior risk dramatically.

**Non-example 2:** thinking that a diffuse prior means no modeling choice was made. Choosing to be diffuse is still a choice, and in weakly identified problems it can change the posterior materially.

### 3.5 What Conjugacy Buys and What It Misses

Conjugacy gives at least five major benefits:

1. exact posterior formulas
2. exact posterior predictive formulas
3. transparent prior-strength interpretation
4. fast sequential updates
5. easy pedagogical access to Bayesian logic

Those benefits make conjugate models indispensable for intuition, prototyping, online systems, and settings where the likelihood family already matches the application well.

But conjugacy also has limits:

- it encourages priors chosen for convenience rather than domain realism
- it can hide model mismatch
- it does not scale automatically to rich neural architectures
- it often excludes multimodal or structured posteriors

Three examples of what conjugacy misses:

1. **Deep neural posteriors:** distributions over millions of weights are generally nonconjugate.
2. **Complex latent-variable models:** exact integration is usually impossible.
3. **Structured priors:** low-rank, graph-based, or combinatorial priors often leave the conjugate world.

The right lesson is not "conjugate Bayes is simplistic." The right lesson is that exact Bayesian updating is easy only for certain model families, and those families provide the foundation from which approximation methods are built.

---

## 4. Core Theory II: Estimation, Prediction, and Uncertainty

### 4.1 MAP Estimation as Regularised MLE

Maximum a posteriori estimation chooses
$$
\hat{\theta}_{\text{MAP}} = \arg\max_\theta p(\theta \mid \mathcal{D})
= \arg\max_\theta \left[\log p(\mathcal{D} \mid \theta) + \log p(\theta)\right].
$$
Because the evidence does not depend on $\theta$, MAP is equivalent to maximizing log-likelihood plus log-prior.

This makes the relationship to regularization immediate.

#### Gaussian prior

If $\boldsymbol{\theta} \sim \mathcal{N}(\mathbf{0}, \lambda^{-1} I)$, then
$$
\log p(\boldsymbol{\theta}) = -\frac{\lambda}{2}\lVert \boldsymbol{\theta} \rVert_2^2 + C.
$$
So MAP becomes
$$
\hat{\boldsymbol{\theta}}_{\text{MAP}}
=
\arg\max_{\boldsymbol{\theta}}
\left[
\log p(\mathcal{D}\mid\boldsymbol{\theta})
- \frac{\lambda}{2}\lVert \boldsymbol{\theta} \rVert_2^2
\right].
$$
That is L2-regularized MLE, or ridge-style shrinkage.

#### Laplace prior

If the components are independent Laplace:
$$
p(\theta_j) \propto \exp(-\lambda |\theta_j|),
$$
then
$$
\log p(\boldsymbol{\theta}) = -\lambda \lVert \boldsymbol{\theta} \rVert_1 + C,
$$
so MAP becomes L1-regularized MLE.

Examples:

1. ridge regression = Gaussian-prior MAP
2. lasso = Laplace-prior MAP
3. weight decay in neural nets = Gaussian-prior interpretation of penalized optimization

Non-examples:

1. MAP is not full Bayesian inference. It collapses the posterior to one point.
2. A regularizer is not automatically Bayesian unless it can be interpreted as a log-prior under a probabilistic model.

This section should connect back to [Estimation Theory](../02-Estimation-Theory/notes.md#54-mle-as-cross-entropy-minimisation), where MLE was treated as the frequentist workhorse. MAP is the bridge between frequentist optimization and Bayesian inference: the moment we add a prior penalty to the likelihood objective, we have moved into Bayesian territory, even if we stop short of integrating over the posterior.

**For AI:** Weight decay in large language model training is often discussed as a purely optimization-side trick. The Bayesian interpretation is cleaner: it encodes prior preference for smaller parameter norms, which in turn stabilizes generalization and reduces overfitting.

### 4.2 Posterior Covariance and Shrinkage

Point estimates answer "where is the posterior centered?" Posterior covariance answers "how uncertain are we, and in which directions?"

For a vector parameter $\boldsymbol{\theta}$, the posterior covariance is
$$
\operatorname{Cov}(\boldsymbol{\theta} \mid \mathcal{D})
=
\mathbb{E}\left[
(\boldsymbol{\theta} - \mathbb{E}[\boldsymbol{\theta}\mid\mathcal{D}])
(\boldsymbol{\theta} - \mathbb{E}[\boldsymbol{\theta}\mid\mathcal{D}])^\top
\middle|\mathcal{D}
\right].
$$

This matrix tells us:

- which directions are well identified
- which directions remain uncertain
- how parameter uncertainties co-vary
- how much predictions should vary when parameter uncertainty is propagated

Shrinkage is the visible effect of posterior covariance interacting with the prior. In small samples, the posterior often contracts toward structured prior beliefs, reducing variance at the price of some bias. This tradeoff is often beneficial.

Examples:

1. In Gaussian-Gaussian updating, posterior variance is smaller than both prior variance and raw sampling variance once both information sources are combined.
2. In Bayesian linear regression, highly collinear features produce posterior covariance directions with large uncertainty.
3. In hierarchical models, weak groups borrow information from the population prior, dramatically reducing variance.

Non-examples:

1. Shrinkage is not the same as underfitting. It is controlled bias used to reduce variance.
2. Posterior covariance is not simply the Hessian inverse unless a Gaussian approximation is justified.

Posterior covariance also clarifies why Bayesian inference is attractive for unstable estimation problems. If the likelihood is flat in some direction, a frequentist optimizer may still return one arbitrary point. A posterior keeps that uncertainty visible.

**For AI:** In Bayesian deep learning, uncertainty in weights is usually too high-dimensional to represent exactly, so practical methods approximate only a low-rank or diagonal covariance. Methods like SWAG make this tradeoff explicit.

### 4.3 Posterior Predictive Distribution

The posterior predictive distribution is the Bayesian answer to prediction:
$$
p(x_{\text{new}} \mid \mathcal{D})
=
\int p(x_{\text{new}} \mid \theta)\, p(\theta \mid \mathcal{D})\, d\theta.
$$

This is the most important formula in applied Bayesian inference after Bayes' rule itself.

Why? Because it integrates parameter uncertainty into prediction. Plug-in prediction replaces $\theta$ by a point estimate such as MLE or MAP:
$$
p(x_{\text{new}} \mid \hat{\theta}).
$$
Bayesian prediction averages over all plausible parameter values under the posterior. That averaging is often more stable and better calibrated.

Examples:

1. **Beta-Binomial:** the predictive probability of success equals the posterior mean of the success probability.
2. **Gamma-Poisson:** integrating out the Poisson rate yields an overdispersed predictive distribution, more realistic than a pure plug-in Poisson.
3. **Bayesian linear regression:** predictive variance contains both observation noise and parameter uncertainty.

Non-examples:

1. Reporting only a posterior mean parameter and then acting as if uncertainty vanished.
2. Confusing the posterior over parameters with the predictive distribution over future observations.

The distinction matters. A posterior can be concentrated while predictions remain noisy because the observation model itself is noisy. Conversely, predictions can be sharp even when parameter uncertainty is moderate if the downstream quantity is insensitive to the uncertain directions.

**For AI:** Posterior predictive distributions are directly relevant to calibrated confidence, abstention, active learning, and synthetic-data generation under uncertainty.

The posterior predictive can be written in two-stage form:
$$
p(x_{\text{new}} \mid \mathcal{D})
=
\int p(x_{\text{new}} \mid \theta)\, p(\theta \mid \mathcal{D})\, d\theta.
$$
This says:

1. sample a plausible parameter from what the data allow
2. simulate or score the new observation under that parameter
3. average over all such plausible parameters

That averaging is often called **Bayesian model averaging** at the parameter level. It explains why posterior predictive distributions can be more conservative and better calibrated than point-estimate predictions. A point estimate pretends one parameter is true. The posterior predictive averages over uncertainty honestly.

Three useful examples:

**Example 1: Beta-Binomial smoothing.** If one observes 1 success in 1 trial, the MLE predictive probability of success is 1. The posterior predictive with a moderate prior is far lower, avoiding catastrophic overconfidence from tiny samples.

**Example 2: Gamma-Poisson overdispersion.** Plugging in the posterior mean rate yields a Poisson predictive variance equal to the mean. Integrating over posterior rate uncertainty yields larger variance, which is often more realistic for real count data.

**Example 3: Bayesian linear regression.** Predictions far from the observed design points have larger uncertainty because $\mathbf{x}_*^\top \Sigma_n \mathbf{x}_*$ grows in poorly observed directions. This is exactly the kind of uncertainty a safe system should expose.

Two non-examples:

**Non-example 1:** "posterior predictive uncertainty is just aleatoric noise." No. It also includes epistemic uncertainty from the posterior over parameters.

**Non-example 2:** "if the posterior is concentrated, predictive uncertainty must be small." Not necessarily. If the data model itself is noisy, predictive uncertainty can remain large even with very certain parameters.

This distinction between parameter uncertainty and observation uncertainty is crucial in AI safety and active learning. Epistemic uncertainty can often be reduced by collecting more data; aleatoric uncertainty often cannot.

### 4.4 Posterior Predictive Checks

A model can fit the data badly even if posterior computation is exact. Bayesian inference does not rescue a misspecified model. That is why posterior predictive checks (PPCs) matter.

The idea is simple:

1. Sample $\theta^{(s)} \sim p(\theta \mid \mathcal{D})$.
2. Simulate replicated data $\tilde{\mathcal{D}}^{(s)} \sim p(\mathcal{D} \mid \theta^{(s)})$.
3. Compare statistics of $\tilde{\mathcal{D}}^{(s)}$ to those of the observed dataset.

If the model is adequate, the observed data should look typical under these replications. If the observed data sit in the extreme tail of replicated summaries, the model is missing important structure.

Examples of discrepancy statistics:

- mean or variance
- tail behavior
- number of zero counts
- class imbalance
- calibration error
- residual autocorrelation

Examples:

1. A Poisson model may underfit overdispersed counts; PPCs reveal replicated variance far below observed variance.
2. A Gaussian observation model may miss heavy tails; PPCs reveal too few extreme values in the replicated data.
3. A Bayesian classifier may look good on average accuracy but fail PPC-style calibration diagnostics.

Non-examples:

1. Using PPCs as formal proof the model is true. They are diagnostics, not proofs.
2. Comparing only one statistic and declaring the whole model validated.

**For AI:** In probabilistic ML systems, PPCs are a natural way to ask whether the learned generative or predictive model captures the patterns that matter operationally. They are especially useful for drift monitoring and calibration sanity checks.

PPCs are especially valuable because they focus attention on the observable world rather than on latent mathematical elegance. A posterior may look concentrated and well behaved while still generating unrealistic data. The real question is not "did the model optimize cleanly?" but "if I sampled from this model after seeing the data, would the synthetic worlds resemble the one I am actually trying to understand?"

Three examples:

**Example 1:** A language-quality model might match average toxicity rates but fail badly on tail-risk prompts. A PPC based on extreme-quantile toxicity scores can reveal the gap.

**Example 2:** A recommendation model may match mean click rate yet fail to reproduce the observed heterogeneity across users. A PPC on user-level variance catches what global accuracy misses.

**Example 3:** A forecasting model may predict the correct marginal variance but miss autocorrelation structure. A PPC based on lagged residual statistics can reveal the defect.

### 4.5 Bayesian Linear Regression as a Worked Example

Bayesian linear regression is a complete worked example that unifies prior design, posterior inference, shrinkage, and posterior prediction without becoming a full regression chapter.

Assume
$$
\mathbf{y} = X\boldsymbol{\beta} + \boldsymbol{\varepsilon},
\qquad
\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 I_n),
$$
with prior
$$
\boldsymbol{\beta} \sim \mathcal{N}(\mathbf{0}, \tau^2 I_d).
$$

The posterior is Gaussian:
$$
\boldsymbol{\beta} \mid \mathcal{D}
\sim
\mathcal{N}(\boldsymbol{\mu}_n, \Sigma_n)
$$
where
$$
\Sigma_n = \left(\frac{1}{\sigma^2}X^\top X + \frac{1}{\tau^2}I_d\right)^{-1},
\qquad
\boldsymbol{\mu}_n = \Sigma_n \frac{1}{\sigma^2}X^\top \mathbf{y}.
$$

This formula contains several important ideas at once.

**MAP equals ridge.** The posterior mode is the ridge estimator:
$$
\hat{\boldsymbol{\beta}}_{\text{MAP}}
=
\arg\min_{\boldsymbol{\beta}}
\left[
\frac{1}{2\sigma^2}\lVert \mathbf{y} - X\boldsymbol{\beta} \rVert_2^2
+
\frac{1}{2\tau^2}\lVert \boldsymbol{\beta} \rVert_2^2
\right].
$$

**Posterior covariance measures identifiability.** If columns of $X$ are nearly collinear, $X^\top X$ is ill-conditioned and posterior uncertainty grows along unstable directions.

**Posterior predictive is Gaussian.** For a new feature vector $\mathbf{x}_*$,
$$
y_* \mid \mathbf{x}_*, \mathcal{D}
\sim
\mathcal{N}\left(
\mathbf{x}_*^\top \boldsymbol{\mu}_n,\;
\sigma^2 + \mathbf{x}_*^\top \Sigma_n \mathbf{x}_*
\right).
$$
The first term in the variance is observation noise. The second term is parameter uncertainty.

Examples:

1. With many data and moderate noise, $\Sigma_n$ shrinks and predictions approach plug-in regression.
2. With little data, predictions widen far from the training design.
3. With strong prior shrinkage (small $\tau^2$), coefficients stay near zero unless strongly supported by the data.

Non-examples:

1. Calling Bayesian linear regression "just ridge." Ridge corresponds only to the MAP point, not to the full posterior or predictive distribution.
2. Thinking the predictive variance is just the residual variance $\sigma^2$. It also contains uncertainty from estimating $\boldsymbol{\beta}$.

This example is the right stopping point for this section. We use regression as a vehicle for Bayesian ideas, not as a substitute for the full [Regression Analysis](../06-Regression-Analysis/notes.md) section.

---

## 5. Core Theory III: Model Comparison and Hierarchical Bayes

### 5.1 Marginal Likelihood and Occam's Razor

The marginal likelihood
$$
p(\mathcal{D} \mid \mathcal{M}) = \int p(\mathcal{D} \mid \theta, \mathcal{M})\, p(\theta \mid \mathcal{M})\, d\theta
$$
scores an entire model class $\mathcal{M}$ by averaging the likelihood over its parameter space under the prior.

This is conceptually different from maximum likelihood. Maximum likelihood asks: is there one parameter setting in this model that fits the data well? Marginal likelihood asks: does the model place substantial prior mass on parameter settings that fit the data well? A flexible model with huge parameter space can achieve a large maximum likelihood while still receiving a poor marginal likelihood if most of its parameter space fits badly.

This is the Bayesian version of Occam's razor. Complex models are not penalized by an arbitrary extra term. They are penalized automatically because averaging over a larger parameter space dilutes probability mass unless the complexity is truly supported by the data.

Examples:

1. A model with many irrelevant parameters can overfit pointwise but still have weak marginal likelihood.
2. A simpler model with slightly lower peak likelihood may win because it concentrates probability on useful regions.
3. In latent-variable models, richer priors can improve marginal likelihood when they align with real structure rather than adding free flexibility blindly.

Non-examples:

1. Marginal likelihood is not "the likelihood at the posterior mean."
2. Occam's razor here is not a hand-tuned penalty; it is a consequence of integration.

**For AI:** Bayesian model evidence gives a principled language for deciding whether extra complexity in a probabilistic model is truly supported, rather than rewarding any architecture that can fit the training data harder.

### 5.2 Bayes Factors and Bayesian Model Comparison

If two models or hypotheses $\mathcal{M}_1$ and $\mathcal{M}_0$ are being compared, the **Bayes factor** is
$$
B_{10}
=
\frac{p(\mathcal{D} \mid \mathcal{M}_1)}{p(\mathcal{D} \mid \mathcal{M}_0)}.
$$

It updates prior model odds into posterior model odds:
$$
\frac{P(\mathcal{M}_1 \mid \mathcal{D})}{P(\mathcal{M}_0 \mid \mathcal{D})}
=
B_{10}
\cdot
\frac{P(\mathcal{M}_1)}{P(\mathcal{M}_0)}.
$$

This is the Bayesian analog of hypothesis comparison, but it differs fundamentally from p-values and likelihood-ratio tests.

- A p-value asks how surprising the data are under the null.
- A likelihood ratio compares best-fit parameter settings.
- A Bayes factor compares integrated support across the full parameter spaces.

Examples:

1. Comparing a null model to a nonzero-effect model in online experimentation.
2. Comparing a sparse prior to a dense prior when choosing feature structure.
3. Comparing probabilistic sequence models under different smoothing assumptions.

Non-examples:

1. Treating a Bayes factor as a posterior probability without including prior model odds.
2. Treating Bayes factors as numerically interchangeable with p-values.

> **Backward reference:** [Hypothesis Testing](../03-Hypothesis-Testing/notes.md#65-bayesian-alternative-preview) gave only a brief preview of Bayes factors. This section is their canonical home.

**For AI:** In benchmark comparison and model selection, Bayes factors offer an evidence-centric alternative to thresholded significance testing, especially when one wants direct probability updates on competing models.

There is a subtle but important reason Bayes factors can behave very differently from p-values. Under a diffuse alternative prior, the model spreads probability mass over many parameter values that fit the data poorly. Even if one subset of those values fits well, the integrated evidence can remain modest because the average fit under the alternative is diluted. This is sometimes called the **Occam penalty** in action.

That means Bayes factors are sensitive not only to the data but also to the geometry of the alternative model. A large model with a vague prior can be punished more strongly than intuition expects. This is not a bug. It is the mechanism by which Bayesian evidence rewards models that made sharp, successful predictions rather than merely permitting many possibilities.

Three instructive examples:

**Example 1: point null vs broad alternative.** Suppose the null says $\theta=0$ and the alternative says $\theta \sim \mathcal{N}(0, \tau^2)$ with large $\tau^2$. If the observed effect is small but nonzero, the Bayes factor can still favor the null because most of the broad alternative prior mass predicted larger effects than the data delivered.

**Example 2: nested regression models.** A larger regression model may improve the fitted likelihood, yet still lose on marginal likelihood if the extra coefficients are poorly identified and the prior spreads mass widely over them.

**Example 3: model family comparison in sequence modeling.** A more expressive probabilistic decoder can assign very high likelihood to some settings but still lose evidence if the prior support is too diffuse relative to the data size.

Two non-examples:

**Non-example 1:** "Bayes factor close to 1 means the models are equally true." No. It means the observed data do not strongly update the prior odds between the candidate models.

**Non-example 2:** "A strong Bayes factor removes dependence on the prior." The Bayes factor already incorporates the prior over model parameters. Strong evidence can dominate weak prior odds, but it does not erase prior specification.

Practical model comparison therefore often includes:

- clear prior specification on each model
- sensitivity analysis over prior scales
- reporting posterior odds only after stating prior model odds
- avoiding the temptation to compare Bayes factors and p-values as if they answered the same question

For decision-making, this matters because Bayesian comparison asks not only "can this model fit?" but "did this model place meaningful prior probability on what actually happened?" That is a much stronger and often more useful standard.

### 5.3 Empirical Bayes

Empirical Bayes sits between full Bayes and plug-in frequentist estimation. The idea is:

1. posit a prior family $p(\theta \mid \lambda)$ with hyperparameter $\lambda$
2. estimate $\lambda$ from the data, often by maximizing marginal likelihood
3. condition on the estimated $\hat{\lambda}$ and proceed as if it were fixed

Formally,
$$
\hat{\lambda}
=
\arg\max_\lambda p(\mathcal{D} \mid \lambda)
=
\arg\max_\lambda \int p(\mathcal{D} \mid \theta)\, p(\theta \mid \lambda)\, d\theta.
$$

This approach is attractive because it preserves shrinkage and pooled-information benefits without requiring a fully Bayesian posterior over hyperparameters.

Examples:

1. estimating a global variance scale for many related coefficients
2. setting smoothing strength in large collections of sparse count models
3. fitting prior variance in Bayesian linear models from observed tasks

Non-examples:

1. Calling empirical Bayes "fully Bayesian." It is not, because hyperparameters are plugged in rather than integrated out.
2. Thinking empirical Bayes is automatically ad hoc. In high-dimensional settings it can be extremely effective and principled.

**For AI:** Empirical Bayes ideas appear whenever a system learns regularization strength, task-sharing scales, or uncertainty hyperparameters from data instead of fixing them by hand.

Empirical Bayes is often misunderstood because it occupies an uncomfortable middle ground. Purist Bayesians object that uncertainty in $\lambda$ should be integrated out, not plugged in. Pure frequentists may object that the prior is being estimated from the same data it is supposed to regularize. In practice, the method works well in many high-dimensional settings because it shares information efficiently while remaining computationally tractable.

Three examples:

**Example 1: many related effect sizes.** If thousands of sparse task-level coefficients are believed to come from a common Gaussian prior, estimating the prior variance from all tasks can dramatically improve shrinkage relative to tuning each task independently.

**Example 2: empirical Bayes smoothing in count models.** Large collections of low-count rates benefit from a shared prior estimated from the full population, then used to stabilize each local estimate.

**Example 3: layerwise uncertainty scales.** In approximate Bayesian deep learning, one may fit prior or posterior scale hyperparameters from data rather than fixing one global number.

Two non-examples:

**Non-example 1:** "Empirical Bayes uses the data twice, so it is invalid." It uses the data in a coupled estimation problem. Whether the approximation is acceptable depends on the inferential goal and uncertainty requirements, not on a slogan.

**Non-example 2:** "Empirical Bayes removes all need for priors." It still requires a prior family; only the hyperparameters are estimated.

### 5.4 Hierarchical Models and Partial Pooling

Hierarchical Bayes is one of the most practically important Bayesian ideas.

Suppose we estimate task-specific parameters $\theta_1, \ldots, \theta_G$ for $G$ related groups. A non-hierarchical approach either:

- fits each group independently (**no pooling**), or
- forces all groups to share one parameter (**complete pooling**).

Hierarchical Bayes introduces a population distribution:
$$
\theta_g \mid \mu, \tau^2 \sim \mathcal{N}(\mu, \tau^2),
\qquad
g = 1,\ldots,G.
$$
Now each group has its own parameter, but the parameters are coupled through shared hyperparameters.

This creates **partial pooling**:

- data-rich groups are estimated mostly from their own data
- data-poor groups are shrunk toward the population mean
- uncertainty is propagated at both group and population levels

Examples:

1. user-specific click models in recommendation systems
2. per-language or per-region metrics in multilingual products
3. patient- or site-level effects in medical ML

Non-examples:

1. Averaging all groups together and calling it hierarchical. That is complete pooling.
2. Fitting all groups independently and then post-hoc averaging. That misses shared uncertainty structure.

```text
ASCII PARTIAL POOLING
======================================================================

  no pooling:        each group learns alone
  complete pooling:  all groups forced to same value
  partial pooling:   groups share information through a population prior

  weak groups  --->  pulled strongly toward population mean
  strong groups ---> remain close to their own data signal

======================================================================
```

Hierarchical Bayes often outperforms both extremes because it respects heterogeneity while reducing variance.

**For AI:** Partial pooling is valuable whenever many related tasks, users, prompts, or environments share structure but not identity. It is a natural language for transfer, personalization, and multi-task uncertainty.

The hidden principle behind hierarchical Bayes is **exchangeability**. Before seeing group identities in detail, we often judge the groups to be similar enough that their parameters should be modeled as draws from a common population. Exchangeability is weaker than identical equality and stronger than complete independence. It says the ordering of groups should not matter to the prior.

This matters because many AI problems have exactly this form:

- users are different but not unrelated
- prompts are different but drawn from a broader task family
- regional models differ but share common infrastructure
- evaluation suites contain related but nonidentical sub-benchmarks

Hierarchical models let the data decide how much pooling is appropriate. If the estimated population variance $\tau^2$ is small, groups are strongly tied together. If $\tau^2$ is large, the model relaxes toward no pooling.

Three extended examples:

**Example 1: multilingual toxicity filtering.** High-resource languages provide abundant labels, low-resource languages provide few. A hierarchical prior over language-specific parameters lets low-resource languages borrow strength without assuming all languages behave identically.

**Example 2: hospital-level outcome models.** Some hospitals contribute many observations, others very few. Partial pooling prevents extreme estimates for small hospitals while preserving true large-hospital differences.

**Example 3: prompt family evaluation.** If one estimates failure rates across prompt categories, hierarchical Bayes prevents sparse categories from producing overconfident extremes.

Two non-examples:

**Non-example 1:** using complete pooling because it "improves stability." Stability bought by erasing genuine heterogeneity is often misleading.

**Non-example 2:** using no pooling because it "avoids assumptions." Independence across groups is itself a strong assumption and is often worse.

The characteristic output of a hierarchical model is not just one estimate per group. It is a joint posterior over all groups and population-level hyperparameters. That joint uncertainty is what makes principled partial pooling possible.

### 5.5 Prior Sensitivity and Robustness

Bayesian inference is only as trustworthy as the model-prior pair. Sensitivity analysis asks: how much do posterior conclusions change under reasonable prior alternatives?

This is especially important when:

- data are scarce
- the likelihood is weakly informative
- model comparison is sensitive to the prior scale
- posterior tails drive decisions

Examples of sensitivity questions:

1. does a launch decision change if the prior on effect size widens modestly?
2. does a Bayes factor reverse if the alternative prior is made too diffuse?
3. do posterior tail probabilities for safety risk stay stable across plausible priors?

Non-examples:

1. Reporting one prior and pretending prior choice is irrelevant.
2. Declaring the posterior "objective" because the prior was weakly informative.

Robust Bayesian practice often includes:

- prior predictive checks
- multiple plausible prior specifications
- reporting how posterior summaries move with prior strength
- avoiding excessively diffuse priors that create unstable computation or misleading evidence calculations

**For AI:** Prior sensitivity matters in safety-related claims, low-data evaluations, and model-comparison problems where apparently strong evidence can collapse under a slightly different prior scale.

---

## 6. Advanced Topics: Approximate Inference

### 6.1 Why Exact Posteriors Become Intractable

Exact posterior formulas rely on one of two favorable situations:

1. conjugacy gives analytic normalization and closed-form updates
2. the parameter space is small enough for direct numerical integration

Modern ML problems usually satisfy neither. Deep nets have millions or billions of parameters, latent-variable models require integration over large hidden spaces, and structured priors introduce dependencies that destroy conjugacy.

The generic problem is:
$$
p(\theta \mid \mathcal{D}) \propto p(\mathcal{D} \mid \theta)p(\theta)
$$
is easy to write down but hard to normalize, summarize, sample from, or integrate against.

This creates four recurring computational goals:

1. approximate the posterior itself
2. approximate posterior expectations
3. approximate the evidence
4. approximate the posterior predictive

Approximate inference methods trade off bias, variance, scalability, and calibration. There is no universal winner.

### 6.2 MCMC for Posterior Sampling

Markov chain Monte Carlo constructs a Markov chain whose stationary distribution is the posterior. Once the chain has mixed, samples from the chain can approximate posterior expectations:
$$
\mathbb{E}[f(\theta) \mid \mathcal{D}]
\approx
\frac{1}{S}\sum_{s=1}^S f(\theta^{(s)}).
$$

Important families include:

- Metropolis-Hastings
- Gibbs sampling
- Hamiltonian Monte Carlo

We do not re-derive Markov-chain theory here; that belongs to [Markov Chains](../../06-Probability-Theory/07-Markov-Chains/notes.md). Here the emphasis is inferential use.

Examples:

1. Metropolis-Hastings proposes a candidate and accepts it with a ratio preserving the posterior as invariant.
2. Gibbs sampling cycles through tractable conditional distributions.
3. HMC uses gradients and auxiliary momentum to explore high-dimensional posteriors more efficiently.

Non-examples:

1. A sampler that returns one optimized point is not MCMC.
2. A chain that has not mixed does not justify posterior summaries as if they were exact.

Strengths:

- asymptotically exact under appropriate conditions
- flexible across complex posterior shapes
- natural uncertainty propagation

Weaknesses:

- expensive in high dimensions
- convergence diagnostics are nontrivial
- often hard to scale to modern deep learning without approximation

**For AI:** MCMC remains essential in Bayesian modeling and scientific ML, but for very large neural networks it is often replaced by approximate or local methods because full posterior sampling is too costly.

Because MCMC is approximate in finite compute, diagnostics matter.

Important practical diagnostics include:

- trace plots for qualitative mixing
- effective sample size (ESS)
- split-$\hat{R}$ for cross-chain agreement
- autocorrelation decay
- divergence warnings in HMC-like methods

These diagnostics do not prove correctness, but they can reveal obvious failure. A badly mixed chain with high autocorrelation may produce deceptively stable but wrong posterior summaries.

Three examples:

**Example 1: random-walk Metropolis in high dimension.** Acceptance rates can collapse because local proposals do not move efficiently through the posterior.

**Example 2: Gibbs in strongly correlated posteriors.** Each conditional update is easy, but the chain moves slowly across the joint geometry.

**Example 3: HMC in smooth posteriors.** Gradient-guided exploration often gives better mixing and lower autocorrelation than naive local proposals.

Two non-examples:

**Non-example 1:** "the chain ran for many iterations, so it must have converged." Long runtime alone proves nothing.

**Non-example 2:** "MCMC gives exact samples." Only the stationary limit is exact under assumptions. Finite-run practice is always approximate.

There is also a conceptual distinction between optimization and sampling. Optimization seeks one good point. Sampling seeks representative coverage of the whole posterior mass. Mixing time, not just objective decrease, determines success.

### 6.3 Variational Inference and the ELBO

Variational inference replaces integration with optimization. Choose a tractable family $q_\phi(\theta)$ and fit it to the posterior by minimizing KL divergence:
$$
q_\phi^*
=
\arg\min_{q_\phi \in \mathcal{Q}}
D_{\mathrm{KL}}(q_\phi(\theta)\,\|\, p(\theta \mid \mathcal{D})).
$$

Because the true posterior contains the intractable evidence, we optimize the evidence lower bound (ELBO):
$$
\log p(\mathcal{D})
=
\mathcal{L}(q_\phi)
+
D_{\mathrm{KL}}(q_\phi(\theta)\,\|\, p(\theta \mid \mathcal{D})),
$$
where
$$
\mathcal{L}(q_\phi)
=
\mathbb{E}_{q_\phi}[\log p(\mathcal{D},\theta)]
-
\mathbb{E}_{q_\phi}[\log q_\phi(\theta)].
$$

Since KL divergence is nonnegative, maximizing the ELBO minimizes the KL gap to the posterior within the chosen family.

Examples:

1. mean-field VI assumes factorization and gains scalability
2. structured VI adds correlations for better fidelity
3. stochastic VI uses minibatches for large datasets

Non-examples:

1. VI is not exact Bayes unless the true posterior lies in the chosen variational family.
2. Mean-field VI is not "uncertainty solved." It often underestimates posterior variance because minimizing $D_{\mathrm{KL}}(q \| p)$ tends to favor mode-seeking approximations.

**For AI:** VI is central to VAEs, amortized latent inference, approximate Bayesian neural networks, and many large-scale probabilistic models where sampling is too expensive.

The direction of KL divergence matters. Standard VI often minimizes
$$
D_{\mathrm{KL}}(q \| p),
$$
not
$$
D_{\mathrm{KL}}(p \| q).
$$
These are not symmetric. Minimizing $D_{\mathrm{KL}}(q \| p)$ heavily penalizes placing mass where the true posterior has little support, but it is more tolerant of missing some posterior mass. As a result, mean-field VI often becomes **mode-seeking** and underestimates posterior variance.

This explains several common empirical observations:

- VI can produce sharp but overconfident approximations
- multimodal posteriors may be represented by one dominant mode
- diagonal Gaussian approximations miss posterior correlations

Examples:

**Example 1: bimodal posterior.** A mean-field Gaussian may center on one mode and largely ignore the other.

**Example 2: funnel geometry.** A simple Gaussian variational family can misrepresent heavy curvature and tail structure badly.

**Example 3: latent-variable autoencoders.** A diagonal encoder posterior is computationally efficient, but it necessarily suppresses some dependence structure among latent coordinates.

Two non-examples:

**Non-example 1:** "ELBO improvement means posterior quality is good." A better ELBO is good within the variational family, but a limited family can still miss important structure.

**Non-example 2:** "VI is just a faster version of MCMC." It solves a different approximation problem and produces different error modes.

There are several standard responses:

- richer variational families
- normalizing flows
- low-rank covariance structure
- importance-weighted objectives
- hybrid methods mixing variational warm starts with sampling refinements

The correct mental model is not "VI approximates Bayes perfectly if tuned carefully." It is "VI trades posterior fidelity for scalable optimization, and the choice of variational family is the central design decision."

### 6.4 Amortized Inference and VAEs

In many latent-variable models, each datapoint $\mathbf{x}^{(i)}$ has a latent variable $\mathbf{z}^{(i)}$. Running a full optimization to approximate each posterior $p(\mathbf{z}^{(i)} \mid \mathbf{x}^{(i)})$ separately is too expensive. Amortized inference solves this by learning a shared inference network:
$$
q_\phi(\mathbf{z} \mid \mathbf{x}).
$$

The variational autoencoder is the canonical example. It defines:

- prior $p(\mathbf{z})$
- decoder $p_\theta(\mathbf{x} \mid \mathbf{z})$
- encoder $q_\phi(\mathbf{z} \mid \mathbf{x})$

and optimizes
$$
\mathbb{E}_{q_\phi(\mathbf{z}\mid\mathbf{x})}
[\log p_\theta(\mathbf{x}\mid\mathbf{z})]
-
D_{\mathrm{KL}}(q_\phi(\mathbf{z}\mid\mathbf{x}) \| p(\mathbf{z})).
$$

The first term is reconstruction quality. The second term keeps the approximate posterior close to the prior and prevents arbitrary memorization of latent codes.

Examples:

1. image VAEs with Gaussian latent spaces
2. text latent-variable models with approximate posterior encoders
3. structured probabilistic models with learned local posterior approximations

Non-examples:

1. A deterministic autoencoder is not a VAE.
2. The encoder output is not the true posterior unless the approximation family and optimization happen to recover it exactly.

**For AI:** Amortized inference is one of the main places where Bayesian ideas became fully integrated with neural networks and modern autodiff tooling.

### 6.5 Laplace, MC Dropout, SGLD, and SWAG

Exact posterior inference for large neural networks is generally unavailable, so practical Bayesian deep learning uses approximations.

**Laplace approximation.** Approximate the posterior locally around a mode by a Gaussian:
$$
p(\boldsymbol{\theta} \mid \mathcal{D})
\approx
\mathcal{N}(\hat{\boldsymbol{\theta}}_{\text{MAP}}, H^{-1}),
$$
where $H$ is the Hessian of the negative log-posterior at the mode. This is local and can miss multimodality, but it is computationally attractive.

**MC dropout.** Keep dropout active at test time and average predictions across stochastic forward passes. Following Gal and Ghahramani, this can be interpreted as approximate variational Bayesian inference in a restricted family.

**SGLD.** Add Gaussian noise to stochastic-gradient updates so the iterates approximate posterior samples rather than collapsing to one optimizer. This ties posterior sampling to large-scale optimization.

**SWAG.** Fit a low-rank-plus-diagonal Gaussian approximation to SGD iterates near a wide optimum, then sample weights from that approximation for Bayesian model averaging.

Examples:

1. Laplace for local curvature-based uncertainty
2. MC dropout for cheap predictive uncertainty
3. SGLD for sampling-style uncertainty in scalable settings
4. SWAG for practical posterior approximations from standard training trajectories

Non-examples:

1. None of these methods is universally exact.
2. Good calibration from one benchmark does not prove full posterior fidelity.

**For AI:** These methods matter because exact Bayesian neural networks remain expensive. Approximate posterior methods are often used as uncertainty baselines, calibration tools, or practical approximations in safety-sensitive or data-limited applications.

It helps to compare the methods directly.

| Method | Main idea | Strength | Main weakness |
| --- | --- | --- | --- |
| Laplace | Gaussian around MAP mode | Cheap local uncertainty | Misses multimodality and nonlocal geometry |
| MC dropout | Stochastic subnet averaging | Easy to bolt onto existing models | Approximation quality depends strongly on architecture and interpretation |
| SGLD | SGD plus injected noise | Sampling flavor at scale | Sensitive to step-size schedules and mixing quality |
| SWAG | Gaussian fit to SGD iterates | Strong practical calibration baseline | Approximate posterior geometry only near a training trajectory |

Three examples of when each tends to shine:

**Example 1: Laplace** is attractive when one already has a trained MAP network and wants a quick curvature-based uncertainty estimate near the optimum.

**Example 2: MC dropout** is attractive when architecture and code already use dropout, and a cheap predictive-uncertainty baseline is needed.

**Example 3: SWAG** is attractive when one can afford collecting SGD iterates and wants a better uncertainty baseline than plain deterministic softmax confidence.

Two non-examples:

**Non-example 1:** using any of these methods and then assuming posterior probabilities are perfectly calibrated under severe distribution shift.

**Non-example 2:** comparing these methods only by test accuracy. Their purpose is uncertainty, calibration, and risk-sensitive prediction, not just point performance.

In Bayesian deep learning, the decisive question is often not "which method is most Bayesian?" but "which approximation gives the most reliable uncertainty at acceptable compute cost for this deployment setting?"

---

## 7. Applications in Machine Learning

### 7.1 Naive Bayes as Generative Classification

Naive Bayes uses Bayes' rule for classification:
$$
P(y=k \mid \mathbf{x})
\propto
p(\mathbf{x} \mid y=k)\, P(y=k).
$$

Its defining assumption is conditional independence of features given the class:
$$
p(\mathbf{x} \mid y=k) = \prod_j p(x_j \mid y=k).
$$

That assumption is usually false, but the classifier can still perform well because classification needs the relative posterior ordering of classes more than exact density fidelity.

> **Backward reference:** The conditional-independence structure behind Naive Bayes was developed in [Joint Distributions](../../06-Probability-Theory/03-Joint-Distributions/notes.md#43-conditional-independence).

**For AI:** Naive Bayes remains a useful illustration of generative classification, prior smoothing, and posterior class reasoning.

### 7.2 Bayesian Neural Networks

A Bayesian neural network places a distribution over weights:
$$
p(\boldsymbol{\theta} \mid \mathcal{D})
\propto
p(\mathcal{D} \mid \boldsymbol{\theta})\, p(\boldsymbol{\theta}).
$$

Predictions average over weight uncertainty:
$$
p(y_* \mid \mathbf{x}_*, \mathcal{D})
=
\int p(y_* \mid \mathbf{x}_*, \boldsymbol{\theta})\, p(\boldsymbol{\theta} \mid \mathcal{D})\, d\boldsymbol{\theta}.
$$

This is conceptually attractive because the network can express epistemic uncertainty: uncertainty due to limited knowledge rather than inherent observation noise.

In practice, exact BNN posteriors are intractable. That is why the previous subsection's approximations matter so much.

Examples:

1. Bayes by Backprop for variational Gaussian weight posteriors
2. MC dropout as approximate posterior averaging
3. Laplace or SWAG as posterior surrogates around trained networks

Non-examples:

1. A deterministic softmax score is not a posterior uncertainty estimate by itself.
2. High confidence on in-distribution data does not imply good OOD uncertainty.

It is useful to separate two types of uncertainty in Bayesian neural networks.

- **Aleatoric uncertainty:** noise or ambiguity inherent in the data-generating process
- **Epistemic uncertainty:** uncertainty due to limited knowledge of the model parameters

Bayesian weight distributions primarily target epistemic uncertainty. When the model sees more relevant data, epistemic uncertainty should contract. This is why Bayesian approximations are most useful in small-data regimes, under distribution shift, or in active-learning settings where additional data can genuinely reduce uncertainty.

A second key distinction is between **weight-space** and **function-space** uncertainty. Many approximate BNN methods place distributions over weights because weights are the direct parameters of the model. But what matters for decision-making is usually uncertainty over functions or predictions. Different weight configurations can induce similar predictive functions, especially in overparameterized networks. This makes posterior approximation in deep learning difficult: weight-space geometry can be highly redundant, multimodal, and curved even when predictive behavior is comparatively smooth.

Three examples:

**Example 1: data-rich regime.** On a very large in-distribution dataset, a Bayesian neural network may provide only modest epistemic gains over strong ensembles because the posterior is already relatively concentrated in function space.

**Example 2: sparse or rare classes.** For low-frequency classes or rare prompt types, posterior uncertainty can remain large even when average training accuracy is high. This is where Bayesian approximations are most informative.

**Example 3: covariate shift.** A model trained on one image distribution or one prompt distribution may produce confident but unstable predictions off-distribution. Posterior surrogates that widen uncertainty under shift can improve abstention and monitoring behavior, though they do not solve OOD detection perfectly.

Two non-examples:

**Non-example 1:** "A Bayesian neural network always outperforms a deterministic network." The point is not guaranteed accuracy gains. The point is better uncertainty representation for downstream decisions.

**Non-example 2:** "Any ensemble is Bayesian." Ensembles often behave like uncertainty approximators and can resemble Bayesian model averaging in practice, but they are not automatically posterior samples.

There is also an engineering reality. Exact Bayesian neural networks remain too expensive for many large-scale systems, so practitioners often choose between:

- a theoretically closer but expensive approximation
- a cheaper surrogate with weaker posterior interpretation
- a deterministic model plus calibration fix

The right choice depends on the cost of being wrong, the amount of available data, and whether downstream systems actually use uncertainty rather than merely logging it.

**For AI:** This is why Bayesian deep learning should be evaluated by calibration, risk-sensitive utility, abstention quality, active-learning value, and OOD behavior, not only by top-line accuracy.

### 7.3 Uncertainty for Calibration, Active Learning, and OOD Detection

Bayesian uncertainty is operationally useful when a model must decide whether to trust itself.

**Calibration.** Posterior-aware prediction can improve probability calibration by accounting for parameter uncertainty.

**Active learning.** Query the datapoints with highest posterior uncertainty or highest expected information gain.

**OOD detection.** If the posterior predictive is diffuse, unstable across posterior samples, or assigns low mass to a new input, the system can flag potential distribution shift.

Examples:

1. uncertain moderation predictions routed to human review
2. active learning for labeling expensive data
3. uncertainty-aware ranking under sparse feedback

Non-examples:

1. Predictive entropy alone is not a perfect OOD detector.
2. A method with approximate Bayesian flavor is not automatically calibrated.

Calibration itself splits into at least two questions:

1. are predicted probabilities aligned with empirical frequencies in-distribution?
2. does uncertainty rise appropriately under ambiguity, shift, or sparse evidence?

The first can often be improved by temperature scaling or ensembling. The second is harder. Bayesian methods aim at the second because they try to propagate epistemic uncertainty instead of just correcting output logits after the fact.

Three examples:

**Example 1:** An image classifier may achieve good top-1 accuracy while remaining overconfident on corrupted data. Posterior-aware prediction aims to widen uncertainty under such perturbations.

**Example 2:** In active learning, a posterior or posterior surrogate identifies examples where the model's current beliefs are most uncertain, so labeling effort is spent where it will reduce epistemic uncertainty fastest.

**Example 3:** In retrieval and ranking, uncertainty over sparse user interactions can prevent a system from overcommitting to noisy early feedback.

Two non-examples:

**Non-example 1:** using softmax confidence as if it were Bayesian uncertainty.

**Non-example 2:** interpreting disagreement across posterior samples as a guarantee of correctness. It is evidence of uncertainty, not proof of truth.

### 7.4 Bayesian Optimization and Thompson Sampling

Bayesian optimization treats the objective function as uncertain and updates a posterior over it after each expensive evaluation. A surrogate model, often a Gaussian process, balances exploration and exploitation by using both posterior mean and posterior uncertainty.

Thompson sampling uses posterior sampling directly: sample a plausible objective from the posterior and act optimally under that sampled world. Repeating this naturally balances exploration and exploitation.

Examples:

1. hyperparameter tuning for expensive model training
2. prompt or policy selection with expensive evaluations
3. recommendation-bandit systems under uncertainty

Non-examples:

1. random search is not Bayesian optimization
2. fixed epsilon-greedy exploration does not exploit posterior structure

**For AI:** When every training run costs substantial compute, posterior-aware search can save enormous resources by targeting informative trials instead of brute-force sweeps.

Bayesian optimization is especially attractive when the objective is:

- expensive to evaluate
- noisy
- derivative-free
- available only through black-box experimentation

That description fits many modern ML workflows: hyperparameter tuning, inference-time tradeoff tuning, reward-model threshold selection, and evaluation-guided prompt or policy search.

Thompson sampling deserves separate emphasis because it turns posterior reasoning directly into a policy. Sampling one plausible world from the posterior and acting optimally in that sampled world leads to a simple randomized exploration strategy whose randomness is informed, not arbitrary.

This gives a clean contrast:

- epsilon-greedy adds blind random exploration
- UCB adds optimism bonuses
- Thompson sampling samples according to posterior uncertainty

All three can work. Thompson sampling is the most explicitly Bayesian.

### 7.5 Bayesian Views of Regularization and Fine-Tuning

Many ML regularizers admit Bayesian interpretations:

- L2 penalty = Gaussian prior
- L1 penalty = Laplace prior
- group shrinkage = hierarchical prior
- sparse adaptation = structured prior on updates

This perspective is especially useful in fine-tuning. If we believe most weights should remain near their pretrained values, then Bayesian language suggests priors centered at those values rather than at zero. If we believe updates should be low-rank or sparse, Bayesian structure can encode that too.

Examples:

1. posterior or MAP interpretation of weight decay
2. priors centered on pretrained weights for conservative adaptation
3. hierarchical priors across tasks or domains during multi-task fine-tuning

Non-examples:

1. not every optimizer trick has a clean Bayesian meaning
2. saying "Bayesian" does not remove the need to validate adaptation behavior empirically

The Bayesian view is most useful when it changes design decisions. For example:

- Should the prior be centered at zero or at pretrained weights?
- Should updates be globally isotropic or structured by layer and task?
- Should uncertainty be propagated into predictions after fine-tuning?
- Should low-rank adaptation be interpreted as a computational constraint, a prior belief, or both?

These are not merely philosophical questions. They affect which parameters move, how confidently they move, and how much a system trusts its adapted outputs after seeing limited new data.

---

## 8. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
| --- | --- | --- | --- |
| 1 | "The prior is just subjective bias." | Priors can encode symmetry, scale constraints, weak regularization, historical evidence, or hierarchical structure. | State the modeling role of the prior explicitly and perform sensitivity checks. |
| 2 | "The likelihood is a probability distribution over parameters." | The likelihood is a function of the parameter with the data held fixed, not a normalized distribution in $\theta$. | Use posterior language only after multiplying by the prior and normalizing. |
| 3 | "MAP is the same as Bayesian inference." | MAP collapses the posterior to one point and discards posterior uncertainty. | Distinguish posterior summaries from the full posterior and posterior predictive. |
| 4 | "A 95% credible interval and a 95% confidence interval mean the same thing." | They may have similar endpoints but different semantics. | Keep the Bayesian and frequentist probability statements separate. |
| 5 | "Flat priors are objective and harmless." | Flatness depends on parameterization and can create unstable or improper posteriors. | Prefer justified weakly informative priors or invariance-based priors when appropriate. |
| 6 | "Bayes factors are just Bayesian p-values." | Bayes factors compare integrated model evidence, not tail probabilities under a null. | Interpret Bayes factors through posterior odds and prior model odds. |
| 7 | "Conjugate priors are automatically correct." | Conjugacy gives tractability, not truth. | Use conjugacy when it fits the problem or as an instructional baseline, not as a dogma. |
| 8 | "Variational inference gives exact uncertainty if optimized well." | A restricted variational family can remain systematically overconfident. | Evaluate posterior quality, calibration, and sensitivity to the approximation family. |
| 9 | "Softmax confidence is Bayesian uncertainty." | Deterministic confidence scores can be sharply wrong under shift or sparse evidence. | Use posterior or posterior-surrogate uncertainty and validate calibration. |
| 10 | "Posterior predictive checks prove the model is true." | PPCs are diagnostics for mismatch, not proofs of correctness. | Use PPCs to falsify bad fit and combine them with domain checks. |
| 11 | "More MCMC iterations always solve convergence." | Poor geometry and high autocorrelation can persist for a long time. | Check ESS, split-$\hat{R}$, trace behavior, and proposal geometry. |
| 12 | "Bayesian methods eliminate the need for empirical validation." | Bayesian inference still depends on model assumptions, approximations, and deployment conditions. | Validate calibration, robustness, and decision quality on the real use case. |

---

## 9. Exercises

These exercises are mirrored in `exercises.ipynb`, where each one includes a scaffold cell and a full reference solution.

1. **Exercise 1 (*) - Beta-Binomial Updating**
   Let $X_1,\ldots,X_n \sim \operatorname{Bern}(\theta)$ with prior $\theta \sim \operatorname{Beta}(\alpha,\beta)$.
   (a) Derive the posterior density.
   (b) Compute the posterior mean, variance, and MAP.
   (c) Derive the posterior predictive probability of success for one future trial.
   (d) Explain how the answer changes as $\alpha+\beta$ grows with fixed prior mean.

2. **Exercise 2 (*) - Gaussian Prior, Gaussian Likelihood**
   Suppose $X_i \mid \mu \sim \mathcal{N}(\mu,\sigma^2)$ with known $\sigma^2$ and prior $\mu \sim \mathcal{N}(\mu_0,\tau_0^2)$.
   (a) Derive the posterior by completing the square.
   (b) Express the posterior mean as a precision-weighted average.
   (c) Show what happens as $n \to \infty$.
   (d) Interpret the result for a small-data ML monitoring problem.

3. **Exercise 3 (*) - MAP as Regularised MLE**
   (a) Show that a Gaussian prior on $\boldsymbol{\theta}$ yields an L2 penalty in the MAP objective.
   (b) Show that an independent Laplace prior yields an L1 penalty.
   (c) Explain why the resulting optimizer is not yet full Bayesian inference.
   (d) Connect the result to weight decay in deep learning.

4. **Exercise 4 (**) - Posterior Predictive for Count Data**
   Let $X_i \mid \lambda \sim \operatorname{Poisson}(\lambda)$ and $\lambda \sim \operatorname{Gamma}(\alpha,\beta)$.
   (a) Derive the posterior on $\lambda$.
   (b) Derive the posterior predictive for one future count.
   (c) Compare its variance to that of a plug-in Poisson predictor using the posterior mean of $\lambda$.
   (d) Explain why posterior predictive uncertainty can exceed observation-model uncertainty.

5. **Exercise 5 (**) - Bayes Factor vs p-Value**
   Consider a simple null-vs-effect comparison problem.
   (a) Write down the likelihood under $H_0$ and $H_1$.
   (b) Derive or numerically approximate the Bayes factor for a chosen prior under $H_1$.
   (c) Compare the conclusion to a classical significance test.
   (d) Explain how the prior scale affects the Bayes factor.

6. **Exercise 6 (**) - Hierarchical Partial Pooling**
   You observe noisy click-through data for several related groups with very different sample sizes.
   (a) Write down a two-level hierarchical model.
   (b) Explain the difference between no pooling, complete pooling, and partial pooling.
   (c) Show qualitatively which groups shrink most strongly.
   (d) Explain why this is useful in multilingual or multi-market ML systems.

7. **Exercise 7 (***) - ELBO and Variational Inference**
   (a) Starting from $D_{\mathrm{KL}}(q(\theta)\|p(\theta\mid\mathcal{D}))$, derive the ELBO identity.
   (b) Explain why maximizing the ELBO minimizes KL divergence.
   (c) Explain why mean-field VI can underestimate posterior variance.
   (d) Connect the derivation to the VAE objective.

8. **Exercise 8 (***) - Bayesian Uncertainty in ML**
   Compare one or more approximate Bayesian deep-learning methods such as MC dropout, Laplace, SGLD, or SWAG.
   (a) State what posterior object each method approximates.
   (b) Identify one computational advantage.
   (c) Identify one calibration or fidelity limitation.
   (d) Propose which method you would choose for an uncertainty-aware deployment with limited compute and justify the decision.

---

## 10. Why This Matters for AI (2026 Perspective)

| Concept | AI Impact |
| --- | --- |
| Prior as regularizer | Makes assumptions explicit instead of burying them in penalties and initialization choices |
| Posterior uncertainty | Supports abstention, fallback logic, and risk-sensitive prediction |
| Credible intervals | Enable direct probability statements about launch thresholds and safety margins |
| Posterior predictive | Gives calibrated next-step uncertainty rather than just point predictions |
| Conjugate Bayes | Powers fast online updates for bandits, CTR models, and monitoring systems |
| MAP interpretation | Explains weight decay, L1 sparsity, and other regularizers as prior assumptions |
| Hierarchical Bayes | Shares strength across tasks, languages, regions, users, and low-resource groups |
| Bayes factors | Provide an evidence-based alternative to thresholded model-comparison rituals |
| PPCs | Expose misspecification when a probabilistic model fits one metric but misses the real data shape |
| MCMC | Remains the gold-standard asymptotic route for posterior sampling in rich probabilistic models |
| Variational inference | Makes latent-variable Bayes scalable enough for neural architectures and large datasets |
| Amortized inference | Lets neural networks learn approximate posteriors efficiently, as in VAEs |
| Approximate Bayesian DL | Methods like MC dropout, SGLD, and SWAG provide practical uncertainty baselines |
| Bayesian optimization | Saves expensive training budget by using posterior uncertainty to guide search |
| Thompson sampling | Turns posterior uncertainty directly into principled exploration policies |
| Fine-tuning priors | Encourages conservative adaptation around pretrained models and supports structured uncertainty |

Bayesian inference matters for AI because it keeps one crucial quantity visible: what the system still does not know. In modern model deployment, that is often more valuable than squeezing one more decimal place out of average validation accuracy.

---

## 11. Conceptual Bridge

This section sits at a precise point in the curriculum.

Backward, it depends on probability theory and frequentist statistics. From [Joint Distributions](../../06-Probability-Theory/03-Joint-Distributions/notes.md), we inherit Bayes' theorem, conditional distributions, multivariate Gaussian conditioning, and the language of latent variables. From [Estimation Theory](../02-Estimation-Theory/notes.md), we inherit likelihoods, MLE, Fisher information, asymptotic normality, and confidence intervals. From [Hypothesis Testing](../03-Hypothesis-Testing/notes.md), we inherit the classical model-comparison mindset that Bayesian evidence now complements and sometimes challenges.

Conceptually, Bayesian inference is the point where those ideas are reorganized. Bayes' theorem stops being one identity inside probability theory and becomes the architecture of inference. Likelihood stops being only an optimization objective and becomes one term in a full posterior update. Confidence language stops being the only way to talk about uncertainty and is joined by posterior probability statements, credible intervals, and posterior predictive reasoning.

Forward, this section opens several doors.

- Into **Time Series**, repeated Gaussian updating becomes filtering and state estimation.
- Into **Regression Analysis**, penalized estimators acquire full posterior and predictive interpretations.
- Into **Optimization**, variational objectives and stochastic-gradient posterior approximations become algorithmic objects.
- Into **Information Theory**, KL divergence and evidence connect posterior approximation to compression and coding.
- Into **modern ML practice**, approximate Bayesian methods become tools for calibration, active learning, bandits, and uncertainty-aware deployment.

The deepest conceptual move is this: frequentist statistics asks how procedures behave across hypothetical repeated datasets; Bayesian statistics asks how beliefs should change for the dataset we actually observed. Mature ML systems often need both views. They need procedures with strong repeated-use guarantees and posterior-aware decisions on real realized data.

```text
ASCII CURRICULUM POSITION
======================================================================

  Probability Theory
        |
        +-- Bayes' theorem, conditionals, MVN conditioning
        |
        v
  Estimation Theory
        |
        +-- likelihood, MLE, Fisher information, asymptotics
        |
        v
  Hypothesis Testing
        |
        +-- p-values, likelihood ratios, decision rules
        |
        v
  Bayesian Inference
        |
        +-- prior + likelihood -> posterior
        +-- posterior -> prediction, evidence, decisions
        |
        +---------------------> Time Series
        +---------------------> Regression Analysis
        +---------------------> Optimization
        +---------------------> Information Theory

======================================================================
```

If probability theory gave us the syntax of uncertainty and estimation theory gave us the grammar of data-to-parameter inference, Bayesian inference gives us the full probabilistic semantics of learning under uncertainty. It is the chapter where uncertainty stops being a nuisance term and becomes a first-class computational object.

One final way to summarize the bridge is this:

- probability theory told us how to manipulate uncertainty
- estimation theory told us how to fit unknown parameters from data
- hypothesis testing told us how to compare claims under repeated-sampling logic
- Bayesian inference tells us how to update whole distributions of belief and use them for prediction and action

That perspective will keep returning. Whenever a later chapter asks us to choose under uncertainty, propagate uncertainty through a model, or reason about what the system does not know yet, the machinery introduced here is part of the answer.

## References

1. Gelman, A., Carlin, J., Stern, H., Dunson, D., Vehtari, A., and Rubin, D. *Bayesian Data Analysis*. CRC Press.
2. Murphy, K. P. *Machine Learning: A Probabilistic Perspective*. MIT Press.
3. Bishop, C. M. *Pattern Recognition and Machine Learning*. Springer.
4. Bernardo, J. M., and Smith, A. F. M. *Bayesian Theory*. Wiley.
5. Robert, C., and Casella, G. *Monte Carlo Statistical Methods*. Springer.
6. Blei, D. M., Kucukelbir, A., and McAuliffe, J. D. "Variational Inference: A Review for Statisticians." *JASA* (2017).
7. Kingma, D. P., and Welling, M. "Auto-Encoding Variational Bayes." ICLR (2014).
8. Blundell, C., Cornebise, J., Kavukcuoglu, K., and Wierstra, D. "Weight Uncertainty in Neural Networks." ICML (2015).
9. Gal, Y., and Ghahramani, Z. "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning." ICML (2016).
10. Welling, M., and Teh, Y. W. "Bayesian Learning via Stochastic Gradient Langevin Dynamics." ICML (2011).
11. Maddox, W., Garipov, T., Izmailov, P., Vetrov, D., and Wilson, A. G. "A Simple Baseline for Bayesian Uncertainty in Deep Learning." NeurIPS (2019).
12. Johari, R. "Lecture 16: Bayesian Inference." Stanford MS&E 226 notes.
13. Jeffreys, H. *Theory of Probability*. Oxford University Press.
14. Gelman, A. and Shalizi, C. R. "Philosophy and the Practice of Bayesian Statistics." *British Journal of Mathematical and Statistical Psychology* (2013).
15. Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., and Bürkner, P.-C. "Rank-Normalization, Folding, and Localization: An Improved R-hat for Assessing Convergence of MCMC." *Bayesian Analysis* (2021).
