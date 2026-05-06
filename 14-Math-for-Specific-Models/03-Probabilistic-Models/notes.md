[<- Neural Networks](../02-Neural-Networks/notes.md) | [Home](../../README.md) | [RNN and LSTM Math ->](../04-RNN-and-LSTM-Math/notes.md)

---

# Probabilistic Models

Probabilistic models describe data, hidden structure, parameters, and predictions with distributions. They make uncertainty a first-class object rather than an afterthought.

## Overview

The core operations are:

$$
p(x)=\sum_zp(x,z),\qquad p(\theta\mid D)=\frac{p(D\mid\theta)p(\theta)}{p(D)}.
$$

Marginalization handles hidden variables. Bayes rule updates beliefs. Likelihood trains parameters. These ideas appear in naive Bayes, mixture models, HMMs, variational inference, VAEs, uncertainty estimation, and probabilistic interpretations of neural networks.

## Prerequisites

- Probability rules, expectation, and conditional probability
- Log likelihood and cross-entropy
- Linear and neural model sections
- Basic optimization

## Companion Notebooks

| Notebook | Purpose |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Demonstrates likelihoods, Bayes updates, naive Bayes, Gaussian mixtures, EM, HMM forward recursion, Monte Carlo, ELBO intuition, and calibration. |
| [exercises.ipynb](exercises.ipynb) | Ten practice problems for probability normalization, MLE/MAP, Bayes, mixture responsibilities, HMMs, and diagnostics. |

## Learning Objectives

After this section, you should be able to:

- Distinguish joint, marginal, conditional, likelihood, posterior, and predictive distributions.
- Compute MLE and MAP estimates in simple models.
- Apply Bayes rule with conjugate priors.
- Explain naive Bayes and the generative/discriminative distinction.
- Compute mixture responsibilities and one EM update.
- Explain graphical-model factorization and conditional independence.
- Run HMM forward and Viterbi-style recursions at a high level.
- Explain approximate inference, ELBO, and posterior predictive checks.

## Table of Contents

1. [Probabilistic Modeling View](#1-probabilistic-modeling-view)
   - 1.1 [Random variables](#11-random-variables)
   - 1.2 [Joint distribution](#12-joint-distribution)
   - 1.3 [Conditional distribution](#13-conditional-distribution)
   - 1.4 [Marginalization](#14-marginalization)
   - 1.5 [Decision rule](#15-decision-rule)
2. [Likelihood and Estimation](#2-likelihood-and-estimation)
   - 2.1 [Likelihood](#21-likelihood)
   - 2.2 [Log likelihood](#22-log-likelihood)
   - 2.3 [MLE](#23-mle)
   - 2.4 [MAP](#24-map)
   - 2.5 [Predictive distribution](#25-predictive-distribution)
3. [Bayes Rule](#3-bayes-rule)
   - 3.1 [Posterior](#31-posterior)
   - 3.2 [Evidence](#32-evidence)
   - 3.3 [Conjugacy](#33-conjugacy)
   - 3.4 [Prior strength](#34-prior-strength)
   - 3.5 [Posterior predictive](#35-posterior-predictive)
4. [Naive Bayes and Discriminative Contrast](#4-naive-bayes-and-discriminative-contrast)
   - 4.1 [Generative classifier](#41-generative-classifier)
   - 4.2 [Naive assumption](#42-naive-assumption)
   - 4.3 [Classification](#43-classification)
   - 4.4 [Discriminative model](#44-discriminative-model)
   - 4.5 [Calibration](#45-calibration)
5. [Latent Variable Models](#5-latent-variable-models)
   - 5.1 [Latent variable](#51-latent-variable)
   - 5.2 [Mixture model](#52-mixture-model)
   - 5.3 [Responsibilities](#53-responsibilities)
   - 5.4 [Identifiability](#54-identifiability)
   - 5.5 [Representation learning](#55-representation-learning)
6. [Expectation Maximization](#6-expectation-maximization)
   - 6.1 [E-step](#61-estep)
   - 6.2 [M-step](#62-mstep)
   - 6.3 [Lower bound](#63-lower-bound)
   - 6.4 [Gaussian mixture updates](#64-gaussian-mixture-updates)
   - 6.5 [Local optima](#65-local-optima)
7. [Graphical Models](#7-graphical-models)
   - 7.1 [Directed model](#71-directed-model)
   - 7.2 [Undirected model](#72-undirected-model)
   - 7.3 [Conditional independence](#73-conditional-independence)
   - 7.4 [Inference](#74-inference)
   - 7.5 [Message passing](#75-message-passing)
8. [Hidden Markov Models](#8-hidden-markov-models)
   - 8.1 [Markov state](#81-markov-state)
   - 8.2 [Emission](#82-emission)
   - 8.3 [Forward recursion](#83-forward-recursion)
   - 8.4 [Viterbi](#84-viterbi)
   - 8.5 [Sequence modeling bridge](#85-sequence-modeling-bridge)
9. [Approximate Inference](#9-approximate-inference)
   - 9.1 [Monte Carlo](#91-monte-carlo)
   - 9.2 [Variational inference](#92-variational-inference)
   - 9.3 [ELBO](#93-elbo)
   - 9.4 [Reparameterization](#94-reparameterization)
   - 9.5 [Amortized inference](#95-amortized-inference)
10. [Diagnostics](#10-diagnostics)
   - 10.1 [Log-likelihood](#101-loglikelihood)
   - 10.2 [Posterior predictive checks](#102-posterior-predictive-checks)
   - 10.3 [Calibration curves](#103-calibration-curves)
   - 10.4 [Sensitivity to priors](#104-sensitivity-to-priors)
   - 10.5 [Ablations](#105-ablations)

---

## Object Map

```text
observed data:       x, y
latent variables:    z
parameters:          theta
prior:               p(theta)
likelihood:          p(D | theta)
posterior:           p(theta | D)
prediction:          p(y_new | x_new, D)
```

## 1. Probabilistic Modeling View

This part studies probabilistic modeling view as uncertainty-aware modeling. Keep track of observed variables, hidden variables, parameters, and decisions.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Random variables](#1-random-variables) | represent uncertain quantities | $X,Y,Z$ |
| [Joint distribution](#1-joint-distribution) | model all variables together | $p(x,y,z)$ |
| [Conditional distribution](#1-conditional-distribution) | predict one variable given another | $p(y\mid x)$ |
| [Marginalization](#1-marginalization) | sum or integrate out hidden variables | $p(x)=\sum_zp(x,z)$ |
| [Decision rule](#1-decision-rule) | turn probabilities into actions | $a^\star=\arg\min_a E[L(a,Y)\mid x]$ |

### 1.1 Random variables

**Main idea.** Represent uncertain quantities.

Core relation:

$$X,Y,Z$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 1.2 Joint distribution

**Main idea.** Model all variables together.

Core relation:

$$p(x,y,z)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 1.3 Conditional distribution

**Main idea.** Predict one variable given another.

Core relation:

$$p(y\mid x)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 1.4 Marginalization

**Main idea.** Sum or integrate out hidden variables.

Core relation:

$$p(x)=\sum_zp(x,z)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 1.5 Decision rule

**Main idea.** Turn probabilities into actions.

Core relation:

$$a^\star=\arg\min_a E[L(a,Y)\mid x]$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
## 2. Likelihood and Estimation

This part studies likelihood and estimation as uncertainty-aware modeling. Keep track of observed variables, hidden variables, parameters, and decisions.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Likelihood](#2-likelihood) | view data probability as a function of parameters | $L(\theta)=p(D\mid\theta)$ |
| [Log likelihood](#2-log-likelihood) | products become sums | $\ell(\theta)=\sum_i\log p(x_i\mid\theta)$ |
| [MLE](#2-mle) | choose parameters maximizing data likelihood | $\hat\theta_\mathrm{MLE}=\arg\max_\theta\ell(\theta)$ |
| [MAP](#2-map) | include a prior over parameters | $\hat\theta_\mathrm{MAP}=\arg\max_\theta[\log p(D\mid\theta)+\log p(\theta)]$ |
| [Predictive distribution](#2-predictive-distribution) | integrate parameter uncertainty when Bayesian | $p(x_\star\mid D)=\int p(x_\star\mid\theta)p(\theta\mid D)d\theta$ |

### 2.1 Likelihood

**Main idea.** View data probability as a function of parameters.

Core relation:

$$L(\theta)=p(D\mid\theta)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 2.2 Log likelihood

**Main idea.** Products become sums.

Core relation:

$$\ell(\theta)=\sum_i\log p(x_i\mid\theta)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 2.3 MLE

**Main idea.** Choose parameters maximizing data likelihood.

Core relation:

$$\hat\theta_\mathrm{MLE}=\arg\max_\theta\ell(\theta)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 2.4 MAP

**Main idea.** Include a prior over parameters.

Core relation:

$$\hat\theta_\mathrm{MAP}=\arg\max_\theta[\log p(D\mid\theta)+\log p(\theta)]$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 2.5 Predictive distribution

**Main idea.** Integrate parameter uncertainty when bayesian.

Core relation:

$$p(x_\star\mid D)=\int p(x_\star\mid\theta)p(\theta\mid D)d\theta$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
## 3. Bayes Rule

This part studies bayes rule as uncertainty-aware modeling. Keep track of observed variables, hidden variables, parameters, and decisions.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Posterior](#3-posterior) | update beliefs after data | $p(\theta\mid D)=p(D\mid\theta)p(\theta)/p(D)$ |
| [Evidence](#3-evidence) | normalizing probability of data | $p(D)=\int p(D\mid\theta)p(\theta)d\theta$ |
| [Conjugacy](#3-conjugacy) | some priors yield closed-form posteriors | $\mathrm{Beta}+\mathrm{Bernoulli}\rightarrow\mathrm{Beta}$ |
| [Prior strength](#3-prior-strength) | prior counts can regularize low-data estimates | $\alpha,\beta$ |
| [Posterior predictive](#3-posterior-predictive) | predictions average over posterior uncertainty | $p(y\mid x,D)$ |

### 3.1 Posterior

**Main idea.** Update beliefs after data.

Core relation:

$$p(\theta\mid D)=p(D\mid\theta)p(\theta)/p(D)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** Bayes rule is the core update that turns observations into revised uncertainty.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 3.2 Evidence

**Main idea.** Normalizing probability of data.

Core relation:

$$p(D)=\int p(D\mid\theta)p(\theta)d\theta$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 3.3 Conjugacy

**Main idea.** Some priors yield closed-form posteriors.

Core relation:

$$\mathrm{Beta}+\mathrm{Bernoulli}\rightarrow\mathrm{Beta}$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 3.4 Prior strength

**Main idea.** Prior counts can regularize low-data estimates.

Core relation:

$$\alpha,\beta$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 3.5 Posterior predictive

**Main idea.** Predictions average over posterior uncertainty.

Core relation:

$$p(y\mid x,D)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
## 4. Naive Bayes and Discriminative Contrast

This part studies naive bayes and discriminative contrast as uncertainty-aware modeling. Keep track of observed variables, hidden variables, parameters, and decisions.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Generative classifier](#4-generative-classifier) | model class prior and feature likelihood | $p(y,x)=p(y)p(x\mid y)$ |
| [Naive assumption](#4-naive-assumption) | features are conditionally independent given class | $p(x\mid y)=\prod_jp(x_j\mid y)$ |
| [Classification](#4-classification) | choose largest posterior class | $\arg\max_y p(y)\prod_jp(x_j\mid y)$ |
| [Discriminative model](#4-discriminative-model) | model p(y|x) directly | $p_\theta(y\mid x)$ |
| [Calibration](#4-calibration) | probabilistic outputs should match frequencies | $P(Y=y\mid \hat p_y=c)\approx c$ |

### 4.1 Generative classifier

**Main idea.** Model class prior and feature likelihood.

Core relation:

$$p(y,x)=p(y)p(x\mid y)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 4.2 Naive assumption

**Main idea.** Features are conditionally independent given class.

Core relation:

$$p(x\mid y)=\prod_jp(x_j\mid y)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 4.3 Classification

**Main idea.** Choose largest posterior class.

Core relation:

$$\arg\max_y p(y)\prod_jp(x_j\mid y)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 4.4 Discriminative model

**Main idea.** Model p(y|x) directly.

Core relation:

$$p_\theta(y\mid x)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 4.5 Calibration

**Main idea.** Probabilistic outputs should match frequencies.

Core relation:

$$P(Y=y\mid \hat p_y=c)\approx c$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
## 5. Latent Variable Models

This part studies latent variable models as uncertainty-aware modeling. Keep track of observed variables, hidden variables, parameters, and decisions.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Latent variable](#5-latent-variable) | hidden cause explains observed data | $p(x,z)=p(z)p(x\mid z)$ |
| [Mixture model](#5-mixture-model) | data comes from one of several components | $p(x)=\sum_k\pi_kp(x\mid z=k)$ |
| [Responsibilities](#5-responsibilities) | posterior component probabilities | $\gamma_{ik}=p(z_i=k\mid x_i)$ |
| [Identifiability](#5-identifiability) | different latent labels can represent the same distribution | $z$ labels can permute |
| [Representation learning](#5-representation-learning) | latent variables are probabilistic hidden features | $z\rightarrow x$ |

### 5.1 Latent variable

**Main idea.** Hidden cause explains observed data.

Core relation:

$$p(x,z)=p(z)p(x\mid z)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 5.2 Mixture model

**Main idea.** Data comes from one of several components.

Core relation:

$$p(x)=\sum_k\pi_kp(x\mid z=k)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 5.3 Responsibilities

**Main idea.** Posterior component probabilities.

Core relation:

$$\gamma_{ik}=p(z_i=k\mid x_i)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** Responsibilities are soft cluster assignments and the heart of mixture-model EM.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 5.4 Identifiability

**Main idea.** Different latent labels can represent the same distribution.

Core relation:

$$z$ labels can permute$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 5.5 Representation learning

**Main idea.** Latent variables are probabilistic hidden features.

Core relation:

$$z\rightarrow x$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
## 6. Expectation Maximization

This part studies expectation maximization as uncertainty-aware modeling. Keep track of observed variables, hidden variables, parameters, and decisions.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [E-step](#6-estep) | compute posterior over latent variables | $q_i(z)=p(z\mid x_i,\theta^{old})$ |
| [M-step](#6-mstep) | maximize expected complete-data log likelihood | $\theta^{new}=\arg\max_\theta E_q[\log p(x,z\mid\theta)]$ |
| [Lower bound](#6-lower-bound) | EM improves an evidence lower bound | $\log p(x)\ge E_q[\log p(x,z)-\log q(z)]$ |
| [Gaussian mixture updates](#6-gaussian-mixture-updates) | means become responsibility-weighted averages | $\mu_k=\sum_i\gamma_{ik}x_i/\sum_i\gamma_{ik}$ |
| [Local optima](#6-local-optima) | EM depends on initialization | $\theta_0$ matters |

### 6.1 E-step

**Main idea.** Compute posterior over latent variables.

Core relation:

$$q_i(z)=p(z\mid x_i,\theta^{old})$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 6.2 M-step

**Main idea.** Maximize expected complete-data log likelihood.

Core relation:

$$\theta^{new}=\arg\max_\theta E_q[\log p(x,z\mid\theta)]$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 6.3 Lower bound

**Main idea.** Em improves an evidence lower bound.

Core relation:

$$\log p(x)\ge E_q[\log p(x,z)-\log q(z)]$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 6.4 Gaussian mixture updates

**Main idea.** Means become responsibility-weighted averages.

Core relation:

$$\mu_k=\sum_i\gamma_{ik}x_i/\sum_i\gamma_{ik}$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 6.5 Local optima

**Main idea.** Em depends on initialization.

Core relation:

$$\theta_0$ matters$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
## 7. Graphical Models

This part studies graphical models as uncertainty-aware modeling. Keep track of observed variables, hidden variables, parameters, and decisions.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Directed model](#7-directed-model) | factorize a joint distribution by parents | $p(x_{1:n})=\prod_ip(x_i\mid\mathrm{pa}_i)$ |
| [Undirected model](#7-undirected-model) | factorize by potentials over cliques | $p(x)=Z^{-1}\prod_c\psi_c(x_c)$ |
| [Conditional independence](#7-conditional-independence) | graph structure encodes independence assumptions | $X\perp Y\mid Z$ |
| [Inference](#7-inference) | compute marginals or MAP assignments | $p(x_i\mid evidence)$ |
| [Message passing](#7-message-passing) | reuse local computations on graphs | $m_{i\rightarrow j}(x_j)$ |

### 7.1 Directed model

**Main idea.** Factorize a joint distribution by parents.

Core relation:

$$p(x_{1:n})=\prod_ip(x_i\mid\mathrm{pa}_i)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 7.2 Undirected model

**Main idea.** Factorize by potentials over cliques.

Core relation:

$$p(x)=Z^{-1}\prod_c\psi_c(x_c)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 7.3 Conditional independence

**Main idea.** Graph structure encodes independence assumptions.

Core relation:

$$X\perp Y\mid Z$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 7.4 Inference

**Main idea.** Compute marginals or map assignments.

Core relation:

$$p(x_i\mid evidence)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 7.5 Message passing

**Main idea.** Reuse local computations on graphs.

Core relation:

$$m_{i\rightarrow j}(x_j)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
## 8. Hidden Markov Models

This part studies hidden markov models as uncertainty-aware modeling. Keep track of observed variables, hidden variables, parameters, and decisions.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Markov state](#8-markov-state) | current hidden state depends on previous hidden state | $p(z_t\mid z_{t-1})$ |
| [Emission](#8-emission) | observation depends on current hidden state | $p(x_t\mid z_t)$ |
| [Forward recursion](#8-forward-recursion) | compute filtered probabilities | $\alpha_t(j)=p(x_{1:t},z_t=j)$ |
| [Viterbi](#8-viterbi) | find most likely hidden state path | $\max_{z_{1:T}}p(z_{1:T},x_{1:T})$ |
| [Sequence modeling bridge](#8-sequence-modeling-bridge) | HMMs are probabilistic predecessors of neural sequence models | $z_t$ hidden state |

### 8.1 Markov state

**Main idea.** Current hidden state depends on previous hidden state.

Core relation:

$$p(z_t\mid z_{t-1})$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 8.2 Emission

**Main idea.** Observation depends on current hidden state.

Core relation:

$$p(x_t\mid z_t)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 8.3 Forward recursion

**Main idea.** Compute filtered probabilities.

Core relation:

$$\alpha_t(j)=p(x_{1:t},z_t=j)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is dynamic programming for probabilistic sequence inference.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 8.4 Viterbi

**Main idea.** Find most likely hidden state path.

Core relation:

$$\max_{z_{1:T}}p(z_{1:T},x_{1:T})$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 8.5 Sequence modeling bridge

**Main idea.** Hmms are probabilistic predecessors of neural sequence models.

Core relation:

$$z_t$ hidden state$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
## 9. Approximate Inference

This part studies approximate inference as uncertainty-aware modeling. Keep track of observed variables, hidden variables, parameters, and decisions.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Monte Carlo](#9-monte-carlo) | estimate expectations with samples | $E[f(X)]\approx n^{-1}\sum_if(x_i)$ |
| [Variational inference](#9-variational-inference) | approximate posterior with a tractable family | $q_\phi(z)\approx p(z\mid x)$ |
| [ELBO](#9-elbo) | optimize a lower bound on log evidence | $E_q[\log p(x,z)-\log q(z)]$ |
| [Reparameterization](#9-reparameterization) | differentiate through random variables when possible | $z=\mu+\sigma\epsilon$ |
| [Amortized inference](#9-amortized-inference) | use a neural network to predict variational parameters | $q_\phi(z\mid x)$ |

### 9.1 Monte Carlo

**Main idea.** Estimate expectations with samples.

Core relation:

$$E[f(X)]\approx n^{-1}\sum_if(x_i)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 9.2 Variational inference

**Main idea.** Approximate posterior with a tractable family.

Core relation:

$$q_\phi(z)\approx p(z\mid x)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 9.3 ELBO

**Main idea.** Optimize a lower bound on log evidence.

Core relation:

$$E_q[\log p(x,z)-\log q(z)]$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** The ELBO connects classical latent-variable models to VAEs and modern variational methods.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 9.4 Reparameterization

**Main idea.** Differentiate through random variables when possible.

Core relation:

$$z=\mu+\sigma\epsilon$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 9.5 Amortized inference

**Main idea.** Use a neural network to predict variational parameters.

Core relation:

$$q_\phi(z\mid x)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
## 10. Diagnostics

This part studies diagnostics as uncertainty-aware modeling. Keep track of observed variables, hidden variables, parameters, and decisions.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Log-likelihood](#10-loglikelihood) | track data fit under the probabilistic model | $\ell(D)$ |
| [Posterior predictive checks](#10-posterior-predictive-checks) | simulate from the fitted model and compare to data | $x^\mathrm{rep}\sim p(x\mid D)$ |
| [Calibration curves](#10-calibration-curves) | check probability quality | $\mathrm{ECE}$ |
| [Sensitivity to priors](#10-sensitivity-to-priors) | posterior can change under weak data | $p(\theta)$ |
| [Ablations](#10-ablations) | compare generative, discriminative, latent, and neural versions | $\Delta L,\Delta S$ |

### 10.1 Log-likelihood

**Main idea.** Track data fit under the probabilistic model.

Core relation:

$$\ell(D)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 10.2 Posterior predictive checks

**Main idea.** Simulate from the fitted model and compare to data.

Core relation:

$$x^\mathrm{rep}\sim p(x\mid D)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** A probabilistic model should generate data that looks like the data it claims to explain.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 10.3 Calibration curves

**Main idea.** Check probability quality.

Core relation:

$$\mathrm{ECE}$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 10.4 Sensitivity to priors

**Main idea.** Posterior can change under weak data.

Core relation:

$$p(\theta)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 10.5 Ablations

**Main idea.** Compare generative, discriminative, latent, and neural versions.

Core relation:

$$\Delta L,\Delta S$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.

---

## Practice Exercises

1. Normalize a discrete probability vector.
2. Compute Bernoulli log likelihood and MLE.
3. Compute a Beta-Bernoulli posterior.
4. Compute a MAP estimate with pseudo-counts.
5. Classify with naive Bayes.
6. Compute Gaussian mixture responsibilities.
7. Perform one EM mean update.
8. Run one HMM forward step.
9. Estimate an expectation by Monte Carlo.
10. Write a probabilistic-model debugging checklist.

## Why This Matters for AI

Modern AI systems need uncertainty: calibrated classifiers, latent-variable generative models, retrieval confidence, Bayesian decision rules, and probabilistic sequence models. Neural networks often produce the parameters of probability distributions, so probabilistic modeling explains what those outputs mean.

## Bridge to RNN and LSTM Math

Hidden Markov models are probabilistic sequence models with latent states. RNNs replace explicit latent-state probabilities with learned hidden vectors and differentiable recurrence.

## References

- Kevin Murphy, "Probabilistic Machine Learning: An Introduction", 2022: https://probml.github.io/pml-book/book1.html
- Christopher Bishop, "Pattern Recognition and Machine Learning", 2006.
- A. P. Dempster, N. M. Laird, and D. B. Rubin, "Maximum Likelihood from Incomplete Data via the EM Algorithm", 1977: https://academic.oup.com/jrsssb/article/39/1/1/7027539
- Lawrence Rabiner, "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition", 1989.
