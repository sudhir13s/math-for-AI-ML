[<- Reinforcement Learning](../06-Reinforcement-Learning/notes.md) | [Home](../../README.md) | [CNN and Convolution Math ->](../08-CNN-and-Convolution-Math/notes.md)

---

# Generative Models

Generative models learn to sample from, score, or transform data distributions. They are the mathematical foundation behind language generation, image generation, latent-variable modeling, diffusion models, and synthetic data.

## Overview

The goal is to model:

$$
p_\theta(x)\approx p_\mathrm{data}(x).
$$

Different model families choose different paths: autoregressive factorization, latent-variable bounds, adversarial games, invertible transformations, or iterative denoising.

## Prerequisites

- Probability, likelihood, KL divergence, and ELBO
- Neural networks and optimization
- Autoregressive sequence probability
- Basic Gaussian sampling and matrix operations

## Companion Notebooks

| Notebook | Purpose |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Demonstrates autoregressive likelihood, VAE reparameterization and KL, GAN losses, flow change of variables, diffusion noising, score updates, and FID intuition. |
| [exercises.ipynb](exercises.ipynb) | Ten practice problems for generative-model objectives and diagnostics. |

## Learning Objectives

After this section, you should be able to:

- Compare autoregressive models, VAEs, GANs, flows, diffusion models, and score models.
- Compute autoregressive log likelihood.
- Write and interpret the VAE ELBO and reparameterization trick.
- Explain GAN minimax training and mode collapse.
- Apply the change-of-variables formula for a simple flow.
- Simulate diffusion noising and denoising loss.
- Explain score matching and guidance.
- Evaluate generative models with likelihood, diversity, sample quality, and FID-style statistics.

## Table of Contents

1. [Generative Modeling Goal](#1-generative-modeling-goal)
   - 1.1 [Data distribution](#11-data-distribution)
   - 1.2 [Model distribution](#12-model-distribution)
   - 1.3 [Sampling](#13-sampling)
   - 1.4 [Likelihood](#14-likelihood)
   - 1.5 [Conditional generation](#15-conditional-generation)
2. [Autoregressive Models](#2-autoregressive-models)
   - 2.1 [Chain rule](#21-chain-rule)
   - 2.2 [Teacher forcing](#22-teacher-forcing)
   - 2.3 [Sampling loop](#23-sampling-loop)
   - 2.4 [Exact likelihood](#24-exact-likelihood)
   - 2.5 [Cost](#25-cost)
3. [Variational Autoencoders](#3-variational-autoencoders)
   - 3.1 [Latent variable model](#31-latent-variable-model)
   - 3.2 [Encoder](#32-encoder)
   - 3.3 [Decoder](#33-decoder)
   - 3.4 [ELBO](#34-elbo)
   - 3.5 [Reparameterization](#35-reparameterization)
4. [GANs](#4-gans)
   - 4.1 [Generator](#41-generator)
   - 4.2 [Discriminator](#42-discriminator)
   - 4.3 [Minimax objective](#43-minimax-objective)
   - 4.4 [Mode collapse](#44-mode-collapse)
   - 4.5 [No direct likelihood](#45-no-direct-likelihood)
5. [Normalizing Flows](#5-normalizing-flows)
   - 5.1 [Invertible map](#51-invertible-map)
   - 5.2 [Change of variables](#52-change-of-variables)
   - 5.3 [Exact likelihood](#53-exact-likelihood)
   - 5.4 [Architectural constraint](#54-architectural-constraint)
   - 5.5 [Sampling](#55-sampling)
6. [Diffusion Models](#6-diffusion-models)
   - 6.1 [Forward noising](#61-forward-noising)
   - 6.2 [Closed-form noising](#62-closedform-noising)
   - 6.3 [Denoising model](#63-denoising-model)
   - 6.4 [Training objective](#64-training-objective)
   - 6.5 [Reverse sampling](#65-reverse-sampling)
7. [Score-Based View](#7-scorebased-view)
   - 7.1 [Score](#71-score)
   - 7.2 [Denoising score matching](#72-denoising-score-matching)
   - 7.3 [Langevin sampling](#73-langevin-sampling)
   - 7.4 [SDE view](#74-sde-view)
   - 7.5 [Guidance](#75-guidance)
8. [Evaluation](#8-evaluation)
   - 8.1 [Log likelihood](#81-log-likelihood)
   - 8.2 [Sample quality](#82-sample-quality)
   - 8.3 [Diversity](#83-diversity)
   - 8.4 [FID intuition](#84-fid-intuition)
   - 8.5 [Precision recall](#85-precision-recall)
9. [Applications and Tradeoffs](#9-applications-and-tradeoffs)
   - 9.1 [Text](#91-text)
   - 9.2 [Images](#92-images)
   - 9.3 [Representation learning](#93-representation-learning)
   - 9.4 [Data augmentation](#94-data-augmentation)
   - 9.5 [Safety and misuse](#95-safety-and-misuse)
10. [Diagnostics](#10-diagnostics)
   - 10.1 [Likelihood versus samples](#101-likelihood-versus-samples)
   - 10.2 [Latent traversals](#102-latent-traversals)
   - 10.3 [Mode coverage](#103-mode-coverage)
   - 10.4 [Denoising curves](#104-denoising-curves)
   - 10.5 [Ablations](#105-ablations)

---

## Model Family Map

| Family | Likelihood | Sampling | Main strength | Main cost |
| --- | --- | --- | --- | --- |
| Autoregressive | Exact | Sequential | Strong likelihood and text generation | Serial generation |
| VAE | Lower bound | Fast | Latent representation | Blurry samples if decoder weak |
| GAN | Usually unavailable | Fast | Sharp samples | Unstable training, mode collapse |
| Flow | Exact | Fast-ish | Exact density and invertibility | Architecture constraints |
| Diffusion | Variational/score view | Iterative | High sample quality and controllability | Many denoising steps |

## 1. Generative Modeling Goal

This part studies generative modeling goal as ways to model, sample, and evaluate data distributions.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Data distribution](#1-data-distribution) | learn a model of how examples are generated | $p_\mathrm{data}(x)$ |
| [Model distribution](#1-model-distribution) | choose a parametric distribution family | $p_\theta(x)$ |
| [Sampling](#1-sampling) | generate new examples from the model | $x\sim p_\theta$ |
| [Likelihood](#1-likelihood) | some models assign exact or approximate probabilities | $\log p_\theta(x)$ |
| [Conditional generation](#1-conditional-generation) | generate using labels, prompts, or context | $p_\theta(x\mid c)$ |

### 1.1 Data distribution

**Main idea.** Learn a model of how examples are generated.

Core relation:

$$p_\mathrm{data}(x)$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 1.2 Model distribution

**Main idea.** Choose a parametric distribution family.

Core relation:

$$p_\theta(x)$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 1.3 Sampling

**Main idea.** Generate new examples from the model.

Core relation:

$$x\sim p_\theta$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 1.4 Likelihood

**Main idea.** Some models assign exact or approximate probabilities.

Core relation:

$$\log p_\theta(x)$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 1.5 Conditional generation

**Main idea.** Generate using labels, prompts, or context.

Core relation:

$$p_\theta(x\mid c)$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
## 2. Autoregressive Models

This part studies autoregressive models as ways to model, sample, and evaluate data distributions.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Chain rule](#2-chain-rule) | factorize data into conditional predictions | $p(x_{1:T})=\prod_tp(x_t\mid x_{<t})$ |
| [Teacher forcing](#2-teacher-forcing) | train conditionals with true previous tokens | $-\log p_\theta(x_t\mid x_{<t}^\star)$ |
| [Sampling loop](#2-sampling-loop) | generate one variable at a time | $x_t\sim p_\theta(\cdot\mid x_{<t})$ |
| [Exact likelihood](#2-exact-likelihood) | log likelihood is a sum of conditional log probabilities | $\log p(x)=\sum_t\log p(x_t\mid x_{<t})$ |
| [Cost](#2-cost) | generation is sequential in the generated dimension | $O(T)$ serial steps |

### 2.1 Chain rule

**Main idea.** Factorize data into conditional predictions.

Core relation:

$$p(x_{1:T})=\prod_tp(x_t\mid x_{<t})$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 2.2 Teacher forcing

**Main idea.** Train conditionals with true previous tokens.

Core relation:

$$-\log p_\theta(x_t\mid x_{<t}^\star)$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 2.3 Sampling loop

**Main idea.** Generate one variable at a time.

Core relation:

$$x_t\sim p_\theta(\cdot\mid x_{<t})$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 2.4 Exact likelihood

**Main idea.** Log likelihood is a sum of conditional log probabilities.

Core relation:

$$\log p(x)=\sum_t\log p(x_t\mid x_{<t})$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 2.5 Cost

**Main idea.** Generation is sequential in the generated dimension.

Core relation:

$$O(T)$ serial steps$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
## 3. Variational Autoencoders

This part studies variational autoencoders as ways to model, sample, and evaluate data distributions.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Latent variable model](#3-latent-variable-model) | generate data from a latent code | $p_\theta(x,z)=p(z)p_\theta(x\mid z)$ |
| [Encoder](#3-encoder) | approximate posterior over latent variables | $q_\phi(z\mid x)$ |
| [Decoder](#3-decoder) | map latent samples to data distribution | $p_\theta(x\mid z)$ |
| [ELBO](#3-elbo) | optimize a tractable lower bound | $\log p_\theta(x)\ge E_q[\log p_\theta(x\mid z)]-D_\mathrm{KL}(q_\phi(z\mid x)\Vert p(z))$ |
| [Reparameterization](#3-reparameterization) | sample differentiably from a Gaussian posterior | $z=\mu+\sigma\odot\epsilon,\ \epsilon\sim N(0,I)$ |

### 3.1 Latent variable model

**Main idea.** Generate data from a latent code.

Core relation:

$$p_\theta(x,z)=p(z)p_\theta(x\mid z)$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 3.2 Encoder

**Main idea.** Approximate posterior over latent variables.

Core relation:

$$q_\phi(z\mid x)$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 3.3 Decoder

**Main idea.** Map latent samples to data distribution.

Core relation:

$$p_\theta(x\mid z)$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 3.4 ELBO

**Main idea.** Optimize a tractable lower bound.

Core relation:

$$\log p_\theta(x)\ge E_q[\log p_\theta(x\mid z)]-D_\mathrm{KL}(q_\phi(z\mid x)\Vert p(z))$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is the bridge from probabilistic latent variables to trainable variational autoencoders.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 3.5 Reparameterization

**Main idea.** Sample differentiably from a gaussian posterior.

Core relation:

$$z=\mu+\sigma\odot\epsilon,\ \epsilon\sim N(0,I)$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
## 4. GANs

This part studies gans as ways to model, sample, and evaluate data distributions.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Generator](#4-generator) | map noise to synthetic samples | $x=G_\theta(z)$ |
| [Discriminator](#4-discriminator) | score whether samples look real | $D_\psi(x)\in(0,1)$ |
| [Minimax objective](#4-minimax-objective) | train generator and discriminator adversarially | $\min_G\max_D E_{x\sim p_data}\log D(x)+E_z\log(1-D(G(z)))$ |
| [Mode collapse](#4-mode-collapse) | generator may cover only part of the data distribution | $p_G$ misses modes |
| [No direct likelihood](#4-no-direct-likelihood) | standard GANs sample well but do not provide easy likelihoods | $\log p_G(x)$ unavailable |

### 4.1 Generator

**Main idea.** Map noise to synthetic samples.

Core relation:

$$x=G_\theta(z)$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 4.2 Discriminator

**Main idea.** Score whether samples look real.

Core relation:

$$D_\psi(x)\in(0,1)$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 4.3 Minimax objective

**Main idea.** Train generator and discriminator adversarially.

Core relation:

$$\min_G\max_D E_{x\sim p_data}\log D(x)+E_z\log(1-D(G(z)))$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** GANs trade likelihood for an adversarial learning signal that can produce sharp samples.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 4.4 Mode collapse

**Main idea.** Generator may cover only part of the data distribution.

Core relation:

$$p_G$ misses modes$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 4.5 No direct likelihood

**Main idea.** Standard gans sample well but do not provide easy likelihoods.

Core relation:

$$\log p_G(x)$ unavailable$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
## 5. Normalizing Flows

This part studies normalizing flows as ways to model, sample, and evaluate data distributions.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Invertible map](#5-invertible-map) | transform simple noise into data with an invertible function | $x=f_\theta(z)$ |
| [Change of variables](#5-change-of-variables) | density uses Jacobian determinant | $\log p_X(x)=\log p_Z(z)+\log|\det \partial f^{-1}/\partial x|$ |
| [Exact likelihood](#5-exact-likelihood) | flows can train by maximum likelihood | $\max_\theta\sum_i\log p_\theta(x_i)$ |
| [Architectural constraint](#5-architectural-constraint) | layers must be invertible and have tractable Jacobians | $\det J$ tractable |
| [Sampling](#5-sampling) | sample z then apply f | $z\sim p_Z,\ x=f_\theta(z)$ |

### 5.1 Invertible map

**Main idea.** Transform simple noise into data with an invertible function.

Core relation:

$$x=f_\theta(z)$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 5.2 Change of variables

**Main idea.** Density uses jacobian determinant.

Core relation:

$$\log p_X(x)=\log p_Z(z)+\log|\det \partial f^{-1}/\partial x|$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** Flows are powerful because they keep exact likelihood through invertible maps.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 5.3 Exact likelihood

**Main idea.** Flows can train by maximum likelihood.

Core relation:

$$\max_\theta\sum_i\log p_\theta(x_i)$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 5.4 Architectural constraint

**Main idea.** Layers must be invertible and have tractable jacobians.

Core relation:

$$\det J$ tractable$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 5.5 Sampling

**Main idea.** Sample z then apply f.

Core relation:

$$z\sim p_Z,\ x=f_\theta(z)$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
## 6. Diffusion Models

This part studies diffusion models as ways to model, sample, and evaluate data distributions.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Forward noising](#6-forward-noising) | gradually corrupt data with Gaussian noise | $q(x_t\mid x_{t-1})$ |
| [Closed-form noising](#6-closedform-noising) | sample noisy x_t directly from x_0 | $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$ |
| [Denoising model](#6-denoising-model) | learn to predict noise or clean data | $\epsilon_\theta(x_t,t)$ |
| [Training objective](#6-training-objective) | common DDPM loss predicts added noise | $E\Vert \epsilon-\epsilon_\theta(x_t,t)\Vert^2$ |
| [Reverse sampling](#6-reverse-sampling) | start from noise and denoise step by step | $p_\theta(x_{t-1}\mid x_t)$ |

### 6.1 Forward noising

**Main idea.** Gradually corrupt data with gaussian noise.

Core relation:

$$q(x_t\mid x_{t-1})$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 6.2 Closed-form noising

**Main idea.** Sample noisy x_t directly from x_0.

Core relation:

$$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 6.3 Denoising model

**Main idea.** Learn to predict noise or clean data.

Core relation:

$$\epsilon_\theta(x_t,t)$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 6.4 Training objective

**Main idea.** Common ddpm loss predicts added noise.

Core relation:

$$E\Vert \epsilon-\epsilon_\theta(x_t,t)\Vert^2$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** Diffusion training often becomes a supervised denoising problem.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 6.5 Reverse sampling

**Main idea.** Start from noise and denoise step by step.

Core relation:

$$p_\theta(x_{t-1}\mid x_t)$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
## 7. Score-Based View

This part studies score-based view as ways to model, sample, and evaluate data distributions.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Score](#7-score) | gradient of log density with respect to data | $s(x)=\nabla_x\log p(x)$ |
| [Denoising score matching](#7-denoising-score-matching) | learn scores from noisy samples | $s_\theta(x_t,t)$ |
| [Langevin sampling](#7-langevin-sampling) | move samples toward high-density regions plus noise | $x\leftarrow x+\eta s_\theta(x)+\sqrt{2\eta}\xi$ |
| [SDE view](#7-sde-view) | continuous-time noising and denoising processes | $dx=f(x,t)dt+g(t)dw$ |
| [Guidance](#7-guidance) | condition generation by modifying scores or logits | $s_\mathrm{guided}=s_\mathrm{uncond}+w(s_\mathrm{cond}-s_\mathrm{uncond})$ |

### 7.1 Score

**Main idea.** Gradient of log density with respect to data.

Core relation:

$$s(x)=\nabla_x\log p(x)$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 7.2 Denoising score matching

**Main idea.** Learn scores from noisy samples.

Core relation:

$$s_\theta(x_t,t)$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 7.3 Langevin sampling

**Main idea.** Move samples toward high-density regions plus noise.

Core relation:

$$x\leftarrow x+\eta s_\theta(x)+\sqrt{2\eta}\xi$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 7.4 SDE view

**Main idea.** Continuous-time noising and denoising processes.

Core relation:

$$dx=f(x,t)dt+g(t)dw$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 7.5 Guidance

**Main idea.** Condition generation by modifying scores or logits.

Core relation:

$$s_\mathrm{guided}=s_\mathrm{uncond}+w(s_\mathrm{cond}-s_\mathrm{uncond})$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** Guidance is one reason conditional diffusion models can trade diversity for prompt adherence.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
## 8. Evaluation

This part studies evaluation as ways to model, sample, and evaluate data distributions.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Log likelihood](#8-log-likelihood) | evaluate density when tractable | $\log p_\theta(x)$ |
| [Sample quality](#8-sample-quality) | visual or task-specific quality of generated examples | $S_\mathrm{quality}$ |
| [Diversity](#8-diversity) | model should cover data modes | $\mathrm{support}(p_\theta)$ |
| [FID intuition](#8-fid-intuition) | compare feature means and covariances | $\Vert\mu_r-\mu_g\Vert^2+\mathrm{Tr}(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2})$ |
| [Precision recall](#8-precision-recall) | separate sample fidelity from mode coverage | $\mathrm{precision},\mathrm{recall}$ |

### 8.1 Log likelihood

**Main idea.** Evaluate density when tractable.

Core relation:

$$\log p_\theta(x)$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 8.2 Sample quality

**Main idea.** Visual or task-specific quality of generated examples.

Core relation:

$$S_\mathrm{quality}$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 8.3 Diversity

**Main idea.** Model should cover data modes.

Core relation:

$$\mathrm{support}(p_\theta)$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 8.4 FID intuition

**Main idea.** Compare feature means and covariances.

Core relation:

$$\Vert\mu_r-\mu_g\Vert^2+\mathrm{Tr}(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2})$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 8.5 Precision recall

**Main idea.** Separate sample fidelity from mode coverage.

Core relation:

$$\mathrm{precision},\mathrm{recall}$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
## 9. Applications and Tradeoffs

This part studies applications and tradeoffs as ways to model, sample, and evaluate data distributions.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Text](#9-text) | autoregressive LMs dominate discrete sequence generation | $p(x_t\mid x_{<t})$ |
| [Images](#9-images) | diffusion and autoregressive models are common high-quality image generators | $p(x\mid c)$ |
| [Representation learning](#9-representation-learning) | VAEs learn latent spaces | $z$ |
| [Data augmentation](#9-data-augmentation) | synthetic samples can help or hurt depending on quality | $D_\mathrm{real}\cup D_\mathrm{synthetic}$ |
| [Safety and misuse](#9-safety-and-misuse) | generation systems need provenance, filtering, and evaluation | $\mathrm{risk}$ |

### 9.1 Text

**Main idea.** Autoregressive lms dominate discrete sequence generation.

Core relation:

$$p(x_t\mid x_{<t})$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 9.2 Images

**Main idea.** Diffusion and autoregressive models are common high-quality image generators.

Core relation:

$$p(x\mid c)$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 9.3 Representation learning

**Main idea.** Vaes learn latent spaces.

Core relation:

$$z$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 9.4 Data augmentation

**Main idea.** Synthetic samples can help or hurt depending on quality.

Core relation:

$$D_\mathrm{real}\cup D_\mathrm{synthetic}$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 9.5 Safety and misuse

**Main idea.** Generation systems need provenance, filtering, and evaluation.

Core relation:

$$\mathrm{risk}$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
## 10. Diagnostics

This part studies diagnostics as ways to model, sample, and evaluate data distributions.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Likelihood versus samples](#10-likelihood-versus-samples) | good likelihood and good samples do not always align | $\log p$ versus visual quality |
| [Latent traversals](#10-latent-traversals) | inspect smoothness and meaning of latent directions | $z+\alpha v$ |
| [Mode coverage](#10-mode-coverage) | check diversity and rare classes | $\mathrm{coverage}$ |
| [Denoising curves](#10-denoising-curves) | track loss by timestep | $L_t$ |
| [Ablations](#10-ablations) | compare architecture, objective, guidance, sampling steps, and conditioning | $\Delta S,\Delta T$ |

### 10.1 Likelihood versus samples

**Main idea.** Good likelihood and good samples do not always align.

Core relation:

$$\log p$ versus visual quality$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 10.2 Latent traversals

**Main idea.** Inspect smoothness and meaning of latent directions.

Core relation:

$$z+\alpha v$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 10.3 Mode coverage

**Main idea.** Check diversity and rare classes.

Core relation:

$$\mathrm{coverage}$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 10.4 Denoising curves

**Main idea.** Track loss by timestep.

Core relation:

$$L_t$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 10.5 Ablations

**Main idea.** Compare architecture, objective, guidance, sampling steps, and conditioning.

Core relation:

$$\Delta S,\Delta T$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.

---

## Practice Exercises

1. Compute autoregressive log likelihood.
2. Compute a VAE Gaussian KL to a standard normal prior.
3. Apply the reparameterization trick.
4. Compute GAN discriminator and generator losses.
5. Apply a 1D flow change-of-variables formula.
6. Simulate one diffusion noising step.
7. Compute a denoising MSE loss.
8. Take one score-based Langevin update.
9. Compute a simplified FID-style distance.
10. Write a generative-model debugging checklist.

## Why This Matters for AI

Modern AI is largely generative: LLMs generate text, diffusion models generate images, VAEs and flows model latent structure, and synthetic data systems generate training examples. Understanding the objective behind each model prevents shallow comparisons.

## Bridge to CNN and Convolution Math

Image generators often use convolutional or attention-based backbones. The next section studies convolution math, a key building block for many vision generators and discriminators.

## References

- Diederik Kingma and Max Welling, "Auto-Encoding Variational Bayes", 2013: https://arxiv.org/abs/1312.6114
- Ian Goodfellow et al., "Generative Adversarial Nets", 2014: https://papers.nips.cc/paper/5423-generative-adversarial-nets
- Jonathan Ho, Ajay Jain, and Pieter Abbeel, "Denoising Diffusion Probabilistic Models", 2020: https://arxiv.org/abs/2006.11239
- Yang Song et al., "Score-Based Generative Modeling through Stochastic Differential Equations", 2020: https://arxiv.org/abs/2011.13456
- Durk Kingma and Prafulla Dhariwal, "Glow: Generative Flow with Invertible 1x1 Convolutions", 2018: https://arxiv.org/abs/1807.03039
