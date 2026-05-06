[<- Fine-Tuning Math](../07-Fine-Tuning-Math/notes.md) | [Home](../../README.md) | [Efficient Attention and Inference ->](../09-Efficient-Attention-and-Inference/notes.md)

---

# Scaling Laws

Scaling laws are empirical formulas for forecasting how language-model loss changes as parameters, data, and compute change. They are not a replacement for experiments; they are a way to make experiments smaller, cheaper, and more informative.

## Overview

The core form is a power law with a floor:

$$
L(X)=L_\infty + AX^{-\alpha}.
$$

Here $L$ is usually held-out cross-entropy, $X$ is a resource such as parameters, tokens, or compute, $L_\infty$ is the irreducible floor for the setup, and $\alpha$ controls how fast excess loss falls. In LLM planning, the most important resources are parameter count $N$, training tokens $D$, and training compute $C$. For dense decoder-only transformers, a useful first estimate is:

$$
C\approx 6ND.
$$

The practical question is: for a fixed budget, what combination of $N$ and $D$ gives the lowest expected loss, and how uncertain is that forecast?

## Prerequisites

- Cross-entropy, perplexity, and held-out loss
- Logarithms and log-log plots
- Basic curve fitting and residual analysis
- Training FLOPs and token accounting
- The previous sections on training at scale and fine-tuning

## Companion Notebooks

| Notebook | Purpose |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Fits toy power laws, builds IsoFLOP curves, solves compute-optimal allocation by grid search, studies data quality, serving cost, and residual diagnostics. |
| [exercises.ipynb](exercises.ipynb) | Ten practice problems for power-law fitting, compute budgets, Chinchilla-style sizing, undertraining checks, and forecast uncertainty. |

## Learning Objectives

After this section, you should be able to:

- Explain the difference between parameter, data, and compute scaling.
- Fit a power law when the irreducible loss floor is known or hypothesized.
- Estimate training compute with $C\approx 6ND$.
- Draw and interpret IsoFLOP tradeoffs.
- Solve a compute-optimal allocation problem for a two-term loss proxy.
- Explain the Kaplan and Chinchilla lessons without cargo-culting ratios.
- Reason about data quality, repeated data, and effective token counts.
- Include inference cost and latency in model-size decisions.
- Diagnose residuals and uncertainty in scaling-law forecasts.

## Table of Contents

1. [What Scaling Laws Measure](#1-what-scaling-laws-measure)
   - 1.1 [Loss as the target](#11-loss-as-the-target)
   - 1.2 [Resources](#12-resources)
   - 1.3 [Power-law form](#13-powerlaw-form)
   - 1.4 [Forecasting discipline](#14-forecasting-discipline)
   - 1.5 [Scope](#15-scope)
2. [Parameter Scaling](#2-parameter-scaling)
   - 2.1 [Model-size law](#21-modelsize-law)
   - 2.2 [Capacity limit](#22-capacity-limit)
   - 2.3 [Over-large model](#23-overlarge-model)
   - 2.4 [Architecture dependence](#24-architecture-dependence)
   - 2.5 [Fine-tuning implication](#25-finetuning-implication)
3. [Data Scaling](#3-data-scaling)
   - 3.1 [Token-count law](#31-tokencount-law)
   - 3.2 [Data quality](#32-data-quality)
   - 3.3 [Repeated data](#33-repeated-data)
   - 3.4 [Domain match](#34-domain-match)
   - 3.5 [Tokenizer dependence](#35-tokenizer-dependence)
4. [Compute Scaling](#4-compute-scaling)
   - 4.1 [Dense training FLOPs](#41-dense-training-flops)
   - 4.2 [Compute law](#42-compute-law)
   - 4.3 [IsoFLOP curves](#43-isoflop-curves)
   - 4.4 [Compute is not wall-clock](#44-compute-is-not-wallclock)
   - 4.5 [Budget translation](#45-budget-translation)
5. [Compute-Optimal Allocation](#5-computeoptimal-allocation)
   - 5.1 [Two-term loss proxy](#51-twoterm-loss-proxy)
   - 5.2 [Constraint](#52-constraint)
   - 5.3 [Balanced marginal returns](#53-balanced-marginal-returns)
   - 5.4 [Chinchilla intuition](#54-chinchilla-intuition)
   - 5.5 [Practical rule](#55-practical-rule)
6. [Kaplan and Chinchilla Lessons](#6-kaplan-and-chinchilla-lessons)
   - 6.1 [Kaplan-style result](#61-kaplanstyle-result)
   - 6.2 [Chinchilla revision](#62-chinchilla-revision)
   - 6.3 [Undertraining diagnostic](#63-undertraining-diagnostic)
   - 6.4 [Overtraining for inference](#64-overtraining-for-inference)
   - 6.5 [Do not cargo-cult ratios](#65-do-not-cargocult-ratios)
7. [Fitting Scaling Laws](#7-fitting-scaling-laws)
   - 7.1 [Log transformation](#71-log-transformation)
   - 7.2 [Unknown floor](#72-unknown-floor)
   - 7.3 [Holdout validation](#73-holdout-validation)
   - 7.4 [Residual analysis](#74-residual-analysis)
   - 7.5 [Extrapolation risk](#75-extrapolation-risk)
8. [Data Quality and Mixtures](#8-data-quality-and-mixtures)
   - 8.1 [Mixture weights](#81-mixture-weights)
   - 8.2 [Effective tokens](#82-effective-tokens)
   - 8.3 [Contamination](#83-contamination)
   - 8.4 [Curriculum and filtering](#84-curriculum-and-filtering)
   - 8.5 [Evaluation distribution](#85-evaluation-distribution)
9. [Inference-Aware Scaling](#9-inferenceaware-scaling)
   - 9.1 [Serving cost](#91-serving-cost)
   - 9.2 [Latency constraint](#92-latency-constraint)
   - 9.3 [Small overtrained models](#93-small-overtrained-models)
   - 9.4 [Distillation](#94-distillation)
   - 9.5 [Total cost objective](#95-total-cost-objective)
10. [Limits and Debugging](#10-limits-and-debugging)
   - 10.1 [Not a theorem of all AI](#101-not-a-theorem-of-all-ai)
   - 10.2 [Metric artifacts](#102-metric-artifacts)
   - 10.3 [Architecture changes](#103-architecture-changes)
   - 10.4 [Bad small runs](#104-bad-small-runs)
   - 10.5 [Decision checklist](#105-decision-checklist)

---

## Forecasting Workflow

```text
small runs -> held-out loss table -> fit simple law -> holdout forecast -> residual check -> budget decision -> larger validation run
```

At each step, write down the exact setup: architecture, tokenizer, data mixture, optimizer, schedule, batch size, precision, evaluation set, and filtering. A scaling law fitted under one setup is not guaranteed to transfer to another setup.

## 1. What Scaling Laws Measure

This part studies what scaling laws measure as a forecasting tool. The formulas are useful only when you keep the measurement setup fixed and validate predictions against real runs.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Loss as the target](#1-loss-as-the-target) | scaling laws usually model held-out cross-entropy | $L=-E[\log p_\theta(t_i\mid t_{<i})]$ |
| [Resources](#1-resources) | parameters, tokens, and compute are the main axes | $N,\ D,\ C$ |
| [Power-law form](#1-powerlaw-form) | loss often improves approximately linearly on log-log axes after subtracting an irreducible floor | $L(X)=L_\infty + AX^{-\alpha}$ |
| [Forecasting discipline](#1-forecasting-discipline) | fit small runs, predict larger runs, then validate residuals | $\hat L(C_\mathrm{large})$ |
| [Scope](#1-scope) | a scaling law is empirical over a data, architecture, optimizer, and tokenization regime | $\mathrm{law}=\mathrm{fit}\mid\mathrm{setup}$ |

### 1.1 Loss as the target

**Main idea.** Scaling laws usually model held-out cross-entropy.

Core relation:

$$L=-E[\log p_\theta(t_i\mid t_{<i})]$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 1.2 Resources

**Main idea.** Parameters, tokens, and compute are the main axes.

Core relation:

$$N,\ D,\ C$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 1.3 Power-law form

**Main idea.** Loss often improves approximately linearly on log-log axes after subtracting an irreducible floor.

Core relation:

$$L(X)=L_\infty + AX^{-\alpha}$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 1.4 Forecasting discipline

**Main idea.** Fit small runs, predict larger runs, then validate residuals.

Core relation:

$$\hat L(C_\mathrm{large})$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 1.5 Scope

**Main idea.** A scaling law is empirical over a data, architecture, optimizer, and tokenization regime.

Core relation:

$$\mathrm{law}=\mathrm{fit}\mid\mathrm{setup}$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
## 2. Parameter Scaling

This part studies parameter scaling as a forecasting tool. The formulas are useful only when you keep the measurement setup fixed and validate predictions against real runs.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Model-size law](#2-modelsize-law) | larger models reduce loss when data and compute are not limiting | $L(N)=L_\infty + A_NN^{-\alpha_N}$ |
| [Capacity limit](#2-capacity-limit) | too small a model cannot represent the distribution well | $N\downarrow\Rightarrow L\uparrow$ |
| [Over-large model](#2-overlarge-model) | a model can be too large for the available token budget | $D/N$ too small |
| [Architecture dependence](#2-architecture-dependence) | the coefficients depend on architecture and training recipe | $A_N,\alpha_N$ are not universal constants |
| [Fine-tuning implication](#2-finetuning-implication) | adapter rank can be viewed as an adaptation-capacity axis | $r$ controls update capacity |

### 2.1 Model-size law

**Main idea.** Larger models reduce loss when data and compute are not limiting.

Core relation:

$$L(N)=L_\infty + A_NN^{-\alpha_N}$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 2.2 Capacity limit

**Main idea.** Too small a model cannot represent the distribution well.

Core relation:

$$N\downarrow\Rightarrow L\uparrow$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 2.3 Over-large model

**Main idea.** A model can be too large for the available token budget.

Core relation:

$$D/N$ too small$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 2.4 Architecture dependence

**Main idea.** The coefficients depend on architecture and training recipe.

Core relation:

$$A_N,\alpha_N$ are not universal constants$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 2.5 Fine-tuning implication

**Main idea.** Adapter rank can be viewed as an adaptation-capacity axis.

Core relation:

$$r$ controls update capacity$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
## 3. Data Scaling

This part studies data scaling as a forecasting tool. The formulas are useful only when you keep the measurement setup fixed and validate predictions against real runs.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Token-count law](#3-tokencount-law) | more unique useful tokens reduce loss | $L(D)=L_\infty + A_DD^{-\alpha_D}$ |
| [Data quality](#3-data-quality) | one high-quality token can be worth more than one noisy token | $D_\mathrm{eff}\ne D_\mathrm{raw}$ |
| [Repeated data](#3-repeated-data) | repetition gives diminishing returns and can increase memorization | $D_\mathrm{eff}<kD$ for repeats |
| [Domain match](#3-domain-match) | the validation distribution defines what improvement means | $p_\mathrm{train}\approx p_\mathrm{eval}$ |
| [Tokenizer dependence](#3-tokenizer-dependence) | token counts depend on the tokenizer | $D$ is measured in model tokens |

### 3.1 Token-count law

**Main idea.** More unique useful tokens reduce loss.

Core relation:

$$L(D)=L_\infty + A_DD^{-\alpha_D}$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 3.2 Data quality

**Main idea.** One high-quality token can be worth more than one noisy token.

Core relation:

$$D_\mathrm{eff}\ne D_\mathrm{raw}$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 3.3 Repeated data

**Main idea.** Repetition gives diminishing returns and can increase memorization.

Core relation:

$$D_\mathrm{eff}<kD$ for repeats$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 3.4 Domain match

**Main idea.** The validation distribution defines what improvement means.

Core relation:

$$p_\mathrm{train}\approx p_\mathrm{eval}$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 3.5 Tokenizer dependence

**Main idea.** Token counts depend on the tokenizer.

Core relation:

$$D$ is measured in model tokens$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
## 4. Compute Scaling

This part studies compute scaling as a forecasting tool. The formulas are useful only when you keep the measurement setup fixed and validate predictions against real runs.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Dense training FLOPs](#4-dense-training-flops) | a common first estimate for dense transformers is six times parameters times tokens | $C\approx 6ND$ |
| [Compute law](#4-compute-law) | loss can be modeled directly as a function of compute | $L(C)=L_\infty + A_CC^{-\alpha_C}$ |
| [IsoFLOP curves](#4-isoflop-curves) | fixed compute creates a tradeoff between model size and token count | $D=C/(6N)$ |
| [Compute is not wall-clock](#4-compute-is-not-wallclock) | hardware utilization and communication decide time | $T=C/\mathrm{achieved\ FLOPs\ per\ sec}$ |
| [Budget translation](#4-budget-translation) | money and time become constraints after mapping to achievable FLOPs | $\mathrm{cost}=T\cdot\mathrm{price}$ |

### 4.1 Dense training FLOPs

**Main idea.** A common first estimate for dense transformers is six times parameters times tokens.

Core relation:

$$C\approx 6ND$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 4.2 Compute law

**Main idea.** Loss can be modeled directly as a function of compute.

Core relation:

$$L(C)=L_\infty + A_CC^{-\alpha_C}$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 4.3 IsoFLOP curves

**Main idea.** Fixed compute creates a tradeoff between model size and token count.

Core relation:

$$D=C/(6N)$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** These curves are how you ask whether the next dollar should buy a larger model or more training tokens.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 4.4 Compute is not wall-clock

**Main idea.** Hardware utilization and communication decide time.

Core relation:

$$T=C/\mathrm{achieved\ FLOPs\ per\ sec}$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 4.5 Budget translation

**Main idea.** Money and time become constraints after mapping to achievable flops.

Core relation:

$$\mathrm{cost}=T\cdot\mathrm{price}$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
## 5. Compute-Optimal Allocation

This part studies compute-optimal allocation as a forecasting tool. The formulas are useful only when you keep the measurement setup fixed and validate predictions against real runs.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Two-term loss proxy](#5-twoterm-loss-proxy) | separate parameter-limited and data-limited terms | $L(N,D)=L_\infty + A/N^\alpha + B/D^\beta$ |
| [Constraint](#5-constraint) | training compute couples N and D | $C=6ND$ |
| [Balanced marginal returns](#5-balanced-marginal-returns) | at optimum, extra compute should help similarly through N and D | $\partial L/\partial N$ balances $\partial L/\partial D$ under the constraint |
| [Chinchilla intuition](#5-chinchilla-intuition) | many earlier large models were undertrained relative to their size | $D/N$ was too small |
| [Practical rule](#5-practical-rule) | choose N and D together, not one after the other | $(N^\star,D^\star)=\arg\min_{6ND=C}L(N,D)$ |

### 5.1 Two-term loss proxy

**Main idea.** Separate parameter-limited and data-limited terms.

Core relation:

$$L(N,D)=L_\infty + A/N^\alpha + B/D^\beta$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 5.2 Constraint

**Main idea.** Training compute couples n and d.

Core relation:

$$C=6ND$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 5.3 Balanced marginal returns

**Main idea.** At optimum, extra compute should help similarly through n and d.

Core relation:

$$\partial L/\partial N$ balances $\partial L/\partial D$ under the constraint$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 5.4 Chinchilla intuition

**Main idea.** Many earlier large models were undertrained relative to their size.

Core relation:

$$D/N$ was too small$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 5.5 Practical rule

**Main idea.** Choose n and d together, not one after the other.

Core relation:

$$(N^\star,D^\star)=\arg\min_{6ND=C}L(N,D)$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
## 6. Kaplan and Chinchilla Lessons

This part studies kaplan and chinchilla lessons as a forecasting tool. The formulas are useful only when you keep the measurement setup fixed and validate predictions against real runs.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Kaplan-style result](#6-kaplanstyle-result) | smooth power laws made loss forecasting practical | $L$ follows power laws across scale ranges |
| [Chinchilla revision](#6-chinchilla-revision) | more tokens per parameter can be compute-optimal than earlier practice suggested | $D^\star/N^\star$ increases |
| [Undertraining diagnostic](#6-undertraining-diagnostic) | a model with low tokens per parameter may benefit more from data than size | $D/N\ll\mathrm{target}$ |
| [Overtraining for inference](#6-overtraining-for-inference) | training smaller models longer can reduce serving cost | $\mathrm{train\ cost}+\mathrm{inference\ cost}$ |
| [Do not cargo-cult ratios](#6-do-not-cargocult-ratios) | ratios depend on data, architecture, and objective | $D/N$ is a planning prior, not a theorem |

### 6.1 Kaplan-style result

**Main idea.** Smooth power laws made loss forecasting practical.

Core relation:

$$L$ follows power laws across scale ranges$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 6.2 Chinchilla revision

**Main idea.** More tokens per parameter can be compute-optimal than earlier practice suggested.

Core relation:

$$D^\star/N^\star$ increases$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** The practical lesson is that undertrained large models can waste compute even when they look impressive.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 6.3 Undertraining diagnostic

**Main idea.** A model with low tokens per parameter may benefit more from data than size.

Core relation:

$$D/N\ll\mathrm{target}$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 6.4 Overtraining for inference

**Main idea.** Training smaller models longer can reduce serving cost.

Core relation:

$$\mathrm{train\ cost}+\mathrm{inference\ cost}$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 6.5 Do not cargo-cult ratios

**Main idea.** Ratios depend on data, architecture, and objective.

Core relation:

$$D/N$ is a planning prior, not a theorem$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
## 7. Fitting Scaling Laws

This part studies fitting scaling laws as a forecasting tool. The formulas are useful only when you keep the measurement setup fixed and validate predictions against real runs.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Log transformation](#7-log-transformation) | if the floor is known, power-law fitting becomes linear | $\log(L-L_\infty)=\log A-\alpha\log X$ |
| [Unknown floor](#7-unknown-floor) | the irreducible loss floor must be fit or bounded carefully | $L_\infty$ |
| [Holdout validation](#7-holdout-validation) | reserve some runs to test forecasts | $L_\mathrm{actual}-L_\mathrm{pred}$ |
| [Residual analysis](#7-residual-analysis) | curved residuals mean the law is missing a factor | $\epsilon_i=L_i-\hat L_i$ |
| [Extrapolation risk](#7-extrapolation-risk) | small-run fits can fail when regime changes | $X_\mathrm{test}\gg X_\mathrm{fit}$ |

### 7.1 Log transformation

**Main idea.** If the floor is known, power-law fitting becomes linear.

Core relation:

$$\log(L-L_\infty)=\log A-\alpha\log X$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 7.2 Unknown floor

**Main idea.** The irreducible loss floor must be fit or bounded carefully.

Core relation:

$$L_\infty$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** A small error in the loss floor can bend exponent estimates and make forecasts overconfident.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 7.3 Holdout validation

**Main idea.** Reserve some runs to test forecasts.

Core relation:

$$L_\mathrm{actual}-L_\mathrm{pred}$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 7.4 Residual analysis

**Main idea.** Curved residuals mean the law is missing a factor.

Core relation:

$$\epsilon_i=L_i-\hat L_i$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 7.5 Extrapolation risk

**Main idea.** Small-run fits can fail when regime changes.

Core relation:

$$X_\mathrm{test}\gg X_\mathrm{fit}$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
## 8. Data Quality and Mixtures

This part studies data quality and mixtures as a forecasting tool. The formulas are useful only when you keep the measurement setup fixed and validate predictions against real runs.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Mixture weights](#8-mixture-weights) | training data is a weighted mixture of sources | $p_\mathrm{train}=\sum_k w_kp_k$ |
| [Effective tokens](#8-effective-tokens) | quality, deduplication, and domain match alter token value | $D_\mathrm{eff}=\sum_k q_kw_kD$ |
| [Contamination](#8-contamination) | validation leakage makes scaling look better than it is | $D_\mathrm{eval}\cap D_\mathrm{train}\ne\emptyset$ |
| [Curriculum and filtering](#8-curriculum-and-filtering) | data order and filtering can shift constants | $A$ changes even if $\alpha$ is similar |
| [Evaluation distribution](#8-evaluation-distribution) | optimize for the distribution that matters | $L_\mathrm{eval}$, not only $L_\mathrm{web}$ |

### 8.1 Mixture weights

**Main idea.** Training data is a weighted mixture of sources.

Core relation:

$$p_\mathrm{train}=\sum_k w_kp_k$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 8.2 Effective tokens

**Main idea.** Quality, deduplication, and domain match alter token value.

Core relation:

$$D_\mathrm{eff}=\sum_k q_kw_kD$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** Data quantity without data value is a weak planning variable.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 8.3 Contamination

**Main idea.** Validation leakage makes scaling look better than it is.

Core relation:

$$D_\mathrm{eval}\cap D_\mathrm{train}\ne\emptyset$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 8.4 Curriculum and filtering

**Main idea.** Data order and filtering can shift constants.

Core relation:

$$A$ changes even if $\alpha$ is similar$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 8.5 Evaluation distribution

**Main idea.** Optimize for the distribution that matters.

Core relation:

$$L_\mathrm{eval}$, not only $L_\mathrm{web}$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
## 9. Inference-Aware Scaling

This part studies inference-aware scaling as a forecasting tool. The formulas are useful only when you keep the measurement setup fixed and validate predictions against real runs.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Serving cost](#9-serving-cost) | a model trained once may be served many times | $C_\mathrm{serve}\propto N\cdot T_\mathrm{generated}\cdot Q$ |
| [Latency constraint](#9-latency-constraint) | larger models can be better but too slow | $T_\mathrm{latency}\le T_\mathrm{SLA}$ |
| [Small overtrained models](#9-small-overtrained-models) | more training tokens can improve a smaller cheaper model | $D/N$ can exceed pretraining-optimal planning ratios |
| [Distillation](#9-distillation) | teacher quality can transfer into a smaller student | $p_s \approx p_t$ |
| [Total cost objective](#9-total-cost-objective) | choose models by train plus serve plus quality | $J=\mathrm{loss}+\lambda\mathrm{cost}$ |

### 9.1 Serving cost

**Main idea.** A model trained once may be served many times.

Core relation:

$$C_\mathrm{serve}\propto N\cdot T_\mathrm{generated}\cdot Q$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** The best training run is not always the cheapest product model.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 9.2 Latency constraint

**Main idea.** Larger models can be better but too slow.

Core relation:

$$T_\mathrm{latency}\le T_\mathrm{SLA}$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 9.3 Small overtrained models

**Main idea.** More training tokens can improve a smaller cheaper model.

Core relation:

$$D/N$ can exceed pretraining-optimal planning ratios$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 9.4 Distillation

**Main idea.** Teacher quality can transfer into a smaller student.

Core relation:

$$p_s \approx p_t$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 9.5 Total cost objective

**Main idea.** Choose models by train plus serve plus quality.

Core relation:

$$J=\mathrm{loss}+\lambda\mathrm{cost}$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
## 10. Limits and Debugging

This part studies limits and debugging as a forecasting tool. The formulas are useful only when you keep the measurement setup fixed and validate predictions against real runs.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Not a theorem of all AI](#10-not-a-theorem-of-all-ai) | scaling laws are fitted empirical regularities | $\mathrm{valid\ only\ in\ regime}$ |
| [Metric artifacts](#10-metric-artifacts) | thresholded task scores can look sudden even when loss changes smoothly | $\mathbf{1}\{s>\tau\}$ |
| [Architecture changes](#10-architecture-changes) | new attention, MoE, or tokenizer choices can change constants | $A,\alpha,L_\infty$ shift |
| [Bad small runs](#10-bad-small-runs) | bugs in small runs poison the forecast | $\hat L$ inherits measurement error |
| [Decision checklist](#10-decision-checklist) | use scaling laws as planning instruments with uncertainty | $\hat L\pm\mathrm{interval}$ |

### 10.1 Not a theorem of all AI

**Main idea.** Scaling laws are fitted empirical regularities.

Core relation:

$$\mathrm{valid\ only\ in\ regime}$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 10.2 Metric artifacts

**Main idea.** Thresholded task scores can look sudden even when loss changes smoothly.

Core relation:

$$\mathbf{1}\{s>\tau\}$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 10.3 Architecture changes

**Main idea.** New attention, moe, or tokenizer choices can change constants.

Core relation:

$$A,\alpha,L_\infty$ shift$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 10.4 Bad small runs

**Main idea.** Bugs in small runs poison the forecast.

Core relation:

$$\hat L$ inherits measurement error$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 10.5 Decision checklist

**Main idea.** Use scaling laws as planning instruments with uncertainty.

Core relation:

$$\hat L\pm\mathrm{interval}$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.

---

## Practice Exercises

1. Fit a one-resource power law after subtracting a known loss floor.
2. Convert parameters and tokens into approximate dense training FLOPs.
3. Compute token count from a fixed compute budget and model size.
4. Search an IsoFLOP curve for the lowest toy loss.
5. Diagnose whether a model is undertrained by checking tokens per parameter.
6. Compute effective tokens for a data mixture.
7. Estimate serving cost for two model choices.
8. Compute residuals for scaling-law forecasts.
9. Show how a thresholded metric can look abrupt even when loss is smooth.
10. Write a forecast checklist with uncertainty.

## Why This Matters for AI

Scaling laws turn expensive guessing into disciplined planning. They help decide whether to collect more data, train longer, use a smaller model, improve data quality, or avoid a run whose expected gain is too small. They also teach humility: a clean line on a log-log plot is useful, but it is still a fit to a regime, not a theorem about intelligence.

## Bridge to Efficient Attention and Inference

The next section studies how attention and inference systems change the cost side of the scaling equation. Scaling laws say what loss might be worth buying; efficient attention and serving math decide whether that loss can be bought and served under memory, latency, and throughput constraints.

## References

- Jared Kaplan et al., "Scaling Laws for Neural Language Models", 2020: https://arxiv.org/abs/2001.08361
- Jordan Hoffmann et al., "Training Compute-Optimal Large Language Models", 2022: https://arxiv.org/abs/2203.15556
- Tom Henighan et al., "Scaling Laws for Autoregressive Generative Modeling", 2020: https://arxiv.org/abs/2010.14701
- Ethan Caballero et al., "Broken Neural Scaling Laws", 2022: https://arxiv.org/abs/2210.14891
- Sam McCandlish et al., "An Empirical Model of Large-Batch Training", 2018: https://arxiv.org/abs/1812.06162
- Yonatan Bisk et al., "Experience Grounds Language", 2020, for caution about benchmarks and grounding limits: https://arxiv.org/abs/2004.10151
