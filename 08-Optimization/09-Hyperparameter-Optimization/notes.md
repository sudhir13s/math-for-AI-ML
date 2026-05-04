[← Back to Optimization](../README.md) | [Previous: Regularization Methods ←](../08-Regularization-Methods/notes.md) | [Next: Learning Rate Schedules →](../10-Learning-Rate-Schedules/notes.md)

---

# Hyperparameter Optimization

> _"A training run is an experiment; hyperparameter optimization is the mathematics of spending experiments wisely."_

## Overview

Hyperparameter optimization is the outer loop around learning.
The inner loop updates model parameters $\boldsymbol{\theta}$ by minimizing a loss.
The outer loop chooses the configuration $\boldsymbol{\lambda}$ that defines how the inner loop behaves:
learning rate, weight decay, batch size, optimizer, architecture width, dropout rate, data augmentation strength,
LoRA rank, number of training steps, early-stopping patience, and many other choices that are not directly learned by backpropagation.

This section treats hyperparameter optimization as noisy, budgeted, derivative-free optimization over a structured configuration space.
The validation score is expensive because each evaluation may require training a model.
It is noisy because initialization, data order, hardware nondeterminism, and finite validation sets perturb the measured result.
It is constrained because compute, memory, wall-clock time, and safety requirements matter.
The central question is not "which tuner is fashionable?"
The central question is:

$$
\text{How should we allocate a limited experiment budget to find a configuration that generalizes?}
$$

This section is the canonical home for grid search, random search, Bayesian optimization, successive halving, Hyperband,
BOHB, population-based training, statistical selection bias, and practical tuning workflows.
It does not re-teach optimizer internals from [Adaptive Learning Rate](../07-Adaptive-Learning-Rate/notes.md),
regularization theory from [Regularization Methods](../08-Regularization-Methods/notes.md),
or schedule formulas from [Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md).
Those topics appear only as hyperparameters to be selected, evaluated, and reported.

## Prerequisites

- **Gradient descent and stochastic optimization** - training loss, validation loss, gradient noise, mini-batch training - [Gradient Descent](../02-Gradient-Descent/notes.md), [Stochastic Optimization](../05-Stochastic-Optimization/notes.md)
- **Adaptive optimizers** - AdamW, effective learning rate, optimizer state, weight decay - [Adaptive Learning Rate](../07-Adaptive-Learning-Rate/notes.md)
- **Regularization and early stopping** - model complexity controls and validation-based stopping - [Regularization Methods](../08-Regularization-Methods/notes.md)
- **Bayesian inference** - posterior uncertainty and probabilistic modeling - [Bayesian Inference](../../07-Statistics/04-Bayesian-Inference/notes.md)
- **Gaussian processes and kernels** - surrogate modeling intuition for Bayesian optimization - [Kernel Methods](../../12-Functional-Analysis/03-Kernel-Methods/notes.md)
- **Cross-validation and evaluation** - validation sets, test sets, variance, and selection bias - [Statistics](../../07-Statistics/README.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Interactive simulations of search spaces, grid/random search, Bayesian optimization, successive halving, Hyperband, PBT, leakage, and Pareto tuning |
| [exercises.ipynb](exercises.ipynb) | 10 graded exercises covering HPO mechanics, validation noise, search algorithms, bandit allocation, bias, and LLM fine-tuning budgets |

## Learning Objectives

After completing this section, you will:

1. Define hyperparameters, configuration spaces, studies, trials, objectives, budgets, resources, and fidelities precisely.
2. Distinguish model parameters $\boldsymbol{\theta}$ from hyperparameters $\boldsymbol{\lambda}$ and explain why the latter usually need an outer optimization loop.
3. Construct search spaces with categorical, ordinal, continuous, log-scaled, and conditional variables.
4. Explain why grid search becomes inefficient when only a few hyperparameters matter.
5. Derive why random search is a strong baseline for high-dimensional, low-effective-dimensional tuning problems.
6. Implement simple surrogate-based Bayesian optimization with a probabilistic response model and acquisition function.
7. Explain expected improvement, probability of improvement, upper confidence bound, and Thompson-style acquisition policies.
8. Simulate successive halving, Hyperband, and asynchronous resource allocation for multi-fidelity model training.
9. Describe population-based training as joint optimization of weights and time-varying hyperparameters.
10. Detect validation leakage, nested cross-validation mistakes, and selection bias caused by repeated tuning.
11. Design fair HPO experiments with fixed budgets, repeated seeds, logging, and multi-metric reporting.
12. Choose a practical tuning strategy for classical ML, deep learning, and LLM fine-tuning under compute constraints.

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 Hyperparameters as the Outer Optimization Loop](#11-hyperparameters-as-the-outer-optimization-loop)
  - [1.2 Why Manual Tuning and Grid Intuition Fail](#12-why-manual-tuning-and-grid-intuition-fail)
  - [1.3 Expensive, Noisy Validation Objectives](#13-expensive-noisy-validation-objectives)
  - [1.4 Budgets, Fidelities, and Early Signals](#14-budgets-fidelities-and-early-signals)
  - [1.5 HPO Algorithm Families for ML and LLMs](#15-hpo-algorithm-families-for-ml-and-llms)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Configuration Spaces, Trials, Studies, and Objectives](#21-configuration-spaces-trials-studies-and-objectives)
  - [2.2 Search Distributions and Conditional Structure](#22-search-distributions-and-conditional-structure)
  - [2.3 Validation Score as a Noisy Estimator](#23-validation-score-as-a-noisy-estimator)
  - [2.4 Budget, Resource, Fidelity, and Cost](#24-budget-resource-fidelity-and-cost)
  - [2.5 Constraints, Multi-Objective Metrics, and Reproducibility](#25-constraints-multi-objective-metrics-and-reproducibility)
- [3. Baseline Search Methods](#3-baseline-search-methods)
  - [3.1 Manual Search and Ablation Discipline](#31-manual-search-and-ablation-discipline)
  - [3.2 Grid Search and the Curse of Dimensionality](#32-grid-search-and-the-curse-of-dimensionality)
  - [3.3 Random Search and Effective Dimension](#33-random-search-and-effective-dimension)
  - [3.4 Quasi-Random and Space-Filling Designs](#34-quasi-random-and-space-filling-designs)
  - [3.5 Choosing Defensible Baselines](#35-choosing-defensible-baselines)
- [4. Bayesian and Surrogate-Based Optimization](#4-bayesian-and-surrogate-based-optimization)
  - [4.1 Sequential Model-Based Optimization](#41-sequential-model-based-optimization)
  - [4.2 Gaussian-Process Surrogate Intuition](#42-gaussian-process-surrogate-intuition)
  - [4.3 Acquisition Functions: PI, EI, UCB, and Thompson Sampling](#43-acquisition-functions-pi-ei-ucb-and-thompson-sampling)
  - [4.4 Tree-Structured Parzen Estimators](#44-tree-structured-parzen-estimators)
  - [4.5 Noisy, Constrained, Cost-Aware, and Multi-Objective BO](#45-noisy-constrained-cost-aware-and-multi-objective-bo)
- [5. Multi-Fidelity and Bandit Methods](#5-multi-fidelity-and-bandit-methods)
  - [5.1 Resource Allocation as Optimization](#51-resource-allocation-as-optimization)
  - [5.2 Successive Halving](#52-successive-halving)
  - [5.3 Hyperband](#53-hyperband)
  - [5.4 ASHA and Asynchronous Distributed Tuning](#54-asha-and-asynchronous-distributed-tuning)
  - [5.5 BOHB: Bayesian Optimization plus Hyperband](#55-bohb-bayesian-optimization-plus-hyperband)
- [6. Population and Dynamic Hyperparameters](#6-population-and-dynamic-hyperparameters)
  - [6.1 Evolutionary and Population Search](#61-evolutionary-and-population-search)
  - [6.2 Population-Based Training](#62-population-based-training)
  - [6.3 Fixed Hyperparameters versus Learned Schedules](#63-fixed-hyperparameters-versus-learned-schedules)
  - [6.4 Architecture and Augmentation Policy Search](#64-architecture-and-augmentation-policy-search)
  - [6.5 Transfer HPO, Warm Starts, and Meta-Learning](#65-transfer-hpo-warm-starts-and-meta-learning)
- [7. Statistical Reliability and Selection Bias](#7-statistical-reliability-and-selection-bias)
  - [7.1 Train, Validation, and Test Discipline](#71-train-validation-and-test-discipline)
  - [7.2 Nested Cross-Validation and Model-Selection Bias](#72-nested-cross-validation-and-model-selection-bias)
  - [7.3 Multiple Comparisons, Seeds, and Confidence Intervals](#73-multiple-comparisons-seeds-and-confidence-intervals)
  - [7.4 Multi-Metric and Pareto Reporting](#74-multi-metric-and-pareto-reporting)
  - [7.5 Reproducible HPO Reports](#75-reproducible-hpo-reports)
- [8. Applications in ML and LLM Systems](#8-applications-in-ml-and-llm-systems)
  - [8.1 Classical ML Pipelines](#81-classical-ml-pipelines)
  - [8.2 Deep Learning Knobs](#82-deep-learning-knobs)
  - [8.3 LLM Fine-Tuning Knobs](#83-llm-fine-tuning-knobs)
  - [8.4 Practical Tooling](#84-practical-tooling)
  - [8.5 When Not to Tune](#85-when-not-to-tune)
- [9. Common Mistakes](#9-common-mistakes)
- [10. Exercises](#10-exercises)
- [11. Why This Matters for AI](#11-why-this-matters-for-ai)
- [12. Conceptual Bridge](#12-conceptual-bridge)
- [References](#references)

---

## 1. Intuition

### 1.1 Hyperparameters as the Outer Optimization Loop

Training a model usually solves an inner optimization problem:

$$
\boldsymbol{\theta}^{*}(\boldsymbol{\lambda})
\in
\arg\min_{\boldsymbol{\theta}}
\mathcal{L}_{\mathrm{train}}(\boldsymbol{\theta}; \boldsymbol{\lambda}).
$$

Here $\boldsymbol{\theta}$ contains learned parameters such as weights, biases, embedding vectors, LoRA matrices, or classifier coefficients.
The vector $\boldsymbol{\lambda}$ contains choices that shape the training process but are not usually learned by gradient descent.
Examples include:

| Hyperparameter | Meaning | Typical scale |
| --- | --- | --- |
| $\eta$ | Base learning rate | log scale |
| $\lambda_{\mathrm{wd}}$ | Weight decay coefficient | log scale |
| $B$ | Batch size | powers of two or hardware-constrained integers |
| $p_{\mathrm{drop}}$ | Dropout probability | bounded continuous |
| $r_{\mathrm{LoRA}}$ | LoRA rank | small integer |
| $\tau$ | Contrastive temperature | positive continuous |
| $T$ | Number of training steps | resource variable |
| optimizer | SGD, AdamW, Adafactor, etc. | categorical |

The outer problem evaluates the trained model on validation data:

$$
\boldsymbol{\lambda}^{*}
\in
\arg\min_{\boldsymbol{\lambda} \in \mathcal{X}}
F(\boldsymbol{\lambda}),
\qquad
F(\boldsymbol{\lambda})
=
\mathcal{L}_{\mathrm{val}}(\boldsymbol{\theta}^{*}(\boldsymbol{\lambda}); \boldsymbol{\lambda}).
$$

In real systems, we do not observe $F(\boldsymbol{\lambda})$ exactly.
We observe a noisy trial result:

$$
Y(\boldsymbol{\lambda})
=
F(\boldsymbol{\lambda})
+ \varepsilon,
\qquad
\mathbb{E}[\varepsilon \mid \boldsymbol{\lambda}] = 0.
$$

This one equation explains why HPO is different from ordinary calculus-based optimization:

1. Each evaluation of $Y(\boldsymbol{\lambda})$ may require minutes, hours, or days.
2. The gradient $\nabla_{\boldsymbol{\lambda}} F$ is usually unavailable.
3. The objective is noisy.
4. The search space mixes discrete, continuous, conditional, and constrained variables.
5. Good performance is measured on validation or deployment metrics, not training loss alone.

The outer-loop picture is:

```text
configuration lambda
        |
        v
train model parameters theta
        |
        v
measure validation score
        |
        v
choose next lambda
```

Hyperparameter optimization is the design of the final arrow.

**Example: learning rate.**
The learning rate $\eta$ is not a parameter of the model.
It is a parameter of the training process.
Too small and training wastes compute.
Too large and training diverges.
The HPO problem is to find a useful region for $\eta$ while not spending the whole project on trial runs.

**Example: LoRA fine-tuning.**
For a fixed base model, the learned parameters are adapter weights.
The hyperparameters include rank $r$, scaling $\alpha$, adapter target modules, learning rate, batch size, weight decay, number of epochs, and data mixture.
A good configuration can be more important than a clever new adapter trick.

**Example: retrieval-augmented generation.**
The learned model may be frozen.
The hyperparameters are chunk size, overlap, embedding model, retrieval depth $k$, reranker choice, prompt template, and score threshold.
The same mathematical issue appears: evaluate a costly configuration under noisy validation metrics.

### 1.2 Why Manual Tuning and Grid Intuition Fail

Manual tuning is useful when it encodes expert knowledge.
It becomes dangerous when it silently turns into unlogged, biased search.
A human tries a few settings, remembers the best run, forgets failed runs, and reports a test score.
That is not a controlled experiment.
It is a selection process with missing data.

Grid search looks principled:

$$
\mathcal{G}
=
\mathcal{G}_1 \times \mathcal{G}_2 \times \cdots \times \mathcal{G}_d.
$$

If each of $d$ hyperparameters has $m$ candidate values, the full grid has:

$$
\lvert \mathcal{G} \rvert = m^d
$$

trials.
This is already expensive.
But the deeper problem is not only the number of trials.
The deeper problem is that a grid spends equal resolution on every coordinate.

Suppose only two hyperparameters matter:

$$
F(\lambda_1,\ldots,\lambda_d)
\approx
f(\lambda_2,\lambda_7).
$$

A grid with $m=3$ and $d=10$ uses $3^{10}=59049$ total points,
but only $3^2=9$ distinct settings on the two important coordinates.
Most trials differ only in irrelevant coordinates.
Random search, by contrast, samples many distinct values along every coordinate.
Bergstra and Bengio's random-search result is built around this effective-dimension phenomenon.

The grid failure mode is easiest to see in a two-dimensional slice:

```text
grid search:

  x     x     x     x

  x     x     x     x

  x     x     x     x

  x     x     x     x

random search:

    x      x
 x       x    x
      x
   x        x
        x
```

The grid is tidy but rigid.
Random search is untidy but explores more unique coordinate values.
When the important scale is logarithmic, a linear grid can be worse:

$$
\eta \in \{0.001, 0.251, 0.501, 0.751, 1.0\}
$$

is a poor learning-rate grid for neural networks.
A log-scale search such as

$$
\log_{10}\eta \sim \mathcal{U}(-5,-2)
$$

is usually more sensible.

**Manual tuning is not forbidden.**
It is often the first stage.
The rule is:

$$
\text{manual tuning should define the search space, not replace measurement.}
$$

### 1.3 Expensive, Noisy Validation Objectives

If every trial returned the exact value $F(\boldsymbol{\lambda})$, HPO would be hard but clean.
In practice, a trial returns:

$$
Y_j
=
F(\boldsymbol{\lambda}_j)
+ \varepsilon_j.
$$

The noise $\varepsilon_j$ comes from many sources:

| Noise source | Example | Consequence |
| --- | --- | --- |
| Random initialization | different initial weights | one lucky seed can win |
| Data order | shuffled batches | early dynamics vary |
| Dropout and augmentation | stochastic transforms | validation score varies |
| Hardware nondeterminism | GPU kernels | tiny numerical differences can amplify |
| Finite validation data | limited sample size | metric has sampling variance |
| Distributed training | asynchronous timing | effective order differs |

For a validation set of size $n_{\mathrm{val}}$, a simple accuracy estimate has approximate standard error:

$$
\operatorname{SE}(\hat{p})
\approx
\sqrt{\frac{\hat{p}(1-\hat{p})}{n_{\mathrm{val}}}}.
$$

Two configurations whose validation accuracies differ by less than this scale may not be meaningfully different.
For loss metrics, the same idea appears through variance of per-example loss:

$$
\operatorname{SE}(\bar{\ell}_{\mathrm{val}})
=
\sqrt{\frac{1}{n_{\mathrm{val}}}
\operatorname{Var}(\ell(\mathbf{x},y))}.
$$

This matters because HPO optimizes over many noisy measurements.
The best observed trial can be best partly because of favorable noise.
If $N$ independent configurations have equal true performance but noisy observed scores, then the minimum observed score tends to look better as $N$ grows.
This is selection bias.

**For AI systems:** leaderboard chasing, prompt tuning, RAG threshold tuning, and LLM fine-tuning can all overfit validation sets.
The outer loop is still an optimizer.
Anything optimized repeatedly against a finite validation metric can overfit that metric.

### 1.4 Budgets, Fidelities, and Early Signals

A full training run is not the only possible evaluation.
We can evaluate a configuration at a smaller resource level:

$$
y(\boldsymbol{\lambda}, r),
$$

where $r$ might be:

- number of epochs,
- number of training steps,
- number of data samples,
- fraction of dataset,
- model width,
- image resolution,
- sequence length,
- number of trees,
- number of RL environment frames.

The resource $r$ defines the fidelity.
Low fidelity is cheap but biased.
High fidelity is expensive but closer to the target.

Multi-fidelity HPO asks:

$$
\text{Which configurations deserve more resource?}
$$

This is the core idea behind successive halving and Hyperband.
Start many configurations cheaply.
Discard weak ones.
Allocate more budget to survivors.

```text
many cheap trials
  lambda_1 lambda_2 lambda_3 lambda_4 lambda_5 lambda_6 lambda_7 lambda_8
      |        |        |        |        |        |        |        |
      v        v        v        v        v        v        v        v
   score    score    score    score    score    score    score    score
      \        /        \        /        \        /        \        /
       keep best half             keep best half
              \                       /
               v                     v
             train longer        train longer
                    \             /
                     v           v
                     final comparison
```

The key assumption is rank correlation:

$$
\operatorname{corr}(y(\boldsymbol{\lambda}, r_{\mathrm{low}}),
y(\boldsymbol{\lambda}, r_{\mathrm{high}})) > 0.
$$

If early validation scores are completely uninformative, early stopping kills good configurations.
If early validation scores are strongly predictive, multi-fidelity methods save enormous compute.

### 1.5 HPO Algorithm Families for ML and LLMs

The main HPO families differ in what information they use.

| Family | Uses past scores? | Uses resource levels? | Handles dynamic hyperparameters? | Typical use |
| --- | --- | --- | --- | --- |
| Manual search | human memory | maybe | maybe | first pass, debugging |
| Grid search | no | no | no | tiny categorical spaces |
| Random search | no | no | no | strong baseline |
| Quasi-random design | no | no | no | space-filling initial design |
| Bayesian optimization | yes | usually no | no | expensive black-box objective |
| Successive halving | yes, via ranking | yes | no | iterative training |
| Hyperband / ASHA | yes, via ranking | yes | no | parallel deep-learning tuning |
| BOHB | yes | yes | no | guided multi-fidelity HPO |
| Evolutionary search | yes | maybe | yes | architecture and policy search |
| Population-based training | yes | yes | yes | schedules, RL, non-stationary training |

For small tabular models, grid and random search may be enough.
For medium neural networks, random search plus Hyperband-style early stopping is often strong.
For expensive training runs with a small continuous search space, Bayesian optimization can be sample-efficient.
For long-running distributed training where schedules matter, population-based methods can adapt hyperparameters during training.

The correct choice is not universal.
It depends on the objective cost, noise level, search-space dimension, parallelism, budget, and whether partial learning curves are informative.

---

## 2. Formal Definitions

### 2.1 Configuration Spaces, Trials, Studies, and Objectives

A **configuration space** is the set $\mathcal{X}$ of allowed hyperparameter configurations.
A configuration is a point:

$$
\boldsymbol{\lambda}
=
(\lambda_1,\ldots,\lambda_d)
\in
\mathcal{X}.
$$

The components may have different types:

$$
\mathcal{X}
=
\mathcal{X}_1 \times \cdots \times \mathcal{X}_d
$$

when independent, or a structured tree when conditional.

**Examples.**

1. Logistic regression:

$$
\boldsymbol{\lambda}
=
(C,\text{penalty},\text{class weight}).
$$

2. Transformer fine-tuning:

$$
\boldsymbol{\lambda}
=
(\eta,\lambda_{\mathrm{wd}},B,T,r_{\mathrm{LoRA}},\alpha_{\mathrm{LoRA}},p_{\mathrm{drop}}).
$$

3. RAG pipeline:

$$
\boldsymbol{\lambda}
=
(\text{chunk size},\text{overlap},k,\text{embedding model},\text{reranker},\tau_{\mathrm{score}}).
$$

**Non-examples.**

1. A weight matrix $W$ learned by backpropagation is not a hyperparameter in ordinary training.
2. A learned embedding vector $\mathbf{e}_i$ is not a hyperparameter.
3. A batch's sampled gradient $\mathbf{g}_t$ is not a hyperparameter.
4. A validation loss value is not a hyperparameter; it is an observation.

A **trial** is one evaluation of a configuration.
It usually consists of:

1. choose $\boldsymbol{\lambda}_j$,
2. train model parameters $\boldsymbol{\theta}$ under $\boldsymbol{\lambda}_j$,
3. evaluate a metric $Y_j$,
4. log artifacts, cost, seed, and status.

A **study** is the whole sequence:

$$
\mathcal{S}_N
=
\{(\boldsymbol{\lambda}_j,Y_j,c_j,s_j)\}_{j=1}^{N},
$$

where $c_j$ is cost and $s_j$ is status or metadata.

The **objective function** is the unknown mapping:

$$
F:\mathcal{X} \to \mathbb{R}.
$$

For minimization:

$$
\boldsymbol{\lambda}^{*}
\in
\arg\min_{\boldsymbol{\lambda} \in \mathcal{X}} F(\boldsymbol{\lambda}).
$$

For maximization of score $A(\boldsymbol{\lambda})$, convert by minimizing:

$$
F(\boldsymbol{\lambda}) = -A(\boldsymbol{\lambda}).
$$

In practice, $F$ is not directly available.
We observe $Y$:

$$
Y \mid \boldsymbol{\lambda}
\sim
p(y \mid \boldsymbol{\lambda}).
$$

This statistical view is essential.
HPO is not merely deterministic search over a table.
It is decision-making under noisy observations.

### 2.2 Search Distributions and Conditional Structure

A search algorithm needs a way to propose configurations.
Random search defines a distribution $q$ over $\mathcal{X}$:

$$
\boldsymbol{\lambda}_j \sim q(\boldsymbol{\lambda}).
$$

The distribution should respect the geometry of each hyperparameter.

| Hyperparameter type | Example | Sensible distribution |
| --- | --- | --- |
| categorical | optimizer | uniform or weighted categorical |
| ordinal | number of layers | discrete ordered set |
| integer | tree depth | integer interval |
| positive scale | learning rate | log-uniform |
| bounded probability | dropout rate | uniform or beta distribution |
| constrained pair | warmup and total steps | conditional or transformed |
| architecture branch | model family | tree-structured conditional |

For a learning rate, log-uniform means:

$$
\log \eta \sim \mathcal{U}(\log a,\log b).
$$

Equivalently, the density on $\eta$ is:

$$
p(\eta)
=
\frac{1}{\eta(\log b - \log a)},
\qquad
a \le \eta \le b.
$$

This gives equal probability mass to multiplicative intervals:

$$
P(10^{-5} \le \eta \le 10^{-4})
=
P(10^{-4} \le \eta \le 10^{-3}).
$$

That is usually what we want for learning rates and regularization coefficients.

Conditional spaces arise when one choice activates other choices.
For example:

```text
optimizer
  AdamW
    beta_1
    beta_2
    epsilon
  SGD
    momentum
    nesterov
  Adafactor
    relative_step
    factored
```

Mathematically, a tree-structured space can be written as:

$$
\mathcal{X}
=
\bigcup_{k=1}^{K}
\{k\} \times \mathcal{X}^{(k)}.
$$

Each branch $\mathcal{X}^{(k)}$ has its own active hyperparameters.
Treating inactive variables as ordinary numeric dimensions can confuse surrogate models.

**Example: LoRA target modules.**
If `use_lora = false`, then `rank`, `alpha`, and `target_modules` are inactive.
They should not be sampled as if they affect the objective.

**Example: augmentation.**
If `augmentation = none`, then `crop_scale`, `mixup_alpha`, and `color_jitter_strength` are inactive.

**Non-example: post-hoc metric threshold.**
If the threshold is tuned after training on a validation set, it is still a hyperparameter of the decision rule.
It must be logged and included in model selection.

### 2.3 Validation Score as a Noisy Estimator

Let the target deployment risk be:

$$
R(\boldsymbol{\lambda})
=
\mathbb{E}_{(\mathbf{x},y) \sim p_{\mathrm{deploy}}}
\left[
\ell(f_{\boldsymbol{\theta}^{*}(\boldsymbol{\lambda})}(\mathbf{x}),y)
\right].
$$

We cannot compute this exactly.
We estimate it on validation data:

$$
\hat{R}_{\mathrm{val}}(\boldsymbol{\lambda})
=
\frac{1}{n_{\mathrm{val}}}
\sum_{i=1}^{n_{\mathrm{val}}}
\ell(f_{\boldsymbol{\theta}^{*}(\boldsymbol{\lambda})}(\mathbf{x}^{(i)}),y^{(i)}).
$$

This estimate has at least two kinds of error:

$$
\hat{R}_{\mathrm{val}}(\boldsymbol{\lambda})
=
R(\boldsymbol{\lambda})
+ b(\boldsymbol{\lambda})
+ \varepsilon(\boldsymbol{\lambda}).
$$

Here $b(\boldsymbol{\lambda})$ is systematic bias from validation/deployment mismatch,
and $\varepsilon(\boldsymbol{\lambda})$ is random estimation noise.

If the validation set is representative, $b$ may be small.
If the validation set is stale, leaked, too easy, or off-distribution, $b$ can dominate.
HPO will optimize the metric it is given.
It will not magically optimize the metric you intended.

The best observed configuration after $N$ trials is:

$$
\hat{\boldsymbol{\lambda}}_N
\in
\arg\min_{1 \le j \le N} Y_j.
$$

Its observed score $Y(\hat{\boldsymbol{\lambda}}_N)$ is optimistically biased as an estimate of its true future performance.
This is the same problem as model-selection bias.
Cawley and Talbot emphasized that overfitting can occur in the model-selection criterion itself, not only in model parameters.

**Practical implication.**
Keep a final test set untouched until the search process is over.
If the test set influences search-space design, stopping decisions, seed choices, or reporting choices, it has become a validation set.

### 2.4 Budget, Resource, Fidelity, and Cost

A **budget** is the total resource available to the study:

$$
B_{\mathrm{total}}
=
\sum_{j=1}^{N} c_j.
$$

The cost $c_j$ may be measured in:

- GPU-hours,
- wall-clock hours,
- dollars,
- training steps,
- number of examples processed,
- energy,
- memory footprint,
- annotation cost,
- human evaluation cost.

A **resource** $r$ is a controllable amount of training effort for one trial:

$$
y_j(r) = y(\boldsymbol{\lambda}_j,r).
$$

A **fidelity** is the quality level of an approximation to the full objective.
Low-fidelity evaluations are cheaper but less faithful.

| Fidelity variable | Low-fidelity trial | High-fidelity trial | Risk |
| --- | --- | --- | --- |
| epochs | 1 epoch | 30 epochs | late bloomers can be killed |
| data fraction | 10 percent | full data | small data may change ranking |
| model size | small proxy | target model | scaling behavior may differ |
| sequence length | short context | full context | long-context effects missed |
| image resolution | low resolution | target resolution | architecture ranking changes |
| human eval samples | small panel | large panel | high variance |

Multi-fidelity methods are useful only when low-fidelity signals correlate with high-fidelity outcomes.
If the correlation is negative or unstable, aggressive early stopping is harmful.

The cost-aware objective can be written as:

$$
\min_{\boldsymbol{\lambda} \in \mathcal{X}}
\left(F(\boldsymbol{\lambda}), C(\boldsymbol{\lambda})\right),
$$

where $C$ is cost.
Sometimes this is converted to a scalar:

$$
J(\boldsymbol{\lambda})
=
F(\boldsymbol{\lambda})
+ \rho C(\boldsymbol{\lambda}),
$$

but the choice of $\rho$ encodes a tradeoff.
For many AI systems, Pareto reporting is more honest than hiding cost inside one scalar.

### 2.5 Constraints, Multi-Objective Metrics, and Reproducibility

Many HPO problems are constrained:

$$
\begin{aligned}
\min_{\boldsymbol{\lambda} \in \mathcal{X}} \quad
& F(\boldsymbol{\lambda}) \\
\text{s.t.} \quad
& G_k(\boldsymbol{\lambda}) \le 0,
\qquad k = 1,\ldots,K.
\end{aligned}
$$

Examples:

| Constraint | Meaning |
| --- | --- |
| memory $\le$ 24 GB | must fit on one GPU |
| latency $\le$ 100 ms | production serving requirement |
| toxicity rate $\le$ threshold | safety requirement |
| cost $\le$ budget | experiment budget |
| calibration error $\le$ threshold | reliability requirement |
| parameter count $\le$ limit | deployment constraint |

Multi-objective HPO does not have a single best solution in general.
For objectives $F_1,\ldots,F_m$, configuration $\boldsymbol{\lambda}_a$ **Pareto dominates** $\boldsymbol{\lambda}_b$ if:

$$
F_k(\boldsymbol{\lambda}_a)
\le
F_k(\boldsymbol{\lambda}_b)
\quad
\text{for all } k,
$$

and strict inequality holds for at least one $k$.

The **Pareto frontier** is the set of non-dominated configurations.
For model deployment, this is often the right object:

```text
loss
 ^
 |      dominated points
 |        x   x
 |    x
 |       o
 |   o
 | o
 +-----------------> latency

o = Pareto frontier
```

Reproducibility requires logging:

1. search space,
2. sampler or scheduler,
3. random seeds,
4. budget,
5. early-stopping rule,
6. code version,
7. data version,
8. hardware,
9. training metrics,
10. validation metrics,
11. failed trials,
12. final selection rule.

Without failed trials, an HPO report is incomplete.
The failures define the search process as much as the winner.

---

## 3. Baseline Search Methods

### 3.1 Manual Search and Ablation Discipline

Manual search means a human chooses configurations based on prior knowledge, diagnostics, and previous runs.
It is not automatically bad.
Most serious projects begin manually because the search space is not known yet.

A disciplined manual phase answers:

1. Is the training loop correct?
2. Does loss decrease under a known-safe configuration?
3. Which hyperparameters are sensitive?
4. What ranges are plausible?
5. Which metrics should define selection?
6. Which failures are invalid trials rather than poor configurations?

The manual phase should produce a search-space specification such as:

$$
\eta \sim \log\mathcal{U}(10^{-5},3 \cdot 10^{-3}),
\qquad
\lambda_{\mathrm{wd}} \sim \log\mathcal{U}(10^{-6},10^{-1}),
\qquad
r_{\mathrm{LoRA}} \in \{4,8,16,32\}.
$$

Manual search becomes weak science when:

- failed runs are not logged,
- ranges change after seeing test results,
- multiple seeds are tried but only the best seed is reported,
- the stopping rule is informal,
- the final comparison uses unequal compute,
- the chosen configuration is explained after the fact.

**Ablation discipline.**
When changing one hyperparameter, hold others fixed.
When changing many, admit that the result is a configuration-level comparison.
For example, comparing:

$$
(\eta=3\cdot10^{-4}, B=64)
\quad \text{vs.} \quad
(\eta=10^{-4}, B=256)
$$

does not isolate the learning rate.
It tests a pair.

Manual search is best viewed as human-guided experimental design.
It should make later automatic search cheaper and more meaningful.

### 3.2 Grid Search and the Curse of Dimensionality

Grid search enumerates:

$$
\mathcal{G}
=
\mathcal{G}_1 \times \cdots \times \mathcal{G}_d.
$$

If $\mathcal{G}_i$ has $m_i$ values, then:

$$
\lvert \mathcal{G} \rvert
=
\prod_{i=1}^{d} m_i.
$$

This product grows rapidly.
With six hyperparameters and five choices each:

$$
5^6 = 15625
$$

trials.
If each trial costs 30 minutes on one GPU, the total cost is:

$$
15625 \cdot 0.5 = 7812.5
$$

GPU-hours.

Grid search is defensible when:

1. the space is tiny,
2. variables are categorical,
3. every combination is meaningful,
4. parallel compute is abundant,
5. the goal is a complete factorial ablation,
6. the evaluation is cheap.

Grid search is weak when:

1. important variables are continuous,
2. useful scales are logarithmic,
3. only a few dimensions matter,
4. conditional branches exist,
5. the grid boundaries were chosen casually,
6. the budget is fixed and small.

The random-search paper showed that grid search wastes trials when many hyperparameters have little effect.
This is common in ML.
A deep model may be very sensitive to learning rate, moderately sensitive to weight decay, and almost insensitive to some augmentation parameter within a broad range.
A grid does not know this.
It spreads resolution evenly.

The expected best grid result is therefore limited by grid resolution on the important dimensions.
If the best learning rate lies between grid points, the grid cannot find it.
Random search can.

### 3.3 Random Search and Effective Dimension

Random search samples:

$$
\boldsymbol{\lambda}_j \sim q(\boldsymbol{\lambda}),
\qquad
j=1,\ldots,N.
$$

It is embarrassingly parallel.
It needs no surrogate model.
It handles mixed spaces.
It is easy to reproduce.
It is a strong baseline.

The probability that random search samples at least one point in a good region $\mathcal{A} \subseteq \mathcal{X}$ is:

$$
P(\text{hit } \mathcal{A})
=
1 - (1 - q(\mathcal{A}))^N.
$$

If $q(\mathcal{A})=0.05$, then after $N=60$ trials:

$$
1 - 0.95^{60} \approx 0.954.
$$

This simple formula is useful.
It says random search becomes reliable when the good region has nontrivial probability under the search distribution.
The art is defining $q$ so good regions are not tiny.

Effective dimension explains why random search often beats grid search.
Suppose:

$$
F(\boldsymbol{\lambda})
=
f(\lambda_1,\lambda_2)
$$

even though $\boldsymbol{\lambda} \in \mathbb{R}^{20}$.
A grid with two values per coordinate uses $2^{20}$ trials and only two values on $\lambda_1$.
Random search with $N=100$ uses 100 different values on $\lambda_1$ and $\lambda_2$.

**Search-space quality dominates sampler cleverness.**
A random sampler over a bad range is bad.
For learning rates:

$$
\eta \sim \mathcal{U}(0,1)
$$

puts almost all mass on values far too large for many neural networks.
The same sampler over:

$$
\log_{10}\eta \sim \mathcal{U}(-5,-3)
$$

may work well.

Random search should usually be the first automatic baseline.
If a fancy HPO algorithm cannot beat a well-designed random search at the same budget, the fancy algorithm is not helping.

### 3.4 Quasi-Random and Space-Filling Designs

Random points can cluster.
Space-filling designs try to cover the search space more evenly.
Examples include:

- Sobol sequences,
- Halton sequences,
- Latin hypercube sampling,
- orthogonal arrays,
- maximin designs.

The aim is to reduce empty regions in the initial design.
For a normalized box $\mathcal{X}=[0,1]^d$, a space-filling design tries to make the set:

$$
\{\boldsymbol{\lambda}_1,\ldots,\boldsymbol{\lambda}_N\}
$$

spread across the box.

One criterion is maximin distance:

$$
\max_{\boldsymbol{\lambda}_1,\ldots,\boldsymbol{\lambda}_N}
\min_{i \ne j}
\lVert \boldsymbol{\lambda}_i - \boldsymbol{\lambda}_j \rVert_2.
$$

This criterion prefers points far apart.
It is useful for initial Bayesian optimization designs because surrogate models need broad coverage before exploiting.

Space-filling designs are not magic.
They still depend on transformations.
If learning rate is sampled in linear space, a Sobol design will evenly fill the wrong geometry.
First choose the right parameterization, then choose the design.

**Good use.**
Use 20 Sobol or Latin-hypercube initial points before Bayesian optimization in a 6-dimensional continuous search.

**Bad use.**
Use a space-filling design over 100 mostly irrelevant variables and expect it to solve high-dimensional HPO.

### 3.5 Choosing Defensible Baselines

A defensible HPO baseline has:

1. a written search space,
2. a fixed trial budget,
3. a fixed validation metric,
4. a fixed seed policy,
5. failed-trial handling,
6. equal compute across compared methods,
7. final evaluation on data not used for selection.

For most projects:

| Situation | Baseline |
| --- | --- |
| tiny categorical space | grid search |
| medium mixed space | random search |
| expensive continuous objective | random search plus Bayesian optimization |
| iterative training with early metrics | random search plus ASHA or Hyperband |
| schedule-sensitive RL or GAN training | PBT baseline |
| final paper claim | repeated random search and fixed-budget comparison |

The baseline should not be a strawman.
For example, grid search over linearly spaced learning rates is a weak baseline.
A fair comparison uses log-scaled random search.

---

## 4. Bayesian and Surrogate-Based Optimization

### 4.1 Sequential Model-Based Optimization

Bayesian optimization is a special case of sequential model-based optimization.
It maintains a surrogate model of the unknown objective and uses that model to choose the next trial.

At step $t$, the data are:

$$
\mathcal{D}_t
=
\{(\boldsymbol{\lambda}_j,Y_j)\}_{j=1}^{t}.
$$

The surrogate gives a predictive distribution:

$$
p(F(\boldsymbol{\lambda}) \mid \mathcal{D}_t).
$$

An acquisition function $a_t(\boldsymbol{\lambda})$ scores candidate points.
For minimization, the next point is:

$$
\boldsymbol{\lambda}_{t+1}
\in
\arg\max_{\boldsymbol{\lambda} \in \mathcal{X}}
a_t(\boldsymbol{\lambda}).
$$

Then the expensive training run evaluates $Y_{t+1}$.
The loop repeats.

```text
observed trials D_t
        |
        v
fit/update surrogate model
        |
        v
compute acquisition function
        |
        v
choose next configuration
        |
        v
train and validate
        |
        v
append result to D_{t+1}
```

The surrogate does not need to be perfect.
It needs to be useful enough to guide exploration.
Bayesian optimization is most attractive when:

- each trial is expensive,
- dimension is modest,
- parallelism is limited,
- the objective has exploitable structure,
- uncertainty estimates help decide where to sample.

It is less attractive when:

- thousands of cheap trials are available,
- the search space is very high-dimensional,
- many variables are conditional categorical variables,
- the objective changes over time,
- the surrogate overhead is comparable to trial cost.

### 4.2 Gaussian-Process Surrogate Intuition

A Gaussian process places a prior over functions:

$$
F \sim \mathcal{GP}(m,k),
$$

where $m(\boldsymbol{\lambda})$ is a mean function and $k(\boldsymbol{\lambda},\boldsymbol{\lambda}')$ is a kernel.
For any finite set of configurations, function values have a multivariate Gaussian distribution.

Assume noisy observations:

$$
Y_j = F(\boldsymbol{\lambda}_j) + \varepsilon_j,
\qquad
\varepsilon_j \sim \mathcal{N}(0,\sigma_n^2).
$$

Let:

$$
K_{ij}
=
k(\boldsymbol{\lambda}_i,\boldsymbol{\lambda}_j),
\qquad
\mathbf{y}
=
(Y_1,\ldots,Y_t)^\top.
$$

For a candidate $\boldsymbol{\lambda}$, define:

$$
\mathbf{k}_{*}
=
\left(k(\boldsymbol{\lambda},\boldsymbol{\lambda}_1),\ldots,
k(\boldsymbol{\lambda},\boldsymbol{\lambda}_t)\right)^\top.
$$

The posterior predictive mean and variance are:

$$
\mu_t(\boldsymbol{\lambda})
=
m(\boldsymbol{\lambda})
+ \mathbf{k}_{*}^{\top}
(K + \sigma_n^2 I)^{-1}
(\mathbf{y} - \mathbf{m}),
$$

$$
\sigma_t^2(\boldsymbol{\lambda})
=
k(\boldsymbol{\lambda},\boldsymbol{\lambda})
- \mathbf{k}_{*}^{\top}
(K + \sigma_n^2 I)^{-1}
\mathbf{k}_{*}.
$$

The mean $\mu_t$ says where the model expects good performance.
The standard deviation $\sigma_t$ says where the model is uncertain.
Acquisition functions combine both.

The kernel matters.
An RBF kernel assumes smooth variation:

$$
k(\boldsymbol{\lambda},\boldsymbol{\lambda}')
=
\sigma_f^2
\exp\left(
-\frac{1}{2}
\sum_{i=1}^{d}
\frac{(\lambda_i-\lambda_i')^2}{\ell_i^2}
\right).
$$

The lengthscale $\ell_i$ controls sensitivity to dimension $i$.
Small $\ell_i$ means the objective changes quickly along that coordinate.
Large $\ell_i$ means the objective changes slowly.

This connects to effective dimension:
Bayesian optimization can learn that some dimensions matter less,
but it needs enough data and a surrogate family capable of expressing that fact.

### 4.3 Acquisition Functions: PI, EI, UCB, and Thompson Sampling

Let the current best observed value for minimization be:

$$
y_{\min}
=
\min_{1 \le j \le t} Y_j.
$$

Assume the surrogate predictive distribution at $\boldsymbol{\lambda}$ is:

$$
F(\boldsymbol{\lambda}) \mid \mathcal{D}_t
\sim
\mathcal{N}(\mu_t(\boldsymbol{\lambda}),\sigma_t^2(\boldsymbol{\lambda})).
$$

**Probability of Improvement.**
For margin $\xi \ge 0$:

$$
\operatorname{PI}(\boldsymbol{\lambda})
=
P(F(\boldsymbol{\lambda}) \le y_{\min} - \xi)
=
\Phi(z),
$$

where

$$
z
=
\frac{y_{\min} - \xi - \mu_t(\boldsymbol{\lambda})}
{\sigma_t(\boldsymbol{\lambda})}.
$$

PI can be greedy.
It asks for probability of any improvement, not size of improvement.

**Expected Improvement.**
Define improvement:

$$
I(\boldsymbol{\lambda})
=
\max(0, y_{\min} - F(\boldsymbol{\lambda}) - \xi).
$$

Then:

$$
\operatorname{EI}(\boldsymbol{\lambda})
=
\mathbb{E}[I(\boldsymbol{\lambda})]
=
(y_{\min}-\xi-\mu_t)\Phi(z)
+ \sigma_t \phi(z).
$$

EI balances exploitation and exploration:

- low $\mu_t$ gives expected good score,
- high $\sigma_t$ gives room for improvement.

**Lower Confidence Bound.**
For minimization:

$$
\operatorname{LCB}(\boldsymbol{\lambda})
=
\mu_t(\boldsymbol{\lambda})
- \kappa \sigma_t(\boldsymbol{\lambda}).
$$

Choose the point with smallest LCB.
Larger $\kappa$ explores uncertain regions more aggressively.

**Thompson sampling.**
Sample a function from the posterior:

$$
\tilde{F} \sim p(F \mid \mathcal{D}_t),
$$

then choose:

$$
\boldsymbol{\lambda}_{t+1}
\in
\arg\min_{\boldsymbol{\lambda}}
\tilde{F}(\boldsymbol{\lambda}).
$$

This turns posterior uncertainty into randomized exploration.

**Acquisition optimization is itself an optimization problem.**
Acquisition functions can be non-convex.
They are cheap compared with training, so one can optimize them with dense candidate sampling, multistart local search, evolutionary search, or mixed strategies.

### 4.4 Tree-Structured Parzen Estimators

Gaussian processes are elegant but can struggle with high-dimensional, conditional, and categorical HPO spaces.
Tree-structured Parzen estimators (TPE) model the problem differently.

Instead of modeling $p(y \mid \boldsymbol{\lambda})$ directly, TPE models:

$$
p(\boldsymbol{\lambda} \mid y)
$$

by splitting observations into good and bad groups.
Choose a threshold $y^{*}$ such that:

$$
P(y < y^{*}) = \gamma.
$$

For minimization, define:

$$
l(\boldsymbol{\lambda})
=
p(\boldsymbol{\lambda} \mid y < y^{*}),
$$

$$
g(\boldsymbol{\lambda})
=
p(\boldsymbol{\lambda} \mid y \ge y^{*}).
$$

TPE proposes configurations that have high density under the good model and low density under the bad model.
Expected improvement can be connected to the ratio:

$$
\frac{l(\boldsymbol{\lambda})}{g(\boldsymbol{\lambda})}.
$$

This representation handles tree-structured conditional spaces naturally because each branch can have its own density.
That is why TPE-style methods are common in tools such as Hyperopt and Optuna.

**Example.**
If AdamW trials with $\eta$ near $2 \cdot 10^{-4}$ often land in the good group and trials near $10^{-2}$ land in the bad group,
the TPE density ratio will favor the former region.

**Caution.**
TPE is still model-based search.
It can exploit too early if initial trials are unlucky or if the search space is poorly encoded.
Random exploration and good priors still matter.

### 4.5 Noisy, Constrained, Cost-Aware, and Multi-Objective BO

Practical Bayesian optimization extends the simple loop in several ways.

**Noisy BO.**
With noise:

$$
Y = F(\boldsymbol{\lambda}) + \varepsilon,
\qquad
\varepsilon \sim \mathcal{N}(0,\sigma_n^2).
$$

The best observed value may not be the best true value.
Noisy acquisition functions account for uncertainty about the incumbent.
A practical alternative is to repeat promising configurations across seeds.

**Constrained BO.**
Suppose feasibility depends on unknown constraints:

$$
G_k(\boldsymbol{\lambda}) \le 0.
$$

One can model each constraint with a surrogate and use:

$$
a_{\mathrm{constrained}}(\boldsymbol{\lambda})
=
a(\boldsymbol{\lambda})
\prod_{k=1}^{K}
P(G_k(\boldsymbol{\lambda}) \le 0 \mid \mathcal{D}_t).
$$

This favors candidates that are both promising and likely feasible.

**Cost-aware BO.**
If trials have different costs, use improvement per expected cost:

$$
a_{\mathrm{cost}}(\boldsymbol{\lambda})
=
\frac{\operatorname{EI}(\boldsymbol{\lambda})}
\mathbb{E}[C(\boldsymbol{\lambda})]}.
$$

This is useful when larger models or longer sequence lengths cost more.

**Multi-objective BO.**
For objectives such as loss and latency, the acquisition can target expected hypervolume improvement.
The output is a Pareto frontier, not a single scalar best.

**Parallel BO.**
When many workers are available, sequential BO can become a bottleneck.
Batch acquisition functions or asynchronous pending-point heuristics are needed.
In highly parallel settings, random search and ASHA can be more efficient in wall-clock time.

---

## 5. Multi-Fidelity and Bandit Methods

### 5.1 Resource Allocation as Optimization

Multi-fidelity HPO evaluates configurations at different resource levels.
Let:

$$
y(\boldsymbol{\lambda},r)
$$

be the observed score for configuration $\boldsymbol{\lambda}$ trained with resource $r$.
The target is performance at maximum resource $R$:

$$
F(\boldsymbol{\lambda}) = y(\boldsymbol{\lambda},R).
$$

A resource-allocation policy decides which pairs $(\boldsymbol{\lambda},r)$ to evaluate.
The total budget constraint is:

$$
\sum_{j} r_j \le B_{\mathrm{total}}.
$$

The simplest policy trains every configuration to $R$.
Bandit methods do better when early results are informative.

The exploration-exploitation tradeoff becomes:

- explore many configurations at small $r$,
- exploit promising configurations at larger $r$.

This is similar to a multi-armed bandit, but arms are infinitely many possible configurations and rewards are non-stochastic or weakly modeled learning curves.

### 5.2 Successive Halving

Successive halving starts with $n$ configurations and a small resource $r_0$.
At each rung:

1. train all remaining configurations with current resource,
2. rank by validation score,
3. keep the top fraction $1/\eta_{\mathrm{SH}}$,
4. multiply resource by $\eta_{\mathrm{SH}}$.

Let $\eta_{\mathrm{SH}}>1$ be the halving factor.
If lower score is better, survivors are:

$$
\mathcal{S}_{k+1}
=
\operatorname{Top}_{\lceil |\mathcal{S}_k|/\eta_{\mathrm{SH}} \rceil}
\left\{
(\boldsymbol{\lambda},y(\boldsymbol{\lambda},r_k)):
\boldsymbol{\lambda} \in \mathcal{S}_k
\right\}.
$$

Resources follow:

$$
r_{k+1} = \eta_{\mathrm{SH}} r_k.
$$

Candidate count follows approximately:

$$
n_{k+1} = \left\lfloor \frac{n_k}{\eta_{\mathrm{SH}}} \right\rfloor.
$$

The tournament view:

```text
r = 1        A B C D E F G H I
             | | | | | | | | |
keep top 3       B     D   F

r = 3             B     D   F
                  |     |   |
keep top 1              D

r = 9                   D
```

Successive halving works when bad configurations are bad early.
It struggles when good configurations need long warmup.

**Example: neural network learning rate.**
Very high learning rates diverge early.
Very low learning rates may look bad early but become stable later.
Successive halving may discard very low learning rates too aggressively unless the minimum resource is large enough.

### 5.3 Hyperband

Successive halving depends on the starting number of configurations and initial resource.
Hyperband runs multiple successive-halving brackets with different tradeoffs.
Some brackets try many configurations with little initial resource.
Others try fewer configurations with more initial resource.

Let $R$ be the maximum resource and $\eta_{\mathrm{HB}}>1$ the downsampling rate.
Define:

$$
s_{\max}
=
\left\lfloor \log_{\eta_{\mathrm{HB}}} R \right\rfloor.
$$

Hyperband loops over brackets $s \in \{s_{\max},\ldots,0\}$.
Each bracket chooses:

$$
n_s
\approx
\left\lceil
\frac{s_{\max}+1}{s+1}
\eta_{\mathrm{HB}}^{s}
\right\rceil,
$$

and:

$$
r_s
=
R\eta_{\mathrm{HB}}^{-s}.
$$

Large $s$ means many configurations with small initial resource.
Small $s$ means fewer configurations with larger initial resource.

The point is robust hedging.
If early learning curves are reliable, aggressive brackets work.
If early learning curves are unreliable, conservative brackets protect late bloomers.

Hyperband's insight is:

$$
\text{resource allocation can beat smarter configuration choice.}
$$

The original Hyperband paper reported large speedups by combining random search with adaptive resource allocation.
This is why Hyperband remains a practical baseline for deep learning.

### 5.4 ASHA and Asynchronous Distributed Tuning

Synchronous successive halving waits for all trials in a rung to finish before promotion.
In distributed systems, this wastes time.
Slow trials block fast trials.

Asynchronous Successive Halving Algorithm (ASHA) promotes configurations as soon as enough evidence is available.
Instead of waiting for a full rung, each trial reports intermediate metrics.
The scheduler compares against completed trials at the same resource level and decides:

$$
\text{promote, pause, continue, or stop.}
$$

The benefit is better hardware utilization.
Workers rarely sit idle.

The risk is noisier promotion decisions.
A trial may be stopped because the current comparison set is unlucky or biased.
ASHA therefore needs careful choices for:

- minimum resource,
- reduction factor,
- grace period,
- metric smoothing,
- handling missing reports,
- retrying failed trials.

For LLM fine-tuning, ASHA can be useful when many short fine-tunes are possible.
For full pretraining, individual trials may be too expensive for naive HPO; scaling-law-informed sweeps and smaller proxy runs become more important.

### 5.5 BOHB: Bayesian Optimization plus Hyperband

BOHB combines Bayesian optimization with Hyperband.
Hyperband supplies multi-fidelity resource allocation.
The Bayesian component guides which configurations to sample.

The motivation is:

1. Random Hyperband explores broadly and allocates budget well.
2. Bayesian optimization learns from prior results but can be expensive and slow for deep learning.
3. BOHB tries to keep Hyperband's anytime behavior while improving sample quality.

At a high level:

```text
previous trials
      |
      v
fit density/surrogate model
      |
      v
sample promising configurations
      |
      v
run Hyperband-style brackets
      |
      v
update model
```

BOHB is useful when:

- multi-fidelity signals exist,
- the search space is mixed,
- pure random sampling is too wasteful,
- pure Bayesian optimization is too slow,
- parallel resources are available.

It is not a theorem that BOHB always wins.
The practical lesson is broader:

$$
\text{good HPO often combines better sampling with better resource allocation.}
$$

---

## 6. Population and Dynamic Hyperparameters

### 6.1 Evolutionary and Population Search

Evolutionary HPO maintains a population of configurations:

$$
\mathcal{P}_t
=
\{\boldsymbol{\lambda}_{t,1},\ldots,\boldsymbol{\lambda}_{t,M}\}.
$$

The loop is:

1. evaluate population members,
2. select better members,
3. mutate or recombine configurations,
4. repeat.

Mutation might change:

$$
\eta' = \eta \cdot \exp(\delta),
\qquad
\delta \sim \mathcal{N}(0,\sigma^2).
$$

Categorical mutation might switch optimizer or augmentation policy.
Architecture mutation might add a layer or change width.

Population methods are attractive for rugged, mixed, or conditional spaces.
They are less sample-efficient than good Bayesian optimization on smooth low-dimensional objectives, but they parallelize naturally and handle non-smooth choices.

**Non-example.**
Training one model and changing its hyperparameters manually after looking at validation curves is not a reproducible population method.
It is manual intervention unless the rule is specified in advance.

### 6.2 Population-Based Training

Population-Based Training (PBT) jointly trains model weights and hyperparameters.
Each population member has:

$$
(\boldsymbol{\theta}_{t,i}, \boldsymbol{\lambda}_{t,i}).
$$

During training, weak performers copy weights from stronger performers and perturb hyperparameters.
This is often described as exploit and explore:

- **exploit:** replace a weak member with a copy of a strong member,
- **explore:** mutate the copied hyperparameters.

PBT does not merely find a fixed $\boldsymbol{\lambda}$.
It can discover a schedule:

$$
\boldsymbol{\lambda}_i(t).
$$

This is important for learning rates, entropy coefficients in reinforcement learning, augmentation strength, loss weights, and other time-dependent controls.

PBT differs from ordinary HPO:

| Ordinary HPO | PBT |
| --- | --- |
| train each trial independently | trials exchange weights |
| usually fixed hyperparameters | dynamic hyperparameter schedules |
| final configuration selected after training | selection happens during training |
| easier statistical interpretation | more coupled and harder to analyze |

PBT is powerful but less clean for final scientific claims because trials are not independent.
The lineage of weights and hyperparameters must be logged.

### 6.3 Fixed Hyperparameters versus Learned Schedules

Some quantities called hyperparameters are really functions of time:

$$
\eta_t = s(t;\boldsymbol{\lambda}).
$$

Learning-rate schedules belong in [Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md).
Here, the HPO question is how to select schedule parameters such as:

- peak learning rate,
- warmup steps,
- decay shape,
- final learning-rate ratio,
- cycle length,
- restart policy,
- stable phase duration.

> **Preview: Learning Rate Schedules**
>
> Warmup, cosine decay, one-cycle schedules, and WSD schedules have their own mathematical structure.
> This section treats them as tunable configuration families.
>
> -> Full treatment: [Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md)

The distinction matters.
Optimizing a fixed learning rate $\eta$ is a different problem from optimizing a schedule $\eta_t$.
A single scalar may be too weak for long training runs.
But allowing arbitrary schedules makes the search space enormous.

Common compromise:

$$
\eta_t
=
\eta_{\max} \cdot s(t;T_{\mathrm{warm}},T_{\mathrm{decay}},\rho).
$$

Tune a few schedule parameters rather than every time step.

### 6.4 Architecture and Augmentation Policy Search

Architecture search and augmentation search are related to HPO but can become full subfields.
They include:

- neural architecture search,
- depth and width search,
- convolution block search,
- transformer head and MLP ratio search,
- augmentation policy search,
- data mixture search,
- prompt template search,
- tokenizer or vocabulary choices.

This section gives only the HPO view:
these are structured configuration spaces with expensive noisy objectives.

Architecture search often has stronger coupling to model capacity and compute:

$$
F(\boldsymbol{\lambda})
\quad \text{and} \quad
C(\boldsymbol{\lambda})
$$

change together.
A larger model may improve validation loss but violate latency, memory, or training budget.
Therefore architecture HPO should usually be multi-objective.

For LLMs, full architecture search is rare for ordinary teams because pretraining budgets are enormous.
More common search targets are:

- adapter rank,
- target modules,
- data mixture weights,
- context length,
- batch size,
- optimizer and schedule parameters,
- decoding and inference parameters.

### 6.5 Transfer HPO, Warm Starts, and Meta-Learning

HPO studies are not always independent.
Past tasks can inform future search.
Let task $m$ have objective:

$$
F_m(\boldsymbol{\lambda}).
$$

Transfer HPO uses previous studies:

$$
\{\mathcal{D}^{(1)},\ldots,\mathcal{D}^{(M-1)}\}
$$

to improve search on task $M$.

Simple transfer:

1. start from the best historical configurations,
2. narrow search ranges around robust regions,
3. reuse priors for learning rate and weight decay,
4. initialize surrogate models with past observations.

Meta-learning view:

$$
p(F_M \mid \mathcal{D}^{(1:M-1)},\mathcal{D}^{(M)}_t).
$$

This is attractive when an organization repeatedly trains similar models.
It is dangerous when tasks differ in hidden ways.
A hyperparameter prior from one dataset can be harmful on another.

**Rule.**
Transfer prior knowledge, but still measure.
Do not treat historical defaults as proof of current optimality.

---

## 7. Statistical Reliability and Selection Bias

### 7.1 Train, Validation, and Test Discipline

The correct split logic is:

| Split | Used for |
| --- | --- |
| training | fit $\boldsymbol{\theta}$ |
| validation | select $\boldsymbol{\lambda}$ |
| test | estimate final generalization after selection |

The test set must not influence:

- search ranges,
- early stopping,
- architecture choice,
- seed choice,
- prompt choice,
- data cleaning decisions,
- final checkpoint selection,
- retry decisions.

If it does, it is no longer a test set.

The validation objective is:

$$
\hat{R}_{\mathrm{val}}(\boldsymbol{\lambda}).
$$

The test estimate is:

$$
\hat{R}_{\mathrm{test}}(\hat{\boldsymbol{\lambda}}).
$$

where $\hat{\boldsymbol{\lambda}}$ is selected without using test labels.

Leakage can be subtle.
Examples:

1. evaluating every trial on the test set and selecting the best,
2. changing the search space after seeing test failures,
3. selecting the best random seed by test performance,
4. tuning a threshold on test data,
5. removing difficult test examples after inspecting errors,
6. choosing between papers' methods based on your test set.

For LLM evaluation, the same issue appears with benchmark suites.
If prompts, decoding settings, or checkpoints are repeatedly selected on the benchmark, the benchmark has become validation.

### 7.2 Nested Cross-Validation and Model-Selection Bias

In small-data settings, one often uses cross-validation.
For hyperparameter tuning, cross-validation can estimate:

$$
\hat{R}_{\mathrm{CV}}(\boldsymbol{\lambda})
=
\frac{1}{K}
\sum_{k=1}^{K}
\hat{R}^{(k)}_{\mathrm{val}}(\boldsymbol{\lambda}).
$$

If the same cross-validation scores are used both to select hyperparameters and report final performance, the estimate is optimistic.
Nested cross-validation separates selection and evaluation.

Outer loop:

1. hold out fold $k$ as outer test,
2. tune hyperparameters using only the remaining data,
3. train selected model,
4. evaluate on outer fold.

Inner loop:

1. split the outer-training data,
2. compare hyperparameters,
3. choose $\hat{\boldsymbol{\lambda}}^{(k)}$.

Nested CV estimates the performance of the full selection procedure:

$$
\hat{R}_{\mathrm{nested}}
=
\frac{1}{K_{\mathrm{outer}}}
\sum_{k=1}^{K_{\mathrm{outer}}}
\hat{R}_{\mathrm{outer},k}
\left(
\hat{\boldsymbol{\lambda}}^{(k)}
\right).
$$

This is more expensive than ordinary CV.
It is worth it when data are small and performance claims must be reliable.

For large deep learning, nested CV is often infeasible.
The analogous discipline is a clean validation/test split plus limited final test evaluation.

### 7.3 Multiple Comparisons, Seeds, and Confidence Intervals

If we test many configurations, some will look good by chance.
Assume for intuition that all configurations have equal true mean $\mu$ and observed noise:

$$
Y_j \sim \mathcal{N}(\mu,\sigma^2).
$$

Then:

$$
\mathbb{E}[\min_j Y_j] < \mu.
$$

The more trials, the more optimistic the best observed score.
This is the winner's curse.

Repeated seeds help estimate training variability.
For configuration $\boldsymbol{\lambda}$ and seeds $s=1,\ldots,S$:

$$
\bar{Y}(\boldsymbol{\lambda})
=
\frac{1}{S}
\sum_{s=1}^{S}
Y(\boldsymbol{\lambda},s).
$$

The standard error is:

$$
\operatorname{SE}(\bar{Y})
=
\frac{\hat{\sigma}}{\sqrt{S}}.
$$

But repeating every configuration across many seeds can be expensive.
A practical pattern:

1. use one seed for broad exploration,
2. repeat top configurations across several seeds,
3. choose based on mean and variance,
4. report uncertainty.

Do not select the best seed.
A seed is part of the noise process, not a model improvement.

### 7.4 Multi-Metric and Pareto Reporting

Modern AI systems rarely optimize one metric.
Consider:

$$
\left(
\text{validation loss},
\text{latency},
\text{memory},
\text{toxicity rate},
\text{calibration error}
\right).
$$

Scalarizing:

$$
J
=
\alpha_1 F_1 + \cdots + \alpha_m F_m
$$

is convenient but hides tradeoffs.
Pareto reporting is often clearer.

A configuration is Pareto-efficient if no other configuration improves one metric without worsening another.
For model deployment, a slightly worse loss may be acceptable if latency is much better.
For safety-critical systems, a lower loss may not justify a higher violation rate.

Good HPO reports include:

- primary selection metric,
- secondary guardrail metrics,
- Pareto frontier,
- cost per trial,
- total cost,
- failed trials,
- uncertainty over final candidates.

### 7.5 Reproducible HPO Reports

A reproducible HPO report should answer:

1. What was the exact search space?
2. Which sampler and scheduler were used?
3. What was the budget?
4. What metric selected the winner?
5. What seeds were used?
6. Were trials pruned?
7. How were failed trials handled?
8. What code and data version were used?
9. What hardware was used?
10. How many total configurations were evaluated?
11. How many reached full budget?
12. Was the final test set touched before selection?

Minimal table:

| Field | Example |
| --- | --- |
| Study name | `hpo_lora_rank_lr_2026_05_05` |
| Search space | YAML or JSON specification |
| Sampler | TPE with random startup trials |
| Scheduler | ASHA, grace period 500 steps |
| Budget | 80 trials, max 3000 steps |
| Metric | validation negative log-likelihood |
| Seeds | one exploration seed, five final seeds |
| Data version | hash or dataset release |
| Code version | git commit |
| Hardware | 1xA100 80GB |

This may feel bureaucratic.
It is how future-you knows whether the result is real.

---

## 8. Applications in ML and LLM Systems

### 8.1 Classical ML Pipelines

Classical ML HPO often tunes preprocessing, model class, and estimator hyperparameters together.
For an SVM pipeline:

$$
\boldsymbol{\lambda}
=
(\text{scaler}, C, \gamma, \text{kernel}).
$$

For a random forest:

$$
\boldsymbol{\lambda}
=
(\text{n_estimators}, \text{max_depth}, \text{min_samples_leaf}, \text{max_features}).
$$

Cross-validation is common because datasets may be small.
Pipelines are important because preprocessing must be fit inside each training fold.
If scaling is fit before cross-validation on the whole dataset, validation information leaks into training.

Grid search can be fine for tiny categorical spaces.
Randomized search is better when distributions matter.
Successive halving can use resources such as number of samples or number of trees.

### 8.2 Deep Learning Knobs

Deep learning HPO usually centers on:

| Hyperparameter | Notes |
| --- | --- |
| learning rate | most sensitive, usually log scale |
| batch size | coupled with learning rate and hardware |
| optimizer | AdamW is common, but not universal |
| weight decay | interacts with optimizer and architecture |
| dropout | task- and model-dependent |
| gradient clipping | stabilizes but can slow learning |
| warmup | crucial for large models |
| number of steps | resource and performance variable |
| augmentation | domain-specific |
| label smoothing | affects calibration and loss |

Many of these have canonical theory elsewhere in the chapter.
In HPO, we treat them as coordinates of $\boldsymbol{\lambda}$.

Learning rate and batch size are coupled.
A larger batch often supports a larger learning rate up to a point.
Therefore independent one-dimensional sweeps can miss good regions.

Weight decay and learning rate are also coupled in AdamW because the decoupled decay update has scale:

$$
\eta \lambda_{\mathrm{wd}} \boldsymbol{\theta}_t.
$$

Changing $\eta$ changes the effective decay per step.

### 8.3 LLM Fine-Tuning Knobs

LLM fine-tuning HPO is often constrained by memory and data quality.
Common knobs:

| Knob | Meaning |
| --- | --- |
| $\eta$ | adapter or full fine-tuning learning rate |
| $B_{\mathrm{micro}}$ | per-device microbatch size |
| $B_{\mathrm{eff}}$ | effective batch size after gradient accumulation |
| $r$ | LoRA rank |
| $\alpha$ | LoRA scaling |
| target modules | which matrices receive adapters |
| epochs or steps | training resource |
| warmup ratio | schedule shape |
| weight decay | regularization |
| max sequence length | memory and task coverage |
| data mixture weights | instruction, preference, domain data |
| packing policy | sample packing for efficiency |

A practical first search might be:

$$
\log_{10}\eta \sim \mathcal{U}(-5,-3),
\qquad
r \in \{4,8,16,32\},
\qquad
\lambda_{\mathrm{wd}} \in \{0,0.01,0.05\}.
$$

Do not tune only on training loss.
Fine-tuning can overfit style, memorization, or benchmark artifacts.
Use validation tasks aligned with deployment.

For instruction tuning, the objective may combine:

- validation loss,
- task accuracy,
- preference win rate,
- toxicity or safety filters,
- format compliance,
- latency and memory.

This is a multi-objective HPO problem.

### 8.4 Practical Tooling

Common tools encode the same mathematical concepts:

| Tool | Concepts |
| --- | --- |
| scikit-learn search | grid, randomized search, successive halving, cross-validation |
| Optuna | studies, trials, TPE, pruning, conditional spaces |
| Ray Tune | distributed trials, schedulers, ASHA, Hyperband, PBT |
| KerasTuner | random, Bayesian, Hyperband tuners |
| SageMaker tuning | managed random, Bayesian, Hyperband, grid strategies |
| Weights and Biases sweeps | experiment tracking plus sweep strategies |
| MLflow | tracking and model registry integration |

The tool is not the method.
You still must specify:

$$
(\mathcal{X}, q, B_{\mathrm{total}}, \text{metric}, \text{selection rule}).
$$

Good tooling helps with:

- reproducibility,
- failed-trial tracking,
- parallel scheduling,
- visualization,
- artifact storage,
- resuming studies,
- comparing studies.

Bad usage of good tooling is still bad science.
A tuner cannot fix a leaked validation set or meaningless metric.

### 8.5 When Not to Tune

HPO has opportunity cost.
Sometimes the right move is not another sweep.

Do not tune aggressively when:

1. the training loop is still buggy,
2. the validation metric is untrusted,
3. the dataset split is wrong,
4. the search space is arbitrary,
5. compute is better spent collecting data,
6. a robust default is known,
7. the expected improvement is smaller than noise,
8. the model is under-instrumented,
9. you cannot log failed trials,
10. the deployment constraint is not defined.

For LLM work, many teams should start with robust defaults:

- AdamW or Adafactor depending on memory,
- conservative learning-rate ranges,
- modest LoRA rank,
- clean validation set,
- fixed seed policy,
- short pilot runs,
- only then automatic search.

HPO amplifies whatever objective it is given.
If the objective is wrong, HPO makes you wrong faster.

---

## 9. Common Mistakes

| # | Mistake | Why It Is Wrong | Fix |
| --- | --- | --- | --- |
| 1 | Tuning on the test set | The test set becomes part of the optimization loop, so the final score is biased | Keep test data untouched until final evaluation |
| 2 | Searching learning rate on a linear scale | Useful learning-rate regions are usually multiplicative | Search $\log \eta$ or use log-uniform ranges |
| 3 | Comparing HPO methods with unequal budgets | More trials or more resource can masquerade as a better algorithm | Fix total budget, max resource, and parallelism |
| 4 | Reporting only the best trial | The winner may be noise-favored and hides failed attempts | Report full study summary, top trials, and uncertainty |
| 5 | Selecting the best random seed | A seed is not a hyperparameter you can expect to reproduce | Repeat final candidates and report mean and variance |
| 6 | Using grid search in high-dimensional continuous spaces | Grid resolution collapses on important dimensions | Use random or model-based search with sensible distributions |
| 7 | Pruning before learning curves are informative | Late-blooming configurations are discarded | Use a grace period and inspect low/high-fidelity correlation |
| 8 | Tuning too many knobs at once | Search budget becomes too thin for the space | Start with sensitive knobs and expand gradually |
| 9 | Ignoring conditional structure | Inactive hyperparameters confuse samplers and surrogates | Use tree-structured spaces or branch-specific variables |
| 10 | Optimizing one metric while caring about another | HPO finds what the metric rewards, not what deployment needs | Define primary metric and guardrails before search |
| 11 | Forgetting preprocessing leakage in CV | Validation fold information enters training transformations | Put preprocessing inside the cross-validation pipeline |
| 12 | Treating PBT as ordinary independent trials | Weight copying couples runs and changes interpretation | Log lineage and report the dynamic schedule |

---

## 10. Exercises

### Exercise 1: Configuration Space Identification *

For each item below, decide whether it is a model parameter, hyperparameter, metric, resource, or metadata.

(a) A transformer MLP weight matrix $W$.

(b) Learning rate $\eta$.

(c) Validation loss after 2000 steps.

(d) Number of training epochs.

(e) Git commit hash.

Explain each answer.

### Exercise 2: Grid Size and Budget *

A grid search uses:

$$
\eta \in \{10^{-5},10^{-4},10^{-3},10^{-2}\},
\quad
\lambda_{\mathrm{wd}} \in \{0,0.01,0.1\},
\quad
r \in \{4,8,16,32\},
\quad
B \in \{16,32\}.
$$

(a) How many trials are required?

(b) If each trial costs 45 minutes, what is the GPU-hour cost?

(c) What happens if two more 5-value hyperparameters are added?

### Exercise 3: Random Search Hit Probability *

Assume the good region has probability $q(\mathcal{A})=0.03$ under your search distribution.

(a) Derive the probability of at least one hit after $N$ trials.

(b) Compute this probability for $N=20,50,100$.

(c) Solve for $N$ needed to exceed 95 percent hit probability.

### Exercise 4: Log-Uniform Learning Rate **

Let $\log_{10}\eta \sim \mathcal{U}(-5,-2)$.

(a) Derive the probability that $\eta \in [10^{-4},10^{-3}]$.

(b) Compare with $\eta \sim \mathcal{U}(10^{-5},10^{-2})$.

(c) Explain why the log-uniform distribution is usually better for learning rates.

### Exercise 5: Expected Improvement **

Suppose the surrogate predictive distribution at a candidate is:

$$
F(\boldsymbol{\lambda}) \sim \mathcal{N}(\mu,\sigma^2),
$$

with $\mu=0.42$, $\sigma=0.05$, current best $y_{\min}=0.40$, and $\xi=0$.

(a) Compute $z$.

(b) Compute probability of improvement.

(c) Compute expected improvement.

(d) Interpret whether the point is exploratory or exploitative.

### Exercise 6: Successive Halving **

You start with 27 configurations, reduction factor $\eta_{\mathrm{SH}}=3$, and initial resource $r_0=1$ epoch.

(a) How many configurations remain after each rung?

(b) What resource does each rung use?

(c) What is the total epoch budget?

(d) Compare with training all 27 configurations for the final resource.

### Exercise 7: Hyperband Brackets **

Let maximum resource be $R=81$ and reduction factor $\eta_{\mathrm{HB}}=3$.

(a) Compute $s_{\max}$.

(b) Describe the most aggressive and most conservative brackets.

(c) Explain why Hyperband runs multiple brackets instead of one.

### Exercise 8: Validation Selection Bias ***

Assume 100 configurations have identical true score $\mu=0.5$ and observed scores:

$$
Y_j \sim \mathcal{N}(0.5,0.02^2).
$$

(a) Simulate the best observed score.

(b) Repeat the simulation many times and estimate the expected best observed score.

(c) Explain why the selected configuration's observed score is optimistic.

### Exercise 9: Pareto Frontier ***

Given configurations with validation loss and latency:

| Config | Loss | Latency |
| --- | --- | --- |
| A | 0.42 | 120 |
| B | 0.40 | 180 |
| C | 0.45 | 90 |
| D | 0.41 | 110 |
| E | 0.39 | 240 |

(a) Identify dominated configurations.

(b) List the Pareto frontier.

(c) Choose a configuration if latency must be at most 130 ms.

### Exercise 10: LLM Fine-Tuning Study Design ***

Design a 60-trial HPO study for LoRA fine-tuning.

Specify:

(a) search space,

(b) validation metrics,

(c) budget and pruning rule,

(d) seed policy,

(e) final selection rule,

(f) final test evaluation protocol.

---

## 11. Why This Matters for AI

| Concept | AI impact |
| --- | --- |
| Search-space design | Determines whether the tuner can even sample good configurations |
| Random search | Strong baseline for neural networks with a few sensitive knobs |
| Bayesian optimization | Saves expensive trials when the search space is modest and smooth enough |
| TPE | Handles tree-structured conditional spaces common in real ML pipelines |
| Successive halving | Converts early learning curves into compute savings |
| Hyperband and ASHA | Practical for parallel neural-network tuning with intermediate metrics |
| BOHB | Combines guided sampling with multi-fidelity allocation |
| PBT | Learns dynamic hyperparameter schedules in RL, GANs, and long training |
| Nested CV | Prevents optimistic claims in small-data model selection |
| Multi-objective HPO | Makes tradeoffs among loss, latency, memory, cost, and safety explicit |
| Repeated seeds | Separates robust configurations from lucky runs |
| Experiment tracking | Turns HPO from folklore into reproducible engineering |

In 2026-era AI systems, HPO is not only a convenience layer.
It is part of the scientific claim.
If a new optimizer, adapter method, retrieval pipeline, or alignment recipe wins only because it received a better tuning budget, the comparison is weak.
If the validation set leaks into prompt selection or checkpoint choice, the benchmark number is inflated.
If final candidates are not repeated across seeds, a lucky run may become a false default.

For LLMs, the largest runs are too expensive for naive HPO.
That does not make HPO irrelevant.
It makes the experimental design more important:
small proxy runs, scaling-law-informed sweeps, careful data-mixture studies, robust defaults, and limited high-fidelity confirmations.

---

## 12. Conceptual Bridge

This section sits after adaptive optimizers and regularization because HPO treats those earlier ideas as choices in an outer loop.
[Adaptive Learning Rate](../07-Adaptive-Learning-Rate/notes.md) explains the internal mechanics of AdamW, Adafactor, and related methods.
[Regularization Methods](../08-Regularization-Methods/notes.md) explains how weight decay, dropout, early stopping, and penalties control model complexity.
Hyperparameter optimization asks how to choose those knobs under uncertainty and budget constraints.

It points forward to [Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md).
Schedules are time-varying hyperparameters.
This section teaches how to tune schedule families; the next section teaches the mathematical forms and training dynamics of those schedules themselves.

It also points outward to statistics and production ML.
Selection bias, cross-validation, repeated seeds, and test-set discipline are statistical ideas.
Experiment tracking, lineage, failed-trial handling, and reproducible reports are MLOps ideas.
HPO is where mathematical optimization, statistical estimation, and engineering discipline meet.

```text
HYPERPARAMETER OPTIMIZATION IN THE OPTIMIZATION CHAPTER
══════════════════════════════════════════════════════════════════════

  05 Stochastic Optimization
        noisy training dynamics
              |
              v
  07 Adaptive Learning Rate          08 Regularization Methods
        optimizer knobs                    complexity knobs
              \                              /
               \                            /
                v                          v
             09 Hyperparameter Optimization
                outer-loop noisy budgeted search
                         |
                         v
             10 Learning Rate Schedules
                time-varying hyperparameters

══════════════════════════════════════════════════════════════════════
```

The durable lesson is simple:
hyperparameter optimization is not a bag of tricks.
It is the disciplined optimization of an expensive, noisy validation objective under compute, statistical, and deployment constraints.

## References

1. Bergstra, J., and Bengio, Y. (2012). "Random Search for Hyper-Parameter Optimization." *Journal of Machine Learning Research*, 13(10), 281-305. https://jmlr.org/papers/v13/bergstra12a.html
2. Snoek, J., Larochelle, H., and Adams, R. P. (2012). "Practical Bayesian Optimization of Machine Learning Algorithms." *NeurIPS 2012*. https://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms
3. Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., and Talwalkar, A. (2018). "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization." *Journal of Machine Learning Research*. https://www.jmlr.org/beta/papers/v18/16-558.html
4. Falkner, S., Klein, A., and Hutter, F. (2018). "BOHB: Robust and Efficient Hyperparameter Optimization at Scale." *ICML 2018*. https://proceedings.mlr.press/v80/falkner18a.html
5. Jaderberg, M. et al. (2017). "Population Based Training of Neural Networks." https://arxiv.org/abs/1711.09846
6. Cawley, G. C., and Talbot, N. L. C. (2010). "On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation." *Journal of Machine Learning Research*. https://jmlr.csail.mit.edu/beta/papers/v11/cawley10a.html
7. Bergstra, J., Bardenet, R., Bengio, Y., and Kegl, B. (2011). "Algorithms for Hyper-Parameter Optimization." *NeurIPS 2011*. https://proceedings.neurips.cc/paper/4443-algorithms-for-hyper-parameter-optimization
8. scikit-learn Developers. "Tuning the hyper-parameters of an estimator." https://scikit-learn.org/1.5/modules/grid_search.html
9. Optuna Developers. "Optuna: A hyperparameter optimization framework." https://github.com/optuna/optuna
10. Ray Developers. "Ray Tune: Hyperparameter Tuning." https://docs.ray.io/en/master/tune/index.html
11. Amazon Web Services. "Understand the hyperparameter tuning strategies available in Amazon SageMaker AI." https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-how-it-works.html
