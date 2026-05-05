[Back to Curriculum](../../README.md) | [Previous: Calibration and Uncertainty](../02-Calibration-and-Uncertainty/notes.md) | [Next: Error Analysis and Ablations](../04-Error-Analysis-and-Ablations/notes.md)

---

# Robustness and Distribution Shift

> _"A model is reliable only on the distributions it survives."_

## Overview

Robustness evaluation measures how model risk changes when the test distribution, prompt
surface, subgroup, or adversary changes.

The chapter treats evaluation as a mathematical object: a controlled protocol that maps
model behavior into evidence with uncertainty. A result is not just a number; it is an
estimate produced by a task distribution, a scoring rule, and an aggregation rule.

This section is written in LaTeX Markdown. Inline mathematics uses `$...$`, and display
equations use `$$...$$`. The emphasis is practical but rigorous: every metric should be
linked to the statistical assumptions that make it meaningful.

## Prerequisites

- [Descriptive Statistics](../../07-Statistics/01-Descriptive-Statistics/notes.md)
- [Hypothesis Testing](../../07-Statistics/03-Hypothesis-Testing/notes.md)
- [Matrix Norms](../../03-Advanced-Linear-Algebra/06-Matrix-Norms/notes.md)
- [Calibration and Uncertainty](../02-Calibration-and-Uncertainty/notes.md)
- [RAG Math and Retrieval](../../15-Math-for-LLMs/12-RAG-Math-and-Retrieval/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Executable demonstrations for robustness and distribution shift |
| [exercises.ipynb](exercises.ipynb) | Graded practice for robustness and distribution shift |

## Learning Objectives

After completing this section, you will be able to:

- Define the core objects used in robustness and distribution shift
- Write empirical estimators for model scores and risks
- Attach uncertainty statements to finite evaluation results
- Separate protocol design from metric computation
- Identify common leakage, sampling, and aggregation failures
- Use synthetic experiments to test statistical intuitions
- Connect offline metrics to downstream AI decisions
- Explain where this section ends and where adjacent chapters begin
- Design exercises and notebooks that make the math executable
- Read modern LLM evaluation papers with sharper statistical judgment

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 Deployment changes the data distribution](#11-deployment-changes-the-data-distribution)
  - [1.2 Prompt surface as an input distribution](#12-prompt-surface-as-an-input-distribution)
  - [1.3 Rare tails dominate reliability risk](#13-rare-tails-dominate-reliability-risk)
  - [1.4 Robustness is not only adversarial accuracy](#14-robustness-is-not-only-adversarial-accuracy)
  - [1.5 Reliability budgets across shifts](#15-reliability-budgets-across-shifts)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Training and test distributions](#21-training-and-test-distributions)
  - [2.2 Covariate shift, label shift, and concept shift](#22-covariate-shift-label-shift-and-concept-shift)
  - [2.3 Subgroup risk](#23-subgroup-risk)
  - [2.4 Robust risk and worst-case risk](#24-robust-risk-and-worst-case-risk)
  - [2.5 Threat model and perturbation set](#25-threat-model-and-perturbation-set)
- [3. Measuring Shift](#3-measuring-shift)
  - [3.1 Two-sample tests](#31-two-sample-tests)
  - [3.2 MMD and Wasserstein previews](#32-mmd-and-wasserstein-previews)
  - [3.3 Embedding drift](#33-embedding-drift)
  - [3.4 Slice drift](#34-slice-drift)
  - [3.5 OOD score functions](#35-ood-score-functions)
- [4. Robustness Protocols](#4-robustness-protocols)
  - [4.1 Perturbation tests](#41-perturbation-tests)
  - [4.2 Stress tests](#42-stress-tests)
  - [4.3 Adversarial examples](#43-adversarial-examples)
  - [4.4 Common corruptions](#44-common-corruptions)
  - [4.5 Threat-model reporting](#45-threat-model-reporting)
- [5. Group and Tail Reliability](#5-group-and-tail-reliability)
  - [5.1 Worst-group accuracy](#51-worst-group-accuracy)
  - [5.2 Conditional value at risk](#52-conditional-value-at-risk)
  - [5.3 Tail loss](#53-tail-loss)
  - [5.4 Distributionally robust evaluation](#54-distributionally-robust-evaluation)
  - [5.5 Fairness and privacy side effects](#55-fairness-and-privacy-side-effects)
- [6. LLM Robustness](#6-llm-robustness)
  - [6.1 Prompt perturbations](#61-prompt-perturbations)
  - [6.2 Format sensitivity](#62-format-sensitivity)
  - [6.3 Multilingual and cultural shift](#63-multilingual-and-cultural-shift)
  - [6.4 Retrieval-context failures](#64-retrieval-context-failures)
  - [6.5 Long-context degradation](#65-long-context-degradation)
- [7. Boundary with Safety](#7-boundary-with-safety)
  - [7.1 Jailbreak preview](#71-jailbreak-preview)
  - [7.2 Red-team preview](#72-red-team-preview)
  - [7.3 Why alignment training is Chapter 18](#73-why-alignment-training-is-chapter-18)
  - [7.4 Why production drift is Chapter 19](#74-why-production-drift-is-chapter-19)
  - [7.5 Reliability handoff checklist](#75-reliability-handoff-checklist)
- [8. Common Mistakes](#8-common-mistakes)
- [9. Exercises](#9-exercises)
- [10. Why This Matters for AI](#10-why-this-matters-for-ai)
- [11. Conceptual Bridge](#11-conceptual-bridge)
- [References](#references)

---

## 1. Intuition

Intuition is the part of robustness and distribution shift that turns the approved TOC
into a concrete learning path. The subsections below keep the focus on Chapter 17's
canonical job: measurement, reliability, uncertainty, and decision support for AI
systems.

### 1.1 Deployment changes the data distribution

Deployment changes the data distribution is part of the canonical scope of robustness
and distribution shift. In this chapter, the object under study is not merely a dataset
or a model, but the full shifted evaluation distribution: the items, prompts, outputs,
graders, uncertainty statements, and decision rules that turn model behavior into
evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{R}_{\mathrm{shift}} = \frac{1}{n}\sum_{i=1}^n \ell(f_\theta(x), y).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For deployment changes the
data distribution, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in deployment changes the data distribution |
| Scoring rule | Exact formula for \ell(f_\theta(x), y) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report deployment changes the data distribution with item count, prompt protocol,
grader version, and a confidence interval.
- Use paired comparisons when two models answer the same evaluation items.
- Inspect at least one meaningful slice before concluding that the aggregate result is
reliable.
- Store raw outputs so future graders can be replayed without querying the model again.
- Document whether the metric is measuring capability, reliability, user value, or risk.

Non-examples:
- A leaderboard point estimate without sample size.
- A benchmark score produced with an undocumented prompt template.
- A model-graded result without judge identity, rubric, or agreement check.
- A robustness claim measured only on the easiest in-distribution examples.
- An online win declared before the randomization and logging checks pass.

Worked evaluation pattern for deployment changes the data distribution:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, deployment changes the data distribution is especially delicate because
the same model can be used with many prompts, decoding policies, tools, retrieval
contexts, and safety filters. The measured quantity is therefore a property of the
system configuration, not just the base weights.

| AI connection | Evaluation consequence |
| --- | --- |
| Prompting | Treat prompt templates as part of the protocol, not as invisible setup |
| Decoding | Temperature and sampling change both mean score and variance |
| Retrieval | Retrieved context creates an extra source of failure and leakage |
| Tool use | Tool errors need separate attribution from model reasoning errors |
| Safety layer | Guardrail behavior can improve risk metrics while changing capability metrics |

Implementation checklist:
- Use deterministic seeds for synthetic or sampled evaluation subsets.
- Print metric denominators, not only percentages.
- Keep missing, invalid, timeout, and refusal outcomes explicit.
- Prefer typed result records over loose CSV columns.
- Separate raw model outputs from normalized grader inputs.
- Track the smallest reproducible command that generated the result.
- Record whether the estimate is item-weighted, token-weighted, user-weighted, or
domain-weighted.
- Write the decision rule before seeing the final score whenever the result will guide a
release.

The mathematical habit to build is skepticism with structure. A score is not ignored
because it is noisy; it is interpreted through the design that produced it. Deployment
changes the data distribution is one place where that habit becomes concrete.

### 1.2 Prompt surface as an input distribution

Prompt surface as an input distribution is part of the canonical scope of robustness and
distribution shift. In this chapter, the object under study is not merely a dataset or a
model, but the full shifted evaluation distribution: the items, prompts, outputs,
graders, uncertainty statements, and decision rules that turn model behavior into
evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{R}_{\mathrm{shift}} = \frac{1}{n}\sum_{i=1}^n \ell(f_\theta(x), y).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For prompt surface as an
input distribution, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in prompt surface as an input distribution |
| Scoring rule | Exact formula for \ell(f_\theta(x), y) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report prompt surface as an input distribution with item count, prompt protocol,
grader version, and a confidence interval.
- Use paired comparisons when two models answer the same evaluation items.
- Inspect at least one meaningful slice before concluding that the aggregate result is
reliable.
- Store raw outputs so future graders can be replayed without querying the model again.
- Document whether the metric is measuring capability, reliability, user value, or risk.

Non-examples:
- A leaderboard point estimate without sample size.
- A benchmark score produced with an undocumented prompt template.
- A model-graded result without judge identity, rubric, or agreement check.
- A robustness claim measured only on the easiest in-distribution examples.
- An online win declared before the randomization and logging checks pass.

Worked evaluation pattern for prompt surface as an input distribution:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, prompt surface as an input distribution is especially delicate because
the same model can be used with many prompts, decoding policies, tools, retrieval
contexts, and safety filters. The measured quantity is therefore a property of the
system configuration, not just the base weights.

| AI connection | Evaluation consequence |
| --- | --- |
| Prompting | Treat prompt templates as part of the protocol, not as invisible setup |
| Decoding | Temperature and sampling change both mean score and variance |
| Retrieval | Retrieved context creates an extra source of failure and leakage |
| Tool use | Tool errors need separate attribution from model reasoning errors |
| Safety layer | Guardrail behavior can improve risk metrics while changing capability metrics |

Implementation checklist:
- Use deterministic seeds for synthetic or sampled evaluation subsets.
- Print metric denominators, not only percentages.
- Keep missing, invalid, timeout, and refusal outcomes explicit.
- Prefer typed result records over loose CSV columns.
- Separate raw model outputs from normalized grader inputs.
- Track the smallest reproducible command that generated the result.
- Record whether the estimate is item-weighted, token-weighted, user-weighted, or
domain-weighted.
- Write the decision rule before seeing the final score whenever the result will guide a
release.

The mathematical habit to build is skepticism with structure. A score is not ignored
because it is noisy; it is interpreted through the design that produced it. Prompt
surface as an input distribution is one place where that habit becomes concrete.

### 1.3 Rare tails dominate reliability risk

Rare tails dominate reliability risk is part of the canonical scope of robustness and
distribution shift. In this chapter, the object under study is not merely a dataset or a
model, but the full shifted evaluation distribution: the items, prompts, outputs,
graders, uncertainty statements, and decision rules that turn model behavior into
evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{R}_{\mathrm{shift}} = \frac{1}{n}\sum_{i=1}^n \ell(f_\theta(x), y).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For rare tails dominate
reliability risk, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in rare tails dominate reliability risk |
| Scoring rule | Exact formula for \ell(f_\theta(x), y) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report rare tails dominate reliability risk with item count, prompt protocol, grader
version, and a confidence interval.
- Use paired comparisons when two models answer the same evaluation items.
- Inspect at least one meaningful slice before concluding that the aggregate result is
reliable.
- Store raw outputs so future graders can be replayed without querying the model again.
- Document whether the metric is measuring capability, reliability, user value, or risk.

Non-examples:
- A leaderboard point estimate without sample size.
- A benchmark score produced with an undocumented prompt template.
- A model-graded result without judge identity, rubric, or agreement check.
- A robustness claim measured only on the easiest in-distribution examples.
- An online win declared before the randomization and logging checks pass.

Worked evaluation pattern for rare tails dominate reliability risk:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, rare tails dominate reliability risk is especially delicate because the
same model can be used with many prompts, decoding policies, tools, retrieval contexts,
and safety filters. The measured quantity is therefore a property of the system
configuration, not just the base weights.

| AI connection | Evaluation consequence |
| --- | --- |
| Prompting | Treat prompt templates as part of the protocol, not as invisible setup |
| Decoding | Temperature and sampling change both mean score and variance |
| Retrieval | Retrieved context creates an extra source of failure and leakage |
| Tool use | Tool errors need separate attribution from model reasoning errors |
| Safety layer | Guardrail behavior can improve risk metrics while changing capability metrics |

Implementation checklist:
- Use deterministic seeds for synthetic or sampled evaluation subsets.
- Print metric denominators, not only percentages.
- Keep missing, invalid, timeout, and refusal outcomes explicit.
- Prefer typed result records over loose CSV columns.
- Separate raw model outputs from normalized grader inputs.
- Track the smallest reproducible command that generated the result.
- Record whether the estimate is item-weighted, token-weighted, user-weighted, or
domain-weighted.
- Write the decision rule before seeing the final score whenever the result will guide a
release.

The mathematical habit to build is skepticism with structure. A score is not ignored
because it is noisy; it is interpreted through the design that produced it. Rare tails
dominate reliability risk is one place where that habit becomes concrete.

### 1.4 Robustness is not only adversarial accuracy

Robustness is not only adversarial accuracy is part of the canonical scope of robustness
and distribution shift. In this chapter, the object under study is not merely a dataset
or a model, but the full shifted evaluation distribution: the items, prompts, outputs,
graders, uncertainty statements, and decision rules that turn model behavior into
evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{R}_{\mathrm{shift}} = \frac{1}{n}\sum_{i=1}^n \ell(f_\theta(x), y).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For robustness is not only
adversarial accuracy, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in robustness is not only adversarial accuracy |
| Scoring rule | Exact formula for \ell(f_\theta(x), y) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report robustness is not only adversarial accuracy with item count, prompt protocol,
grader version, and a confidence interval.
- Use paired comparisons when two models answer the same evaluation items.
- Inspect at least one meaningful slice before concluding that the aggregate result is
reliable.
- Store raw outputs so future graders can be replayed without querying the model again.
- Document whether the metric is measuring capability, reliability, user value, or risk.

Non-examples:
- A leaderboard point estimate without sample size.
- A benchmark score produced with an undocumented prompt template.
- A model-graded result without judge identity, rubric, or agreement check.
- A robustness claim measured only on the easiest in-distribution examples.
- An online win declared before the randomization and logging checks pass.

Worked evaluation pattern for robustness is not only adversarial accuracy:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, robustness is not only adversarial accuracy is especially delicate
because the same model can be used with many prompts, decoding policies, tools,
retrieval contexts, and safety filters. The measured quantity is therefore a property of
the system configuration, not just the base weights.

| AI connection | Evaluation consequence |
| --- | --- |
| Prompting | Treat prompt templates as part of the protocol, not as invisible setup |
| Decoding | Temperature and sampling change both mean score and variance |
| Retrieval | Retrieved context creates an extra source of failure and leakage |
| Tool use | Tool errors need separate attribution from model reasoning errors |
| Safety layer | Guardrail behavior can improve risk metrics while changing capability metrics |

Implementation checklist:
- Use deterministic seeds for synthetic or sampled evaluation subsets.
- Print metric denominators, not only percentages.
- Keep missing, invalid, timeout, and refusal outcomes explicit.
- Prefer typed result records over loose CSV columns.
- Separate raw model outputs from normalized grader inputs.
- Track the smallest reproducible command that generated the result.
- Record whether the estimate is item-weighted, token-weighted, user-weighted, or
domain-weighted.
- Write the decision rule before seeing the final score whenever the result will guide a
release.

The mathematical habit to build is skepticism with structure. A score is not ignored
because it is noisy; it is interpreted through the design that produced it. Robustness
is not only adversarial accuracy is one place where that habit becomes concrete.

### 1.5 Reliability budgets across shifts

Reliability budgets across shifts is part of the canonical scope of robustness and
distribution shift. In this chapter, the object under study is not merely a dataset or a
model, but the full shifted evaluation distribution: the items, prompts, outputs,
graders, uncertainty statements, and decision rules that turn model behavior into
evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{R}_{\mathrm{shift}} = \frac{1}{n}\sum_{i=1}^n \ell(f_\theta(x), y).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For reliability budgets
across shifts, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in reliability budgets across shifts |
| Scoring rule | Exact formula for \ell(f_\theta(x), y) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report reliability budgets across shifts with item count, prompt protocol, grader
version, and a confidence interval.
- Use paired comparisons when two models answer the same evaluation items.
- Inspect at least one meaningful slice before concluding that the aggregate result is
reliable.
- Store raw outputs so future graders can be replayed without querying the model again.
- Document whether the metric is measuring capability, reliability, user value, or risk.

Non-examples:
- A leaderboard point estimate without sample size.
- A benchmark score produced with an undocumented prompt template.
- A model-graded result without judge identity, rubric, or agreement check.
- A robustness claim measured only on the easiest in-distribution examples.
- An online win declared before the randomization and logging checks pass.

Worked evaluation pattern for reliability budgets across shifts:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, reliability budgets across shifts is especially delicate because the
same model can be used with many prompts, decoding policies, tools, retrieval contexts,
and safety filters. The measured quantity is therefore a property of the system
configuration, not just the base weights.

| AI connection | Evaluation consequence |
| --- | --- |
| Prompting | Treat prompt templates as part of the protocol, not as invisible setup |
| Decoding | Temperature and sampling change both mean score and variance |
| Retrieval | Retrieved context creates an extra source of failure and leakage |
| Tool use | Tool errors need separate attribution from model reasoning errors |
| Safety layer | Guardrail behavior can improve risk metrics while changing capability metrics |

Implementation checklist:
- Use deterministic seeds for synthetic or sampled evaluation subsets.
- Print metric denominators, not only percentages.
- Keep missing, invalid, timeout, and refusal outcomes explicit.
- Prefer typed result records over loose CSV columns.
- Separate raw model outputs from normalized grader inputs.
- Track the smallest reproducible command that generated the result.
- Record whether the estimate is item-weighted, token-weighted, user-weighted, or
domain-weighted.
- Write the decision rule before seeing the final score whenever the result will guide a
release.

The mathematical habit to build is skepticism with structure. A score is not ignored
because it is noisy; it is interpreted through the design that produced it. Reliability
budgets across shifts is one place where that habit becomes concrete.

## 2. Formal Definitions

Formal Definitions is the part of robustness and distribution shift that turns the
approved TOC into a concrete learning path. The subsections below keep the focus on
Chapter 17's canonical job: measurement, reliability, uncertainty, and decision support
for AI systems.

### 2.1 Training and test distributions

Training and test distributions is part of the canonical scope of robustness and
distribution shift. In this chapter, the object under study is not merely a dataset or a
model, but the full shifted evaluation distribution: the items, prompts, outputs,
graders, uncertainty statements, and decision rules that turn model behavior into
evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{R}_{\mathrm{shift}} = \frac{1}{n}\sum_{i=1}^n \ell(f_\theta(x), y).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For training and test
distributions, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in training and test distributions |
| Scoring rule | Exact formula for \ell(f_\theta(x), y) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report training and test distributions with item count, prompt protocol, grader
version, and a confidence interval.
- Use paired comparisons when two models answer the same evaluation items.
- Inspect at least one meaningful slice before concluding that the aggregate result is
reliable.
- Store raw outputs so future graders can be replayed without querying the model again.
- Document whether the metric is measuring capability, reliability, user value, or risk.

Non-examples:
- A leaderboard point estimate without sample size.
- A benchmark score produced with an undocumented prompt template.
- A model-graded result without judge identity, rubric, or agreement check.
- A robustness claim measured only on the easiest in-distribution examples.
- An online win declared before the randomization and logging checks pass.

Worked evaluation pattern for training and test distributions:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, training and test distributions is especially delicate because the same
model can be used with many prompts, decoding policies, tools, retrieval contexts, and
safety filters. The measured quantity is therefore a property of the system
configuration, not just the base weights.

| AI connection | Evaluation consequence |
| --- | --- |
| Prompting | Treat prompt templates as part of the protocol, not as invisible setup |
| Decoding | Temperature and sampling change both mean score and variance |
| Retrieval | Retrieved context creates an extra source of failure and leakage |
| Tool use | Tool errors need separate attribution from model reasoning errors |
| Safety layer | Guardrail behavior can improve risk metrics while changing capability metrics |

Implementation checklist:
- Use deterministic seeds for synthetic or sampled evaluation subsets.
- Print metric denominators, not only percentages.
- Keep missing, invalid, timeout, and refusal outcomes explicit.
- Prefer typed result records over loose CSV columns.
- Separate raw model outputs from normalized grader inputs.
- Track the smallest reproducible command that generated the result.
- Record whether the estimate is item-weighted, token-weighted, user-weighted, or
domain-weighted.
- Write the decision rule before seeing the final score whenever the result will guide a
release.

The mathematical habit to build is skepticism with structure. A score is not ignored
because it is noisy; it is interpreted through the design that produced it. Training and
test distributions is one place where that habit becomes concrete.

### 2.2 Covariate shift, label shift, and concept shift

Covariate shift, label shift, and concept shift is part of the canonical scope of
robustness and distribution shift. In this chapter, the object under study is not merely
a dataset or a model, but the full shifted evaluation distribution: the items, prompts,
outputs, graders, uncertainty statements, and decision rules that turn model behavior
into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{R}_{\mathrm{shift}} = \frac{1}{n}\sum_{i=1}^n \ell(f_\theta(x), y).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For covariate shift, label
shift, and concept shift, those choices determine whether the reported number is
evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in covariate shift, label shift, and concept shift |
| Scoring rule | Exact formula for \ell(f_\theta(x), y) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report covariate shift, label shift, and concept shift with item count, prompt
protocol, grader version, and a confidence interval.
- Use paired comparisons when two models answer the same evaluation items.
- Inspect at least one meaningful slice before concluding that the aggregate result is
reliable.
- Store raw outputs so future graders can be replayed without querying the model again.
- Document whether the metric is measuring capability, reliability, user value, or risk.

Non-examples:
- A leaderboard point estimate without sample size.
- A benchmark score produced with an undocumented prompt template.
- A model-graded result without judge identity, rubric, or agreement check.
- A robustness claim measured only on the easiest in-distribution examples.
- An online win declared before the randomization and logging checks pass.

Worked evaluation pattern for covariate shift, label shift, and concept shift:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, covariate shift, label shift, and concept shift is especially delicate
because the same model can be used with many prompts, decoding policies, tools,
retrieval contexts, and safety filters. The measured quantity is therefore a property of
the system configuration, not just the base weights.

| AI connection | Evaluation consequence |
| --- | --- |
| Prompting | Treat prompt templates as part of the protocol, not as invisible setup |
| Decoding | Temperature and sampling change both mean score and variance |
| Retrieval | Retrieved context creates an extra source of failure and leakage |
| Tool use | Tool errors need separate attribution from model reasoning errors |
| Safety layer | Guardrail behavior can improve risk metrics while changing capability metrics |

Implementation checklist:
- Use deterministic seeds for synthetic or sampled evaluation subsets.
- Print metric denominators, not only percentages.
- Keep missing, invalid, timeout, and refusal outcomes explicit.
- Prefer typed result records over loose CSV columns.
- Separate raw model outputs from normalized grader inputs.
- Track the smallest reproducible command that generated the result.
- Record whether the estimate is item-weighted, token-weighted, user-weighted, or
domain-weighted.
- Write the decision rule before seeing the final score whenever the result will guide a
release.

The mathematical habit to build is skepticism with structure. A score is not ignored
because it is noisy; it is interpreted through the design that produced it. Covariate
shift, label shift, and concept shift is one place where that habit becomes concrete.

### 2.3 Subgroup risk

Subgroup risk is part of the canonical scope of robustness and distribution shift. In
this chapter, the object under study is not merely a dataset or a model, but the full
shifted evaluation distribution: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{R}_{\mathrm{shift}} = \frac{1}{n}\sum_{i=1}^n \ell(f_\theta(x), y).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For subgroup risk, those
choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in subgroup risk |
| Scoring rule | Exact formula for \ell(f_\theta(x), y) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report subgroup risk with item count, prompt protocol, grader version, and a
confidence interval.
- Use paired comparisons when two models answer the same evaluation items.
- Inspect at least one meaningful slice before concluding that the aggregate result is
reliable.
- Store raw outputs so future graders can be replayed without querying the model again.
- Document whether the metric is measuring capability, reliability, user value, or risk.

Non-examples:
- A leaderboard point estimate without sample size.
- A benchmark score produced with an undocumented prompt template.
- A model-graded result without judge identity, rubric, or agreement check.
- A robustness claim measured only on the easiest in-distribution examples.
- An online win declared before the randomization and logging checks pass.

Worked evaluation pattern for subgroup risk:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, subgroup risk is especially delicate because the same model can be used
with many prompts, decoding policies, tools, retrieval contexts, and safety filters. The
measured quantity is therefore a property of the system configuration, not just the base
weights.

| AI connection | Evaluation consequence |
| --- | --- |
| Prompting | Treat prompt templates as part of the protocol, not as invisible setup |
| Decoding | Temperature and sampling change both mean score and variance |
| Retrieval | Retrieved context creates an extra source of failure and leakage |
| Tool use | Tool errors need separate attribution from model reasoning errors |
| Safety layer | Guardrail behavior can improve risk metrics while changing capability metrics |

Implementation checklist:
- Use deterministic seeds for synthetic or sampled evaluation subsets.
- Print metric denominators, not only percentages.
- Keep missing, invalid, timeout, and refusal outcomes explicit.
- Prefer typed result records over loose CSV columns.
- Separate raw model outputs from normalized grader inputs.
- Track the smallest reproducible command that generated the result.
- Record whether the estimate is item-weighted, token-weighted, user-weighted, or
domain-weighted.
- Write the decision rule before seeing the final score whenever the result will guide a
release.

The mathematical habit to build is skepticism with structure. A score is not ignored
because it is noisy; it is interpreted through the design that produced it. Subgroup
risk is one place where that habit becomes concrete.

### 2.4 Robust risk and worst-case risk

Robust risk and worst-case risk is part of the canonical scope of robustness and
distribution shift. In this chapter, the object under study is not merely a dataset or a
model, but the full shifted evaluation distribution: the items, prompts, outputs,
graders, uncertainty statements, and decision rules that turn model behavior into
evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{R}_{\mathrm{shift}} = \frac{1}{n}\sum_{i=1}^n \ell(f_\theta(x), y).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For robust risk and worst-
case risk, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in robust risk and worst-case risk |
| Scoring rule | Exact formula for \ell(f_\theta(x), y) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report robust risk and worst-case risk with item count, prompt protocol, grader
version, and a confidence interval.
- Use paired comparisons when two models answer the same evaluation items.
- Inspect at least one meaningful slice before concluding that the aggregate result is
reliable.
- Store raw outputs so future graders can be replayed without querying the model again.
- Document whether the metric is measuring capability, reliability, user value, or risk.

Non-examples:
- A leaderboard point estimate without sample size.
- A benchmark score produced with an undocumented prompt template.
- A model-graded result without judge identity, rubric, or agreement check.
- A robustness claim measured only on the easiest in-distribution examples.
- An online win declared before the randomization and logging checks pass.

Worked evaluation pattern for robust risk and worst-case risk:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, robust risk and worst-case risk is especially delicate because the same
model can be used with many prompts, decoding policies, tools, retrieval contexts, and
safety filters. The measured quantity is therefore a property of the system
configuration, not just the base weights.

| AI connection | Evaluation consequence |
| --- | --- |
| Prompting | Treat prompt templates as part of the protocol, not as invisible setup |
| Decoding | Temperature and sampling change both mean score and variance |
| Retrieval | Retrieved context creates an extra source of failure and leakage |
| Tool use | Tool errors need separate attribution from model reasoning errors |
| Safety layer | Guardrail behavior can improve risk metrics while changing capability metrics |

Implementation checklist:
- Use deterministic seeds for synthetic or sampled evaluation subsets.
- Print metric denominators, not only percentages.
- Keep missing, invalid, timeout, and refusal outcomes explicit.
- Prefer typed result records over loose CSV columns.
- Separate raw model outputs from normalized grader inputs.
- Track the smallest reproducible command that generated the result.
- Record whether the estimate is item-weighted, token-weighted, user-weighted, or
domain-weighted.
- Write the decision rule before seeing the final score whenever the result will guide a
release.

The mathematical habit to build is skepticism with structure. A score is not ignored
because it is noisy; it is interpreted through the design that produced it. Robust risk
and worst-case risk is one place where that habit becomes concrete.

### 2.5 Threat model and perturbation set

Threat model and perturbation set is part of the canonical scope of robustness and
distribution shift. In this chapter, the object under study is not merely a dataset or a
model, but the full shifted evaluation distribution: the items, prompts, outputs,
graders, uncertainty statements, and decision rules that turn model behavior into
evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{R}_{\mathrm{shift}} = \frac{1}{n}\sum_{i=1}^n \ell(f_\theta(x), y).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For threat model and
perturbation set, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in threat model and perturbation set |
| Scoring rule | Exact formula for \ell(f_\theta(x), y) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report threat model and perturbation set with item count, prompt protocol, grader
version, and a confidence interval.
- Use paired comparisons when two models answer the same evaluation items.
- Inspect at least one meaningful slice before concluding that the aggregate result is
reliable.
- Store raw outputs so future graders can be replayed without querying the model again.
- Document whether the metric is measuring capability, reliability, user value, or risk.

Non-examples:
- A leaderboard point estimate without sample size.
- A benchmark score produced with an undocumented prompt template.
- A model-graded result without judge identity, rubric, or agreement check.
- A robustness claim measured only on the easiest in-distribution examples.
- An online win declared before the randomization and logging checks pass.

Worked evaluation pattern for threat model and perturbation set:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, threat model and perturbation set is especially delicate because the
same model can be used with many prompts, decoding policies, tools, retrieval contexts,
and safety filters. The measured quantity is therefore a property of the system
configuration, not just the base weights.

| AI connection | Evaluation consequence |
| --- | --- |
| Prompting | Treat prompt templates as part of the protocol, not as invisible setup |
| Decoding | Temperature and sampling change both mean score and variance |
| Retrieval | Retrieved context creates an extra source of failure and leakage |
| Tool use | Tool errors need separate attribution from model reasoning errors |
| Safety layer | Guardrail behavior can improve risk metrics while changing capability metrics |

Implementation checklist:
- Use deterministic seeds for synthetic or sampled evaluation subsets.
- Print metric denominators, not only percentages.
- Keep missing, invalid, timeout, and refusal outcomes explicit.
- Prefer typed result records over loose CSV columns.
- Separate raw model outputs from normalized grader inputs.
- Track the smallest reproducible command that generated the result.
- Record whether the estimate is item-weighted, token-weighted, user-weighted, or
domain-weighted.
- Write the decision rule before seeing the final score whenever the result will guide a
release.

The mathematical habit to build is skepticism with structure. A score is not ignored
because it is noisy; it is interpreted through the design that produced it. Threat model
and perturbation set is one place where that habit becomes concrete.

## 3. Measuring Shift

Measuring Shift is the part of robustness and distribution shift that turns the approved
TOC into a concrete learning path. The subsections below keep the focus on Chapter 17's
canonical job: measurement, reliability, uncertainty, and decision support for AI
systems.

### 3.1 Two-sample tests

Two-sample tests is part of the canonical scope of robustness and distribution shift. In
this chapter, the object under study is not merely a dataset or a model, but the full
shifted evaluation distribution: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{R}_{\mathrm{shift}} = \frac{1}{n}\sum_{i=1}^n \ell(f_\theta(x), y).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For two-sample tests, those
choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in two-sample tests |
| Scoring rule | Exact formula for \ell(f_\theta(x), y) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report two-sample tests with item count, prompt protocol, grader version, and a
confidence interval.
- Use paired comparisons when two models answer the same evaluation items.
- Inspect at least one meaningful slice before concluding that the aggregate result is
reliable.
- Store raw outputs so future graders can be replayed without querying the model again.
- Document whether the metric is measuring capability, reliability, user value, or risk.

Non-examples:
- A leaderboard point estimate without sample size.
- A benchmark score produced with an undocumented prompt template.
- A model-graded result without judge identity, rubric, or agreement check.
- A robustness claim measured only on the easiest in-distribution examples.
- An online win declared before the randomization and logging checks pass.

Worked evaluation pattern for two-sample tests:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, two-sample tests is especially delicate because the same model can be
used with many prompts, decoding policies, tools, retrieval contexts, and safety
filters. The measured quantity is therefore a property of the system configuration, not
just the base weights.

| AI connection | Evaluation consequence |
| --- | --- |
| Prompting | Treat prompt templates as part of the protocol, not as invisible setup |
| Decoding | Temperature and sampling change both mean score and variance |
| Retrieval | Retrieved context creates an extra source of failure and leakage |
| Tool use | Tool errors need separate attribution from model reasoning errors |
| Safety layer | Guardrail behavior can improve risk metrics while changing capability metrics |

Implementation checklist:
- Use deterministic seeds for synthetic or sampled evaluation subsets.
- Print metric denominators, not only percentages.
- Keep missing, invalid, timeout, and refusal outcomes explicit.
- Prefer typed result records over loose CSV columns.
- Separate raw model outputs from normalized grader inputs.
- Track the smallest reproducible command that generated the result.
- Record whether the estimate is item-weighted, token-weighted, user-weighted, or
domain-weighted.
- Write the decision rule before seeing the final score whenever the result will guide a
release.

The mathematical habit to build is skepticism with structure. A score is not ignored
because it is noisy; it is interpreted through the design that produced it. Two-sample
tests is one place where that habit becomes concrete.

### 3.2 MMD and Wasserstein previews

MMD and Wasserstein previews is part of the canonical scope of robustness and
distribution shift. In this chapter, the object under study is not merely a dataset or a
model, but the full shifted evaluation distribution: the items, prompts, outputs,
graders, uncertainty statements, and decision rules that turn model behavior into
evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{R}_{\mathrm{shift}} = \frac{1}{n}\sum_{i=1}^n \ell(f_\theta(x), y).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For mmd and wasserstein
previews, those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in mmd and wasserstein previews |
| Scoring rule | Exact formula for \ell(f_\theta(x), y) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report mmd and wasserstein previews with item count, prompt protocol, grader version,
and a confidence interval.
- Use paired comparisons when two models answer the same evaluation items.
- Inspect at least one meaningful slice before concluding that the aggregate result is
reliable.
- Store raw outputs so future graders can be replayed without querying the model again.
- Document whether the metric is measuring capability, reliability, user value, or risk.

Non-examples:
- A leaderboard point estimate without sample size.
- A benchmark score produced with an undocumented prompt template.
- A model-graded result without judge identity, rubric, or agreement check.
- A robustness claim measured only on the easiest in-distribution examples.
- An online win declared before the randomization and logging checks pass.

Worked evaluation pattern for mmd and wasserstein previews:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, mmd and wasserstein previews is especially delicate because the same
model can be used with many prompts, decoding policies, tools, retrieval contexts, and
safety filters. The measured quantity is therefore a property of the system
configuration, not just the base weights.

| AI connection | Evaluation consequence |
| --- | --- |
| Prompting | Treat prompt templates as part of the protocol, not as invisible setup |
| Decoding | Temperature and sampling change both mean score and variance |
| Retrieval | Retrieved context creates an extra source of failure and leakage |
| Tool use | Tool errors need separate attribution from model reasoning errors |
| Safety layer | Guardrail behavior can improve risk metrics while changing capability metrics |

Implementation checklist:
- Use deterministic seeds for synthetic or sampled evaluation subsets.
- Print metric denominators, not only percentages.
- Keep missing, invalid, timeout, and refusal outcomes explicit.
- Prefer typed result records over loose CSV columns.
- Separate raw model outputs from normalized grader inputs.
- Track the smallest reproducible command that generated the result.
- Record whether the estimate is item-weighted, token-weighted, user-weighted, or
domain-weighted.
- Write the decision rule before seeing the final score whenever the result will guide a
release.

The mathematical habit to build is skepticism with structure. A score is not ignored
because it is noisy; it is interpreted through the design that produced it. MMD and
Wasserstein previews is one place where that habit becomes concrete.

### 3.3 Embedding drift

Embedding drift is part of the canonical scope of robustness and distribution shift. In
this chapter, the object under study is not merely a dataset or a model, but the full
shifted evaluation distribution: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{R}_{\mathrm{shift}} = \frac{1}{n}\sum_{i=1}^n \ell(f_\theta(x), y).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For embedding drift, those
choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in embedding drift |
| Scoring rule | Exact formula for \ell(f_\theta(x), y) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report embedding drift with item count, prompt protocol, grader version, and a
confidence interval.
- Use paired comparisons when two models answer the same evaluation items.
- Inspect at least one meaningful slice before concluding that the aggregate result is
reliable.
- Store raw outputs so future graders can be replayed without querying the model again.
- Document whether the metric is measuring capability, reliability, user value, or risk.

Non-examples:
- A leaderboard point estimate without sample size.
- A benchmark score produced with an undocumented prompt template.
- A model-graded result without judge identity, rubric, or agreement check.
- A robustness claim measured only on the easiest in-distribution examples.
- An online win declared before the randomization and logging checks pass.

Worked evaluation pattern for embedding drift:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, embedding drift is especially delicate because the same model can be
used with many prompts, decoding policies, tools, retrieval contexts, and safety
filters. The measured quantity is therefore a property of the system configuration, not
just the base weights.

| AI connection | Evaluation consequence |
| --- | --- |
| Prompting | Treat prompt templates as part of the protocol, not as invisible setup |
| Decoding | Temperature and sampling change both mean score and variance |
| Retrieval | Retrieved context creates an extra source of failure and leakage |
| Tool use | Tool errors need separate attribution from model reasoning errors |
| Safety layer | Guardrail behavior can improve risk metrics while changing capability metrics |

Implementation checklist:
- Use deterministic seeds for synthetic or sampled evaluation subsets.
- Print metric denominators, not only percentages.
- Keep missing, invalid, timeout, and refusal outcomes explicit.
- Prefer typed result records over loose CSV columns.
- Separate raw model outputs from normalized grader inputs.
- Track the smallest reproducible command that generated the result.
- Record whether the estimate is item-weighted, token-weighted, user-weighted, or
domain-weighted.
- Write the decision rule before seeing the final score whenever the result will guide a
release.

The mathematical habit to build is skepticism with structure. A score is not ignored
because it is noisy; it is interpreted through the design that produced it. Embedding
drift is one place where that habit becomes concrete.

### 3.4 Slice drift

Slice drift is part of the canonical scope of robustness and distribution shift. In this
chapter, the object under study is not merely a dataset or a model, but the full shifted
evaluation distribution: the items, prompts, outputs, graders, uncertainty statements,
and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{R}_{\mathrm{shift}} = \frac{1}{n}\sum_{i=1}^n \ell(f_\theta(x), y).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For slice drift, those
choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in slice drift |
| Scoring rule | Exact formula for \ell(f_\theta(x), y) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report slice drift with item count, prompt protocol, grader version, and a confidence
interval.
- Use paired comparisons when two models answer the same evaluation items.
- Inspect at least one meaningful slice before concluding that the aggregate result is
reliable.
- Store raw outputs so future graders can be replayed without querying the model again.
- Document whether the metric is measuring capability, reliability, user value, or risk.

Non-examples:
- A leaderboard point estimate without sample size.
- A benchmark score produced with an undocumented prompt template.
- A model-graded result without judge identity, rubric, or agreement check.
- A robustness claim measured only on the easiest in-distribution examples.
- An online win declared before the randomization and logging checks pass.

Worked evaluation pattern for slice drift:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, slice drift is especially delicate because the same model can be used
with many prompts, decoding policies, tools, retrieval contexts, and safety filters. The
measured quantity is therefore a property of the system configuration, not just the base
weights.

| AI connection | Evaluation consequence |
| --- | --- |
| Prompting | Treat prompt templates as part of the protocol, not as invisible setup |
| Decoding | Temperature and sampling change both mean score and variance |
| Retrieval | Retrieved context creates an extra source of failure and leakage |
| Tool use | Tool errors need separate attribution from model reasoning errors |
| Safety layer | Guardrail behavior can improve risk metrics while changing capability metrics |

Implementation checklist:
- Use deterministic seeds for synthetic or sampled evaluation subsets.
- Print metric denominators, not only percentages.
- Keep missing, invalid, timeout, and refusal outcomes explicit.
- Prefer typed result records over loose CSV columns.
- Separate raw model outputs from normalized grader inputs.
- Track the smallest reproducible command that generated the result.
- Record whether the estimate is item-weighted, token-weighted, user-weighted, or
domain-weighted.
- Write the decision rule before seeing the final score whenever the result will guide a
release.

The mathematical habit to build is skepticism with structure. A score is not ignored
because it is noisy; it is interpreted through the design that produced it. Slice drift
is one place where that habit becomes concrete.

### 3.5 OOD score functions

OOD score functions is part of the canonical scope of robustness and distribution shift.
In this chapter, the object under study is not merely a dataset or a model, but the full
shifted evaluation distribution: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{R}_{\mathrm{shift}} = \frac{1}{n}\sum_{i=1}^n \ell(f_\theta(x), y).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For ood score functions,
those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in ood score functions |
| Scoring rule | Exact formula for \ell(f_\theta(x), y) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report ood score functions with item count, prompt protocol, grader version, and a
confidence interval.
- Use paired comparisons when two models answer the same evaluation items.
- Inspect at least one meaningful slice before concluding that the aggregate result is
reliable.
- Store raw outputs so future graders can be replayed without querying the model again.
- Document whether the metric is measuring capability, reliability, user value, or risk.

Non-examples:
- A leaderboard point estimate without sample size.
- A benchmark score produced with an undocumented prompt template.
- A model-graded result without judge identity, rubric, or agreement check.
- A robustness claim measured only on the easiest in-distribution examples.
- An online win declared before the randomization and logging checks pass.

Worked evaluation pattern for ood score functions:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, ood score functions is especially delicate because the same model can be
used with many prompts, decoding policies, tools, retrieval contexts, and safety
filters. The measured quantity is therefore a property of the system configuration, not
just the base weights.

| AI connection | Evaluation consequence |
| --- | --- |
| Prompting | Treat prompt templates as part of the protocol, not as invisible setup |
| Decoding | Temperature and sampling change both mean score and variance |
| Retrieval | Retrieved context creates an extra source of failure and leakage |
| Tool use | Tool errors need separate attribution from model reasoning errors |
| Safety layer | Guardrail behavior can improve risk metrics while changing capability metrics |

Implementation checklist:
- Use deterministic seeds for synthetic or sampled evaluation subsets.
- Print metric denominators, not only percentages.
- Keep missing, invalid, timeout, and refusal outcomes explicit.
- Prefer typed result records over loose CSV columns.
- Separate raw model outputs from normalized grader inputs.
- Track the smallest reproducible command that generated the result.
- Record whether the estimate is item-weighted, token-weighted, user-weighted, or
domain-weighted.
- Write the decision rule before seeing the final score whenever the result will guide a
release.

The mathematical habit to build is skepticism with structure. A score is not ignored
because it is noisy; it is interpreted through the design that produced it. OOD score
functions is one place where that habit becomes concrete.

## 4. Robustness Protocols

Robustness Protocols is the part of robustness and distribution shift that turns the
approved TOC into a concrete learning path. The subsections below keep the focus on
Chapter 17's canonical job: measurement, reliability, uncertainty, and decision support
for AI systems.

### 4.1 Perturbation tests

Perturbation tests is part of the canonical scope of robustness and distribution shift.
In this chapter, the object under study is not merely a dataset or a model, but the full
shifted evaluation distribution: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{R}_{\mathrm{shift}} = \frac{1}{n}\sum_{i=1}^n \ell(f_\theta(x), y).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For perturbation tests,
those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in perturbation tests |
| Scoring rule | Exact formula for \ell(f_\theta(x), y) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report perturbation tests with item count, prompt protocol, grader version, and a
confidence interval.
- Use paired comparisons when two models answer the same evaluation items.
- Inspect at least one meaningful slice before concluding that the aggregate result is
reliable.
- Store raw outputs so future graders can be replayed without querying the model again.
- Document whether the metric is measuring capability, reliability, user value, or risk.

Non-examples:
- A leaderboard point estimate without sample size.
- A benchmark score produced with an undocumented prompt template.
- A model-graded result without judge identity, rubric, or agreement check.
- A robustness claim measured only on the easiest in-distribution examples.
- An online win declared before the randomization and logging checks pass.

Worked evaluation pattern for perturbation tests:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, perturbation tests is especially delicate because the same model can be
used with many prompts, decoding policies, tools, retrieval contexts, and safety
filters. The measured quantity is therefore a property of the system configuration, not
just the base weights.

| AI connection | Evaluation consequence |
| --- | --- |
| Prompting | Treat prompt templates as part of the protocol, not as invisible setup |
| Decoding | Temperature and sampling change both mean score and variance |
| Retrieval | Retrieved context creates an extra source of failure and leakage |
| Tool use | Tool errors need separate attribution from model reasoning errors |
| Safety layer | Guardrail behavior can improve risk metrics while changing capability metrics |

Implementation checklist:
- Use deterministic seeds for synthetic or sampled evaluation subsets.
- Print metric denominators, not only percentages.
- Keep missing, invalid, timeout, and refusal outcomes explicit.
- Prefer typed result records over loose CSV columns.
- Separate raw model outputs from normalized grader inputs.
- Track the smallest reproducible command that generated the result.
- Record whether the estimate is item-weighted, token-weighted, user-weighted, or
domain-weighted.
- Write the decision rule before seeing the final score whenever the result will guide a
release.

The mathematical habit to build is skepticism with structure. A score is not ignored
because it is noisy; it is interpreted through the design that produced it. Perturbation
tests is one place where that habit becomes concrete.

### 4.2 Stress tests

Stress tests is part of the canonical scope of robustness and distribution shift. In
this chapter, the object under study is not merely a dataset or a model, but the full
shifted evaluation distribution: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{R}_{\mathrm{shift}} = \frac{1}{n}\sum_{i=1}^n \ell(f_\theta(x), y).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For stress tests, those
choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in stress tests |
| Scoring rule | Exact formula for \ell(f_\theta(x), y) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report stress tests with item count, prompt protocol, grader version, and a confidence
interval.
- Use paired comparisons when two models answer the same evaluation items.
- Inspect at least one meaningful slice before concluding that the aggregate result is
reliable.
- Store raw outputs so future graders can be replayed without querying the model again.
- Document whether the metric is measuring capability, reliability, user value, or risk.

Non-examples:
- A leaderboard point estimate without sample size.
- A benchmark score produced with an undocumented prompt template.
- A model-graded result without judge identity, rubric, or agreement check.
- A robustness claim measured only on the easiest in-distribution examples.
- An online win declared before the randomization and logging checks pass.

Worked evaluation pattern for stress tests:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, stress tests is especially delicate because the same model can be used
with many prompts, decoding policies, tools, retrieval contexts, and safety filters. The
measured quantity is therefore a property of the system configuration, not just the base
weights.

| AI connection | Evaluation consequence |
| --- | --- |
| Prompting | Treat prompt templates as part of the protocol, not as invisible setup |
| Decoding | Temperature and sampling change both mean score and variance |
| Retrieval | Retrieved context creates an extra source of failure and leakage |
| Tool use | Tool errors need separate attribution from model reasoning errors |
| Safety layer | Guardrail behavior can improve risk metrics while changing capability metrics |

Implementation checklist:
- Use deterministic seeds for synthetic or sampled evaluation subsets.
- Print metric denominators, not only percentages.
- Keep missing, invalid, timeout, and refusal outcomes explicit.
- Prefer typed result records over loose CSV columns.
- Separate raw model outputs from normalized grader inputs.
- Track the smallest reproducible command that generated the result.
- Record whether the estimate is item-weighted, token-weighted, user-weighted, or
domain-weighted.
- Write the decision rule before seeing the final score whenever the result will guide a
release.

The mathematical habit to build is skepticism with structure. A score is not ignored
because it is noisy; it is interpreted through the design that produced it. Stress tests
is one place where that habit becomes concrete.

### 4.3 Adversarial examples

Adversarial examples is part of the canonical scope of robustness and distribution
shift. In this chapter, the object under study is not merely a dataset or a model, but
the full shifted evaluation distribution: the items, prompts, outputs, graders,
uncertainty statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{R}_{\mathrm{shift}} = \frac{1}{n}\sum_{i=1}^n \ell(f_\theta(x), y).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For adversarial examples,
those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in adversarial examples |
| Scoring rule | Exact formula for \ell(f_\theta(x), y) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report adversarial examples with item count, prompt protocol, grader version, and a
confidence interval.
- Use paired comparisons when two models answer the same evaluation items.
- Inspect at least one meaningful slice before concluding that the aggregate result is
reliable.
- Store raw outputs so future graders can be replayed without querying the model again.
- Document whether the metric is measuring capability, reliability, user value, or risk.

Non-examples:
- A leaderboard point estimate without sample size.
- A benchmark score produced with an undocumented prompt template.
- A model-graded result without judge identity, rubric, or agreement check.
- A robustness claim measured only on the easiest in-distribution examples.
- An online win declared before the randomization and logging checks pass.

Worked evaluation pattern for adversarial examples:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, adversarial examples is especially delicate because the same model can
be used with many prompts, decoding policies, tools, retrieval contexts, and safety
filters. The measured quantity is therefore a property of the system configuration, not
just the base weights.

| AI connection | Evaluation consequence |
| --- | --- |
| Prompting | Treat prompt templates as part of the protocol, not as invisible setup |
| Decoding | Temperature and sampling change both mean score and variance |
| Retrieval | Retrieved context creates an extra source of failure and leakage |
| Tool use | Tool errors need separate attribution from model reasoning errors |
| Safety layer | Guardrail behavior can improve risk metrics while changing capability metrics |

Implementation checklist:
- Use deterministic seeds for synthetic or sampled evaluation subsets.
- Print metric denominators, not only percentages.
- Keep missing, invalid, timeout, and refusal outcomes explicit.
- Prefer typed result records over loose CSV columns.
- Separate raw model outputs from normalized grader inputs.
- Track the smallest reproducible command that generated the result.
- Record whether the estimate is item-weighted, token-weighted, user-weighted, or
domain-weighted.
- Write the decision rule before seeing the final score whenever the result will guide a
release.

The mathematical habit to build is skepticism with structure. A score is not ignored
because it is noisy; it is interpreted through the design that produced it. Adversarial
examples is one place where that habit becomes concrete.

### 4.4 Common corruptions

Common corruptions is part of the canonical scope of robustness and distribution shift.
In this chapter, the object under study is not merely a dataset or a model, but the full
shifted evaluation distribution: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{R}_{\mathrm{shift}} = \frac{1}{n}\sum_{i=1}^n \ell(f_\theta(x), y).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For common corruptions,
those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in common corruptions |
| Scoring rule | Exact formula for \ell(f_\theta(x), y) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report common corruptions with item count, prompt protocol, grader version, and a
confidence interval.
- Use paired comparisons when two models answer the same evaluation items.
- Inspect at least one meaningful slice before concluding that the aggregate result is
reliable.
- Store raw outputs so future graders can be replayed without querying the model again.
- Document whether the metric is measuring capability, reliability, user value, or risk.

Non-examples:
- A leaderboard point estimate without sample size.
- A benchmark score produced with an undocumented prompt template.
- A model-graded result without judge identity, rubric, or agreement check.
- A robustness claim measured only on the easiest in-distribution examples.
- An online win declared before the randomization and logging checks pass.

Worked evaluation pattern for common corruptions:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, common corruptions is especially delicate because the same model can be
used with many prompts, decoding policies, tools, retrieval contexts, and safety
filters. The measured quantity is therefore a property of the system configuration, not
just the base weights.

| AI connection | Evaluation consequence |
| --- | --- |
| Prompting | Treat prompt templates as part of the protocol, not as invisible setup |
| Decoding | Temperature and sampling change both mean score and variance |
| Retrieval | Retrieved context creates an extra source of failure and leakage |
| Tool use | Tool errors need separate attribution from model reasoning errors |
| Safety layer | Guardrail behavior can improve risk metrics while changing capability metrics |

Implementation checklist:
- Use deterministic seeds for synthetic or sampled evaluation subsets.
- Print metric denominators, not only percentages.
- Keep missing, invalid, timeout, and refusal outcomes explicit.
- Prefer typed result records over loose CSV columns.
- Separate raw model outputs from normalized grader inputs.
- Track the smallest reproducible command that generated the result.
- Record whether the estimate is item-weighted, token-weighted, user-weighted, or
domain-weighted.
- Write the decision rule before seeing the final score whenever the result will guide a
release.

The mathematical habit to build is skepticism with structure. A score is not ignored
because it is noisy; it is interpreted through the design that produced it. Common
corruptions is one place where that habit becomes concrete.

### 4.5 Threat-model reporting

Threat-model reporting is part of the canonical scope of robustness and distribution
shift. In this chapter, the object under study is not merely a dataset or a model, but
the full shifted evaluation distribution: the items, prompts, outputs, graders,
uncertainty statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{R}_{\mathrm{shift}} = \frac{1}{n}\sum_{i=1}^n \ell(f_\theta(x), y).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For threat-model reporting,
those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in threat-model reporting |
| Scoring rule | Exact formula for \ell(f_\theta(x), y) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report threat-model reporting with item count, prompt protocol, grader version, and a
confidence interval.
- Use paired comparisons when two models answer the same evaluation items.
- Inspect at least one meaningful slice before concluding that the aggregate result is
reliable.
- Store raw outputs so future graders can be replayed without querying the model again.
- Document whether the metric is measuring capability, reliability, user value, or risk.

Non-examples:
- A leaderboard point estimate without sample size.
- A benchmark score produced with an undocumented prompt template.
- A model-graded result without judge identity, rubric, or agreement check.
- A robustness claim measured only on the easiest in-distribution examples.
- An online win declared before the randomization and logging checks pass.

Worked evaluation pattern for threat-model reporting:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, threat-model reporting is especially delicate because the same model can
be used with many prompts, decoding policies, tools, retrieval contexts, and safety
filters. The measured quantity is therefore a property of the system configuration, not
just the base weights.

| AI connection | Evaluation consequence |
| --- | --- |
| Prompting | Treat prompt templates as part of the protocol, not as invisible setup |
| Decoding | Temperature and sampling change both mean score and variance |
| Retrieval | Retrieved context creates an extra source of failure and leakage |
| Tool use | Tool errors need separate attribution from model reasoning errors |
| Safety layer | Guardrail behavior can improve risk metrics while changing capability metrics |

Implementation checklist:
- Use deterministic seeds for synthetic or sampled evaluation subsets.
- Print metric denominators, not only percentages.
- Keep missing, invalid, timeout, and refusal outcomes explicit.
- Prefer typed result records over loose CSV columns.
- Separate raw model outputs from normalized grader inputs.
- Track the smallest reproducible command that generated the result.
- Record whether the estimate is item-weighted, token-weighted, user-weighted, or
domain-weighted.
- Write the decision rule before seeing the final score whenever the result will guide a
release.

The mathematical habit to build is skepticism with structure. A score is not ignored
because it is noisy; it is interpreted through the design that produced it. Threat-model
reporting is one place where that habit becomes concrete.

## 5. Group and Tail Reliability

Group and Tail Reliability is the part of robustness and distribution shift that turns
the approved TOC into a concrete learning path. The subsections below keep the focus on
Chapter 17's canonical job: measurement, reliability, uncertainty, and decision support
for AI systems.

### 5.1 Worst-group accuracy

Worst-group accuracy is part of the canonical scope of robustness and distribution
shift. In this chapter, the object under study is not merely a dataset or a model, but
the full shifted evaluation distribution: the items, prompts, outputs, graders,
uncertainty statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{R}_{\mathrm{shift}} = \frac{1}{n}\sum_{i=1}^n \ell(f_\theta(x), y).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For worst-group accuracy,
those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in worst-group accuracy |
| Scoring rule | Exact formula for \ell(f_\theta(x), y) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report worst-group accuracy with item count, prompt protocol, grader version, and a
confidence interval.
- Use paired comparisons when two models answer the same evaluation items.
- Inspect at least one meaningful slice before concluding that the aggregate result is
reliable.
- Store raw outputs so future graders can be replayed without querying the model again.
- Document whether the metric is measuring capability, reliability, user value, or risk.

Non-examples:
- A leaderboard point estimate without sample size.
- A benchmark score produced with an undocumented prompt template.
- A model-graded result without judge identity, rubric, or agreement check.
- A robustness claim measured only on the easiest in-distribution examples.
- An online win declared before the randomization and logging checks pass.

Worked evaluation pattern for worst-group accuracy:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, worst-group accuracy is especially delicate because the same model can
be used with many prompts, decoding policies, tools, retrieval contexts, and safety
filters. The measured quantity is therefore a property of the system configuration, not
just the base weights.

| AI connection | Evaluation consequence |
| --- | --- |
| Prompting | Treat prompt templates as part of the protocol, not as invisible setup |
| Decoding | Temperature and sampling change both mean score and variance |
| Retrieval | Retrieved context creates an extra source of failure and leakage |
| Tool use | Tool errors need separate attribution from model reasoning errors |
| Safety layer | Guardrail behavior can improve risk metrics while changing capability metrics |

Implementation checklist:
- Use deterministic seeds for synthetic or sampled evaluation subsets.
- Print metric denominators, not only percentages.
- Keep missing, invalid, timeout, and refusal outcomes explicit.
- Prefer typed result records over loose CSV columns.
- Separate raw model outputs from normalized grader inputs.
- Track the smallest reproducible command that generated the result.
- Record whether the estimate is item-weighted, token-weighted, user-weighted, or
domain-weighted.
- Write the decision rule before seeing the final score whenever the result will guide a
release.

The mathematical habit to build is skepticism with structure. A score is not ignored
because it is noisy; it is interpreted through the design that produced it. Worst-group
accuracy is one place where that habit becomes concrete.

### 5.2 Conditional value at risk

Conditional value at risk is part of the canonical scope of robustness and distribution
shift. In this chapter, the object under study is not merely a dataset or a model, but
the full shifted evaluation distribution: the items, prompts, outputs, graders,
uncertainty statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{R}_{\mathrm{shift}} = \frac{1}{n}\sum_{i=1}^n \ell(f_\theta(x), y).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For conditional value at
risk, those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in conditional value at risk |
| Scoring rule | Exact formula for \ell(f_\theta(x), y) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report conditional value at risk with item count, prompt protocol, grader version, and
a confidence interval.
- Use paired comparisons when two models answer the same evaluation items.
- Inspect at least one meaningful slice before concluding that the aggregate result is
reliable.
- Store raw outputs so future graders can be replayed without querying the model again.
- Document whether the metric is measuring capability, reliability, user value, or risk.

Non-examples:
- A leaderboard point estimate without sample size.
- A benchmark score produced with an undocumented prompt template.
- A model-graded result without judge identity, rubric, or agreement check.
- A robustness claim measured only on the easiest in-distribution examples.
- An online win declared before the randomization and logging checks pass.

Worked evaluation pattern for conditional value at risk:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, conditional value at risk is especially delicate because the same model
can be used with many prompts, decoding policies, tools, retrieval contexts, and safety
filters. The measured quantity is therefore a property of the system configuration, not
just the base weights.

| AI connection | Evaluation consequence |
| --- | --- |
| Prompting | Treat prompt templates as part of the protocol, not as invisible setup |
| Decoding | Temperature and sampling change both mean score and variance |
| Retrieval | Retrieved context creates an extra source of failure and leakage |
| Tool use | Tool errors need separate attribution from model reasoning errors |
| Safety layer | Guardrail behavior can improve risk metrics while changing capability metrics |

Implementation checklist:
- Use deterministic seeds for synthetic or sampled evaluation subsets.
- Print metric denominators, not only percentages.
- Keep missing, invalid, timeout, and refusal outcomes explicit.
- Prefer typed result records over loose CSV columns.
- Separate raw model outputs from normalized grader inputs.
- Track the smallest reproducible command that generated the result.
- Record whether the estimate is item-weighted, token-weighted, user-weighted, or
domain-weighted.
- Write the decision rule before seeing the final score whenever the result will guide a
release.

The mathematical habit to build is skepticism with structure. A score is not ignored
because it is noisy; it is interpreted through the design that produced it. Conditional
value at risk is one place where that habit becomes concrete.

### 5.3 Tail loss

Tail loss is part of the canonical scope of robustness and distribution shift. In this
chapter, the object under study is not merely a dataset or a model, but the full shifted
evaluation distribution: the items, prompts, outputs, graders, uncertainty statements,
and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{R}_{\mathrm{shift}} = \frac{1}{n}\sum_{i=1}^n \ell(f_\theta(x), y).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For tail loss, those choices
determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in tail loss |
| Scoring rule | Exact formula for \ell(f_\theta(x), y) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report tail loss with item count, prompt protocol, grader version, and a confidence
interval.
- Use paired comparisons when two models answer the same evaluation items.
- Inspect at least one meaningful slice before concluding that the aggregate result is
reliable.
- Store raw outputs so future graders can be replayed without querying the model again.
- Document whether the metric is measuring capability, reliability, user value, or risk.

Non-examples:
- A leaderboard point estimate without sample size.
- A benchmark score produced with an undocumented prompt template.
- A model-graded result without judge identity, rubric, or agreement check.
- A robustness claim measured only on the easiest in-distribution examples.
- An online win declared before the randomization and logging checks pass.

Worked evaluation pattern for tail loss:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, tail loss is especially delicate because the same model can be used with
many prompts, decoding policies, tools, retrieval contexts, and safety filters. The
measured quantity is therefore a property of the system configuration, not just the base
weights.

| AI connection | Evaluation consequence |
| --- | --- |
| Prompting | Treat prompt templates as part of the protocol, not as invisible setup |
| Decoding | Temperature and sampling change both mean score and variance |
| Retrieval | Retrieved context creates an extra source of failure and leakage |
| Tool use | Tool errors need separate attribution from model reasoning errors |
| Safety layer | Guardrail behavior can improve risk metrics while changing capability metrics |

Implementation checklist:
- Use deterministic seeds for synthetic or sampled evaluation subsets.
- Print metric denominators, not only percentages.
- Keep missing, invalid, timeout, and refusal outcomes explicit.
- Prefer typed result records over loose CSV columns.
- Separate raw model outputs from normalized grader inputs.
- Track the smallest reproducible command that generated the result.
- Record whether the estimate is item-weighted, token-weighted, user-weighted, or
domain-weighted.
- Write the decision rule before seeing the final score whenever the result will guide a
release.

The mathematical habit to build is skepticism with structure. A score is not ignored
because it is noisy; it is interpreted through the design that produced it. Tail loss is
one place where that habit becomes concrete.

### 5.4 Distributionally robust evaluation

Distributionally robust evaluation is part of the canonical scope of robustness and
distribution shift. In this chapter, the object under study is not merely a dataset or a
model, but the full shifted evaluation distribution: the items, prompts, outputs,
graders, uncertainty statements, and decision rules that turn model behavior into
evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{R}_{\mathrm{shift}} = \frac{1}{n}\sum_{i=1}^n \ell(f_\theta(x), y).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For distributionally robust
evaluation, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in distributionally robust evaluation |
| Scoring rule | Exact formula for \ell(f_\theta(x), y) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report distributionally robust evaluation with item count, prompt protocol, grader
version, and a confidence interval.
- Use paired comparisons when two models answer the same evaluation items.
- Inspect at least one meaningful slice before concluding that the aggregate result is
reliable.
- Store raw outputs so future graders can be replayed without querying the model again.
- Document whether the metric is measuring capability, reliability, user value, or risk.

Non-examples:
- A leaderboard point estimate without sample size.
- A benchmark score produced with an undocumented prompt template.
- A model-graded result without judge identity, rubric, or agreement check.
- A robustness claim measured only on the easiest in-distribution examples.
- An online win declared before the randomization and logging checks pass.

Worked evaluation pattern for distributionally robust evaluation:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, distributionally robust evaluation is especially delicate because the
same model can be used with many prompts, decoding policies, tools, retrieval contexts,
and safety filters. The measured quantity is therefore a property of the system
configuration, not just the base weights.

| AI connection | Evaluation consequence |
| --- | --- |
| Prompting | Treat prompt templates as part of the protocol, not as invisible setup |
| Decoding | Temperature and sampling change both mean score and variance |
| Retrieval | Retrieved context creates an extra source of failure and leakage |
| Tool use | Tool errors need separate attribution from model reasoning errors |
| Safety layer | Guardrail behavior can improve risk metrics while changing capability metrics |

Implementation checklist:
- Use deterministic seeds for synthetic or sampled evaluation subsets.
- Print metric denominators, not only percentages.
- Keep missing, invalid, timeout, and refusal outcomes explicit.
- Prefer typed result records over loose CSV columns.
- Separate raw model outputs from normalized grader inputs.
- Track the smallest reproducible command that generated the result.
- Record whether the estimate is item-weighted, token-weighted, user-weighted, or
domain-weighted.
- Write the decision rule before seeing the final score whenever the result will guide a
release.

The mathematical habit to build is skepticism with structure. A score is not ignored
because it is noisy; it is interpreted through the design that produced it.
Distributionally robust evaluation is one place where that habit becomes concrete.

### 5.5 Fairness and privacy side effects

Fairness and privacy side effects is part of the canonical scope of robustness and
distribution shift. In this chapter, the object under study is not merely a dataset or a
model, but the full shifted evaluation distribution: the items, prompts, outputs,
graders, uncertainty statements, and decision rules that turn model behavior into
evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{R}_{\mathrm{shift}} = \frac{1}{n}\sum_{i=1}^n \ell(f_\theta(x), y).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For fairness and privacy
side effects, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in fairness and privacy side effects |
| Scoring rule | Exact formula for \ell(f_\theta(x), y) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report fairness and privacy side effects with item count, prompt protocol, grader
version, and a confidence interval.
- Use paired comparisons when two models answer the same evaluation items.
- Inspect at least one meaningful slice before concluding that the aggregate result is
reliable.
- Store raw outputs so future graders can be replayed without querying the model again.
- Document whether the metric is measuring capability, reliability, user value, or risk.

Non-examples:
- A leaderboard point estimate without sample size.
- A benchmark score produced with an undocumented prompt template.
- A model-graded result without judge identity, rubric, or agreement check.
- A robustness claim measured only on the easiest in-distribution examples.
- An online win declared before the randomization and logging checks pass.

Worked evaluation pattern for fairness and privacy side effects:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, fairness and privacy side effects is especially delicate because the
same model can be used with many prompts, decoding policies, tools, retrieval contexts,
and safety filters. The measured quantity is therefore a property of the system
configuration, not just the base weights.

| AI connection | Evaluation consequence |
| --- | --- |
| Prompting | Treat prompt templates as part of the protocol, not as invisible setup |
| Decoding | Temperature and sampling change both mean score and variance |
| Retrieval | Retrieved context creates an extra source of failure and leakage |
| Tool use | Tool errors need separate attribution from model reasoning errors |
| Safety layer | Guardrail behavior can improve risk metrics while changing capability metrics |

Implementation checklist:
- Use deterministic seeds for synthetic or sampled evaluation subsets.
- Print metric denominators, not only percentages.
- Keep missing, invalid, timeout, and refusal outcomes explicit.
- Prefer typed result records over loose CSV columns.
- Separate raw model outputs from normalized grader inputs.
- Track the smallest reproducible command that generated the result.
- Record whether the estimate is item-weighted, token-weighted, user-weighted, or
domain-weighted.
- Write the decision rule before seeing the final score whenever the result will guide a
release.

The mathematical habit to build is skepticism with structure. A score is not ignored
because it is noisy; it is interpreted through the design that produced it. Fairness and
privacy side effects is one place where that habit becomes concrete.

## 6. LLM Robustness

LLM Robustness is the part of robustness and distribution shift that turns the approved
TOC into a concrete learning path. The subsections below keep the focus on Chapter 17's
canonical job: measurement, reliability, uncertainty, and decision support for AI
systems.

### 6.1 Prompt perturbations

Prompt perturbations is part of the canonical scope of robustness and distribution
shift. In this chapter, the object under study is not merely a dataset or a model, but
the full shifted evaluation distribution: the items, prompts, outputs, graders,
uncertainty statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{R}_{\mathrm{shift}} = \frac{1}{n}\sum_{i=1}^n \ell(f_\theta(x), y).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For prompt perturbations,
those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in prompt perturbations |
| Scoring rule | Exact formula for \ell(f_\theta(x), y) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report prompt perturbations with item count, prompt protocol, grader version, and a
confidence interval.
- Use paired comparisons when two models answer the same evaluation items.
- Inspect at least one meaningful slice before concluding that the aggregate result is
reliable.
- Store raw outputs so future graders can be replayed without querying the model again.
- Document whether the metric is measuring capability, reliability, user value, or risk.

Non-examples:
- A leaderboard point estimate without sample size.
- A benchmark score produced with an undocumented prompt template.
- A model-graded result without judge identity, rubric, or agreement check.
- A robustness claim measured only on the easiest in-distribution examples.
- An online win declared before the randomization and logging checks pass.

Worked evaluation pattern for prompt perturbations:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, prompt perturbations is especially delicate because the same model can
be used with many prompts, decoding policies, tools, retrieval contexts, and safety
filters. The measured quantity is therefore a property of the system configuration, not
just the base weights.

| AI connection | Evaluation consequence |
| --- | --- |
| Prompting | Treat prompt templates as part of the protocol, not as invisible setup |
| Decoding | Temperature and sampling change both mean score and variance |
| Retrieval | Retrieved context creates an extra source of failure and leakage |
| Tool use | Tool errors need separate attribution from model reasoning errors |
| Safety layer | Guardrail behavior can improve risk metrics while changing capability metrics |

Implementation checklist:
- Use deterministic seeds for synthetic or sampled evaluation subsets.
- Print metric denominators, not only percentages.
- Keep missing, invalid, timeout, and refusal outcomes explicit.
- Prefer typed result records over loose CSV columns.
- Separate raw model outputs from normalized grader inputs.
- Track the smallest reproducible command that generated the result.
- Record whether the estimate is item-weighted, token-weighted, user-weighted, or
domain-weighted.
- Write the decision rule before seeing the final score whenever the result will guide a
release.

The mathematical habit to build is skepticism with structure. A score is not ignored
because it is noisy; it is interpreted through the design that produced it. Prompt
perturbations is one place where that habit becomes concrete.

### 6.2 Format sensitivity

Format sensitivity is part of the canonical scope of robustness and distribution shift.
In this chapter, the object under study is not merely a dataset or a model, but the full
shifted evaluation distribution: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{R}_{\mathrm{shift}} = \frac{1}{n}\sum_{i=1}^n \ell(f_\theta(x), y).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For format sensitivity,
those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in format sensitivity |
| Scoring rule | Exact formula for \ell(f_\theta(x), y) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report format sensitivity with item count, prompt protocol, grader version, and a
confidence interval.
- Use paired comparisons when two models answer the same evaluation items.
- Inspect at least one meaningful slice before concluding that the aggregate result is
reliable.
- Store raw outputs so future graders can be replayed without querying the model again.
- Document whether the metric is measuring capability, reliability, user value, or risk.

Non-examples:
- A leaderboard point estimate without sample size.
- A benchmark score produced with an undocumented prompt template.
- A model-graded result without judge identity, rubric, or agreement check.
- A robustness claim measured only on the easiest in-distribution examples.
- An online win declared before the randomization and logging checks pass.

Worked evaluation pattern for format sensitivity:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, format sensitivity is especially delicate because the same model can be
used with many prompts, decoding policies, tools, retrieval contexts, and safety
filters. The measured quantity is therefore a property of the system configuration, not
just the base weights.

| AI connection | Evaluation consequence |
| --- | --- |
| Prompting | Treat prompt templates as part of the protocol, not as invisible setup |
| Decoding | Temperature and sampling change both mean score and variance |
| Retrieval | Retrieved context creates an extra source of failure and leakage |
| Tool use | Tool errors need separate attribution from model reasoning errors |
| Safety layer | Guardrail behavior can improve risk metrics while changing capability metrics |

Implementation checklist:
- Use deterministic seeds for synthetic or sampled evaluation subsets.
- Print metric denominators, not only percentages.
- Keep missing, invalid, timeout, and refusal outcomes explicit.
- Prefer typed result records over loose CSV columns.
- Separate raw model outputs from normalized grader inputs.
- Track the smallest reproducible command that generated the result.
- Record whether the estimate is item-weighted, token-weighted, user-weighted, or
domain-weighted.
- Write the decision rule before seeing the final score whenever the result will guide a
release.

The mathematical habit to build is skepticism with structure. A score is not ignored
because it is noisy; it is interpreted through the design that produced it. Format
sensitivity is one place where that habit becomes concrete.

### 6.3 Multilingual and cultural shift

Multilingual and cultural shift is part of the canonical scope of robustness and
distribution shift. In this chapter, the object under study is not merely a dataset or a
model, but the full shifted evaluation distribution: the items, prompts, outputs,
graders, uncertainty statements, and decision rules that turn model behavior into
evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{R}_{\mathrm{shift}} = \frac{1}{n}\sum_{i=1}^n \ell(f_\theta(x), y).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For multilingual and
cultural shift, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in multilingual and cultural shift |
| Scoring rule | Exact formula for \ell(f_\theta(x), y) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report multilingual and cultural shift with item count, prompt protocol, grader
version, and a confidence interval.
- Use paired comparisons when two models answer the same evaluation items.
- Inspect at least one meaningful slice before concluding that the aggregate result is
reliable.
- Store raw outputs so future graders can be replayed without querying the model again.
- Document whether the metric is measuring capability, reliability, user value, or risk.

Non-examples:
- A leaderboard point estimate without sample size.
- A benchmark score produced with an undocumented prompt template.
- A model-graded result without judge identity, rubric, or agreement check.
- A robustness claim measured only on the easiest in-distribution examples.
- An online win declared before the randomization and logging checks pass.

Worked evaluation pattern for multilingual and cultural shift:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, multilingual and cultural shift is especially delicate because the same
model can be used with many prompts, decoding policies, tools, retrieval contexts, and
safety filters. The measured quantity is therefore a property of the system
configuration, not just the base weights.

| AI connection | Evaluation consequence |
| --- | --- |
| Prompting | Treat prompt templates as part of the protocol, not as invisible setup |
| Decoding | Temperature and sampling change both mean score and variance |
| Retrieval | Retrieved context creates an extra source of failure and leakage |
| Tool use | Tool errors need separate attribution from model reasoning errors |
| Safety layer | Guardrail behavior can improve risk metrics while changing capability metrics |

Implementation checklist:
- Use deterministic seeds for synthetic or sampled evaluation subsets.
- Print metric denominators, not only percentages.
- Keep missing, invalid, timeout, and refusal outcomes explicit.
- Prefer typed result records over loose CSV columns.
- Separate raw model outputs from normalized grader inputs.
- Track the smallest reproducible command that generated the result.
- Record whether the estimate is item-weighted, token-weighted, user-weighted, or
domain-weighted.
- Write the decision rule before seeing the final score whenever the result will guide a
release.

The mathematical habit to build is skepticism with structure. A score is not ignored
because it is noisy; it is interpreted through the design that produced it. Multilingual
and cultural shift is one place where that habit becomes concrete.

### 6.4 Retrieval-context failures

Retrieval-context failures is part of the canonical scope of robustness and distribution
shift. In this chapter, the object under study is not merely a dataset or a model, but
the full shifted evaluation distribution: the items, prompts, outputs, graders,
uncertainty statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{R}_{\mathrm{shift}} = \frac{1}{n}\sum_{i=1}^n \ell(f_\theta(x), y).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For retrieval-context
failures, those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in retrieval-context failures |
| Scoring rule | Exact formula for \ell(f_\theta(x), y) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report retrieval-context failures with item count, prompt protocol, grader version,
and a confidence interval.
- Use paired comparisons when two models answer the same evaluation items.
- Inspect at least one meaningful slice before concluding that the aggregate result is
reliable.
- Store raw outputs so future graders can be replayed without querying the model again.
- Document whether the metric is measuring capability, reliability, user value, or risk.

Non-examples:
- A leaderboard point estimate without sample size.
- A benchmark score produced with an undocumented prompt template.
- A model-graded result without judge identity, rubric, or agreement check.
- A robustness claim measured only on the easiest in-distribution examples.
- An online win declared before the randomization and logging checks pass.

Worked evaluation pattern for retrieval-context failures:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, retrieval-context failures is especially delicate because the same model
can be used with many prompts, decoding policies, tools, retrieval contexts, and safety
filters. The measured quantity is therefore a property of the system configuration, not
just the base weights.

| AI connection | Evaluation consequence |
| --- | --- |
| Prompting | Treat prompt templates as part of the protocol, not as invisible setup |
| Decoding | Temperature and sampling change both mean score and variance |
| Retrieval | Retrieved context creates an extra source of failure and leakage |
| Tool use | Tool errors need separate attribution from model reasoning errors |
| Safety layer | Guardrail behavior can improve risk metrics while changing capability metrics |

Implementation checklist:
- Use deterministic seeds for synthetic or sampled evaluation subsets.
- Print metric denominators, not only percentages.
- Keep missing, invalid, timeout, and refusal outcomes explicit.
- Prefer typed result records over loose CSV columns.
- Separate raw model outputs from normalized grader inputs.
- Track the smallest reproducible command that generated the result.
- Record whether the estimate is item-weighted, token-weighted, user-weighted, or
domain-weighted.
- Write the decision rule before seeing the final score whenever the result will guide a
release.

The mathematical habit to build is skepticism with structure. A score is not ignored
because it is noisy; it is interpreted through the design that produced it. Retrieval-
context failures is one place where that habit becomes concrete.

### 6.5 Long-context degradation

Long-context degradation is part of the canonical scope of robustness and distribution
shift. In this chapter, the object under study is not merely a dataset or a model, but
the full shifted evaluation distribution: the items, prompts, outputs, graders,
uncertainty statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{R}_{\mathrm{shift}} = \frac{1}{n}\sum_{i=1}^n \ell(f_\theta(x), y).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For long-context
degradation, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in long-context degradation |
| Scoring rule | Exact formula for \ell(f_\theta(x), y) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report long-context degradation with item count, prompt protocol, grader version, and
a confidence interval.
- Use paired comparisons when two models answer the same evaluation items.
- Inspect at least one meaningful slice before concluding that the aggregate result is
reliable.
- Store raw outputs so future graders can be replayed without querying the model again.
- Document whether the metric is measuring capability, reliability, user value, or risk.

Non-examples:
- A leaderboard point estimate without sample size.
- A benchmark score produced with an undocumented prompt template.
- A model-graded result without judge identity, rubric, or agreement check.
- A robustness claim measured only on the easiest in-distribution examples.
- An online win declared before the randomization and logging checks pass.

Worked evaluation pattern for long-context degradation:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, long-context degradation is especially delicate because the same model
can be used with many prompts, decoding policies, tools, retrieval contexts, and safety
filters. The measured quantity is therefore a property of the system configuration, not
just the base weights.

| AI connection | Evaluation consequence |
| --- | --- |
| Prompting | Treat prompt templates as part of the protocol, not as invisible setup |
| Decoding | Temperature and sampling change both mean score and variance |
| Retrieval | Retrieved context creates an extra source of failure and leakage |
| Tool use | Tool errors need separate attribution from model reasoning errors |
| Safety layer | Guardrail behavior can improve risk metrics while changing capability metrics |

Implementation checklist:
- Use deterministic seeds for synthetic or sampled evaluation subsets.
- Print metric denominators, not only percentages.
- Keep missing, invalid, timeout, and refusal outcomes explicit.
- Prefer typed result records over loose CSV columns.
- Separate raw model outputs from normalized grader inputs.
- Track the smallest reproducible command that generated the result.
- Record whether the estimate is item-weighted, token-weighted, user-weighted, or
domain-weighted.
- Write the decision rule before seeing the final score whenever the result will guide a
release.

The mathematical habit to build is skepticism with structure. A score is not ignored
because it is noisy; it is interpreted through the design that produced it. Long-context
degradation is one place where that habit becomes concrete.

## 7. Boundary with Safety

Boundary with Safety is the part of robustness and distribution shift that turns the
approved TOC into a concrete learning path. The subsections below keep the focus on
Chapter 17's canonical job: measurement, reliability, uncertainty, and decision support
for AI systems.

### 7.1 Jailbreak preview

Jailbreak preview is part of the canonical scope of robustness and distribution shift.
In this chapter, the object under study is not merely a dataset or a model, but the full
shifted evaluation distribution: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{R}_{\mathrm{shift}} = \frac{1}{n}\sum_{i=1}^n \ell(f_\theta(x), y).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For jailbreak preview, those
choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in jailbreak preview |
| Scoring rule | Exact formula for \ell(f_\theta(x), y) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report jailbreak preview with item count, prompt protocol, grader version, and a
confidence interval.
- Use paired comparisons when two models answer the same evaluation items.
- Inspect at least one meaningful slice before concluding that the aggregate result is
reliable.
- Store raw outputs so future graders can be replayed without querying the model again.
- Document whether the metric is measuring capability, reliability, user value, or risk.

Non-examples:
- A leaderboard point estimate without sample size.
- A benchmark score produced with an undocumented prompt template.
- A model-graded result without judge identity, rubric, or agreement check.
- A robustness claim measured only on the easiest in-distribution examples.
- An online win declared before the randomization and logging checks pass.

Worked evaluation pattern for jailbreak preview:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, jailbreak preview is especially delicate because the same model can be
used with many prompts, decoding policies, tools, retrieval contexts, and safety
filters. The measured quantity is therefore a property of the system configuration, not
just the base weights.

| AI connection | Evaluation consequence |
| --- | --- |
| Prompting | Treat prompt templates as part of the protocol, not as invisible setup |
| Decoding | Temperature and sampling change both mean score and variance |
| Retrieval | Retrieved context creates an extra source of failure and leakage |
| Tool use | Tool errors need separate attribution from model reasoning errors |
| Safety layer | Guardrail behavior can improve risk metrics while changing capability metrics |

Implementation checklist:
- Use deterministic seeds for synthetic or sampled evaluation subsets.
- Print metric denominators, not only percentages.
- Keep missing, invalid, timeout, and refusal outcomes explicit.
- Prefer typed result records over loose CSV columns.
- Separate raw model outputs from normalized grader inputs.
- Track the smallest reproducible command that generated the result.
- Record whether the estimate is item-weighted, token-weighted, user-weighted, or
domain-weighted.
- Write the decision rule before seeing the final score whenever the result will guide a
release.

The mathematical habit to build is skepticism with structure. A score is not ignored
because it is noisy; it is interpreted through the design that produced it. Jailbreak
preview is one place where that habit becomes concrete.

### 7.2 Red-team preview

Red-team preview is part of the canonical scope of robustness and distribution shift. In
this chapter, the object under study is not merely a dataset or a model, but the full
shifted evaluation distribution: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{R}_{\mathrm{shift}} = \frac{1}{n}\sum_{i=1}^n \ell(f_\theta(x), y).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For red-team preview, those
choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in red-team preview |
| Scoring rule | Exact formula for \ell(f_\theta(x), y) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report red-team preview with item count, prompt protocol, grader version, and a
confidence interval.
- Use paired comparisons when two models answer the same evaluation items.
- Inspect at least one meaningful slice before concluding that the aggregate result is
reliable.
- Store raw outputs so future graders can be replayed without querying the model again.
- Document whether the metric is measuring capability, reliability, user value, or risk.

Non-examples:
- A leaderboard point estimate without sample size.
- A benchmark score produced with an undocumented prompt template.
- A model-graded result without judge identity, rubric, or agreement check.
- A robustness claim measured only on the easiest in-distribution examples.
- An online win declared before the randomization and logging checks pass.

Worked evaluation pattern for red-team preview:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, red-team preview is especially delicate because the same model can be
used with many prompts, decoding policies, tools, retrieval contexts, and safety
filters. The measured quantity is therefore a property of the system configuration, not
just the base weights.

| AI connection | Evaluation consequence |
| --- | --- |
| Prompting | Treat prompt templates as part of the protocol, not as invisible setup |
| Decoding | Temperature and sampling change both mean score and variance |
| Retrieval | Retrieved context creates an extra source of failure and leakage |
| Tool use | Tool errors need separate attribution from model reasoning errors |
| Safety layer | Guardrail behavior can improve risk metrics while changing capability metrics |

Implementation checklist:
- Use deterministic seeds for synthetic or sampled evaluation subsets.
- Print metric denominators, not only percentages.
- Keep missing, invalid, timeout, and refusal outcomes explicit.
- Prefer typed result records over loose CSV columns.
- Separate raw model outputs from normalized grader inputs.
- Track the smallest reproducible command that generated the result.
- Record whether the estimate is item-weighted, token-weighted, user-weighted, or
domain-weighted.
- Write the decision rule before seeing the final score whenever the result will guide a
release.

The mathematical habit to build is skepticism with structure. A score is not ignored
because it is noisy; it is interpreted through the design that produced it. Red-team
preview is one place where that habit becomes concrete.

### 7.3 Why alignment training is Chapter 18

Why alignment training is Chapter 18 is part of the canonical scope of robustness and
distribution shift. In this chapter, the object under study is not merely a dataset or a
model, but the full shifted evaluation distribution: the items, prompts, outputs,
graders, uncertainty statements, and decision rules that turn model behavior into
evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{R}_{\mathrm{shift}} = \frac{1}{n}\sum_{i=1}^n \ell(f_\theta(x), y).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For why alignment training
is chapter 18, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in why alignment training is chapter 18 |
| Scoring rule | Exact formula for \ell(f_\theta(x), y) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report why alignment training is chapter 18 with item count, prompt protocol, grader
version, and a confidence interval.
- Use paired comparisons when two models answer the same evaluation items.
- Inspect at least one meaningful slice before concluding that the aggregate result is
reliable.
- Store raw outputs so future graders can be replayed without querying the model again.
- Document whether the metric is measuring capability, reliability, user value, or risk.

Non-examples:
- A leaderboard point estimate without sample size.
- A benchmark score produced with an undocumented prompt template.
- A model-graded result without judge identity, rubric, or agreement check.
- A robustness claim measured only on the easiest in-distribution examples.
- An online win declared before the randomization and logging checks pass.

Worked evaluation pattern for why alignment training is chapter 18:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, why alignment training is chapter 18 is especially delicate because the
same model can be used with many prompts, decoding policies, tools, retrieval contexts,
and safety filters. The measured quantity is therefore a property of the system
configuration, not just the base weights.

| AI connection | Evaluation consequence |
| --- | --- |
| Prompting | Treat prompt templates as part of the protocol, not as invisible setup |
| Decoding | Temperature and sampling change both mean score and variance |
| Retrieval | Retrieved context creates an extra source of failure and leakage |
| Tool use | Tool errors need separate attribution from model reasoning errors |
| Safety layer | Guardrail behavior can improve risk metrics while changing capability metrics |

Implementation checklist:
- Use deterministic seeds for synthetic or sampled evaluation subsets.
- Print metric denominators, not only percentages.
- Keep missing, invalid, timeout, and refusal outcomes explicit.
- Prefer typed result records over loose CSV columns.
- Separate raw model outputs from normalized grader inputs.
- Track the smallest reproducible command that generated the result.
- Record whether the estimate is item-weighted, token-weighted, user-weighted, or
domain-weighted.
- Write the decision rule before seeing the final score whenever the result will guide a
release.

The mathematical habit to build is skepticism with structure. A score is not ignored
because it is noisy; it is interpreted through the design that produced it. Why
alignment training is Chapter 18 is one place where that habit becomes concrete.

### 7.4 Why production drift is Chapter 19

Why production drift is Chapter 19 is part of the canonical scope of robustness and
distribution shift. In this chapter, the object under study is not merely a dataset or a
model, but the full shifted evaluation distribution: the items, prompts, outputs,
graders, uncertainty statements, and decision rules that turn model behavior into
evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{R}_{\mathrm{shift}} = \frac{1}{n}\sum_{i=1}^n \ell(f_\theta(x), y).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For why production drift is
chapter 19, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in why production drift is chapter 19 |
| Scoring rule | Exact formula for \ell(f_\theta(x), y) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report why production drift is chapter 19 with item count, prompt protocol, grader
version, and a confidence interval.
- Use paired comparisons when two models answer the same evaluation items.
- Inspect at least one meaningful slice before concluding that the aggregate result is
reliable.
- Store raw outputs so future graders can be replayed without querying the model again.
- Document whether the metric is measuring capability, reliability, user value, or risk.

Non-examples:
- A leaderboard point estimate without sample size.
- A benchmark score produced with an undocumented prompt template.
- A model-graded result without judge identity, rubric, or agreement check.
- A robustness claim measured only on the easiest in-distribution examples.
- An online win declared before the randomization and logging checks pass.

Worked evaluation pattern for why production drift is chapter 19:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, why production drift is chapter 19 is especially delicate because the
same model can be used with many prompts, decoding policies, tools, retrieval contexts,
and safety filters. The measured quantity is therefore a property of the system
configuration, not just the base weights.

| AI connection | Evaluation consequence |
| --- | --- |
| Prompting | Treat prompt templates as part of the protocol, not as invisible setup |
| Decoding | Temperature and sampling change both mean score and variance |
| Retrieval | Retrieved context creates an extra source of failure and leakage |
| Tool use | Tool errors need separate attribution from model reasoning errors |
| Safety layer | Guardrail behavior can improve risk metrics while changing capability metrics |

Implementation checklist:
- Use deterministic seeds for synthetic or sampled evaluation subsets.
- Print metric denominators, not only percentages.
- Keep missing, invalid, timeout, and refusal outcomes explicit.
- Prefer typed result records over loose CSV columns.
- Separate raw model outputs from normalized grader inputs.
- Track the smallest reproducible command that generated the result.
- Record whether the estimate is item-weighted, token-weighted, user-weighted, or
domain-weighted.
- Write the decision rule before seeing the final score whenever the result will guide a
release.

The mathematical habit to build is skepticism with structure. A score is not ignored
because it is noisy; it is interpreted through the design that produced it. Why
production drift is Chapter 19 is one place where that habit becomes concrete.

### 7.5 Reliability handoff checklist

Reliability handoff checklist is part of the canonical scope of robustness and
distribution shift. In this chapter, the object under study is not merely a dataset or a
model, but the full shifted evaluation distribution: the items, prompts, outputs,
graders, uncertainty statements, and decision rules that turn model behavior into
evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{R}_{\mathrm{shift}} = \frac{1}{n}\sum_{i=1}^n \ell(f_\theta(x), y).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For reliability handoff
checklist, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in reliability handoff checklist |
| Scoring rule | Exact formula for \ell(f_\theta(x), y) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report reliability handoff checklist with item count, prompt protocol, grader version,
and a confidence interval.
- Use paired comparisons when two models answer the same evaluation items.
- Inspect at least one meaningful slice before concluding that the aggregate result is
reliable.
- Store raw outputs so future graders can be replayed without querying the model again.
- Document whether the metric is measuring capability, reliability, user value, or risk.

Non-examples:
- A leaderboard point estimate without sample size.
- A benchmark score produced with an undocumented prompt template.
- A model-graded result without judge identity, rubric, or agreement check.
- A robustness claim measured only on the easiest in-distribution examples.
- An online win declared before the randomization and logging checks pass.

Worked evaluation pattern for reliability handoff checklist:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, reliability handoff checklist is especially delicate because the same
model can be used with many prompts, decoding policies, tools, retrieval contexts, and
safety filters. The measured quantity is therefore a property of the system
configuration, not just the base weights.

| AI connection | Evaluation consequence |
| --- | --- |
| Prompting | Treat prompt templates as part of the protocol, not as invisible setup |
| Decoding | Temperature and sampling change both mean score and variance |
| Retrieval | Retrieved context creates an extra source of failure and leakage |
| Tool use | Tool errors need separate attribution from model reasoning errors |
| Safety layer | Guardrail behavior can improve risk metrics while changing capability metrics |

Implementation checklist:
- Use deterministic seeds for synthetic or sampled evaluation subsets.
- Print metric denominators, not only percentages.
- Keep missing, invalid, timeout, and refusal outcomes explicit.
- Prefer typed result records over loose CSV columns.
- Separate raw model outputs from normalized grader inputs.
- Track the smallest reproducible command that generated the result.
- Record whether the estimate is item-weighted, token-weighted, user-weighted, or
domain-weighted.
- Write the decision rule before seeing the final score whenever the result will guide a
release.

The mathematical habit to build is skepticism with structure. A score is not ignored
because it is noisy; it is interpreted through the design that produced it. Reliability
handoff checklist is one place where that habit becomes concrete.

## 8. Common Mistakes

| # | Mistake | Why It Is Wrong | Fix |
| --- | --- | --- | --- |
| 1 | Treating a point estimate as exact | Every finite evaluation has sampling error in robustness and distribution shift. | Report uncertainty with the point estimate. |
| 2 | Changing prompts between models | The protocol changed with the treatment in robustness and distribution shift. | Lock prompt, decoding, and grader before comparison. |
| 3 | Ignoring invalid outputs | Missingness can be correlated with model quality in robustness and distribution shift. | Track invalid, timeout, refusal, and parse-failure rates. |
| 4 | Overfitting to a public leaderboard | Repeated testing leaks information from the benchmark in robustness and distribution shift. | Use private holdouts and regression suites. |
| 5 | Averaging incomparable metrics | Different scales do not have shared units in robustness and distribution shift. | Normalize by a stated decision rule or report separately. |
| 6 | Forgetting paired structure | Two models often answer the same items in robustness and distribution shift. | Use paired bootstrap or paired tests where possible. |
| 7 | Reporting only aggregate performance | Subgroup failures can be hidden in robustness and distribution shift. | Add slice and tail-risk views. |
| 8 | Trusting model judges blindly | LLM judges have position, verbosity, and self-preference biases in robustness and distribution shift. | Calibrate judges against human labels. |
| 9 | Peeking during online experiments | Optional stopping inflates false positives in robustness and distribution shift. | Use fixed horizons or sequential-valid methods. |
| 10 | Conflating evaluation with monitoring | Chapter 17 measures controlled evidence; production monitoring is ongoing operations in robustness and distribution shift. | Hand off drift dashboards to Chapter 19 concepts. |

## 9. Exercises

1. (*) Deployment changes the data distribution.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

2. (*) Prompt surface as an input distribution.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

3. (*) Rare tails dominate reliability risk.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

4. (**) Robustness is not only adversarial accuracy.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

5. (**) Reliability budgets across shifts.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

6. (**) Training and test distributions.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

7. (***) Covariate shift, label shift, and concept shift.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

8. (***) Subgroup risk.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

9. (***) Robust risk and worst-case risk.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

10. (***) Threat model and perturbation set.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

## 10. Why This Matters for AI

| Concept | AI Impact |
| --- | --- |
| Protocol as measurement | Prevents hidden prompt or grader changes from masquerading as model progress |
| Uncertainty intervals | Keeps model rankings honest when differences are smaller than sampling noise |
| Slice metrics | Reveals failures on languages, domains, formats, or user groups hidden by averages |
| Calibration | Lets systems decide when to answer, abstain, ask for help, or escalate |
| Robustness | Tests whether behavior survives realistic perturbations and distribution shift |
| Ablations | Separates real improvements from accidental metric movement |
| Online tests | Measures causal user impact rather than offline proxy success |
| Audit trails | Turns evaluation from a screenshot into reproducible scientific evidence |

## 11. Conceptual Bridge

This section sits after the training-data pipeline because evaluation depends on clean
holdouts, contamination audits, and well-documented data provenance. It does not repeat
those pipeline mechanics; it consumes their outputs as the basis for credible
measurement.

It also sits before alignment and production chapters. Alignment asks how to shape model
behavior with supervised data, preferences, policies, and feedback. Production MLOps
asks how deployed systems are observed and maintained over time. Robustness and
Distribution Shift supplies the measurement discipline both chapters need.

The recurring mathematical pattern is empirical risk with uncertainty. Whether the
object is a benchmark item, a calibrated probability, a shifted subgroup, an ablation
comparison, or an online treatment effect, the learner should ask: what distribution
generated this evidence, what estimator did we compute, and what decision is justified
by the uncertainty?

```text
16 Data Pipeline
    -> clean eval data, manifests, decontamination
17 Evaluation and Reliability
    -> benchmarks, calibration, robustness, ablations, online tests
18 Alignment and Safety
    -> SFT, preferences, policies, human feedback
19 Production ML and MLOps
    -> monitoring, serving, retraining, observability
```

## References

- [WILDS: A Benchmark of in-the-Wild Distribution Shifts](https://arxiv.org/abs/2012.07421)
- [A Baseline for Detecting Misclassified and Out-of-Distribution Examples](https://arxiv.org/abs/1610.02136)
- [RobustBench](https://arxiv.org/abs/2010.09670)
- [HELM](https://arxiv.org/abs/2211.09110)
- [Matrix Norms](../../03-Advanced-Linear-Algebra/06-Matrix-Norms/notes.md)
- [Concentration Inequalities](../../06-Probability-Theory/05-Concentration-Inequalities/notes.md)
- [Estimation Theory](../../07-Statistics/02-Estimation-Theory/notes.md)
- [Hypothesis Testing](../../07-Statistics/03-Hypothesis-Testing/notes.md)

