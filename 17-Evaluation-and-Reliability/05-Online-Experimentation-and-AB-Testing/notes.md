[Back to Curriculum](../../README.md) | [Previous: Error Analysis and Ablations](../04-Error-Analysis-and-Ablations/notes.md) | [Next: Instruction Tuning and SFT](../../18-Alignment-and-Safety/01-Instruction-Tuning-and-SFT/notes.md)

---

# Online Experimentation and AB Testing

> _"An online experiment is a randomized proof attempt about the real world."_

## Overview

Online experiments connect offline model evidence to causal user and system impact
through randomized comparison, statistical inference, and trust checks.

The chapter treats evaluation as a mathematical object: a controlled protocol that maps
model behavior into evidence with uncertainty. A result is not just a number; it is an
estimate produced by a task distribution, a scoring rule, and an aggregation rule.

This section is written in LaTeX Markdown. Inline mathematics uses `$...$`, and display
equations use `$$...$$`. The emphasis is practical but rigorous: every metric should be
linked to the statistical assumptions that make it meaningful.

## Prerequisites

- [Common Distributions](../../06-Probability-Theory/02-Common-Distributions/notes.md)
- [Estimation Theory](../../07-Statistics/02-Estimation-Theory/notes.md)
- [Hypothesis Testing](../../07-Statistics/03-Hypothesis-Testing/notes.md)
- [Capability Benchmarks](../01-Capability-Benchmarks/notes.md)
- [Error Analysis and Ablations](../04-Error-Analysis-and-Ablations/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Executable demonstrations for online experimentation and ab testing |
| [exercises.ipynb](exercises.ipynb) | Graded practice for online experimentation and ab testing |

## Learning Objectives

After completing this section, you will be able to:

- Define the core objects used in online experimentation and ab testing
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
  - [1.1 Offline eval predicts, online tests measure](#11-offline-eval-predicts-online-tests-measure)
  - [1.2 Randomization as causal design](#12-randomization-as-causal-design)
  - [1.3 Overall evaluation criterion](#13-overall-evaluation-criterion)
  - [1.4 Guardrails and downside risk](#14-guardrails-and-downside-risk)
  - [1.5 Experiment culture for LLM systems](#15-experiment-culture-for-llm-systems)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Treatment and control](#21-treatment-and-control)
  - [2.2 Randomization unit](#22-randomization-unit)
  - [2.3 Overall evaluation criterion and guardrails](#23-overall-evaluation-criterion-and-guardrails)
  - [2.4 Average treatment effect](#24-average-treatment-effect)
  - [2.5 Power and Type I or Type II errors](#25-power-and-type-i-or-type-ii-errors)
- [3. Experiment Design](#3-experiment-design)
  - [3.1 Hypotheses and decision rules](#31-hypotheses-and-decision-rules)
  - [3.2 Metric hierarchy](#32-metric-hierarchy)
  - [3.3 Sample sizing](#33-sample-sizing)
  - [3.4 Stratification and blocking](#34-stratification-and-blocking)
  - [3.5 Randomization checks](#35-randomization-checks)
- [4. Inference for AB Tests](#4-inference-for-ab-tests)
  - [4.1 Difference in means and proportions](#41-difference-in-means-and-proportions)
  - [4.2 t tests and z tests](#42-t-tests-and-z-tests)
  - [4.3 Bootstrap confidence intervals](#43-bootstrap-confidence-intervals)
  - [4.4 CUPED preview](#44-cuped-preview)
  - [4.5 Heterogeneous treatment effects](#45-heterogeneous-treatment-effects)
- [5. Sequential Testing](#5-sequential-testing)
  - [5.1 Peeking risk](#51-peeking-risk)
  - [5.2 Alpha spending](#52-alpha-spending)
  - [5.3 Always-valid p-values](#53-always-valid-p-values)
  - [5.4 Stopping rules](#54-stopping-rules)
  - [5.5 Multiple online tests](#55-multiple-online-tests)
- [6. LLM Product Experiments](#6-llm-product-experiments)
  - [6.1 Model routing](#61-model-routing)
  - [6.2 Prompt variants](#62-prompt-variants)
  - [6.3 Latency-quality tradeoffs](#63-latency-quality-tradeoffs)
  - [6.4 Human ratings](#64-human-ratings)
  - [6.5 Safety guardrails](#65-safety-guardrails)
- [7. Trustworthiness Checks](#7-trustworthiness-checks)
  - [7.1 A/A tests](#71-aa-tests)
  - [7.2 Sample-ratio mismatch](#72-sample-ratio-mismatch)
  - [7.3 Novelty and carryover effects](#73-novelty-and-carryover-effects)
  - [7.4 Interference](#74-interference)
  - [7.5 Logging and auditability](#75-logging-and-auditability)
- [8. Common Mistakes](#8-common-mistakes)
- [9. Exercises](#9-exercises)
- [10. Why This Matters for AI](#10-why-this-matters-for-ai)
- [11. Conceptual Bridge](#11-conceptual-bridge)
- [References](#references)

---

## 1. Intuition

Intuition is the part of online experimentation and ab testing that turns the approved
TOC into a concrete learning path. The subsections below keep the focus on Chapter 17's
canonical job: measurement, reliability, uncertainty, and decision support for AI
systems.

### 1.1 Offline eval predicts, online tests measure

Offline eval predicts, online tests measure is part of the canonical scope of online
experimentation and ab testing. In this chapter, the object under study is not merely a
dataset or a model, but the full online randomized experiment: the items, prompts,
outputs, graders, uncertainty statements, and decision rules that turn model behavior
into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\widehat{\operatorname{ATE}} = \frac{1}{n}\sum_{i=1}^n Y_i(1)-Y_i(0).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For offline eval predicts,
online tests measure, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in offline eval predicts, online tests measure |
| Scoring rule | Exact formula for Y_i(1)-Y_i(0) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report offline eval predicts, online tests measure with item count, prompt protocol,
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

Worked evaluation pattern for offline eval predicts, online tests measure:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, offline eval predicts, online tests measure is especially delicate
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
because it is noisy; it is interpreted through the design that produced it. Offline eval
predicts, online tests measure is one place where that habit becomes concrete.

### 1.2 Randomization as causal design

Randomization as causal design is part of the canonical scope of online experimentation
and ab testing. In this chapter, the object under study is not merely a dataset or a
model, but the full online randomized experiment: the items, prompts, outputs, graders,
uncertainty statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\widehat{\operatorname{ATE}} = \frac{1}{n}\sum_{i=1}^n Y_i(1)-Y_i(0).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For randomization as causal
design, those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in randomization as causal design |
| Scoring rule | Exact formula for Y_i(1)-Y_i(0) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report randomization as causal design with item count, prompt protocol, grader
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

Worked evaluation pattern for randomization as causal design:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, randomization as causal design is especially delicate because the same
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
because it is noisy; it is interpreted through the design that produced it.
Randomization as causal design is one place where that habit becomes concrete.

### 1.3 Overall evaluation criterion

Overall evaluation criterion is part of the canonical scope of online experimentation
and ab testing. In this chapter, the object under study is not merely a dataset or a
model, but the full online randomized experiment: the items, prompts, outputs, graders,
uncertainty statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\widehat{\operatorname{ATE}} = \frac{1}{n}\sum_{i=1}^n Y_i(1)-Y_i(0).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For overall evaluation
criterion, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in overall evaluation criterion |
| Scoring rule | Exact formula for Y_i(1)-Y_i(0) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report overall evaluation criterion with item count, prompt protocol, grader version,
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

Worked evaluation pattern for overall evaluation criterion:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, overall evaluation criterion is especially delicate because the same
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
because it is noisy; it is interpreted through the design that produced it. Overall
evaluation criterion is one place where that habit becomes concrete.

### 1.4 Guardrails and downside risk

Guardrails and downside risk is part of the canonical scope of online experimentation
and ab testing. In this chapter, the object under study is not merely a dataset or a
model, but the full online randomized experiment: the items, prompts, outputs, graders,
uncertainty statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\widehat{\operatorname{ATE}} = \frac{1}{n}\sum_{i=1}^n Y_i(1)-Y_i(0).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For guardrails and downside
risk, those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in guardrails and downside risk |
| Scoring rule | Exact formula for Y_i(1)-Y_i(0) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report guardrails and downside risk with item count, prompt protocol, grader version,
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

Worked evaluation pattern for guardrails and downside risk:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, guardrails and downside risk is especially delicate because the same
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
because it is noisy; it is interpreted through the design that produced it. Guardrails
and downside risk is one place where that habit becomes concrete.

### 1.5 Experiment culture for LLM systems

Experiment culture for LLM systems is part of the canonical scope of online
experimentation and ab testing. In this chapter, the object under study is not merely a
dataset or a model, but the full online randomized experiment: the items, prompts,
outputs, graders, uncertainty statements, and decision rules that turn model behavior
into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\widehat{\operatorname{ATE}} = \frac{1}{n}\sum_{i=1}^n Y_i(1)-Y_i(0).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For experiment culture for
llm systems, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in experiment culture for llm systems |
| Scoring rule | Exact formula for Y_i(1)-Y_i(0) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report experiment culture for llm systems with item count, prompt protocol, grader
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

Worked evaluation pattern for experiment culture for llm systems:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, experiment culture for llm systems is especially delicate because the
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
because it is noisy; it is interpreted through the design that produced it. Experiment
culture for LLM systems is one place where that habit becomes concrete.

## 2. Formal Definitions

Formal Definitions is the part of online experimentation and ab testing that turns the
approved TOC into a concrete learning path. The subsections below keep the focus on
Chapter 17's canonical job: measurement, reliability, uncertainty, and decision support
for AI systems.

### 2.1 Treatment and control

Treatment and control is part of the canonical scope of online experimentation and ab
testing. In this chapter, the object under study is not merely a dataset or a model, but
the full online randomized experiment: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\widehat{\operatorname{ATE}} = \frac{1}{n}\sum_{i=1}^n Y_i(1)-Y_i(0).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For treatment and control,
those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in treatment and control |
| Scoring rule | Exact formula for Y_i(1)-Y_i(0) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report treatment and control with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for treatment and control:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, treatment and control is especially delicate because the same model can
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
because it is noisy; it is interpreted through the design that produced it. Treatment
and control is one place where that habit becomes concrete.

### 2.2 Randomization unit

Randomization unit is part of the canonical scope of online experimentation and ab
testing. In this chapter, the object under study is not merely a dataset or a model, but
the full online randomized experiment: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\widehat{\operatorname{ATE}} = \frac{1}{n}\sum_{i=1}^n Y_i(1)-Y_i(0).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For randomization unit,
those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in randomization unit |
| Scoring rule | Exact formula for Y_i(1)-Y_i(0) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report randomization unit with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for randomization unit:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, randomization unit is especially delicate because the same model can be
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
because it is noisy; it is interpreted through the design that produced it.
Randomization unit is one place where that habit becomes concrete.

### 2.3 Overall evaluation criterion and guardrails

Overall evaluation criterion and guardrails is part of the canonical scope of online
experimentation and ab testing. In this chapter, the object under study is not merely a
dataset or a model, but the full online randomized experiment: the items, prompts,
outputs, graders, uncertainty statements, and decision rules that turn model behavior
into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\widehat{\operatorname{ATE}} = \frac{1}{n}\sum_{i=1}^n Y_i(1)-Y_i(0).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For overall evaluation
criterion and guardrails, those choices determine whether the reported number is
evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in overall evaluation criterion and guardrails |
| Scoring rule | Exact formula for Y_i(1)-Y_i(0) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report overall evaluation criterion and guardrails with item count, prompt protocol,
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

Worked evaluation pattern for overall evaluation criterion and guardrails:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, overall evaluation criterion and guardrails is especially delicate
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
because it is noisy; it is interpreted through the design that produced it. Overall
evaluation criterion and guardrails is one place where that habit becomes concrete.

### 2.4 Average treatment effect

Average treatment effect is part of the canonical scope of online experimentation and ab
testing. In this chapter, the object under study is not merely a dataset or a model, but
the full online randomized experiment: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\widehat{\operatorname{ATE}} = \frac{1}{n}\sum_{i=1}^n Y_i(1)-Y_i(0).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For average treatment
effect, those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in average treatment effect |
| Scoring rule | Exact formula for Y_i(1)-Y_i(0) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report average treatment effect with item count, prompt protocol, grader version, and
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

Worked evaluation pattern for average treatment effect:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, average treatment effect is especially delicate because the same model
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
because it is noisy; it is interpreted through the design that produced it. Average
treatment effect is one place where that habit becomes concrete.

### 2.5 Power and Type I or Type II errors

Power and Type I or Type II errors is part of the canonical scope of online
experimentation and ab testing. In this chapter, the object under study is not merely a
dataset or a model, but the full online randomized experiment: the items, prompts,
outputs, graders, uncertainty statements, and decision rules that turn model behavior
into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\widehat{\operatorname{ATE}} = \frac{1}{n}\sum_{i=1}^n Y_i(1)-Y_i(0).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For power and type i or type
ii errors, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in power and type i or type ii errors |
| Scoring rule | Exact formula for Y_i(1)-Y_i(0) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report power and type i or type ii errors with item count, prompt protocol, grader
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

Worked evaluation pattern for power and type i or type ii errors:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, power and type i or type ii errors is especially delicate because the
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
because it is noisy; it is interpreted through the design that produced it. Power and
Type I or Type II errors is one place where that habit becomes concrete.

## 3. Experiment Design

Experiment Design is the part of online experimentation and ab testing that turns the
approved TOC into a concrete learning path. The subsections below keep the focus on
Chapter 17's canonical job: measurement, reliability, uncertainty, and decision support
for AI systems.

### 3.1 Hypotheses and decision rules

Hypotheses and decision rules is part of the canonical scope of online experimentation
and ab testing. In this chapter, the object under study is not merely a dataset or a
model, but the full online randomized experiment: the items, prompts, outputs, graders,
uncertainty statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\widehat{\operatorname{ATE}} = \frac{1}{n}\sum_{i=1}^n Y_i(1)-Y_i(0).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For hypotheses and decision
rules, those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in hypotheses and decision rules |
| Scoring rule | Exact formula for Y_i(1)-Y_i(0) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report hypotheses and decision rules with item count, prompt protocol, grader version,
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

Worked evaluation pattern for hypotheses and decision rules:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, hypotheses and decision rules is especially delicate because the same
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
because it is noisy; it is interpreted through the design that produced it. Hypotheses
and decision rules is one place where that habit becomes concrete.

### 3.2 Metric hierarchy

Metric hierarchy is part of the canonical scope of online experimentation and ab
testing. In this chapter, the object under study is not merely a dataset or a model, but
the full online randomized experiment: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\widehat{\operatorname{ATE}} = \frac{1}{n}\sum_{i=1}^n Y_i(1)-Y_i(0).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For metric hierarchy, those
choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in metric hierarchy |
| Scoring rule | Exact formula for Y_i(1)-Y_i(0) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report metric hierarchy with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for metric hierarchy:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, metric hierarchy is especially delicate because the same model can be
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
because it is noisy; it is interpreted through the design that produced it. Metric
hierarchy is one place where that habit becomes concrete.

### 3.3 Sample sizing

Sample sizing is part of the canonical scope of online experimentation and ab testing.
In this chapter, the object under study is not merely a dataset or a model, but the full
online randomized experiment: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\widehat{\operatorname{ATE}} = \frac{1}{n}\sum_{i=1}^n Y_i(1)-Y_i(0).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For sample sizing, those
choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in sample sizing |
| Scoring rule | Exact formula for Y_i(1)-Y_i(0) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report sample sizing with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for sample sizing:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, sample sizing is especially delicate because the same model can be used
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
because it is noisy; it is interpreted through the design that produced it. Sample
sizing is one place where that habit becomes concrete.

### 3.4 Stratification and blocking

Stratification and blocking is part of the canonical scope of online experimentation and
ab testing. In this chapter, the object under study is not merely a dataset or a model,
but the full online randomized experiment: the items, prompts, outputs, graders,
uncertainty statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\widehat{\operatorname{ATE}} = \frac{1}{n}\sum_{i=1}^n Y_i(1)-Y_i(0).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For stratification and
blocking, those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in stratification and blocking |
| Scoring rule | Exact formula for Y_i(1)-Y_i(0) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report stratification and blocking with item count, prompt protocol, grader version,
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

Worked evaluation pattern for stratification and blocking:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, stratification and blocking is especially delicate because the same
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
because it is noisy; it is interpreted through the design that produced it.
Stratification and blocking is one place where that habit becomes concrete.

### 3.5 Randomization checks

Randomization checks is part of the canonical scope of online experimentation and ab
testing. In this chapter, the object under study is not merely a dataset or a model, but
the full online randomized experiment: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\widehat{\operatorname{ATE}} = \frac{1}{n}\sum_{i=1}^n Y_i(1)-Y_i(0).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For randomization checks,
those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in randomization checks |
| Scoring rule | Exact formula for Y_i(1)-Y_i(0) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report randomization checks with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for randomization checks:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, randomization checks is especially delicate because the same model can
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
because it is noisy; it is interpreted through the design that produced it.
Randomization checks is one place where that habit becomes concrete.

## 4. Inference for AB Tests

Inference for AB Tests is the part of online experimentation and ab testing that turns
the approved TOC into a concrete learning path. The subsections below keep the focus on
Chapter 17's canonical job: measurement, reliability, uncertainty, and decision support
for AI systems.

### 4.1 Difference in means and proportions

Difference in means and proportions is part of the canonical scope of online
experimentation and ab testing. In this chapter, the object under study is not merely a
dataset or a model, but the full online randomized experiment: the items, prompts,
outputs, graders, uncertainty statements, and decision rules that turn model behavior
into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\widehat{\operatorname{ATE}} = \frac{1}{n}\sum_{i=1}^n Y_i(1)-Y_i(0).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For difference in means and
proportions, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in difference in means and proportions |
| Scoring rule | Exact formula for Y_i(1)-Y_i(0) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report difference in means and proportions with item count, prompt protocol, grader
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

Worked evaluation pattern for difference in means and proportions:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, difference in means and proportions is especially delicate because the
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
because it is noisy; it is interpreted through the design that produced it. Difference
in means and proportions is one place where that habit becomes concrete.

### 4.2 t tests and z tests

t tests and z tests is part of the canonical scope of online experimentation and ab
testing. In this chapter, the object under study is not merely a dataset or a model, but
the full online randomized experiment: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\widehat{\operatorname{ATE}} = \frac{1}{n}\sum_{i=1}^n Y_i(1)-Y_i(0).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For t tests and z tests,
those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in t tests and z tests |
| Scoring rule | Exact formula for Y_i(1)-Y_i(0) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report t tests and z tests with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for t tests and z tests:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, t tests and z tests is especially delicate because the same model can be
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
because it is noisy; it is interpreted through the design that produced it. t tests and
z tests is one place where that habit becomes concrete.

### 4.3 Bootstrap confidence intervals

Bootstrap confidence intervals is part of the canonical scope of online experimentation
and ab testing. In this chapter, the object under study is not merely a dataset or a
model, but the full online randomized experiment: the items, prompts, outputs, graders,
uncertainty statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\widehat{\operatorname{ATE}} = \frac{1}{n}\sum_{i=1}^n Y_i(1)-Y_i(0).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For bootstrap confidence
intervals, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in bootstrap confidence intervals |
| Scoring rule | Exact formula for Y_i(1)-Y_i(0) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report bootstrap confidence intervals with item count, prompt protocol, grader
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

Worked evaluation pattern for bootstrap confidence intervals:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, bootstrap confidence intervals is especially delicate because the same
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
because it is noisy; it is interpreted through the design that produced it. Bootstrap
confidence intervals is one place where that habit becomes concrete.

### 4.4 CUPED preview

CUPED preview is part of the canonical scope of online experimentation and ab testing.
In this chapter, the object under study is not merely a dataset or a model, but the full
online randomized experiment: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\widehat{\operatorname{ATE}} = \frac{1}{n}\sum_{i=1}^n Y_i(1)-Y_i(0).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For cuped preview, those
choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in cuped preview |
| Scoring rule | Exact formula for Y_i(1)-Y_i(0) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report cuped preview with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for cuped preview:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, cuped preview is especially delicate because the same model can be used
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
because it is noisy; it is interpreted through the design that produced it. CUPED
preview is one place where that habit becomes concrete.

### 4.5 Heterogeneous treatment effects

Heterogeneous treatment effects is part of the canonical scope of online experimentation
and ab testing. In this chapter, the object under study is not merely a dataset or a
model, but the full online randomized experiment: the items, prompts, outputs, graders,
uncertainty statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\widehat{\operatorname{ATE}} = \frac{1}{n}\sum_{i=1}^n Y_i(1)-Y_i(0).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For heterogeneous treatment
effects, those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in heterogeneous treatment effects |
| Scoring rule | Exact formula for Y_i(1)-Y_i(0) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report heterogeneous treatment effects with item count, prompt protocol, grader
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

Worked evaluation pattern for heterogeneous treatment effects:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, heterogeneous treatment effects is especially delicate because the same
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
because it is noisy; it is interpreted through the design that produced it.
Heterogeneous treatment effects is one place where that habit becomes concrete.

## 5. Sequential Testing

Sequential Testing is the part of online experimentation and ab testing that turns the
approved TOC into a concrete learning path. The subsections below keep the focus on
Chapter 17's canonical job: measurement, reliability, uncertainty, and decision support
for AI systems.

### 5.1 Peeking risk

Peeking risk is part of the canonical scope of online experimentation and ab testing. In
this chapter, the object under study is not merely a dataset or a model, but the full
online randomized experiment: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\widehat{\operatorname{ATE}} = \frac{1}{n}\sum_{i=1}^n Y_i(1)-Y_i(0).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For peeking risk, those
choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in peeking risk |
| Scoring rule | Exact formula for Y_i(1)-Y_i(0) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report peeking risk with item count, prompt protocol, grader version, and a confidence
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

Worked evaluation pattern for peeking risk:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, peeking risk is especially delicate because the same model can be used
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
because it is noisy; it is interpreted through the design that produced it. Peeking risk
is one place where that habit becomes concrete.

### 5.2 Alpha spending

Alpha spending is part of the canonical scope of online experimentation and ab testing.
In this chapter, the object under study is not merely a dataset or a model, but the full
online randomized experiment: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\widehat{\operatorname{ATE}} = \frac{1}{n}\sum_{i=1}^n Y_i(1)-Y_i(0).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For alpha spending, those
choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in alpha spending |
| Scoring rule | Exact formula for Y_i(1)-Y_i(0) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report alpha spending with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for alpha spending:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, alpha spending is especially delicate because the same model can be used
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
because it is noisy; it is interpreted through the design that produced it. Alpha
spending is one place where that habit becomes concrete.

### 5.3 Always-valid p-values

Always-valid p-values is part of the canonical scope of online experimentation and ab
testing. In this chapter, the object under study is not merely a dataset or a model, but
the full online randomized experiment: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\widehat{\operatorname{ATE}} = \frac{1}{n}\sum_{i=1}^n Y_i(1)-Y_i(0).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For always-valid p-values,
those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in always-valid p-values |
| Scoring rule | Exact formula for Y_i(1)-Y_i(0) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report always-valid p-values with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for always-valid p-values:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, always-valid p-values is especially delicate because the same model can
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
because it is noisy; it is interpreted through the design that produced it. Always-valid
p-values is one place where that habit becomes concrete.

### 5.4 Stopping rules

Stopping rules is part of the canonical scope of online experimentation and ab testing.
In this chapter, the object under study is not merely a dataset or a model, but the full
online randomized experiment: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\widehat{\operatorname{ATE}} = \frac{1}{n}\sum_{i=1}^n Y_i(1)-Y_i(0).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For stopping rules, those
choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in stopping rules |
| Scoring rule | Exact formula for Y_i(1)-Y_i(0) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report stopping rules with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for stopping rules:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, stopping rules is especially delicate because the same model can be used
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
because it is noisy; it is interpreted through the design that produced it. Stopping
rules is one place where that habit becomes concrete.

### 5.5 Multiple online tests

Multiple online tests is part of the canonical scope of online experimentation and ab
testing. In this chapter, the object under study is not merely a dataset or a model, but
the full online randomized experiment: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\widehat{\operatorname{ATE}} = \frac{1}{n}\sum_{i=1}^n Y_i(1)-Y_i(0).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For multiple online tests,
those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in multiple online tests |
| Scoring rule | Exact formula for Y_i(1)-Y_i(0) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report multiple online tests with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for multiple online tests:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, multiple online tests is especially delicate because the same model can
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
because it is noisy; it is interpreted through the design that produced it. Multiple
online tests is one place where that habit becomes concrete.

## 6. LLM Product Experiments

LLM Product Experiments is the part of online experimentation and ab testing that turns
the approved TOC into a concrete learning path. The subsections below keep the focus on
Chapter 17's canonical job: measurement, reliability, uncertainty, and decision support
for AI systems.

### 6.1 Model routing

Model routing is part of the canonical scope of online experimentation and ab testing.
In this chapter, the object under study is not merely a dataset or a model, but the full
online randomized experiment: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\widehat{\operatorname{ATE}} = \frac{1}{n}\sum_{i=1}^n Y_i(1)-Y_i(0).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For model routing, those
choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in model routing |
| Scoring rule | Exact formula for Y_i(1)-Y_i(0) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report model routing with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for model routing:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, model routing is especially delicate because the same model can be used
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
because it is noisy; it is interpreted through the design that produced it. Model
routing is one place where that habit becomes concrete.

### 6.2 Prompt variants

Prompt variants is part of the canonical scope of online experimentation and ab testing.
In this chapter, the object under study is not merely a dataset or a model, but the full
online randomized experiment: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\widehat{\operatorname{ATE}} = \frac{1}{n}\sum_{i=1}^n Y_i(1)-Y_i(0).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For prompt variants, those
choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in prompt variants |
| Scoring rule | Exact formula for Y_i(1)-Y_i(0) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report prompt variants with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for prompt variants:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, prompt variants is especially delicate because the same model can be
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
because it is noisy; it is interpreted through the design that produced it. Prompt
variants is one place where that habit becomes concrete.

### 6.3 Latency-quality tradeoffs

Latency-quality tradeoffs is part of the canonical scope of online experimentation and
ab testing. In this chapter, the object under study is not merely a dataset or a model,
but the full online randomized experiment: the items, prompts, outputs, graders,
uncertainty statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\widehat{\operatorname{ATE}} = \frac{1}{n}\sum_{i=1}^n Y_i(1)-Y_i(0).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For latency-quality
tradeoffs, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in latency-quality tradeoffs |
| Scoring rule | Exact formula for Y_i(1)-Y_i(0) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report latency-quality tradeoffs with item count, prompt protocol, grader version, and
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

Worked evaluation pattern for latency-quality tradeoffs:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, latency-quality tradeoffs is especially delicate because the same model
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
because it is noisy; it is interpreted through the design that produced it. Latency-
quality tradeoffs is one place where that habit becomes concrete.

### 6.4 Human ratings

Human ratings is part of the canonical scope of online experimentation and ab testing.
In this chapter, the object under study is not merely a dataset or a model, but the full
online randomized experiment: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\widehat{\operatorname{ATE}} = \frac{1}{n}\sum_{i=1}^n Y_i(1)-Y_i(0).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For human ratings, those
choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in human ratings |
| Scoring rule | Exact formula for Y_i(1)-Y_i(0) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report human ratings with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for human ratings:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, human ratings is especially delicate because the same model can be used
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
because it is noisy; it is interpreted through the design that produced it. Human
ratings is one place where that habit becomes concrete.

### 6.5 Safety guardrails

Safety guardrails is part of the canonical scope of online experimentation and ab
testing. In this chapter, the object under study is not merely a dataset or a model, but
the full online randomized experiment: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\widehat{\operatorname{ATE}} = \frac{1}{n}\sum_{i=1}^n Y_i(1)-Y_i(0).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For safety guardrails, those
choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in safety guardrails |
| Scoring rule | Exact formula for Y_i(1)-Y_i(0) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report safety guardrails with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for safety guardrails:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, safety guardrails is especially delicate because the same model can be
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
because it is noisy; it is interpreted through the design that produced it. Safety
guardrails is one place where that habit becomes concrete.

## 7. Trustworthiness Checks

Trustworthiness Checks is the part of online experimentation and ab testing that turns
the approved TOC into a concrete learning path. The subsections below keep the focus on
Chapter 17's canonical job: measurement, reliability, uncertainty, and decision support
for AI systems.

### 7.1 A/A tests

A/A tests is part of the canonical scope of online experimentation and ab testing. In
this chapter, the object under study is not merely a dataset or a model, but the full
online randomized experiment: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\widehat{\operatorname{ATE}} = \frac{1}{n}\sum_{i=1}^n Y_i(1)-Y_i(0).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For a/a tests, those choices
determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in a/a tests |
| Scoring rule | Exact formula for Y_i(1)-Y_i(0) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report a/a tests with item count, prompt protocol, grader version, and a confidence
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

Worked evaluation pattern for a/a tests:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, a/a tests is especially delicate because the same model can be used with
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
because it is noisy; it is interpreted through the design that produced it. A/A tests is
one place where that habit becomes concrete.

### 7.2 Sample-ratio mismatch

Sample-ratio mismatch is part of the canonical scope of online experimentation and ab
testing. In this chapter, the object under study is not merely a dataset or a model, but
the full online randomized experiment: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\widehat{\operatorname{ATE}} = \frac{1}{n}\sum_{i=1}^n Y_i(1)-Y_i(0).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For sample-ratio mismatch,
those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in sample-ratio mismatch |
| Scoring rule | Exact formula for Y_i(1)-Y_i(0) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report sample-ratio mismatch with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for sample-ratio mismatch:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, sample-ratio mismatch is especially delicate because the same model can
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
because it is noisy; it is interpreted through the design that produced it. Sample-ratio
mismatch is one place where that habit becomes concrete.

### 7.3 Novelty and carryover effects

Novelty and carryover effects is part of the canonical scope of online experimentation
and ab testing. In this chapter, the object under study is not merely a dataset or a
model, but the full online randomized experiment: the items, prompts, outputs, graders,
uncertainty statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\widehat{\operatorname{ATE}} = \frac{1}{n}\sum_{i=1}^n Y_i(1)-Y_i(0).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For novelty and carryover
effects, those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in novelty and carryover effects |
| Scoring rule | Exact formula for Y_i(1)-Y_i(0) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report novelty and carryover effects with item count, prompt protocol, grader version,
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

Worked evaluation pattern for novelty and carryover effects:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, novelty and carryover effects is especially delicate because the same
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
because it is noisy; it is interpreted through the design that produced it. Novelty and
carryover effects is one place where that habit becomes concrete.

### 7.4 Interference

Interference is part of the canonical scope of online experimentation and ab testing. In
this chapter, the object under study is not merely a dataset or a model, but the full
online randomized experiment: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\widehat{\operatorname{ATE}} = \frac{1}{n}\sum_{i=1}^n Y_i(1)-Y_i(0).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For interference, those
choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in interference |
| Scoring rule | Exact formula for Y_i(1)-Y_i(0) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report interference with item count, prompt protocol, grader version, and a confidence
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

Worked evaluation pattern for interference:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, interference is especially delicate because the same model can be used
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
because it is noisy; it is interpreted through the design that produced it. Interference
is one place where that habit becomes concrete.

### 7.5 Logging and auditability

Logging and auditability is part of the canonical scope of online experimentation and ab
testing. In this chapter, the object under study is not merely a dataset or a model, but
the full online randomized experiment: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\widehat{\operatorname{ATE}} = \frac{1}{n}\sum_{i=1}^n Y_i(1)-Y_i(0).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For logging and
auditability, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in logging and auditability |
| Scoring rule | Exact formula for Y_i(1)-Y_i(0) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report logging and auditability with item count, prompt protocol, grader version, and
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

Worked evaluation pattern for logging and auditability:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, logging and auditability is especially delicate because the same model
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
because it is noisy; it is interpreted through the design that produced it. Logging and
auditability is one place where that habit becomes concrete.

## 8. Common Mistakes

| # | Mistake | Why It Is Wrong | Fix |
| --- | --- | --- | --- |
| 1 | Treating a point estimate as exact | Every finite evaluation has sampling error in online experimentation and ab testing. | Report uncertainty with the point estimate. |
| 2 | Changing prompts between models | The protocol changed with the treatment in online experimentation and ab testing. | Lock prompt, decoding, and grader before comparison. |
| 3 | Ignoring invalid outputs | Missingness can be correlated with model quality in online experimentation and ab testing. | Track invalid, timeout, refusal, and parse-failure rates. |
| 4 | Overfitting to a public leaderboard | Repeated testing leaks information from the benchmark in online experimentation and ab testing. | Use private holdouts and regression suites. |
| 5 | Averaging incomparable metrics | Different scales do not have shared units in online experimentation and ab testing. | Normalize by a stated decision rule or report separately. |
| 6 | Forgetting paired structure | Two models often answer the same items in online experimentation and ab testing. | Use paired bootstrap or paired tests where possible. |
| 7 | Reporting only aggregate performance | Subgroup failures can be hidden in online experimentation and ab testing. | Add slice and tail-risk views. |
| 8 | Trusting model judges blindly | LLM judges have position, verbosity, and self-preference biases in online experimentation and ab testing. | Calibrate judges against human labels. |
| 9 | Peeking during online experiments | Optional stopping inflates false positives in online experimentation and ab testing. | Use fixed horizons or sequential-valid methods. |
| 10 | Conflating evaluation with monitoring | Chapter 17 measures controlled evidence; production monitoring is ongoing operations in online experimentation and ab testing. | Hand off drift dashboards to Chapter 19 concepts. |

## 9. Exercises

1. (*) Offline eval predicts, online tests measure.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

2. (*) Randomization as causal design.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

3. (*) Overall evaluation criterion.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

4. (**) Guardrails and downside risk.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

5. (**) Experiment culture for LLM systems.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

6. (**) Treatment and control.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

7. (***) Randomization unit.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

8. (***) Overall evaluation criterion and guardrails.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

9. (***) Average treatment effect.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

10. (***) Power and Type I or Type II errors.
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
asks how deployed systems are observed and maintained over time. Online Experimentation
and AB Testing supplies the measurement discipline both chapters need.

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

- [Trustworthy Online Controlled Experiments](https://www.cambridge.org/core/books/trustworthy-online-controlled-experiments/trustworthy-online-controlled-experiments/524D7D0FDF6A542F732D047118618645)
- [Controlled experiments on the web: survey and practical guide](https://www.microsoft.com/en-us/research/publication/controlled-experiments-on-the-web-survey-and-practical-guide/)
- [Always Valid Inference: Bringing Sequential Analysis to A/B Testing](https://arxiv.org/abs/1512.04922)
- [Hypothesis Testing](../../07-Statistics/03-Hypothesis-Testing/notes.md)
- [OpenAI Evals](https://github.com/openai/evals)
- [Concentration Inequalities](../../06-Probability-Theory/05-Concentration-Inequalities/notes.md)
- [Estimation Theory](../../07-Statistics/02-Estimation-Theory/notes.md)
- [Hypothesis Testing](../../07-Statistics/03-Hypothesis-Testing/notes.md)

