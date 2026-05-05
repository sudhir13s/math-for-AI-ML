[Back to Curriculum](../../README.md) | [Previous: Data Mixture Optimization](../../16-LLM-Training-Data-Pipeline/07-Data-Mixture-Optimization/notes.md) | [Next: Calibration and Uncertainty](../02-Calibration-and-Uncertainty/notes.md)

---

# Capability Benchmarks

> _"A benchmark is a measuring instrument, not a model of reality."_

## Overview

Capability benchmarks estimate what a model can do under a stated protocol; reliability
begins when the protocol, metric, and uncertainty are explicit.

The chapter treats evaluation as a mathematical object: a controlled protocol that maps
model behavior into evidence with uncertainty. A result is not just a number; it is an
estimate produced by a task distribution, a scoring rule, and an aggregation rule.

This section is written in LaTeX Markdown. Inline mathematics uses `$...$`, and display
equations use `$$...$$`. The emphasis is practical but rigorous: every metric should be
linked to the statistical assumptions that make it meaningful.

## Prerequisites

- [Estimation Theory](../../07-Statistics/02-Estimation-Theory/notes.md)
- [Hypothesis Testing](../../07-Statistics/03-Hypothesis-Testing/notes.md)
- [Concentration Inequalities](../../06-Probability-Theory/05-Concentration-Inequalities/notes.md)
- [Language Model Probability](../../15-Math-for-LLMs/05-Language-Model-Probability/notes.md)
- [Contamination and Dedup Audits](../../16-LLM-Training-Data-Pipeline/05-Contamination-and-Dedup-Audits/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Executable demonstrations for capability benchmarks |
| [exercises.ipynb](exercises.ipynb) | Graded practice for capability benchmarks |

## Learning Objectives

After completing this section, you will be able to:

- Define the core objects used in capability benchmarks
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
  - [1.1 Benchmarks as noisy estimators](#11-benchmarks-as-noisy-estimators)
  - [1.2 Capability versus observed score](#12-capability-versus-observed-score)
  - [1.3 Metric pluralism for LLM systems](#13-metric-pluralism-for-llm-systems)
  - [1.4 Benchmark lifecycle and saturation](#14-benchmark-lifecycle-and-saturation)
  - [1.5 What benchmark scores can and cannot certify](#15-what-benchmark-scores-can-and-cannot-certify)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Model and system under test](#21-model-and-system-under-test)
  - [2.2 Task, item, and evaluation sample](#22-task-item-and-evaluation-sample)
  - [2.3 Prompt protocol and decoding policy](#23-prompt-protocol-and-decoding-policy)
  - [2.4 Scorer, metric, and aggregate estimate](#24-scorer-metric-and-aggregate-estimate)
  - [2.5 Confidence interval and leaderboard rank](#25-confidence-interval-and-leaderboard-rank)
- [3. Benchmark Design](#3-benchmark-design)
  - [3.1 Task taxonomy and coverage](#31-task-taxonomy-and-coverage)
  - [3.2 Dataset sampling and item independence](#32-dataset-sampling-and-item-independence)
  - [3.3 Prompt templates and few-shot policy](#33-prompt-templates-and-few-shot-policy)
  - [3.4 Grading functions and rubrics](#34-grading-functions-and-rubrics)
  - [3.5 Contamination flags and eval provenance](#35-contamination-flags-and-eval-provenance)
- [4. Core Metrics](#4-core-metrics)
  - [4.1 Accuracy and exact match](#41-accuracy-and-exact-match)
  - [4.2 Precision, recall, and F1](#42-precision-recall-and-f1)
  - [4.3 Pass at k for code generation](#43-pass-at-k-for-code-generation)
  - [4.4 Log-probability and perplexity](#44-log-probability-and-perplexity)
  - [4.5 Pairwise preference and judge agreement](#45-pairwise-preference-and-judge-agreement)
- [5. Statistical Reliability](#5-statistical-reliability)
  - [5.1 Bootstrap intervals](#51-bootstrap-intervals)
  - [5.2 Paired model comparisons](#52-paired-model-comparisons)
  - [5.3 Leaderboard uncertainty](#53-leaderboard-uncertainty)
  - [5.4 Multiple comparisons](#54-multiple-comparisons)
  - [5.5 Benchmark power and sample size](#55-benchmark-power-and-sample-size)
- [6. LLM Benchmark Families](#6-llm-benchmark-families)
  - [6.1 MMLU and multitask knowledge](#61-mmlu-and-multitask-knowledge)
  - [6.2 HumanEval and functional correctness](#62-humaneval-and-functional-correctness)
  - [6.3 BIG-bench and capability extrapolation](#63-big-bench-and-capability-extrapolation)
  - [6.4 HELM and multi-metric transparency](#64-helm-and-multi-metric-transparency)
  - [6.5 MT-Bench, Chatbot Arena, and LLM-as-judge](#65-mt-bench-chatbot-arena-and-llm-as-judge)
- [7. Common Mistakes](#7-common-mistakes)
- [8. Exercises](#8-exercises)
- [9. Why This Matters for AI](#9-why-this-matters-for-ai)
- [10. Conceptual Bridge](#10-conceptual-bridge)
- [References](#references)

---

## 1. Intuition

Intuition is the part of capability benchmarks that turns the approved TOC into a
concrete learning path. The subsections below keep the focus on Chapter 17's canonical
job: measurement, reliability, uncertainty, and decision support for AI systems.

### 1.1 Benchmarks as noisy estimators

Benchmarks as noisy estimators is part of the canonical scope of capability benchmarks.
In this chapter, the object under study is not merely a dataset or a model, but the full
benchmark protocol: the items, prompts, outputs, graders, uncertainty statements, and
decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{\mu}_{m,t} = \frac{1}{n}\sum_{i=1}^n s_m(z_i).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For benchmarks as noisy
estimators, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in benchmarks as noisy estimators |
| Scoring rule | Exact formula for s_m(z_i) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report benchmarks as noisy estimators with item count, prompt protocol, grader
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

Worked evaluation pattern for benchmarks as noisy estimators:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, benchmarks as noisy estimators is especially delicate because the same
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
because it is noisy; it is interpreted through the design that produced it. Benchmarks
as noisy estimators is one place where that habit becomes concrete.

### 1.2 Capability versus observed score

Capability versus observed score is part of the canonical scope of capability
benchmarks. In this chapter, the object under study is not merely a dataset or a model,
but the full benchmark protocol: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{\mu}_{m,t} = \frac{1}{n}\sum_{i=1}^n s_m(z_i).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For capability versus
observed score, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in capability versus observed score |
| Scoring rule | Exact formula for s_m(z_i) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report capability versus observed score with item count, prompt protocol, grader
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

Worked evaluation pattern for capability versus observed score:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, capability versus observed score is especially delicate because the same
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
because it is noisy; it is interpreted through the design that produced it. Capability
versus observed score is one place where that habit becomes concrete.

### 1.3 Metric pluralism for LLM systems

Metric pluralism for LLM systems is part of the canonical scope of capability
benchmarks. In this chapter, the object under study is not merely a dataset or a model,
but the full benchmark protocol: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{\mu}_{m,t} = \frac{1}{n}\sum_{i=1}^n s_m(z_i).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For metric pluralism for llm
systems, those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in metric pluralism for llm systems |
| Scoring rule | Exact formula for s_m(z_i) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report metric pluralism for llm systems with item count, prompt protocol, grader
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

Worked evaluation pattern for metric pluralism for llm systems:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, metric pluralism for llm systems is especially delicate because the same
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
because it is noisy; it is interpreted through the design that produced it. Metric
pluralism for LLM systems is one place where that habit becomes concrete.

### 1.4 Benchmark lifecycle and saturation

Benchmark lifecycle and saturation is part of the canonical scope of capability
benchmarks. In this chapter, the object under study is not merely a dataset or a model,
but the full benchmark protocol: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{\mu}_{m,t} = \frac{1}{n}\sum_{i=1}^n s_m(z_i).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For benchmark lifecycle and
saturation, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in benchmark lifecycle and saturation |
| Scoring rule | Exact formula for s_m(z_i) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report benchmark lifecycle and saturation with item count, prompt protocol, grader
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

Worked evaluation pattern for benchmark lifecycle and saturation:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, benchmark lifecycle and saturation is especially delicate because the
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
because it is noisy; it is interpreted through the design that produced it. Benchmark
lifecycle and saturation is one place where that habit becomes concrete.

### 1.5 What benchmark scores can and cannot certify

What benchmark scores can and cannot certify is part of the canonical scope of
capability benchmarks. In this chapter, the object under study is not merely a dataset
or a model, but the full benchmark protocol: the items, prompts, outputs, graders,
uncertainty statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{\mu}_{m,t} = \frac{1}{n}\sum_{i=1}^n s_m(z_i).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For what benchmark scores
can and cannot certify, those choices determine whether the reported number is evidence
or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in what benchmark scores can and cannot certify |
| Scoring rule | Exact formula for s_m(z_i) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report what benchmark scores can and cannot certify with item count, prompt protocol,
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

Worked evaluation pattern for what benchmark scores can and cannot certify:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, what benchmark scores can and cannot certify is especially delicate
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
because it is noisy; it is interpreted through the design that produced it. What
benchmark scores can and cannot certify is one place where that habit becomes concrete.

## 2. Formal Definitions

Formal Definitions is the part of capability benchmarks that turns the approved TOC into
a concrete learning path. The subsections below keep the focus on Chapter 17's canonical
job: measurement, reliability, uncertainty, and decision support for AI systems.

### 2.1 Model and system under test

Model and system under test is part of the canonical scope of capability benchmarks. In
this chapter, the object under study is not merely a dataset or a model, but the full
benchmark protocol: the items, prompts, outputs, graders, uncertainty statements, and
decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{\mu}_{m,t} = \frac{1}{n}\sum_{i=1}^n s_m(z_i).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For model and system under
test, those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in model and system under test |
| Scoring rule | Exact formula for s_m(z_i) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report model and system under test with item count, prompt protocol, grader version,
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

Worked evaluation pattern for model and system under test:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, model and system under test is especially delicate because the same
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
because it is noisy; it is interpreted through the design that produced it. Model and
system under test is one place where that habit becomes concrete.

### 2.2 Task, item, and evaluation sample

Task, item, and evaluation sample is part of the canonical scope of capability
benchmarks. In this chapter, the object under study is not merely a dataset or a model,
but the full benchmark protocol: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{\mu}_{m,t} = \frac{1}{n}\sum_{i=1}^n s_m(z_i).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For task, item, and
evaluation sample, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in task, item, and evaluation sample |
| Scoring rule | Exact formula for s_m(z_i) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report task, item, and evaluation sample with item count, prompt protocol, grader
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

Worked evaluation pattern for task, item, and evaluation sample:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, task, item, and evaluation sample is especially delicate because the
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
because it is noisy; it is interpreted through the design that produced it. Task, item,
and evaluation sample is one place where that habit becomes concrete.

### 2.3 Prompt protocol and decoding policy

Prompt protocol and decoding policy is part of the canonical scope of capability
benchmarks. In this chapter, the object under study is not merely a dataset or a model,
but the full benchmark protocol: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{\mu}_{m,t} = \frac{1}{n}\sum_{i=1}^n s_m(z_i).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For prompt protocol and
decoding policy, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in prompt protocol and decoding policy |
| Scoring rule | Exact formula for s_m(z_i) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report prompt protocol and decoding policy with item count, prompt protocol, grader
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

Worked evaluation pattern for prompt protocol and decoding policy:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, prompt protocol and decoding policy is especially delicate because the
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
because it is noisy; it is interpreted through the design that produced it. Prompt
protocol and decoding policy is one place where that habit becomes concrete.

### 2.4 Scorer, metric, and aggregate estimate

Scorer, metric, and aggregate estimate is part of the canonical scope of capability
benchmarks. In this chapter, the object under study is not merely a dataset or a model,
but the full benchmark protocol: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{\mu}_{m,t} = \frac{1}{n}\sum_{i=1}^n s_m(z_i).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For scorer, metric, and
aggregate estimate, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in scorer, metric, and aggregate estimate |
| Scoring rule | Exact formula for s_m(z_i) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report scorer, metric, and aggregate estimate with item count, prompt protocol, grader
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

Worked evaluation pattern for scorer, metric, and aggregate estimate:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, scorer, metric, and aggregate estimate is especially delicate because
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
because it is noisy; it is interpreted through the design that produced it. Scorer,
metric, and aggregate estimate is one place where that habit becomes concrete.

### 2.5 Confidence interval and leaderboard rank

Confidence interval and leaderboard rank is part of the canonical scope of capability
benchmarks. In this chapter, the object under study is not merely a dataset or a model,
but the full benchmark protocol: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{\mu}_{m,t} = \frac{1}{n}\sum_{i=1}^n s_m(z_i).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For confidence interval and
leaderboard rank, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in confidence interval and leaderboard rank |
| Scoring rule | Exact formula for s_m(z_i) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report confidence interval and leaderboard rank with item count, prompt protocol,
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

Worked evaluation pattern for confidence interval and leaderboard rank:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, confidence interval and leaderboard rank is especially delicate because
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
because it is noisy; it is interpreted through the design that produced it. Confidence
interval and leaderboard rank is one place where that habit becomes concrete.

## 3. Benchmark Design

Benchmark Design is the part of capability benchmarks that turns the approved TOC into a
concrete learning path. The subsections below keep the focus on Chapter 17's canonical
job: measurement, reliability, uncertainty, and decision support for AI systems.

### 3.1 Task taxonomy and coverage

Task taxonomy and coverage is part of the canonical scope of capability benchmarks. In
this chapter, the object under study is not merely a dataset or a model, but the full
benchmark protocol: the items, prompts, outputs, graders, uncertainty statements, and
decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{\mu}_{m,t} = \frac{1}{n}\sum_{i=1}^n s_m(z_i).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For task taxonomy and
coverage, those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in task taxonomy and coverage |
| Scoring rule | Exact formula for s_m(z_i) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report task taxonomy and coverage with item count, prompt protocol, grader version,
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

Worked evaluation pattern for task taxonomy and coverage:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, task taxonomy and coverage is especially delicate because the same model
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
because it is noisy; it is interpreted through the design that produced it. Task
taxonomy and coverage is one place where that habit becomes concrete.

### 3.2 Dataset sampling and item independence

Dataset sampling and item independence is part of the canonical scope of capability
benchmarks. In this chapter, the object under study is not merely a dataset or a model,
but the full benchmark protocol: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{\mu}_{m,t} = \frac{1}{n}\sum_{i=1}^n s_m(z_i).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For dataset sampling and
item independence, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in dataset sampling and item independence |
| Scoring rule | Exact formula for s_m(z_i) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report dataset sampling and item independence with item count, prompt protocol, grader
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

Worked evaluation pattern for dataset sampling and item independence:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, dataset sampling and item independence is especially delicate because
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
because it is noisy; it is interpreted through the design that produced it. Dataset
sampling and item independence is one place where that habit becomes concrete.

### 3.3 Prompt templates and few-shot policy

Prompt templates and few-shot policy is part of the canonical scope of capability
benchmarks. In this chapter, the object under study is not merely a dataset or a model,
but the full benchmark protocol: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{\mu}_{m,t} = \frac{1}{n}\sum_{i=1}^n s_m(z_i).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For prompt templates and
few-shot policy, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in prompt templates and few-shot policy |
| Scoring rule | Exact formula for s_m(z_i) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report prompt templates and few-shot policy with item count, prompt protocol, grader
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

Worked evaluation pattern for prompt templates and few-shot policy:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, prompt templates and few-shot policy is especially delicate because the
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
because it is noisy; it is interpreted through the design that produced it. Prompt
templates and few-shot policy is one place where that habit becomes concrete.

### 3.4 Grading functions and rubrics

Grading functions and rubrics is part of the canonical scope of capability benchmarks.
In this chapter, the object under study is not merely a dataset or a model, but the full
benchmark protocol: the items, prompts, outputs, graders, uncertainty statements, and
decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{\mu}_{m,t} = \frac{1}{n}\sum_{i=1}^n s_m(z_i).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For grading functions and
rubrics, those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in grading functions and rubrics |
| Scoring rule | Exact formula for s_m(z_i) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report grading functions and rubrics with item count, prompt protocol, grader version,
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

Worked evaluation pattern for grading functions and rubrics:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, grading functions and rubrics is especially delicate because the same
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
because it is noisy; it is interpreted through the design that produced it. Grading
functions and rubrics is one place where that habit becomes concrete.

### 3.5 Contamination flags and eval provenance

Contamination flags and eval provenance is part of the canonical scope of capability
benchmarks. In this chapter, the object under study is not merely a dataset or a model,
but the full benchmark protocol: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{\mu}_{m,t} = \frac{1}{n}\sum_{i=1}^n s_m(z_i).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For contamination flags and
eval provenance, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in contamination flags and eval provenance |
| Scoring rule | Exact formula for s_m(z_i) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report contamination flags and eval provenance with item count, prompt protocol,
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

Worked evaluation pattern for contamination flags and eval provenance:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, contamination flags and eval provenance is especially delicate because
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
because it is noisy; it is interpreted through the design that produced it.
Contamination flags and eval provenance is one place where that habit becomes concrete.

## 4. Core Metrics

Core Metrics is the part of capability benchmarks that turns the approved TOC into a
concrete learning path. The subsections below keep the focus on Chapter 17's canonical
job: measurement, reliability, uncertainty, and decision support for AI systems.

### 4.1 Accuracy and exact match

Accuracy and exact match is part of the canonical scope of capability benchmarks. In
this chapter, the object under study is not merely a dataset or a model, but the full
benchmark protocol: the items, prompts, outputs, graders, uncertainty statements, and
decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{\mu}_{m,t} = \frac{1}{n}\sum_{i=1}^n s_m(z_i).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For accuracy and exact
match, those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in accuracy and exact match |
| Scoring rule | Exact formula for s_m(z_i) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report accuracy and exact match with item count, prompt protocol, grader version, and
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

Worked evaluation pattern for accuracy and exact match:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, accuracy and exact match is especially delicate because the same model
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
because it is noisy; it is interpreted through the design that produced it. Accuracy and
exact match is one place where that habit becomes concrete.

### 4.2 Precision, recall, and F1

Precision, recall, and F1 is part of the canonical scope of capability benchmarks. In
this chapter, the object under study is not merely a dataset or a model, but the full
benchmark protocol: the items, prompts, outputs, graders, uncertainty statements, and
decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{\mu}_{m,t} = \frac{1}{n}\sum_{i=1}^n s_m(z_i).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For precision, recall, and
f1, those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in precision, recall, and f1 |
| Scoring rule | Exact formula for s_m(z_i) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report precision, recall, and f1 with item count, prompt protocol, grader version, and
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

Worked evaluation pattern for precision, recall, and f1:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, precision, recall, and f1 is especially delicate because the same model
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
because it is noisy; it is interpreted through the design that produced it. Precision,
recall, and F1 is one place where that habit becomes concrete.

### 4.3 Pass at k for code generation

Pass at k for code generation is part of the canonical scope of capability benchmarks.
In this chapter, the object under study is not merely a dataset or a model, but the full
benchmark protocol: the items, prompts, outputs, graders, uncertainty statements, and
decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{\mu}_{m,t} = \frac{1}{n}\sum_{i=1}^n s_m(z_i).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For pass at k for code
generation, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in pass at k for code generation |
| Scoring rule | Exact formula for s_m(z_i) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report pass at k for code generation with item count, prompt protocol, grader version,
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

Worked evaluation pattern for pass at k for code generation:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, pass at k for code generation is especially delicate because the same
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
because it is noisy; it is interpreted through the design that produced it. Pass at k
for code generation is one place where that habit becomes concrete.

### 4.4 Log-probability and perplexity

Log-probability and perplexity is part of the canonical scope of capability benchmarks.
In this chapter, the object under study is not merely a dataset or a model, but the full
benchmark protocol: the items, prompts, outputs, graders, uncertainty statements, and
decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{\mu}_{m,t} = \frac{1}{n}\sum_{i=1}^n s_m(z_i).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For log-probability and
perplexity, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in log-probability and perplexity |
| Scoring rule | Exact formula for s_m(z_i) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report log-probability and perplexity with item count, prompt protocol, grader
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

Worked evaluation pattern for log-probability and perplexity:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, log-probability and perplexity is especially delicate because the same
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
because it is noisy; it is interpreted through the design that produced it. Log-
probability and perplexity is one place where that habit becomes concrete.

### 4.5 Pairwise preference and judge agreement

Pairwise preference and judge agreement is part of the canonical scope of capability
benchmarks. In this chapter, the object under study is not merely a dataset or a model,
but the full benchmark protocol: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{\mu}_{m,t} = \frac{1}{n}\sum_{i=1}^n s_m(z_i).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For pairwise preference and
judge agreement, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in pairwise preference and judge agreement |
| Scoring rule | Exact formula for s_m(z_i) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report pairwise preference and judge agreement with item count, prompt protocol,
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

Worked evaluation pattern for pairwise preference and judge agreement:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, pairwise preference and judge agreement is especially delicate because
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
because it is noisy; it is interpreted through the design that produced it. Pairwise
preference and judge agreement is one place where that habit becomes concrete.

## 5. Statistical Reliability

Statistical Reliability is the part of capability benchmarks that turns the approved TOC
into a concrete learning path. The subsections below keep the focus on Chapter 17's
canonical job: measurement, reliability, uncertainty, and decision support for AI
systems.

### 5.1 Bootstrap intervals

Bootstrap intervals is part of the canonical scope of capability benchmarks. In this
chapter, the object under study is not merely a dataset or a model, but the full
benchmark protocol: the items, prompts, outputs, graders, uncertainty statements, and
decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{\mu}_{m,t} = \frac{1}{n}\sum_{i=1}^n s_m(z_i).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For bootstrap intervals,
those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in bootstrap intervals |
| Scoring rule | Exact formula for s_m(z_i) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report bootstrap intervals with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for bootstrap intervals:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, bootstrap intervals is especially delicate because the same model can be
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
because it is noisy; it is interpreted through the design that produced it. Bootstrap
intervals is one place where that habit becomes concrete.

### 5.2 Paired model comparisons

Paired model comparisons is part of the canonical scope of capability benchmarks. In
this chapter, the object under study is not merely a dataset or a model, but the full
benchmark protocol: the items, prompts, outputs, graders, uncertainty statements, and
decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{\mu}_{m,t} = \frac{1}{n}\sum_{i=1}^n s_m(z_i).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For paired model
comparisons, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in paired model comparisons |
| Scoring rule | Exact formula for s_m(z_i) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report paired model comparisons with item count, prompt protocol, grader version, and
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

Worked evaluation pattern for paired model comparisons:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, paired model comparisons is especially delicate because the same model
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
because it is noisy; it is interpreted through the design that produced it. Paired model
comparisons is one place where that habit becomes concrete.

### 5.3 Leaderboard uncertainty

Leaderboard uncertainty is part of the canonical scope of capability benchmarks. In this
chapter, the object under study is not merely a dataset or a model, but the full
benchmark protocol: the items, prompts, outputs, graders, uncertainty statements, and
decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{\mu}_{m,t} = \frac{1}{n}\sum_{i=1}^n s_m(z_i).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For leaderboard uncertainty,
those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in leaderboard uncertainty |
| Scoring rule | Exact formula for s_m(z_i) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report leaderboard uncertainty with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for leaderboard uncertainty:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, leaderboard uncertainty is especially delicate because the same model
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
because it is noisy; it is interpreted through the design that produced it. Leaderboard
uncertainty is one place where that habit becomes concrete.

### 5.4 Multiple comparisons

Multiple comparisons is part of the canonical scope of capability benchmarks. In this
chapter, the object under study is not merely a dataset or a model, but the full
benchmark protocol: the items, prompts, outputs, graders, uncertainty statements, and
decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{\mu}_{m,t} = \frac{1}{n}\sum_{i=1}^n s_m(z_i).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For multiple comparisons,
those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in multiple comparisons |
| Scoring rule | Exact formula for s_m(z_i) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report multiple comparisons with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for multiple comparisons:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, multiple comparisons is especially delicate because the same model can
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
comparisons is one place where that habit becomes concrete.

### 5.5 Benchmark power and sample size

Benchmark power and sample size is part of the canonical scope of capability benchmarks.
In this chapter, the object under study is not merely a dataset or a model, but the full
benchmark protocol: the items, prompts, outputs, graders, uncertainty statements, and
decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{\mu}_{m,t} = \frac{1}{n}\sum_{i=1}^n s_m(z_i).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For benchmark power and
sample size, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in benchmark power and sample size |
| Scoring rule | Exact formula for s_m(z_i) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report benchmark power and sample size with item count, prompt protocol, grader
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

Worked evaluation pattern for benchmark power and sample size:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, benchmark power and sample size is especially delicate because the same
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
because it is noisy; it is interpreted through the design that produced it. Benchmark
power and sample size is one place where that habit becomes concrete.

## 6. LLM Benchmark Families

LLM Benchmark Families is the part of capability benchmarks that turns the approved TOC
into a concrete learning path. The subsections below keep the focus on Chapter 17's
canonical job: measurement, reliability, uncertainty, and decision support for AI
systems.

### 6.1 MMLU and multitask knowledge

MMLU and multitask knowledge is part of the canonical scope of capability benchmarks. In
this chapter, the object under study is not merely a dataset or a model, but the full
benchmark protocol: the items, prompts, outputs, graders, uncertainty statements, and
decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{\mu}_{m,t} = \frac{1}{n}\sum_{i=1}^n s_m(z_i).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For mmlu and multitask
knowledge, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in mmlu and multitask knowledge |
| Scoring rule | Exact formula for s_m(z_i) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report mmlu and multitask knowledge with item count, prompt protocol, grader version,
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

Worked evaluation pattern for mmlu and multitask knowledge:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, mmlu and multitask knowledge is especially delicate because the same
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
because it is noisy; it is interpreted through the design that produced it. MMLU and
multitask knowledge is one place where that habit becomes concrete.

### 6.2 HumanEval and functional correctness

HumanEval and functional correctness is part of the canonical scope of capability
benchmarks. In this chapter, the object under study is not merely a dataset or a model,
but the full benchmark protocol: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{\mu}_{m,t} = \frac{1}{n}\sum_{i=1}^n s_m(z_i).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For humaneval and functional
correctness, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in humaneval and functional correctness |
| Scoring rule | Exact formula for s_m(z_i) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report humaneval and functional correctness with item count, prompt protocol, grader
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

Worked evaluation pattern for humaneval and functional correctness:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, humaneval and functional correctness is especially delicate because the
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
because it is noisy; it is interpreted through the design that produced it. HumanEval
and functional correctness is one place where that habit becomes concrete.

### 6.3 BIG-bench and capability extrapolation

BIG-bench and capability extrapolation is part of the canonical scope of capability
benchmarks. In this chapter, the object under study is not merely a dataset or a model,
but the full benchmark protocol: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{\mu}_{m,t} = \frac{1}{n}\sum_{i=1}^n s_m(z_i).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For big-bench and capability
extrapolation, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in big-bench and capability extrapolation |
| Scoring rule | Exact formula for s_m(z_i) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report big-bench and capability extrapolation with item count, prompt protocol, grader
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

Worked evaluation pattern for big-bench and capability extrapolation:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, big-bench and capability extrapolation is especially delicate because
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
because it is noisy; it is interpreted through the design that produced it. BIG-bench
and capability extrapolation is one place where that habit becomes concrete.

### 6.4 HELM and multi-metric transparency

HELM and multi-metric transparency is part of the canonical scope of capability
benchmarks. In this chapter, the object under study is not merely a dataset or a model,
but the full benchmark protocol: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{\mu}_{m,t} = \frac{1}{n}\sum_{i=1}^n s_m(z_i).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For helm and multi-metric
transparency, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in helm and multi-metric transparency |
| Scoring rule | Exact formula for s_m(z_i) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report helm and multi-metric transparency with item count, prompt protocol, grader
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

Worked evaluation pattern for helm and multi-metric transparency:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, helm and multi-metric transparency is especially delicate because the
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
because it is noisy; it is interpreted through the design that produced it. HELM and
multi-metric transparency is one place where that habit becomes concrete.

### 6.5 MT-Bench, Chatbot Arena, and LLM-as-judge

MT-Bench, Chatbot Arena, and LLM-as-judge is part of the canonical scope of capability
benchmarks. In this chapter, the object under study is not merely a dataset or a model,
but the full benchmark protocol: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\hat{\mu}_{m,t} = \frac{1}{n}\sum_{i=1}^n s_m(z_i).
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For mt-bench, chatbot arena,
and llm-as-judge, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in mt-bench, chatbot arena, and llm-as-judge |
| Scoring rule | Exact formula for s_m(z_i) | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report mt-bench, chatbot arena, and llm-as-judge with item count, prompt protocol,
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

Worked evaluation pattern for mt-bench, chatbot arena, and llm-as-judge:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, mt-bench, chatbot arena, and llm-as-judge is especially delicate because
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
because it is noisy; it is interpreted through the design that produced it. MT-Bench,
Chatbot Arena, and LLM-as-judge is one place where that habit becomes concrete.

## 7. Common Mistakes

| # | Mistake | Why It Is Wrong | Fix |
| --- | --- | --- | --- |
| 1 | Treating a point estimate as exact | Every finite evaluation has sampling error in capability benchmarks. | Report uncertainty with the point estimate. |
| 2 | Changing prompts between models | The protocol changed with the treatment in capability benchmarks. | Lock prompt, decoding, and grader before comparison. |
| 3 | Ignoring invalid outputs | Missingness can be correlated with model quality in capability benchmarks. | Track invalid, timeout, refusal, and parse-failure rates. |
| 4 | Overfitting to a public leaderboard | Repeated testing leaks information from the benchmark in capability benchmarks. | Use private holdouts and regression suites. |
| 5 | Averaging incomparable metrics | Different scales do not have shared units in capability benchmarks. | Normalize by a stated decision rule or report separately. |
| 6 | Forgetting paired structure | Two models often answer the same items in capability benchmarks. | Use paired bootstrap or paired tests where possible. |
| 7 | Reporting only aggregate performance | Subgroup failures can be hidden in capability benchmarks. | Add slice and tail-risk views. |
| 8 | Trusting model judges blindly | LLM judges have position, verbosity, and self-preference biases in capability benchmarks. | Calibrate judges against human labels. |
| 9 | Peeking during online experiments | Optional stopping inflates false positives in capability benchmarks. | Use fixed horizons or sequential-valid methods. |
| 10 | Conflating evaluation with monitoring | Chapter 17 measures controlled evidence; production monitoring is ongoing operations in capability benchmarks. | Hand off drift dashboards to Chapter 19 concepts. |

## 8. Exercises

1. (*) Benchmarks as noisy estimators.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

2. (*) Capability versus observed score.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

3. (*) Metric pluralism for LLM systems.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

4. (**) Benchmark lifecycle and saturation.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

5. (**) What benchmark scores can and cannot certify.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

6. (**) Model and system under test.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

7. (***) Task, item, and evaluation sample.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

8. (***) Prompt protocol and decoding policy.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

9. (***) Scorer, metric, and aggregate estimate.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

10. (***) Confidence interval and leaderboard rank.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

## 9. Why This Matters for AI

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

## 10. Conceptual Bridge

This section sits after the training-data pipeline because evaluation depends on clean
holdouts, contamination audits, and well-documented data provenance. It does not repeat
those pipeline mechanics; it consumes their outputs as the basis for credible
measurement.

It also sits before alignment and production chapters. Alignment asks how to shape model
behavior with supervised data, preferences, policies, and feedback. Production MLOps
asks how deployed systems are observed and maintained over time. Capability Benchmarks
supplies the measurement discipline both chapters need.

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

- [MMLU: Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300)
- [HELM: Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110)
- [HumanEval: Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374)
- [BIG-bench](https://arxiv.org/abs/2206.04615)
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)
- [OpenAI Evals](https://github.com/openai/evals)
- [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [Concentration Inequalities](../../06-Probability-Theory/05-Concentration-Inequalities/notes.md)
- [Estimation Theory](../../07-Statistics/02-Estimation-Theory/notes.md)
- [Hypothesis Testing](../../07-Statistics/03-Hypothesis-Testing/notes.md)

