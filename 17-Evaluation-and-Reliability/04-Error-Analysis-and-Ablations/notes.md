[Back to Curriculum](../../README.md) | [Previous: Robustness and Distribution Shift](../03-Robustness-and-Distribution-Shift/notes.md) | [Next: Online Experimentation and AB Testing](../05-Online-Experimentation-and-AB-Testing/notes.md)

---

# Error Analysis and Ablations

> _"A failed example is not an embarrassment; it is a coordinate system for improvement."_

## Overview

Error analysis turns aggregate scores into failure structure; ablations test which
component actually caused an improvement.

The chapter treats evaluation as a mathematical object: a controlled protocol that maps
model behavior into evidence with uncertainty. A result is not just a number; it is an
estimate produced by a task distribution, a scoring rule, and an aggregation rule.

This section is written in LaTeX Markdown. Inline mathematics uses `$...$`, and display
equations use `$$...$$`. The emphasis is practical but rigorous: every metric should be
linked to the statistical assumptions that make it meaningful.

## Prerequisites

- [Descriptive Statistics](../../07-Statistics/01-Descriptive-Statistics/notes.md)
- [Hypothesis Testing](../../07-Statistics/03-Hypothesis-Testing/notes.md)
- [Regression Analysis](../../07-Statistics/06-Regression-Analysis/notes.md)
- [Capability Benchmarks](../01-Capability-Benchmarks/notes.md)
- [Robustness and Distribution Shift](../03-Robustness-and-Distribution-Shift/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Executable demonstrations for error analysis and ablations |
| [exercises.ipynb](exercises.ipynb) | Graded practice for error analysis and ablations |

## Learning Objectives

After completing this section, you will be able to:

- Define the core objects used in error analysis and ablations
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
  - [1.1 Aggregate metrics hide failure modes](#11-aggregate-metrics-hide-failure-modes)
  - [1.2 Failures as structured data](#12-failures-as-structured-data)
  - [1.3 Ablations as causal probes](#13-ablations-as-causal-probes)
  - [1.4 Debugging without benchmark overfitting](#14-debugging-without-benchmark-overfitting)
  - [1.5 From one bug to a regression suite](#15-from-one-bug-to-a-regression-suite)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Error set](#21-error-set)
  - [2.2 Confusion matrix](#22-confusion-matrix)
  - [2.3 Slice and subgroup](#23-slice-and-subgroup)
  - [2.4 Counterfactual example](#24-counterfactual-example)
  - [2.5 Ablation effect and interaction](#25-ablation-effect-and-interaction)
- [3. Error Taxonomies](#3-error-taxonomies)
  - [3.1 False positives and false negatives](#31-false-positives-and-false-negatives)
  - [3.2 Hallucination and unsupported claims](#32-hallucination-and-unsupported-claims)
  - [3.3 Format and instruction errors](#33-format-and-instruction-errors)
  - [3.4 Reasoning errors](#34-reasoning-errors)
  - [3.5 Tool and retrieval failures](#35-tool-and-retrieval-failures)
- [4. Slice Analysis](#4-slice-analysis)
  - [4.1 Stratified metrics](#41-stratified-metrics)
  - [4.2 Subgroup confidence intervals](#42-subgroup-confidence-intervals)
  - [4.3 Multiple-testing control for slices](#43-multiple-testing-control-for-slices)
  - [4.4 Prioritizing failures](#44-prioritizing-failures)
  - [4.5 Dashboard and report design](#45-dashboard-and-report-design)
- [5. Ablation Design](#5-ablation-design)
  - [5.1 Model ablations](#51-model-ablations)
  - [5.2 Data ablations](#52-data-ablations)
  - [5.3 Prompt and decoding ablations](#53-prompt-and-decoding-ablations)
  - [5.4 Retrieval and tool ablations](#54-retrieval-and-tool-ablations)
  - [5.5 Metric ablations](#55-metric-ablations)
- [6. Factorial Experiments](#6-factorial-experiments)
  - [6.1 Main effects](#61-main-effects)
  - [6.2 Interactions](#62-interactions)
  - [6.3 Response surfaces](#63-response-surfaces)
  - [6.4 ANOVA preview](#64-anova-preview)
  - [6.5 Attribution limits](#65-attribution-limits)
- [7. Regression-Test Workflow](#7-regression-test-workflow)
  - [7.1 Converting failures into eval cases](#71-converting-failures-into-eval-cases)
  - [7.2 Versioning error suites](#72-versioning-error-suites)
  - [7.3 Avoiding overfit fixes](#73-avoiding-overfit-fixes)
  - [7.4 Budgeting regression tests](#74-budgeting-regression-tests)
  - [7.5 Closing the debug loop](#75-closing-the-debug-loop)
- [8. Common Mistakes](#8-common-mistakes)
- [9. Exercises](#9-exercises)
- [10. Why This Matters for AI](#10-why-this-matters-for-ai)
- [11. Conceptual Bridge](#11-conceptual-bridge)
- [References](#references)

---

## 1. Intuition

Intuition is the part of error analysis and ablations that turns the approved TOC into a
concrete learning path. The subsections below keep the focus on Chapter 17's canonical
job: measurement, reliability, uncertainty, and decision support for AI systems.

### 1.1 Aggregate metrics hide failure modes

Aggregate metrics hide failure modes is part of the canonical scope of error analysis
and ablations. In this chapter, the object under study is not merely a dataset or a
model, but the full failure slice and causal comparison: the items, prompts, outputs,
graders, uncertainty statements, and decision rules that turn model behavior into
evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\Delta_{\mathrm{ablate}} = \frac{1}{n}\sum_{i=1}^n \mathbb{1}\{e_i \in E\}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For aggregate metrics hide
failure modes, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in aggregate metrics hide failure modes |
| Scoring rule | Exact formula for \mathbb{1}\{e_i \in E\} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report aggregate metrics hide failure modes with item count, prompt protocol, grader
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

Worked evaluation pattern for aggregate metrics hide failure modes:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, aggregate metrics hide failure modes is especially delicate because the
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
because it is noisy; it is interpreted through the design that produced it. Aggregate
metrics hide failure modes is one place where that habit becomes concrete.

### 1.2 Failures as structured data

Failures as structured data is part of the canonical scope of error analysis and
ablations. In this chapter, the object under study is not merely a dataset or a model,
but the full failure slice and causal comparison: the items, prompts, outputs, graders,
uncertainty statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\Delta_{\mathrm{ablate}} = \frac{1}{n}\sum_{i=1}^n \mathbb{1}\{e_i \in E\}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For failures as structured
data, those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in failures as structured data |
| Scoring rule | Exact formula for \mathbb{1}\{e_i \in E\} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report failures as structured data with item count, prompt protocol, grader version,
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

Worked evaluation pattern for failures as structured data:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, failures as structured data is especially delicate because the same
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
because it is noisy; it is interpreted through the design that produced it. Failures as
structured data is one place where that habit becomes concrete.

### 1.3 Ablations as causal probes

Ablations as causal probes is part of the canonical scope of error analysis and
ablations. In this chapter, the object under study is not merely a dataset or a model,
but the full failure slice and causal comparison: the items, prompts, outputs, graders,
uncertainty statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\Delta_{\mathrm{ablate}} = \frac{1}{n}\sum_{i=1}^n \mathbb{1}\{e_i \in E\}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For ablations as causal
probes, those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in ablations as causal probes |
| Scoring rule | Exact formula for \mathbb{1}\{e_i \in E\} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report ablations as causal probes with item count, prompt protocol, grader version,
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

Worked evaluation pattern for ablations as causal probes:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, ablations as causal probes is especially delicate because the same model
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
because it is noisy; it is interpreted through the design that produced it. Ablations as
causal probes is one place where that habit becomes concrete.

### 1.4 Debugging without benchmark overfitting

Debugging without benchmark overfitting is part of the canonical scope of error analysis
and ablations. In this chapter, the object under study is not merely a dataset or a
model, but the full failure slice and causal comparison: the items, prompts, outputs,
graders, uncertainty statements, and decision rules that turn model behavior into
evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\Delta_{\mathrm{ablate}} = \frac{1}{n}\sum_{i=1}^n \mathbb{1}\{e_i \in E\}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For debugging without
benchmark overfitting, those choices determine whether the reported number is evidence
or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in debugging without benchmark overfitting |
| Scoring rule | Exact formula for \mathbb{1}\{e_i \in E\} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report debugging without benchmark overfitting with item count, prompt protocol,
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

Worked evaluation pattern for debugging without benchmark overfitting:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, debugging without benchmark overfitting is especially delicate because
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
because it is noisy; it is interpreted through the design that produced it. Debugging
without benchmark overfitting is one place where that habit becomes concrete.

### 1.5 From one bug to a regression suite

From one bug to a regression suite is part of the canonical scope of error analysis and
ablations. In this chapter, the object under study is not merely a dataset or a model,
but the full failure slice and causal comparison: the items, prompts, outputs, graders,
uncertainty statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\Delta_{\mathrm{ablate}} = \frac{1}{n}\sum_{i=1}^n \mathbb{1}\{e_i \in E\}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For from one bug to a
regression suite, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in from one bug to a regression suite |
| Scoring rule | Exact formula for \mathbb{1}\{e_i \in E\} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report from one bug to a regression suite with item count, prompt protocol, grader
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

Worked evaluation pattern for from one bug to a regression suite:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, from one bug to a regression suite is especially delicate because the
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
because it is noisy; it is interpreted through the design that produced it. From one bug
to a regression suite is one place where that habit becomes concrete.

## 2. Formal Definitions

Formal Definitions is the part of error analysis and ablations that turns the approved
TOC into a concrete learning path. The subsections below keep the focus on Chapter 17's
canonical job: measurement, reliability, uncertainty, and decision support for AI
systems.

### 2.1 Error set

Error set is part of the canonical scope of error analysis and ablations. In this
chapter, the object under study is not merely a dataset or a model, but the full failure
slice and causal comparison: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\Delta_{\mathrm{ablate}} = \frac{1}{n}\sum_{i=1}^n \mathbb{1}\{e_i \in E\}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For error set, those choices
determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in error set |
| Scoring rule | Exact formula for \mathbb{1}\{e_i \in E\} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report error set with item count, prompt protocol, grader version, and a confidence
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

Worked evaluation pattern for error set:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, error set is especially delicate because the same model can be used with
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
because it is noisy; it is interpreted through the design that produced it. Error set is
one place where that habit becomes concrete.

### 2.2 Confusion matrix

Confusion matrix is part of the canonical scope of error analysis and ablations. In this
chapter, the object under study is not merely a dataset or a model, but the full failure
slice and causal comparison: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\Delta_{\mathrm{ablate}} = \frac{1}{n}\sum_{i=1}^n \mathbb{1}\{e_i \in E\}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For confusion matrix, those
choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in confusion matrix |
| Scoring rule | Exact formula for \mathbb{1}\{e_i \in E\} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report confusion matrix with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for confusion matrix:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, confusion matrix is especially delicate because the same model can be
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
because it is noisy; it is interpreted through the design that produced it. Confusion
matrix is one place where that habit becomes concrete.

### 2.3 Slice and subgroup

Slice and subgroup is part of the canonical scope of error analysis and ablations. In
this chapter, the object under study is not merely a dataset or a model, but the full
failure slice and causal comparison: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\Delta_{\mathrm{ablate}} = \frac{1}{n}\sum_{i=1}^n \mathbb{1}\{e_i \in E\}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For slice and subgroup,
those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in slice and subgroup |
| Scoring rule | Exact formula for \mathbb{1}\{e_i \in E\} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report slice and subgroup with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for slice and subgroup:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, slice and subgroup is especially delicate because the same model can be
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
because it is noisy; it is interpreted through the design that produced it. Slice and
subgroup is one place where that habit becomes concrete.

### 2.4 Counterfactual example

Counterfactual example is part of the canonical scope of error analysis and ablations.
In this chapter, the object under study is not merely a dataset or a model, but the full
failure slice and causal comparison: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\Delta_{\mathrm{ablate}} = \frac{1}{n}\sum_{i=1}^n \mathbb{1}\{e_i \in E\}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For counterfactual example,
those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in counterfactual example |
| Scoring rule | Exact formula for \mathbb{1}\{e_i \in E\} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report counterfactual example with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for counterfactual example:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, counterfactual example is especially delicate because the same model can
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
Counterfactual example is one place where that habit becomes concrete.

### 2.5 Ablation effect and interaction

Ablation effect and interaction is part of the canonical scope of error analysis and
ablations. In this chapter, the object under study is not merely a dataset or a model,
but the full failure slice and causal comparison: the items, prompts, outputs, graders,
uncertainty statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\Delta_{\mathrm{ablate}} = \frac{1}{n}\sum_{i=1}^n \mathbb{1}\{e_i \in E\}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For ablation effect and
interaction, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in ablation effect and interaction |
| Scoring rule | Exact formula for \mathbb{1}\{e_i \in E\} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report ablation effect and interaction with item count, prompt protocol, grader
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

Worked evaluation pattern for ablation effect and interaction:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, ablation effect and interaction is especially delicate because the same
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
because it is noisy; it is interpreted through the design that produced it. Ablation
effect and interaction is one place where that habit becomes concrete.

## 3. Error Taxonomies

Error Taxonomies is the part of error analysis and ablations that turns the approved TOC
into a concrete learning path. The subsections below keep the focus on Chapter 17's
canonical job: measurement, reliability, uncertainty, and decision support for AI
systems.

### 3.1 False positives and false negatives

False positives and false negatives is part of the canonical scope of error analysis and
ablations. In this chapter, the object under study is not merely a dataset or a model,
but the full failure slice and causal comparison: the items, prompts, outputs, graders,
uncertainty statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\Delta_{\mathrm{ablate}} = \frac{1}{n}\sum_{i=1}^n \mathbb{1}\{e_i \in E\}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For false positives and
false negatives, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in false positives and false negatives |
| Scoring rule | Exact formula for \mathbb{1}\{e_i \in E\} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report false positives and false negatives with item count, prompt protocol, grader
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

Worked evaluation pattern for false positives and false negatives:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, false positives and false negatives is especially delicate because the
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
because it is noisy; it is interpreted through the design that produced it. False
positives and false negatives is one place where that habit becomes concrete.

### 3.2 Hallucination and unsupported claims

Hallucination and unsupported claims is part of the canonical scope of error analysis
and ablations. In this chapter, the object under study is not merely a dataset or a
model, but the full failure slice and causal comparison: the items, prompts, outputs,
graders, uncertainty statements, and decision rules that turn model behavior into
evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\Delta_{\mathrm{ablate}} = \frac{1}{n}\sum_{i=1}^n \mathbb{1}\{e_i \in E\}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For hallucination and
unsupported claims, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in hallucination and unsupported claims |
| Scoring rule | Exact formula for \mathbb{1}\{e_i \in E\} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report hallucination and unsupported claims with item count, prompt protocol, grader
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

Worked evaluation pattern for hallucination and unsupported claims:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, hallucination and unsupported claims is especially delicate because the
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
Hallucination and unsupported claims is one place where that habit becomes concrete.

### 3.3 Format and instruction errors

Format and instruction errors is part of the canonical scope of error analysis and
ablations. In this chapter, the object under study is not merely a dataset or a model,
but the full failure slice and causal comparison: the items, prompts, outputs, graders,
uncertainty statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\Delta_{\mathrm{ablate}} = \frac{1}{n}\sum_{i=1}^n \mathbb{1}\{e_i \in E\}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For format and instruction
errors, those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in format and instruction errors |
| Scoring rule | Exact formula for \mathbb{1}\{e_i \in E\} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report format and instruction errors with item count, prompt protocol, grader version,
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

Worked evaluation pattern for format and instruction errors:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, format and instruction errors is especially delicate because the same
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
because it is noisy; it is interpreted through the design that produced it. Format and
instruction errors is one place where that habit becomes concrete.

### 3.4 Reasoning errors

Reasoning errors is part of the canonical scope of error analysis and ablations. In this
chapter, the object under study is not merely a dataset or a model, but the full failure
slice and causal comparison: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\Delta_{\mathrm{ablate}} = \frac{1}{n}\sum_{i=1}^n \mathbb{1}\{e_i \in E\}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For reasoning errors, those
choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in reasoning errors |
| Scoring rule | Exact formula for \mathbb{1}\{e_i \in E\} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report reasoning errors with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for reasoning errors:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, reasoning errors is especially delicate because the same model can be
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
because it is noisy; it is interpreted through the design that produced it. Reasoning
errors is one place where that habit becomes concrete.

### 3.5 Tool and retrieval failures

Tool and retrieval failures is part of the canonical scope of error analysis and
ablations. In this chapter, the object under study is not merely a dataset or a model,
but the full failure slice and causal comparison: the items, prompts, outputs, graders,
uncertainty statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\Delta_{\mathrm{ablate}} = \frac{1}{n}\sum_{i=1}^n \mathbb{1}\{e_i \in E\}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For tool and retrieval
failures, those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in tool and retrieval failures |
| Scoring rule | Exact formula for \mathbb{1}\{e_i \in E\} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report tool and retrieval failures with item count, prompt protocol, grader version,
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

Worked evaluation pattern for tool and retrieval failures:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, tool and retrieval failures is especially delicate because the same
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
because it is noisy; it is interpreted through the design that produced it. Tool and
retrieval failures is one place where that habit becomes concrete.

## 4. Slice Analysis

Slice Analysis is the part of error analysis and ablations that turns the approved TOC
into a concrete learning path. The subsections below keep the focus on Chapter 17's
canonical job: measurement, reliability, uncertainty, and decision support for AI
systems.

### 4.1 Stratified metrics

Stratified metrics is part of the canonical scope of error analysis and ablations. In
this chapter, the object under study is not merely a dataset or a model, but the full
failure slice and causal comparison: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\Delta_{\mathrm{ablate}} = \frac{1}{n}\sum_{i=1}^n \mathbb{1}\{e_i \in E\}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For stratified metrics,
those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in stratified metrics |
| Scoring rule | Exact formula for \mathbb{1}\{e_i \in E\} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report stratified metrics with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for stratified metrics:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, stratified metrics is especially delicate because the same model can be
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
because it is noisy; it is interpreted through the design that produced it. Stratified
metrics is one place where that habit becomes concrete.

### 4.2 Subgroup confidence intervals

Subgroup confidence intervals is part of the canonical scope of error analysis and
ablations. In this chapter, the object under study is not merely a dataset or a model,
but the full failure slice and causal comparison: the items, prompts, outputs, graders,
uncertainty statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\Delta_{\mathrm{ablate}} = \frac{1}{n}\sum_{i=1}^n \mathbb{1}\{e_i \in E\}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For subgroup confidence
intervals, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in subgroup confidence intervals |
| Scoring rule | Exact formula for \mathbb{1}\{e_i \in E\} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report subgroup confidence intervals with item count, prompt protocol, grader version,
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

Worked evaluation pattern for subgroup confidence intervals:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, subgroup confidence intervals is especially delicate because the same
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
because it is noisy; it is interpreted through the design that produced it. Subgroup
confidence intervals is one place where that habit becomes concrete.

### 4.3 Multiple-testing control for slices

Multiple-testing control for slices is part of the canonical scope of error analysis and
ablations. In this chapter, the object under study is not merely a dataset or a model,
but the full failure slice and causal comparison: the items, prompts, outputs, graders,
uncertainty statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\Delta_{\mathrm{ablate}} = \frac{1}{n}\sum_{i=1}^n \mathbb{1}\{e_i \in E\}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For multiple-testing control
for slices, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in multiple-testing control for slices |
| Scoring rule | Exact formula for \mathbb{1}\{e_i \in E\} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report multiple-testing control for slices with item count, prompt protocol, grader
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

Worked evaluation pattern for multiple-testing control for slices:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, multiple-testing control for slices is especially delicate because the
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
because it is noisy; it is interpreted through the design that produced it. Multiple-
testing control for slices is one place where that habit becomes concrete.

### 4.4 Prioritizing failures

Prioritizing failures is part of the canonical scope of error analysis and ablations. In
this chapter, the object under study is not merely a dataset or a model, but the full
failure slice and causal comparison: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\Delta_{\mathrm{ablate}} = \frac{1}{n}\sum_{i=1}^n \mathbb{1}\{e_i \in E\}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For prioritizing failures,
those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in prioritizing failures |
| Scoring rule | Exact formula for \mathbb{1}\{e_i \in E\} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report prioritizing failures with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for prioritizing failures:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, prioritizing failures is especially delicate because the same model can
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
because it is noisy; it is interpreted through the design that produced it. Prioritizing
failures is one place where that habit becomes concrete.

### 4.5 Dashboard and report design

Dashboard and report design is part of the canonical scope of error analysis and
ablations. In this chapter, the object under study is not merely a dataset or a model,
but the full failure slice and causal comparison: the items, prompts, outputs, graders,
uncertainty statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\Delta_{\mathrm{ablate}} = \frac{1}{n}\sum_{i=1}^n \mathbb{1}\{e_i \in E\}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For dashboard and report
design, those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in dashboard and report design |
| Scoring rule | Exact formula for \mathbb{1}\{e_i \in E\} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report dashboard and report design with item count, prompt protocol, grader version,
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

Worked evaluation pattern for dashboard and report design:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, dashboard and report design is especially delicate because the same
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
because it is noisy; it is interpreted through the design that produced it. Dashboard
and report design is one place where that habit becomes concrete.

## 5. Ablation Design

Ablation Design is the part of error analysis and ablations that turns the approved TOC
into a concrete learning path. The subsections below keep the focus on Chapter 17's
canonical job: measurement, reliability, uncertainty, and decision support for AI
systems.

### 5.1 Model ablations

Model ablations is part of the canonical scope of error analysis and ablations. In this
chapter, the object under study is not merely a dataset or a model, but the full failure
slice and causal comparison: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\Delta_{\mathrm{ablate}} = \frac{1}{n}\sum_{i=1}^n \mathbb{1}\{e_i \in E\}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For model ablations, those
choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in model ablations |
| Scoring rule | Exact formula for \mathbb{1}\{e_i \in E\} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report model ablations with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for model ablations:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, model ablations is especially delicate because the same model can be
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
because it is noisy; it is interpreted through the design that produced it. Model
ablations is one place where that habit becomes concrete.

### 5.2 Data ablations

Data ablations is part of the canonical scope of error analysis and ablations. In this
chapter, the object under study is not merely a dataset or a model, but the full failure
slice and causal comparison: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\Delta_{\mathrm{ablate}} = \frac{1}{n}\sum_{i=1}^n \mathbb{1}\{e_i \in E\}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For data ablations, those
choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in data ablations |
| Scoring rule | Exact formula for \mathbb{1}\{e_i \in E\} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report data ablations with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for data ablations:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, data ablations is especially delicate because the same model can be used
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
because it is noisy; it is interpreted through the design that produced it. Data
ablations is one place where that habit becomes concrete.

### 5.3 Prompt and decoding ablations

Prompt and decoding ablations is part of the canonical scope of error analysis and
ablations. In this chapter, the object under study is not merely a dataset or a model,
but the full failure slice and causal comparison: the items, prompts, outputs, graders,
uncertainty statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\Delta_{\mathrm{ablate}} = \frac{1}{n}\sum_{i=1}^n \mathbb{1}\{e_i \in E\}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For prompt and decoding
ablations, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in prompt and decoding ablations |
| Scoring rule | Exact formula for \mathbb{1}\{e_i \in E\} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report prompt and decoding ablations with item count, prompt protocol, grader version,
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

Worked evaluation pattern for prompt and decoding ablations:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, prompt and decoding ablations is especially delicate because the same
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
because it is noisy; it is interpreted through the design that produced it. Prompt and
decoding ablations is one place where that habit becomes concrete.

### 5.4 Retrieval and tool ablations

Retrieval and tool ablations is part of the canonical scope of error analysis and
ablations. In this chapter, the object under study is not merely a dataset or a model,
but the full failure slice and causal comparison: the items, prompts, outputs, graders,
uncertainty statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\Delta_{\mathrm{ablate}} = \frac{1}{n}\sum_{i=1}^n \mathbb{1}\{e_i \in E\}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For retrieval and tool
ablations, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in retrieval and tool ablations |
| Scoring rule | Exact formula for \mathbb{1}\{e_i \in E\} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report retrieval and tool ablations with item count, prompt protocol, grader version,
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

Worked evaluation pattern for retrieval and tool ablations:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, retrieval and tool ablations is especially delicate because the same
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
because it is noisy; it is interpreted through the design that produced it. Retrieval
and tool ablations is one place where that habit becomes concrete.

### 5.5 Metric ablations

Metric ablations is part of the canonical scope of error analysis and ablations. In this
chapter, the object under study is not merely a dataset or a model, but the full failure
slice and causal comparison: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\Delta_{\mathrm{ablate}} = \frac{1}{n}\sum_{i=1}^n \mathbb{1}\{e_i \in E\}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For metric ablations, those
choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in metric ablations |
| Scoring rule | Exact formula for \mathbb{1}\{e_i \in E\} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report metric ablations with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for metric ablations:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, metric ablations is especially delicate because the same model can be
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
ablations is one place where that habit becomes concrete.

## 6. Factorial Experiments

Factorial Experiments is the part of error analysis and ablations that turns the
approved TOC into a concrete learning path. The subsections below keep the focus on
Chapter 17's canonical job: measurement, reliability, uncertainty, and decision support
for AI systems.

### 6.1 Main effects

Main effects is part of the canonical scope of error analysis and ablations. In this
chapter, the object under study is not merely a dataset or a model, but the full failure
slice and causal comparison: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\Delta_{\mathrm{ablate}} = \frac{1}{n}\sum_{i=1}^n \mathbb{1}\{e_i \in E\}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For main effects, those
choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in main effects |
| Scoring rule | Exact formula for \mathbb{1}\{e_i \in E\} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report main effects with item count, prompt protocol, grader version, and a confidence
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

Worked evaluation pattern for main effects:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, main effects is especially delicate because the same model can be used
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
because it is noisy; it is interpreted through the design that produced it. Main effects
is one place where that habit becomes concrete.

### 6.2 Interactions

Interactions is part of the canonical scope of error analysis and ablations. In this
chapter, the object under study is not merely a dataset or a model, but the full failure
slice and causal comparison: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\Delta_{\mathrm{ablate}} = \frac{1}{n}\sum_{i=1}^n \mathbb{1}\{e_i \in E\}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For interactions, those
choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in interactions |
| Scoring rule | Exact formula for \mathbb{1}\{e_i \in E\} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report interactions with item count, prompt protocol, grader version, and a confidence
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

Worked evaluation pattern for interactions:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, interactions is especially delicate because the same model can be used
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
because it is noisy; it is interpreted through the design that produced it. Interactions
is one place where that habit becomes concrete.

### 6.3 Response surfaces

Response surfaces is part of the canonical scope of error analysis and ablations. In
this chapter, the object under study is not merely a dataset or a model, but the full
failure slice and causal comparison: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\Delta_{\mathrm{ablate}} = \frac{1}{n}\sum_{i=1}^n \mathbb{1}\{e_i \in E\}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For response surfaces, those
choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in response surfaces |
| Scoring rule | Exact formula for \mathbb{1}\{e_i \in E\} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report response surfaces with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for response surfaces:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, response surfaces is especially delicate because the same model can be
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
because it is noisy; it is interpreted through the design that produced it. Response
surfaces is one place where that habit becomes concrete.

### 6.4 ANOVA preview

ANOVA preview is part of the canonical scope of error analysis and ablations. In this
chapter, the object under study is not merely a dataset or a model, but the full failure
slice and causal comparison: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\Delta_{\mathrm{ablate}} = \frac{1}{n}\sum_{i=1}^n \mathbb{1}\{e_i \in E\}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For anova preview, those
choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in anova preview |
| Scoring rule | Exact formula for \mathbb{1}\{e_i \in E\} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report anova preview with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for anova preview:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, anova preview is especially delicate because the same model can be used
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
because it is noisy; it is interpreted through the design that produced it. ANOVA
preview is one place where that habit becomes concrete.

### 6.5 Attribution limits

Attribution limits is part of the canonical scope of error analysis and ablations. In
this chapter, the object under study is not merely a dataset or a model, but the full
failure slice and causal comparison: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\Delta_{\mathrm{ablate}} = \frac{1}{n}\sum_{i=1}^n \mathbb{1}\{e_i \in E\}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For attribution limits,
those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in attribution limits |
| Scoring rule | Exact formula for \mathbb{1}\{e_i \in E\} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report attribution limits with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for attribution limits:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, attribution limits is especially delicate because the same model can be
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
because it is noisy; it is interpreted through the design that produced it. Attribution
limits is one place where that habit becomes concrete.

## 7. Regression-Test Workflow

Regression-Test Workflow is the part of error analysis and ablations that turns the
approved TOC into a concrete learning path. The subsections below keep the focus on
Chapter 17's canonical job: measurement, reliability, uncertainty, and decision support
for AI systems.

### 7.1 Converting failures into eval cases

Converting failures into eval cases is part of the canonical scope of error analysis and
ablations. In this chapter, the object under study is not merely a dataset or a model,
but the full failure slice and causal comparison: the items, prompts, outputs, graders,
uncertainty statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\Delta_{\mathrm{ablate}} = \frac{1}{n}\sum_{i=1}^n \mathbb{1}\{e_i \in E\}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For converting failures into
eval cases, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in converting failures into eval cases |
| Scoring rule | Exact formula for \mathbb{1}\{e_i \in E\} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report converting failures into eval cases with item count, prompt protocol, grader
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

Worked evaluation pattern for converting failures into eval cases:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, converting failures into eval cases is especially delicate because the
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
because it is noisy; it is interpreted through the design that produced it. Converting
failures into eval cases is one place where that habit becomes concrete.

### 7.2 Versioning error suites

Versioning error suites is part of the canonical scope of error analysis and ablations.
In this chapter, the object under study is not merely a dataset or a model, but the full
failure slice and causal comparison: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\Delta_{\mathrm{ablate}} = \frac{1}{n}\sum_{i=1}^n \mathbb{1}\{e_i \in E\}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For versioning error suites,
those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in versioning error suites |
| Scoring rule | Exact formula for \mathbb{1}\{e_i \in E\} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report versioning error suites with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for versioning error suites:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, versioning error suites is especially delicate because the same model
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
because it is noisy; it is interpreted through the design that produced it. Versioning
error suites is one place where that habit becomes concrete.

### 7.3 Avoiding overfit fixes

Avoiding overfit fixes is part of the canonical scope of error analysis and ablations.
In this chapter, the object under study is not merely a dataset or a model, but the full
failure slice and causal comparison: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\Delta_{\mathrm{ablate}} = \frac{1}{n}\sum_{i=1}^n \mathbb{1}\{e_i \in E\}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For avoiding overfit fixes,
those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in avoiding overfit fixes |
| Scoring rule | Exact formula for \mathbb{1}\{e_i \in E\} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report avoiding overfit fixes with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for avoiding overfit fixes:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, avoiding overfit fixes is especially delicate because the same model can
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
because it is noisy; it is interpreted through the design that produced it. Avoiding
overfit fixes is one place where that habit becomes concrete.

### 7.4 Budgeting regression tests

Budgeting regression tests is part of the canonical scope of error analysis and
ablations. In this chapter, the object under study is not merely a dataset or a model,
but the full failure slice and causal comparison: the items, prompts, outputs, graders,
uncertainty statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\Delta_{\mathrm{ablate}} = \frac{1}{n}\sum_{i=1}^n \mathbb{1}\{e_i \in E\}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For budgeting regression
tests, those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in budgeting regression tests |
| Scoring rule | Exact formula for \mathbb{1}\{e_i \in E\} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report budgeting regression tests with item count, prompt protocol, grader version,
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

Worked evaluation pattern for budgeting regression tests:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, budgeting regression tests is especially delicate because the same model
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
because it is noisy; it is interpreted through the design that produced it. Budgeting
regression tests is one place where that habit becomes concrete.

### 7.5 Closing the debug loop

Closing the debug loop is part of the canonical scope of error analysis and ablations.
In this chapter, the object under study is not merely a dataset or a model, but the full
failure slice and causal comparison: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\Delta_{\mathrm{ablate}} = \frac{1}{n}\sum_{i=1}^n \mathbb{1}\{e_i \in E\}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For closing the debug loop,
those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in closing the debug loop |
| Scoring rule | Exact formula for \mathbb{1}\{e_i \in E\} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report closing the debug loop with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for closing the debug loop:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, closing the debug loop is especially delicate because the same model can
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
because it is noisy; it is interpreted through the design that produced it. Closing the
debug loop is one place where that habit becomes concrete.

## 8. Common Mistakes

| # | Mistake | Why It Is Wrong | Fix |
| --- | --- | --- | --- |
| 1 | Treating a point estimate as exact | Every finite evaluation has sampling error in error analysis and ablations. | Report uncertainty with the point estimate. |
| 2 | Changing prompts between models | The protocol changed with the treatment in error analysis and ablations. | Lock prompt, decoding, and grader before comparison. |
| 3 | Ignoring invalid outputs | Missingness can be correlated with model quality in error analysis and ablations. | Track invalid, timeout, refusal, and parse-failure rates. |
| 4 | Overfitting to a public leaderboard | Repeated testing leaks information from the benchmark in error analysis and ablations. | Use private holdouts and regression suites. |
| 5 | Averaging incomparable metrics | Different scales do not have shared units in error analysis and ablations. | Normalize by a stated decision rule or report separately. |
| 6 | Forgetting paired structure | Two models often answer the same items in error analysis and ablations. | Use paired bootstrap or paired tests where possible. |
| 7 | Reporting only aggregate performance | Subgroup failures can be hidden in error analysis and ablations. | Add slice and tail-risk views. |
| 8 | Trusting model judges blindly | LLM judges have position, verbosity, and self-preference biases in error analysis and ablations. | Calibrate judges against human labels. |
| 9 | Peeking during online experiments | Optional stopping inflates false positives in error analysis and ablations. | Use fixed horizons or sequential-valid methods. |
| 10 | Conflating evaluation with monitoring | Chapter 17 measures controlled evidence; production monitoring is ongoing operations in error analysis and ablations. | Hand off drift dashboards to Chapter 19 concepts. |

## 9. Exercises

1. (*) Aggregate metrics hide failure modes.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

2. (*) Failures as structured data.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

3. (*) Ablations as causal probes.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

4. (**) Debugging without benchmark overfitting.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

5. (**) From one bug to a regression suite.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

6. (**) Error set.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

7. (***) Confusion matrix.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

8. (***) Slice and subgroup.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

9. (***) Counterfactual example.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

10. (***) Ablation effect and interaction.
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
asks how deployed systems are observed and maintained over time. Error Analysis and
Ablations supplies the measurement discipline both chapters need.

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

- [Hypothesis Testing](../../07-Statistics/03-Hypothesis-Testing/notes.md)
- [HELM](https://arxiv.org/abs/2211.09110)
- [OpenAI Evals](https://github.com/openai/evals)
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)
- [Trustworthy Online Controlled Experiments](https://www.cambridge.org/core/books/trustworthy-online-controlled-experiments/trustworthy-online-controlled-experiments/524D7D0FDF6A542F732D047118618645)
- [Concentration Inequalities](../../06-Probability-Theory/05-Concentration-Inequalities/notes.md)
- [Estimation Theory](../../07-Statistics/02-Estimation-Theory/notes.md)
- [Hypothesis Testing](../../07-Statistics/03-Hypothesis-Testing/notes.md)

