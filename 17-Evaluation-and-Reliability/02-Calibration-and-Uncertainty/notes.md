[Back to Curriculum](../../README.md) | [Previous: Capability Benchmarks](../01-Capability-Benchmarks/notes.md) | [Next: Robustness and Distribution Shift](../03-Robustness-and-Distribution-Shift/notes.md)

---

# Calibration and Uncertainty

> _"A probability is a promise about frequency."_

## Overview

Calibration asks whether confidence matches correctness; uncertainty methods decide when
a model should answer, abstain, or return a set of plausible answers.

The chapter treats evaluation as a mathematical object: a controlled protocol that maps
model behavior into evidence with uncertainty. A result is not just a number; it is an
estimate produced by a task distribution, a scoring rule, and an aggregation rule.

This section is written in LaTeX Markdown. Inline mathematics uses `$...$`, and display
equations use `$$...$$`. The emphasis is practical but rigorous: every metric should be
linked to the statistical assumptions that make it meaningful.

## Prerequisites

- [Common Distributions](../../06-Probability-Theory/02-Common-Distributions/notes.md)
- [Estimation Theory](../../07-Statistics/02-Estimation-Theory/notes.md)
- [Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md)
- [Loss Functions](../../13-ML-Specific-Math/01-Loss-Functions/notes.md)
- [Language Model Probability](../../15-Math-for-LLMs/05-Language-Model-Probability/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Executable demonstrations for calibration and uncertainty |
| [exercises.ipynb](exercises.ipynb) | Graded practice for calibration and uncertainty |

## Learning Objectives

After completing this section, you will be able to:

- Define the core objects used in calibration and uncertainty
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
  - [1.1 Confidence should match correctness](#11-confidence-should-match-correctness)
  - [1.2 High accuracy can still be unsafe](#12-high-accuracy-can-still-be-unsafe)
  - [1.3 Selective prediction and abstention](#13-selective-prediction-and-abstention)
  - [1.4 Epistemic and aleatoric uncertainty](#14-epistemic-and-aleatoric-uncertainty)
  - [1.5 Why LLM verbal confidence is unreliable](#15-why-llm-verbal-confidence-is-unreliable)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Calibration condition](#21-calibration-condition)
  - [2.2 Reliability function](#22-reliability-function)
  - [2.3 Expected and maximum calibration error](#23-expected-and-maximum-calibration-error)
  - [2.4 Brier score and negative log-likelihood](#24-brier-score-and-negative-log-likelihood)
  - [2.5 Coverage, set size, and selective risk](#25-coverage-set-size-and-selective-risk)
- [3. Calibration Diagnostics](#3-calibration-diagnostics)
  - [3.1 Reliability diagrams](#31-reliability-diagrams)
  - [3.2 Confidence histograms](#32-confidence-histograms)
  - [3.3 Binning bias and adaptive bins](#33-binning-bias-and-adaptive-bins)
  - [3.4 Bootstrap uncertainty for ECE](#34-bootstrap-uncertainty-for-ece)
  - [3.5 Prompt and slice-level calibration](#35-prompt-and-slice-level-calibration)
- [4. Proper Scoring Rules](#4-proper-scoring-rules)
  - [4.1 Log score](#41-log-score)
  - [4.2 Brier score](#42-brier-score)
  - [4.3 Sharpness versus calibration](#43-sharpness-versus-calibration)
  - [4.4 Strict propriety](#44-strict-propriety)
  - [4.5 Scoring rules for LLM outputs](#45-scoring-rules-for-llm-outputs)
- [5. Post-Hoc Calibration](#5-post-hoc-calibration)
  - [5.1 Temperature scaling](#51-temperature-scaling)
  - [5.2 Platt scaling](#52-platt-scaling)
  - [5.3 Isotonic regression](#53-isotonic-regression)
  - [5.4 Validation split discipline](#54-validation-split-discipline)
  - [5.5 Calibration under shift](#55-calibration-under-shift)
- [6. Uncertainty Sets and Abstention](#6-uncertainty-sets-and-abstention)
  - [6.1 Conformal prediction](#61-conformal-prediction)
  - [6.2 Prediction sets and marginal coverage](#62-prediction-sets-and-marginal-coverage)
  - [6.3 Risk-coverage curves](#63-risk-coverage-curves)
  - [6.4 Reject option and triage](#64-reject-option-and-triage)
  - [6.5 Structured-output uncertainty previews](#65-structured-output-uncertainty-previews)
- [7. LLM-Specific Uncertainty](#7-llm-specific-uncertainty)
  - [7.1 Token confidence](#71-token-confidence)
  - [7.2 Sequence confidence](#72-sequence-confidence)
  - [7.3 Self-consistency and sample disagreement](#73-self-consistency-and-sample-disagreement)
  - [7.4 Semantic uncertainty](#74-semantic-uncertainty)
  - [7.5 Knowing what it knows](#75-knowing-what-it-knows)
- [8. Common Mistakes](#8-common-mistakes)
- [9. Exercises](#9-exercises)
- [10. Why This Matters for AI](#10-why-this-matters-for-ai)
- [11. Conceptual Bridge](#11-conceptual-bridge)
- [References](#references)

---

## 1. Intuition

Intuition is the part of calibration and uncertainty that turns the approved TOC into a
concrete learning path. The subsections below keep the focus on Chapter 17's canonical
job: measurement, reliability, uncertainty, and decision support for AI systems.

### 1.1 Confidence should match correctness

Confidence should match correctness is part of the canonical scope of calibration and
uncertainty. In this chapter, the object under study is not merely a dataset or a model,
but the full probabilistic forecast: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\operatorname{ECE} = \frac{1}{n}\sum_{i=1}^n \ell_{\mathrm{NLL}}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For confidence should match
correctness, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in confidence should match correctness |
| Scoring rule | Exact formula for \ell_{\mathrm{NLL}} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report confidence should match correctness with item count, prompt protocol, grader
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

Worked evaluation pattern for confidence should match correctness:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, confidence should match correctness is especially delicate because the
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
because it is noisy; it is interpreted through the design that produced it. Confidence
should match correctness is one place where that habit becomes concrete.

### 1.2 High accuracy can still be unsafe

High accuracy can still be unsafe is part of the canonical scope of calibration and
uncertainty. In this chapter, the object under study is not merely a dataset or a model,
but the full probabilistic forecast: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\operatorname{ECE} = \frac{1}{n}\sum_{i=1}^n \ell_{\mathrm{NLL}}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For high accuracy can still
be unsafe, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in high accuracy can still be unsafe |
| Scoring rule | Exact formula for \ell_{\mathrm{NLL}} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report high accuracy can still be unsafe with item count, prompt protocol, grader
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

Worked evaluation pattern for high accuracy can still be unsafe:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, high accuracy can still be unsafe is especially delicate because the
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
because it is noisy; it is interpreted through the design that produced it. High
accuracy can still be unsafe is one place where that habit becomes concrete.

### 1.3 Selective prediction and abstention

Selective prediction and abstention is part of the canonical scope of calibration and
uncertainty. In this chapter, the object under study is not merely a dataset or a model,
but the full probabilistic forecast: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\operatorname{ECE} = \frac{1}{n}\sum_{i=1}^n \ell_{\mathrm{NLL}}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For selective prediction and
abstention, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in selective prediction and abstention |
| Scoring rule | Exact formula for \ell_{\mathrm{NLL}} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report selective prediction and abstention with item count, prompt protocol, grader
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

Worked evaluation pattern for selective prediction and abstention:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, selective prediction and abstention is especially delicate because the
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
because it is noisy; it is interpreted through the design that produced it. Selective
prediction and abstention is one place where that habit becomes concrete.

### 1.4 Epistemic and aleatoric uncertainty

Epistemic and aleatoric uncertainty is part of the canonical scope of calibration and
uncertainty. In this chapter, the object under study is not merely a dataset or a model,
but the full probabilistic forecast: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\operatorname{ECE} = \frac{1}{n}\sum_{i=1}^n \ell_{\mathrm{NLL}}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For epistemic and aleatoric
uncertainty, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in epistemic and aleatoric uncertainty |
| Scoring rule | Exact formula for \ell_{\mathrm{NLL}} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report epistemic and aleatoric uncertainty with item count, prompt protocol, grader
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

Worked evaluation pattern for epistemic and aleatoric uncertainty:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, epistemic and aleatoric uncertainty is especially delicate because the
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
because it is noisy; it is interpreted through the design that produced it. Epistemic
and aleatoric uncertainty is one place where that habit becomes concrete.

### 1.5 Why LLM verbal confidence is unreliable

Why LLM verbal confidence is unreliable is part of the canonical scope of calibration
and uncertainty. In this chapter, the object under study is not merely a dataset or a
model, but the full probabilistic forecast: the items, prompts, outputs, graders,
uncertainty statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\operatorname{ECE} = \frac{1}{n}\sum_{i=1}^n \ell_{\mathrm{NLL}}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For why llm verbal
confidence is unreliable, those choices determine whether the reported number is
evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in why llm verbal confidence is unreliable |
| Scoring rule | Exact formula for \ell_{\mathrm{NLL}} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report why llm verbal confidence is unreliable with item count, prompt protocol,
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

Worked evaluation pattern for why llm verbal confidence is unreliable:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, why llm verbal confidence is unreliable is especially delicate because
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
because it is noisy; it is interpreted through the design that produced it. Why LLM
verbal confidence is unreliable is one place where that habit becomes concrete.

## 2. Formal Definitions

Formal Definitions is the part of calibration and uncertainty that turns the approved
TOC into a concrete learning path. The subsections below keep the focus on Chapter 17's
canonical job: measurement, reliability, uncertainty, and decision support for AI
systems.

### 2.1 Calibration condition

Calibration condition is part of the canonical scope of calibration and uncertainty. In
this chapter, the object under study is not merely a dataset or a model, but the full
probabilistic forecast: the items, prompts, outputs, graders, uncertainty statements,
and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\operatorname{ECE} = \frac{1}{n}\sum_{i=1}^n \ell_{\mathrm{NLL}}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For calibration condition,
those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in calibration condition |
| Scoring rule | Exact formula for \ell_{\mathrm{NLL}} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report calibration condition with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for calibration condition:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, calibration condition is especially delicate because the same model can
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
because it is noisy; it is interpreted through the design that produced it. Calibration
condition is one place where that habit becomes concrete.

### 2.2 Reliability function

Reliability function is part of the canonical scope of calibration and uncertainty. In
this chapter, the object under study is not merely a dataset or a model, but the full
probabilistic forecast: the items, prompts, outputs, graders, uncertainty statements,
and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\operatorname{ECE} = \frac{1}{n}\sum_{i=1}^n \ell_{\mathrm{NLL}}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For reliability function,
those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in reliability function |
| Scoring rule | Exact formula for \ell_{\mathrm{NLL}} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report reliability function with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for reliability function:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, reliability function is especially delicate because the same model can
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
because it is noisy; it is interpreted through the design that produced it. Reliability
function is one place where that habit becomes concrete.

### 2.3 Expected and maximum calibration error

Expected and maximum calibration error is part of the canonical scope of calibration and
uncertainty. In this chapter, the object under study is not merely a dataset or a model,
but the full probabilistic forecast: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\operatorname{ECE} = \frac{1}{n}\sum_{i=1}^n \ell_{\mathrm{NLL}}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For expected and maximum
calibration error, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in expected and maximum calibration error |
| Scoring rule | Exact formula for \ell_{\mathrm{NLL}} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report expected and maximum calibration error with item count, prompt protocol, grader
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

Worked evaluation pattern for expected and maximum calibration error:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, expected and maximum calibration error is especially delicate because
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
because it is noisy; it is interpreted through the design that produced it. Expected and
maximum calibration error is one place where that habit becomes concrete.

### 2.4 Brier score and negative log-likelihood

Brier score and negative log-likelihood is part of the canonical scope of calibration
and uncertainty. In this chapter, the object under study is not merely a dataset or a
model, but the full probabilistic forecast: the items, prompts, outputs, graders,
uncertainty statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\operatorname{ECE} = \frac{1}{n}\sum_{i=1}^n \ell_{\mathrm{NLL}}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For brier score and negative
log-likelihood, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in brier score and negative log-likelihood |
| Scoring rule | Exact formula for \ell_{\mathrm{NLL}} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report brier score and negative log-likelihood with item count, prompt protocol,
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

Worked evaluation pattern for brier score and negative log-likelihood:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, brier score and negative log-likelihood is especially delicate because
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
because it is noisy; it is interpreted through the design that produced it. Brier score
and negative log-likelihood is one place where that habit becomes concrete.

### 2.5 Coverage, set size, and selective risk

Coverage, set size, and selective risk is part of the canonical scope of calibration and
uncertainty. In this chapter, the object under study is not merely a dataset or a model,
but the full probabilistic forecast: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\operatorname{ECE} = \frac{1}{n}\sum_{i=1}^n \ell_{\mathrm{NLL}}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For coverage, set size, and
selective risk, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in coverage, set size, and selective risk |
| Scoring rule | Exact formula for \ell_{\mathrm{NLL}} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report coverage, set size, and selective risk with item count, prompt protocol, grader
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

Worked evaluation pattern for coverage, set size, and selective risk:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, coverage, set size, and selective risk is especially delicate because
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
because it is noisy; it is interpreted through the design that produced it. Coverage,
set size, and selective risk is one place where that habit becomes concrete.

## 3. Calibration Diagnostics

Calibration Diagnostics is the part of calibration and uncertainty that turns the
approved TOC into a concrete learning path. The subsections below keep the focus on
Chapter 17's canonical job: measurement, reliability, uncertainty, and decision support
for AI systems.

### 3.1 Reliability diagrams

Reliability diagrams is part of the canonical scope of calibration and uncertainty. In
this chapter, the object under study is not merely a dataset or a model, but the full
probabilistic forecast: the items, prompts, outputs, graders, uncertainty statements,
and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\operatorname{ECE} = \frac{1}{n}\sum_{i=1}^n \ell_{\mathrm{NLL}}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For reliability diagrams,
those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in reliability diagrams |
| Scoring rule | Exact formula for \ell_{\mathrm{NLL}} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report reliability diagrams with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for reliability diagrams:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, reliability diagrams is especially delicate because the same model can
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
because it is noisy; it is interpreted through the design that produced it. Reliability
diagrams is one place where that habit becomes concrete.

### 3.2 Confidence histograms

Confidence histograms is part of the canonical scope of calibration and uncertainty. In
this chapter, the object under study is not merely a dataset or a model, but the full
probabilistic forecast: the items, prompts, outputs, graders, uncertainty statements,
and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\operatorname{ECE} = \frac{1}{n}\sum_{i=1}^n \ell_{\mathrm{NLL}}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For confidence histograms,
those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in confidence histograms |
| Scoring rule | Exact formula for \ell_{\mathrm{NLL}} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report confidence histograms with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for confidence histograms:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, confidence histograms is especially delicate because the same model can
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
because it is noisy; it is interpreted through the design that produced it. Confidence
histograms is one place where that habit becomes concrete.

### 3.3 Binning bias and adaptive bins

Binning bias and adaptive bins is part of the canonical scope of calibration and
uncertainty. In this chapter, the object under study is not merely a dataset or a model,
but the full probabilistic forecast: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\operatorname{ECE} = \frac{1}{n}\sum_{i=1}^n \ell_{\mathrm{NLL}}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For binning bias and
adaptive bins, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in binning bias and adaptive bins |
| Scoring rule | Exact formula for \ell_{\mathrm{NLL}} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report binning bias and adaptive bins with item count, prompt protocol, grader
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

Worked evaluation pattern for binning bias and adaptive bins:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, binning bias and adaptive bins is especially delicate because the same
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
because it is noisy; it is interpreted through the design that produced it. Binning bias
and adaptive bins is one place where that habit becomes concrete.

### 3.4 Bootstrap uncertainty for ECE

Bootstrap uncertainty for ECE is part of the canonical scope of calibration and
uncertainty. In this chapter, the object under study is not merely a dataset or a model,
but the full probabilistic forecast: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\operatorname{ECE} = \frac{1}{n}\sum_{i=1}^n \ell_{\mathrm{NLL}}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For bootstrap uncertainty
for ece, those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in bootstrap uncertainty for ece |
| Scoring rule | Exact formula for \ell_{\mathrm{NLL}} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report bootstrap uncertainty for ece with item count, prompt protocol, grader version,
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

Worked evaluation pattern for bootstrap uncertainty for ece:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, bootstrap uncertainty for ece is especially delicate because the same
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
uncertainty for ECE is one place where that habit becomes concrete.

### 3.5 Prompt and slice-level calibration

Prompt and slice-level calibration is part of the canonical scope of calibration and
uncertainty. In this chapter, the object under study is not merely a dataset or a model,
but the full probabilistic forecast: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\operatorname{ECE} = \frac{1}{n}\sum_{i=1}^n \ell_{\mathrm{NLL}}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For prompt and slice-level
calibration, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in prompt and slice-level calibration |
| Scoring rule | Exact formula for \ell_{\mathrm{NLL}} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report prompt and slice-level calibration with item count, prompt protocol, grader
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

Worked evaluation pattern for prompt and slice-level calibration:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, prompt and slice-level calibration is especially delicate because the
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
because it is noisy; it is interpreted through the design that produced it. Prompt and
slice-level calibration is one place where that habit becomes concrete.

## 4. Proper Scoring Rules

Proper Scoring Rules is the part of calibration and uncertainty that turns the approved
TOC into a concrete learning path. The subsections below keep the focus on Chapter 17's
canonical job: measurement, reliability, uncertainty, and decision support for AI
systems.

### 4.1 Log score

Log score is part of the canonical scope of calibration and uncertainty. In this
chapter, the object under study is not merely a dataset or a model, but the full
probabilistic forecast: the items, prompts, outputs, graders, uncertainty statements,
and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\operatorname{ECE} = \frac{1}{n}\sum_{i=1}^n \ell_{\mathrm{NLL}}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For log score, those choices
determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in log score |
| Scoring rule | Exact formula for \ell_{\mathrm{NLL}} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report log score with item count, prompt protocol, grader version, and a confidence
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

Worked evaluation pattern for log score:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, log score is especially delicate because the same model can be used with
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
because it is noisy; it is interpreted through the design that produced it. Log score is
one place where that habit becomes concrete.

### 4.2 Brier score

Brier score is part of the canonical scope of calibration and uncertainty. In this
chapter, the object under study is not merely a dataset or a model, but the full
probabilistic forecast: the items, prompts, outputs, graders, uncertainty statements,
and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\operatorname{ECE} = \frac{1}{n}\sum_{i=1}^n \ell_{\mathrm{NLL}}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For brier score, those
choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in brier score |
| Scoring rule | Exact formula for \ell_{\mathrm{NLL}} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report brier score with item count, prompt protocol, grader version, and a confidence
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

Worked evaluation pattern for brier score:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, brier score is especially delicate because the same model can be used
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
because it is noisy; it is interpreted through the design that produced it. Brier score
is one place where that habit becomes concrete.

### 4.3 Sharpness versus calibration

Sharpness versus calibration is part of the canonical scope of calibration and
uncertainty. In this chapter, the object under study is not merely a dataset or a model,
but the full probabilistic forecast: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\operatorname{ECE} = \frac{1}{n}\sum_{i=1}^n \ell_{\mathrm{NLL}}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For sharpness versus
calibration, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in sharpness versus calibration |
| Scoring rule | Exact formula for \ell_{\mathrm{NLL}} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report sharpness versus calibration with item count, prompt protocol, grader version,
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

Worked evaluation pattern for sharpness versus calibration:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, sharpness versus calibration is especially delicate because the same
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
because it is noisy; it is interpreted through the design that produced it. Sharpness
versus calibration is one place where that habit becomes concrete.

### 4.4 Strict propriety

Strict propriety is part of the canonical scope of calibration and uncertainty. In this
chapter, the object under study is not merely a dataset or a model, but the full
probabilistic forecast: the items, prompts, outputs, graders, uncertainty statements,
and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\operatorname{ECE} = \frac{1}{n}\sum_{i=1}^n \ell_{\mathrm{NLL}}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For strict propriety, those
choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in strict propriety |
| Scoring rule | Exact formula for \ell_{\mathrm{NLL}} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report strict propriety with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for strict propriety:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, strict propriety is especially delicate because the same model can be
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
because it is noisy; it is interpreted through the design that produced it. Strict
propriety is one place where that habit becomes concrete.

### 4.5 Scoring rules for LLM outputs

Scoring rules for LLM outputs is part of the canonical scope of calibration and
uncertainty. In this chapter, the object under study is not merely a dataset or a model,
but the full probabilistic forecast: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\operatorname{ECE} = \frac{1}{n}\sum_{i=1}^n \ell_{\mathrm{NLL}}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For scoring rules for llm
outputs, those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in scoring rules for llm outputs |
| Scoring rule | Exact formula for \ell_{\mathrm{NLL}} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report scoring rules for llm outputs with item count, prompt protocol, grader version,
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

Worked evaluation pattern for scoring rules for llm outputs:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, scoring rules for llm outputs is especially delicate because the same
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
because it is noisy; it is interpreted through the design that produced it. Scoring
rules for LLM outputs is one place where that habit becomes concrete.

## 5. Post-Hoc Calibration

Post-Hoc Calibration is the part of calibration and uncertainty that turns the approved
TOC into a concrete learning path. The subsections below keep the focus on Chapter 17's
canonical job: measurement, reliability, uncertainty, and decision support for AI
systems.

### 5.1 Temperature scaling

Temperature scaling is part of the canonical scope of calibration and uncertainty. In
this chapter, the object under study is not merely a dataset or a model, but the full
probabilistic forecast: the items, prompts, outputs, graders, uncertainty statements,
and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\operatorname{ECE} = \frac{1}{n}\sum_{i=1}^n \ell_{\mathrm{NLL}}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For temperature scaling,
those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in temperature scaling |
| Scoring rule | Exact formula for \ell_{\mathrm{NLL}} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report temperature scaling with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for temperature scaling:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, temperature scaling is especially delicate because the same model can be
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
because it is noisy; it is interpreted through the design that produced it. Temperature
scaling is one place where that habit becomes concrete.

### 5.2 Platt scaling

Platt scaling is part of the canonical scope of calibration and uncertainty. In this
chapter, the object under study is not merely a dataset or a model, but the full
probabilistic forecast: the items, prompts, outputs, graders, uncertainty statements,
and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\operatorname{ECE} = \frac{1}{n}\sum_{i=1}^n \ell_{\mathrm{NLL}}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For platt scaling, those
choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in platt scaling |
| Scoring rule | Exact formula for \ell_{\mathrm{NLL}} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report platt scaling with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for platt scaling:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, platt scaling is especially delicate because the same model can be used
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
because it is noisy; it is interpreted through the design that produced it. Platt
scaling is one place where that habit becomes concrete.

### 5.3 Isotonic regression

Isotonic regression is part of the canonical scope of calibration and uncertainty. In
this chapter, the object under study is not merely a dataset or a model, but the full
probabilistic forecast: the items, prompts, outputs, graders, uncertainty statements,
and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\operatorname{ECE} = \frac{1}{n}\sum_{i=1}^n \ell_{\mathrm{NLL}}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For isotonic regression,
those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in isotonic regression |
| Scoring rule | Exact formula for \ell_{\mathrm{NLL}} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report isotonic regression with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for isotonic regression:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, isotonic regression is especially delicate because the same model can be
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
because it is noisy; it is interpreted through the design that produced it. Isotonic
regression is one place where that habit becomes concrete.

### 5.4 Validation split discipline

Validation split discipline is part of the canonical scope of calibration and
uncertainty. In this chapter, the object under study is not merely a dataset or a model,
but the full probabilistic forecast: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\operatorname{ECE} = \frac{1}{n}\sum_{i=1}^n \ell_{\mathrm{NLL}}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For validation split
discipline, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in validation split discipline |
| Scoring rule | Exact formula for \ell_{\mathrm{NLL}} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report validation split discipline with item count, prompt protocol, grader version,
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

Worked evaluation pattern for validation split discipline:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, validation split discipline is especially delicate because the same
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
because it is noisy; it is interpreted through the design that produced it. Validation
split discipline is one place where that habit becomes concrete.

### 5.5 Calibration under shift

Calibration under shift is part of the canonical scope of calibration and uncertainty.
In this chapter, the object under study is not merely a dataset or a model, but the full
probabilistic forecast: the items, prompts, outputs, graders, uncertainty statements,
and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\operatorname{ECE} = \frac{1}{n}\sum_{i=1}^n \ell_{\mathrm{NLL}}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For calibration under shift,
those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in calibration under shift |
| Scoring rule | Exact formula for \ell_{\mathrm{NLL}} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report calibration under shift with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for calibration under shift:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, calibration under shift is especially delicate because the same model
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
because it is noisy; it is interpreted through the design that produced it. Calibration
under shift is one place where that habit becomes concrete.

## 6. Uncertainty Sets and Abstention

Uncertainty Sets and Abstention is the part of calibration and uncertainty that turns
the approved TOC into a concrete learning path. The subsections below keep the focus on
Chapter 17's canonical job: measurement, reliability, uncertainty, and decision support
for AI systems.

### 6.1 Conformal prediction

Conformal prediction is part of the canonical scope of calibration and uncertainty. In
this chapter, the object under study is not merely a dataset or a model, but the full
probabilistic forecast: the items, prompts, outputs, graders, uncertainty statements,
and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\operatorname{ECE} = \frac{1}{n}\sum_{i=1}^n \ell_{\mathrm{NLL}}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For conformal prediction,
those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in conformal prediction |
| Scoring rule | Exact formula for \ell_{\mathrm{NLL}} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report conformal prediction with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for conformal prediction:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, conformal prediction is especially delicate because the same model can
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
because it is noisy; it is interpreted through the design that produced it. Conformal
prediction is one place where that habit becomes concrete.

### 6.2 Prediction sets and marginal coverage

Prediction sets and marginal coverage is part of the canonical scope of calibration and
uncertainty. In this chapter, the object under study is not merely a dataset or a model,
but the full probabilistic forecast: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\operatorname{ECE} = \frac{1}{n}\sum_{i=1}^n \ell_{\mathrm{NLL}}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For prediction sets and
marginal coverage, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in prediction sets and marginal coverage |
| Scoring rule | Exact formula for \ell_{\mathrm{NLL}} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report prediction sets and marginal coverage with item count, prompt protocol, grader
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

Worked evaluation pattern for prediction sets and marginal coverage:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, prediction sets and marginal coverage is especially delicate because the
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
because it is noisy; it is interpreted through the design that produced it. Prediction
sets and marginal coverage is one place where that habit becomes concrete.

### 6.3 Risk-coverage curves

Risk-coverage curves is part of the canonical scope of calibration and uncertainty. In
this chapter, the object under study is not merely a dataset or a model, but the full
probabilistic forecast: the items, prompts, outputs, graders, uncertainty statements,
and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\operatorname{ECE} = \frac{1}{n}\sum_{i=1}^n \ell_{\mathrm{NLL}}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For risk-coverage curves,
those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in risk-coverage curves |
| Scoring rule | Exact formula for \ell_{\mathrm{NLL}} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report risk-coverage curves with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for risk-coverage curves:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, risk-coverage curves is especially delicate because the same model can
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
because it is noisy; it is interpreted through the design that produced it. Risk-
coverage curves is one place where that habit becomes concrete.

### 6.4 Reject option and triage

Reject option and triage is part of the canonical scope of calibration and uncertainty.
In this chapter, the object under study is not merely a dataset or a model, but the full
probabilistic forecast: the items, prompts, outputs, graders, uncertainty statements,
and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\operatorname{ECE} = \frac{1}{n}\sum_{i=1}^n \ell_{\mathrm{NLL}}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For reject option and
triage, those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in reject option and triage |
| Scoring rule | Exact formula for \ell_{\mathrm{NLL}} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report reject option and triage with item count, prompt protocol, grader version, and
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

Worked evaluation pattern for reject option and triage:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, reject option and triage is especially delicate because the same model
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
because it is noisy; it is interpreted through the design that produced it. Reject
option and triage is one place where that habit becomes concrete.

### 6.5 Structured-output uncertainty previews

Structured-output uncertainty previews is part of the canonical scope of calibration and
uncertainty. In this chapter, the object under study is not merely a dataset or a model,
but the full probabilistic forecast: the items, prompts, outputs, graders, uncertainty
statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\operatorname{ECE} = \frac{1}{n}\sum_{i=1}^n \ell_{\mathrm{NLL}}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For structured-output
uncertainty previews, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in structured-output uncertainty previews |
| Scoring rule | Exact formula for \ell_{\mathrm{NLL}} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report structured-output uncertainty previews with item count, prompt protocol, grader
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

Worked evaluation pattern for structured-output uncertainty previews:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, structured-output uncertainty previews is especially delicate because
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
because it is noisy; it is interpreted through the design that produced it. Structured-
output uncertainty previews is one place where that habit becomes concrete.

## 7. LLM-Specific Uncertainty

LLM-Specific Uncertainty is the part of calibration and uncertainty that turns the
approved TOC into a concrete learning path. The subsections below keep the focus on
Chapter 17's canonical job: measurement, reliability, uncertainty, and decision support
for AI systems.

### 7.1 Token confidence

Token confidence is part of the canonical scope of calibration and uncertainty. In this
chapter, the object under study is not merely a dataset or a model, but the full
probabilistic forecast: the items, prompts, outputs, graders, uncertainty statements,
and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\operatorname{ECE} = \frac{1}{n}\sum_{i=1}^n \ell_{\mathrm{NLL}}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For token confidence, those
choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in token confidence |
| Scoring rule | Exact formula for \ell_{\mathrm{NLL}} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report token confidence with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for token confidence:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, token confidence is especially delicate because the same model can be
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
because it is noisy; it is interpreted through the design that produced it. Token
confidence is one place where that habit becomes concrete.

### 7.2 Sequence confidence

Sequence confidence is part of the canonical scope of calibration and uncertainty. In
this chapter, the object under study is not merely a dataset or a model, but the full
probabilistic forecast: the items, prompts, outputs, graders, uncertainty statements,
and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\operatorname{ECE} = \frac{1}{n}\sum_{i=1}^n \ell_{\mathrm{NLL}}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For sequence confidence,
those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in sequence confidence |
| Scoring rule | Exact formula for \ell_{\mathrm{NLL}} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report sequence confidence with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for sequence confidence:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, sequence confidence is especially delicate because the same model can be
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
because it is noisy; it is interpreted through the design that produced it. Sequence
confidence is one place where that habit becomes concrete.

### 7.3 Self-consistency and sample disagreement

Self-consistency and sample disagreement is part of the canonical scope of calibration
and uncertainty. In this chapter, the object under study is not merely a dataset or a
model, but the full probabilistic forecast: the items, prompts, outputs, graders,
uncertainty statements, and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\operatorname{ECE} = \frac{1}{n}\sum_{i=1}^n \ell_{\mathrm{NLL}}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For self-consistency and
sample disagreement, those choices determine whether the reported number is evidence or
decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in self-consistency and sample disagreement |
| Scoring rule | Exact formula for \ell_{\mathrm{NLL}} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report self-consistency and sample disagreement with item count, prompt protocol,
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

Worked evaluation pattern for self-consistency and sample disagreement:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, self-consistency and sample disagreement is especially delicate because
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
because it is noisy; it is interpreted through the design that produced it. Self-
consistency and sample disagreement is one place where that habit becomes concrete.

### 7.4 Semantic uncertainty

Semantic uncertainty is part of the canonical scope of calibration and uncertainty. In
this chapter, the object under study is not merely a dataset or a model, but the full
probabilistic forecast: the items, prompts, outputs, graders, uncertainty statements,
and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\operatorname{ECE} = \frac{1}{n}\sum_{i=1}^n \ell_{\mathrm{NLL}}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For semantic uncertainty,
those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in semantic uncertainty |
| Scoring rule | Exact formula for \ell_{\mathrm{NLL}} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report semantic uncertainty with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for semantic uncertainty:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, semantic uncertainty is especially delicate because the same model can
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
because it is noisy; it is interpreted through the design that produced it. Semantic
uncertainty is one place where that habit becomes concrete.

### 7.5 Knowing what it knows

Knowing what it knows is part of the canonical scope of calibration and uncertainty. In
this chapter, the object under study is not merely a dataset or a model, but the full
probabilistic forecast: the items, prompts, outputs, graders, uncertainty statements,
and decision rules that turn model behavior into evidence.

The basic mathematical pattern is an empirical estimator. For a model or system $m$
evaluated on items $z_1,\ldots,z_n$, the local estimate is written

$$
\operatorname{ECE} = \frac{1}{n}\sum_{i=1}^n \ell_{\mathrm{NLL}}.
$$

The formula is intentionally simple. The difficulty lies in deciding what counts as an
item, which loss or score is meaningful, whether the items are independent, and whether
the estimate answers the real product or research question. For knowing what it knows,
those choices determine whether the reported number is evidence or decoration.

A useful invariant is that every evaluation claim should be reproducible as a tuple
$(m,\mathcal{T},\pi,g,\rho)$, where $m$ is the system, $\mathcal{T}$ is the task sample,
$\pi$ is the prompt or intervention policy, $g$ is the grader, and $\rho$ is the
aggregation rule. If any part of this tuple is missing, the number cannot be audited.

| Component | What to record | Why it matters |
| --- | --- | --- |
| Item definition | IDs, source, split, and allowed transformations | Prevents accidental drift in knowing what it knows |
| Scoring rule | Exact formula for \ell_{\mathrm{NLL}} | Makes comparisons repeatable |
| Aggregation | Mean, weighted mean, worst group, or pairwise model | Determines the scientific claim |
| Uncertainty | Standard error, interval, or posterior summary | Separates signal from sampling noise |
| Audit trail | Code version and random seeds | Makes failures debuggable |

Examples of correct use:
- Report knowing what it knows with item count, prompt protocol, grader version, and a
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

Worked evaluation pattern for knowing what it knows:
1. Define the evaluation population in words before writing code.
2. Choose the smallest metric set that answers the decision question.
3. Compute the point estimate and an uncertainty statement together.
4. Run a slice or paired analysis to check whether the aggregate hides structure.
5. Archive raw outputs, scores, and seeds before changing the prompt or grader.

For AI systems, knowing what it knows is especially delicate because the same model can
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
because it is noisy; it is interpreted through the design that produced it. Knowing what
it knows is one place where that habit becomes concrete.

## 8. Common Mistakes

| # | Mistake | Why It Is Wrong | Fix |
| --- | --- | --- | --- |
| 1 | Treating a point estimate as exact | Every finite evaluation has sampling error in calibration and uncertainty. | Report uncertainty with the point estimate. |
| 2 | Changing prompts between models | The protocol changed with the treatment in calibration and uncertainty. | Lock prompt, decoding, and grader before comparison. |
| 3 | Ignoring invalid outputs | Missingness can be correlated with model quality in calibration and uncertainty. | Track invalid, timeout, refusal, and parse-failure rates. |
| 4 | Overfitting to a public leaderboard | Repeated testing leaks information from the benchmark in calibration and uncertainty. | Use private holdouts and regression suites. |
| 5 | Averaging incomparable metrics | Different scales do not have shared units in calibration and uncertainty. | Normalize by a stated decision rule or report separately. |
| 6 | Forgetting paired structure | Two models often answer the same items in calibration and uncertainty. | Use paired bootstrap or paired tests where possible. |
| 7 | Reporting only aggregate performance | Subgroup failures can be hidden in calibration and uncertainty. | Add slice and tail-risk views. |
| 8 | Trusting model judges blindly | LLM judges have position, verbosity, and self-preference biases in calibration and uncertainty. | Calibrate judges against human labels. |
| 9 | Peeking during online experiments | Optional stopping inflates false positives in calibration and uncertainty. | Use fixed horizons or sequential-valid methods. |
| 10 | Conflating evaluation with monitoring | Chapter 17 measures controlled evidence; production monitoring is ongoing operations in calibration and uncertainty. | Hand off drift dashboards to Chapter 19 concepts. |

## 9. Exercises

1. (*) Confidence should match correctness.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

2. (*) High accuracy can still be unsafe.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

3. (*) Selective prediction and abstention.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

4. (**) Epistemic and aleatoric uncertainty.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

5. (**) Why LLM verbal confidence is unreliable.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

6. (**) Calibration condition.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

7. (***) Reliability function.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

8. (***) Expected and maximum calibration error.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

9. (***) Brier score and negative log-likelihood.
(a) Define the relevant evaluation object. (b) Write the estimator in LaTeX notation.
(c) Give one example where the estimator is reliable. (d) Give one example where the
same number would be misleading. (e) Describe what the theory notebook should verify
computationally.

10. (***) Coverage, set size, and selective risk.
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
asks how deployed systems are observed and maintained over time. Calibration and
Uncertainty supplies the measurement discipline both chapters need.

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

- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)
- [A Gentle Introduction to Conformal Prediction](https://arxiv.org/abs/2107.07511)
- [Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md)
- [Language Model Probability](../../15-Math-for-LLMs/05-Language-Model-Probability/notes.md)
- [SimpleQA: Measuring short-form factuality](https://arxiv.org/abs/2411.04368)
- [Concentration Inequalities](../../06-Probability-Theory/05-Concentration-Inequalities/notes.md)
- [Estimation Theory](../../07-Statistics/02-Estimation-Theory/notes.md)
- [Hypothesis Testing](../../07-Statistics/03-Hypothesis-Testing/notes.md)

