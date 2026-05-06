[Back to Curriculum](../../README.md) | [Previous: Preference Optimization RLHF and DPO](../02-Preference-Optimization-RLHF-and-DPO/notes.md) | [Next: Policy and Guardrails](../04-Policy-and-Guardrails/notes.md)

---

# Red Teaming and Safety Evaluations

> _"A safety case improves when failures are searched for deliberately."_

## Overview

Red teaming treats harmful behavior discovery as an adaptive search problem over
prompts, contexts, policies, and scoring rules.

Alignment and safety sit between model training and model deployment. The chapter
studies how desired behavior is specified, learned, stress-tested, constrained, and
improved through feedback.

This section is written in LaTeX Markdown. Inline mathematics uses `$...$`, and display
equations use `$$...$$`. The notes use math to expose the operational contract: which
behavior is optimized, which behavior is forbidden, and which evidence proves the system
changed.

## Prerequisites

- [Robustness and Distribution Shift](../../17-Evaluation-and-Reliability/03-Robustness-and-Distribution-Shift/notes.md)
- [Capability Benchmarks](../../17-Evaluation-and-Reliability/01-Capability-Benchmarks/notes.md)
- [Policy and Guardrails](../04-Policy-and-Guardrails/notes.md)
- [Contamination and Dedup Audits](../../16-LLM-Training-Data-Pipeline/05-Contamination-and-Dedup-Audits/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Executable demonstrations for red teaming and safety evaluations |
| [exercises.ipynb](exercises.ipynb) | Graded practice for red teaming and safety evaluations |

## Learning Objectives

After completing this section, you will be able to:

- Define the core mathematical objects in red teaming and safety evaluations
- Write alignment losses and decision rules in repo notation
- Separate data, objective, policy, and evaluation responsibilities
- Explain how safety failures become training or guardrail updates
- Identify reward hacking, over-refusal, proxy misspecification, and drift
- Design synthetic experiments that demonstrate alignment tradeoffs
- Connect human feedback to SFT, reward modeling, DPO, or runtime policies
- State the boundary with Chapter 17 evaluation and Chapter 19 production MLOps
- Read modern alignment papers with attention to assumptions and proxies
- Build exercises that test both formulas and system-level judgment

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 Red teaming searches before users find failures](#11-red-teaming-searches-before-users-find-failures)
  - [1.2 Safety failures are rare-event measurements](#12-safety-failures-are-rare-event-measurements)
  - [1.3 Attackers adapt to defenses](#13-attackers-adapt-to-defenses)
  - [1.4 Red-team findings become training data](#14-red-team-findings-become-training-data)
  - [1.5 Safety evaluation versus general robustness](#15-safety-evaluation-versus-general-robustness)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Harm taxonomy](#21-harm-taxonomy)
  - [2.2 Attack prompt](#22-attack-prompt)
  - [2.3 Target model](#23-target-model)
  - [2.4 Violation score](#24-violation-score)
  - [2.5 Attack success rate](#25-attack-success-rate)
- [3. Human Red Teaming](#3-human-red-teaming)
  - [3.1 Protocols](#31-protocols)
  - [3.2 Severity labels](#32-severity-labels)
  - [3.3 Inter-rater agreement](#33-inter-rater-agreement)
  - [3.4 Coverage gaps](#34-coverage-gaps)
  - [3.5 Triage and reproduction](#35-triage-and-reproduction)
- [4. Automated Red Teaming](#4-automated-red-teaming)
  - [4.1 Attacker models](#41-attacker-models)
  - [4.2 Prompt mutation](#42-prompt-mutation)
  - [4.3 Search objectives](#43-search-objectives)
  - [4.4 Adaptive attacks](#44-adaptive-attacks)
  - [4.5 Budgeted exploration](#45-budgeted-exploration)
- [5. Safety Benchmarks and Datasets](#5-safety-benchmarks-and-datasets)
  - [5.1 SafetyPrompts](#51-safetyprompts)
  - [5.2 HarmBench-style suites](#52-harmbench-style-suites)
  - [5.3 Jailbreak sets](#53-jailbreak-sets)
  - [5.4 Refusal measurement](#54-refusal-measurement)
  - [5.5 Over-refusal measurement](#55-over-refusal-measurement)
- [6. Measurement Math](#6-measurement-math)
  - [6.1 Attack success rate](#61-attack-success-rate)
  - [6.2 Confidence intervals](#62-confidence-intervals)
  - [6.3 False positive and false negative tradeoffs](#63-false-positive-and-false-negative-tradeoffs)
  - [6.4 Severity weighting](#64-severity-weighting)
  - [6.5 Paired safety comparisons](#65-paired-safety-comparisons)
- [7. From Findings to Fixes](#7-from-findings-to-fixes)
  - [7.1 Regression tests](#71-regression-tests)
  - [7.2 Safety SFT](#72-safety-sft)
  - [7.3 Preference data](#73-preference-data)
  - [7.4 Guardrail rules](#74-guardrail-rules)
  - [7.5 Non-overfitting discipline](#75-non-overfitting-discipline)
- [8. Common Mistakes](#8-common-mistakes)
- [9. Exercises](#9-exercises)
- [10. Why This Matters for AI](#10-why-this-matters-for-ai)
- [11. Conceptual Bridge](#11-conceptual-bridge)
- [References](#references)

---

## 1. Intuition

Intuition develops the part of red teaming and safety evaluations that the approved TOC
assigns to Chapter 18. The emphasis is alignment behavior, safety constraints, and
feedback loops, not generic fine-tuning or production monitoring.

### 1.1 Red teaming searches before users find failures

Red teaming searches before users find failures belongs in the canonical scope of red
teaming and safety evaluations. The object is the safety attack and evaluation protocol,
not merely a prompt trick or a moderation label. We study how data, losses, policies,
review processes, and safety constraints shape a model's conditional distribution over
responses.

A compact way to read this subsection is through the local symbol v(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\widehat{\operatorname{ASR}} = \frac{1}{n}\sum_{i=1}^{n}\mathbb{1}\{v(x_i,y_i)=1\}.
$$

For red teaming searches before users find failures, this formula should not be treated
as a slogan. It defines which tokens, responses, comparisons, or decisions receive
gradient or operational weight. A change in masking, sampling, rubric wording, or
thresholding changes the effective objective even if the model architecture is
unchanged.

| Alignment object | Mathematical question | Engineering question |
| --- | --- | --- |
| Data | Which examples define the target behavior? | Who wrote, filtered, and approved them? |
| Objective | Which terms receive weight? | Are masks, margins, and thresholds logged? |
| Policy | Which actions are allowed or disallowed? | Can reviewers reproduce the decision? |
| Evaluation | Which metric detects regression? | Is the test private, stable, and sliced? |
| Feedback | Which new evidence changes training? | How does it enter the next dataset version? |

Examples:
- Treat red teaming searches before users find failures as part of the model contract
and store the exact data version.
- Record the prompt template, role format, policy version, and decoder settings.
- Compare aligned and reference policies on both helpfulness and safety slices.
- Use held-out examples that were not used to tune refusals or rewards.
- Inspect failure cases before declaring the objective successful.

Non-examples:
- Calling a model aligned because it sounds polite on a few prompts.
- Training on refusals without measuring over-refusal on benign requests.
- Using a reward model as ground truth without calibration or adversarial checks.
- Shipping a guardrail threshold without measuring false positive and false negative
rates.
- Letting feedback logs change training without provenance or consent controls.

A useful implementation pattern is to separate policy, data, and measurement. The policy
says what behavior is desired. The data supplies examples, comparisons, attacks, or
feedback events. The measurement checks whether the updated system moved in the intended
direction without unacceptable regressions.

```text
policy text/rubric
      |
      v
training or guardrail data  ->  objective/threshold  ->  aligned system
      |                                                   |
      v                                                   v
audit metadata                                      held-out safety eval
```

Worked reasoning pattern for red teaming searches before users find failures:
1. Name the target behavior in plain language.
2. Write the mathematical variable that represents it.
3. Specify which examples or comparisons estimate it.
4. Choose the optimization loss or runtime decision rule.
5. Define the regression metric that would prove the change became worse.

Three details are especially easy to miss in alignment work. First, the user intent
distribution is not the same as the pretraining distribution. Second, safety labels are
not ordinary class labels; they encode policy judgments that can change by context.
Third, optimization pressure finds shortcuts, so every proxy must be monitored for
Goodhart-style failures.

| Failure pressure | Typical symptom | Mitigation |
| --- | --- | --- |
| Proxy reward | High reward but worse human judgment | Holdout preferences and adversarial review |
| Refusal shortcut | Safe but unhelpful responses | Measure benign refusal rate separately |
| Template overfit | Good on training chat format only | Evaluate alternate templates and languages |
| Policy ambiguity | Inconsistent labels | Adjudication and rubric revision |
| Feedback drift | New labels change old policy silently | Version policy, rubric, and dataset together |

AI connection: Red teaming searches before users find failures is part of the post-
training stack used by modern assistant systems. It links the base language model to
human intent, safety policy, and deployment constraints without pretending that a single
loss can capture all values. The goal is not perfect alignment by formula; it is a
repeatable loop where evidence, objectives, and safeguards improve together.

### 1.2 Safety failures are rare-event measurements

Safety failures are rare-event measurements belongs in the canonical scope of red
teaming and safety evaluations. The object is the safety attack and evaluation protocol,
not merely a prompt trick or a moderation label. We study how data, losses, policies,
review processes, and safety constraints shape a model's conditional distribution over
responses.

A compact way to read this subsection is through the local symbol v(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\widehat{\operatorname{ASR}} = \frac{1}{n}\sum_{i=1}^{n}\mathbb{1}\{v(x_i,y_i)=1\}.
$$

For safety failures are rare-event measurements, this formula should not be treated as a
slogan. It defines which tokens, responses, comparisons, or decisions receive gradient
or operational weight. A change in masking, sampling, rubric wording, or thresholding
changes the effective objective even if the model architecture is unchanged.

| Alignment object | Mathematical question | Engineering question |
| --- | --- | --- |
| Data | Which examples define the target behavior? | Who wrote, filtered, and approved them? |
| Objective | Which terms receive weight? | Are masks, margins, and thresholds logged? |
| Policy | Which actions are allowed or disallowed? | Can reviewers reproduce the decision? |
| Evaluation | Which metric detects regression? | Is the test private, stable, and sliced? |
| Feedback | Which new evidence changes training? | How does it enter the next dataset version? |

Examples:
- Treat safety failures are rare-event measurements as part of the model contract and
store the exact data version.
- Record the prompt template, role format, policy version, and decoder settings.
- Compare aligned and reference policies on both helpfulness and safety slices.
- Use held-out examples that were not used to tune refusals or rewards.
- Inspect failure cases before declaring the objective successful.

Non-examples:
- Calling a model aligned because it sounds polite on a few prompts.
- Training on refusals without measuring over-refusal on benign requests.
- Using a reward model as ground truth without calibration or adversarial checks.
- Shipping a guardrail threshold without measuring false positive and false negative
rates.
- Letting feedback logs change training without provenance or consent controls.

A useful implementation pattern is to separate policy, data, and measurement. The policy
says what behavior is desired. The data supplies examples, comparisons, attacks, or
feedback events. The measurement checks whether the updated system moved in the intended
direction without unacceptable regressions.

```text
policy text/rubric
      |
      v
training or guardrail data  ->  objective/threshold  ->  aligned system
      |                                                   |
      v                                                   v
audit metadata                                      held-out safety eval
```

Worked reasoning pattern for safety failures are rare-event measurements:
1. Name the target behavior in plain language.
2. Write the mathematical variable that represents it.
3. Specify which examples or comparisons estimate it.
4. Choose the optimization loss or runtime decision rule.
5. Define the regression metric that would prove the change became worse.

Three details are especially easy to miss in alignment work. First, the user intent
distribution is not the same as the pretraining distribution. Second, safety labels are
not ordinary class labels; they encode policy judgments that can change by context.
Third, optimization pressure finds shortcuts, so every proxy must be monitored for
Goodhart-style failures.

| Failure pressure | Typical symptom | Mitigation |
| --- | --- | --- |
| Proxy reward | High reward but worse human judgment | Holdout preferences and adversarial review |
| Refusal shortcut | Safe but unhelpful responses | Measure benign refusal rate separately |
| Template overfit | Good on training chat format only | Evaluate alternate templates and languages |
| Policy ambiguity | Inconsistent labels | Adjudication and rubric revision |
| Feedback drift | New labels change old policy silently | Version policy, rubric, and dataset together |

AI connection: Safety failures are rare-event measurements is part of the post-training
stack used by modern assistant systems. It links the base language model to human
intent, safety policy, and deployment constraints without pretending that a single loss
can capture all values. The goal is not perfect alignment by formula; it is a repeatable
loop where evidence, objectives, and safeguards improve together.

### 1.3 Attackers adapt to defenses

Attackers adapt to defenses belongs in the canonical scope of red teaming and safety
evaluations. The object is the safety attack and evaluation protocol, not merely a
prompt trick or a moderation label. We study how data, losses, policies, review
processes, and safety constraints shape a model's conditional distribution over
responses.

A compact way to read this subsection is through the local symbol v(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\widehat{\operatorname{ASR}} = \frac{1}{n}\sum_{i=1}^{n}\mathbb{1}\{v(x_i,y_i)=1\}.
$$

For attackers adapt to defenses, this formula should not be treated as a slogan. It
defines which tokens, responses, comparisons, or decisions receive gradient or
operational weight. A change in masking, sampling, rubric wording, or thresholding
changes the effective objective even if the model architecture is unchanged.

| Alignment object | Mathematical question | Engineering question |
| --- | --- | --- |
| Data | Which examples define the target behavior? | Who wrote, filtered, and approved them? |
| Objective | Which terms receive weight? | Are masks, margins, and thresholds logged? |
| Policy | Which actions are allowed or disallowed? | Can reviewers reproduce the decision? |
| Evaluation | Which metric detects regression? | Is the test private, stable, and sliced? |
| Feedback | Which new evidence changes training? | How does it enter the next dataset version? |

Examples:
- Treat attackers adapt to defenses as part of the model contract and store the exact
data version.
- Record the prompt template, role format, policy version, and decoder settings.
- Compare aligned and reference policies on both helpfulness and safety slices.
- Use held-out examples that were not used to tune refusals or rewards.
- Inspect failure cases before declaring the objective successful.

Non-examples:
- Calling a model aligned because it sounds polite on a few prompts.
- Training on refusals without measuring over-refusal on benign requests.
- Using a reward model as ground truth without calibration or adversarial checks.
- Shipping a guardrail threshold without measuring false positive and false negative
rates.
- Letting feedback logs change training without provenance or consent controls.

A useful implementation pattern is to separate policy, data, and measurement. The policy
says what behavior is desired. The data supplies examples, comparisons, attacks, or
feedback events. The measurement checks whether the updated system moved in the intended
direction without unacceptable regressions.

```text
policy text/rubric
      |
      v
training or guardrail data  ->  objective/threshold  ->  aligned system
      |                                                   |
      v                                                   v
audit metadata                                      held-out safety eval
```

Worked reasoning pattern for attackers adapt to defenses:
1. Name the target behavior in plain language.
2. Write the mathematical variable that represents it.
3. Specify which examples or comparisons estimate it.
4. Choose the optimization loss or runtime decision rule.
5. Define the regression metric that would prove the change became worse.

Three details are especially easy to miss in alignment work. First, the user intent
distribution is not the same as the pretraining distribution. Second, safety labels are
not ordinary class labels; they encode policy judgments that can change by context.
Third, optimization pressure finds shortcuts, so every proxy must be monitored for
Goodhart-style failures.

| Failure pressure | Typical symptom | Mitigation |
| --- | --- | --- |
| Proxy reward | High reward but worse human judgment | Holdout preferences and adversarial review |
| Refusal shortcut | Safe but unhelpful responses | Measure benign refusal rate separately |
| Template overfit | Good on training chat format only | Evaluate alternate templates and languages |
| Policy ambiguity | Inconsistent labels | Adjudication and rubric revision |
| Feedback drift | New labels change old policy silently | Version policy, rubric, and dataset together |

AI connection: Attackers adapt to defenses is part of the post-training stack used by
modern assistant systems. It links the base language model to human intent, safety
policy, and deployment constraints without pretending that a single loss can capture all
values. The goal is not perfect alignment by formula; it is a repeatable loop where
evidence, objectives, and safeguards improve together.

### 1.4 Red-team findings become training data

Red-team findings become training data belongs in the canonical scope of red teaming and
safety evaluations. The object is the safety attack and evaluation protocol, not merely
a prompt trick or a moderation label. We study how data, losses, policies, review
processes, and safety constraints shape a model's conditional distribution over
responses.

A compact way to read this subsection is through the local symbol v(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\widehat{\operatorname{ASR}} = \frac{1}{n}\sum_{i=1}^{n}\mathbb{1}\{v(x_i,y_i)=1\}.
$$

For red-team findings become training data, this formula should not be treated as a
slogan. It defines which tokens, responses, comparisons, or decisions receive gradient
or operational weight. A change in masking, sampling, rubric wording, or thresholding
changes the effective objective even if the model architecture is unchanged.

| Alignment object | Mathematical question | Engineering question |
| --- | --- | --- |
| Data | Which examples define the target behavior? | Who wrote, filtered, and approved them? |
| Objective | Which terms receive weight? | Are masks, margins, and thresholds logged? |
| Policy | Which actions are allowed or disallowed? | Can reviewers reproduce the decision? |
| Evaluation | Which metric detects regression? | Is the test private, stable, and sliced? |
| Feedback | Which new evidence changes training? | How does it enter the next dataset version? |

Examples:
- Treat red-team findings become training data as part of the model contract and store
the exact data version.
- Record the prompt template, role format, policy version, and decoder settings.
- Compare aligned and reference policies on both helpfulness and safety slices.
- Use held-out examples that were not used to tune refusals or rewards.
- Inspect failure cases before declaring the objective successful.

Non-examples:
- Calling a model aligned because it sounds polite on a few prompts.
- Training on refusals without measuring over-refusal on benign requests.
- Using a reward model as ground truth without calibration or adversarial checks.
- Shipping a guardrail threshold without measuring false positive and false negative
rates.
- Letting feedback logs change training without provenance or consent controls.

A useful implementation pattern is to separate policy, data, and measurement. The policy
says what behavior is desired. The data supplies examples, comparisons, attacks, or
feedback events. The measurement checks whether the updated system moved in the intended
direction without unacceptable regressions.

```text
policy text/rubric
      |
      v
training or guardrail data  ->  objective/threshold  ->  aligned system
      |                                                   |
      v                                                   v
audit metadata                                      held-out safety eval
```

Worked reasoning pattern for red-team findings become training data:
1. Name the target behavior in plain language.
2. Write the mathematical variable that represents it.
3. Specify which examples or comparisons estimate it.
4. Choose the optimization loss or runtime decision rule.
5. Define the regression metric that would prove the change became worse.

Three details are especially easy to miss in alignment work. First, the user intent
distribution is not the same as the pretraining distribution. Second, safety labels are
not ordinary class labels; they encode policy judgments that can change by context.
Third, optimization pressure finds shortcuts, so every proxy must be monitored for
Goodhart-style failures.

| Failure pressure | Typical symptom | Mitigation |
| --- | --- | --- |
| Proxy reward | High reward but worse human judgment | Holdout preferences and adversarial review |
| Refusal shortcut | Safe but unhelpful responses | Measure benign refusal rate separately |
| Template overfit | Good on training chat format only | Evaluate alternate templates and languages |
| Policy ambiguity | Inconsistent labels | Adjudication and rubric revision |
| Feedback drift | New labels change old policy silently | Version policy, rubric, and dataset together |

AI connection: Red-team findings become training data is part of the post-training stack
used by modern assistant systems. It links the base language model to human intent,
safety policy, and deployment constraints without pretending that a single loss can
capture all values. The goal is not perfect alignment by formula; it is a repeatable
loop where evidence, objectives, and safeguards improve together.

### 1.5 Safety evaluation versus general robustness

Safety evaluation versus general robustness belongs in the canonical scope of red
teaming and safety evaluations. The object is the safety attack and evaluation protocol,
not merely a prompt trick or a moderation label. We study how data, losses, policies,
review processes, and safety constraints shape a model's conditional distribution over
responses.

A compact way to read this subsection is through the local symbol v(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\widehat{\operatorname{ASR}} = \frac{1}{n}\sum_{i=1}^{n}\mathbb{1}\{v(x_i,y_i)=1\}.
$$

For safety evaluation versus general robustness, this formula should not be treated as a
slogan. It defines which tokens, responses, comparisons, or decisions receive gradient
or operational weight. A change in masking, sampling, rubric wording, or thresholding
changes the effective objective even if the model architecture is unchanged.

| Alignment object | Mathematical question | Engineering question |
| --- | --- | --- |
| Data | Which examples define the target behavior? | Who wrote, filtered, and approved them? |
| Objective | Which terms receive weight? | Are masks, margins, and thresholds logged? |
| Policy | Which actions are allowed or disallowed? | Can reviewers reproduce the decision? |
| Evaluation | Which metric detects regression? | Is the test private, stable, and sliced? |
| Feedback | Which new evidence changes training? | How does it enter the next dataset version? |

Examples:
- Treat safety evaluation versus general robustness as part of the model contract and
store the exact data version.
- Record the prompt template, role format, policy version, and decoder settings.
- Compare aligned and reference policies on both helpfulness and safety slices.
- Use held-out examples that were not used to tune refusals or rewards.
- Inspect failure cases before declaring the objective successful.

Non-examples:
- Calling a model aligned because it sounds polite on a few prompts.
- Training on refusals without measuring over-refusal on benign requests.
- Using a reward model as ground truth without calibration or adversarial checks.
- Shipping a guardrail threshold without measuring false positive and false negative
rates.
- Letting feedback logs change training without provenance or consent controls.

A useful implementation pattern is to separate policy, data, and measurement. The policy
says what behavior is desired. The data supplies examples, comparisons, attacks, or
feedback events. The measurement checks whether the updated system moved in the intended
direction without unacceptable regressions.

```text
policy text/rubric
      |
      v
training or guardrail data  ->  objective/threshold  ->  aligned system
      |                                                   |
      v                                                   v
audit metadata                                      held-out safety eval
```

Worked reasoning pattern for safety evaluation versus general robustness:
1. Name the target behavior in plain language.
2. Write the mathematical variable that represents it.
3. Specify which examples or comparisons estimate it.
4. Choose the optimization loss or runtime decision rule.
5. Define the regression metric that would prove the change became worse.

Three details are especially easy to miss in alignment work. First, the user intent
distribution is not the same as the pretraining distribution. Second, safety labels are
not ordinary class labels; they encode policy judgments that can change by context.
Third, optimization pressure finds shortcuts, so every proxy must be monitored for
Goodhart-style failures.

| Failure pressure | Typical symptom | Mitigation |
| --- | --- | --- |
| Proxy reward | High reward but worse human judgment | Holdout preferences and adversarial review |
| Refusal shortcut | Safe but unhelpful responses | Measure benign refusal rate separately |
| Template overfit | Good on training chat format only | Evaluate alternate templates and languages |
| Policy ambiguity | Inconsistent labels | Adjudication and rubric revision |
| Feedback drift | New labels change old policy silently | Version policy, rubric, and dataset together |

AI connection: Safety evaluation versus general robustness is part of the post-training
stack used by modern assistant systems. It links the base language model to human
intent, safety policy, and deployment constraints without pretending that a single loss
can capture all values. The goal is not perfect alignment by formula; it is a repeatable
loop where evidence, objectives, and safeguards improve together.

## 2. Formal Definitions

Formal Definitions develops the part of red teaming and safety evaluations that the
approved TOC assigns to Chapter 18. The emphasis is alignment behavior, safety
constraints, and feedback loops, not generic fine-tuning or production monitoring.

### 2.1 Harm taxonomy

Harm taxonomy belongs in the canonical scope of red teaming and safety evaluations. The
object is the safety attack and evaluation protocol, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol v(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\widehat{\operatorname{ASR}} = \frac{1}{n}\sum_{i=1}^{n}\mathbb{1}\{v(x_i,y_i)=1\}.
$$

For harm taxonomy, this formula should not be treated as a slogan. It defines which
tokens, responses, comparisons, or decisions receive gradient or operational weight. A
change in masking, sampling, rubric wording, or thresholding changes the effective
objective even if the model architecture is unchanged.

| Alignment object | Mathematical question | Engineering question |
| --- | --- | --- |
| Data | Which examples define the target behavior? | Who wrote, filtered, and approved them? |
| Objective | Which terms receive weight? | Are masks, margins, and thresholds logged? |
| Policy | Which actions are allowed or disallowed? | Can reviewers reproduce the decision? |
| Evaluation | Which metric detects regression? | Is the test private, stable, and sliced? |
| Feedback | Which new evidence changes training? | How does it enter the next dataset version? |

Examples:
- Treat harm taxonomy as part of the model contract and store the exact data version.
- Record the prompt template, role format, policy version, and decoder settings.
- Compare aligned and reference policies on both helpfulness and safety slices.
- Use held-out examples that were not used to tune refusals or rewards.
- Inspect failure cases before declaring the objective successful.

Non-examples:
- Calling a model aligned because it sounds polite on a few prompts.
- Training on refusals without measuring over-refusal on benign requests.
- Using a reward model as ground truth without calibration or adversarial checks.
- Shipping a guardrail threshold without measuring false positive and false negative
rates.
- Letting feedback logs change training without provenance or consent controls.

A useful implementation pattern is to separate policy, data, and measurement. The policy
says what behavior is desired. The data supplies examples, comparisons, attacks, or
feedback events. The measurement checks whether the updated system moved in the intended
direction without unacceptable regressions.

```text
policy text/rubric
      |
      v
training or guardrail data  ->  objective/threshold  ->  aligned system
      |                                                   |
      v                                                   v
audit metadata                                      held-out safety eval
```

Worked reasoning pattern for harm taxonomy:
1. Name the target behavior in plain language.
2. Write the mathematical variable that represents it.
3. Specify which examples or comparisons estimate it.
4. Choose the optimization loss or runtime decision rule.
5. Define the regression metric that would prove the change became worse.

Three details are especially easy to miss in alignment work. First, the user intent
distribution is not the same as the pretraining distribution. Second, safety labels are
not ordinary class labels; they encode policy judgments that can change by context.
Third, optimization pressure finds shortcuts, so every proxy must be monitored for
Goodhart-style failures.

| Failure pressure | Typical symptom | Mitigation |
| --- | --- | --- |
| Proxy reward | High reward but worse human judgment | Holdout preferences and adversarial review |
| Refusal shortcut | Safe but unhelpful responses | Measure benign refusal rate separately |
| Template overfit | Good on training chat format only | Evaluate alternate templates and languages |
| Policy ambiguity | Inconsistent labels | Adjudication and rubric revision |
| Feedback drift | New labels change old policy silently | Version policy, rubric, and dataset together |

AI connection: Harm taxonomy is part of the post-training stack used by modern assistant
systems. It links the base language model to human intent, safety policy, and deployment
constraints without pretending that a single loss can capture all values. The goal is
not perfect alignment by formula; it is a repeatable loop where evidence, objectives,
and safeguards improve together.

### 2.2 Attack prompt

Attack prompt belongs in the canonical scope of red teaming and safety evaluations. The
object is the safety attack and evaluation protocol, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol v(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\widehat{\operatorname{ASR}} = \frac{1}{n}\sum_{i=1}^{n}\mathbb{1}\{v(x_i,y_i)=1\}.
$$

For attack prompt, this formula should not be treated as a slogan. It defines which
tokens, responses, comparisons, or decisions receive gradient or operational weight. A
change in masking, sampling, rubric wording, or thresholding changes the effective
objective even if the model architecture is unchanged.

| Alignment object | Mathematical question | Engineering question |
| --- | --- | --- |
| Data | Which examples define the target behavior? | Who wrote, filtered, and approved them? |
| Objective | Which terms receive weight? | Are masks, margins, and thresholds logged? |
| Policy | Which actions are allowed or disallowed? | Can reviewers reproduce the decision? |
| Evaluation | Which metric detects regression? | Is the test private, stable, and sliced? |
| Feedback | Which new evidence changes training? | How does it enter the next dataset version? |

Examples:
- Treat attack prompt as part of the model contract and store the exact data version.
- Record the prompt template, role format, policy version, and decoder settings.
- Compare aligned and reference policies on both helpfulness and safety slices.
- Use held-out examples that were not used to tune refusals or rewards.
- Inspect failure cases before declaring the objective successful.

Non-examples:
- Calling a model aligned because it sounds polite on a few prompts.
- Training on refusals without measuring over-refusal on benign requests.
- Using a reward model as ground truth without calibration or adversarial checks.
- Shipping a guardrail threshold without measuring false positive and false negative
rates.
- Letting feedback logs change training without provenance or consent controls.

A useful implementation pattern is to separate policy, data, and measurement. The policy
says what behavior is desired. The data supplies examples, comparisons, attacks, or
feedback events. The measurement checks whether the updated system moved in the intended
direction without unacceptable regressions.

```text
policy text/rubric
      |
      v
training or guardrail data  ->  objective/threshold  ->  aligned system
      |                                                   |
      v                                                   v
audit metadata                                      held-out safety eval
```

Worked reasoning pattern for attack prompt:
1. Name the target behavior in plain language.
2. Write the mathematical variable that represents it.
3. Specify which examples or comparisons estimate it.
4. Choose the optimization loss or runtime decision rule.
5. Define the regression metric that would prove the change became worse.

Three details are especially easy to miss in alignment work. First, the user intent
distribution is not the same as the pretraining distribution. Second, safety labels are
not ordinary class labels; they encode policy judgments that can change by context.
Third, optimization pressure finds shortcuts, so every proxy must be monitored for
Goodhart-style failures.

| Failure pressure | Typical symptom | Mitigation |
| --- | --- | --- |
| Proxy reward | High reward but worse human judgment | Holdout preferences and adversarial review |
| Refusal shortcut | Safe but unhelpful responses | Measure benign refusal rate separately |
| Template overfit | Good on training chat format only | Evaluate alternate templates and languages |
| Policy ambiguity | Inconsistent labels | Adjudication and rubric revision |
| Feedback drift | New labels change old policy silently | Version policy, rubric, and dataset together |

AI connection: Attack prompt is part of the post-training stack used by modern assistant
systems. It links the base language model to human intent, safety policy, and deployment
constraints without pretending that a single loss can capture all values. The goal is
not perfect alignment by formula; it is a repeatable loop where evidence, objectives,
and safeguards improve together.

### 2.3 Target model

Target model belongs in the canonical scope of red teaming and safety evaluations. The
object is the safety attack and evaluation protocol, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol v(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\widehat{\operatorname{ASR}} = \frac{1}{n}\sum_{i=1}^{n}\mathbb{1}\{v(x_i,y_i)=1\}.
$$

For target model, this formula should not be treated as a slogan. It defines which
tokens, responses, comparisons, or decisions receive gradient or operational weight. A
change in masking, sampling, rubric wording, or thresholding changes the effective
objective even if the model architecture is unchanged.

| Alignment object | Mathematical question | Engineering question |
| --- | --- | --- |
| Data | Which examples define the target behavior? | Who wrote, filtered, and approved them? |
| Objective | Which terms receive weight? | Are masks, margins, and thresholds logged? |
| Policy | Which actions are allowed or disallowed? | Can reviewers reproduce the decision? |
| Evaluation | Which metric detects regression? | Is the test private, stable, and sliced? |
| Feedback | Which new evidence changes training? | How does it enter the next dataset version? |

Examples:
- Treat target model as part of the model contract and store the exact data version.
- Record the prompt template, role format, policy version, and decoder settings.
- Compare aligned and reference policies on both helpfulness and safety slices.
- Use held-out examples that were not used to tune refusals or rewards.
- Inspect failure cases before declaring the objective successful.

Non-examples:
- Calling a model aligned because it sounds polite on a few prompts.
- Training on refusals without measuring over-refusal on benign requests.
- Using a reward model as ground truth without calibration or adversarial checks.
- Shipping a guardrail threshold without measuring false positive and false negative
rates.
- Letting feedback logs change training without provenance or consent controls.

A useful implementation pattern is to separate policy, data, and measurement. The policy
says what behavior is desired. The data supplies examples, comparisons, attacks, or
feedback events. The measurement checks whether the updated system moved in the intended
direction without unacceptable regressions.

```text
policy text/rubric
      |
      v
training or guardrail data  ->  objective/threshold  ->  aligned system
      |                                                   |
      v                                                   v
audit metadata                                      held-out safety eval
```

Worked reasoning pattern for target model:
1. Name the target behavior in plain language.
2. Write the mathematical variable that represents it.
3. Specify which examples or comparisons estimate it.
4. Choose the optimization loss or runtime decision rule.
5. Define the regression metric that would prove the change became worse.

Three details are especially easy to miss in alignment work. First, the user intent
distribution is not the same as the pretraining distribution. Second, safety labels are
not ordinary class labels; they encode policy judgments that can change by context.
Third, optimization pressure finds shortcuts, so every proxy must be monitored for
Goodhart-style failures.

| Failure pressure | Typical symptom | Mitigation |
| --- | --- | --- |
| Proxy reward | High reward but worse human judgment | Holdout preferences and adversarial review |
| Refusal shortcut | Safe but unhelpful responses | Measure benign refusal rate separately |
| Template overfit | Good on training chat format only | Evaluate alternate templates and languages |
| Policy ambiguity | Inconsistent labels | Adjudication and rubric revision |
| Feedback drift | New labels change old policy silently | Version policy, rubric, and dataset together |

AI connection: Target model is part of the post-training stack used by modern assistant
systems. It links the base language model to human intent, safety policy, and deployment
constraints without pretending that a single loss can capture all values. The goal is
not perfect alignment by formula; it is a repeatable loop where evidence, objectives,
and safeguards improve together.

### 2.4 Violation score

Violation score belongs in the canonical scope of red teaming and safety evaluations.
The object is the safety attack and evaluation protocol, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol v(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\widehat{\operatorname{ASR}} = \frac{1}{n}\sum_{i=1}^{n}\mathbb{1}\{v(x_i,y_i)=1\}.
$$

For violation score, this formula should not be treated as a slogan. It defines which
tokens, responses, comparisons, or decisions receive gradient or operational weight. A
change in masking, sampling, rubric wording, or thresholding changes the effective
objective even if the model architecture is unchanged.

| Alignment object | Mathematical question | Engineering question |
| --- | --- | --- |
| Data | Which examples define the target behavior? | Who wrote, filtered, and approved them? |
| Objective | Which terms receive weight? | Are masks, margins, and thresholds logged? |
| Policy | Which actions are allowed or disallowed? | Can reviewers reproduce the decision? |
| Evaluation | Which metric detects regression? | Is the test private, stable, and sliced? |
| Feedback | Which new evidence changes training? | How does it enter the next dataset version? |

Examples:
- Treat violation score as part of the model contract and store the exact data version.
- Record the prompt template, role format, policy version, and decoder settings.
- Compare aligned and reference policies on both helpfulness and safety slices.
- Use held-out examples that were not used to tune refusals or rewards.
- Inspect failure cases before declaring the objective successful.

Non-examples:
- Calling a model aligned because it sounds polite on a few prompts.
- Training on refusals without measuring over-refusal on benign requests.
- Using a reward model as ground truth without calibration or adversarial checks.
- Shipping a guardrail threshold without measuring false positive and false negative
rates.
- Letting feedback logs change training without provenance or consent controls.

A useful implementation pattern is to separate policy, data, and measurement. The policy
says what behavior is desired. The data supplies examples, comparisons, attacks, or
feedback events. The measurement checks whether the updated system moved in the intended
direction without unacceptable regressions.

```text
policy text/rubric
      |
      v
training or guardrail data  ->  objective/threshold  ->  aligned system
      |                                                   |
      v                                                   v
audit metadata                                      held-out safety eval
```

Worked reasoning pattern for violation score:
1. Name the target behavior in plain language.
2. Write the mathematical variable that represents it.
3. Specify which examples or comparisons estimate it.
4. Choose the optimization loss or runtime decision rule.
5. Define the regression metric that would prove the change became worse.

Three details are especially easy to miss in alignment work. First, the user intent
distribution is not the same as the pretraining distribution. Second, safety labels are
not ordinary class labels; they encode policy judgments that can change by context.
Third, optimization pressure finds shortcuts, so every proxy must be monitored for
Goodhart-style failures.

| Failure pressure | Typical symptom | Mitigation |
| --- | --- | --- |
| Proxy reward | High reward but worse human judgment | Holdout preferences and adversarial review |
| Refusal shortcut | Safe but unhelpful responses | Measure benign refusal rate separately |
| Template overfit | Good on training chat format only | Evaluate alternate templates and languages |
| Policy ambiguity | Inconsistent labels | Adjudication and rubric revision |
| Feedback drift | New labels change old policy silently | Version policy, rubric, and dataset together |

AI connection: Violation score is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 2.5 Attack success rate

Attack success rate belongs in the canonical scope of red teaming and safety
evaluations. The object is the safety attack and evaluation protocol, not merely a
prompt trick or a moderation label. We study how data, losses, policies, review
processes, and safety constraints shape a model's conditional distribution over
responses.

A compact way to read this subsection is through the local symbol v(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\widehat{\operatorname{ASR}} = \frac{1}{n}\sum_{i=1}^{n}\mathbb{1}\{v(x_i,y_i)=1\}.
$$

For attack success rate, this formula should not be treated as a slogan. It defines
which tokens, responses, comparisons, or decisions receive gradient or operational
weight. A change in masking, sampling, rubric wording, or thresholding changes the
effective objective even if the model architecture is unchanged.

| Alignment object | Mathematical question | Engineering question |
| --- | --- | --- |
| Data | Which examples define the target behavior? | Who wrote, filtered, and approved them? |
| Objective | Which terms receive weight? | Are masks, margins, and thresholds logged? |
| Policy | Which actions are allowed or disallowed? | Can reviewers reproduce the decision? |
| Evaluation | Which metric detects regression? | Is the test private, stable, and sliced? |
| Feedback | Which new evidence changes training? | How does it enter the next dataset version? |

Examples:
- Treat attack success rate as part of the model contract and store the exact data
version.
- Record the prompt template, role format, policy version, and decoder settings.
- Compare aligned and reference policies on both helpfulness and safety slices.
- Use held-out examples that were not used to tune refusals or rewards.
- Inspect failure cases before declaring the objective successful.

Non-examples:
- Calling a model aligned because it sounds polite on a few prompts.
- Training on refusals without measuring over-refusal on benign requests.
- Using a reward model as ground truth without calibration or adversarial checks.
- Shipping a guardrail threshold without measuring false positive and false negative
rates.
- Letting feedback logs change training without provenance or consent controls.

A useful implementation pattern is to separate policy, data, and measurement. The policy
says what behavior is desired. The data supplies examples, comparisons, attacks, or
feedback events. The measurement checks whether the updated system moved in the intended
direction without unacceptable regressions.

```text
policy text/rubric
      |
      v
training or guardrail data  ->  objective/threshold  ->  aligned system
      |                                                   |
      v                                                   v
audit metadata                                      held-out safety eval
```

Worked reasoning pattern for attack success rate:
1. Name the target behavior in plain language.
2. Write the mathematical variable that represents it.
3. Specify which examples or comparisons estimate it.
4. Choose the optimization loss or runtime decision rule.
5. Define the regression metric that would prove the change became worse.

Three details are especially easy to miss in alignment work. First, the user intent
distribution is not the same as the pretraining distribution. Second, safety labels are
not ordinary class labels; they encode policy judgments that can change by context.
Third, optimization pressure finds shortcuts, so every proxy must be monitored for
Goodhart-style failures.

| Failure pressure | Typical symptom | Mitigation |
| --- | --- | --- |
| Proxy reward | High reward but worse human judgment | Holdout preferences and adversarial review |
| Refusal shortcut | Safe but unhelpful responses | Measure benign refusal rate separately |
| Template overfit | Good on training chat format only | Evaluate alternate templates and languages |
| Policy ambiguity | Inconsistent labels | Adjudication and rubric revision |
| Feedback drift | New labels change old policy silently | Version policy, rubric, and dataset together |

AI connection: Attack success rate is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

## 3. Human Red Teaming

Human Red Teaming develops the part of red teaming and safety evaluations that the
approved TOC assigns to Chapter 18. The emphasis is alignment behavior, safety
constraints, and feedback loops, not generic fine-tuning or production monitoring.

### 3.1 Protocols

Protocols belongs in the canonical scope of red teaming and safety evaluations. The
object is the safety attack and evaluation protocol, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol v(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\widehat{\operatorname{ASR}} = \frac{1}{n}\sum_{i=1}^{n}\mathbb{1}\{v(x_i,y_i)=1\}.
$$

For protocols, this formula should not be treated as a slogan. It defines which tokens,
responses, comparisons, or decisions receive gradient or operational weight. A change in
masking, sampling, rubric wording, or thresholding changes the effective objective even
if the model architecture is unchanged.

| Alignment object | Mathematical question | Engineering question |
| --- | --- | --- |
| Data | Which examples define the target behavior? | Who wrote, filtered, and approved them? |
| Objective | Which terms receive weight? | Are masks, margins, and thresholds logged? |
| Policy | Which actions are allowed or disallowed? | Can reviewers reproduce the decision? |
| Evaluation | Which metric detects regression? | Is the test private, stable, and sliced? |
| Feedback | Which new evidence changes training? | How does it enter the next dataset version? |

Examples:
- Treat protocols as part of the model contract and store the exact data version.
- Record the prompt template, role format, policy version, and decoder settings.
- Compare aligned and reference policies on both helpfulness and safety slices.
- Use held-out examples that were not used to tune refusals or rewards.
- Inspect failure cases before declaring the objective successful.

Non-examples:
- Calling a model aligned because it sounds polite on a few prompts.
- Training on refusals without measuring over-refusal on benign requests.
- Using a reward model as ground truth without calibration or adversarial checks.
- Shipping a guardrail threshold without measuring false positive and false negative
rates.
- Letting feedback logs change training without provenance or consent controls.

A useful implementation pattern is to separate policy, data, and measurement. The policy
says what behavior is desired. The data supplies examples, comparisons, attacks, or
feedback events. The measurement checks whether the updated system moved in the intended
direction without unacceptable regressions.

```text
policy text/rubric
      |
      v
training or guardrail data  ->  objective/threshold  ->  aligned system
      |                                                   |
      v                                                   v
audit metadata                                      held-out safety eval
```

Worked reasoning pattern for protocols:
1. Name the target behavior in plain language.
2. Write the mathematical variable that represents it.
3. Specify which examples or comparisons estimate it.
4. Choose the optimization loss or runtime decision rule.
5. Define the regression metric that would prove the change became worse.

Three details are especially easy to miss in alignment work. First, the user intent
distribution is not the same as the pretraining distribution. Second, safety labels are
not ordinary class labels; they encode policy judgments that can change by context.
Third, optimization pressure finds shortcuts, so every proxy must be monitored for
Goodhart-style failures.

| Failure pressure | Typical symptom | Mitigation |
| --- | --- | --- |
| Proxy reward | High reward but worse human judgment | Holdout preferences and adversarial review |
| Refusal shortcut | Safe but unhelpful responses | Measure benign refusal rate separately |
| Template overfit | Good on training chat format only | Evaluate alternate templates and languages |
| Policy ambiguity | Inconsistent labels | Adjudication and rubric revision |
| Feedback drift | New labels change old policy silently | Version policy, rubric, and dataset together |

AI connection: Protocols is part of the post-training stack used by modern assistant
systems. It links the base language model to human intent, safety policy, and deployment
constraints without pretending that a single loss can capture all values. The goal is
not perfect alignment by formula; it is a repeatable loop where evidence, objectives,
and safeguards improve together.

### 3.2 Severity labels

Severity labels belongs in the canonical scope of red teaming and safety evaluations.
The object is the safety attack and evaluation protocol, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol v(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\widehat{\operatorname{ASR}} = \frac{1}{n}\sum_{i=1}^{n}\mathbb{1}\{v(x_i,y_i)=1\}.
$$

For severity labels, this formula should not be treated as a slogan. It defines which
tokens, responses, comparisons, or decisions receive gradient or operational weight. A
change in masking, sampling, rubric wording, or thresholding changes the effective
objective even if the model architecture is unchanged.

| Alignment object | Mathematical question | Engineering question |
| --- | --- | --- |
| Data | Which examples define the target behavior? | Who wrote, filtered, and approved them? |
| Objective | Which terms receive weight? | Are masks, margins, and thresholds logged? |
| Policy | Which actions are allowed or disallowed? | Can reviewers reproduce the decision? |
| Evaluation | Which metric detects regression? | Is the test private, stable, and sliced? |
| Feedback | Which new evidence changes training? | How does it enter the next dataset version? |

Examples:
- Treat severity labels as part of the model contract and store the exact data version.
- Record the prompt template, role format, policy version, and decoder settings.
- Compare aligned and reference policies on both helpfulness and safety slices.
- Use held-out examples that were not used to tune refusals or rewards.
- Inspect failure cases before declaring the objective successful.

Non-examples:
- Calling a model aligned because it sounds polite on a few prompts.
- Training on refusals without measuring over-refusal on benign requests.
- Using a reward model as ground truth without calibration or adversarial checks.
- Shipping a guardrail threshold without measuring false positive and false negative
rates.
- Letting feedback logs change training without provenance or consent controls.

A useful implementation pattern is to separate policy, data, and measurement. The policy
says what behavior is desired. The data supplies examples, comparisons, attacks, or
feedback events. The measurement checks whether the updated system moved in the intended
direction without unacceptable regressions.

```text
policy text/rubric
      |
      v
training or guardrail data  ->  objective/threshold  ->  aligned system
      |                                                   |
      v                                                   v
audit metadata                                      held-out safety eval
```

Worked reasoning pattern for severity labels:
1. Name the target behavior in plain language.
2. Write the mathematical variable that represents it.
3. Specify which examples or comparisons estimate it.
4. Choose the optimization loss or runtime decision rule.
5. Define the regression metric that would prove the change became worse.

Three details are especially easy to miss in alignment work. First, the user intent
distribution is not the same as the pretraining distribution. Second, safety labels are
not ordinary class labels; they encode policy judgments that can change by context.
Third, optimization pressure finds shortcuts, so every proxy must be monitored for
Goodhart-style failures.

| Failure pressure | Typical symptom | Mitigation |
| --- | --- | --- |
| Proxy reward | High reward but worse human judgment | Holdout preferences and adversarial review |
| Refusal shortcut | Safe but unhelpful responses | Measure benign refusal rate separately |
| Template overfit | Good on training chat format only | Evaluate alternate templates and languages |
| Policy ambiguity | Inconsistent labels | Adjudication and rubric revision |
| Feedback drift | New labels change old policy silently | Version policy, rubric, and dataset together |

AI connection: Severity labels is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 3.3 Inter-rater agreement

Inter-rater agreement belongs in the canonical scope of red teaming and safety
evaluations. The object is the safety attack and evaluation protocol, not merely a
prompt trick or a moderation label. We study how data, losses, policies, review
processes, and safety constraints shape a model's conditional distribution over
responses.

A compact way to read this subsection is through the local symbol v(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\widehat{\operatorname{ASR}} = \frac{1}{n}\sum_{i=1}^{n}\mathbb{1}\{v(x_i,y_i)=1\}.
$$

For inter-rater agreement, this formula should not be treated as a slogan. It defines
which tokens, responses, comparisons, or decisions receive gradient or operational
weight. A change in masking, sampling, rubric wording, or thresholding changes the
effective objective even if the model architecture is unchanged.

| Alignment object | Mathematical question | Engineering question |
| --- | --- | --- |
| Data | Which examples define the target behavior? | Who wrote, filtered, and approved them? |
| Objective | Which terms receive weight? | Are masks, margins, and thresholds logged? |
| Policy | Which actions are allowed or disallowed? | Can reviewers reproduce the decision? |
| Evaluation | Which metric detects regression? | Is the test private, stable, and sliced? |
| Feedback | Which new evidence changes training? | How does it enter the next dataset version? |

Examples:
- Treat inter-rater agreement as part of the model contract and store the exact data
version.
- Record the prompt template, role format, policy version, and decoder settings.
- Compare aligned and reference policies on both helpfulness and safety slices.
- Use held-out examples that were not used to tune refusals or rewards.
- Inspect failure cases before declaring the objective successful.

Non-examples:
- Calling a model aligned because it sounds polite on a few prompts.
- Training on refusals without measuring over-refusal on benign requests.
- Using a reward model as ground truth without calibration or adversarial checks.
- Shipping a guardrail threshold without measuring false positive and false negative
rates.
- Letting feedback logs change training without provenance or consent controls.

A useful implementation pattern is to separate policy, data, and measurement. The policy
says what behavior is desired. The data supplies examples, comparisons, attacks, or
feedback events. The measurement checks whether the updated system moved in the intended
direction without unacceptable regressions.

```text
policy text/rubric
      |
      v
training or guardrail data  ->  objective/threshold  ->  aligned system
      |                                                   |
      v                                                   v
audit metadata                                      held-out safety eval
```

Worked reasoning pattern for inter-rater agreement:
1. Name the target behavior in plain language.
2. Write the mathematical variable that represents it.
3. Specify which examples or comparisons estimate it.
4. Choose the optimization loss or runtime decision rule.
5. Define the regression metric that would prove the change became worse.

Three details are especially easy to miss in alignment work. First, the user intent
distribution is not the same as the pretraining distribution. Second, safety labels are
not ordinary class labels; they encode policy judgments that can change by context.
Third, optimization pressure finds shortcuts, so every proxy must be monitored for
Goodhart-style failures.

| Failure pressure | Typical symptom | Mitigation |
| --- | --- | --- |
| Proxy reward | High reward but worse human judgment | Holdout preferences and adversarial review |
| Refusal shortcut | Safe but unhelpful responses | Measure benign refusal rate separately |
| Template overfit | Good on training chat format only | Evaluate alternate templates and languages |
| Policy ambiguity | Inconsistent labels | Adjudication and rubric revision |
| Feedback drift | New labels change old policy silently | Version policy, rubric, and dataset together |

AI connection: Inter-rater agreement is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 3.4 Coverage gaps

Coverage gaps belongs in the canonical scope of red teaming and safety evaluations. The
object is the safety attack and evaluation protocol, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol v(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\widehat{\operatorname{ASR}} = \frac{1}{n}\sum_{i=1}^{n}\mathbb{1}\{v(x_i,y_i)=1\}.
$$

For coverage gaps, this formula should not be treated as a slogan. It defines which
tokens, responses, comparisons, or decisions receive gradient or operational weight. A
change in masking, sampling, rubric wording, or thresholding changes the effective
objective even if the model architecture is unchanged.

| Alignment object | Mathematical question | Engineering question |
| --- | --- | --- |
| Data | Which examples define the target behavior? | Who wrote, filtered, and approved them? |
| Objective | Which terms receive weight? | Are masks, margins, and thresholds logged? |
| Policy | Which actions are allowed or disallowed? | Can reviewers reproduce the decision? |
| Evaluation | Which metric detects regression? | Is the test private, stable, and sliced? |
| Feedback | Which new evidence changes training? | How does it enter the next dataset version? |

Examples:
- Treat coverage gaps as part of the model contract and store the exact data version.
- Record the prompt template, role format, policy version, and decoder settings.
- Compare aligned and reference policies on both helpfulness and safety slices.
- Use held-out examples that were not used to tune refusals or rewards.
- Inspect failure cases before declaring the objective successful.

Non-examples:
- Calling a model aligned because it sounds polite on a few prompts.
- Training on refusals without measuring over-refusal on benign requests.
- Using a reward model as ground truth without calibration or adversarial checks.
- Shipping a guardrail threshold without measuring false positive and false negative
rates.
- Letting feedback logs change training without provenance or consent controls.

A useful implementation pattern is to separate policy, data, and measurement. The policy
says what behavior is desired. The data supplies examples, comparisons, attacks, or
feedback events. The measurement checks whether the updated system moved in the intended
direction without unacceptable regressions.

```text
policy text/rubric
      |
      v
training or guardrail data  ->  objective/threshold  ->  aligned system
      |                                                   |
      v                                                   v
audit metadata                                      held-out safety eval
```

Worked reasoning pattern for coverage gaps:
1. Name the target behavior in plain language.
2. Write the mathematical variable that represents it.
3. Specify which examples or comparisons estimate it.
4. Choose the optimization loss or runtime decision rule.
5. Define the regression metric that would prove the change became worse.

Three details are especially easy to miss in alignment work. First, the user intent
distribution is not the same as the pretraining distribution. Second, safety labels are
not ordinary class labels; they encode policy judgments that can change by context.
Third, optimization pressure finds shortcuts, so every proxy must be monitored for
Goodhart-style failures.

| Failure pressure | Typical symptom | Mitigation |
| --- | --- | --- |
| Proxy reward | High reward but worse human judgment | Holdout preferences and adversarial review |
| Refusal shortcut | Safe but unhelpful responses | Measure benign refusal rate separately |
| Template overfit | Good on training chat format only | Evaluate alternate templates and languages |
| Policy ambiguity | Inconsistent labels | Adjudication and rubric revision |
| Feedback drift | New labels change old policy silently | Version policy, rubric, and dataset together |

AI connection: Coverage gaps is part of the post-training stack used by modern assistant
systems. It links the base language model to human intent, safety policy, and deployment
constraints without pretending that a single loss can capture all values. The goal is
not perfect alignment by formula; it is a repeatable loop where evidence, objectives,
and safeguards improve together.

### 3.5 Triage and reproduction

Triage and reproduction belongs in the canonical scope of red teaming and safety
evaluations. The object is the safety attack and evaluation protocol, not merely a
prompt trick or a moderation label. We study how data, losses, policies, review
processes, and safety constraints shape a model's conditional distribution over
responses.

A compact way to read this subsection is through the local symbol v(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\widehat{\operatorname{ASR}} = \frac{1}{n}\sum_{i=1}^{n}\mathbb{1}\{v(x_i,y_i)=1\}.
$$

For triage and reproduction, this formula should not be treated as a slogan. It defines
which tokens, responses, comparisons, or decisions receive gradient or operational
weight. A change in masking, sampling, rubric wording, or thresholding changes the
effective objective even if the model architecture is unchanged.

| Alignment object | Mathematical question | Engineering question |
| --- | --- | --- |
| Data | Which examples define the target behavior? | Who wrote, filtered, and approved them? |
| Objective | Which terms receive weight? | Are masks, margins, and thresholds logged? |
| Policy | Which actions are allowed or disallowed? | Can reviewers reproduce the decision? |
| Evaluation | Which metric detects regression? | Is the test private, stable, and sliced? |
| Feedback | Which new evidence changes training? | How does it enter the next dataset version? |

Examples:
- Treat triage and reproduction as part of the model contract and store the exact data
version.
- Record the prompt template, role format, policy version, and decoder settings.
- Compare aligned and reference policies on both helpfulness and safety slices.
- Use held-out examples that were not used to tune refusals or rewards.
- Inspect failure cases before declaring the objective successful.

Non-examples:
- Calling a model aligned because it sounds polite on a few prompts.
- Training on refusals without measuring over-refusal on benign requests.
- Using a reward model as ground truth without calibration or adversarial checks.
- Shipping a guardrail threshold without measuring false positive and false negative
rates.
- Letting feedback logs change training without provenance or consent controls.

A useful implementation pattern is to separate policy, data, and measurement. The policy
says what behavior is desired. The data supplies examples, comparisons, attacks, or
feedback events. The measurement checks whether the updated system moved in the intended
direction without unacceptable regressions.

```text
policy text/rubric
      |
      v
training or guardrail data  ->  objective/threshold  ->  aligned system
      |                                                   |
      v                                                   v
audit metadata                                      held-out safety eval
```

Worked reasoning pattern for triage and reproduction:
1. Name the target behavior in plain language.
2. Write the mathematical variable that represents it.
3. Specify which examples or comparisons estimate it.
4. Choose the optimization loss or runtime decision rule.
5. Define the regression metric that would prove the change became worse.

Three details are especially easy to miss in alignment work. First, the user intent
distribution is not the same as the pretraining distribution. Second, safety labels are
not ordinary class labels; they encode policy judgments that can change by context.
Third, optimization pressure finds shortcuts, so every proxy must be monitored for
Goodhart-style failures.

| Failure pressure | Typical symptom | Mitigation |
| --- | --- | --- |
| Proxy reward | High reward but worse human judgment | Holdout preferences and adversarial review |
| Refusal shortcut | Safe but unhelpful responses | Measure benign refusal rate separately |
| Template overfit | Good on training chat format only | Evaluate alternate templates and languages |
| Policy ambiguity | Inconsistent labels | Adjudication and rubric revision |
| Feedback drift | New labels change old policy silently | Version policy, rubric, and dataset together |

AI connection: Triage and reproduction is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

## 4. Automated Red Teaming

Automated Red Teaming develops the part of red teaming and safety evaluations that the
approved TOC assigns to Chapter 18. The emphasis is alignment behavior, safety
constraints, and feedback loops, not generic fine-tuning or production monitoring.

### 4.1 Attacker models

Attacker models belongs in the canonical scope of red teaming and safety evaluations.
The object is the safety attack and evaluation protocol, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol v(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\widehat{\operatorname{ASR}} = \frac{1}{n}\sum_{i=1}^{n}\mathbb{1}\{v(x_i,y_i)=1\}.
$$

For attacker models, this formula should not be treated as a slogan. It defines which
tokens, responses, comparisons, or decisions receive gradient or operational weight. A
change in masking, sampling, rubric wording, or thresholding changes the effective
objective even if the model architecture is unchanged.

| Alignment object | Mathematical question | Engineering question |
| --- | --- | --- |
| Data | Which examples define the target behavior? | Who wrote, filtered, and approved them? |
| Objective | Which terms receive weight? | Are masks, margins, and thresholds logged? |
| Policy | Which actions are allowed or disallowed? | Can reviewers reproduce the decision? |
| Evaluation | Which metric detects regression? | Is the test private, stable, and sliced? |
| Feedback | Which new evidence changes training? | How does it enter the next dataset version? |

Examples:
- Treat attacker models as part of the model contract and store the exact data version.
- Record the prompt template, role format, policy version, and decoder settings.
- Compare aligned and reference policies on both helpfulness and safety slices.
- Use held-out examples that were not used to tune refusals or rewards.
- Inspect failure cases before declaring the objective successful.

Non-examples:
- Calling a model aligned because it sounds polite on a few prompts.
- Training on refusals without measuring over-refusal on benign requests.
- Using a reward model as ground truth without calibration or adversarial checks.
- Shipping a guardrail threshold without measuring false positive and false negative
rates.
- Letting feedback logs change training without provenance or consent controls.

A useful implementation pattern is to separate policy, data, and measurement. The policy
says what behavior is desired. The data supplies examples, comparisons, attacks, or
feedback events. The measurement checks whether the updated system moved in the intended
direction without unacceptable regressions.

```text
policy text/rubric
      |
      v
training or guardrail data  ->  objective/threshold  ->  aligned system
      |                                                   |
      v                                                   v
audit metadata                                      held-out safety eval
```

Worked reasoning pattern for attacker models:
1. Name the target behavior in plain language.
2. Write the mathematical variable that represents it.
3. Specify which examples or comparisons estimate it.
4. Choose the optimization loss or runtime decision rule.
5. Define the regression metric that would prove the change became worse.

Three details are especially easy to miss in alignment work. First, the user intent
distribution is not the same as the pretraining distribution. Second, safety labels are
not ordinary class labels; they encode policy judgments that can change by context.
Third, optimization pressure finds shortcuts, so every proxy must be monitored for
Goodhart-style failures.

| Failure pressure | Typical symptom | Mitigation |
| --- | --- | --- |
| Proxy reward | High reward but worse human judgment | Holdout preferences and adversarial review |
| Refusal shortcut | Safe but unhelpful responses | Measure benign refusal rate separately |
| Template overfit | Good on training chat format only | Evaluate alternate templates and languages |
| Policy ambiguity | Inconsistent labels | Adjudication and rubric revision |
| Feedback drift | New labels change old policy silently | Version policy, rubric, and dataset together |

AI connection: Attacker models is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 4.2 Prompt mutation

Prompt mutation belongs in the canonical scope of red teaming and safety evaluations.
The object is the safety attack and evaluation protocol, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol v(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\widehat{\operatorname{ASR}} = \frac{1}{n}\sum_{i=1}^{n}\mathbb{1}\{v(x_i,y_i)=1\}.
$$

For prompt mutation, this formula should not be treated as a slogan. It defines which
tokens, responses, comparisons, or decisions receive gradient or operational weight. A
change in masking, sampling, rubric wording, or thresholding changes the effective
objective even if the model architecture is unchanged.

| Alignment object | Mathematical question | Engineering question |
| --- | --- | --- |
| Data | Which examples define the target behavior? | Who wrote, filtered, and approved them? |
| Objective | Which terms receive weight? | Are masks, margins, and thresholds logged? |
| Policy | Which actions are allowed or disallowed? | Can reviewers reproduce the decision? |
| Evaluation | Which metric detects regression? | Is the test private, stable, and sliced? |
| Feedback | Which new evidence changes training? | How does it enter the next dataset version? |

Examples:
- Treat prompt mutation as part of the model contract and store the exact data version.
- Record the prompt template, role format, policy version, and decoder settings.
- Compare aligned and reference policies on both helpfulness and safety slices.
- Use held-out examples that were not used to tune refusals or rewards.
- Inspect failure cases before declaring the objective successful.

Non-examples:
- Calling a model aligned because it sounds polite on a few prompts.
- Training on refusals without measuring over-refusal on benign requests.
- Using a reward model as ground truth without calibration or adversarial checks.
- Shipping a guardrail threshold without measuring false positive and false negative
rates.
- Letting feedback logs change training without provenance or consent controls.

A useful implementation pattern is to separate policy, data, and measurement. The policy
says what behavior is desired. The data supplies examples, comparisons, attacks, or
feedback events. The measurement checks whether the updated system moved in the intended
direction without unacceptable regressions.

```text
policy text/rubric
      |
      v
training or guardrail data  ->  objective/threshold  ->  aligned system
      |                                                   |
      v                                                   v
audit metadata                                      held-out safety eval
```

Worked reasoning pattern for prompt mutation:
1. Name the target behavior in plain language.
2. Write the mathematical variable that represents it.
3. Specify which examples or comparisons estimate it.
4. Choose the optimization loss or runtime decision rule.
5. Define the regression metric that would prove the change became worse.

Three details are especially easy to miss in alignment work. First, the user intent
distribution is not the same as the pretraining distribution. Second, safety labels are
not ordinary class labels; they encode policy judgments that can change by context.
Third, optimization pressure finds shortcuts, so every proxy must be monitored for
Goodhart-style failures.

| Failure pressure | Typical symptom | Mitigation |
| --- | --- | --- |
| Proxy reward | High reward but worse human judgment | Holdout preferences and adversarial review |
| Refusal shortcut | Safe but unhelpful responses | Measure benign refusal rate separately |
| Template overfit | Good on training chat format only | Evaluate alternate templates and languages |
| Policy ambiguity | Inconsistent labels | Adjudication and rubric revision |
| Feedback drift | New labels change old policy silently | Version policy, rubric, and dataset together |

AI connection: Prompt mutation is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 4.3 Search objectives

Search objectives belongs in the canonical scope of red teaming and safety evaluations.
The object is the safety attack and evaluation protocol, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol v(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\widehat{\operatorname{ASR}} = \frac{1}{n}\sum_{i=1}^{n}\mathbb{1}\{v(x_i,y_i)=1\}.
$$

For search objectives, this formula should not be treated as a slogan. It defines which
tokens, responses, comparisons, or decisions receive gradient or operational weight. A
change in masking, sampling, rubric wording, or thresholding changes the effective
objective even if the model architecture is unchanged.

| Alignment object | Mathematical question | Engineering question |
| --- | --- | --- |
| Data | Which examples define the target behavior? | Who wrote, filtered, and approved them? |
| Objective | Which terms receive weight? | Are masks, margins, and thresholds logged? |
| Policy | Which actions are allowed or disallowed? | Can reviewers reproduce the decision? |
| Evaluation | Which metric detects regression? | Is the test private, stable, and sliced? |
| Feedback | Which new evidence changes training? | How does it enter the next dataset version? |

Examples:
- Treat search objectives as part of the model contract and store the exact data
version.
- Record the prompt template, role format, policy version, and decoder settings.
- Compare aligned and reference policies on both helpfulness and safety slices.
- Use held-out examples that were not used to tune refusals or rewards.
- Inspect failure cases before declaring the objective successful.

Non-examples:
- Calling a model aligned because it sounds polite on a few prompts.
- Training on refusals without measuring over-refusal on benign requests.
- Using a reward model as ground truth without calibration or adversarial checks.
- Shipping a guardrail threshold without measuring false positive and false negative
rates.
- Letting feedback logs change training without provenance or consent controls.

A useful implementation pattern is to separate policy, data, and measurement. The policy
says what behavior is desired. The data supplies examples, comparisons, attacks, or
feedback events. The measurement checks whether the updated system moved in the intended
direction without unacceptable regressions.

```text
policy text/rubric
      |
      v
training or guardrail data  ->  objective/threshold  ->  aligned system
      |                                                   |
      v                                                   v
audit metadata                                      held-out safety eval
```

Worked reasoning pattern for search objectives:
1. Name the target behavior in plain language.
2. Write the mathematical variable that represents it.
3. Specify which examples or comparisons estimate it.
4. Choose the optimization loss or runtime decision rule.
5. Define the regression metric that would prove the change became worse.

Three details are especially easy to miss in alignment work. First, the user intent
distribution is not the same as the pretraining distribution. Second, safety labels are
not ordinary class labels; they encode policy judgments that can change by context.
Third, optimization pressure finds shortcuts, so every proxy must be monitored for
Goodhart-style failures.

| Failure pressure | Typical symptom | Mitigation |
| --- | --- | --- |
| Proxy reward | High reward but worse human judgment | Holdout preferences and adversarial review |
| Refusal shortcut | Safe but unhelpful responses | Measure benign refusal rate separately |
| Template overfit | Good on training chat format only | Evaluate alternate templates and languages |
| Policy ambiguity | Inconsistent labels | Adjudication and rubric revision |
| Feedback drift | New labels change old policy silently | Version policy, rubric, and dataset together |

AI connection: Search objectives is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 4.4 Adaptive attacks

Adaptive attacks belongs in the canonical scope of red teaming and safety evaluations.
The object is the safety attack and evaluation protocol, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol v(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\widehat{\operatorname{ASR}} = \frac{1}{n}\sum_{i=1}^{n}\mathbb{1}\{v(x_i,y_i)=1\}.
$$

For adaptive attacks, this formula should not be treated as a slogan. It defines which
tokens, responses, comparisons, or decisions receive gradient or operational weight. A
change in masking, sampling, rubric wording, or thresholding changes the effective
objective even if the model architecture is unchanged.

| Alignment object | Mathematical question | Engineering question |
| --- | --- | --- |
| Data | Which examples define the target behavior? | Who wrote, filtered, and approved them? |
| Objective | Which terms receive weight? | Are masks, margins, and thresholds logged? |
| Policy | Which actions are allowed or disallowed? | Can reviewers reproduce the decision? |
| Evaluation | Which metric detects regression? | Is the test private, stable, and sliced? |
| Feedback | Which new evidence changes training? | How does it enter the next dataset version? |

Examples:
- Treat adaptive attacks as part of the model contract and store the exact data version.
- Record the prompt template, role format, policy version, and decoder settings.
- Compare aligned and reference policies on both helpfulness and safety slices.
- Use held-out examples that were not used to tune refusals or rewards.
- Inspect failure cases before declaring the objective successful.

Non-examples:
- Calling a model aligned because it sounds polite on a few prompts.
- Training on refusals without measuring over-refusal on benign requests.
- Using a reward model as ground truth without calibration or adversarial checks.
- Shipping a guardrail threshold without measuring false positive and false negative
rates.
- Letting feedback logs change training without provenance or consent controls.

A useful implementation pattern is to separate policy, data, and measurement. The policy
says what behavior is desired. The data supplies examples, comparisons, attacks, or
feedback events. The measurement checks whether the updated system moved in the intended
direction without unacceptable regressions.

```text
policy text/rubric
      |
      v
training or guardrail data  ->  objective/threshold  ->  aligned system
      |                                                   |
      v                                                   v
audit metadata                                      held-out safety eval
```

Worked reasoning pattern for adaptive attacks:
1. Name the target behavior in plain language.
2. Write the mathematical variable that represents it.
3. Specify which examples or comparisons estimate it.
4. Choose the optimization loss or runtime decision rule.
5. Define the regression metric that would prove the change became worse.

Three details are especially easy to miss in alignment work. First, the user intent
distribution is not the same as the pretraining distribution. Second, safety labels are
not ordinary class labels; they encode policy judgments that can change by context.
Third, optimization pressure finds shortcuts, so every proxy must be monitored for
Goodhart-style failures.

| Failure pressure | Typical symptom | Mitigation |
| --- | --- | --- |
| Proxy reward | High reward but worse human judgment | Holdout preferences and adversarial review |
| Refusal shortcut | Safe but unhelpful responses | Measure benign refusal rate separately |
| Template overfit | Good on training chat format only | Evaluate alternate templates and languages |
| Policy ambiguity | Inconsistent labels | Adjudication and rubric revision |
| Feedback drift | New labels change old policy silently | Version policy, rubric, and dataset together |

AI connection: Adaptive attacks is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 4.5 Budgeted exploration

Budgeted exploration belongs in the canonical scope of red teaming and safety
evaluations. The object is the safety attack and evaluation protocol, not merely a
prompt trick or a moderation label. We study how data, losses, policies, review
processes, and safety constraints shape a model's conditional distribution over
responses.

A compact way to read this subsection is through the local symbol v(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\widehat{\operatorname{ASR}} = \frac{1}{n}\sum_{i=1}^{n}\mathbb{1}\{v(x_i,y_i)=1\}.
$$

For budgeted exploration, this formula should not be treated as a slogan. It defines
which tokens, responses, comparisons, or decisions receive gradient or operational
weight. A change in masking, sampling, rubric wording, or thresholding changes the
effective objective even if the model architecture is unchanged.

| Alignment object | Mathematical question | Engineering question |
| --- | --- | --- |
| Data | Which examples define the target behavior? | Who wrote, filtered, and approved them? |
| Objective | Which terms receive weight? | Are masks, margins, and thresholds logged? |
| Policy | Which actions are allowed or disallowed? | Can reviewers reproduce the decision? |
| Evaluation | Which metric detects regression? | Is the test private, stable, and sliced? |
| Feedback | Which new evidence changes training? | How does it enter the next dataset version? |

Examples:
- Treat budgeted exploration as part of the model contract and store the exact data
version.
- Record the prompt template, role format, policy version, and decoder settings.
- Compare aligned and reference policies on both helpfulness and safety slices.
- Use held-out examples that were not used to tune refusals or rewards.
- Inspect failure cases before declaring the objective successful.

Non-examples:
- Calling a model aligned because it sounds polite on a few prompts.
- Training on refusals without measuring over-refusal on benign requests.
- Using a reward model as ground truth without calibration or adversarial checks.
- Shipping a guardrail threshold without measuring false positive and false negative
rates.
- Letting feedback logs change training without provenance or consent controls.

A useful implementation pattern is to separate policy, data, and measurement. The policy
says what behavior is desired. The data supplies examples, comparisons, attacks, or
feedback events. The measurement checks whether the updated system moved in the intended
direction without unacceptable regressions.

```text
policy text/rubric
      |
      v
training or guardrail data  ->  objective/threshold  ->  aligned system
      |                                                   |
      v                                                   v
audit metadata                                      held-out safety eval
```

Worked reasoning pattern for budgeted exploration:
1. Name the target behavior in plain language.
2. Write the mathematical variable that represents it.
3. Specify which examples or comparisons estimate it.
4. Choose the optimization loss or runtime decision rule.
5. Define the regression metric that would prove the change became worse.

Three details are especially easy to miss in alignment work. First, the user intent
distribution is not the same as the pretraining distribution. Second, safety labels are
not ordinary class labels; they encode policy judgments that can change by context.
Third, optimization pressure finds shortcuts, so every proxy must be monitored for
Goodhart-style failures.

| Failure pressure | Typical symptom | Mitigation |
| --- | --- | --- |
| Proxy reward | High reward but worse human judgment | Holdout preferences and adversarial review |
| Refusal shortcut | Safe but unhelpful responses | Measure benign refusal rate separately |
| Template overfit | Good on training chat format only | Evaluate alternate templates and languages |
| Policy ambiguity | Inconsistent labels | Adjudication and rubric revision |
| Feedback drift | New labels change old policy silently | Version policy, rubric, and dataset together |

AI connection: Budgeted exploration is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

## 5. Safety Benchmarks and Datasets

Safety Benchmarks and Datasets develops the part of red teaming and safety evaluations
that the approved TOC assigns to Chapter 18. The emphasis is alignment behavior, safety
constraints, and feedback loops, not generic fine-tuning or production monitoring.

### 5.1 SafetyPrompts

SafetyPrompts belongs in the canonical scope of red teaming and safety evaluations. The
object is the safety attack and evaluation protocol, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol v(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\widehat{\operatorname{ASR}} = \frac{1}{n}\sum_{i=1}^{n}\mathbb{1}\{v(x_i,y_i)=1\}.
$$

For safetyprompts, this formula should not be treated as a slogan. It defines which
tokens, responses, comparisons, or decisions receive gradient or operational weight. A
change in masking, sampling, rubric wording, or thresholding changes the effective
objective even if the model architecture is unchanged.

| Alignment object | Mathematical question | Engineering question |
| --- | --- | --- |
| Data | Which examples define the target behavior? | Who wrote, filtered, and approved them? |
| Objective | Which terms receive weight? | Are masks, margins, and thresholds logged? |
| Policy | Which actions are allowed or disallowed? | Can reviewers reproduce the decision? |
| Evaluation | Which metric detects regression? | Is the test private, stable, and sliced? |
| Feedback | Which new evidence changes training? | How does it enter the next dataset version? |

Examples:
- Treat safetyprompts as part of the model contract and store the exact data version.
- Record the prompt template, role format, policy version, and decoder settings.
- Compare aligned and reference policies on both helpfulness and safety slices.
- Use held-out examples that were not used to tune refusals or rewards.
- Inspect failure cases before declaring the objective successful.

Non-examples:
- Calling a model aligned because it sounds polite on a few prompts.
- Training on refusals without measuring over-refusal on benign requests.
- Using a reward model as ground truth without calibration or adversarial checks.
- Shipping a guardrail threshold without measuring false positive and false negative
rates.
- Letting feedback logs change training without provenance or consent controls.

A useful implementation pattern is to separate policy, data, and measurement. The policy
says what behavior is desired. The data supplies examples, comparisons, attacks, or
feedback events. The measurement checks whether the updated system moved in the intended
direction without unacceptable regressions.

```text
policy text/rubric
      |
      v
training or guardrail data  ->  objective/threshold  ->  aligned system
      |                                                   |
      v                                                   v
audit metadata                                      held-out safety eval
```

Worked reasoning pattern for safetyprompts:
1. Name the target behavior in plain language.
2. Write the mathematical variable that represents it.
3. Specify which examples or comparisons estimate it.
4. Choose the optimization loss or runtime decision rule.
5. Define the regression metric that would prove the change became worse.

Three details are especially easy to miss in alignment work. First, the user intent
distribution is not the same as the pretraining distribution. Second, safety labels are
not ordinary class labels; they encode policy judgments that can change by context.
Third, optimization pressure finds shortcuts, so every proxy must be monitored for
Goodhart-style failures.

| Failure pressure | Typical symptom | Mitigation |
| --- | --- | --- |
| Proxy reward | High reward but worse human judgment | Holdout preferences and adversarial review |
| Refusal shortcut | Safe but unhelpful responses | Measure benign refusal rate separately |
| Template overfit | Good on training chat format only | Evaluate alternate templates and languages |
| Policy ambiguity | Inconsistent labels | Adjudication and rubric revision |
| Feedback drift | New labels change old policy silently | Version policy, rubric, and dataset together |

AI connection: SafetyPrompts is part of the post-training stack used by modern assistant
systems. It links the base language model to human intent, safety policy, and deployment
constraints without pretending that a single loss can capture all values. The goal is
not perfect alignment by formula; it is a repeatable loop where evidence, objectives,
and safeguards improve together.

### 5.2 HarmBench-style suites

HarmBench-style suites belongs in the canonical scope of red teaming and safety
evaluations. The object is the safety attack and evaluation protocol, not merely a
prompt trick or a moderation label. We study how data, losses, policies, review
processes, and safety constraints shape a model's conditional distribution over
responses.

A compact way to read this subsection is through the local symbol v(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\widehat{\operatorname{ASR}} = \frac{1}{n}\sum_{i=1}^{n}\mathbb{1}\{v(x_i,y_i)=1\}.
$$

For harmbench-style suites, this formula should not be treated as a slogan. It defines
which tokens, responses, comparisons, or decisions receive gradient or operational
weight. A change in masking, sampling, rubric wording, or thresholding changes the
effective objective even if the model architecture is unchanged.

| Alignment object | Mathematical question | Engineering question |
| --- | --- | --- |
| Data | Which examples define the target behavior? | Who wrote, filtered, and approved them? |
| Objective | Which terms receive weight? | Are masks, margins, and thresholds logged? |
| Policy | Which actions are allowed or disallowed? | Can reviewers reproduce the decision? |
| Evaluation | Which metric detects regression? | Is the test private, stable, and sliced? |
| Feedback | Which new evidence changes training? | How does it enter the next dataset version? |

Examples:
- Treat harmbench-style suites as part of the model contract and store the exact data
version.
- Record the prompt template, role format, policy version, and decoder settings.
- Compare aligned and reference policies on both helpfulness and safety slices.
- Use held-out examples that were not used to tune refusals or rewards.
- Inspect failure cases before declaring the objective successful.

Non-examples:
- Calling a model aligned because it sounds polite on a few prompts.
- Training on refusals without measuring over-refusal on benign requests.
- Using a reward model as ground truth without calibration or adversarial checks.
- Shipping a guardrail threshold without measuring false positive and false negative
rates.
- Letting feedback logs change training without provenance or consent controls.

A useful implementation pattern is to separate policy, data, and measurement. The policy
says what behavior is desired. The data supplies examples, comparisons, attacks, or
feedback events. The measurement checks whether the updated system moved in the intended
direction without unacceptable regressions.

```text
policy text/rubric
      |
      v
training or guardrail data  ->  objective/threshold  ->  aligned system
      |                                                   |
      v                                                   v
audit metadata                                      held-out safety eval
```

Worked reasoning pattern for harmbench-style suites:
1. Name the target behavior in plain language.
2. Write the mathematical variable that represents it.
3. Specify which examples or comparisons estimate it.
4. Choose the optimization loss or runtime decision rule.
5. Define the regression metric that would prove the change became worse.

Three details are especially easy to miss in alignment work. First, the user intent
distribution is not the same as the pretraining distribution. Second, safety labels are
not ordinary class labels; they encode policy judgments that can change by context.
Third, optimization pressure finds shortcuts, so every proxy must be monitored for
Goodhart-style failures.

| Failure pressure | Typical symptom | Mitigation |
| --- | --- | --- |
| Proxy reward | High reward but worse human judgment | Holdout preferences and adversarial review |
| Refusal shortcut | Safe but unhelpful responses | Measure benign refusal rate separately |
| Template overfit | Good on training chat format only | Evaluate alternate templates and languages |
| Policy ambiguity | Inconsistent labels | Adjudication and rubric revision |
| Feedback drift | New labels change old policy silently | Version policy, rubric, and dataset together |

AI connection: HarmBench-style suites is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 5.3 Jailbreak sets

Jailbreak sets belongs in the canonical scope of red teaming and safety evaluations. The
object is the safety attack and evaluation protocol, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol v(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\widehat{\operatorname{ASR}} = \frac{1}{n}\sum_{i=1}^{n}\mathbb{1}\{v(x_i,y_i)=1\}.
$$

For jailbreak sets, this formula should not be treated as a slogan. It defines which
tokens, responses, comparisons, or decisions receive gradient or operational weight. A
change in masking, sampling, rubric wording, or thresholding changes the effective
objective even if the model architecture is unchanged.

| Alignment object | Mathematical question | Engineering question |
| --- | --- | --- |
| Data | Which examples define the target behavior? | Who wrote, filtered, and approved them? |
| Objective | Which terms receive weight? | Are masks, margins, and thresholds logged? |
| Policy | Which actions are allowed or disallowed? | Can reviewers reproduce the decision? |
| Evaluation | Which metric detects regression? | Is the test private, stable, and sliced? |
| Feedback | Which new evidence changes training? | How does it enter the next dataset version? |

Examples:
- Treat jailbreak sets as part of the model contract and store the exact data version.
- Record the prompt template, role format, policy version, and decoder settings.
- Compare aligned and reference policies on both helpfulness and safety slices.
- Use held-out examples that were not used to tune refusals or rewards.
- Inspect failure cases before declaring the objective successful.

Non-examples:
- Calling a model aligned because it sounds polite on a few prompts.
- Training on refusals without measuring over-refusal on benign requests.
- Using a reward model as ground truth without calibration or adversarial checks.
- Shipping a guardrail threshold without measuring false positive and false negative
rates.
- Letting feedback logs change training without provenance or consent controls.

A useful implementation pattern is to separate policy, data, and measurement. The policy
says what behavior is desired. The data supplies examples, comparisons, attacks, or
feedback events. The measurement checks whether the updated system moved in the intended
direction without unacceptable regressions.

```text
policy text/rubric
      |
      v
training or guardrail data  ->  objective/threshold  ->  aligned system
      |                                                   |
      v                                                   v
audit metadata                                      held-out safety eval
```

Worked reasoning pattern for jailbreak sets:
1. Name the target behavior in plain language.
2. Write the mathematical variable that represents it.
3. Specify which examples or comparisons estimate it.
4. Choose the optimization loss or runtime decision rule.
5. Define the regression metric that would prove the change became worse.

Three details are especially easy to miss in alignment work. First, the user intent
distribution is not the same as the pretraining distribution. Second, safety labels are
not ordinary class labels; they encode policy judgments that can change by context.
Third, optimization pressure finds shortcuts, so every proxy must be monitored for
Goodhart-style failures.

| Failure pressure | Typical symptom | Mitigation |
| --- | --- | --- |
| Proxy reward | High reward but worse human judgment | Holdout preferences and adversarial review |
| Refusal shortcut | Safe but unhelpful responses | Measure benign refusal rate separately |
| Template overfit | Good on training chat format only | Evaluate alternate templates and languages |
| Policy ambiguity | Inconsistent labels | Adjudication and rubric revision |
| Feedback drift | New labels change old policy silently | Version policy, rubric, and dataset together |

AI connection: Jailbreak sets is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 5.4 Refusal measurement

Refusal measurement belongs in the canonical scope of red teaming and safety
evaluations. The object is the safety attack and evaluation protocol, not merely a
prompt trick or a moderation label. We study how data, losses, policies, review
processes, and safety constraints shape a model's conditional distribution over
responses.

A compact way to read this subsection is through the local symbol v(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\widehat{\operatorname{ASR}} = \frac{1}{n}\sum_{i=1}^{n}\mathbb{1}\{v(x_i,y_i)=1\}.
$$

For refusal measurement, this formula should not be treated as a slogan. It defines
which tokens, responses, comparisons, or decisions receive gradient or operational
weight. A change in masking, sampling, rubric wording, or thresholding changes the
effective objective even if the model architecture is unchanged.

| Alignment object | Mathematical question | Engineering question |
| --- | --- | --- |
| Data | Which examples define the target behavior? | Who wrote, filtered, and approved them? |
| Objective | Which terms receive weight? | Are masks, margins, and thresholds logged? |
| Policy | Which actions are allowed or disallowed? | Can reviewers reproduce the decision? |
| Evaluation | Which metric detects regression? | Is the test private, stable, and sliced? |
| Feedback | Which new evidence changes training? | How does it enter the next dataset version? |

Examples:
- Treat refusal measurement as part of the model contract and store the exact data
version.
- Record the prompt template, role format, policy version, and decoder settings.
- Compare aligned and reference policies on both helpfulness and safety slices.
- Use held-out examples that were not used to tune refusals or rewards.
- Inspect failure cases before declaring the objective successful.

Non-examples:
- Calling a model aligned because it sounds polite on a few prompts.
- Training on refusals without measuring over-refusal on benign requests.
- Using a reward model as ground truth without calibration or adversarial checks.
- Shipping a guardrail threshold without measuring false positive and false negative
rates.
- Letting feedback logs change training without provenance or consent controls.

A useful implementation pattern is to separate policy, data, and measurement. The policy
says what behavior is desired. The data supplies examples, comparisons, attacks, or
feedback events. The measurement checks whether the updated system moved in the intended
direction without unacceptable regressions.

```text
policy text/rubric
      |
      v
training or guardrail data  ->  objective/threshold  ->  aligned system
      |                                                   |
      v                                                   v
audit metadata                                      held-out safety eval
```

Worked reasoning pattern for refusal measurement:
1. Name the target behavior in plain language.
2. Write the mathematical variable that represents it.
3. Specify which examples or comparisons estimate it.
4. Choose the optimization loss or runtime decision rule.
5. Define the regression metric that would prove the change became worse.

Three details are especially easy to miss in alignment work. First, the user intent
distribution is not the same as the pretraining distribution. Second, safety labels are
not ordinary class labels; they encode policy judgments that can change by context.
Third, optimization pressure finds shortcuts, so every proxy must be monitored for
Goodhart-style failures.

| Failure pressure | Typical symptom | Mitigation |
| --- | --- | --- |
| Proxy reward | High reward but worse human judgment | Holdout preferences and adversarial review |
| Refusal shortcut | Safe but unhelpful responses | Measure benign refusal rate separately |
| Template overfit | Good on training chat format only | Evaluate alternate templates and languages |
| Policy ambiguity | Inconsistent labels | Adjudication and rubric revision |
| Feedback drift | New labels change old policy silently | Version policy, rubric, and dataset together |

AI connection: Refusal measurement is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 5.5 Over-refusal measurement

Over-refusal measurement belongs in the canonical scope of red teaming and safety
evaluations. The object is the safety attack and evaluation protocol, not merely a
prompt trick or a moderation label. We study how data, losses, policies, review
processes, and safety constraints shape a model's conditional distribution over
responses.

A compact way to read this subsection is through the local symbol v(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\widehat{\operatorname{ASR}} = \frac{1}{n}\sum_{i=1}^{n}\mathbb{1}\{v(x_i,y_i)=1\}.
$$

For over-refusal measurement, this formula should not be treated as a slogan. It defines
which tokens, responses, comparisons, or decisions receive gradient or operational
weight. A change in masking, sampling, rubric wording, or thresholding changes the
effective objective even if the model architecture is unchanged.

| Alignment object | Mathematical question | Engineering question |
| --- | --- | --- |
| Data | Which examples define the target behavior? | Who wrote, filtered, and approved them? |
| Objective | Which terms receive weight? | Are masks, margins, and thresholds logged? |
| Policy | Which actions are allowed or disallowed? | Can reviewers reproduce the decision? |
| Evaluation | Which metric detects regression? | Is the test private, stable, and sliced? |
| Feedback | Which new evidence changes training? | How does it enter the next dataset version? |

Examples:
- Treat over-refusal measurement as part of the model contract and store the exact data
version.
- Record the prompt template, role format, policy version, and decoder settings.
- Compare aligned and reference policies on both helpfulness and safety slices.
- Use held-out examples that were not used to tune refusals or rewards.
- Inspect failure cases before declaring the objective successful.

Non-examples:
- Calling a model aligned because it sounds polite on a few prompts.
- Training on refusals without measuring over-refusal on benign requests.
- Using a reward model as ground truth without calibration or adversarial checks.
- Shipping a guardrail threshold without measuring false positive and false negative
rates.
- Letting feedback logs change training without provenance or consent controls.

A useful implementation pattern is to separate policy, data, and measurement. The policy
says what behavior is desired. The data supplies examples, comparisons, attacks, or
feedback events. The measurement checks whether the updated system moved in the intended
direction without unacceptable regressions.

```text
policy text/rubric
      |
      v
training or guardrail data  ->  objective/threshold  ->  aligned system
      |                                                   |
      v                                                   v
audit metadata                                      held-out safety eval
```

Worked reasoning pattern for over-refusal measurement:
1. Name the target behavior in plain language.
2. Write the mathematical variable that represents it.
3. Specify which examples or comparisons estimate it.
4. Choose the optimization loss or runtime decision rule.
5. Define the regression metric that would prove the change became worse.

Three details are especially easy to miss in alignment work. First, the user intent
distribution is not the same as the pretraining distribution. Second, safety labels are
not ordinary class labels; they encode policy judgments that can change by context.
Third, optimization pressure finds shortcuts, so every proxy must be monitored for
Goodhart-style failures.

| Failure pressure | Typical symptom | Mitigation |
| --- | --- | --- |
| Proxy reward | High reward but worse human judgment | Holdout preferences and adversarial review |
| Refusal shortcut | Safe but unhelpful responses | Measure benign refusal rate separately |
| Template overfit | Good on training chat format only | Evaluate alternate templates and languages |
| Policy ambiguity | Inconsistent labels | Adjudication and rubric revision |
| Feedback drift | New labels change old policy silently | Version policy, rubric, and dataset together |

AI connection: Over-refusal measurement is part of the post-training stack used by
modern assistant systems. It links the base language model to human intent, safety
policy, and deployment constraints without pretending that a single loss can capture all
values. The goal is not perfect alignment by formula; it is a repeatable loop where
evidence, objectives, and safeguards improve together.

## 6. Measurement Math

Measurement Math develops the part of red teaming and safety evaluations that the
approved TOC assigns to Chapter 18. The emphasis is alignment behavior, safety
constraints, and feedback loops, not generic fine-tuning or production monitoring.

### 6.1 Attack success rate

Attack success rate belongs in the canonical scope of red teaming and safety
evaluations. The object is the safety attack and evaluation protocol, not merely a
prompt trick or a moderation label. We study how data, losses, policies, review
processes, and safety constraints shape a model's conditional distribution over
responses.

A compact way to read this subsection is through the local symbol v(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\widehat{\operatorname{ASR}} = \frac{1}{n}\sum_{i=1}^{n}\mathbb{1}\{v(x_i,y_i)=1\}.
$$

For attack success rate, this formula should not be treated as a slogan. It defines
which tokens, responses, comparisons, or decisions receive gradient or operational
weight. A change in masking, sampling, rubric wording, or thresholding changes the
effective objective even if the model architecture is unchanged.

| Alignment object | Mathematical question | Engineering question |
| --- | --- | --- |
| Data | Which examples define the target behavior? | Who wrote, filtered, and approved them? |
| Objective | Which terms receive weight? | Are masks, margins, and thresholds logged? |
| Policy | Which actions are allowed or disallowed? | Can reviewers reproduce the decision? |
| Evaluation | Which metric detects regression? | Is the test private, stable, and sliced? |
| Feedback | Which new evidence changes training? | How does it enter the next dataset version? |

Examples:
- Treat attack success rate as part of the model contract and store the exact data
version.
- Record the prompt template, role format, policy version, and decoder settings.
- Compare aligned and reference policies on both helpfulness and safety slices.
- Use held-out examples that were not used to tune refusals or rewards.
- Inspect failure cases before declaring the objective successful.

Non-examples:
- Calling a model aligned because it sounds polite on a few prompts.
- Training on refusals without measuring over-refusal on benign requests.
- Using a reward model as ground truth without calibration or adversarial checks.
- Shipping a guardrail threshold without measuring false positive and false negative
rates.
- Letting feedback logs change training without provenance or consent controls.

A useful implementation pattern is to separate policy, data, and measurement. The policy
says what behavior is desired. The data supplies examples, comparisons, attacks, or
feedback events. The measurement checks whether the updated system moved in the intended
direction without unacceptable regressions.

```text
policy text/rubric
      |
      v
training or guardrail data  ->  objective/threshold  ->  aligned system
      |                                                   |
      v                                                   v
audit metadata                                      held-out safety eval
```

Worked reasoning pattern for attack success rate:
1. Name the target behavior in plain language.
2. Write the mathematical variable that represents it.
3. Specify which examples or comparisons estimate it.
4. Choose the optimization loss or runtime decision rule.
5. Define the regression metric that would prove the change became worse.

Three details are especially easy to miss in alignment work. First, the user intent
distribution is not the same as the pretraining distribution. Second, safety labels are
not ordinary class labels; they encode policy judgments that can change by context.
Third, optimization pressure finds shortcuts, so every proxy must be monitored for
Goodhart-style failures.

| Failure pressure | Typical symptom | Mitigation |
| --- | --- | --- |
| Proxy reward | High reward but worse human judgment | Holdout preferences and adversarial review |
| Refusal shortcut | Safe but unhelpful responses | Measure benign refusal rate separately |
| Template overfit | Good on training chat format only | Evaluate alternate templates and languages |
| Policy ambiguity | Inconsistent labels | Adjudication and rubric revision |
| Feedback drift | New labels change old policy silently | Version policy, rubric, and dataset together |

AI connection: Attack success rate is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 6.2 Confidence intervals

Confidence intervals belongs in the canonical scope of red teaming and safety
evaluations. The object is the safety attack and evaluation protocol, not merely a
prompt trick or a moderation label. We study how data, losses, policies, review
processes, and safety constraints shape a model's conditional distribution over
responses.

A compact way to read this subsection is through the local symbol v(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\widehat{\operatorname{ASR}} = \frac{1}{n}\sum_{i=1}^{n}\mathbb{1}\{v(x_i,y_i)=1\}.
$$

For confidence intervals, this formula should not be treated as a slogan. It defines
which tokens, responses, comparisons, or decisions receive gradient or operational
weight. A change in masking, sampling, rubric wording, or thresholding changes the
effective objective even if the model architecture is unchanged.

| Alignment object | Mathematical question | Engineering question |
| --- | --- | --- |
| Data | Which examples define the target behavior? | Who wrote, filtered, and approved them? |
| Objective | Which terms receive weight? | Are masks, margins, and thresholds logged? |
| Policy | Which actions are allowed or disallowed? | Can reviewers reproduce the decision? |
| Evaluation | Which metric detects regression? | Is the test private, stable, and sliced? |
| Feedback | Which new evidence changes training? | How does it enter the next dataset version? |

Examples:
- Treat confidence intervals as part of the model contract and store the exact data
version.
- Record the prompt template, role format, policy version, and decoder settings.
- Compare aligned and reference policies on both helpfulness and safety slices.
- Use held-out examples that were not used to tune refusals or rewards.
- Inspect failure cases before declaring the objective successful.

Non-examples:
- Calling a model aligned because it sounds polite on a few prompts.
- Training on refusals without measuring over-refusal on benign requests.
- Using a reward model as ground truth without calibration or adversarial checks.
- Shipping a guardrail threshold without measuring false positive and false negative
rates.
- Letting feedback logs change training without provenance or consent controls.

A useful implementation pattern is to separate policy, data, and measurement. The policy
says what behavior is desired. The data supplies examples, comparisons, attacks, or
feedback events. The measurement checks whether the updated system moved in the intended
direction without unacceptable regressions.

```text
policy text/rubric
      |
      v
training or guardrail data  ->  objective/threshold  ->  aligned system
      |                                                   |
      v                                                   v
audit metadata                                      held-out safety eval
```

Worked reasoning pattern for confidence intervals:
1. Name the target behavior in plain language.
2. Write the mathematical variable that represents it.
3. Specify which examples or comparisons estimate it.
4. Choose the optimization loss or runtime decision rule.
5. Define the regression metric that would prove the change became worse.

Three details are especially easy to miss in alignment work. First, the user intent
distribution is not the same as the pretraining distribution. Second, safety labels are
not ordinary class labels; they encode policy judgments that can change by context.
Third, optimization pressure finds shortcuts, so every proxy must be monitored for
Goodhart-style failures.

| Failure pressure | Typical symptom | Mitigation |
| --- | --- | --- |
| Proxy reward | High reward but worse human judgment | Holdout preferences and adversarial review |
| Refusal shortcut | Safe but unhelpful responses | Measure benign refusal rate separately |
| Template overfit | Good on training chat format only | Evaluate alternate templates and languages |
| Policy ambiguity | Inconsistent labels | Adjudication and rubric revision |
| Feedback drift | New labels change old policy silently | Version policy, rubric, and dataset together |

AI connection: Confidence intervals is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 6.3 False positive and false negative tradeoffs

False positive and false negative tradeoffs belongs in the canonical scope of red
teaming and safety evaluations. The object is the safety attack and evaluation protocol,
not merely a prompt trick or a moderation label. We study how data, losses, policies,
review processes, and safety constraints shape a model's conditional distribution over
responses.

A compact way to read this subsection is through the local symbol v(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\widehat{\operatorname{ASR}} = \frac{1}{n}\sum_{i=1}^{n}\mathbb{1}\{v(x_i,y_i)=1\}.
$$

For false positive and false negative tradeoffs, this formula should not be treated as a
slogan. It defines which tokens, responses, comparisons, or decisions receive gradient
or operational weight. A change in masking, sampling, rubric wording, or thresholding
changes the effective objective even if the model architecture is unchanged.

| Alignment object | Mathematical question | Engineering question |
| --- | --- | --- |
| Data | Which examples define the target behavior? | Who wrote, filtered, and approved them? |
| Objective | Which terms receive weight? | Are masks, margins, and thresholds logged? |
| Policy | Which actions are allowed or disallowed? | Can reviewers reproduce the decision? |
| Evaluation | Which metric detects regression? | Is the test private, stable, and sliced? |
| Feedback | Which new evidence changes training? | How does it enter the next dataset version? |

Examples:
- Treat false positive and false negative tradeoffs as part of the model contract and
store the exact data version.
- Record the prompt template, role format, policy version, and decoder settings.
- Compare aligned and reference policies on both helpfulness and safety slices.
- Use held-out examples that were not used to tune refusals or rewards.
- Inspect failure cases before declaring the objective successful.

Non-examples:
- Calling a model aligned because it sounds polite on a few prompts.
- Training on refusals without measuring over-refusal on benign requests.
- Using a reward model as ground truth without calibration or adversarial checks.
- Shipping a guardrail threshold without measuring false positive and false negative
rates.
- Letting feedback logs change training without provenance or consent controls.

A useful implementation pattern is to separate policy, data, and measurement. The policy
says what behavior is desired. The data supplies examples, comparisons, attacks, or
feedback events. The measurement checks whether the updated system moved in the intended
direction without unacceptable regressions.

```text
policy text/rubric
      |
      v
training or guardrail data  ->  objective/threshold  ->  aligned system
      |                                                   |
      v                                                   v
audit metadata                                      held-out safety eval
```

Worked reasoning pattern for false positive and false negative tradeoffs:
1. Name the target behavior in plain language.
2. Write the mathematical variable that represents it.
3. Specify which examples or comparisons estimate it.
4. Choose the optimization loss or runtime decision rule.
5. Define the regression metric that would prove the change became worse.

Three details are especially easy to miss in alignment work. First, the user intent
distribution is not the same as the pretraining distribution. Second, safety labels are
not ordinary class labels; they encode policy judgments that can change by context.
Third, optimization pressure finds shortcuts, so every proxy must be monitored for
Goodhart-style failures.

| Failure pressure | Typical symptom | Mitigation |
| --- | --- | --- |
| Proxy reward | High reward but worse human judgment | Holdout preferences and adversarial review |
| Refusal shortcut | Safe but unhelpful responses | Measure benign refusal rate separately |
| Template overfit | Good on training chat format only | Evaluate alternate templates and languages |
| Policy ambiguity | Inconsistent labels | Adjudication and rubric revision |
| Feedback drift | New labels change old policy silently | Version policy, rubric, and dataset together |

AI connection: False positive and false negative tradeoffs is part of the post-training
stack used by modern assistant systems. It links the base language model to human
intent, safety policy, and deployment constraints without pretending that a single loss
can capture all values. The goal is not perfect alignment by formula; it is a repeatable
loop where evidence, objectives, and safeguards improve together.

### 6.4 Severity weighting

Severity weighting belongs in the canonical scope of red teaming and safety evaluations.
The object is the safety attack and evaluation protocol, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol v(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\widehat{\operatorname{ASR}} = \frac{1}{n}\sum_{i=1}^{n}\mathbb{1}\{v(x_i,y_i)=1\}.
$$

For severity weighting, this formula should not be treated as a slogan. It defines which
tokens, responses, comparisons, or decisions receive gradient or operational weight. A
change in masking, sampling, rubric wording, or thresholding changes the effective
objective even if the model architecture is unchanged.

| Alignment object | Mathematical question | Engineering question |
| --- | --- | --- |
| Data | Which examples define the target behavior? | Who wrote, filtered, and approved them? |
| Objective | Which terms receive weight? | Are masks, margins, and thresholds logged? |
| Policy | Which actions are allowed or disallowed? | Can reviewers reproduce the decision? |
| Evaluation | Which metric detects regression? | Is the test private, stable, and sliced? |
| Feedback | Which new evidence changes training? | How does it enter the next dataset version? |

Examples:
- Treat severity weighting as part of the model contract and store the exact data
version.
- Record the prompt template, role format, policy version, and decoder settings.
- Compare aligned and reference policies on both helpfulness and safety slices.
- Use held-out examples that were not used to tune refusals or rewards.
- Inspect failure cases before declaring the objective successful.

Non-examples:
- Calling a model aligned because it sounds polite on a few prompts.
- Training on refusals without measuring over-refusal on benign requests.
- Using a reward model as ground truth without calibration or adversarial checks.
- Shipping a guardrail threshold without measuring false positive and false negative
rates.
- Letting feedback logs change training without provenance or consent controls.

A useful implementation pattern is to separate policy, data, and measurement. The policy
says what behavior is desired. The data supplies examples, comparisons, attacks, or
feedback events. The measurement checks whether the updated system moved in the intended
direction without unacceptable regressions.

```text
policy text/rubric
      |
      v
training or guardrail data  ->  objective/threshold  ->  aligned system
      |                                                   |
      v                                                   v
audit metadata                                      held-out safety eval
```

Worked reasoning pattern for severity weighting:
1. Name the target behavior in plain language.
2. Write the mathematical variable that represents it.
3. Specify which examples or comparisons estimate it.
4. Choose the optimization loss or runtime decision rule.
5. Define the regression metric that would prove the change became worse.

Three details are especially easy to miss in alignment work. First, the user intent
distribution is not the same as the pretraining distribution. Second, safety labels are
not ordinary class labels; they encode policy judgments that can change by context.
Third, optimization pressure finds shortcuts, so every proxy must be monitored for
Goodhart-style failures.

| Failure pressure | Typical symptom | Mitigation |
| --- | --- | --- |
| Proxy reward | High reward but worse human judgment | Holdout preferences and adversarial review |
| Refusal shortcut | Safe but unhelpful responses | Measure benign refusal rate separately |
| Template overfit | Good on training chat format only | Evaluate alternate templates and languages |
| Policy ambiguity | Inconsistent labels | Adjudication and rubric revision |
| Feedback drift | New labels change old policy silently | Version policy, rubric, and dataset together |

AI connection: Severity weighting is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 6.5 Paired safety comparisons

Paired safety comparisons belongs in the canonical scope of red teaming and safety
evaluations. The object is the safety attack and evaluation protocol, not merely a
prompt trick or a moderation label. We study how data, losses, policies, review
processes, and safety constraints shape a model's conditional distribution over
responses.

A compact way to read this subsection is through the local symbol v(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\widehat{\operatorname{ASR}} = \frac{1}{n}\sum_{i=1}^{n}\mathbb{1}\{v(x_i,y_i)=1\}.
$$

For paired safety comparisons, this formula should not be treated as a slogan. It
defines which tokens, responses, comparisons, or decisions receive gradient or
operational weight. A change in masking, sampling, rubric wording, or thresholding
changes the effective objective even if the model architecture is unchanged.

| Alignment object | Mathematical question | Engineering question |
| --- | --- | --- |
| Data | Which examples define the target behavior? | Who wrote, filtered, and approved them? |
| Objective | Which terms receive weight? | Are masks, margins, and thresholds logged? |
| Policy | Which actions are allowed or disallowed? | Can reviewers reproduce the decision? |
| Evaluation | Which metric detects regression? | Is the test private, stable, and sliced? |
| Feedback | Which new evidence changes training? | How does it enter the next dataset version? |

Examples:
- Treat paired safety comparisons as part of the model contract and store the exact data
version.
- Record the prompt template, role format, policy version, and decoder settings.
- Compare aligned and reference policies on both helpfulness and safety slices.
- Use held-out examples that were not used to tune refusals or rewards.
- Inspect failure cases before declaring the objective successful.

Non-examples:
- Calling a model aligned because it sounds polite on a few prompts.
- Training on refusals without measuring over-refusal on benign requests.
- Using a reward model as ground truth without calibration or adversarial checks.
- Shipping a guardrail threshold without measuring false positive and false negative
rates.
- Letting feedback logs change training without provenance or consent controls.

A useful implementation pattern is to separate policy, data, and measurement. The policy
says what behavior is desired. The data supplies examples, comparisons, attacks, or
feedback events. The measurement checks whether the updated system moved in the intended
direction without unacceptable regressions.

```text
policy text/rubric
      |
      v
training or guardrail data  ->  objective/threshold  ->  aligned system
      |                                                   |
      v                                                   v
audit metadata                                      held-out safety eval
```

Worked reasoning pattern for paired safety comparisons:
1. Name the target behavior in plain language.
2. Write the mathematical variable that represents it.
3. Specify which examples or comparisons estimate it.
4. Choose the optimization loss or runtime decision rule.
5. Define the regression metric that would prove the change became worse.

Three details are especially easy to miss in alignment work. First, the user intent
distribution is not the same as the pretraining distribution. Second, safety labels are
not ordinary class labels; they encode policy judgments that can change by context.
Third, optimization pressure finds shortcuts, so every proxy must be monitored for
Goodhart-style failures.

| Failure pressure | Typical symptom | Mitigation |
| --- | --- | --- |
| Proxy reward | High reward but worse human judgment | Holdout preferences and adversarial review |
| Refusal shortcut | Safe but unhelpful responses | Measure benign refusal rate separately |
| Template overfit | Good on training chat format only | Evaluate alternate templates and languages |
| Policy ambiguity | Inconsistent labels | Adjudication and rubric revision |
| Feedback drift | New labels change old policy silently | Version policy, rubric, and dataset together |

AI connection: Paired safety comparisons is part of the post-training stack used by
modern assistant systems. It links the base language model to human intent, safety
policy, and deployment constraints without pretending that a single loss can capture all
values. The goal is not perfect alignment by formula; it is a repeatable loop where
evidence, objectives, and safeguards improve together.

## 7. From Findings to Fixes

From Findings to Fixes develops the part of red teaming and safety evaluations that the
approved TOC assigns to Chapter 18. The emphasis is alignment behavior, safety
constraints, and feedback loops, not generic fine-tuning or production monitoring.

### 7.1 Regression tests

Regression tests belongs in the canonical scope of red teaming and safety evaluations.
The object is the safety attack and evaluation protocol, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol v(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\widehat{\operatorname{ASR}} = \frac{1}{n}\sum_{i=1}^{n}\mathbb{1}\{v(x_i,y_i)=1\}.
$$

For regression tests, this formula should not be treated as a slogan. It defines which
tokens, responses, comparisons, or decisions receive gradient or operational weight. A
change in masking, sampling, rubric wording, or thresholding changes the effective
objective even if the model architecture is unchanged.

| Alignment object | Mathematical question | Engineering question |
| --- | --- | --- |
| Data | Which examples define the target behavior? | Who wrote, filtered, and approved them? |
| Objective | Which terms receive weight? | Are masks, margins, and thresholds logged? |
| Policy | Which actions are allowed or disallowed? | Can reviewers reproduce the decision? |
| Evaluation | Which metric detects regression? | Is the test private, stable, and sliced? |
| Feedback | Which new evidence changes training? | How does it enter the next dataset version? |

Examples:
- Treat regression tests as part of the model contract and store the exact data version.
- Record the prompt template, role format, policy version, and decoder settings.
- Compare aligned and reference policies on both helpfulness and safety slices.
- Use held-out examples that were not used to tune refusals or rewards.
- Inspect failure cases before declaring the objective successful.

Non-examples:
- Calling a model aligned because it sounds polite on a few prompts.
- Training on refusals without measuring over-refusal on benign requests.
- Using a reward model as ground truth without calibration or adversarial checks.
- Shipping a guardrail threshold without measuring false positive and false negative
rates.
- Letting feedback logs change training without provenance or consent controls.

A useful implementation pattern is to separate policy, data, and measurement. The policy
says what behavior is desired. The data supplies examples, comparisons, attacks, or
feedback events. The measurement checks whether the updated system moved in the intended
direction without unacceptable regressions.

```text
policy text/rubric
      |
      v
training or guardrail data  ->  objective/threshold  ->  aligned system
      |                                                   |
      v                                                   v
audit metadata                                      held-out safety eval
```

Worked reasoning pattern for regression tests:
1. Name the target behavior in plain language.
2. Write the mathematical variable that represents it.
3. Specify which examples or comparisons estimate it.
4. Choose the optimization loss or runtime decision rule.
5. Define the regression metric that would prove the change became worse.

Three details are especially easy to miss in alignment work. First, the user intent
distribution is not the same as the pretraining distribution. Second, safety labels are
not ordinary class labels; they encode policy judgments that can change by context.
Third, optimization pressure finds shortcuts, so every proxy must be monitored for
Goodhart-style failures.

| Failure pressure | Typical symptom | Mitigation |
| --- | --- | --- |
| Proxy reward | High reward but worse human judgment | Holdout preferences and adversarial review |
| Refusal shortcut | Safe but unhelpful responses | Measure benign refusal rate separately |
| Template overfit | Good on training chat format only | Evaluate alternate templates and languages |
| Policy ambiguity | Inconsistent labels | Adjudication and rubric revision |
| Feedback drift | New labels change old policy silently | Version policy, rubric, and dataset together |

AI connection: Regression tests is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 7.2 Safety SFT

Safety SFT belongs in the canonical scope of red teaming and safety evaluations. The
object is the safety attack and evaluation protocol, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol v(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\widehat{\operatorname{ASR}} = \frac{1}{n}\sum_{i=1}^{n}\mathbb{1}\{v(x_i,y_i)=1\}.
$$

For safety sft, this formula should not be treated as a slogan. It defines which tokens,
responses, comparisons, or decisions receive gradient or operational weight. A change in
masking, sampling, rubric wording, or thresholding changes the effective objective even
if the model architecture is unchanged.

| Alignment object | Mathematical question | Engineering question |
| --- | --- | --- |
| Data | Which examples define the target behavior? | Who wrote, filtered, and approved them? |
| Objective | Which terms receive weight? | Are masks, margins, and thresholds logged? |
| Policy | Which actions are allowed or disallowed? | Can reviewers reproduce the decision? |
| Evaluation | Which metric detects regression? | Is the test private, stable, and sliced? |
| Feedback | Which new evidence changes training? | How does it enter the next dataset version? |

Examples:
- Treat safety sft as part of the model contract and store the exact data version.
- Record the prompt template, role format, policy version, and decoder settings.
- Compare aligned and reference policies on both helpfulness and safety slices.
- Use held-out examples that were not used to tune refusals or rewards.
- Inspect failure cases before declaring the objective successful.

Non-examples:
- Calling a model aligned because it sounds polite on a few prompts.
- Training on refusals without measuring over-refusal on benign requests.
- Using a reward model as ground truth without calibration or adversarial checks.
- Shipping a guardrail threshold without measuring false positive and false negative
rates.
- Letting feedback logs change training without provenance or consent controls.

A useful implementation pattern is to separate policy, data, and measurement. The policy
says what behavior is desired. The data supplies examples, comparisons, attacks, or
feedback events. The measurement checks whether the updated system moved in the intended
direction without unacceptable regressions.

```text
policy text/rubric
      |
      v
training or guardrail data  ->  objective/threshold  ->  aligned system
      |                                                   |
      v                                                   v
audit metadata                                      held-out safety eval
```

Worked reasoning pattern for safety sft:
1. Name the target behavior in plain language.
2. Write the mathematical variable that represents it.
3. Specify which examples or comparisons estimate it.
4. Choose the optimization loss or runtime decision rule.
5. Define the regression metric that would prove the change became worse.

Three details are especially easy to miss in alignment work. First, the user intent
distribution is not the same as the pretraining distribution. Second, safety labels are
not ordinary class labels; they encode policy judgments that can change by context.
Third, optimization pressure finds shortcuts, so every proxy must be monitored for
Goodhart-style failures.

| Failure pressure | Typical symptom | Mitigation |
| --- | --- | --- |
| Proxy reward | High reward but worse human judgment | Holdout preferences and adversarial review |
| Refusal shortcut | Safe but unhelpful responses | Measure benign refusal rate separately |
| Template overfit | Good on training chat format only | Evaluate alternate templates and languages |
| Policy ambiguity | Inconsistent labels | Adjudication and rubric revision |
| Feedback drift | New labels change old policy silently | Version policy, rubric, and dataset together |

AI connection: Safety SFT is part of the post-training stack used by modern assistant
systems. It links the base language model to human intent, safety policy, and deployment
constraints without pretending that a single loss can capture all values. The goal is
not perfect alignment by formula; it is a repeatable loop where evidence, objectives,
and safeguards improve together.

### 7.3 Preference data

Preference data belongs in the canonical scope of red teaming and safety evaluations.
The object is the safety attack and evaluation protocol, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol v(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\widehat{\operatorname{ASR}} = \frac{1}{n}\sum_{i=1}^{n}\mathbb{1}\{v(x_i,y_i)=1\}.
$$

For preference data, this formula should not be treated as a slogan. It defines which
tokens, responses, comparisons, or decisions receive gradient or operational weight. A
change in masking, sampling, rubric wording, or thresholding changes the effective
objective even if the model architecture is unchanged.

| Alignment object | Mathematical question | Engineering question |
| --- | --- | --- |
| Data | Which examples define the target behavior? | Who wrote, filtered, and approved them? |
| Objective | Which terms receive weight? | Are masks, margins, and thresholds logged? |
| Policy | Which actions are allowed or disallowed? | Can reviewers reproduce the decision? |
| Evaluation | Which metric detects regression? | Is the test private, stable, and sliced? |
| Feedback | Which new evidence changes training? | How does it enter the next dataset version? |

Examples:
- Treat preference data as part of the model contract and store the exact data version.
- Record the prompt template, role format, policy version, and decoder settings.
- Compare aligned and reference policies on both helpfulness and safety slices.
- Use held-out examples that were not used to tune refusals or rewards.
- Inspect failure cases before declaring the objective successful.

Non-examples:
- Calling a model aligned because it sounds polite on a few prompts.
- Training on refusals without measuring over-refusal on benign requests.
- Using a reward model as ground truth without calibration or adversarial checks.
- Shipping a guardrail threshold without measuring false positive and false negative
rates.
- Letting feedback logs change training without provenance or consent controls.

A useful implementation pattern is to separate policy, data, and measurement. The policy
says what behavior is desired. The data supplies examples, comparisons, attacks, or
feedback events. The measurement checks whether the updated system moved in the intended
direction without unacceptable regressions.

```text
policy text/rubric
      |
      v
training or guardrail data  ->  objective/threshold  ->  aligned system
      |                                                   |
      v                                                   v
audit metadata                                      held-out safety eval
```

Worked reasoning pattern for preference data:
1. Name the target behavior in plain language.
2. Write the mathematical variable that represents it.
3. Specify which examples or comparisons estimate it.
4. Choose the optimization loss or runtime decision rule.
5. Define the regression metric that would prove the change became worse.

Three details are especially easy to miss in alignment work. First, the user intent
distribution is not the same as the pretraining distribution. Second, safety labels are
not ordinary class labels; they encode policy judgments that can change by context.
Third, optimization pressure finds shortcuts, so every proxy must be monitored for
Goodhart-style failures.

| Failure pressure | Typical symptom | Mitigation |
| --- | --- | --- |
| Proxy reward | High reward but worse human judgment | Holdout preferences and adversarial review |
| Refusal shortcut | Safe but unhelpful responses | Measure benign refusal rate separately |
| Template overfit | Good on training chat format only | Evaluate alternate templates and languages |
| Policy ambiguity | Inconsistent labels | Adjudication and rubric revision |
| Feedback drift | New labels change old policy silently | Version policy, rubric, and dataset together |

AI connection: Preference data is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 7.4 Guardrail rules

Guardrail rules belongs in the canonical scope of red teaming and safety evaluations.
The object is the safety attack and evaluation protocol, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol v(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\widehat{\operatorname{ASR}} = \frac{1}{n}\sum_{i=1}^{n}\mathbb{1}\{v(x_i,y_i)=1\}.
$$

For guardrail rules, this formula should not be treated as a slogan. It defines which
tokens, responses, comparisons, or decisions receive gradient or operational weight. A
change in masking, sampling, rubric wording, or thresholding changes the effective
objective even if the model architecture is unchanged.

| Alignment object | Mathematical question | Engineering question |
| --- | --- | --- |
| Data | Which examples define the target behavior? | Who wrote, filtered, and approved them? |
| Objective | Which terms receive weight? | Are masks, margins, and thresholds logged? |
| Policy | Which actions are allowed or disallowed? | Can reviewers reproduce the decision? |
| Evaluation | Which metric detects regression? | Is the test private, stable, and sliced? |
| Feedback | Which new evidence changes training? | How does it enter the next dataset version? |

Examples:
- Treat guardrail rules as part of the model contract and store the exact data version.
- Record the prompt template, role format, policy version, and decoder settings.
- Compare aligned and reference policies on both helpfulness and safety slices.
- Use held-out examples that were not used to tune refusals or rewards.
- Inspect failure cases before declaring the objective successful.

Non-examples:
- Calling a model aligned because it sounds polite on a few prompts.
- Training on refusals without measuring over-refusal on benign requests.
- Using a reward model as ground truth without calibration or adversarial checks.
- Shipping a guardrail threshold without measuring false positive and false negative
rates.
- Letting feedback logs change training without provenance or consent controls.

A useful implementation pattern is to separate policy, data, and measurement. The policy
says what behavior is desired. The data supplies examples, comparisons, attacks, or
feedback events. The measurement checks whether the updated system moved in the intended
direction without unacceptable regressions.

```text
policy text/rubric
      |
      v
training or guardrail data  ->  objective/threshold  ->  aligned system
      |                                                   |
      v                                                   v
audit metadata                                      held-out safety eval
```

Worked reasoning pattern for guardrail rules:
1. Name the target behavior in plain language.
2. Write the mathematical variable that represents it.
3. Specify which examples or comparisons estimate it.
4. Choose the optimization loss or runtime decision rule.
5. Define the regression metric that would prove the change became worse.

Three details are especially easy to miss in alignment work. First, the user intent
distribution is not the same as the pretraining distribution. Second, safety labels are
not ordinary class labels; they encode policy judgments that can change by context.
Third, optimization pressure finds shortcuts, so every proxy must be monitored for
Goodhart-style failures.

| Failure pressure | Typical symptom | Mitigation |
| --- | --- | --- |
| Proxy reward | High reward but worse human judgment | Holdout preferences and adversarial review |
| Refusal shortcut | Safe but unhelpful responses | Measure benign refusal rate separately |
| Template overfit | Good on training chat format only | Evaluate alternate templates and languages |
| Policy ambiguity | Inconsistent labels | Adjudication and rubric revision |
| Feedback drift | New labels change old policy silently | Version policy, rubric, and dataset together |

AI connection: Guardrail rules is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 7.5 Non-overfitting discipline

Non-overfitting discipline belongs in the canonical scope of red teaming and safety
evaluations. The object is the safety attack and evaluation protocol, not merely a
prompt trick or a moderation label. We study how data, losses, policies, review
processes, and safety constraints shape a model's conditional distribution over
responses.

A compact way to read this subsection is through the local symbol v(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\widehat{\operatorname{ASR}} = \frac{1}{n}\sum_{i=1}^{n}\mathbb{1}\{v(x_i,y_i)=1\}.
$$

For non-overfitting discipline, this formula should not be treated as a slogan. It
defines which tokens, responses, comparisons, or decisions receive gradient or
operational weight. A change in masking, sampling, rubric wording, or thresholding
changes the effective objective even if the model architecture is unchanged.

| Alignment object | Mathematical question | Engineering question |
| --- | --- | --- |
| Data | Which examples define the target behavior? | Who wrote, filtered, and approved them? |
| Objective | Which terms receive weight? | Are masks, margins, and thresholds logged? |
| Policy | Which actions are allowed or disallowed? | Can reviewers reproduce the decision? |
| Evaluation | Which metric detects regression? | Is the test private, stable, and sliced? |
| Feedback | Which new evidence changes training? | How does it enter the next dataset version? |

Examples:
- Treat non-overfitting discipline as part of the model contract and store the exact
data version.
- Record the prompt template, role format, policy version, and decoder settings.
- Compare aligned and reference policies on both helpfulness and safety slices.
- Use held-out examples that were not used to tune refusals or rewards.
- Inspect failure cases before declaring the objective successful.

Non-examples:
- Calling a model aligned because it sounds polite on a few prompts.
- Training on refusals without measuring over-refusal on benign requests.
- Using a reward model as ground truth without calibration or adversarial checks.
- Shipping a guardrail threshold without measuring false positive and false negative
rates.
- Letting feedback logs change training without provenance or consent controls.

A useful implementation pattern is to separate policy, data, and measurement. The policy
says what behavior is desired. The data supplies examples, comparisons, attacks, or
feedback events. The measurement checks whether the updated system moved in the intended
direction without unacceptable regressions.

```text
policy text/rubric
      |
      v
training or guardrail data  ->  objective/threshold  ->  aligned system
      |                                                   |
      v                                                   v
audit metadata                                      held-out safety eval
```

Worked reasoning pattern for non-overfitting discipline:
1. Name the target behavior in plain language.
2. Write the mathematical variable that represents it.
3. Specify which examples or comparisons estimate it.
4. Choose the optimization loss or runtime decision rule.
5. Define the regression metric that would prove the change became worse.

Three details are especially easy to miss in alignment work. First, the user intent
distribution is not the same as the pretraining distribution. Second, safety labels are
not ordinary class labels; they encode policy judgments that can change by context.
Third, optimization pressure finds shortcuts, so every proxy must be monitored for
Goodhart-style failures.

| Failure pressure | Typical symptom | Mitigation |
| --- | --- | --- |
| Proxy reward | High reward but worse human judgment | Holdout preferences and adversarial review |
| Refusal shortcut | Safe but unhelpful responses | Measure benign refusal rate separately |
| Template overfit | Good on training chat format only | Evaluate alternate templates and languages |
| Policy ambiguity | Inconsistent labels | Adjudication and rubric revision |
| Feedback drift | New labels change old policy silently | Version policy, rubric, and dataset together |

AI connection: Non-overfitting discipline is part of the post-training stack used by
modern assistant systems. It links the base language model to human intent, safety
policy, and deployment constraints without pretending that a single loss can capture all
values. The goal is not perfect alignment by formula; it is a repeatable loop where
evidence, objectives, and safeguards improve together.

## 8. Common Mistakes

| # | Mistake | Why It Is Wrong | Fix |
| --- | --- | --- | --- |
| 1 | Treating SFT as full alignment | SFT imitates demonstrations but does not optimize preferences or robust safety. | Use preference optimization and safety evals after SFT. |
| 2 | Masking prompt tokens incorrectly | The model is trained to copy user prompts instead of answer them. | Use response-only loss masks for chat SFT. |
| 3 | Trusting reward scores as truth | Reward models are learned proxies with bias and calibration error. | Evaluate reward models on held-out preference and safety sets. |
| 4 | Ignoring KL drift | A policy can become high reward but lose language quality or capability. | Track KL to the reference policy and capability regressions. |
| 5 | Optimizing only refusal rate | High refusal can hide low helpfulness and overblocking. | Measure safe compliance and benign refusal separately. |
| 6 | Using public jailbreaks as the only red team | Static attacks overfit quickly. | Mix human, automated, private, and adaptive attacks. |
| 7 | Changing policy text without versioning | Labels become incomparable across time. | Version policy, rubric, data, and model together. |
| 8 | Skipping reviewer calibration | Human feedback becomes noisy and inconsistent. | Use gold tasks, overlap, adjudication, and disagreement analysis. |
| 9 | Letting guardrails replace model training | Runtime filters cannot fix every model behavior. | Use layered defenses: data, training, policies, and gates. |
| 10 | Confusing safety monitoring with production observability | Chapter 18 feedback loops are not full MLOps dashboards. | Hand production telemetry to Chapter 19 while preserving safety feedback evidence. |

## 9. Exercises

1. (*) Red teaming searches before users find failures.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

2. (*) Safety failures are rare-event measurements.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

3. (*) Attackers adapt to defenses.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

4. (**) Red-team findings become training data.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

5. (**) Safety evaluation versus general robustness.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

6. (**) Harm taxonomy.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

7. (***) Attack prompt.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

8. (***) Target model.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

9. (***) Violation score.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

10. (***) Attack success rate.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

## 10. Why This Matters for AI

| Concept | AI Impact |
| --- | --- |
| Instruction tuning | Converts raw next-token prediction into usable assistant behavior |
| Preference learning | Optimizes choices that are hard to express as reference answers |
| KL control | Limits destructive policy drift during reward optimization |
| Red teaming | Finds harmful behavior before deployment and creates regression cases |
| Guardrails | Adds runtime control when training alone is insufficient |
| Policy versioning | Keeps safety labels auditable across changing rules |
| Human feedback | Supplies sparse but high-value evidence about user intent and risk |
| Release gates | Connects alignment work to measurable safety and capability thresholds |

## 11. Conceptual Bridge

Chapter 17 taught how to measure model behavior with benchmarks, uncertainty, robustness
tests, ablations, and online experiments. Chapter 18 uses those measurements to change
behavior through data, objectives, policies, guardrails, and human feedback.

Chapter 15 remains the home for general fine-tuning mechanics: parameter-efficient
updates, memory cost, and broad training details. This chapter narrows the focus to
post-training methods whose purpose is alignment with instructions, preferences, and
safety policies.

Chapter 19 will pick up production lineage, monitoring, observability, drift, and
serving systems. Chapter 18 stops at the safety feedback loop: how evidence becomes
alignment data or runtime policy, not how every deployed metric is stored forever.

```text
15 LLM training and fine-tuning math
        -> objectives and update mechanics
17 Evaluation and Reliability
        -> evidence about model behavior
18 Alignment and Safety
        -> SFT, preferences, red teams, policies, feedback
19 Production ML and MLOps
        -> deployment, observability, drift, retraining
```

## References

- [Red Teaming Language Models with Language Models](https://arxiv.org/abs/2202.03286)
- [Red Teaming Language Models to Reduce Harms](https://arxiv.org/abs/2209.07858)
- [SafetyPrompts](https://arxiv.org/abs/2404.05399)
- [HELM](https://arxiv.org/abs/2211.09110)
- [KL Divergence](../../09-Information-Theory/02-KL-Divergence/notes.md)
- [Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md)
- [Fine-Tuning Math](../../15-Math-for-LLMs/07-Fine-Tuning-Math/notes.md)
- [Evaluation and Reliability](../../17-Evaluation-and-Reliability/01-Capability-Benchmarks/notes.md)

