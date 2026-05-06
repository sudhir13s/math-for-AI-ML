[Back to Curriculum](../../README.md) | [Previous: Policy and Guardrails](../04-Policy-and-Guardrails/notes.md) | [Next: Data Versioning and Lineage](../../19-Production-ML-and-MLOps/01-Data-Versioning-and-Lineage/notes.md)

---

# Human in the Loop and Monitoring

> _"Alignment becomes real when feedback changes the next training set."_

## Overview

Human-in-the-loop alignment treats review, escalation, feedback, and retraining as a
controlled feedback system rather than a one-time annotation job.

Alignment and safety sit between model training and model deployment. The chapter
studies how desired behavior is specified, learned, stress-tested, constrained, and
improved through feedback.

This section is written in LaTeX Markdown. Inline mathematics uses `$...$`, and display
equations use `$$...$$`. The notes use math to expose the operational contract: which
behavior is optimized, which behavior is forbidden, and which evidence proves the system
changed.

## Prerequisites

- [Online Experimentation and AB Testing](../../17-Evaluation-and-Reliability/05-Online-Experimentation-and-AB-Testing/notes.md)
- [Policy and Guardrails](../04-Policy-and-Guardrails/notes.md)
- [Preference Optimization RLHF and DPO](../02-Preference-Optimization-RLHF-and-DPO/notes.md)
- [Data Versioning and Lineage](../../19-Production-ML-and-MLOps/01-Data-Versioning-and-Lineage/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Executable demonstrations for human in the loop and monitoring |
| [exercises.ipynb](exercises.ipynb) | Graded practice for human in the loop and monitoring |

## Learning Objectives

After completing this section, you will be able to:

- Define the core mathematical objects in human in the loop and monitoring
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
  - [1.1 Alignment as a feedback system](#11-alignment-as-a-feedback-system)
  - [1.2 Humans as sparse high-value sensors](#12-humans-as-sparse-high-value-sensors)
  - [1.3 Escalation as risk control](#13-escalation-as-risk-control)
  - [1.4 Feedback loops versus production dashboards](#14-feedback-loops-versus-production-dashboards)
  - [1.5 Why monitoring must become data](#15-why-monitoring-must-become-data)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Feedback event](#21-feedback-event)
  - [2.2 Label budget](#22-label-budget)
  - [2.3 Active learning score](#23-active-learning-score)
  - [2.4 Escalation policy](#24-escalation-policy)
  - [2.5 Preference queue](#25-preference-queue)
- [3. Feedback Collection](#3-feedback-collection)
  - [3.1 Rankings](#31-rankings)
  - [3.2 Ratings](#32-ratings)
  - [3.3 Edits](#33-edits)
  - [3.4 Demonstrations](#34-demonstrations)
  - [3.5 Natural-language critiques](#35-natural-language-critiques)
- [4. Active Learning](#4-active-learning)
  - [4.1 Uncertainty sampling](#41-uncertainty-sampling)
  - [4.2 Diversity sampling](#42-diversity-sampling)
  - [4.3 Risk-weighted sampling](#43-risk-weighted-sampling)
  - [4.4 Marginal value of labels](#44-marginal-value-of-labels)
  - [4.5 Exploration budget](#45-exploration-budget)
- [5. Human Review Workflows](#5-human-review-workflows)
  - [5.1 Escalation](#51-escalation)
  - [5.2 Triage](#52-triage)
  - [5.3 Reviewer calibration](#53-reviewer-calibration)
  - [5.4 Rubric drift](#54-rubric-drift)
  - [5.5 Adjudication](#55-adjudication)
- [6. Feedback-to-Training Loop](#6-feedback-to-training-loop)
  - [6.1 Data selection](#61-data-selection)
  - [6.2 Reward-model refresh](#62-reward-model-refresh)
  - [6.3 SFT refresh](#63-sft-refresh)
  - [6.4 Regression safety checks](#64-regression-safety-checks)
  - [6.5 Release gate](#65-release-gate)
- [7. Monitoring Boundary](#7-monitoring-boundary)
  - [7.1 Safety feedback loops here](#71-safety-feedback-loops-here)
  - [7.2 Production telemetry belongs to Chapter 19](#72-production-telemetry-belongs-to-chapter-19)
  - [7.3 Drift dashboards as inputs](#73-drift-dashboards-as-inputs)
  - [7.4 Privacy and consent](#74-privacy-and-consent)
  - [7.5 Feedback auditability](#75-feedback-auditability)
- [8. Common Mistakes](#8-common-mistakes)
- [9. Exercises](#9-exercises)
- [10. Why This Matters for AI](#10-why-this-matters-for-ai)
- [11. Conceptual Bridge](#11-conceptual-bridge)
- [References](#references)

---

## 1. Intuition

Intuition develops the part of human in the loop and monitoring that the approved TOC
assigns to Chapter 18. The emphasis is alignment behavior, safety constraints, and
feedback loops, not generic fine-tuning or production monitoring.

### 1.1 Alignment as a feedback system

Alignment as a feedback system belongs in the canonical scope of human in the loop and
monitoring. The object is the human feedback loop, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x_i,y_i,h_i). It
marks the alignment object being transformed: an instruction policy, a preference pair,
a violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
u_i = \lambda_{\mathrm{risk}} r_i + \lambda_{\mathrm{unc}} h_i + \lambda_{\mathrm{div}} d_i.
$$

For alignment as a feedback system, this formula should not be treated as a slogan. It
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
- Treat alignment as a feedback system as part of the model contract and store the exact
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

Worked reasoning pattern for alignment as a feedback system:
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

AI connection: Alignment as a feedback system is part of the post-training stack used by
modern assistant systems. It links the base language model to human intent, safety
policy, and deployment constraints without pretending that a single loss can capture all
values. The goal is not perfect alignment by formula; it is a repeatable loop where
evidence, objectives, and safeguards improve together.

### 1.2 Humans as sparse high-value sensors

Humans as sparse high-value sensors belongs in the canonical scope of human in the loop
and monitoring. The object is the human feedback loop, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x_i,y_i,h_i). It
marks the alignment object being transformed: an instruction policy, a preference pair,
a violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
u_i = \lambda_{\mathrm{risk}} r_i + \lambda_{\mathrm{unc}} h_i + \lambda_{\mathrm{div}} d_i.
$$

For humans as sparse high-value sensors, this formula should not be treated as a slogan.
It defines which tokens, responses, comparisons, or decisions receive gradient or
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
- Treat humans as sparse high-value sensors as part of the model contract and store the
exact data version.
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

Worked reasoning pattern for humans as sparse high-value sensors:
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

AI connection: Humans as sparse high-value sensors is part of the post-training stack
used by modern assistant systems. It links the base language model to human intent,
safety policy, and deployment constraints without pretending that a single loss can
capture all values. The goal is not perfect alignment by formula; it is a repeatable
loop where evidence, objectives, and safeguards improve together.

### 1.3 Escalation as risk control

Escalation as risk control belongs in the canonical scope of human in the loop and
monitoring. The object is the human feedback loop, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x_i,y_i,h_i). It
marks the alignment object being transformed: an instruction policy, a preference pair,
a violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
u_i = \lambda_{\mathrm{risk}} r_i + \lambda_{\mathrm{unc}} h_i + \lambda_{\mathrm{div}} d_i.
$$

For escalation as risk control, this formula should not be treated as a slogan. It
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
- Treat escalation as risk control as part of the model contract and store the exact
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

Worked reasoning pattern for escalation as risk control:
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

AI connection: Escalation as risk control is part of the post-training stack used by
modern assistant systems. It links the base language model to human intent, safety
policy, and deployment constraints without pretending that a single loss can capture all
values. The goal is not perfect alignment by formula; it is a repeatable loop where
evidence, objectives, and safeguards improve together.

### 1.4 Feedback loops versus production dashboards

Feedback loops versus production dashboards belongs in the canonical scope of human in
the loop and monitoring. The object is the human feedback loop, not merely a prompt
trick or a moderation label. We study how data, losses, policies, review processes, and
safety constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x_i,y_i,h_i). It
marks the alignment object being transformed: an instruction policy, a preference pair,
a violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
u_i = \lambda_{\mathrm{risk}} r_i + \lambda_{\mathrm{unc}} h_i + \lambda_{\mathrm{div}} d_i.
$$

For feedback loops versus production dashboards, this formula should not be treated as a
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
- Treat feedback loops versus production dashboards as part of the model contract and
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

Worked reasoning pattern for feedback loops versus production dashboards:
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

AI connection: Feedback loops versus production dashboards is part of the post-training
stack used by modern assistant systems. It links the base language model to human
intent, safety policy, and deployment constraints without pretending that a single loss
can capture all values. The goal is not perfect alignment by formula; it is a repeatable
loop where evidence, objectives, and safeguards improve together.

### 1.5 Why monitoring must become data

Why monitoring must become data belongs in the canonical scope of human in the loop and
monitoring. The object is the human feedback loop, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x_i,y_i,h_i). It
marks the alignment object being transformed: an instruction policy, a preference pair,
a violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
u_i = \lambda_{\mathrm{risk}} r_i + \lambda_{\mathrm{unc}} h_i + \lambda_{\mathrm{div}} d_i.
$$

For why monitoring must become data, this formula should not be treated as a slogan. It
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
- Treat why monitoring must become data as part of the model contract and store the
exact data version.
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

Worked reasoning pattern for why monitoring must become data:
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

AI connection: Why monitoring must become data is part of the post-training stack used
by modern assistant systems. It links the base language model to human intent, safety
policy, and deployment constraints without pretending that a single loss can capture all
values. The goal is not perfect alignment by formula; it is a repeatable loop where
evidence, objectives, and safeguards improve together.

## 2. Formal Definitions

Formal Definitions develops the part of human in the loop and monitoring that the
approved TOC assigns to Chapter 18. The emphasis is alignment behavior, safety
constraints, and feedback loops, not generic fine-tuning or production monitoring.

### 2.1 Feedback event

Feedback event belongs in the canonical scope of human in the loop and monitoring. The
object is the human feedback loop, not merely a prompt trick or a moderation label. We
study how data, losses, policies, review processes, and safety constraints shape a
model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x_i,y_i,h_i). It
marks the alignment object being transformed: an instruction policy, a preference pair,
a violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
u_i = \lambda_{\mathrm{risk}} r_i + \lambda_{\mathrm{unc}} h_i + \lambda_{\mathrm{div}} d_i.
$$

For feedback event, this formula should not be treated as a slogan. It defines which
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
- Treat feedback event as part of the model contract and store the exact data version.
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

Worked reasoning pattern for feedback event:
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

AI connection: Feedback event is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 2.2 Label budget

Label budget belongs in the canonical scope of human in the loop and monitoring. The
object is the human feedback loop, not merely a prompt trick or a moderation label. We
study how data, losses, policies, review processes, and safety constraints shape a
model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x_i,y_i,h_i). It
marks the alignment object being transformed: an instruction policy, a preference pair,
a violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
u_i = \lambda_{\mathrm{risk}} r_i + \lambda_{\mathrm{unc}} h_i + \lambda_{\mathrm{div}} d_i.
$$

For label budget, this formula should not be treated as a slogan. It defines which
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
- Treat label budget as part of the model contract and store the exact data version.
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

Worked reasoning pattern for label budget:
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

AI connection: Label budget is part of the post-training stack used by modern assistant
systems. It links the base language model to human intent, safety policy, and deployment
constraints without pretending that a single loss can capture all values. The goal is
not perfect alignment by formula; it is a repeatable loop where evidence, objectives,
and safeguards improve together.

### 2.3 Active learning score

Active learning score belongs in the canonical scope of human in the loop and
monitoring. The object is the human feedback loop, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x_i,y_i,h_i). It
marks the alignment object being transformed: an instruction policy, a preference pair,
a violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
u_i = \lambda_{\mathrm{risk}} r_i + \lambda_{\mathrm{unc}} h_i + \lambda_{\mathrm{div}} d_i.
$$

For active learning score, this formula should not be treated as a slogan. It defines
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
- Treat active learning score as part of the model contract and store the exact data
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

Worked reasoning pattern for active learning score:
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

AI connection: Active learning score is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 2.4 Escalation policy

Escalation policy belongs in the canonical scope of human in the loop and monitoring.
The object is the human feedback loop, not merely a prompt trick or a moderation label.
We study how data, losses, policies, review processes, and safety constraints shape a
model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x_i,y_i,h_i). It
marks the alignment object being transformed: an instruction policy, a preference pair,
a violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
u_i = \lambda_{\mathrm{risk}} r_i + \lambda_{\mathrm{unc}} h_i + \lambda_{\mathrm{div}} d_i.
$$

For escalation policy, this formula should not be treated as a slogan. It defines which
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
- Treat escalation policy as part of the model contract and store the exact data
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

Worked reasoning pattern for escalation policy:
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

AI connection: Escalation policy is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 2.5 Preference queue

Preference queue belongs in the canonical scope of human in the loop and monitoring. The
object is the human feedback loop, not merely a prompt trick or a moderation label. We
study how data, losses, policies, review processes, and safety constraints shape a
model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x_i,y_i,h_i). It
marks the alignment object being transformed: an instruction policy, a preference pair,
a violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
u_i = \lambda_{\mathrm{risk}} r_i + \lambda_{\mathrm{unc}} h_i + \lambda_{\mathrm{div}} d_i.
$$

For preference queue, this formula should not be treated as a slogan. It defines which
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
- Treat preference queue as part of the model contract and store the exact data version.
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

Worked reasoning pattern for preference queue:
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

AI connection: Preference queue is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

## 3. Feedback Collection

Feedback Collection develops the part of human in the loop and monitoring that the
approved TOC assigns to Chapter 18. The emphasis is alignment behavior, safety
constraints, and feedback loops, not generic fine-tuning or production monitoring.

### 3.1 Rankings

Rankings belongs in the canonical scope of human in the loop and monitoring. The object
is the human feedback loop, not merely a prompt trick or a moderation label. We study
how data, losses, policies, review processes, and safety constraints shape a model's
conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x_i,y_i,h_i). It
marks the alignment object being transformed: an instruction policy, a preference pair,
a violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
u_i = \lambda_{\mathrm{risk}} r_i + \lambda_{\mathrm{unc}} h_i + \lambda_{\mathrm{div}} d_i.
$$

For rankings, this formula should not be treated as a slogan. It defines which tokens,
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
- Treat rankings as part of the model contract and store the exact data version.
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

Worked reasoning pattern for rankings:
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

AI connection: Rankings is part of the post-training stack used by modern assistant
systems. It links the base language model to human intent, safety policy, and deployment
constraints without pretending that a single loss can capture all values. The goal is
not perfect alignment by formula; it is a repeatable loop where evidence, objectives,
and safeguards improve together.

### 3.2 Ratings

Ratings belongs in the canonical scope of human in the loop and monitoring. The object
is the human feedback loop, not merely a prompt trick or a moderation label. We study
how data, losses, policies, review processes, and safety constraints shape a model's
conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x_i,y_i,h_i). It
marks the alignment object being transformed: an instruction policy, a preference pair,
a violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
u_i = \lambda_{\mathrm{risk}} r_i + \lambda_{\mathrm{unc}} h_i + \lambda_{\mathrm{div}} d_i.
$$

For ratings, this formula should not be treated as a slogan. It defines which tokens,
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
- Treat ratings as part of the model contract and store the exact data version.
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

Worked reasoning pattern for ratings:
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

AI connection: Ratings is part of the post-training stack used by modern assistant
systems. It links the base language model to human intent, safety policy, and deployment
constraints without pretending that a single loss can capture all values. The goal is
not perfect alignment by formula; it is a repeatable loop where evidence, objectives,
and safeguards improve together.

### 3.3 Edits

Edits belongs in the canonical scope of human in the loop and monitoring. The object is
the human feedback loop, not merely a prompt trick or a moderation label. We study how
data, losses, policies, review processes, and safety constraints shape a model's
conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x_i,y_i,h_i). It
marks the alignment object being transformed: an instruction policy, a preference pair,
a violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
u_i = \lambda_{\mathrm{risk}} r_i + \lambda_{\mathrm{unc}} h_i + \lambda_{\mathrm{div}} d_i.
$$

For edits, this formula should not be treated as a slogan. It defines which tokens,
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
- Treat edits as part of the model contract and store the exact data version.
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

Worked reasoning pattern for edits:
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

AI connection: Edits is part of the post-training stack used by modern assistant
systems. It links the base language model to human intent, safety policy, and deployment
constraints without pretending that a single loss can capture all values. The goal is
not perfect alignment by formula; it is a repeatable loop where evidence, objectives,
and safeguards improve together.

### 3.4 Demonstrations

Demonstrations belongs in the canonical scope of human in the loop and monitoring. The
object is the human feedback loop, not merely a prompt trick or a moderation label. We
study how data, losses, policies, review processes, and safety constraints shape a
model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x_i,y_i,h_i). It
marks the alignment object being transformed: an instruction policy, a preference pair,
a violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
u_i = \lambda_{\mathrm{risk}} r_i + \lambda_{\mathrm{unc}} h_i + \lambda_{\mathrm{div}} d_i.
$$

For demonstrations, this formula should not be treated as a slogan. It defines which
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
- Treat demonstrations as part of the model contract and store the exact data version.
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

Worked reasoning pattern for demonstrations:
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

AI connection: Demonstrations is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 3.5 Natural-language critiques

Natural-language critiques belongs in the canonical scope of human in the loop and
monitoring. The object is the human feedback loop, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x_i,y_i,h_i). It
marks the alignment object being transformed: an instruction policy, a preference pair,
a violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
u_i = \lambda_{\mathrm{risk}} r_i + \lambda_{\mathrm{unc}} h_i + \lambda_{\mathrm{div}} d_i.
$$

For natural-language critiques, this formula should not be treated as a slogan. It
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
- Treat natural-language critiques as part of the model contract and store the exact
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

Worked reasoning pattern for natural-language critiques:
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

AI connection: Natural-language critiques is part of the post-training stack used by
modern assistant systems. It links the base language model to human intent, safety
policy, and deployment constraints without pretending that a single loss can capture all
values. The goal is not perfect alignment by formula; it is a repeatable loop where
evidence, objectives, and safeguards improve together.

## 4. Active Learning

Active Learning develops the part of human in the loop and monitoring that the approved
TOC assigns to Chapter 18. The emphasis is alignment behavior, safety constraints, and
feedback loops, not generic fine-tuning or production monitoring.

### 4.1 Uncertainty sampling

Uncertainty sampling belongs in the canonical scope of human in the loop and monitoring.
The object is the human feedback loop, not merely a prompt trick or a moderation label.
We study how data, losses, policies, review processes, and safety constraints shape a
model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x_i,y_i,h_i). It
marks the alignment object being transformed: an instruction policy, a preference pair,
a violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
u_i = \lambda_{\mathrm{risk}} r_i + \lambda_{\mathrm{unc}} h_i + \lambda_{\mathrm{div}} d_i.
$$

For uncertainty sampling, this formula should not be treated as a slogan. It defines
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
- Treat uncertainty sampling as part of the model contract and store the exact data
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

Worked reasoning pattern for uncertainty sampling:
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

AI connection: Uncertainty sampling is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 4.2 Diversity sampling

Diversity sampling belongs in the canonical scope of human in the loop and monitoring.
The object is the human feedback loop, not merely a prompt trick or a moderation label.
We study how data, losses, policies, review processes, and safety constraints shape a
model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x_i,y_i,h_i). It
marks the alignment object being transformed: an instruction policy, a preference pair,
a violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
u_i = \lambda_{\mathrm{risk}} r_i + \lambda_{\mathrm{unc}} h_i + \lambda_{\mathrm{div}} d_i.
$$

For diversity sampling, this formula should not be treated as a slogan. It defines which
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
- Treat diversity sampling as part of the model contract and store the exact data
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

Worked reasoning pattern for diversity sampling:
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

AI connection: Diversity sampling is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 4.3 Risk-weighted sampling

Risk-weighted sampling belongs in the canonical scope of human in the loop and
monitoring. The object is the human feedback loop, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x_i,y_i,h_i). It
marks the alignment object being transformed: an instruction policy, a preference pair,
a violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
u_i = \lambda_{\mathrm{risk}} r_i + \lambda_{\mathrm{unc}} h_i + \lambda_{\mathrm{div}} d_i.
$$

For risk-weighted sampling, this formula should not be treated as a slogan. It defines
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
- Treat risk-weighted sampling as part of the model contract and store the exact data
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

Worked reasoning pattern for risk-weighted sampling:
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

AI connection: Risk-weighted sampling is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 4.4 Marginal value of labels

Marginal value of labels belongs in the canonical scope of human in the loop and
monitoring. The object is the human feedback loop, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x_i,y_i,h_i). It
marks the alignment object being transformed: an instruction policy, a preference pair,
a violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
u_i = \lambda_{\mathrm{risk}} r_i + \lambda_{\mathrm{unc}} h_i + \lambda_{\mathrm{div}} d_i.
$$

For marginal value of labels, this formula should not be treated as a slogan. It defines
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
- Treat marginal value of labels as part of the model contract and store the exact data
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

Worked reasoning pattern for marginal value of labels:
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

AI connection: Marginal value of labels is part of the post-training stack used by
modern assistant systems. It links the base language model to human intent, safety
policy, and deployment constraints without pretending that a single loss can capture all
values. The goal is not perfect alignment by formula; it is a repeatable loop where
evidence, objectives, and safeguards improve together.

### 4.5 Exploration budget

Exploration budget belongs in the canonical scope of human in the loop and monitoring.
The object is the human feedback loop, not merely a prompt trick or a moderation label.
We study how data, losses, policies, review processes, and safety constraints shape a
model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x_i,y_i,h_i). It
marks the alignment object being transformed: an instruction policy, a preference pair,
a violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
u_i = \lambda_{\mathrm{risk}} r_i + \lambda_{\mathrm{unc}} h_i + \lambda_{\mathrm{div}} d_i.
$$

For exploration budget, this formula should not be treated as a slogan. It defines which
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
- Treat exploration budget as part of the model contract and store the exact data
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

Worked reasoning pattern for exploration budget:
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

AI connection: Exploration budget is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

## 5. Human Review Workflows

Human Review Workflows develops the part of human in the loop and monitoring that the
approved TOC assigns to Chapter 18. The emphasis is alignment behavior, safety
constraints, and feedback loops, not generic fine-tuning or production monitoring.

### 5.1 Escalation

Escalation belongs in the canonical scope of human in the loop and monitoring. The
object is the human feedback loop, not merely a prompt trick or a moderation label. We
study how data, losses, policies, review processes, and safety constraints shape a
model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x_i,y_i,h_i). It
marks the alignment object being transformed: an instruction policy, a preference pair,
a violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
u_i = \lambda_{\mathrm{risk}} r_i + \lambda_{\mathrm{unc}} h_i + \lambda_{\mathrm{div}} d_i.
$$

For escalation, this formula should not be treated as a slogan. It defines which tokens,
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
- Treat escalation as part of the model contract and store the exact data version.
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

Worked reasoning pattern for escalation:
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

AI connection: Escalation is part of the post-training stack used by modern assistant
systems. It links the base language model to human intent, safety policy, and deployment
constraints without pretending that a single loss can capture all values. The goal is
not perfect alignment by formula; it is a repeatable loop where evidence, objectives,
and safeguards improve together.

### 5.2 Triage

Triage belongs in the canonical scope of human in the loop and monitoring. The object is
the human feedback loop, not merely a prompt trick or a moderation label. We study how
data, losses, policies, review processes, and safety constraints shape a model's
conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x_i,y_i,h_i). It
marks the alignment object being transformed: an instruction policy, a preference pair,
a violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
u_i = \lambda_{\mathrm{risk}} r_i + \lambda_{\mathrm{unc}} h_i + \lambda_{\mathrm{div}} d_i.
$$

For triage, this formula should not be treated as a slogan. It defines which tokens,
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
- Treat triage as part of the model contract and store the exact data version.
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

Worked reasoning pattern for triage:
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

AI connection: Triage is part of the post-training stack used by modern assistant
systems. It links the base language model to human intent, safety policy, and deployment
constraints without pretending that a single loss can capture all values. The goal is
not perfect alignment by formula; it is a repeatable loop where evidence, objectives,
and safeguards improve together.

### 5.3 Reviewer calibration

Reviewer calibration belongs in the canonical scope of human in the loop and monitoring.
The object is the human feedback loop, not merely a prompt trick or a moderation label.
We study how data, losses, policies, review processes, and safety constraints shape a
model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x_i,y_i,h_i). It
marks the alignment object being transformed: an instruction policy, a preference pair,
a violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
u_i = \lambda_{\mathrm{risk}} r_i + \lambda_{\mathrm{unc}} h_i + \lambda_{\mathrm{div}} d_i.
$$

For reviewer calibration, this formula should not be treated as a slogan. It defines
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
- Treat reviewer calibration as part of the model contract and store the exact data
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

Worked reasoning pattern for reviewer calibration:
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

AI connection: Reviewer calibration is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 5.4 Rubric drift

Rubric drift belongs in the canonical scope of human in the loop and monitoring. The
object is the human feedback loop, not merely a prompt trick or a moderation label. We
study how data, losses, policies, review processes, and safety constraints shape a
model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x_i,y_i,h_i). It
marks the alignment object being transformed: an instruction policy, a preference pair,
a violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
u_i = \lambda_{\mathrm{risk}} r_i + \lambda_{\mathrm{unc}} h_i + \lambda_{\mathrm{div}} d_i.
$$

For rubric drift, this formula should not be treated as a slogan. It defines which
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
- Treat rubric drift as part of the model contract and store the exact data version.
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

Worked reasoning pattern for rubric drift:
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

AI connection: Rubric drift is part of the post-training stack used by modern assistant
systems. It links the base language model to human intent, safety policy, and deployment
constraints without pretending that a single loss can capture all values. The goal is
not perfect alignment by formula; it is a repeatable loop where evidence, objectives,
and safeguards improve together.

### 5.5 Adjudication

Adjudication belongs in the canonical scope of human in the loop and monitoring. The
object is the human feedback loop, not merely a prompt trick or a moderation label. We
study how data, losses, policies, review processes, and safety constraints shape a
model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x_i,y_i,h_i). It
marks the alignment object being transformed: an instruction policy, a preference pair,
a violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
u_i = \lambda_{\mathrm{risk}} r_i + \lambda_{\mathrm{unc}} h_i + \lambda_{\mathrm{div}} d_i.
$$

For adjudication, this formula should not be treated as a slogan. It defines which
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
- Treat adjudication as part of the model contract and store the exact data version.
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

Worked reasoning pattern for adjudication:
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

AI connection: Adjudication is part of the post-training stack used by modern assistant
systems. It links the base language model to human intent, safety policy, and deployment
constraints without pretending that a single loss can capture all values. The goal is
not perfect alignment by formula; it is a repeatable loop where evidence, objectives,
and safeguards improve together.

## 6. Feedback-to-Training Loop

Feedback-to-Training Loop develops the part of human in the loop and monitoring that the
approved TOC assigns to Chapter 18. The emphasis is alignment behavior, safety
constraints, and feedback loops, not generic fine-tuning or production monitoring.

### 6.1 Data selection

Data selection belongs in the canonical scope of human in the loop and monitoring. The
object is the human feedback loop, not merely a prompt trick or a moderation label. We
study how data, losses, policies, review processes, and safety constraints shape a
model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x_i,y_i,h_i). It
marks the alignment object being transformed: an instruction policy, a preference pair,
a violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
u_i = \lambda_{\mathrm{risk}} r_i + \lambda_{\mathrm{unc}} h_i + \lambda_{\mathrm{div}} d_i.
$$

For data selection, this formula should not be treated as a slogan. It defines which
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
- Treat data selection as part of the model contract and store the exact data version.
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

Worked reasoning pattern for data selection:
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

AI connection: Data selection is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 6.2 Reward-model refresh

Reward-model refresh belongs in the canonical scope of human in the loop and monitoring.
The object is the human feedback loop, not merely a prompt trick or a moderation label.
We study how data, losses, policies, review processes, and safety constraints shape a
model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x_i,y_i,h_i). It
marks the alignment object being transformed: an instruction policy, a preference pair,
a violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
u_i = \lambda_{\mathrm{risk}} r_i + \lambda_{\mathrm{unc}} h_i + \lambda_{\mathrm{div}} d_i.
$$

For reward-model refresh, this formula should not be treated as a slogan. It defines
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
- Treat reward-model refresh as part of the model contract and store the exact data
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

Worked reasoning pattern for reward-model refresh:
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

AI connection: Reward-model refresh is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 6.3 SFT refresh

SFT refresh belongs in the canonical scope of human in the loop and monitoring. The
object is the human feedback loop, not merely a prompt trick or a moderation label. We
study how data, losses, policies, review processes, and safety constraints shape a
model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x_i,y_i,h_i). It
marks the alignment object being transformed: an instruction policy, a preference pair,
a violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
u_i = \lambda_{\mathrm{risk}} r_i + \lambda_{\mathrm{unc}} h_i + \lambda_{\mathrm{div}} d_i.
$$

For sft refresh, this formula should not be treated as a slogan. It defines which
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
- Treat sft refresh as part of the model contract and store the exact data version.
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

Worked reasoning pattern for sft refresh:
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

AI connection: SFT refresh is part of the post-training stack used by modern assistant
systems. It links the base language model to human intent, safety policy, and deployment
constraints without pretending that a single loss can capture all values. The goal is
not perfect alignment by formula; it is a repeatable loop where evidence, objectives,
and safeguards improve together.

### 6.4 Regression safety checks

Regression safety checks belongs in the canonical scope of human in the loop and
monitoring. The object is the human feedback loop, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x_i,y_i,h_i). It
marks the alignment object being transformed: an instruction policy, a preference pair,
a violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
u_i = \lambda_{\mathrm{risk}} r_i + \lambda_{\mathrm{unc}} h_i + \lambda_{\mathrm{div}} d_i.
$$

For regression safety checks, this formula should not be treated as a slogan. It defines
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
- Treat regression safety checks as part of the model contract and store the exact data
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

Worked reasoning pattern for regression safety checks:
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

AI connection: Regression safety checks is part of the post-training stack used by
modern assistant systems. It links the base language model to human intent, safety
policy, and deployment constraints without pretending that a single loss can capture all
values. The goal is not perfect alignment by formula; it is a repeatable loop where
evidence, objectives, and safeguards improve together.

### 6.5 Release gate

Release gate belongs in the canonical scope of human in the loop and monitoring. The
object is the human feedback loop, not merely a prompt trick or a moderation label. We
study how data, losses, policies, review processes, and safety constraints shape a
model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x_i,y_i,h_i). It
marks the alignment object being transformed: an instruction policy, a preference pair,
a violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
u_i = \lambda_{\mathrm{risk}} r_i + \lambda_{\mathrm{unc}} h_i + \lambda_{\mathrm{div}} d_i.
$$

For release gate, this formula should not be treated as a slogan. It defines which
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
- Treat release gate as part of the model contract and store the exact data version.
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

Worked reasoning pattern for release gate:
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

AI connection: Release gate is part of the post-training stack used by modern assistant
systems. It links the base language model to human intent, safety policy, and deployment
constraints without pretending that a single loss can capture all values. The goal is
not perfect alignment by formula; it is a repeatable loop where evidence, objectives,
and safeguards improve together.

## 7. Monitoring Boundary

Monitoring Boundary develops the part of human in the loop and monitoring that the
approved TOC assigns to Chapter 18. The emphasis is alignment behavior, safety
constraints, and feedback loops, not generic fine-tuning or production monitoring.

### 7.1 Safety feedback loops here

Safety feedback loops here belongs in the canonical scope of human in the loop and
monitoring. The object is the human feedback loop, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x_i,y_i,h_i). It
marks the alignment object being transformed: an instruction policy, a preference pair,
a violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
u_i = \lambda_{\mathrm{risk}} r_i + \lambda_{\mathrm{unc}} h_i + \lambda_{\mathrm{div}} d_i.
$$

For safety feedback loops here, this formula should not be treated as a slogan. It
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
- Treat safety feedback loops here as part of the model contract and store the exact
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

Worked reasoning pattern for safety feedback loops here:
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

AI connection: Safety feedback loops here is part of the post-training stack used by
modern assistant systems. It links the base language model to human intent, safety
policy, and deployment constraints without pretending that a single loss can capture all
values. The goal is not perfect alignment by formula; it is a repeatable loop where
evidence, objectives, and safeguards improve together.

### 7.2 Production telemetry belongs to Chapter 19

Production telemetry belongs to Chapter 19 belongs in the canonical scope of human in
the loop and monitoring. The object is the human feedback loop, not merely a prompt
trick or a moderation label. We study how data, losses, policies, review processes, and
safety constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x_i,y_i,h_i). It
marks the alignment object being transformed: an instruction policy, a preference pair,
a violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
u_i = \lambda_{\mathrm{risk}} r_i + \lambda_{\mathrm{unc}} h_i + \lambda_{\mathrm{div}} d_i.
$$

For production telemetry belongs to chapter 19, this formula should not be treated as a
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
- Treat production telemetry belongs to chapter 19 as part of the model contract and
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

Worked reasoning pattern for production telemetry belongs to chapter 19:
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

AI connection: Production telemetry belongs to Chapter 19 is part of the post-training
stack used by modern assistant systems. It links the base language model to human
intent, safety policy, and deployment constraints without pretending that a single loss
can capture all values. The goal is not perfect alignment by formula; it is a repeatable
loop where evidence, objectives, and safeguards improve together.

### 7.3 Drift dashboards as inputs

Drift dashboards as inputs belongs in the canonical scope of human in the loop and
monitoring. The object is the human feedback loop, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x_i,y_i,h_i). It
marks the alignment object being transformed: an instruction policy, a preference pair,
a violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
u_i = \lambda_{\mathrm{risk}} r_i + \lambda_{\mathrm{unc}} h_i + \lambda_{\mathrm{div}} d_i.
$$

For drift dashboards as inputs, this formula should not be treated as a slogan. It
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
- Treat drift dashboards as inputs as part of the model contract and store the exact
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

Worked reasoning pattern for drift dashboards as inputs:
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

AI connection: Drift dashboards as inputs is part of the post-training stack used by
modern assistant systems. It links the base language model to human intent, safety
policy, and deployment constraints without pretending that a single loss can capture all
values. The goal is not perfect alignment by formula; it is a repeatable loop where
evidence, objectives, and safeguards improve together.

### 7.4 Privacy and consent

Privacy and consent belongs in the canonical scope of human in the loop and monitoring.
The object is the human feedback loop, not merely a prompt trick or a moderation label.
We study how data, losses, policies, review processes, and safety constraints shape a
model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x_i,y_i,h_i). It
marks the alignment object being transformed: an instruction policy, a preference pair,
a violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
u_i = \lambda_{\mathrm{risk}} r_i + \lambda_{\mathrm{unc}} h_i + \lambda_{\mathrm{div}} d_i.
$$

For privacy and consent, this formula should not be treated as a slogan. It defines
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
- Treat privacy and consent as part of the model contract and store the exact data
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

Worked reasoning pattern for privacy and consent:
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

AI connection: Privacy and consent is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 7.5 Feedback auditability

Feedback auditability belongs in the canonical scope of human in the loop and
monitoring. The object is the human feedback loop, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x_i,y_i,h_i). It
marks the alignment object being transformed: an instruction policy, a preference pair,
a violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
u_i = \lambda_{\mathrm{risk}} r_i + \lambda_{\mathrm{unc}} h_i + \lambda_{\mathrm{div}} d_i.
$$

For feedback auditability, this formula should not be treated as a slogan. It defines
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
- Treat feedback auditability as part of the model contract and store the exact data
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

Worked reasoning pattern for feedback auditability:
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

AI connection: Feedback auditability is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

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

1. (*) Alignment as a feedback system.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

2. (*) Humans as sparse high-value sensors.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

3. (*) Escalation as risk control.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

4. (**) Feedback loops versus production dashboards.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

5. (**) Why monitoring must become data.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

6. (**) Feedback event.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

7. (***) Label budget.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

8. (***) Active learning score.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

9. (***) Escalation policy.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

10. (***) Preference queue.
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

- [Learning from human preferences](https://arxiv.org/abs/1706.03741)
- [WebGPT](https://arxiv.org/abs/2112.09332)
- [RewardBench](https://arxiv.org/abs/2403.13787)
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
- [Online Experimentation and AB Testing](../../17-Evaluation-and-Reliability/05-Online-Experimentation-and-AB-Testing/notes.md)
- [KL Divergence](../../09-Information-Theory/02-KL-Divergence/notes.md)
- [Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md)
- [Fine-Tuning Math](../../15-Math-for-LLMs/07-Fine-Tuning-Math/notes.md)
- [Evaluation and Reliability](../../17-Evaluation-and-Reliability/01-Capability-Benchmarks/notes.md)

