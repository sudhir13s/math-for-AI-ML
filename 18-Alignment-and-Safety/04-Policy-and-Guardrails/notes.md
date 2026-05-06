[Back to Curriculum](../../README.md) | [Previous: Red Teaming and Safety Evaluations](../03-Red-Teaming-and-Safety-Evaluations/notes.md) | [Next: Human in the Loop and Monitoring](../05-Human-in-the-Loop-and-Monitoring/notes.md)

---

# Policy and Guardrails

> _"A policy is a behavioral specification; a guardrail is its runtime test."_

## Overview

Policies define desired and disallowed behavior, while guardrails translate those rules
into classifiers, validators, gates, and intervention actions.

Alignment and safety sit between model training and model deployment. The chapter
studies how desired behavior is specified, learned, stress-tested, constrained, and
improved through feedback.

This section is written in LaTeX Markdown. Inline mathematics uses `$...$`, and display
equations use `$$...$$`. The notes use math to expose the operational contract: which
behavior is optimized, which behavior is forbidden, and which evidence proves the system
changed.

## Prerequisites

- [Red Teaming and Safety Evaluations](../03-Red-Teaming-and-Safety-Evaluations/notes.md)
- [Calibration and Uncertainty](../../17-Evaluation-and-Reliability/02-Calibration-and-Uncertainty/notes.md)
- [Hypothesis Testing](../../07-Statistics/03-Hypothesis-Testing/notes.md)
- [Language Model Probability](../../15-Math-for-LLMs/05-Language-Model-Probability/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Executable demonstrations for policy and guardrails |
| [exercises.ipynb](exercises.ipynb) | Graded practice for policy and guardrails |

## Learning Objectives

After completing this section, you will be able to:

- Define the core mathematical objects in policy and guardrails
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
  - [1.1 Alignment needs explicit policy](#11-alignment-needs-explicit-policy)
  - [1.2 Guardrails as runtime control](#12-guardrails-as-runtime-control)
  - [1.3 Policy hierarchy and conflict resolution](#13-policy-hierarchy-and-conflict-resolution)
  - [1.4 Safety is not only refusal](#14-safety-is-not-only-refusal)
  - [1.5 Why system prompts alone are insufficient](#15-why-system-prompts-alone-are-insufficient)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Policy rule](#21-policy-rule)
  - [2.2 Allowed and disallowed sets](#22-allowed-and-disallowed-sets)
  - [2.3 Classifier $c(x,y)$](#23-classifier-cxy)
  - [2.4 Guardrail action $a$](#24-guardrail-action-a)
  - [2.5 Risk threshold $\tau$](#25-risk-threshold-tau)
- [3. Policy Taxonomies](#3-policy-taxonomies)
  - [3.1 Harm categories](#31-harm-categories)
  - [3.2 Capability boundaries](#32-capability-boundaries)
  - [3.3 User intent](#33-user-intent)
  - [3.4 Context sensitivity](#34-context-sensitivity)
  - [3.5 Jurisdiction notes](#35-jurisdiction-notes)
- [4. Constitutional AI and Rule-Based Training](#4-constitutional-ai-and-rule-based-training)
  - [4.1 Critique-revise loops](#41-critique-revise-loops)
  - [4.2 AI feedback](#42-ai-feedback)
  - [4.3 Rule hierarchy](#43-rule-hierarchy)
  - [4.4 Policy distillation](#44-policy-distillation)
  - [4.5 Constitutional evaluation](#45-constitutional-evaluation)
- [5. Guardrail Architectures](#5-guardrail-architectures)
  - [5.1 Input filters](#51-input-filters)
  - [5.2 Output filters](#52-output-filters)
  - [5.3 Tool gates](#53-tool-gates)
  - [5.4 Retrieval gates](#54-retrieval-gates)
  - [5.5 Structured validators](#55-structured-validators)
- [6. Moderation Models](#6-moderation-models)
  - [6.1 Llama Guard](#61-llama-guard)
  - [6.2 ShieldGemma](#62-shieldgemma)
  - [6.3 AEGIS-style taxonomies](#63-aegis-style-taxonomies)
  - [6.4 Classifier calibration](#64-classifier-calibration)
  - [6.5 Multilingual moderation](#65-multilingual-moderation)
- [7. Tradeoffs](#7-tradeoffs)
  - [7.1 Safety versus helpfulness](#71-safety-versus-helpfulness)
  - [7.2 Refusal versus compliance](#72-refusal-versus-compliance)
  - [7.3 Latency](#73-latency)
  - [7.4 Adversarial bypass](#74-adversarial-bypass)
  - [7.5 Audit logs](#75-audit-logs)
- [8. Common Mistakes](#8-common-mistakes)
- [9. Exercises](#9-exercises)
- [10. Why This Matters for AI](#10-why-this-matters-for-ai)
- [11. Conceptual Bridge](#11-conceptual-bridge)
- [References](#references)

---

## 1. Intuition

Intuition develops the part of policy and guardrails that the approved TOC assigns to
Chapter 18. The emphasis is alignment behavior, safety constraints, and feedback loops,
not generic fine-tuning or production monitoring.

### 1.1 Alignment needs explicit policy

Alignment needs explicit policy belongs in the canonical scope of policy and guardrails.
The object is the policy-constrained generation system, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol c(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
a(x,y)=\begin{cases}\mathrm{allow},&c(x,y)<\tau\\ \mathrm{block},&c(x,y)\ge \tau\end{cases}.
$$

For alignment needs explicit policy, this formula should not be treated as a slogan. It
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
- Treat alignment needs explicit policy as part of the model contract and store the
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

Worked reasoning pattern for alignment needs explicit policy:
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

AI connection: Alignment needs explicit policy is part of the post-training stack used
by modern assistant systems. It links the base language model to human intent, safety
policy, and deployment constraints without pretending that a single loss can capture all
values. The goal is not perfect alignment by formula; it is a repeatable loop where
evidence, objectives, and safeguards improve together.

### 1.2 Guardrails as runtime control

Guardrails as runtime control belongs in the canonical scope of policy and guardrails.
The object is the policy-constrained generation system, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol c(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
a(x,y)=\begin{cases}\mathrm{allow},&c(x,y)<\tau\\ \mathrm{block},&c(x,y)\ge \tau\end{cases}.
$$

For guardrails as runtime control, this formula should not be treated as a slogan. It
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
- Treat guardrails as runtime control as part of the model contract and store the exact
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

Worked reasoning pattern for guardrails as runtime control:
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

AI connection: Guardrails as runtime control is part of the post-training stack used by
modern assistant systems. It links the base language model to human intent, safety
policy, and deployment constraints without pretending that a single loss can capture all
values. The goal is not perfect alignment by formula; it is a repeatable loop where
evidence, objectives, and safeguards improve together.

### 1.3 Policy hierarchy and conflict resolution

Policy hierarchy and conflict resolution belongs in the canonical scope of policy and
guardrails. The object is the policy-constrained generation system, not merely a prompt
trick or a moderation label. We study how data, losses, policies, review processes, and
safety constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol c(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
a(x,y)=\begin{cases}\mathrm{allow},&c(x,y)<\tau\\ \mathrm{block},&c(x,y)\ge \tau\end{cases}.
$$

For policy hierarchy and conflict resolution, this formula should not be treated as a
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
- Treat policy hierarchy and conflict resolution as part of the model contract and store
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

Worked reasoning pattern for policy hierarchy and conflict resolution:
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

AI connection: Policy hierarchy and conflict resolution is part of the post-training
stack used by modern assistant systems. It links the base language model to human
intent, safety policy, and deployment constraints without pretending that a single loss
can capture all values. The goal is not perfect alignment by formula; it is a repeatable
loop where evidence, objectives, and safeguards improve together.

### 1.4 Safety is not only refusal

Safety is not only refusal belongs in the canonical scope of policy and guardrails. The
object is the policy-constrained generation system, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol c(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
a(x,y)=\begin{cases}\mathrm{allow},&c(x,y)<\tau\\ \mathrm{block},&c(x,y)\ge \tau\end{cases}.
$$

For safety is not only refusal, this formula should not be treated as a slogan. It
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
- Treat safety is not only refusal as part of the model contract and store the exact
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

Worked reasoning pattern for safety is not only refusal:
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

AI connection: Safety is not only refusal is part of the post-training stack used by
modern assistant systems. It links the base language model to human intent, safety
policy, and deployment constraints without pretending that a single loss can capture all
values. The goal is not perfect alignment by formula; it is a repeatable loop where
evidence, objectives, and safeguards improve together.

### 1.5 Why system prompts alone are insufficient

Why system prompts alone are insufficient belongs in the canonical scope of policy and
guardrails. The object is the policy-constrained generation system, not merely a prompt
trick or a moderation label. We study how data, losses, policies, review processes, and
safety constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol c(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
a(x,y)=\begin{cases}\mathrm{allow},&c(x,y)<\tau\\ \mathrm{block},&c(x,y)\ge \tau\end{cases}.
$$

For why system prompts alone are insufficient, this formula should not be treated as a
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
- Treat why system prompts alone are insufficient as part of the model contract and
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

Worked reasoning pattern for why system prompts alone are insufficient:
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

AI connection: Why system prompts alone are insufficient is part of the post-training
stack used by modern assistant systems. It links the base language model to human
intent, safety policy, and deployment constraints without pretending that a single loss
can capture all values. The goal is not perfect alignment by formula; it is a repeatable
loop where evidence, objectives, and safeguards improve together.

## 2. Formal Definitions

Formal Definitions develops the part of policy and guardrails that the approved TOC
assigns to Chapter 18. The emphasis is alignment behavior, safety constraints, and
feedback loops, not generic fine-tuning or production monitoring.

### 2.1 Policy rule

Policy rule belongs in the canonical scope of policy and guardrails. The object is the
policy-constrained generation system, not merely a prompt trick or a moderation label.
We study how data, losses, policies, review processes, and safety constraints shape a
model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol c(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
a(x,y)=\begin{cases}\mathrm{allow},&c(x,y)<\tau\\ \mathrm{block},&c(x,y)\ge \tau\end{cases}.
$$

For policy rule, this formula should not be treated as a slogan. It defines which
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
- Treat policy rule as part of the model contract and store the exact data version.
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

Worked reasoning pattern for policy rule:
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

AI connection: Policy rule is part of the post-training stack used by modern assistant
systems. It links the base language model to human intent, safety policy, and deployment
constraints without pretending that a single loss can capture all values. The goal is
not perfect alignment by formula; it is a repeatable loop where evidence, objectives,
and safeguards improve together.

### 2.2 Allowed and disallowed sets

Allowed and disallowed sets belongs in the canonical scope of policy and guardrails. The
object is the policy-constrained generation system, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol c(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
a(x,y)=\begin{cases}\mathrm{allow},&c(x,y)<\tau\\ \mathrm{block},&c(x,y)\ge \tau\end{cases}.
$$

For allowed and disallowed sets, this formula should not be treated as a slogan. It
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
- Treat allowed and disallowed sets as part of the model contract and store the exact
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

Worked reasoning pattern for allowed and disallowed sets:
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

AI connection: Allowed and disallowed sets is part of the post-training stack used by
modern assistant systems. It links the base language model to human intent, safety
policy, and deployment constraints without pretending that a single loss can capture all
values. The goal is not perfect alignment by formula; it is a repeatable loop where
evidence, objectives, and safeguards improve together.

### 2.3 Classifier $c(x,y)$

Classifier $c(x,y)$ belongs in the canonical scope of policy and guardrails. The object
is the policy-constrained generation system, not merely a prompt trick or a moderation
label. We study how data, losses, policies, review processes, and safety constraints
shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol c(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
a(x,y)=\begin{cases}\mathrm{allow},&c(x,y)<\tau\\ \mathrm{block},&c(x,y)\ge \tau\end{cases}.
$$

For classifier $c(x,y)$, this formula should not be treated as a slogan. It defines
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
- Treat classifier $c(x,y)$ as part of the model contract and store the exact data
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

Worked reasoning pattern for classifier $c(x,y)$:
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

AI connection: Classifier $c(x,y)$ is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 2.4 Guardrail action $a$

Guardrail action $a$ belongs in the canonical scope of policy and guardrails. The object
is the policy-constrained generation system, not merely a prompt trick or a moderation
label. We study how data, losses, policies, review processes, and safety constraints
shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol c(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
a(x,y)=\begin{cases}\mathrm{allow},&c(x,y)<\tau\\ \mathrm{block},&c(x,y)\ge \tau\end{cases}.
$$

For guardrail action $a$, this formula should not be treated as a slogan. It defines
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
- Treat guardrail action $a$ as part of the model contract and store the exact data
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

Worked reasoning pattern for guardrail action $a$:
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

AI connection: Guardrail action $a$ is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 2.5 Risk threshold $\tau$

Risk threshold $\tau$ belongs in the canonical scope of policy and guardrails. The
object is the policy-constrained generation system, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol c(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
a(x,y)=\begin{cases}\mathrm{allow},&c(x,y)<\tau\\ \mathrm{block},&c(x,y)\ge \tau\end{cases}.
$$

For risk threshold $\tau$, this formula should not be treated as a slogan. It defines
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
- Treat risk threshold $\tau$ as part of the model contract and store the exact data
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

Worked reasoning pattern for risk threshold $\tau$:
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

AI connection: Risk threshold $\tau$ is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

## 3. Policy Taxonomies

Policy Taxonomies develops the part of policy and guardrails that the approved TOC
assigns to Chapter 18. The emphasis is alignment behavior, safety constraints, and
feedback loops, not generic fine-tuning or production monitoring.

### 3.1 Harm categories

Harm categories belongs in the canonical scope of policy and guardrails. The object is
the policy-constrained generation system, not merely a prompt trick or a moderation
label. We study how data, losses, policies, review processes, and safety constraints
shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol c(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
a(x,y)=\begin{cases}\mathrm{allow},&c(x,y)<\tau\\ \mathrm{block},&c(x,y)\ge \tau\end{cases}.
$$

For harm categories, this formula should not be treated as a slogan. It defines which
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
- Treat harm categories as part of the model contract and store the exact data version.
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

Worked reasoning pattern for harm categories:
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

AI connection: Harm categories is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 3.2 Capability boundaries

Capability boundaries belongs in the canonical scope of policy and guardrails. The
object is the policy-constrained generation system, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol c(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
a(x,y)=\begin{cases}\mathrm{allow},&c(x,y)<\tau\\ \mathrm{block},&c(x,y)\ge \tau\end{cases}.
$$

For capability boundaries, this formula should not be treated as a slogan. It defines
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
- Treat capability boundaries as part of the model contract and store the exact data
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

Worked reasoning pattern for capability boundaries:
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

AI connection: Capability boundaries is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 3.3 User intent

User intent belongs in the canonical scope of policy and guardrails. The object is the
policy-constrained generation system, not merely a prompt trick or a moderation label.
We study how data, losses, policies, review processes, and safety constraints shape a
model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol c(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
a(x,y)=\begin{cases}\mathrm{allow},&c(x,y)<\tau\\ \mathrm{block},&c(x,y)\ge \tau\end{cases}.
$$

For user intent, this formula should not be treated as a slogan. It defines which
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
- Treat user intent as part of the model contract and store the exact data version.
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

Worked reasoning pattern for user intent:
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

AI connection: User intent is part of the post-training stack used by modern assistant
systems. It links the base language model to human intent, safety policy, and deployment
constraints without pretending that a single loss can capture all values. The goal is
not perfect alignment by formula; it is a repeatable loop where evidence, objectives,
and safeguards improve together.

### 3.4 Context sensitivity

Context sensitivity belongs in the canonical scope of policy and guardrails. The object
is the policy-constrained generation system, not merely a prompt trick or a moderation
label. We study how data, losses, policies, review processes, and safety constraints
shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol c(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
a(x,y)=\begin{cases}\mathrm{allow},&c(x,y)<\tau\\ \mathrm{block},&c(x,y)\ge \tau\end{cases}.
$$

For context sensitivity, this formula should not be treated as a slogan. It defines
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
- Treat context sensitivity as part of the model contract and store the exact data
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

Worked reasoning pattern for context sensitivity:
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

AI connection: Context sensitivity is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 3.5 Jurisdiction notes

Jurisdiction notes belongs in the canonical scope of policy and guardrails. The object
is the policy-constrained generation system, not merely a prompt trick or a moderation
label. We study how data, losses, policies, review processes, and safety constraints
shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol c(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
a(x,y)=\begin{cases}\mathrm{allow},&c(x,y)<\tau\\ \mathrm{block},&c(x,y)\ge \tau\end{cases}.
$$

For jurisdiction notes, this formula should not be treated as a slogan. It defines which
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
- Treat jurisdiction notes as part of the model contract and store the exact data
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

Worked reasoning pattern for jurisdiction notes:
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

AI connection: Jurisdiction notes is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

## 4. Constitutional AI and Rule-Based Training

Constitutional AI and Rule-Based Training develops the part of policy and guardrails
that the approved TOC assigns to Chapter 18. The emphasis is alignment behavior, safety
constraints, and feedback loops, not generic fine-tuning or production monitoring.

### 4.1 Critique-revise loops

Critique-revise loops belongs in the canonical scope of policy and guardrails. The
object is the policy-constrained generation system, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol c(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
a(x,y)=\begin{cases}\mathrm{allow},&c(x,y)<\tau\\ \mathrm{block},&c(x,y)\ge \tau\end{cases}.
$$

For critique-revise loops, this formula should not be treated as a slogan. It defines
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
- Treat critique-revise loops as part of the model contract and store the exact data
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

Worked reasoning pattern for critique-revise loops:
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

AI connection: Critique-revise loops is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 4.2 AI feedback

AI feedback belongs in the canonical scope of policy and guardrails. The object is the
policy-constrained generation system, not merely a prompt trick or a moderation label.
We study how data, losses, policies, review processes, and safety constraints shape a
model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol c(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
a(x,y)=\begin{cases}\mathrm{allow},&c(x,y)<\tau\\ \mathrm{block},&c(x,y)\ge \tau\end{cases}.
$$

For ai feedback, this formula should not be treated as a slogan. It defines which
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
- Treat ai feedback as part of the model contract and store the exact data version.
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

Worked reasoning pattern for ai feedback:
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

AI connection: AI feedback is part of the post-training stack used by modern assistant
systems. It links the base language model to human intent, safety policy, and deployment
constraints without pretending that a single loss can capture all values. The goal is
not perfect alignment by formula; it is a repeatable loop where evidence, objectives,
and safeguards improve together.

### 4.3 Rule hierarchy

Rule hierarchy belongs in the canonical scope of policy and guardrails. The object is
the policy-constrained generation system, not merely a prompt trick or a moderation
label. We study how data, losses, policies, review processes, and safety constraints
shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol c(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
a(x,y)=\begin{cases}\mathrm{allow},&c(x,y)<\tau\\ \mathrm{block},&c(x,y)\ge \tau\end{cases}.
$$

For rule hierarchy, this formula should not be treated as a slogan. It defines which
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
- Treat rule hierarchy as part of the model contract and store the exact data version.
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

Worked reasoning pattern for rule hierarchy:
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

AI connection: Rule hierarchy is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 4.4 Policy distillation

Policy distillation belongs in the canonical scope of policy and guardrails. The object
is the policy-constrained generation system, not merely a prompt trick or a moderation
label. We study how data, losses, policies, review processes, and safety constraints
shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol c(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
a(x,y)=\begin{cases}\mathrm{allow},&c(x,y)<\tau\\ \mathrm{block},&c(x,y)\ge \tau\end{cases}.
$$

For policy distillation, this formula should not be treated as a slogan. It defines
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
- Treat policy distillation as part of the model contract and store the exact data
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

Worked reasoning pattern for policy distillation:
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

AI connection: Policy distillation is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 4.5 Constitutional evaluation

Constitutional evaluation belongs in the canonical scope of policy and guardrails. The
object is the policy-constrained generation system, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol c(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
a(x,y)=\begin{cases}\mathrm{allow},&c(x,y)<\tau\\ \mathrm{block},&c(x,y)\ge \tau\end{cases}.
$$

For constitutional evaluation, this formula should not be treated as a slogan. It
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
- Treat constitutional evaluation as part of the model contract and store the exact data
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

Worked reasoning pattern for constitutional evaluation:
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

AI connection: Constitutional evaluation is part of the post-training stack used by
modern assistant systems. It links the base language model to human intent, safety
policy, and deployment constraints without pretending that a single loss can capture all
values. The goal is not perfect alignment by formula; it is a repeatable loop where
evidence, objectives, and safeguards improve together.

## 5. Guardrail Architectures

Guardrail Architectures develops the part of policy and guardrails that the approved TOC
assigns to Chapter 18. The emphasis is alignment behavior, safety constraints, and
feedback loops, not generic fine-tuning or production monitoring.

### 5.1 Input filters

Input filters belongs in the canonical scope of policy and guardrails. The object is the
policy-constrained generation system, not merely a prompt trick or a moderation label.
We study how data, losses, policies, review processes, and safety constraints shape a
model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol c(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
a(x,y)=\begin{cases}\mathrm{allow},&c(x,y)<\tau\\ \mathrm{block},&c(x,y)\ge \tau\end{cases}.
$$

For input filters, this formula should not be treated as a slogan. It defines which
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
- Treat input filters as part of the model contract and store the exact data version.
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

Worked reasoning pattern for input filters:
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

AI connection: Input filters is part of the post-training stack used by modern assistant
systems. It links the base language model to human intent, safety policy, and deployment
constraints without pretending that a single loss can capture all values. The goal is
not perfect alignment by formula; it is a repeatable loop where evidence, objectives,
and safeguards improve together.

### 5.2 Output filters

Output filters belongs in the canonical scope of policy and guardrails. The object is
the policy-constrained generation system, not merely a prompt trick or a moderation
label. We study how data, losses, policies, review processes, and safety constraints
shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol c(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
a(x,y)=\begin{cases}\mathrm{allow},&c(x,y)<\tau\\ \mathrm{block},&c(x,y)\ge \tau\end{cases}.
$$

For output filters, this formula should not be treated as a slogan. It defines which
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
- Treat output filters as part of the model contract and store the exact data version.
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

Worked reasoning pattern for output filters:
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

AI connection: Output filters is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 5.3 Tool gates

Tool gates belongs in the canonical scope of policy and guardrails. The object is the
policy-constrained generation system, not merely a prompt trick or a moderation label.
We study how data, losses, policies, review processes, and safety constraints shape a
model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol c(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
a(x,y)=\begin{cases}\mathrm{allow},&c(x,y)<\tau\\ \mathrm{block},&c(x,y)\ge \tau\end{cases}.
$$

For tool gates, this formula should not be treated as a slogan. It defines which tokens,
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
- Treat tool gates as part of the model contract and store the exact data version.
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

Worked reasoning pattern for tool gates:
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

AI connection: Tool gates is part of the post-training stack used by modern assistant
systems. It links the base language model to human intent, safety policy, and deployment
constraints without pretending that a single loss can capture all values. The goal is
not perfect alignment by formula; it is a repeatable loop where evidence, objectives,
and safeguards improve together.

### 5.4 Retrieval gates

Retrieval gates belongs in the canonical scope of policy and guardrails. The object is
the policy-constrained generation system, not merely a prompt trick or a moderation
label. We study how data, losses, policies, review processes, and safety constraints
shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol c(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
a(x,y)=\begin{cases}\mathrm{allow},&c(x,y)<\tau\\ \mathrm{block},&c(x,y)\ge \tau\end{cases}.
$$

For retrieval gates, this formula should not be treated as a slogan. It defines which
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
- Treat retrieval gates as part of the model contract and store the exact data version.
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

Worked reasoning pattern for retrieval gates:
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

AI connection: Retrieval gates is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 5.5 Structured validators

Structured validators belongs in the canonical scope of policy and guardrails. The
object is the policy-constrained generation system, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol c(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
a(x,y)=\begin{cases}\mathrm{allow},&c(x,y)<\tau\\ \mathrm{block},&c(x,y)\ge \tau\end{cases}.
$$

For structured validators, this formula should not be treated as a slogan. It defines
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
- Treat structured validators as part of the model contract and store the exact data
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

Worked reasoning pattern for structured validators:
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

AI connection: Structured validators is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

## 6. Moderation Models

Moderation Models develops the part of policy and guardrails that the approved TOC
assigns to Chapter 18. The emphasis is alignment behavior, safety constraints, and
feedback loops, not generic fine-tuning or production monitoring.

### 6.1 Llama Guard

Llama Guard belongs in the canonical scope of policy and guardrails. The object is the
policy-constrained generation system, not merely a prompt trick or a moderation label.
We study how data, losses, policies, review processes, and safety constraints shape a
model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol c(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
a(x,y)=\begin{cases}\mathrm{allow},&c(x,y)<\tau\\ \mathrm{block},&c(x,y)\ge \tau\end{cases}.
$$

For llama guard, this formula should not be treated as a slogan. It defines which
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
- Treat llama guard as part of the model contract and store the exact data version.
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

Worked reasoning pattern for llama guard:
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

AI connection: Llama Guard is part of the post-training stack used by modern assistant
systems. It links the base language model to human intent, safety policy, and deployment
constraints without pretending that a single loss can capture all values. The goal is
not perfect alignment by formula; it is a repeatable loop where evidence, objectives,
and safeguards improve together.

### 6.2 ShieldGemma

ShieldGemma belongs in the canonical scope of policy and guardrails. The object is the
policy-constrained generation system, not merely a prompt trick or a moderation label.
We study how data, losses, policies, review processes, and safety constraints shape a
model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol c(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
a(x,y)=\begin{cases}\mathrm{allow},&c(x,y)<\tau\\ \mathrm{block},&c(x,y)\ge \tau\end{cases}.
$$

For shieldgemma, this formula should not be treated as a slogan. It defines which
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
- Treat shieldgemma as part of the model contract and store the exact data version.
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

Worked reasoning pattern for shieldgemma:
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

AI connection: ShieldGemma is part of the post-training stack used by modern assistant
systems. It links the base language model to human intent, safety policy, and deployment
constraints without pretending that a single loss can capture all values. The goal is
not perfect alignment by formula; it is a repeatable loop where evidence, objectives,
and safeguards improve together.

### 6.3 AEGIS-style taxonomies

AEGIS-style taxonomies belongs in the canonical scope of policy and guardrails. The
object is the policy-constrained generation system, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol c(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
a(x,y)=\begin{cases}\mathrm{allow},&c(x,y)<\tau\\ \mathrm{block},&c(x,y)\ge \tau\end{cases}.
$$

For aegis-style taxonomies, this formula should not be treated as a slogan. It defines
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
- Treat aegis-style taxonomies as part of the model contract and store the exact data
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

Worked reasoning pattern for aegis-style taxonomies:
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

AI connection: AEGIS-style taxonomies is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 6.4 Classifier calibration

Classifier calibration belongs in the canonical scope of policy and guardrails. The
object is the policy-constrained generation system, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol c(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
a(x,y)=\begin{cases}\mathrm{allow},&c(x,y)<\tau\\ \mathrm{block},&c(x,y)\ge \tau\end{cases}.
$$

For classifier calibration, this formula should not be treated as a slogan. It defines
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
- Treat classifier calibration as part of the model contract and store the exact data
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

Worked reasoning pattern for classifier calibration:
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

AI connection: Classifier calibration is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 6.5 Multilingual moderation

Multilingual moderation belongs in the canonical scope of policy and guardrails. The
object is the policy-constrained generation system, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol c(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
a(x,y)=\begin{cases}\mathrm{allow},&c(x,y)<\tau\\ \mathrm{block},&c(x,y)\ge \tau\end{cases}.
$$

For multilingual moderation, this formula should not be treated as a slogan. It defines
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
- Treat multilingual moderation as part of the model contract and store the exact data
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

Worked reasoning pattern for multilingual moderation:
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

AI connection: Multilingual moderation is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

## 7. Tradeoffs

Tradeoffs develops the part of policy and guardrails that the approved TOC assigns to
Chapter 18. The emphasis is alignment behavior, safety constraints, and feedback loops,
not generic fine-tuning or production monitoring.

### 7.1 Safety versus helpfulness

Safety versus helpfulness belongs in the canonical scope of policy and guardrails. The
object is the policy-constrained generation system, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol c(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
a(x,y)=\begin{cases}\mathrm{allow},&c(x,y)<\tau\\ \mathrm{block},&c(x,y)\ge \tau\end{cases}.
$$

For safety versus helpfulness, this formula should not be treated as a slogan. It
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
- Treat safety versus helpfulness as part of the model contract and store the exact data
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

Worked reasoning pattern for safety versus helpfulness:
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

AI connection: Safety versus helpfulness is part of the post-training stack used by
modern assistant systems. It links the base language model to human intent, safety
policy, and deployment constraints without pretending that a single loss can capture all
values. The goal is not perfect alignment by formula; it is a repeatable loop where
evidence, objectives, and safeguards improve together.

### 7.2 Refusal versus compliance

Refusal versus compliance belongs in the canonical scope of policy and guardrails. The
object is the policy-constrained generation system, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol c(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
a(x,y)=\begin{cases}\mathrm{allow},&c(x,y)<\tau\\ \mathrm{block},&c(x,y)\ge \tau\end{cases}.
$$

For refusal versus compliance, this formula should not be treated as a slogan. It
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
- Treat refusal versus compliance as part of the model contract and store the exact data
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

Worked reasoning pattern for refusal versus compliance:
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

AI connection: Refusal versus compliance is part of the post-training stack used by
modern assistant systems. It links the base language model to human intent, safety
policy, and deployment constraints without pretending that a single loss can capture all
values. The goal is not perfect alignment by formula; it is a repeatable loop where
evidence, objectives, and safeguards improve together.

### 7.3 Latency

Latency belongs in the canonical scope of policy and guardrails. The object is the
policy-constrained generation system, not merely a prompt trick or a moderation label.
We study how data, losses, policies, review processes, and safety constraints shape a
model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol c(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
a(x,y)=\begin{cases}\mathrm{allow},&c(x,y)<\tau\\ \mathrm{block},&c(x,y)\ge \tau\end{cases}.
$$

For latency, this formula should not be treated as a slogan. It defines which tokens,
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
- Treat latency as part of the model contract and store the exact data version.
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

Worked reasoning pattern for latency:
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

AI connection: Latency is part of the post-training stack used by modern assistant
systems. It links the base language model to human intent, safety policy, and deployment
constraints without pretending that a single loss can capture all values. The goal is
not perfect alignment by formula; it is a repeatable loop where evidence, objectives,
and safeguards improve together.

### 7.4 Adversarial bypass

Adversarial bypass belongs in the canonical scope of policy and guardrails. The object
is the policy-constrained generation system, not merely a prompt trick or a moderation
label. We study how data, losses, policies, review processes, and safety constraints
shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol c(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
a(x,y)=\begin{cases}\mathrm{allow},&c(x,y)<\tau\\ \mathrm{block},&c(x,y)\ge \tau\end{cases}.
$$

For adversarial bypass, this formula should not be treated as a slogan. It defines which
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
- Treat adversarial bypass as part of the model contract and store the exact data
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

Worked reasoning pattern for adversarial bypass:
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

AI connection: Adversarial bypass is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 7.5 Audit logs

Audit logs belongs in the canonical scope of policy and guardrails. The object is the
policy-constrained generation system, not merely a prompt trick or a moderation label.
We study how data, losses, policies, review processes, and safety constraints shape a
model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol c(x,y). It marks the
alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
a(x,y)=\begin{cases}\mathrm{allow},&c(x,y)<\tau\\ \mathrm{block},&c(x,y)\ge \tau\end{cases}.
$$

For audit logs, this formula should not be treated as a slogan. It defines which tokens,
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
- Treat audit logs as part of the model contract and store the exact data version.
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

Worked reasoning pattern for audit logs:
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

AI connection: Audit logs is part of the post-training stack used by modern assistant
systems. It links the base language model to human intent, safety policy, and deployment
constraints without pretending that a single loss can capture all values. The goal is
not perfect alignment by formula; it is a repeatable loop where evidence, objectives,
and safeguards improve together.

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

1. (*) Alignment needs explicit policy.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

2. (*) Guardrails as runtime control.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

3. (*) Policy hierarchy and conflict resolution.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

4. (**) Safety is not only refusal.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

5. (**) Why system prompts alone are insufficient.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

6. (**) Policy rule.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

7. (***) Allowed and disallowed sets.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

8. (***) Classifier $c(x,y)$.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

9. (***) Guardrail action $a$.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

10. (***) Risk threshold $\tau$.
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

- [Constitutional AI](https://arxiv.org/abs/2212.08073)
- [Llama Guard](https://arxiv.org/abs/2312.06674)
- [ShieldGemma](https://arxiv.org/abs/2407.21772)
- [AEGIS](https://arxiv.org/abs/2404.05993)
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
- [OpenAI Model Spec](https://cdn.openai.com/spec/model-spec-2024-05-08.html)
- [KL Divergence](../../09-Information-Theory/02-KL-Divergence/notes.md)
- [Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md)
- [Fine-Tuning Math](../../15-Math-for-LLMs/07-Fine-Tuning-Math/notes.md)
- [Evaluation and Reliability](../../17-Evaluation-and-Reliability/01-Capability-Benchmarks/notes.md)

