[Back to Curriculum](../../README.md) | [Previous: Online Experimentation and AB Testing](../../17-Evaluation-and-Reliability/05-Online-Experimentation-and-AB-Testing/notes.md) | [Next: Preference Optimization RLHF and DPO](../02-Preference-Optimization-RLHF-and-DPO/notes.md)

---

# Instruction Tuning and SFT

> _"Instruction tuning turns a language model into a cooperative interface."_

## Overview

Supervised fine-tuning aligns a pretrained next-token model with demonstrated
instruction-following behavior by optimizing response tokens under a curated chat
protocol.

Alignment and safety sit between model training and model deployment. The chapter
studies how desired behavior is specified, learned, stress-tested, constrained, and
improved through feedback.

This section is written in LaTeX Markdown. Inline mathematics uses `$...$`, and display
equations use `$$...$$`. The notes use math to expose the operational contract: which
behavior is optimized, which behavior is forbidden, and which evidence proves the system
changed.

## Prerequisites

- [Fine-Tuning Math](../../15-Math-for-LLMs/07-Fine-Tuning-Math/notes.md)
- [Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md)
- [Data Format Standards](../../16-LLM-Training-Data-Pipeline/01-Data-Format-Standards/notes.md)
- [Capability Benchmarks](../../17-Evaluation-and-Reliability/01-Capability-Benchmarks/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Executable demonstrations for instruction tuning and sft |
| [exercises.ipynb](exercises.ipynb) | Graded practice for instruction tuning and sft |

## Learning Objectives

After completing this section, you will be able to:

- Define the core mathematical objects in instruction tuning and sft
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
  - [1.1 Why pretrained next-token models need instruction-following alignment](#11-why-pretrained-next-token-models-need-instruction-following-alignment)
  - [1.2 Instruction following as behavior conditioning](#12-instruction-following-as-behavior-conditioning)
  - [1.3 Demonstrations versus preferences](#13-demonstrations-versus-preferences)
  - [1.4 Chat templates as part of the objective](#14-chat-templates-as-part-of-the-objective)
  - [1.5 Historical arc from FLAN to InstructGPT](#15-historical-arc-from-flan-to-instructgpt)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Prompt $x$ and response $y$](#21-prompt-x-and-response-y)
  - [2.2 Demonstration dataset $\mathcal{D}_{\mathrm{SFT}}$](#22-demonstration-dataset-mathcaldmathrmsft)
  - [2.3 Policy $\pi_\theta(y \mid x)$](#23-policy-pithetay-mid-x)
  - [2.4 Response-token mask](#24-response-token-mask)
  - [2.5 Instruction distribution and validation split](#25-instruction-distribution-and-validation-split)
- [3. Instruction Data Design](#3-instruction-data-design)
  - [3.1 Task diversity](#31-task-diversity)
  - [3.2 Chat templates](#32-chat-templates)
  - [3.3 Role tokens](#33-role-tokens)
  - [3.4 Refusal examples](#34-refusal-examples)
  - [3.5 Multi-turn demonstrations](#35-multi-turn-demonstrations)
- [4. SFT Objective](#4-sft-objective)
  - [4.1 Response-only cross-entropy](#41-response-only-cross-entropy)
  - [4.2 Packed examples](#42-packed-examples)
  - [4.3 Loss masks](#43-loss-masks)
  - [4.4 Class imbalance](#44-class-imbalance)
  - [4.5 Validation curves](#45-validation-curves)
- [5. Alignment Behavior](#5-alignment-behavior)
  - [5.1 Helpfulness](#51-helpfulness)
  - [5.2 Honesty](#52-honesty)
  - [5.3 Harmlessness](#53-harmlessness)
  - [5.4 Style control](#54-style-control)
  - [5.5 Sycophancy risk](#55-sycophancy-risk)
- [6. SFT Limits](#6-sft-limits)
  - [6.1 Imitation ceiling](#61-imitation-ceiling)
  - [6.2 Hallucination](#62-hallucination)
  - [6.3 Over-refusal](#63-over-refusal)
  - [6.4 Catastrophic forgetting](#64-catastrophic-forgetting)
  - [6.5 Distribution mismatch](#65-distribution-mismatch)
- [7. Synthetic and Self-Generated Instructions](#7-synthetic-and-self-generated-instructions)
  - [7.1 FLAN-style multitask instruction tuning](#71-flan-style-multitask-instruction-tuning)
  - [7.2 Self-Instruct bootstrapping](#72-self-instruct-bootstrapping)
  - [7.3 Data filtering](#73-data-filtering)
  - [7.4 Quality scoring](#74-quality-scoring)
  - [7.5 Bootstrapping loops](#75-bootstrapping-loops)
- [8. Common Mistakes](#8-common-mistakes)
- [9. Exercises](#9-exercises)
- [10. Why This Matters for AI](#10-why-this-matters-for-ai)
- [11. Conceptual Bridge](#11-conceptual-bridge)
- [References](#references)

---

## 1. Intuition

Intuition develops the part of instruction tuning and sft that the approved TOC assigns
to Chapter 18. The emphasis is alignment behavior, safety constraints, and feedback
loops, not generic fine-tuning or production monitoring.

### 1.1 Why pretrained next-token models need instruction-following alignment

Why pretrained next-token models need instruction-following alignment belongs in the
canonical scope of instruction tuning and sft. The object is the instruction-following
policy, not merely a prompt trick or a moderation label. We study how data, losses,
policies, review processes, and safety constraints shape a model's conditional
distribution over responses.

A compact way to read this subsection is through the local symbol \pi_\theta(y \mid x).
It marks the alignment object being transformed: an instruction policy, a preference
pair, a violation classifier, a guardrail action, or a feedback event. The details
differ, but the discipline is the same: state the object, state the loss or decision
rule, then audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{SFT}}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t \in R_i}\log \pi_\theta(y_{i,t} \mid x_i,y_{i,<t}).
$$

For why pretrained next-token models need instruction-following alignment, this formula
should not be treated as a slogan. It defines which tokens, responses, comparisons, or
decisions receive gradient or operational weight. A change in masking, sampling, rubric
wording, or thresholding changes the effective objective even if the model architecture
is unchanged.

| Alignment object | Mathematical question | Engineering question |
| --- | --- | --- |
| Data | Which examples define the target behavior? | Who wrote, filtered, and approved them? |
| Objective | Which terms receive weight? | Are masks, margins, and thresholds logged? |
| Policy | Which actions are allowed or disallowed? | Can reviewers reproduce the decision? |
| Evaluation | Which metric detects regression? | Is the test private, stable, and sliced? |
| Feedback | Which new evidence changes training? | How does it enter the next dataset version? |

Examples:
- Treat why pretrained next-token models need instruction-following alignment as part of
the model contract and store the exact data version.
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

Worked reasoning pattern for why pretrained next-token models need instruction-following
alignment:
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

AI connection: Why pretrained next-token models need instruction-following alignment is
part of the post-training stack used by modern assistant systems. It links the base
language model to human intent, safety policy, and deployment constraints without
pretending that a single loss can capture all values. The goal is not perfect alignment
by formula; it is a repeatable loop where evidence, objectives, and safeguards improve
together.

### 1.2 Instruction following as behavior conditioning

Instruction following as behavior conditioning belongs in the canonical scope of
instruction tuning and sft. The object is the instruction-following policy, not merely a
prompt trick or a moderation label. We study how data, losses, policies, review
processes, and safety constraints shape a model's conditional distribution over
responses.

A compact way to read this subsection is through the local symbol \pi_\theta(y \mid x).
It marks the alignment object being transformed: an instruction policy, a preference
pair, a violation classifier, a guardrail action, or a feedback event. The details
differ, but the discipline is the same: state the object, state the loss or decision
rule, then audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{SFT}}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t \in R_i}\log \pi_\theta(y_{i,t} \mid x_i,y_{i,<t}).
$$

For instruction following as behavior conditioning, this formula should not be treated
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
- Treat instruction following as behavior conditioning as part of the model contract and
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

Worked reasoning pattern for instruction following as behavior conditioning:
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

AI connection: Instruction following as behavior conditioning is part of the post-
training stack used by modern assistant systems. It links the base language model to
human intent, safety policy, and deployment constraints without pretending that a single
loss can capture all values. The goal is not perfect alignment by formula; it is a
repeatable loop where evidence, objectives, and safeguards improve together.

### 1.3 Demonstrations versus preferences

Demonstrations versus preferences belongs in the canonical scope of instruction tuning
and sft. The object is the instruction-following policy, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol \pi_\theta(y \mid x).
It marks the alignment object being transformed: an instruction policy, a preference
pair, a violation classifier, a guardrail action, or a feedback event. The details
differ, but the discipline is the same: state the object, state the loss or decision
rule, then audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{SFT}}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t \in R_i}\log \pi_\theta(y_{i,t} \mid x_i,y_{i,<t}).
$$

For demonstrations versus preferences, this formula should not be treated as a slogan.
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
- Treat demonstrations versus preferences as part of the model contract and store the
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

Worked reasoning pattern for demonstrations versus preferences:
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

AI connection: Demonstrations versus preferences is part of the post-training stack used
by modern assistant systems. It links the base language model to human intent, safety
policy, and deployment constraints without pretending that a single loss can capture all
values. The goal is not perfect alignment by formula; it is a repeatable loop where
evidence, objectives, and safeguards improve together.

### 1.4 Chat templates as part of the objective

Chat templates as part of the objective belongs in the canonical scope of instruction
tuning and sft. The object is the instruction-following policy, not merely a prompt
trick or a moderation label. We study how data, losses, policies, review processes, and
safety constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol \pi_\theta(y \mid x).
It marks the alignment object being transformed: an instruction policy, a preference
pair, a violation classifier, a guardrail action, or a feedback event. The details
differ, but the discipline is the same: state the object, state the loss or decision
rule, then audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{SFT}}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t \in R_i}\log \pi_\theta(y_{i,t} \mid x_i,y_{i,<t}).
$$

For chat templates as part of the objective, this formula should not be treated as a
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
- Treat chat templates as part of the objective as part of the model contract and store
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

Worked reasoning pattern for chat templates as part of the objective:
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

AI connection: Chat templates as part of the objective is part of the post-training
stack used by modern assistant systems. It links the base language model to human
intent, safety policy, and deployment constraints without pretending that a single loss
can capture all values. The goal is not perfect alignment by formula; it is a repeatable
loop where evidence, objectives, and safeguards improve together.

### 1.5 Historical arc from FLAN to InstructGPT

Historical arc from FLAN to InstructGPT belongs in the canonical scope of instruction
tuning and sft. The object is the instruction-following policy, not merely a prompt
trick or a moderation label. We study how data, losses, policies, review processes, and
safety constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol \pi_\theta(y \mid x).
It marks the alignment object being transformed: an instruction policy, a preference
pair, a violation classifier, a guardrail action, or a feedback event. The details
differ, but the discipline is the same: state the object, state the loss or decision
rule, then audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{SFT}}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t \in R_i}\log \pi_\theta(y_{i,t} \mid x_i,y_{i,<t}).
$$

For historical arc from flan to instructgpt, this formula should not be treated as a
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
- Treat historical arc from flan to instructgpt as part of the model contract and store
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

Worked reasoning pattern for historical arc from flan to instructgpt:
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

AI connection: Historical arc from FLAN to InstructGPT is part of the post-training
stack used by modern assistant systems. It links the base language model to human
intent, safety policy, and deployment constraints without pretending that a single loss
can capture all values. The goal is not perfect alignment by formula; it is a repeatable
loop where evidence, objectives, and safeguards improve together.

## 2. Formal Definitions

Formal Definitions develops the part of instruction tuning and sft that the approved TOC
assigns to Chapter 18. The emphasis is alignment behavior, safety constraints, and
feedback loops, not generic fine-tuning or production monitoring.

### 2.1 Prompt $x$ and response $y$

Prompt $x$ and response $y$ belongs in the canonical scope of instruction tuning and
sft. The object is the instruction-following policy, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol \pi_\theta(y \mid x).
It marks the alignment object being transformed: an instruction policy, a preference
pair, a violation classifier, a guardrail action, or a feedback event. The details
differ, but the discipline is the same: state the object, state the loss or decision
rule, then audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{SFT}}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t \in R_i}\log \pi_\theta(y_{i,t} \mid x_i,y_{i,<t}).
$$

For prompt $x$ and response $y$, this formula should not be treated as a slogan. It
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
- Treat prompt $x$ and response $y$ as part of the model contract and store the exact
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

Worked reasoning pattern for prompt $x$ and response $y$:
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

AI connection: Prompt $x$ and response $y$ is part of the post-training stack used by
modern assistant systems. It links the base language model to human intent, safety
policy, and deployment constraints without pretending that a single loss can capture all
values. The goal is not perfect alignment by formula; it is a repeatable loop where
evidence, objectives, and safeguards improve together.

### 2.2 Demonstration dataset $\mathcal{D}_{\mathrm{SFT}}$

Demonstration dataset $\mathcal{D}_{\mathrm{SFT}}$ belongs in the canonical scope of
instruction tuning and sft. The object is the instruction-following policy, not merely a
prompt trick or a moderation label. We study how data, losses, policies, review
processes, and safety constraints shape a model's conditional distribution over
responses.

A compact way to read this subsection is through the local symbol \pi_\theta(y \mid x).
It marks the alignment object being transformed: an instruction policy, a preference
pair, a violation classifier, a guardrail action, or a feedback event. The details
differ, but the discipline is the same: state the object, state the loss or decision
rule, then audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{SFT}}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t \in R_i}\log \pi_\theta(y_{i,t} \mid x_i,y_{i,<t}).
$$

For demonstration dataset $\mathcal{d}_{\mathrm{sft}}$, this formula should not be
treated as a slogan. It defines which tokens, responses, comparisons, or decisions
receive gradient or operational weight. A change in masking, sampling, rubric wording,
or thresholding changes the effective objective even if the model architecture is
unchanged.

| Alignment object | Mathematical question | Engineering question |
| --- | --- | --- |
| Data | Which examples define the target behavior? | Who wrote, filtered, and approved them? |
| Objective | Which terms receive weight? | Are masks, margins, and thresholds logged? |
| Policy | Which actions are allowed or disallowed? | Can reviewers reproduce the decision? |
| Evaluation | Which metric detects regression? | Is the test private, stable, and sliced? |
| Feedback | Which new evidence changes training? | How does it enter the next dataset version? |

Examples:
- Treat demonstration dataset $\mathcal{d}_{\mathrm{sft}}$ as part of the model contract
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

Worked reasoning pattern for demonstration dataset $\mathcal{d}_{\mathrm{sft}}$:
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

AI connection: Demonstration dataset $\mathcal{D}_{\mathrm{SFT}}$ is part of the post-
training stack used by modern assistant systems. It links the base language model to
human intent, safety policy, and deployment constraints without pretending that a single
loss can capture all values. The goal is not perfect alignment by formula; it is a
repeatable loop where evidence, objectives, and safeguards improve together.

### 2.3 Policy $\pi_\theta(y \mid x)$

Policy $\pi_\theta(y \mid x)$ belongs in the canonical scope of instruction tuning and
sft. The object is the instruction-following policy, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol \pi_\theta(y \mid x).
It marks the alignment object being transformed: an instruction policy, a preference
pair, a violation classifier, a guardrail action, or a feedback event. The details
differ, but the discipline is the same: state the object, state the loss or decision
rule, then audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{SFT}}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t \in R_i}\log \pi_\theta(y_{i,t} \mid x_i,y_{i,<t}).
$$

For policy $\pi_\theta(y \mid x)$, this formula should not be treated as a slogan. It
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
- Treat policy $\pi_\theta(y \mid x)$ as part of the model contract and store the exact
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

Worked reasoning pattern for policy $\pi_\theta(y \mid x)$:
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

AI connection: Policy $\pi_\theta(y \mid x)$ is part of the post-training stack used by
modern assistant systems. It links the base language model to human intent, safety
policy, and deployment constraints without pretending that a single loss can capture all
values. The goal is not perfect alignment by formula; it is a repeatable loop where
evidence, objectives, and safeguards improve together.

### 2.4 Response-token mask

Response-token mask belongs in the canonical scope of instruction tuning and sft. The
object is the instruction-following policy, not merely a prompt trick or a moderation
label. We study how data, losses, policies, review processes, and safety constraints
shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol \pi_\theta(y \mid x).
It marks the alignment object being transformed: an instruction policy, a preference
pair, a violation classifier, a guardrail action, or a feedback event. The details
differ, but the discipline is the same: state the object, state the loss or decision
rule, then audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{SFT}}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t \in R_i}\log \pi_\theta(y_{i,t} \mid x_i,y_{i,<t}).
$$

For response-token mask, this formula should not be treated as a slogan. It defines
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
- Treat response-token mask as part of the model contract and store the exact data
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

Worked reasoning pattern for response-token mask:
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

AI connection: Response-token mask is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 2.5 Instruction distribution and validation split

Instruction distribution and validation split belongs in the canonical scope of
instruction tuning and sft. The object is the instruction-following policy, not merely a
prompt trick or a moderation label. We study how data, losses, policies, review
processes, and safety constraints shape a model's conditional distribution over
responses.

A compact way to read this subsection is through the local symbol \pi_\theta(y \mid x).
It marks the alignment object being transformed: an instruction policy, a preference
pair, a violation classifier, a guardrail action, or a feedback event. The details
differ, but the discipline is the same: state the object, state the loss or decision
rule, then audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{SFT}}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t \in R_i}\log \pi_\theta(y_{i,t} \mid x_i,y_{i,<t}).
$$

For instruction distribution and validation split, this formula should not be treated as
a slogan. It defines which tokens, responses, comparisons, or decisions receive gradient
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
- Treat instruction distribution and validation split as part of the model contract and
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

Worked reasoning pattern for instruction distribution and validation split:
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

AI connection: Instruction distribution and validation split is part of the post-
training stack used by modern assistant systems. It links the base language model to
human intent, safety policy, and deployment constraints without pretending that a single
loss can capture all values. The goal is not perfect alignment by formula; it is a
repeatable loop where evidence, objectives, and safeguards improve together.

## 3. Instruction Data Design

Instruction Data Design develops the part of instruction tuning and sft that the
approved TOC assigns to Chapter 18. The emphasis is alignment behavior, safety
constraints, and feedback loops, not generic fine-tuning or production monitoring.

### 3.1 Task diversity

Task diversity belongs in the canonical scope of instruction tuning and sft. The object
is the instruction-following policy, not merely a prompt trick or a moderation label. We
study how data, losses, policies, review processes, and safety constraints shape a
model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol \pi_\theta(y \mid x).
It marks the alignment object being transformed: an instruction policy, a preference
pair, a violation classifier, a guardrail action, or a feedback event. The details
differ, but the discipline is the same: state the object, state the loss or decision
rule, then audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{SFT}}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t \in R_i}\log \pi_\theta(y_{i,t} \mid x_i,y_{i,<t}).
$$

For task diversity, this formula should not be treated as a slogan. It defines which
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
- Treat task diversity as part of the model contract and store the exact data version.
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

Worked reasoning pattern for task diversity:
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

AI connection: Task diversity is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 3.2 Chat templates

Chat templates belongs in the canonical scope of instruction tuning and sft. The object
is the instruction-following policy, not merely a prompt trick or a moderation label. We
study how data, losses, policies, review processes, and safety constraints shape a
model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol \pi_\theta(y \mid x).
It marks the alignment object being transformed: an instruction policy, a preference
pair, a violation classifier, a guardrail action, or a feedback event. The details
differ, but the discipline is the same: state the object, state the loss or decision
rule, then audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{SFT}}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t \in R_i}\log \pi_\theta(y_{i,t} \mid x_i,y_{i,<t}).
$$

For chat templates, this formula should not be treated as a slogan. It defines which
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
- Treat chat templates as part of the model contract and store the exact data version.
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

Worked reasoning pattern for chat templates:
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

AI connection: Chat templates is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 3.3 Role tokens

Role tokens belongs in the canonical scope of instruction tuning and sft. The object is
the instruction-following policy, not merely a prompt trick or a moderation label. We
study how data, losses, policies, review processes, and safety constraints shape a
model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol \pi_\theta(y \mid x).
It marks the alignment object being transformed: an instruction policy, a preference
pair, a violation classifier, a guardrail action, or a feedback event. The details
differ, but the discipline is the same: state the object, state the loss or decision
rule, then audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{SFT}}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t \in R_i}\log \pi_\theta(y_{i,t} \mid x_i,y_{i,<t}).
$$

For role tokens, this formula should not be treated as a slogan. It defines which
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
- Treat role tokens as part of the model contract and store the exact data version.
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

Worked reasoning pattern for role tokens:
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

AI connection: Role tokens is part of the post-training stack used by modern assistant
systems. It links the base language model to human intent, safety policy, and deployment
constraints without pretending that a single loss can capture all values. The goal is
not perfect alignment by formula; it is a repeatable loop where evidence, objectives,
and safeguards improve together.

### 3.4 Refusal examples

Refusal examples belongs in the canonical scope of instruction tuning and sft. The
object is the instruction-following policy, not merely a prompt trick or a moderation
label. We study how data, losses, policies, review processes, and safety constraints
shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol \pi_\theta(y \mid x).
It marks the alignment object being transformed: an instruction policy, a preference
pair, a violation classifier, a guardrail action, or a feedback event. The details
differ, but the discipline is the same: state the object, state the loss or decision
rule, then audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{SFT}}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t \in R_i}\log \pi_\theta(y_{i,t} \mid x_i,y_{i,<t}).
$$

For refusal examples, this formula should not be treated as a slogan. It defines which
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
- Treat refusal examples as part of the model contract and store the exact data version.
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

Worked reasoning pattern for refusal examples:
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

AI connection: Refusal examples is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 3.5 Multi-turn demonstrations

Multi-turn demonstrations belongs in the canonical scope of instruction tuning and sft.
The object is the instruction-following policy, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol \pi_\theta(y \mid x).
It marks the alignment object being transformed: an instruction policy, a preference
pair, a violation classifier, a guardrail action, or a feedback event. The details
differ, but the discipline is the same: state the object, state the loss or decision
rule, then audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{SFT}}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t \in R_i}\log \pi_\theta(y_{i,t} \mid x_i,y_{i,<t}).
$$

For multi-turn demonstrations, this formula should not be treated as a slogan. It
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
- Treat multi-turn demonstrations as part of the model contract and store the exact data
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

Worked reasoning pattern for multi-turn demonstrations:
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

AI connection: Multi-turn demonstrations is part of the post-training stack used by
modern assistant systems. It links the base language model to human intent, safety
policy, and deployment constraints without pretending that a single loss can capture all
values. The goal is not perfect alignment by formula; it is a repeatable loop where
evidence, objectives, and safeguards improve together.

## 4. SFT Objective

SFT Objective develops the part of instruction tuning and sft that the approved TOC
assigns to Chapter 18. The emphasis is alignment behavior, safety constraints, and
feedback loops, not generic fine-tuning or production monitoring.

### 4.1 Response-only cross-entropy

Response-only cross-entropy belongs in the canonical scope of instruction tuning and
sft. The object is the instruction-following policy, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol \pi_\theta(y \mid x).
It marks the alignment object being transformed: an instruction policy, a preference
pair, a violation classifier, a guardrail action, or a feedback event. The details
differ, but the discipline is the same: state the object, state the loss or decision
rule, then audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{SFT}}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t \in R_i}\log \pi_\theta(y_{i,t} \mid x_i,y_{i,<t}).
$$

For response-only cross-entropy, this formula should not be treated as a slogan. It
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
- Treat response-only cross-entropy as part of the model contract and store the exact
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

Worked reasoning pattern for response-only cross-entropy:
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

AI connection: Response-only cross-entropy is part of the post-training stack used by
modern assistant systems. It links the base language model to human intent, safety
policy, and deployment constraints without pretending that a single loss can capture all
values. The goal is not perfect alignment by formula; it is a repeatable loop where
evidence, objectives, and safeguards improve together.

### 4.2 Packed examples

Packed examples belongs in the canonical scope of instruction tuning and sft. The object
is the instruction-following policy, not merely a prompt trick or a moderation label. We
study how data, losses, policies, review processes, and safety constraints shape a
model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol \pi_\theta(y \mid x).
It marks the alignment object being transformed: an instruction policy, a preference
pair, a violation classifier, a guardrail action, or a feedback event. The details
differ, but the discipline is the same: state the object, state the loss or decision
rule, then audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{SFT}}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t \in R_i}\log \pi_\theta(y_{i,t} \mid x_i,y_{i,<t}).
$$

For packed examples, this formula should not be treated as a slogan. It defines which
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
- Treat packed examples as part of the model contract and store the exact data version.
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

Worked reasoning pattern for packed examples:
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

AI connection: Packed examples is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 4.3 Loss masks

Loss masks belongs in the canonical scope of instruction tuning and sft. The object is
the instruction-following policy, not merely a prompt trick or a moderation label. We
study how data, losses, policies, review processes, and safety constraints shape a
model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol \pi_\theta(y \mid x).
It marks the alignment object being transformed: an instruction policy, a preference
pair, a violation classifier, a guardrail action, or a feedback event. The details
differ, but the discipline is the same: state the object, state the loss or decision
rule, then audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{SFT}}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t \in R_i}\log \pi_\theta(y_{i,t} \mid x_i,y_{i,<t}).
$$

For loss masks, this formula should not be treated as a slogan. It defines which tokens,
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
- Treat loss masks as part of the model contract and store the exact data version.
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

Worked reasoning pattern for loss masks:
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

AI connection: Loss masks is part of the post-training stack used by modern assistant
systems. It links the base language model to human intent, safety policy, and deployment
constraints without pretending that a single loss can capture all values. The goal is
not perfect alignment by formula; it is a repeatable loop where evidence, objectives,
and safeguards improve together.

### 4.4 Class imbalance

Class imbalance belongs in the canonical scope of instruction tuning and sft. The object
is the instruction-following policy, not merely a prompt trick or a moderation label. We
study how data, losses, policies, review processes, and safety constraints shape a
model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol \pi_\theta(y \mid x).
It marks the alignment object being transformed: an instruction policy, a preference
pair, a violation classifier, a guardrail action, or a feedback event. The details
differ, but the discipline is the same: state the object, state the loss or decision
rule, then audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{SFT}}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t \in R_i}\log \pi_\theta(y_{i,t} \mid x_i,y_{i,<t}).
$$

For class imbalance, this formula should not be treated as a slogan. It defines which
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
- Treat class imbalance as part of the model contract and store the exact data version.
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

Worked reasoning pattern for class imbalance:
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

AI connection: Class imbalance is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 4.5 Validation curves

Validation curves belongs in the canonical scope of instruction tuning and sft. The
object is the instruction-following policy, not merely a prompt trick or a moderation
label. We study how data, losses, policies, review processes, and safety constraints
shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol \pi_\theta(y \mid x).
It marks the alignment object being transformed: an instruction policy, a preference
pair, a violation classifier, a guardrail action, or a feedback event. The details
differ, but the discipline is the same: state the object, state the loss or decision
rule, then audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{SFT}}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t \in R_i}\log \pi_\theta(y_{i,t} \mid x_i,y_{i,<t}).
$$

For validation curves, this formula should not be treated as a slogan. It defines which
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
- Treat validation curves as part of the model contract and store the exact data
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

Worked reasoning pattern for validation curves:
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

AI connection: Validation curves is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

## 5. Alignment Behavior

Alignment Behavior develops the part of instruction tuning and sft that the approved TOC
assigns to Chapter 18. The emphasis is alignment behavior, safety constraints, and
feedback loops, not generic fine-tuning or production monitoring.

### 5.1 Helpfulness

Helpfulness belongs in the canonical scope of instruction tuning and sft. The object is
the instruction-following policy, not merely a prompt trick or a moderation label. We
study how data, losses, policies, review processes, and safety constraints shape a
model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol \pi_\theta(y \mid x).
It marks the alignment object being transformed: an instruction policy, a preference
pair, a violation classifier, a guardrail action, or a feedback event. The details
differ, but the discipline is the same: state the object, state the loss or decision
rule, then audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{SFT}}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t \in R_i}\log \pi_\theta(y_{i,t} \mid x_i,y_{i,<t}).
$$

For helpfulness, this formula should not be treated as a slogan. It defines which
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
- Treat helpfulness as part of the model contract and store the exact data version.
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

Worked reasoning pattern for helpfulness:
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

AI connection: Helpfulness is part of the post-training stack used by modern assistant
systems. It links the base language model to human intent, safety policy, and deployment
constraints without pretending that a single loss can capture all values. The goal is
not perfect alignment by formula; it is a repeatable loop where evidence, objectives,
and safeguards improve together.

### 5.2 Honesty

Honesty belongs in the canonical scope of instruction tuning and sft. The object is the
instruction-following policy, not merely a prompt trick or a moderation label. We study
how data, losses, policies, review processes, and safety constraints shape a model's
conditional distribution over responses.

A compact way to read this subsection is through the local symbol \pi_\theta(y \mid x).
It marks the alignment object being transformed: an instruction policy, a preference
pair, a violation classifier, a guardrail action, or a feedback event. The details
differ, but the discipline is the same: state the object, state the loss or decision
rule, then audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{SFT}}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t \in R_i}\log \pi_\theta(y_{i,t} \mid x_i,y_{i,<t}).
$$

For honesty, this formula should not be treated as a slogan. It defines which tokens,
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
- Treat honesty as part of the model contract and store the exact data version.
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

Worked reasoning pattern for honesty:
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

AI connection: Honesty is part of the post-training stack used by modern assistant
systems. It links the base language model to human intent, safety policy, and deployment
constraints without pretending that a single loss can capture all values. The goal is
not perfect alignment by formula; it is a repeatable loop where evidence, objectives,
and safeguards improve together.

### 5.3 Harmlessness

Harmlessness belongs in the canonical scope of instruction tuning and sft. The object is
the instruction-following policy, not merely a prompt trick or a moderation label. We
study how data, losses, policies, review processes, and safety constraints shape a
model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol \pi_\theta(y \mid x).
It marks the alignment object being transformed: an instruction policy, a preference
pair, a violation classifier, a guardrail action, or a feedback event. The details
differ, but the discipline is the same: state the object, state the loss or decision
rule, then audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{SFT}}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t \in R_i}\log \pi_\theta(y_{i,t} \mid x_i,y_{i,<t}).
$$

For harmlessness, this formula should not be treated as a slogan. It defines which
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
- Treat harmlessness as part of the model contract and store the exact data version.
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

Worked reasoning pattern for harmlessness:
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

AI connection: Harmlessness is part of the post-training stack used by modern assistant
systems. It links the base language model to human intent, safety policy, and deployment
constraints without pretending that a single loss can capture all values. The goal is
not perfect alignment by formula; it is a repeatable loop where evidence, objectives,
and safeguards improve together.

### 5.4 Style control

Style control belongs in the canonical scope of instruction tuning and sft. The object
is the instruction-following policy, not merely a prompt trick or a moderation label. We
study how data, losses, policies, review processes, and safety constraints shape a
model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol \pi_\theta(y \mid x).
It marks the alignment object being transformed: an instruction policy, a preference
pair, a violation classifier, a guardrail action, or a feedback event. The details
differ, but the discipline is the same: state the object, state the loss or decision
rule, then audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{SFT}}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t \in R_i}\log \pi_\theta(y_{i,t} \mid x_i,y_{i,<t}).
$$

For style control, this formula should not be treated as a slogan. It defines which
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
- Treat style control as part of the model contract and store the exact data version.
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

Worked reasoning pattern for style control:
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

AI connection: Style control is part of the post-training stack used by modern assistant
systems. It links the base language model to human intent, safety policy, and deployment
constraints without pretending that a single loss can capture all values. The goal is
not perfect alignment by formula; it is a repeatable loop where evidence, objectives,
and safeguards improve together.

### 5.5 Sycophancy risk

Sycophancy risk belongs in the canonical scope of instruction tuning and sft. The object
is the instruction-following policy, not merely a prompt trick or a moderation label. We
study how data, losses, policies, review processes, and safety constraints shape a
model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol \pi_\theta(y \mid x).
It marks the alignment object being transformed: an instruction policy, a preference
pair, a violation classifier, a guardrail action, or a feedback event. The details
differ, but the discipline is the same: state the object, state the loss or decision
rule, then audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{SFT}}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t \in R_i}\log \pi_\theta(y_{i,t} \mid x_i,y_{i,<t}).
$$

For sycophancy risk, this formula should not be treated as a slogan. It defines which
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
- Treat sycophancy risk as part of the model contract and store the exact data version.
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

Worked reasoning pattern for sycophancy risk:
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

AI connection: Sycophancy risk is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

## 6. SFT Limits

SFT Limits develops the part of instruction tuning and sft that the approved TOC assigns
to Chapter 18. The emphasis is alignment behavior, safety constraints, and feedback
loops, not generic fine-tuning or production monitoring.

### 6.1 Imitation ceiling

Imitation ceiling belongs in the canonical scope of instruction tuning and sft. The
object is the instruction-following policy, not merely a prompt trick or a moderation
label. We study how data, losses, policies, review processes, and safety constraints
shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol \pi_\theta(y \mid x).
It marks the alignment object being transformed: an instruction policy, a preference
pair, a violation classifier, a guardrail action, or a feedback event. The details
differ, but the discipline is the same: state the object, state the loss or decision
rule, then audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{SFT}}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t \in R_i}\log \pi_\theta(y_{i,t} \mid x_i,y_{i,<t}).
$$

For imitation ceiling, this formula should not be treated as a slogan. It defines which
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
- Treat imitation ceiling as part of the model contract and store the exact data
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

Worked reasoning pattern for imitation ceiling:
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

AI connection: Imitation ceiling is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 6.2 Hallucination

Hallucination belongs in the canonical scope of instruction tuning and sft. The object
is the instruction-following policy, not merely a prompt trick or a moderation label. We
study how data, losses, policies, review processes, and safety constraints shape a
model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol \pi_\theta(y \mid x).
It marks the alignment object being transformed: an instruction policy, a preference
pair, a violation classifier, a guardrail action, or a feedback event. The details
differ, but the discipline is the same: state the object, state the loss or decision
rule, then audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{SFT}}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t \in R_i}\log \pi_\theta(y_{i,t} \mid x_i,y_{i,<t}).
$$

For hallucination, this formula should not be treated as a slogan. It defines which
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
- Treat hallucination as part of the model contract and store the exact data version.
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

Worked reasoning pattern for hallucination:
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

AI connection: Hallucination is part of the post-training stack used by modern assistant
systems. It links the base language model to human intent, safety policy, and deployment
constraints without pretending that a single loss can capture all values. The goal is
not perfect alignment by formula; it is a repeatable loop where evidence, objectives,
and safeguards improve together.

### 6.3 Over-refusal

Over-refusal belongs in the canonical scope of instruction tuning and sft. The object is
the instruction-following policy, not merely a prompt trick or a moderation label. We
study how data, losses, policies, review processes, and safety constraints shape a
model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol \pi_\theta(y \mid x).
It marks the alignment object being transformed: an instruction policy, a preference
pair, a violation classifier, a guardrail action, or a feedback event. The details
differ, but the discipline is the same: state the object, state the loss or decision
rule, then audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{SFT}}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t \in R_i}\log \pi_\theta(y_{i,t} \mid x_i,y_{i,<t}).
$$

For over-refusal, this formula should not be treated as a slogan. It defines which
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
- Treat over-refusal as part of the model contract and store the exact data version.
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

Worked reasoning pattern for over-refusal:
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

AI connection: Over-refusal is part of the post-training stack used by modern assistant
systems. It links the base language model to human intent, safety policy, and deployment
constraints without pretending that a single loss can capture all values. The goal is
not perfect alignment by formula; it is a repeatable loop where evidence, objectives,
and safeguards improve together.

### 6.4 Catastrophic forgetting

Catastrophic forgetting belongs in the canonical scope of instruction tuning and sft.
The object is the instruction-following policy, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol \pi_\theta(y \mid x).
It marks the alignment object being transformed: an instruction policy, a preference
pair, a violation classifier, a guardrail action, or a feedback event. The details
differ, but the discipline is the same: state the object, state the loss or decision
rule, then audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{SFT}}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t \in R_i}\log \pi_\theta(y_{i,t} \mid x_i,y_{i,<t}).
$$

For catastrophic forgetting, this formula should not be treated as a slogan. It defines
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
- Treat catastrophic forgetting as part of the model contract and store the exact data
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

Worked reasoning pattern for catastrophic forgetting:
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

AI connection: Catastrophic forgetting is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 6.5 Distribution mismatch

Distribution mismatch belongs in the canonical scope of instruction tuning and sft. The
object is the instruction-following policy, not merely a prompt trick or a moderation
label. We study how data, losses, policies, review processes, and safety constraints
shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol \pi_\theta(y \mid x).
It marks the alignment object being transformed: an instruction policy, a preference
pair, a violation classifier, a guardrail action, or a feedback event. The details
differ, but the discipline is the same: state the object, state the loss or decision
rule, then audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{SFT}}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t \in R_i}\log \pi_\theta(y_{i,t} \mid x_i,y_{i,<t}).
$$

For distribution mismatch, this formula should not be treated as a slogan. It defines
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
- Treat distribution mismatch as part of the model contract and store the exact data
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

Worked reasoning pattern for distribution mismatch:
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

AI connection: Distribution mismatch is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

## 7. Synthetic and Self-Generated Instructions

Synthetic and Self-Generated Instructions develops the part of instruction tuning and
sft that the approved TOC assigns to Chapter 18. The emphasis is alignment behavior,
safety constraints, and feedback loops, not generic fine-tuning or production
monitoring.

### 7.1 FLAN-style multitask instruction tuning

FLAN-style multitask instruction tuning belongs in the canonical scope of instruction
tuning and sft. The object is the instruction-following policy, not merely a prompt
trick or a moderation label. We study how data, losses, policies, review processes, and
safety constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol \pi_\theta(y \mid x).
It marks the alignment object being transformed: an instruction policy, a preference
pair, a violation classifier, a guardrail action, or a feedback event. The details
differ, but the discipline is the same: state the object, state the loss or decision
rule, then audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{SFT}}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t \in R_i}\log \pi_\theta(y_{i,t} \mid x_i,y_{i,<t}).
$$

For flan-style multitask instruction tuning, this formula should not be treated as a
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
- Treat flan-style multitask instruction tuning as part of the model contract and store
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

Worked reasoning pattern for flan-style multitask instruction tuning:
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

AI connection: FLAN-style multitask instruction tuning is part of the post-training
stack used by modern assistant systems. It links the base language model to human
intent, safety policy, and deployment constraints without pretending that a single loss
can capture all values. The goal is not perfect alignment by formula; it is a repeatable
loop where evidence, objectives, and safeguards improve together.

### 7.2 Self-Instruct bootstrapping

Self-Instruct bootstrapping belongs in the canonical scope of instruction tuning and
sft. The object is the instruction-following policy, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol \pi_\theta(y \mid x).
It marks the alignment object being transformed: an instruction policy, a preference
pair, a violation classifier, a guardrail action, or a feedback event. The details
differ, but the discipline is the same: state the object, state the loss or decision
rule, then audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{SFT}}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t \in R_i}\log \pi_\theta(y_{i,t} \mid x_i,y_{i,<t}).
$$

For self-instruct bootstrapping, this formula should not be treated as a slogan. It
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
- Treat self-instruct bootstrapping as part of the model contract and store the exact
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

Worked reasoning pattern for self-instruct bootstrapping:
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

AI connection: Self-Instruct bootstrapping is part of the post-training stack used by
modern assistant systems. It links the base language model to human intent, safety
policy, and deployment constraints without pretending that a single loss can capture all
values. The goal is not perfect alignment by formula; it is a repeatable loop where
evidence, objectives, and safeguards improve together.

### 7.3 Data filtering

Data filtering belongs in the canonical scope of instruction tuning and sft. The object
is the instruction-following policy, not merely a prompt trick or a moderation label. We
study how data, losses, policies, review processes, and safety constraints shape a
model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol \pi_\theta(y \mid x).
It marks the alignment object being transformed: an instruction policy, a preference
pair, a violation classifier, a guardrail action, or a feedback event. The details
differ, but the discipline is the same: state the object, state the loss or decision
rule, then audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{SFT}}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t \in R_i}\log \pi_\theta(y_{i,t} \mid x_i,y_{i,<t}).
$$

For data filtering, this formula should not be treated as a slogan. It defines which
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
- Treat data filtering as part of the model contract and store the exact data version.
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

Worked reasoning pattern for data filtering:
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

AI connection: Data filtering is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 7.4 Quality scoring

Quality scoring belongs in the canonical scope of instruction tuning and sft. The object
is the instruction-following policy, not merely a prompt trick or a moderation label. We
study how data, losses, policies, review processes, and safety constraints shape a
model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol \pi_\theta(y \mid x).
It marks the alignment object being transformed: an instruction policy, a preference
pair, a violation classifier, a guardrail action, or a feedback event. The details
differ, but the discipline is the same: state the object, state the loss or decision
rule, then audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{SFT}}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t \in R_i}\log \pi_\theta(y_{i,t} \mid x_i,y_{i,<t}).
$$

For quality scoring, this formula should not be treated as a slogan. It defines which
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
- Treat quality scoring as part of the model contract and store the exact data version.
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

Worked reasoning pattern for quality scoring:
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

AI connection: Quality scoring is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 7.5 Bootstrapping loops

Bootstrapping loops belongs in the canonical scope of instruction tuning and sft. The
object is the instruction-following policy, not merely a prompt trick or a moderation
label. We study how data, losses, policies, review processes, and safety constraints
shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol \pi_\theta(y \mid x).
It marks the alignment object being transformed: an instruction policy, a preference
pair, a violation classifier, a guardrail action, or a feedback event. The details
differ, but the discipline is the same: state the object, state the loss or decision
rule, then audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{SFT}}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t \in R_i}\log \pi_\theta(y_{i,t} \mid x_i,y_{i,<t}).
$$

For bootstrapping loops, this formula should not be treated as a slogan. It defines
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
- Treat bootstrapping loops as part of the model contract and store the exact data
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

Worked reasoning pattern for bootstrapping loops:
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

AI connection: Bootstrapping loops is part of the post-training stack used by modern
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

1. (*) Why pretrained next-token models need instruction-following alignment.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

2. (*) Instruction following as behavior conditioning.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

3. (*) Demonstrations versus preferences.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

4. (**) Chat templates as part of the objective.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

5. (**) Historical arc from FLAN to InstructGPT.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

6. (**) Prompt $x$ and response $y$.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

7. (***) Demonstration dataset $\mathcal{D}_{\mathrm{SFT}}$.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

8. (***) Policy $\pi_\theta(y \mid x)$.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

9. (***) Response-token mask.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

10. (***) Instruction distribution and validation split.
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

- [FLAN: Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652)
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [Self-Instruct](https://arxiv.org/abs/2212.10560)
- [Fine-Tuning Math](../../15-Math-for-LLMs/07-Fine-Tuning-Math/notes.md)
- [KL Divergence](../../09-Information-Theory/02-KL-Divergence/notes.md)
- [Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md)
- [Fine-Tuning Math](../../15-Math-for-LLMs/07-Fine-Tuning-Math/notes.md)
- [Evaluation and Reliability](../../17-Evaluation-and-Reliability/01-Capability-Benchmarks/notes.md)

