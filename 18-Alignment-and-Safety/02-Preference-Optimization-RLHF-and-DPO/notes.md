[Back to Curriculum](../../README.md) | [Previous: Instruction Tuning and SFT](../01-Instruction-Tuning-and-SFT/notes.md) | [Next: Red Teaming and Safety Evaluations](../03-Red-Teaming-and-Safety-Evaluations/notes.md)

---

# Preference Optimization RLHF and DPO

> _"Preferences teach the model which good-looking answers people actually choose."_

## Overview

Preference optimization learns from comparisons, either by fitting a reward model and
optimizing a KL-regularized policy or by directly optimizing policy log-ratios.

Alignment and safety sit between model training and model deployment. The chapter
studies how desired behavior is specified, learned, stress-tested, constrained, and
improved through feedback.

This section is written in LaTeX Markdown. Inline mathematics uses `$...$`, and display
equations use `$$...$$`. The notes use math to expose the operational contract: which
behavior is optimized, which behavior is forbidden, and which evidence proves the system
changed.

## Prerequisites

- [KL Divergence](../../09-Information-Theory/02-KL-Divergence/notes.md)
- [Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md)
- [Fine-Tuning Math](../../15-Math-for-LLMs/07-Fine-Tuning-Math/notes.md)
- [Calibration and Uncertainty](../../17-Evaluation-and-Reliability/02-Calibration-and-Uncertainty/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Executable demonstrations for preference optimization rlhf and dpo |
| [exercises.ipynb](exercises.ipynb) | Graded practice for preference optimization rlhf and dpo |

## Learning Objectives

After completing this section, you will be able to:

- Define the core mathematical objects in preference optimization rlhf and dpo
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
  - [1.1 Preferences optimize choices rather than demonstrations](#11-preferences-optimize-choices-rather-than-demonstrations)
  - [1.2 Reward models as learned proxies](#12-reward-models-as-learned-proxies)
  - [1.3 Policy shift under a KL budget](#13-policy-shift-under-a-kl-budget)
  - [1.4 DPO as direct reward-model-free training](#14-dpo-as-direct-reward-model-free-training)
  - [1.5 Why preference data is noisy but valuable](#15-why-preference-data-is-noisy-but-valuable)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Preference pair $(x,y_w,y_l)$](#21-preference-pair-xywyl)
  - [2.2 Reward model $r_\phi(x,y)$](#22-reward-model-rphixy)
  - [2.3 Bradley-Terry preference model](#23-bradley-terry-preference-model)
  - [2.4 Reference policy $\pi_{\mathrm{ref}}$](#24-reference-policy-pimathrmref)
  - [2.5 KL coefficient and inverse temperature $\beta$](#25-kl-coefficient-and-inverse-temperature-beta)
- [3. Reward Modeling](#3-reward-modeling)
  - [3.1 Pairwise logistic loss](#31-pairwise-logistic-loss)
  - [3.2 Reward margins](#32-reward-margins)
  - [3.3 Annotator disagreement](#33-annotator-disagreement)
  - [3.4 Reward calibration](#34-reward-calibration)
  - [3.5 RewardBench-style evaluation](#35-rewardbench-style-evaluation)
- [4. RLHF Pipeline](#4-rlhf-pipeline)
  - [4.1 SFT policy initialization](#41-sft-policy-initialization)
  - [4.2 Reward model training](#42-reward-model-training)
  - [4.3 PPO objective](#43-ppo-objective)
  - [4.4 KL penalty](#44-kl-penalty)
  - [4.5 Reward hacking controls](#45-reward-hacking-controls)
- [5. DPO Derivation](#5-dpo-derivation)
  - [5.1 KL-constrained optimal policy](#51-kl-constrained-optimal-policy)
  - [5.2 Implicit reward](#52-implicit-reward)
  - [5.3 DPO loss](#53-dpo-loss)
  - [5.4 $\beta$ tradeoff](#54-beta-tradeoff)
  - [5.5 Gradient interpretation](#55-gradient-interpretation)
- [6. Preference Optimization Variants](#6-preference-optimization-variants)
  - [6.1 IPO preview](#61-ipo-preview)
  - [6.2 ORPO preview](#62-orpo-preview)
  - [6.3 KTO preview](#63-kto-preview)
  - [6.4 RLAIF](#64-rlaif)
  - [6.5 GRPO and reasoning-RL preview](#65-grpo-and-reasoning-rl-preview)
- [7. Failure Modes](#7-failure-modes)
  - [7.1 Reward hacking](#71-reward-hacking)
  - [7.2 Preference overfitting](#72-preference-overfitting)
  - [7.3 Length bias](#73-length-bias)
  - [7.4 Judge bias](#74-judge-bias)
  - [7.5 Alignment tax](#75-alignment-tax)
- [8. Common Mistakes](#8-common-mistakes)
- [9. Exercises](#9-exercises)
- [10. Why This Matters for AI](#10-why-this-matters-for-ai)
- [11. Conceptual Bridge](#11-conceptual-bridge)
- [References](#references)

---

## 1. Intuition

Intuition develops the part of preference optimization rlhf and dpo that the approved
TOC assigns to Chapter 18. The emphasis is alignment behavior, safety constraints, and
feedback loops, not generic fine-tuning or production monitoring.

### 1.1 Preferences optimize choices rather than demonstrations

Preferences optimize choices rather than demonstrations belongs in the canonical scope
of preference optimization rlhf and dpo. The object is the preference-aligned policy,
not merely a prompt trick or a moderation label. We study how data, losses, policies,
review processes, and safety constraints shape a model's conditional distribution over
responses.

A compact way to read this subsection is through the local symbol (x,y_w,y_l). It marks
the alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right).
$$

For preferences optimize choices rather than demonstrations, this formula should not be
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
- Treat preferences optimize choices rather than demonstrations as part of the model
contract and store the exact data version.
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

Worked reasoning pattern for preferences optimize choices rather than demonstrations:
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

AI connection: Preferences optimize choices rather than demonstrations is part of the
post-training stack used by modern assistant systems. It links the base language model
to human intent, safety policy, and deployment constraints without pretending that a
single loss can capture all values. The goal is not perfect alignment by formula; it is
a repeatable loop where evidence, objectives, and safeguards improve together.

### 1.2 Reward models as learned proxies

Reward models as learned proxies belongs in the canonical scope of preference
optimization rlhf and dpo. The object is the preference-aligned policy, not merely a
prompt trick or a moderation label. We study how data, losses, policies, review
processes, and safety constraints shape a model's conditional distribution over
responses.

A compact way to read this subsection is through the local symbol (x,y_w,y_l). It marks
the alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right).
$$

For reward models as learned proxies, this formula should not be treated as a slogan. It
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
- Treat reward models as learned proxies as part of the model contract and store the
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

Worked reasoning pattern for reward models as learned proxies:
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

AI connection: Reward models as learned proxies is part of the post-training stack used
by modern assistant systems. It links the base language model to human intent, safety
policy, and deployment constraints without pretending that a single loss can capture all
values. The goal is not perfect alignment by formula; it is a repeatable loop where
evidence, objectives, and safeguards improve together.

### 1.3 Policy shift under a KL budget

Policy shift under a KL budget belongs in the canonical scope of preference optimization
rlhf and dpo. The object is the preference-aligned policy, not merely a prompt trick or
a moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x,y_w,y_l). It marks
the alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right).
$$

For policy shift under a kl budget, this formula should not be treated as a slogan. It
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
- Treat policy shift under a kl budget as part of the model contract and store the exact
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

Worked reasoning pattern for policy shift under a kl budget:
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

AI connection: Policy shift under a KL budget is part of the post-training stack used by
modern assistant systems. It links the base language model to human intent, safety
policy, and deployment constraints without pretending that a single loss can capture all
values. The goal is not perfect alignment by formula; it is a repeatable loop where
evidence, objectives, and safeguards improve together.

### 1.4 DPO as direct reward-model-free training

DPO as direct reward-model-free training belongs in the canonical scope of preference
optimization rlhf and dpo. The object is the preference-aligned policy, not merely a
prompt trick or a moderation label. We study how data, losses, policies, review
processes, and safety constraints shape a model's conditional distribution over
responses.

A compact way to read this subsection is through the local symbol (x,y_w,y_l). It marks
the alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right).
$$

For dpo as direct reward-model-free training, this formula should not be treated as a
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
- Treat dpo as direct reward-model-free training as part of the model contract and store
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

Worked reasoning pattern for dpo as direct reward-model-free training:
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

AI connection: DPO as direct reward-model-free training is part of the post-training
stack used by modern assistant systems. It links the base language model to human
intent, safety policy, and deployment constraints without pretending that a single loss
can capture all values. The goal is not perfect alignment by formula; it is a repeatable
loop where evidence, objectives, and safeguards improve together.

### 1.5 Why preference data is noisy but valuable

Why preference data is noisy but valuable belongs in the canonical scope of preference
optimization rlhf and dpo. The object is the preference-aligned policy, not merely a
prompt trick or a moderation label. We study how data, losses, policies, review
processes, and safety constraints shape a model's conditional distribution over
responses.

A compact way to read this subsection is through the local symbol (x,y_w,y_l). It marks
the alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right).
$$

For why preference data is noisy but valuable, this formula should not be treated as a
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
- Treat why preference data is noisy but valuable as part of the model contract and
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

Worked reasoning pattern for why preference data is noisy but valuable:
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

AI connection: Why preference data is noisy but valuable is part of the post-training
stack used by modern assistant systems. It links the base language model to human
intent, safety policy, and deployment constraints without pretending that a single loss
can capture all values. The goal is not perfect alignment by formula; it is a repeatable
loop where evidence, objectives, and safeguards improve together.

## 2. Formal Definitions

Formal Definitions develops the part of preference optimization rlhf and dpo that the
approved TOC assigns to Chapter 18. The emphasis is alignment behavior, safety
constraints, and feedback loops, not generic fine-tuning or production monitoring.

### 2.1 Preference pair $(x,y_w,y_l)$

Preference pair $(x,y_w,y_l)$ belongs in the canonical scope of preference optimization
rlhf and dpo. The object is the preference-aligned policy, not merely a prompt trick or
a moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x,y_w,y_l). It marks
the alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right).
$$

For preference pair $(x,y_w,y_l)$, this formula should not be treated as a slogan. It
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
- Treat preference pair $(x,y_w,y_l)$ as part of the model contract and store the exact
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

Worked reasoning pattern for preference pair $(x,y_w,y_l)$:
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

AI connection: Preference pair $(x,y_w,y_l)$ is part of the post-training stack used by
modern assistant systems. It links the base language model to human intent, safety
policy, and deployment constraints without pretending that a single loss can capture all
values. The goal is not perfect alignment by formula; it is a repeatable loop where
evidence, objectives, and safeguards improve together.

### 2.2 Reward model $r_\phi(x,y)$

Reward model $r_\phi(x,y)$ belongs in the canonical scope of preference optimization
rlhf and dpo. The object is the preference-aligned policy, not merely a prompt trick or
a moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x,y_w,y_l). It marks
the alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right).
$$

For reward model $r_\phi(x,y)$, this formula should not be treated as a slogan. It
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
- Treat reward model $r_\phi(x,y)$ as part of the model contract and store the exact
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

Worked reasoning pattern for reward model $r_\phi(x,y)$:
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

AI connection: Reward model $r_\phi(x,y)$ is part of the post-training stack used by
modern assistant systems. It links the base language model to human intent, safety
policy, and deployment constraints without pretending that a single loss can capture all
values. The goal is not perfect alignment by formula; it is a repeatable loop where
evidence, objectives, and safeguards improve together.

### 2.3 Bradley-Terry preference model

Bradley-Terry preference model belongs in the canonical scope of preference optimization
rlhf and dpo. The object is the preference-aligned policy, not merely a prompt trick or
a moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x,y_w,y_l). It marks
the alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right).
$$

For bradley-terry preference model, this formula should not be treated as a slogan. It
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
- Treat bradley-terry preference model as part of the model contract and store the exact
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

Worked reasoning pattern for bradley-terry preference model:
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

AI connection: Bradley-Terry preference model is part of the post-training stack used by
modern assistant systems. It links the base language model to human intent, safety
policy, and deployment constraints without pretending that a single loss can capture all
values. The goal is not perfect alignment by formula; it is a repeatable loop where
evidence, objectives, and safeguards improve together.

### 2.4 Reference policy $\pi_{\mathrm{ref}}$

Reference policy $\pi_{\mathrm{ref}}$ belongs in the canonical scope of preference
optimization rlhf and dpo. The object is the preference-aligned policy, not merely a
prompt trick or a moderation label. We study how data, losses, policies, review
processes, and safety constraints shape a model's conditional distribution over
responses.

A compact way to read this subsection is through the local symbol (x,y_w,y_l). It marks
the alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right).
$$

For reference policy $\pi_{\mathrm{ref}}$, this formula should not be treated as a
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
- Treat reference policy $\pi_{\mathrm{ref}}$ as part of the model contract and store
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

Worked reasoning pattern for reference policy $\pi_{\mathrm{ref}}$:
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

AI connection: Reference policy $\pi_{\mathrm{ref}}$ is part of the post-training stack
used by modern assistant systems. It links the base language model to human intent,
safety policy, and deployment constraints without pretending that a single loss can
capture all values. The goal is not perfect alignment by formula; it is a repeatable
loop where evidence, objectives, and safeguards improve together.

### 2.5 KL coefficient and inverse temperature $\beta$

KL coefficient and inverse temperature $\beta$ belongs in the canonical scope of
preference optimization rlhf and dpo. The object is the preference-aligned policy, not
merely a prompt trick or a moderation label. We study how data, losses, policies, review
processes, and safety constraints shape a model's conditional distribution over
responses.

A compact way to read this subsection is through the local symbol (x,y_w,y_l). It marks
the alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right).
$$

For kl coefficient and inverse temperature $\beta$, this formula should not be treated
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
- Treat kl coefficient and inverse temperature $\beta$ as part of the model contract and
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

Worked reasoning pattern for kl coefficient and inverse temperature $\beta$:
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

AI connection: KL coefficient and inverse temperature $\beta$ is part of the post-
training stack used by modern assistant systems. It links the base language model to
human intent, safety policy, and deployment constraints without pretending that a single
loss can capture all values. The goal is not perfect alignment by formula; it is a
repeatable loop where evidence, objectives, and safeguards improve together.

## 3. Reward Modeling

Reward Modeling develops the part of preference optimization rlhf and dpo that the
approved TOC assigns to Chapter 18. The emphasis is alignment behavior, safety
constraints, and feedback loops, not generic fine-tuning or production monitoring.

### 3.1 Pairwise logistic loss

Pairwise logistic loss belongs in the canonical scope of preference optimization rlhf
and dpo. The object is the preference-aligned policy, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x,y_w,y_l). It marks
the alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right).
$$

For pairwise logistic loss, this formula should not be treated as a slogan. It defines
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
- Treat pairwise logistic loss as part of the model contract and store the exact data
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

Worked reasoning pattern for pairwise logistic loss:
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

AI connection: Pairwise logistic loss is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 3.2 Reward margins

Reward margins belongs in the canonical scope of preference optimization rlhf and dpo.
The object is the preference-aligned policy, not merely a prompt trick or a moderation
label. We study how data, losses, policies, review processes, and safety constraints
shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x,y_w,y_l). It marks
the alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right).
$$

For reward margins, this formula should not be treated as a slogan. It defines which
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
- Treat reward margins as part of the model contract and store the exact data version.
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

Worked reasoning pattern for reward margins:
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

AI connection: Reward margins is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 3.3 Annotator disagreement

Annotator disagreement belongs in the canonical scope of preference optimization rlhf
and dpo. The object is the preference-aligned policy, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x,y_w,y_l). It marks
the alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right).
$$

For annotator disagreement, this formula should not be treated as a slogan. It defines
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
- Treat annotator disagreement as part of the model contract and store the exact data
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

Worked reasoning pattern for annotator disagreement:
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

AI connection: Annotator disagreement is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 3.4 Reward calibration

Reward calibration belongs in the canonical scope of preference optimization rlhf and
dpo. The object is the preference-aligned policy, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x,y_w,y_l). It marks
the alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right).
$$

For reward calibration, this formula should not be treated as a slogan. It defines which
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
- Treat reward calibration as part of the model contract and store the exact data
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

Worked reasoning pattern for reward calibration:
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

AI connection: Reward calibration is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 3.5 RewardBench-style evaluation

RewardBench-style evaluation belongs in the canonical scope of preference optimization
rlhf and dpo. The object is the preference-aligned policy, not merely a prompt trick or
a moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x,y_w,y_l). It marks
the alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right).
$$

For rewardbench-style evaluation, this formula should not be treated as a slogan. It
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
- Treat rewardbench-style evaluation as part of the model contract and store the exact
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

Worked reasoning pattern for rewardbench-style evaluation:
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

AI connection: RewardBench-style evaluation is part of the post-training stack used by
modern assistant systems. It links the base language model to human intent, safety
policy, and deployment constraints without pretending that a single loss can capture all
values. The goal is not perfect alignment by formula; it is a repeatable loop where
evidence, objectives, and safeguards improve together.

## 4. RLHF Pipeline

RLHF Pipeline develops the part of preference optimization rlhf and dpo that the
approved TOC assigns to Chapter 18. The emphasis is alignment behavior, safety
constraints, and feedback loops, not generic fine-tuning or production monitoring.

### 4.1 SFT policy initialization

SFT policy initialization belongs in the canonical scope of preference optimization rlhf
and dpo. The object is the preference-aligned policy, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x,y_w,y_l). It marks
the alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right).
$$

For sft policy initialization, this formula should not be treated as a slogan. It
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
- Treat sft policy initialization as part of the model contract and store the exact data
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

Worked reasoning pattern for sft policy initialization:
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

AI connection: SFT policy initialization is part of the post-training stack used by
modern assistant systems. It links the base language model to human intent, safety
policy, and deployment constraints without pretending that a single loss can capture all
values. The goal is not perfect alignment by formula; it is a repeatable loop where
evidence, objectives, and safeguards improve together.

### 4.2 Reward model training

Reward model training belongs in the canonical scope of preference optimization rlhf and
dpo. The object is the preference-aligned policy, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x,y_w,y_l). It marks
the alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right).
$$

For reward model training, this formula should not be treated as a slogan. It defines
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
- Treat reward model training as part of the model contract and store the exact data
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

Worked reasoning pattern for reward model training:
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

AI connection: Reward model training is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 4.3 PPO objective

PPO objective belongs in the canonical scope of preference optimization rlhf and dpo.
The object is the preference-aligned policy, not merely a prompt trick or a moderation
label. We study how data, losses, policies, review processes, and safety constraints
shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x,y_w,y_l). It marks
the alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right).
$$

For ppo objective, this formula should not be treated as a slogan. It defines which
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
- Treat ppo objective as part of the model contract and store the exact data version.
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

Worked reasoning pattern for ppo objective:
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

AI connection: PPO objective is part of the post-training stack used by modern assistant
systems. It links the base language model to human intent, safety policy, and deployment
constraints without pretending that a single loss can capture all values. The goal is
not perfect alignment by formula; it is a repeatable loop where evidence, objectives,
and safeguards improve together.

### 4.4 KL penalty

KL penalty belongs in the canonical scope of preference optimization rlhf and dpo. The
object is the preference-aligned policy, not merely a prompt trick or a moderation
label. We study how data, losses, policies, review processes, and safety constraints
shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x,y_w,y_l). It marks
the alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right).
$$

For kl penalty, this formula should not be treated as a slogan. It defines which tokens,
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
- Treat kl penalty as part of the model contract and store the exact data version.
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

Worked reasoning pattern for kl penalty:
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

AI connection: KL penalty is part of the post-training stack used by modern assistant
systems. It links the base language model to human intent, safety policy, and deployment
constraints without pretending that a single loss can capture all values. The goal is
not perfect alignment by formula; it is a repeatable loop where evidence, objectives,
and safeguards improve together.

### 4.5 Reward hacking controls

Reward hacking controls belongs in the canonical scope of preference optimization rlhf
and dpo. The object is the preference-aligned policy, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x,y_w,y_l). It marks
the alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right).
$$

For reward hacking controls, this formula should not be treated as a slogan. It defines
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
- Treat reward hacking controls as part of the model contract and store the exact data
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

Worked reasoning pattern for reward hacking controls:
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

AI connection: Reward hacking controls is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

## 5. DPO Derivation

DPO Derivation develops the part of preference optimization rlhf and dpo that the
approved TOC assigns to Chapter 18. The emphasis is alignment behavior, safety
constraints, and feedback loops, not generic fine-tuning or production monitoring.

### 5.1 KL-constrained optimal policy

KL-constrained optimal policy belongs in the canonical scope of preference optimization
rlhf and dpo. The object is the preference-aligned policy, not merely a prompt trick or
a moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x,y_w,y_l). It marks
the alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right).
$$

For kl-constrained optimal policy, this formula should not be treated as a slogan. It
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
- Treat kl-constrained optimal policy as part of the model contract and store the exact
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

Worked reasoning pattern for kl-constrained optimal policy:
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

AI connection: KL-constrained optimal policy is part of the post-training stack used by
modern assistant systems. It links the base language model to human intent, safety
policy, and deployment constraints without pretending that a single loss can capture all
values. The goal is not perfect alignment by formula; it is a repeatable loop where
evidence, objectives, and safeguards improve together.

### 5.2 Implicit reward

Implicit reward belongs in the canonical scope of preference optimization rlhf and dpo.
The object is the preference-aligned policy, not merely a prompt trick or a moderation
label. We study how data, losses, policies, review processes, and safety constraints
shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x,y_w,y_l). It marks
the alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right).
$$

For implicit reward, this formula should not be treated as a slogan. It defines which
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
- Treat implicit reward as part of the model contract and store the exact data version.
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

Worked reasoning pattern for implicit reward:
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

AI connection: Implicit reward is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 5.3 DPO loss

DPO loss belongs in the canonical scope of preference optimization rlhf and dpo. The
object is the preference-aligned policy, not merely a prompt trick or a moderation
label. We study how data, losses, policies, review processes, and safety constraints
shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x,y_w,y_l). It marks
the alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right).
$$

For dpo loss, this formula should not be treated as a slogan. It defines which tokens,
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
- Treat dpo loss as part of the model contract and store the exact data version.
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

Worked reasoning pattern for dpo loss:
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

AI connection: DPO loss is part of the post-training stack used by modern assistant
systems. It links the base language model to human intent, safety policy, and deployment
constraints without pretending that a single loss can capture all values. The goal is
not perfect alignment by formula; it is a repeatable loop where evidence, objectives,
and safeguards improve together.

### 5.4 $\beta$ tradeoff

$\beta$ tradeoff belongs in the canonical scope of preference optimization rlhf and dpo.
The object is the preference-aligned policy, not merely a prompt trick or a moderation
label. We study how data, losses, policies, review processes, and safety constraints
shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x,y_w,y_l). It marks
the alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right).
$$

For $\beta$ tradeoff, this formula should not be treated as a slogan. It defines which
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
- Treat $\beta$ tradeoff as part of the model contract and store the exact data version.
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

Worked reasoning pattern for $\beta$ tradeoff:
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

AI connection: $\beta$ tradeoff is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 5.5 Gradient interpretation

Gradient interpretation belongs in the canonical scope of preference optimization rlhf
and dpo. The object is the preference-aligned policy, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x,y_w,y_l). It marks
the alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right).
$$

For gradient interpretation, this formula should not be treated as a slogan. It defines
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
- Treat gradient interpretation as part of the model contract and store the exact data
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

Worked reasoning pattern for gradient interpretation:
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

AI connection: Gradient interpretation is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

## 6. Preference Optimization Variants

Preference Optimization Variants develops the part of preference optimization rlhf and
dpo that the approved TOC assigns to Chapter 18. The emphasis is alignment behavior,
safety constraints, and feedback loops, not generic fine-tuning or production
monitoring.

### 6.1 IPO preview

IPO preview belongs in the canonical scope of preference optimization rlhf and dpo. The
object is the preference-aligned policy, not merely a prompt trick or a moderation
label. We study how data, losses, policies, review processes, and safety constraints
shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x,y_w,y_l). It marks
the alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right).
$$

For ipo preview, this formula should not be treated as a slogan. It defines which
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
- Treat ipo preview as part of the model contract and store the exact data version.
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

Worked reasoning pattern for ipo preview:
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

AI connection: IPO preview is part of the post-training stack used by modern assistant
systems. It links the base language model to human intent, safety policy, and deployment
constraints without pretending that a single loss can capture all values. The goal is
not perfect alignment by formula; it is a repeatable loop where evidence, objectives,
and safeguards improve together.

### 6.2 ORPO preview

ORPO preview belongs in the canonical scope of preference optimization rlhf and dpo. The
object is the preference-aligned policy, not merely a prompt trick or a moderation
label. We study how data, losses, policies, review processes, and safety constraints
shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x,y_w,y_l). It marks
the alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right).
$$

For orpo preview, this formula should not be treated as a slogan. It defines which
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
- Treat orpo preview as part of the model contract and store the exact data version.
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

Worked reasoning pattern for orpo preview:
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

AI connection: ORPO preview is part of the post-training stack used by modern assistant
systems. It links the base language model to human intent, safety policy, and deployment
constraints without pretending that a single loss can capture all values. The goal is
not perfect alignment by formula; it is a repeatable loop where evidence, objectives,
and safeguards improve together.

### 6.3 KTO preview

KTO preview belongs in the canonical scope of preference optimization rlhf and dpo. The
object is the preference-aligned policy, not merely a prompt trick or a moderation
label. We study how data, losses, policies, review processes, and safety constraints
shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x,y_w,y_l). It marks
the alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right).
$$

For kto preview, this formula should not be treated as a slogan. It defines which
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
- Treat kto preview as part of the model contract and store the exact data version.
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

Worked reasoning pattern for kto preview:
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

AI connection: KTO preview is part of the post-training stack used by modern assistant
systems. It links the base language model to human intent, safety policy, and deployment
constraints without pretending that a single loss can capture all values. The goal is
not perfect alignment by formula; it is a repeatable loop where evidence, objectives,
and safeguards improve together.

### 6.4 RLAIF

RLAIF belongs in the canonical scope of preference optimization rlhf and dpo. The object
is the preference-aligned policy, not merely a prompt trick or a moderation label. We
study how data, losses, policies, review processes, and safety constraints shape a
model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x,y_w,y_l). It marks
the alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right).
$$

For rlaif, this formula should not be treated as a slogan. It defines which tokens,
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
- Treat rlaif as part of the model contract and store the exact data version.
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

Worked reasoning pattern for rlaif:
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

AI connection: RLAIF is part of the post-training stack used by modern assistant
systems. It links the base language model to human intent, safety policy, and deployment
constraints without pretending that a single loss can capture all values. The goal is
not perfect alignment by formula; it is a repeatable loop where evidence, objectives,
and safeguards improve together.

### 6.5 GRPO and reasoning-RL preview

GRPO and reasoning-RL preview belongs in the canonical scope of preference optimization
rlhf and dpo. The object is the preference-aligned policy, not merely a prompt trick or
a moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x,y_w,y_l). It marks
the alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right).
$$

For grpo and reasoning-rl preview, this formula should not be treated as a slogan. It
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
- Treat grpo and reasoning-rl preview as part of the model contract and store the exact
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

Worked reasoning pattern for grpo and reasoning-rl preview:
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

AI connection: GRPO and reasoning-RL preview is part of the post-training stack used by
modern assistant systems. It links the base language model to human intent, safety
policy, and deployment constraints without pretending that a single loss can capture all
values. The goal is not perfect alignment by formula; it is a repeatable loop where
evidence, objectives, and safeguards improve together.

## 7. Failure Modes

Failure Modes develops the part of preference optimization rlhf and dpo that the
approved TOC assigns to Chapter 18. The emphasis is alignment behavior, safety
constraints, and feedback loops, not generic fine-tuning or production monitoring.

### 7.1 Reward hacking

Reward hacking belongs in the canonical scope of preference optimization rlhf and dpo.
The object is the preference-aligned policy, not merely a prompt trick or a moderation
label. We study how data, losses, policies, review processes, and safety constraints
shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x,y_w,y_l). It marks
the alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right).
$$

For reward hacking, this formula should not be treated as a slogan. It defines which
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
- Treat reward hacking as part of the model contract and store the exact data version.
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

Worked reasoning pattern for reward hacking:
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

AI connection: Reward hacking is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 7.2 Preference overfitting

Preference overfitting belongs in the canonical scope of preference optimization rlhf
and dpo. The object is the preference-aligned policy, not merely a prompt trick or a
moderation label. We study how data, losses, policies, review processes, and safety
constraints shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x,y_w,y_l). It marks
the alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right).
$$

For preference overfitting, this formula should not be treated as a slogan. It defines
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
- Treat preference overfitting as part of the model contract and store the exact data
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

Worked reasoning pattern for preference overfitting:
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

AI connection: Preference overfitting is part of the post-training stack used by modern
assistant systems. It links the base language model to human intent, safety policy, and
deployment constraints without pretending that a single loss can capture all values. The
goal is not perfect alignment by formula; it is a repeatable loop where evidence,
objectives, and safeguards improve together.

### 7.3 Length bias

Length bias belongs in the canonical scope of preference optimization rlhf and dpo. The
object is the preference-aligned policy, not merely a prompt trick or a moderation
label. We study how data, losses, policies, review processes, and safety constraints
shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x,y_w,y_l). It marks
the alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right).
$$

For length bias, this formula should not be treated as a slogan. It defines which
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
- Treat length bias as part of the model contract and store the exact data version.
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

Worked reasoning pattern for length bias:
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

AI connection: Length bias is part of the post-training stack used by modern assistant
systems. It links the base language model to human intent, safety policy, and deployment
constraints without pretending that a single loss can capture all values. The goal is
not perfect alignment by formula; it is a repeatable loop where evidence, objectives,
and safeguards improve together.

### 7.4 Judge bias

Judge bias belongs in the canonical scope of preference optimization rlhf and dpo. The
object is the preference-aligned policy, not merely a prompt trick or a moderation
label. We study how data, losses, policies, review processes, and safety constraints
shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x,y_w,y_l). It marks
the alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right).
$$

For judge bias, this formula should not be treated as a slogan. It defines which tokens,
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
- Treat judge bias as part of the model contract and store the exact data version.
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

Worked reasoning pattern for judge bias:
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

AI connection: Judge bias is part of the post-training stack used by modern assistant
systems. It links the base language model to human intent, safety policy, and deployment
constraints without pretending that a single loss can capture all values. The goal is
not perfect alignment by formula; it is a repeatable loop where evidence, objectives,
and safeguards improve together.

### 7.5 Alignment tax

Alignment tax belongs in the canonical scope of preference optimization rlhf and dpo.
The object is the preference-aligned policy, not merely a prompt trick or a moderation
label. We study how data, losses, policies, review processes, and safety constraints
shape a model's conditional distribution over responses.

A compact way to read this subsection is through the local symbol (x,y_w,y_l). It marks
the alignment object being transformed: an instruction policy, a preference pair, a
violation classifier, a guardrail action, or a feedback event. The details differ, but
the discipline is the same: state the object, state the loss or decision rule, then
audit the behavioral side effects.

$$
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right).
$$

For alignment tax, this formula should not be treated as a slogan. It defines which
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
- Treat alignment tax as part of the model contract and store the exact data version.
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

Worked reasoning pattern for alignment tax:
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

AI connection: Alignment tax is part of the post-training stack used by modern assistant
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

1. (*) Preferences optimize choices rather than demonstrations.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

2. (*) Reward models as learned proxies.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

3. (*) Policy shift under a KL budget.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

4. (**) DPO as direct reward-model-free training.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

5. (**) Why preference data is noisy but valuable.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

6. (**) Preference pair $(x,y_w,y_l)$.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

7. (***) Reward model $r_\phi(x,y)$.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

8. (***) Bradley-Terry preference model.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

9. (***) Reference policy $\pi_{\mathrm{ref}}$.
Define the alignment object, write the relevant loss or decision rule, give one safe
example and one unsafe edge case, then explain which held-out metric would catch
regression.

10. (***) KL coefficient and inverse temperature $\beta$.
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

- [Deep reinforcement learning from human preferences](https://arxiv.org/abs/1706.03741)
- [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325)
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- [RewardBench](https://arxiv.org/abs/2403.13787)
- [KL Divergence](../../09-Information-Theory/02-KL-Divergence/notes.md)
- [Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md)
- [Fine-Tuning Math](../../15-Math-for-LLMs/07-Fine-Tuning-Math/notes.md)
- [Evaluation and Reliability](../../17-Evaluation-and-Reliability/01-Capability-Benchmarks/notes.md)

