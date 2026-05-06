[<- Training at Scale](../06-Training-at-Scale/notes.md) | [Home](../../README.md) | [Scaling Laws ->](../08-Scaling-Laws/notes.md)

---

# Fine-Tuning Math

Fine-tuning is controlled movement away from a pretrained model. The mathematics asks four questions: what loss is optimized, which parameters can move, how large is the update space, and how do we measure useful adaptation without damaging the base model.

## Overview

Pretraining learns a broad conditional distribution. Fine-tuning shifts that distribution toward a task, domain, instruction style, preference pattern, or deployment constraint. The shift can be full-rank and full-model, or it can be restricted to a small trainable subspace such as adapters, soft prompts, prefixes, or LoRA matrices.

The central decomposition is:

$$
\theta = \theta_0 + \Delta\theta.
$$

The base weights $\theta_0$ carry pretrained capability. The update $\Delta\theta$ carries adaptation. Fine-tuning math is the study of how to choose, constrain, optimize, and evaluate that update.

## Prerequisites

- Cross-entropy and answer-only token loss
- Training memory and optimizer-state accounting
- Matrix ranks, low-rank factorization, and SVD intuition
- KL divergence and conditional language-model scoring

## Companion Notebooks

| Notebook | Purpose |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Executable demos for SFT masking, LoRA parameter counts, low-rank updates, adapter counts, prefix counts, DPO loss, forgetting tradeoffs, and merge checks. |
| [exercises.ipynb](exercises.ipynb) | Ten practice problems for the update geometry and objective bookkeeping used in real fine-tuning runs. |

## Learning Objectives

After this section, you should be able to:

- Write the fine-tuning objective as local adaptation from pretrained weights.
- Compute answer-only SFT loss with prompt and padding masks.
- Explain full fine-tuning, linear probing, adapters, soft prompts, prefix tuning, and LoRA.
- Count trainable parameters for LoRA, adapters, and prefix methods.
- Derive the LoRA update $W'=W+(\alpha/r)BA$ and merge it for inference.
- Explain why PEFT reduces optimizer-state memory but not all activation memory.
- Compute a DPO preference loss from model and reference log probabilities.
- Diagnose forgetting, overfitting, mask bugs, and incorrect parameter freezing.

## Table of Contents

1. [Fine-Tuning as Local Adaptation](#1-finetuning-as-local-adaptation)
   - 1.1 [Starting from pretrained weights](#11-starting-from-pretrained-weights)
   - 1.2 [Task loss](#12-task-loss)
   - 1.3 [Regularized adaptation](#13-regularized-adaptation)
   - 1.4 [Function-space shift](#14-functionspace-shift)
   - 1.5 [Catastrophic forgetting](#15-catastrophic-forgetting)
2. [Supervised Fine-Tuning Objective](#2-supervised-finetuning-objective)
   - 2.1 [Instruction-response pairs](#21-instructionresponse-pairs)
   - 2.2 [Answer-only loss](#22-answeronly-loss)
   - 2.3 [Teacher forcing](#23-teacher-forcing)
   - 2.4 [Label smoothing](#24-label-smoothing)
   - 2.5 [Dataset mixture weights](#25-dataset-mixture-weights)
3. [Full Fine-Tuning](#3-full-finetuning)
   - 3.1 [All parameters trainable](#31-all-parameters-trainable)
   - 3.2 [Memory cost](#32-memory-cost)
   - 3.3 [Layer-wise learning rates](#33-layerwise-learning-rates)
   - 3.4 [Weight decay](#34-weight-decay)
   - 3.5 [When full tuning helps](#35-when-full-tuning-helps)
4. [Parameter-Efficient Fine-Tuning](#4-parameterefficient-finetuning)
   - 4.1 [Linear probing](#41-linear-probing)
   - 4.2 [Adapters](#42-adapters)
   - 4.3 [Soft prompt tuning](#43-soft-prompt-tuning)
   - 4.4 [Prefix tuning](#44-prefix-tuning)
   - 4.5 [Low-rank adaptation](#45-lowrank-adaptation)
5. [LoRA Algebra](#5-lora-algebra)
   - 5.1 [Rank constraint](#51-rank-constraint)
   - 5.2 [Parameter count](#52-parameter-count)
   - 5.3 [Scaling](#53-scaling)
   - 5.4 [Merge for inference](#54-merge-for-inference)
   - 5.5 [Target modules](#55-target-modules)
6. [Quantized and Memory-Aware Tuning](#6-quantized-and-memoryaware-tuning)
   - 6.1 [Frozen quantized base](#61-frozen-quantized-base)
   - 6.2 [Adapter optimizer states](#62-adapter-optimizer-states)
   - 6.3 [Activation memory remains](#63-activation-memory-remains)
   - 6.4 [Gradient checkpointing](#64-gradient-checkpointing)
   - 6.5 [Rank-memory tradeoff](#65-rankmemory-tradeoff)
7. [Preference Fine-Tuning](#7-preference-finetuning)
   - 7.1 [Preference pairs](#71-preference-pairs)
   - 7.2 [Reward-model view](#72-rewardmodel-view)
   - 7.3 [KL-regularized policy update](#73-klregularized-policy-update)
   - 7.4 [DPO loss](#74-dpo-loss)
   - 7.5 [Preference over-optimization](#75-preference-overoptimization)
8. [Evaluation and Diagnostics](#8-evaluation-and-diagnostics)
   - 8.1 [Train and validation loss](#81-train-and-validation-loss)
   - 8.2 [Base-task retention](#82-basetask-retention)
   - 8.3 [Task quality](#83-task-quality)
   - 8.4 [Distribution shift](#84-distribution-shift)
   - 8.5 [Adapter sanity checks](#85-adapter-sanity-checks)
9. [Choosing a Method](#9-choosing-a-method)
   - 9.1 [No-update methods](#91-noupdate-methods)
   - 9.2 [PEFT methods](#92-peft-methods)
   - 9.3 [Full tuning](#93-full-tuning)
   - 9.4 [Preference tuning](#94-preference-tuning)
   - 9.5 [Deployment constraints](#95-deployment-constraints)
10. [Implementation Checklist](#10-implementation-checklist)
   - 10.1 [Masking](#101-masking)
   - 10.2 [Parameter freeze audit](#102-parameter-freeze-audit)
   - 10.3 [Learning-rate groups](#103-learningrate-groups)
   - 10.4 [Reference model](#104-reference-model)
   - 10.5 [Ablations](#105-ablations)

---

## Method Map

| Method | Base weights | Trainable object | Main advantage | Main risk |
| --- | --- | --- | --- | --- |
| Prompting/RAG | Frozen | None | Zero training | May not enforce consistent behavior |
| Linear probing | Frozen | Output head | Cheap diagnostic | Limited generation adaptation |
| Adapters | Frozen mostly | Bottleneck modules | Multi-task modularity | Extra inference modules |
| Prompt/prefix tuning | Frozen | Continuous prompt or KV prefix | Very small parameter count | Capacity can be limited |
| LoRA | Frozen mostly | Low-rank matrix updates | Strong PEFT default | Rank/target choice matters |
| Full fine-tuning | Trainable | All weights | Maximum capacity | Expensive and can forget |
| Preference tuning | Usually partial or full | Policy update | Aligns comparative behavior | Over-optimization and reward hacking |

## 1. Fine-Tuning as Local Adaptation

This part treats fine-tuning as local adaptation as a measurable adaptation decision. The goal is to know what is updated, what objective is optimized, how much capacity the update has, and how to detect damage to the base model.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Starting from pretrained weights](#1-starting-from-pretrained-weights) | fine-tuning begins near a useful parameter point | $\theta=\theta_0+\Delta\theta$ |
| [Task loss](#1-task-loss) | the target distribution changes from broad pretraining to task behavior | $\min_\theta L_\mathrm{task}(\theta)$ |
| [Regularized adaptation](#1-regularized-adaptation) | penalize moving too far from the base model | $L(\theta)=L_\mathrm{task}(\theta)+\lambda R(\theta,\theta_0)$ |
| [Function-space shift](#1-functionspace-shift) | behavioral movement is better measured on outputs than raw parameters | $D_\mathrm{KL}(p_\theta(\cdot\mid x)\Vert p_{\theta_0}(\cdot\mid x))$ |
| [Catastrophic forgetting](#1-catastrophic-forgetting) | task learning can reduce general ability | $\Delta_\mathrm{forget}=S_\mathrm{base}(\theta_0)-S_\mathrm{base}(\theta)$ |

### 1.1 Starting from pretrained weights

**Main idea.** Fine-tuning begins near a useful parameter point.

Core relation:

$$\theta=\theta_0+\Delta\theta$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 1.2 Task loss

**Main idea.** The target distribution changes from broad pretraining to task behavior.

Core relation:

$$\min_\theta L_\mathrm{task}(\theta)$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 1.3 Regularized adaptation

**Main idea.** Penalize moving too far from the base model.

Core relation:

$$L(\theta)=L_\mathrm{task}(\theta)+\lambda R(\theta,\theta_0)$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 1.4 Function-space shift

**Main idea.** Behavioral movement is better measured on outputs than raw parameters.

Core relation:

$$D_\mathrm{KL}(p_\theta(\cdot\mid x)\Vert p_{\theta_0}(\cdot\mid x))$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 1.5 Catastrophic forgetting

**Main idea.** Task learning can reduce general ability.

Core relation:

$$\Delta_\mathrm{forget}=S_\mathrm{base}(\theta_0)-S_\mathrm{base}(\theta)$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
## 2. Supervised Fine-Tuning Objective

This part treats supervised fine-tuning objective as a measurable adaptation decision. The goal is to know what is updated, what objective is optimized, how much capacity the update has, and how to detect damage to the base model.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Instruction-response pairs](#2-instructionresponse-pairs) | train on prompt and desired completion pairs | $(x,y)\sim D_\mathrm{sft}$ |
| [Answer-only loss](#2-answeronly-loss) | mask prompt tokens when the objective is response imitation | $L_\mathrm{sft}=-\sum_j m_j\log p_\theta(y_j\mid x,y_{<j})/\sum_j m_j$ |
| [Teacher forcing](#2-teacher-forcing) | condition on the gold prefix during training | $p_\theta(y_j\mid x,y_{<j}^\star)$ |
| [Label smoothing](#2-label-smoothing) | soften one-hot targets when useful | $q=(1-\epsilon)y+\epsilon/|V|$ |
| [Dataset mixture weights](#2-dataset-mixture-weights) | combine multiple tasks by weighted expectation | $L=\sum_k \alpha_k E_{D_k}[\ell]$ |

### 2.1 Instruction-response pairs

**Main idea.** Train on prompt and desired completion pairs.

Core relation:

$$(x,y)\sim D_\mathrm{sft}$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 2.2 Answer-only loss

**Main idea.** Mask prompt tokens when the objective is response imitation.

Core relation:

$$L_\mathrm{sft}=-\sum_j m_j\log p_\theta(y_j\mid x,y_{<j})/\sum_j m_j$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is the difference between teaching the model to imitate the response and wasting loss on tokens it was handed in the prompt.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 2.3 Teacher forcing

**Main idea.** Condition on the gold prefix during training.

Core relation:

$$p_\theta(y_j\mid x,y_{<j}^\star)$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 2.4 Label smoothing

**Main idea.** Soften one-hot targets when useful.

Core relation:

$$q=(1-\epsilon)y+\epsilon/|V|$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 2.5 Dataset mixture weights

**Main idea.** Combine multiple tasks by weighted expectation.

Core relation:

$$L=\sum_k \alpha_k E_{D_k}[\ell]$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
## 3. Full Fine-Tuning

This part treats full fine-tuning as a measurable adaptation decision. The goal is to know what is updated, what objective is optimized, how much capacity the update has, and how to detect damage to the base model.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [All parameters trainable](#3-all-parameters-trainable) | full fine-tuning updates every weight tensor | $\Delta\theta\in\mathbb{R}^{|\theta|}$ |
| [Memory cost](#3-memory-cost) | trainable parameters require gradients and optimizer states | $M\approx M_\mathrm{weights}+M_\mathrm{grads}+M_\mathrm{opt}+M_\mathrm{act}$ |
| [Layer-wise learning rates](#3-layerwise-learning-rates) | lower layers can move more slowly than higher layers | $\eta_\ell=\eta_0\gamma^{L-\ell}$ |
| [Weight decay](#3-weight-decay) | regularize parameter norm during adaptation | $\theta\leftarrow\theta-\eta(\nabla L+\lambda\theta)$ |
| [When full tuning helps](#3-when-full-tuning-helps) | large domain shift or maximum quality can justify the cost | $\mathrm{benefit}>\mathrm{compute}+\mathrm{forgetting\ risk}$ |

### 3.1 All parameters trainable

**Main idea.** Full fine-tuning updates every weight tensor.

Core relation:

$$\Delta\theta\in\mathbb{R}^{|\theta|}$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 3.2 Memory cost

**Main idea.** Trainable parameters require gradients and optimizer states.

Core relation:

$$M\approx M_\mathrm{weights}+M_\mathrm{grads}+M_\mathrm{opt}+M_\mathrm{act}$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 3.3 Layer-wise learning rates

**Main idea.** Lower layers can move more slowly than higher layers.

Core relation:

$$\eta_\ell=\eta_0\gamma^{L-\ell}$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 3.4 Weight decay

**Main idea.** Regularize parameter norm during adaptation.

Core relation:

$$\theta\leftarrow\theta-\eta(\nabla L+\lambda\theta)$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 3.5 When full tuning helps

**Main idea.** Large domain shift or maximum quality can justify the cost.

Core relation:

$$\mathrm{benefit}>\mathrm{compute}+\mathrm{forgetting\ risk}$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
## 4. Parameter-Efficient Fine-Tuning

This part treats parameter-efficient fine-tuning as a measurable adaptation decision. The goal is to know what is updated, what objective is optimized, how much capacity the update has, and how to detect damage to the base model.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Linear probing](#4-linear-probing) | freeze the backbone and train only a head | $h=f_{\theta_0}(x),\quad \hat y=Wh+b$ |
| [Adapters](#4-adapters) | insert small bottleneck modules inside layers | $h'=h+W_\mathrm{up}\sigma(W_\mathrm{down}h)$ |
| [Soft prompt tuning](#4-soft-prompt-tuning) | learn continuous input embeddings | $[p_1,\ldots,p_m,x_1,\ldots,x_T]$ |
| [Prefix tuning](#4-prefix-tuning) | learn virtual key-value prefixes for attention | $K'=[K_\mathrm{prefix};K],\quad V'=[V_\mathrm{prefix};V]$ |
| [Low-rank adaptation](#4-lowrank-adaptation) | learn a low-rank update while freezing the base matrix | $W'=W+\frac{\alpha}{r}BA$ |

### 4.1 Linear probing

**Main idea.** Freeze the backbone and train only a head.

Core relation:

$$h=f_{\theta_0}(x),\quad \hat y=Wh+b$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 4.2 Adapters

**Main idea.** Insert small bottleneck modules inside layers.

Core relation:

$$h'=h+W_\mathrm{up}\sigma(W_\mathrm{down}h)$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 4.3 Soft prompt tuning

**Main idea.** Learn continuous input embeddings.

Core relation:

$$[p_1,\ldots,p_m,x_1,\ldots,x_T]$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 4.4 Prefix tuning

**Main idea.** Learn virtual key-value prefixes for attention.

Core relation:

$$K'=[K_\mathrm{prefix};K],\quad V'=[V_\mathrm{prefix};V]$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 4.5 Low-rank adaptation

**Main idea.** Learn a low-rank update while freezing the base matrix.

Core relation:

$$W'=W+\frac{\alpha}{r}BA$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** LoRA works because many useful task updates can be represented well inside a small trainable subspace.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
## 5. LoRA Algebra

This part treats lora algebra as a measurable adaptation decision. The goal is to know what is updated, what objective is optimized, how much capacity the update has, and how to detect damage to the base model.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Rank constraint](#5-rank-constraint) | the update matrix has rank at most r | $\mathrm{rank}(BA)\le r$ |
| [Parameter count](#5-parameter-count) | a d_out by d_in matrix gets r(d_in+d_out) trainable parameters | $P_\mathrm{LoRA}=r(d_\mathrm{in}+d_\mathrm{out})$ |
| [Scaling](#5-scaling) | the alpha over r factor controls update magnitude | $\Delta W=(\alpha/r)BA$ |
| [Merge for inference](#5-merge-for-inference) | after training, add the low-rank update into the base matrix | $W_\mathrm{merged}=W+\Delta W$ |
| [Target modules](#5-target-modules) | attention and MLP projections can receive separate adapters | $W_q,W_k,W_v,W_o,W_\mathrm{up},W_\mathrm{down}$ |

### 5.1 Rank constraint

**Main idea.** The update matrix has rank at most r.

Core relation:

$$\mathrm{rank}(BA)\le r$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 5.2 Parameter count

**Main idea.** A d_out by d_in matrix gets r(d_in+d_out) trainable parameters.

Core relation:

$$P_\mathrm{LoRA}=r(d_\mathrm{in}+d_\mathrm{out})$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 5.3 Scaling

**Main idea.** The alpha over r factor controls update magnitude.

Core relation:

$$\Delta W=(\alpha/r)BA$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 5.4 Merge for inference

**Main idea.** After training, add the low-rank update into the base matrix.

Core relation:

$$W_\mathrm{merged}=W+\Delta W$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is why LoRA can add no extra matrix multiplication at serving time after merging.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 5.5 Target modules

**Main idea.** Attention and mlp projections can receive separate adapters.

Core relation:

$$W_q,W_k,W_v,W_o,W_\mathrm{up},W_\mathrm{down}$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
## 6. Quantized and Memory-Aware Tuning

This part treats quantized and memory-aware tuning as a measurable adaptation decision. The goal is to know what is updated, what objective is optimized, how much capacity the update has, and how to detect damage to the base model.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Frozen quantized base](#6-frozen-quantized-base) | store base weights in low precision and train small adapters | $W_0\approx Q(W_0)$ |
| [Adapter optimizer states](#6-adapter-optimizer-states) | optimizer states are needed only for trainable adapter weights | $M_\mathrm{opt}\propto P_\mathrm{trainable}$ |
| [Activation memory remains](#6-activation-memory-remains) | PEFT reduces parameter-state memory but still backpropagates through the model | $M_\mathrm{act}$ can dominate |
| [Gradient checkpointing](#6-gradient-checkpointing) | recompute activations to fit longer sequences or larger batches | $\mathrm{memory}\downarrow,\quad\mathrm{compute}\uparrow$ |
| [Rank-memory tradeoff](#6-rankmemory-tradeoff) | higher rank improves capacity but increases trainable parameters | $P_\mathrm{LoRA}\propto r$ |

### 6.1 Frozen quantized base

**Main idea.** Store base weights in low precision and train small adapters.

Core relation:

$$W_0\approx Q(W_0)$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 6.2 Adapter optimizer states

**Main idea.** Optimizer states are needed only for trainable adapter weights.

Core relation:

$$M_\mathrm{opt}\propto P_\mathrm{trainable}$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 6.3 Activation memory remains

**Main idea.** Peft reduces parameter-state memory but still backpropagates through the model.

Core relation:

$$M_\mathrm{act}$ can dominate$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 6.4 Gradient checkpointing

**Main idea.** Recompute activations to fit longer sequences or larger batches.

Core relation:

$$\mathrm{memory}\downarrow,\quad\mathrm{compute}\uparrow$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 6.5 Rank-memory tradeoff

**Main idea.** Higher rank improves capacity but increases trainable parameters.

Core relation:

$$P_\mathrm{LoRA}\propto r$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
## 7. Preference Fine-Tuning

This part treats preference fine-tuning as a measurable adaptation decision. The goal is to know what is updated, what objective is optimized, how much capacity the update has, and how to detect damage to the base model.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Preference pairs](#7-preference-pairs) | learn from chosen and rejected responses | $(x,y^+,y^-)$ |
| [Reward-model view](#7-rewardmodel-view) | RLHF trains or uses a reward signal for completions | $r_\phi(x,y)$ |
| [KL-regularized policy update](#7-klregularized-policy-update) | keep the tuned model near a reference model | $E[r]-\beta D_\mathrm{KL}(\pi_\theta\Vert\pi_\mathrm{ref})$ |
| [DPO loss](#7-dpo-loss) | optimize preference likelihood directly with a reference model | $-\log\sigma(\beta[(\log\pi_\theta^+-\log\pi_\theta^-)-(\log\pi_\mathrm{ref}^+-\log\pi_\mathrm{ref}^-)])$ |
| [Preference over-optimization](#7-preference-overoptimization) | too much preference pressure can reduce diversity or factuality | $r\uparrow$ does not guarantee all qualities improve |

### 7.1 Preference pairs

**Main idea.** Learn from chosen and rejected responses.

Core relation:

$$(x,y^+,y^-)$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 7.2 Reward-model view

**Main idea.** Rlhf trains or uses a reward signal for completions.

Core relation:

$$r_\phi(x,y)$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 7.3 KL-regularized policy update

**Main idea.** Keep the tuned model near a reference model.

Core relation:

$$E[r]-\beta D_\mathrm{KL}(\pi_\theta\Vert\pi_\mathrm{ref})$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 7.4 DPO loss

**Main idea.** Optimize preference likelihood directly with a reference model.

Core relation:

$$-\log\sigma(\beta[(\log\pi_\theta^+-\log\pi_\theta^-)-(\log\pi_\mathrm{ref}^+-\log\pi_\mathrm{ref}^-)])$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** DPO turns a preference pair into a logistic loss over relative log probabilities.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 7.5 Preference over-optimization

**Main idea.** Too much preference pressure can reduce diversity or factuality.

Core relation:

$$r\uparrow$ does not guarantee all qualities improve$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
## 8. Evaluation and Diagnostics

This part treats evaluation and diagnostics as a measurable adaptation decision. The goal is to know what is updated, what objective is optimized, how much capacity the update has, and how to detect damage to the base model.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Train and validation loss](#8-train-and-validation-loss) | memorization shows up as train loss improving while validation stalls | $L_\mathrm{train}\downarrow,\quad L_\mathrm{val}\not\downarrow$ |
| [Base-task retention](#8-basetask-retention) | evaluate old skills after adaptation | $S_\mathrm{retain}(\theta)$ |
| [Task quality](#8-task-quality) | use task-specific automatic and human checks | $S_\mathrm{task}(\theta)$ |
| [Distribution shift](#8-distribution-shift) | fine-tune data should match deployment use | $p_\mathrm{train}(x,y)\approx p_\mathrm{deploy}(x,y)$ |
| [Adapter sanity checks](#8-adapter-sanity-checks) | disable the adapter to confirm measured change comes from the adapter | $f_{W+\Delta W}(x)-f_W(x)$ |

### 8.1 Train and validation loss

**Main idea.** Memorization shows up as train loss improving while validation stalls.

Core relation:

$$L_\mathrm{train}\downarrow,\quad L_\mathrm{val}\not\downarrow$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 8.2 Base-task retention

**Main idea.** Evaluate old skills after adaptation.

Core relation:

$$S_\mathrm{retain}(\theta)$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 8.3 Task quality

**Main idea.** Use task-specific automatic and human checks.

Core relation:

$$S_\mathrm{task}(\theta)$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 8.4 Distribution shift

**Main idea.** Fine-tune data should match deployment use.

Core relation:

$$p_\mathrm{train}(x,y)\approx p_\mathrm{deploy}(x,y)$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 8.5 Adapter sanity checks

**Main idea.** Disable the adapter to confirm measured change comes from the adapter.

Core relation:

$$f_{W+\Delta W}(x)-f_W(x)$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
## 9. Choosing a Method

This part treats choosing a method as a measurable adaptation decision. The goal is to know what is updated, what objective is optimized, how much capacity the update has, and how to detect damage to the base model.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [No-update methods](#9-noupdate-methods) | prompting and retrieval are cheapest when they work | $\Delta\theta=0$ |
| [PEFT methods](#9-peft-methods) | adapters and LoRA are the default when cost and deployment flexibility matter | $|\Delta\theta|\ll|\theta|$ |
| [Full tuning](#9-full-tuning) | use when the task requires broad representation movement | $|\Delta\theta|$ can be large |
| [Preference tuning](#9-preference-tuning) | use when the target is comparative behavior rather than exact demonstrations | $y^+\succ y^-$ |
| [Deployment constraints](#9-deployment-constraints) | latency, adapter routing, merging, and safety evaluation are part of the method choice | $\mathrm{quality}/\mathrm{cost}$ is the real metric |

### 9.1 No-update methods

**Main idea.** Prompting and retrieval are cheapest when they work.

Core relation:

$$\Delta\theta=0$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 9.2 PEFT methods

**Main idea.** Adapters and lora are the default when cost and deployment flexibility matter.

Core relation:

$$|\Delta\theta|\ll|\theta|$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 9.3 Full tuning

**Main idea.** Use when the task requires broad representation movement.

Core relation:

$$|\Delta\theta|$ can be large$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 9.4 Preference tuning

**Main idea.** Use when the target is comparative behavior rather than exact demonstrations.

Core relation:

$$y^+\succ y^-$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 9.5 Deployment constraints

**Main idea.** Latency, adapter routing, merging, and safety evaluation are part of the method choice.

Core relation:

$$\mathrm{quality}/\mathrm{cost}$ is the real metric$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
## 10. Implementation Checklist

This part treats implementation checklist as a measurable adaptation decision. The goal is to know what is updated, what objective is optimized, how much capacity the update has, and how to detect damage to the base model.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Masking](#10-masking) | prompt, padding, and answer masks must match the intended objective | $m_j\in\{0,1\}$ |
| [Parameter freeze audit](#10-parameter-freeze-audit) | verify only intended tensors require gradients | $\mathrm{requires\_grad}$ |
| [Learning-rate groups](#10-learningrate-groups) | base weights and adapters need different optimizer groups if both train | $\eta_\mathrm{base}\ne\eta_\mathrm{adapter}$ |
| [Reference model](#10-reference-model) | preference losses require a stable reference distribution | $\pi_\mathrm{ref}$ |
| [Ablations](#10-ablations) | compare base, prompt-only, PEFT, and full-tune when feasible | $\Delta S=S_\mathrm{method}-S_\mathrm{base}$ |

### 10.1 Masking

**Main idea.** Prompt, padding, and answer masks must match the intended objective.

Core relation:

$$m_j\in\{0,1\}$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 10.2 Parameter freeze audit

**Main idea.** Verify only intended tensors require gradients.

Core relation:

$$\mathrm{requires\_grad}$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** A PEFT run with the wrong tensors trainable is just an expensive surprise.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 10.3 Learning-rate groups

**Main idea.** Base weights and adapters need different optimizer groups if both train.

Core relation:

$$\eta_\mathrm{base}\ne\eta_\mathrm{adapter}$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 10.4 Reference model

**Main idea.** Preference losses require a stable reference distribution.

Core relation:

$$\pi_\mathrm{ref}$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 10.5 Ablations

**Main idea.** Compare base, prompt-only, peft, and full-tune when feasible.

Core relation:

$$\Delta S=S_\mathrm{method}-S_\mathrm{base}$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.

---

## Practice Exercises

1. Compute answer-only SFT loss using a binary mask.
2. Compute KL shift between base and tuned next-token distributions.
3. Count LoRA trainable parameters for a projection matrix.
4. Verify the shape of a LoRA update.
5. Approximate a matrix update with truncated SVD.
6. Count adapter bottleneck parameters.
7. Count prefix-tuning parameters for layers and heads.
8. Compute DPO loss for one preference pair.
9. Build a forgetting versus task-quality scorecard.
10. Write a parameter-freeze and masking checklist.

## Why This Matters for AI

Most applied LLM work is not pretraining from scratch. It is adaptation: making a capable base model behave correctly in a domain, workflow, policy regime, or product setting. Fine-tuning math keeps that adaptation honest. It tells you whether you are training the intended tokens, moving the intended parameters, using enough rank, preserving base capability, and measuring the right behavior.

## Bridge to Scaling Laws

The next section studies how loss changes with model size, data size, and compute. Fine-tuning adds another axis: adaptation capacity. A small low-rank update may be enough for format and style, while deeper domain shifts may require more data, rank, layers, or full-model movement.

## References

- Jeremy Howard and Sebastian Ruder, "Universal Language Model Fine-tuning for Text Classification", 2018: https://arxiv.org/abs/1801.06146
- Neil Houlsby et al., "Parameter-Efficient Transfer Learning for NLP", 2019: https://proceedings.mlr.press/v97/houlsby19a.html
- Xiang Lisa Li and Percy Liang, "Prefix-Tuning: Optimizing Continuous Prompts for Generation", 2021: https://arxiv.org/abs/2101.00190
- Edward J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", 2021: https://arxiv.org/abs/2106.09685
- Long Ouyang et al., "Training language models to follow instructions with human feedback", 2022: https://arxiv.org/abs/2203.02155
- Tim Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs", 2023: https://arxiv.org/abs/2305.14314
- Rafael Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", 2023: https://arxiv.org/abs/2305.18290
