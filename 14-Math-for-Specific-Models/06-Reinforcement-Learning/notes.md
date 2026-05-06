[Back to Math for Specific Models](../README.md) | [Previous: Transformer Architecture](../05-Transformer-Architecture/notes.md) | [Next: Generative Models](../07-Generative-Models/notes.md)

---

# Reinforcement Learning

> _"An RL algorithm is a way to turn consequences into better future decisions."_

## Overview

Reinforcement learning studies agents that learn by acting. The data are not a fixed table of labeled examples; the policy changes which states are visited, which rewards are observed, and which mistakes are possible. That single fact is why RL needs Markov decision processes, Bellman equations, temporal-difference learning, exploration, and policy-gradient estimators.

For AI systems, RL matters in two directions. Classical RL explains games, robotics, recommender feedback loops, and online control. Modern LLM work uses the same math when a model is fine-tuned from human preferences, constrained by KL divergence, or evaluated through interactive feedback rather than static labels.

This section is written as LaTeX Markdown. Inline math uses `$...$`, display math uses `$$...$$`, and the notebooks use small synthetic MDPs so every update can be inspected without external data.

## Prerequisites

- [Markov Chains](../../06-Probability-Theory/07-Markov-Chains/notes.md)
- [Expectation and Moments](../../06-Probability-Theory/04-Expectation-and-Moments/notes.md)
- [Gradient Descent](../../08-Optimization/02-Gradient-Descent/notes.md)
- [KL Divergence](../../09-Information-Theory/02-KL-Divergence/notes.md)
- [Transformer Architecture](../05-Transformer-Architecture/notes.md)
- [Preference Optimization, RLHF, and DPO](../../18-Alignment-and-Safety/02-Preference-Optimization-RLHF-and-DPO/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Executable demonstrations of MDPs, Bellman backups, TD learning, Q-learning, policy gradients, PPO, and preference optimization. |
| [exercises.ipynb](exercises.ipynb) | Ten graded practice problems with runnable scaffolds and full checked solutions. |

## Learning Objectives

After completing this section, you will be able to:

- Define a finite Markov decision process and identify its state, action, reward, transition, and discount components.
- Compute discounted returns and explain temporal credit assignment.
- Derive Bellman expectation and optimality equations.
- Run policy evaluation, policy iteration, and value iteration in a tabular MDP.
- Explain Monte Carlo, TD(0), n-step, and eligibility-trace prediction.
- Implement SARSA and Q-learning updates and distinguish on-policy from off-policy control.
- Explain why replay buffers and target networks stabilize DQN-style learning.
- Derive the policy-gradient estimator and explain baseline variance reduction.
- Interpret actor-critic, GAE, PPO clipping, and KL regularization.
- Connect RL math to RLHF, reward modeling, DPO, and reward hacking risk in LLM systems.

---

## Table of Contents

- [1. Intuition and Motivation](#1-intuition-and-motivation)
  - [1.1 Sequential decision making](#11-sequential-decision-making)
  - [1.2 Rewards returns and delayed credit](#12-rewards-returns-and-delayed-credit)
  - [1.3 Exploration versus exploitation](#13-exploration-versus-exploitation)
  - [1.4 RL versus supervised learning](#14-rl-versus-supervised-learning)
  - [1.5 Where RL appears in LLM systems](#15-where-rl-appears-in-llm-systems)
- [2. Formal MDP Setup](#2-formal-mdp-setup)
  - [2.1 States actions rewards and transitions](#21-states-actions-rewards-and-transitions)
  - [2.2 The Markov property](#22-the-markov-property)
  - [2.3 Episodic continuing finite and infinite horizon tasks](#23-episodic-continuing-finite-and-infinite-horizon-tasks)
  - [2.4 Transition kernels and reward functions](#24-transition-kernels-and-reward-functions)
  - [2.5 Partial observability and belief states](#25-partial-observability-and-belief-states)
- [3. Returns Policies and Value Functions](#3-returns-policies-and-value-functions)
  - [3.1 Discounted return](#31-discounted-return)
  - [3.2 Deterministic and stochastic policies](#32-deterministic-and-stochastic-policies)
  - [3.3 State-value and action-value functions](#33-statevalue-and-actionvalue-functions)
  - [3.4 Advantage functions](#34-advantage-functions)
  - [3.5 Occupancy measures](#35-occupancy-measures)
- [4. Bellman Equations](#4-bellman-equations)
  - [4.1 Bellman expectation equation](#41-bellman-expectation-equation)
  - [4.2 Bellman optimality equation](#42-bellman-optimality-equation)
  - [4.3 Contraction mapping intuition](#43-contraction-mapping-intuition)
  - [4.4 Matrix form for finite MDPs](#44-matrix-form-for-finite-mdps)
  - [4.5 Bellman residuals](#45-bellman-residuals)
- [5. Dynamic Programming](#5-dynamic-programming)
  - [5.1 Policy evaluation](#51-policy-evaluation)
  - [5.2 Policy improvement](#52-policy-improvement)
  - [5.3 Policy iteration](#53-policy-iteration)
  - [5.4 Value iteration](#54-value-iteration)
  - [5.5 Planning versus learning](#55-planning-versus-learning)
- [6. Sampling-Based Prediction](#6-samplingbased-prediction)
  - [6.1 Monte Carlo returns](#61-monte-carlo-returns)
  - [6.2 Temporal-difference learning](#62-temporaldifference-learning)
  - [6.3 Bias variance and bootstrapping](#63-bias-variance-and-bootstrapping)
  - [6.4 N-step returns](#64-nstep-returns)
  - [6.5 Eligibility traces and TD lambda](#65-eligibility-traces-and-td-lambda)
- [7. Control Algorithms](#7-control-algorithms)
  - [7.1 SARSA](#71-sarsa)
  - [7.2 Q-learning](#72-qlearning)
  - [7.3 Exploration schedules](#73-exploration-schedules)
  - [7.4 Double Q-learning](#74-double-qlearning)
  - [7.5 Convergence conditions](#75-convergence-conditions)
- [8. Function Approximation and Deep RL](#8-function-approximation-and-deep-rl)
  - [8.1 Why tables fail](#81-why-tables-fail)
  - [8.2 Linear value approximation](#82-linear-value-approximation)
  - [8.3 The deadly triad](#83-the-deadly-triad)
  - [8.4 DQN stabilization](#84-dqn-stabilization)
  - [8.5 Representation learning for policies and critics](#85-representation-learning-for-policies-and-critics)
- [9. Policy Gradients](#9-policy-gradients)
  - [9.1 Policy objective](#91-policy-objective)
  - [9.2 Log-derivative trick](#92-logderivative-trick)
  - [9.3 Policy gradient theorem](#93-policy-gradient-theorem)
  - [9.4 Baselines and variance reduction](#94-baselines-and-variance-reduction)
  - [9.5 Entropy regularization](#95-entropy-regularization)
- [10. Actor-Critic PPO and RLHF](#10-actorcritic-ppo-and-rlhf)
  - [10.1 Actor-critic decomposition](#101-actorcritic-decomposition)
  - [10.2 Generalized advantage estimation](#102-generalized-advantage-estimation)
  - [10.3 Trust regions and KL control](#103-trust-regions-and-kl-control)
  - [10.4 PPO clipped surrogate objective](#104-ppo-clipped-surrogate-objective)
  - [10.5 Reward modeling and preference optimization](#105-reward-modeling-and-preference-optimization)
- [11. Common Mistakes](#11-common-mistakes)
- [12. Exercises](#12-exercises)
- [13. Why This Matters for AI](#13-why-this-matters-for-ai)
- [14. Conceptual Bridge](#14-conceptual-bridge)
- [References](#references)

---

## 1. Intuition and Motivation

Intuition and Motivation is part of the core mathematical path from Markov chains to modern AI agents. The emphasis is on the object definitions and update equations a learner must be able to inspect in code.

### 1.1 Sequential decision making

**Purpose.** Sequential decision making focuses on why actions change future data. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$\mathcal{M}=(\mathcal{S},\mathcal{A},P,r,\gamma),\qquad P(s'\mid s,a)=\Pr(S_{t+1}=s'\mid S_t=s,A_t=a).$$

**Operational definition.**

RL is the mathematics of decisions whose consequences alter later observations. A training example is no longer fixed before learning; the policy helps create the future dataset.

**Worked reading.**

At time $t$, the agent sees $S_t$, chooses $A_t$, receives $R_{t+1}$, and reaches $S_{t+1}$. The learning signal is attached to a trajectory, not a single independent example.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. robot navigation.
2. game play.
3. dialogue policy tuning.

Non-examples:

1. ordinary regression with fixed labels.
2. a bandit problem with no state evolution.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

### 1.2 Rewards returns and delayed credit

**Purpose.** Rewards returns and delayed credit focuses on why scalar feedback is hard to assign. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$G_t=\sum_{k=0}^{\infty}\gamma^kR_{t+k+1},\qquad 0\le \gamma<1.$$

**Operational definition.**

The reward signal is local, but the objective is cumulative. The central difficulty is assigning a future return back to earlier actions.

**Worked reading.**

The discounted return is $G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots$.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. sparse game win/loss rewards.
2. human preference scores for full responses.
3. robot task completion bonuses.

Non-examples:

1. per-token supervised labels.
2. a deterministic lookup table with no delayed consequence.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

### 1.3 Exploration versus exploitation

**Purpose.** Exploration versus exploitation focuses on why the learner must choose data. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$V^\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s],\qquad Q^\pi(s,a)=\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a].$$

**Operational definition.**

The agent must balance actions that look good now with actions that reveal useful information. This makes data collection part of the optimization problem.

**Worked reading.**

An $\epsilon$-greedy policy chooses a greedy action with probability $1-\epsilon$ and explores otherwise.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. optimistic initialization.
2. Boltzmann exploration.
3. UCB-style uncertainty bonuses.

Non-examples:

1. evaluating one fixed logged policy only.
2. training on a static supervised corpus.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

### 1.4 RL versus supervised learning

**Purpose.** RL versus supervised learning focuses on why labels are not enough. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$V^\pi(s)=\sum_a\pi(a\mid s)\sum_{s'}P(s'\mid s,a)\left(r(s,a,s')+\gamma V^\pi(s')\right).$$

**Operational definition.**

Supervised learning assumes labeled targets. RL instead observes consequences from an interactive process and must handle shifting state-action distributions.

**Worked reading.**

A policy update changes $d^\pi(s)$, the visitation distribution, so future data are policy-dependent.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. offline RL from logged data.
2. online policy improvement.
3. preference-tuned language models.

Non-examples:

1. i.i.d. image classification.
2. least-squares regression with fixed design.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

### 1.5 Where RL appears in LLM systems

**Purpose.** Where RL appears in LLM systems focuses on why preference optimization uses this math. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$Q^*(s,a)=\sum_{s'}P(s'\mid s,a)\left(r(s,a,s')+\gamma\max_{a'}Q^*(s',a')\right).$$

**Operational definition.**

RL enters LLM systems through preference learning, reward modeling, KL-regularized policy updates, and evaluation policies that adapt from feedback.

**Worked reading.**

RLHF typically optimizes a reward model while penalizing divergence from a reference model with $D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}})$.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. PPO-style RLHF.
2. DPO-style preference optimization.
3. bandit feedback for ranking.

Non-examples:

1. plain next-token pretraining.
2. static instruction tuning without preference feedback.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

## 2. Formal MDP Setup

Formal MDP Setup is part of the core mathematical path from Markov chains to modern AI agents. The emphasis is on the object definitions and update equations a learner must be able to inspect in code.

### 2.1 States actions rewards and transitions

**Purpose.** States actions rewards and transitions focuses on the tuple $(\mathcal{S},\mathcal{A},P,r,\gamma)$. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$G_t=\sum_{k=0}^{\infty}\gamma^kR_{t+k+1},\qquad 0\le \gamma<1.$$

**Operational definition.**

An MDP is the formal object that makes sequential decision-making mathematically precise.

**Worked reading.**

The Markov property says the current state contains the predictive information needed for the next transition and reward.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. finite gridworld.
2. inventory control.
3. dialogue state tracking.

Non-examples:

1. raw observations that omit hidden state.
2. a static labeled dataset with no actions.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

### 2.2 The Markov property

**Purpose.** The Markov property focuses on why the present state summarizes the useful past. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$V^\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s],\qquad Q^\pi(s,a)=\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a].$$

**Operational definition.**

An MDP is the formal object that makes sequential decision-making mathematically precise.

**Worked reading.**

The Markov property says the current state contains the predictive information needed for the next transition and reward.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. finite gridworld.
2. inventory control.
3. dialogue state tracking.

Non-examples:

1. raw observations that omit hidden state.
2. a static labeled dataset with no actions.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

### 2.3 Episodic continuing finite and infinite horizon tasks

**Purpose.** Episodic continuing finite and infinite horizon tasks focuses on how time changes the objective. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$V^\pi(s)=\sum_a\pi(a\mid s)\sum_{s'}P(s'\mid s,a)\left(r(s,a,s')+\gamma V^\pi(s')\right).$$

**Operational definition.**

This concept is part of the bridge from sequential probability models to practical RL algorithms.

**Worked reading.**

The key habit is to name the state, action, reward, transition, policy, and value object before writing an update.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. small tabular MDPs.
2. neural value functions.
3. preference-optimized language policies.

Non-examples:

1. static regression.
2. uncontrolled simulation traces.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

### 2.4 Transition kernels and reward functions

**Purpose.** Transition kernels and reward functions focuses on how stochastic environments are represented. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$Q^*(s,a)=\sum_{s'}P(s'\mid s,a)\left(r(s,a,s')+\gamma\max_{a'}Q^*(s',a')\right).$$

**Operational definition.**

Function approximation replaces tables with parameterized models so the agent can generalize across large state spaces.

**Worked reading.**

Deep RL is powerful because neural networks share statistical strength, but unstable because approximate bootstrapping can amplify errors.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. DQN target networks.
2. experience replay.
3. critic networks.

Non-examples:

1. exact dynamic programming in a tiny known MDP.
2. memorizing every state-action value in a table.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

### 2.5 Partial observability and belief states

**Purpose.** Partial observability and belief states focuses on what breaks when observations are not states. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$\nabla_\theta J(\theta)=\mathbb{E}_{\pi_\theta}\left[\nabla_\theta\log\pi_\theta(A_t\mid S_t)A^{\pi_\theta}(S_t,A_t)\right].$$

**Operational definition.**

An MDP is the formal object that makes sequential decision-making mathematically precise.

**Worked reading.**

The Markov property says the current state contains the predictive information needed for the next transition and reward.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. finite gridworld.
2. inventory control.
3. dialogue state tracking.

Non-examples:

1. raw observations that omit hidden state.
2. a static labeled dataset with no actions.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

## 3. Returns Policies and Value Functions

Returns Policies and Value Functions is part of the core mathematical path from Markov chains to modern AI agents. The emphasis is on the object definitions and update equations a learner must be able to inspect in code.

### 3.1 Discounted return

**Purpose.** Discounted return focuses on why $G_t=\sum_{k\ge 0}\gamma^k R_{t+k+1}$ is central. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$V^\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s],\qquad Q^\pi(s,a)=\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a].$$

**Operational definition.**

Value functions summarize future consequences. Advantage functions compare an action with the policy's average behavior at the same state.

**Worked reading.**

Occupancy measures explain why RL gradients weight states by how often the current policy visits them.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. $V^\pi$.
2. $Q^\pi$.
3. $A^\pi$.

Non-examples:

1. instant reward only.
2. a metric computed on states never reached by the policy.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

### 3.2 Deterministic and stochastic policies

**Purpose.** Deterministic and stochastic policies focuses on how $\pi(a\mid s)$ controls the data distribution. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$V^\pi(s)=\sum_a\pi(a\mid s)\sum_{s'}P(s'\mid s,a)\left(r(s,a,s')+\gamma V^\pi(s')\right).$$

**Operational definition.**

This concept is part of the bridge from sequential probability models to practical RL algorithms.

**Worked reading.**

The key habit is to name the state, action, reward, transition, policy, and value object before writing an update.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. small tabular MDPs.
2. neural value functions.
3. preference-optimized language policies.

Non-examples:

1. static regression.
2. uncontrolled simulation traces.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

### 3.3 State-value and action-value functions

**Purpose.** State-value and action-value functions focuses on what $V^\pi$ and $Q^\pi$ estimate. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$Q^*(s,a)=\sum_{s'}P(s'\mid s,a)\left(r(s,a,s')+\gamma\max_{a'}Q^*(s',a')\right).$$

**Operational definition.**

Function approximation replaces tables with parameterized models so the agent can generalize across large state spaces.

**Worked reading.**

Deep RL is powerful because neural networks share statistical strength, but unstable because approximate bootstrapping can amplify errors.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. DQN target networks.
2. experience replay.
3. critic networks.

Non-examples:

1. exact dynamic programming in a tiny known MDP.
2. memorizing every state-action value in a table.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

### 3.4 Advantage functions

**Purpose.** Advantage functions focuses on why $A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$ reduces variance. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$\nabla_\theta J(\theta)=\mathbb{E}_{\pi_\theta}\left[\nabla_\theta\log\pi_\theta(A_t\mid S_t)A^{\pi_\theta}(S_t,A_t)\right].$$

**Operational definition.**

Function approximation replaces tables with parameterized models so the agent can generalize across large state spaces.

**Worked reading.**

Deep RL is powerful because neural networks share statistical strength, but unstable because approximate bootstrapping can amplify errors.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. DQN target networks.
2. experience replay.
3. critic networks.

Non-examples:

1. exact dynamic programming in a tiny known MDP.
2. memorizing every state-action value in a table.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

### 3.5 Occupancy measures

**Purpose.** Occupancy measures focuses on how policies induce weighted state-action distributions. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$L^{\mathrm{CLIP}}(\theta)=\mathbb{E}\left[\min(r_t(\theta)\hat{A}_t,\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t)\right].$$

**Operational definition.**

Value functions summarize future consequences. Advantage functions compare an action with the policy's average behavior at the same state.

**Worked reading.**

Occupancy measures explain why RL gradients weight states by how often the current policy visits them.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. $V^\pi$.
2. $Q^\pi$.
3. $A^\pi$.

Non-examples:

1. instant reward only.
2. a metric computed on states never reached by the policy.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

## 4. Bellman Equations

Bellman Equations is part of the core mathematical path from Markov chains to modern AI agents. The emphasis is on the object definitions and update equations a learner must be able to inspect in code.

### 4.1 Bellman expectation equation

**Purpose.** Bellman expectation equation focuses on recursive consistency for a fixed policy. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$V^\pi(s)=\sum_a\pi(a\mid s)\sum_{s'}P(s'\mid s,a)\left(r(s,a,s')+\gamma V^\pi(s')\right).$$

**Operational definition.**

Bellman equations express a global return as immediate reward plus the value of the next state. They are recursive consistency equations.

**Worked reading.**

The Bellman backup replaces a value estimate by a reward-plus-next-value target under either a fixed policy or an optimal action.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. policy evaluation.
2. value iteration.
3. TD target construction.

Non-examples:

1. a one-step supervised label.
2. a loss that ignores future value.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

### 4.2 Bellman optimality equation

**Purpose.** Bellman optimality equation focuses on recursive consistency for the best policy. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$Q^*(s,a)=\sum_{s'}P(s'\mid s,a)\left(r(s,a,s')+\gamma\max_{a'}Q^*(s',a')\right).$$

**Operational definition.**

Bellman equations express a global return as immediate reward plus the value of the next state. They are recursive consistency equations.

**Worked reading.**

The Bellman backup replaces a value estimate by a reward-plus-next-value target under either a fixed policy or an optimal action.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. policy evaluation.
2. value iteration.
3. TD target construction.

Non-examples:

1. a one-step supervised label.
2. a loss that ignores future value.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

### 4.3 Contraction mapping intuition

**Purpose.** Contraction mapping intuition focuses on why dynamic programming converges when $\gamma<1$. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$\nabla_\theta J(\theta)=\mathbb{E}_{\pi_\theta}\left[\nabla_\theta\log\pi_\theta(A_t\mid S_t)A^{\pi_\theta}(S_t,A_t)\right].$$

**Operational definition.**

This concept is part of the bridge from sequential probability models to practical RL algorithms.

**Worked reading.**

The key habit is to name the state, action, reward, transition, policy, and value object before writing an update.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. small tabular MDPs.
2. neural value functions.
3. preference-optimized language policies.

Non-examples:

1. static regression.
2. uncontrolled simulation traces.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

### 4.4 Matrix form for finite MDPs

**Purpose.** Matrix form for finite MDPs focuses on how policy evaluation becomes a linear system. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$L^{\mathrm{CLIP}}(\theta)=\mathbb{E}\left[\min(r_t(\theta)\hat{A}_t,\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t)\right].$$

**Operational definition.**

An MDP is the formal object that makes sequential decision-making mathematically precise.

**Worked reading.**

The Markov property says the current state contains the predictive information needed for the next transition and reward.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. finite gridworld.
2. inventory control.
3. dialogue state tracking.

Non-examples:

1. raw observations that omit hidden state.
2. a static labeled dataset with no actions.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

### 4.5 Bellman residuals

**Purpose.** Bellman residuals focuses on how to diagnose approximate value functions. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$\mathcal{M}=(\mathcal{S},\mathcal{A},P,r,\gamma),\qquad P(s'\mid s,a)=\Pr(S_{t+1}=s'\mid S_t=s,A_t=a).$$

**Operational definition.**

Bellman equations express a global return as immediate reward plus the value of the next state. They are recursive consistency equations.

**Worked reading.**

The Bellman backup replaces a value estimate by a reward-plus-next-value target under either a fixed policy or an optimal action.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. policy evaluation.
2. value iteration.
3. TD target construction.

Non-examples:

1. a one-step supervised label.
2. a loss that ignores future value.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

## 5. Dynamic Programming

Dynamic Programming is part of the core mathematical path from Markov chains to modern AI agents. The emphasis is on the object definitions and update equations a learner must be able to inspect in code.

### 5.1 Policy evaluation

**Purpose.** Policy evaluation focuses on computing $V^\pi$ when the model is known. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$Q^*(s,a)=\sum_{s'}P(s'\mid s,a)\left(r(s,a,s')+\gamma\max_{a'}Q^*(s',a')\right).$$

**Operational definition.**

Policy methods optimize the action distribution directly. This is essential when actions are continuous, structured, or generated token by token.

**Worked reading.**

The score-function identity converts a derivative of an expectation into an expectation of $\nabla_\theta\log\pi_\theta(a\mid s)$ times a return-like signal.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. REINFORCE.
2. actor-critic.
3. PPO for RLHF.

Non-examples:

1. choosing $\arg\max_a Q(s,a)$ from a tiny action table.
2. behavior cloning without reward feedback.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

### 5.2 Policy improvement

**Purpose.** Policy improvement focuses on turning a value function into a better greedy policy. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$\nabla_\theta J(\theta)=\mathbb{E}_{\pi_\theta}\left[\nabla_\theta\log\pi_\theta(A_t\mid S_t)A^{\pi_\theta}(S_t,A_t)\right].$$

**Operational definition.**

Policy methods optimize the action distribution directly. This is essential when actions are continuous, structured, or generated token by token.

**Worked reading.**

The score-function identity converts a derivative of an expectation into an expectation of $\nabla_\theta\log\pi_\theta(a\mid s)$ times a return-like signal.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. REINFORCE.
2. actor-critic.
3. PPO for RLHF.

Non-examples:

1. choosing $\arg\max_a Q(s,a)$ from a tiny action table.
2. behavior cloning without reward feedback.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

### 5.3 Policy iteration

**Purpose.** Policy iteration focuses on alternating evaluation and improvement. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$L^{\mathrm{CLIP}}(\theta)=\mathbb{E}\left[\min(r_t(\theta)\hat{A}_t,\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t)\right].$$

**Operational definition.**

Policy methods optimize the action distribution directly. This is essential when actions are continuous, structured, or generated token by token.

**Worked reading.**

The score-function identity converts a derivative of an expectation into an expectation of $\nabla_\theta\log\pi_\theta(a\mid s)$ times a return-like signal.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. REINFORCE.
2. actor-critic.
3. PPO for RLHF.

Non-examples:

1. choosing $\arg\max_a Q(s,a)$ from a tiny action table.
2. behavior cloning without reward feedback.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

### 5.4 Value iteration

**Purpose.** Value iteration focuses on combining backup and improvement into one operator. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$\mathcal{M}=(\mathcal{S},\mathcal{A},P,r,\gamma),\qquad P(s'\mid s,a)=\Pr(S_{t+1}=s'\mid S_t=s,A_t=a).$$

**Operational definition.**

Value functions summarize future consequences. Advantage functions compare an action with the policy's average behavior at the same state.

**Worked reading.**

Occupancy measures explain why RL gradients weight states by how often the current policy visits them.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. $V^\pi$.
2. $Q^\pi$.
3. $A^\pi$.

Non-examples:

1. instant reward only.
2. a metric computed on states never reached by the policy.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

### 5.5 Planning versus learning

**Purpose.** Planning versus learning focuses on why model access changes the algorithm. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$G_t=\sum_{k=0}^{\infty}\gamma^kR_{t+k+1},\qquad 0\le \gamma<1.$$

**Operational definition.**

Dynamic programming uses a known model to compute values and policies through Bellman backups.

**Worked reading.**

Policy iteration alternates evaluation and greedy improvement; value iteration applies optimal backups directly.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. policy evaluation.
2. policy iteration.
3. value iteration.

Non-examples:

1. model-free Q-learning.
2. one-step supervised classification.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

## 6. Sampling-Based Prediction

Sampling-Based Prediction is part of the core mathematical path from Markov chains to modern AI agents. The emphasis is on the object definitions and update equations a learner must be able to inspect in code.

### 6.1 Monte Carlo returns

**Purpose.** Monte Carlo returns focuses on learning from complete sampled episodes. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$\nabla_\theta J(\theta)=\mathbb{E}_{\pi_\theta}\left[\nabla_\theta\log\pi_\theta(A_t\mid S_t)A^{\pi_\theta}(S_t,A_t)\right].$$

**Operational definition.**

Sampling-based prediction learns values from trajectories. Monte Carlo waits for full returns; TD bootstraps from the next estimate.

**Worked reading.**

The TD error $\delta_t=R_{t+1}+\gamma V(S_{t+1})-V(S_t)$ is a local surprise signal.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. TD(0).
2. n-step returns.
3. TD(lambda).

Non-examples:

1. solving the Bellman linear system exactly.
2. using labels independent of the current policy.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

### 6.2 Temporal-difference learning

**Purpose.** Temporal-difference learning focuses on learning from bootstrapped one-step targets. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$L^{\mathrm{CLIP}}(\theta)=\mathbb{E}\left[\min(r_t(\theta)\hat{A}_t,\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t)\right].$$

**Operational definition.**

Sampling-based prediction learns values from trajectories. Monte Carlo waits for full returns; TD bootstraps from the next estimate.

**Worked reading.**

The TD error $\delta_t=R_{t+1}+\gamma V(S_{t+1})-V(S_t)$ is a local surprise signal.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. TD(0).
2. n-step returns.
3. TD(lambda).

Non-examples:

1. solving the Bellman linear system exactly.
2. using labels independent of the current policy.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

### 6.3 Bias variance and bootstrapping

**Purpose.** Bias variance and bootstrapping focuses on why MC and TD make different errors. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$\mathcal{M}=(\mathcal{S},\mathcal{A},P,r,\gamma),\qquad P(s'\mid s,a)=\Pr(S_{t+1}=s'\mid S_t=s,A_t=a).$$

**Operational definition.**

This concept is part of the bridge from sequential probability models to practical RL algorithms.

**Worked reading.**

The key habit is to name the state, action, reward, transition, policy, and value object before writing an update.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. small tabular MDPs.
2. neural value functions.
3. preference-optimized language policies.

Non-examples:

1. static regression.
2. uncontrolled simulation traces.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

### 6.4 N-step returns

**Purpose.** N-step returns focuses on interpolating between TD(0) and Monte Carlo. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$G_t=\sum_{k=0}^{\infty}\gamma^kR_{t+k+1},\qquad 0\le \gamma<1.$$

**Operational definition.**

Value functions summarize future consequences. Advantage functions compare an action with the policy's average behavior at the same state.

**Worked reading.**

Occupancy measures explain why RL gradients weight states by how often the current policy visits them.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. $V^\pi$.
2. $Q^\pi$.
3. $A^\pi$.

Non-examples:

1. instant reward only.
2. a metric computed on states never reached by the policy.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

### 6.5 Eligibility traces and TD lambda

**Purpose.** Eligibility traces and TD lambda focuses on credit assignment across recent visits. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$V^\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s],\qquad Q^\pi(s,a)=\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a].$$

**Operational definition.**

Sampling-based prediction learns values from trajectories. Monte Carlo waits for full returns; TD bootstraps from the next estimate.

**Worked reading.**

The TD error $\delta_t=R_{t+1}+\gamma V(S_{t+1})-V(S_t)$ is a local surprise signal.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. TD(0).
2. n-step returns.
3. TD(lambda).

Non-examples:

1. solving the Bellman linear system exactly.
2. using labels independent of the current policy.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

## 7. Control Algorithms

Control Algorithms is part of the core mathematical path from Markov chains to modern AI agents. The emphasis is on the object definitions and update equations a learner must be able to inspect in code.

### 7.1 SARSA

**Purpose.** SARSA focuses on on-policy TD control. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$L^{\mathrm{CLIP}}(\theta)=\mathbb{E}\left[\min(r_t(\theta)\hat{A}_t,\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t)\right].$$

**Operational definition.**

Control algorithms learn how to choose actions, not just how to evaluate a fixed policy.

**Worked reading.**

SARSA uses the action actually sampled by the behavior policy; Q-learning uses a greedy target and is therefore off-policy.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. tabular gridworld control.
2. DQN-style value learning.
3. epsilon-greedy exploration.

Non-examples:

1. estimating $V^\pi$ only.
2. planning with a perfect model and no samples.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

### 7.2 Q-learning

**Purpose.** Q-learning focuses on off-policy TD control with greedy targets. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$\mathcal{M}=(\mathcal{S},\mathcal{A},P,r,\gamma),\qquad P(s'\mid s,a)=\Pr(S_{t+1}=s'\mid S_t=s,A_t=a).$$

**Operational definition.**

Control algorithms learn how to choose actions, not just how to evaluate a fixed policy.

**Worked reading.**

SARSA uses the action actually sampled by the behavior policy; Q-learning uses a greedy target and is therefore off-policy.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. tabular gridworld control.
2. DQN-style value learning.
3. epsilon-greedy exploration.

Non-examples:

1. estimating $V^\pi$ only.
2. planning with a perfect model and no samples.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

### 7.3 Exploration schedules

**Purpose.** Exploration schedules focuses on epsilon greedy softmax and optimism. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$G_t=\sum_{k=0}^{\infty}\gamma^kR_{t+k+1},\qquad 0\le \gamma<1.$$

**Operational definition.**

This concept is part of the bridge from sequential probability models to practical RL algorithms.

**Worked reading.**

The key habit is to name the state, action, reward, transition, policy, and value object before writing an update.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. small tabular MDPs.
2. neural value functions.
3. preference-optimized language policies.

Non-examples:

1. static regression.
2. uncontrolled simulation traces.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

### 7.4 Double Q-learning

**Purpose.** Double Q-learning focuses on reducing maximization bias. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$V^\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s],\qquad Q^\pi(s,a)=\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a].$$

**Operational definition.**

Control algorithms learn how to choose actions, not just how to evaluate a fixed policy.

**Worked reading.**

SARSA uses the action actually sampled by the behavior policy; Q-learning uses a greedy target and is therefore off-policy.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. tabular gridworld control.
2. DQN-style value learning.
3. epsilon-greedy exploration.

Non-examples:

1. estimating $V^\pi$ only.
2. planning with a perfect model and no samples.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

### 7.5 Convergence conditions

**Purpose.** Convergence conditions focuses on what tabular guarantees assume. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$V^\pi(s)=\sum_a\pi(a\mid s)\sum_{s'}P(s'\mid s,a)\left(r(s,a,s')+\gamma V^\pi(s')\right).$$

**Operational definition.**

This concept is part of the bridge from sequential probability models to practical RL algorithms.

**Worked reading.**

The key habit is to name the state, action, reward, transition, policy, and value object before writing an update.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. small tabular MDPs.
2. neural value functions.
3. preference-optimized language policies.

Non-examples:

1. static regression.
2. uncontrolled simulation traces.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

## 8. Function Approximation and Deep RL

Function Approximation and Deep RL is part of the core mathematical path from Markov chains to modern AI agents. The emphasis is on the object definitions and update equations a learner must be able to inspect in code.

### 8.1 Why tables fail

**Purpose.** Why tables fail focuses on large state spaces and generalization. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$\mathcal{M}=(\mathcal{S},\mathcal{A},P,r,\gamma),\qquad P(s'\mid s,a)=\Pr(S_{t+1}=s'\mid S_t=s,A_t=a).$$

**Operational definition.**

This concept is part of the bridge from sequential probability models to practical RL algorithms.

**Worked reading.**

The key habit is to name the state, action, reward, transition, policy, and value object before writing an update.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. small tabular MDPs.
2. neural value functions.
3. preference-optimized language policies.

Non-examples:

1. static regression.
2. uncontrolled simulation traces.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

### 8.2 Linear value approximation

**Purpose.** Linear value approximation focuses on projected Bellman equations. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$G_t=\sum_{k=0}^{\infty}\gamma^kR_{t+k+1},\qquad 0\le \gamma<1.$$

**Operational definition.**

Value functions summarize future consequences. Advantage functions compare an action with the policy's average behavior at the same state.

**Worked reading.**

Occupancy measures explain why RL gradients weight states by how often the current policy visits them.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. $V^\pi$.
2. $Q^\pi$.
3. $A^\pi$.

Non-examples:

1. instant reward only.
2. a metric computed on states never reached by the policy.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

### 8.3 The deadly triad

**Purpose.** The deadly triad focuses on bootstrapping off-policy learning and approximation. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$V^\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s],\qquad Q^\pi(s,a)=\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a].$$

**Operational definition.**

This concept is part of the bridge from sequential probability models to practical RL algorithms.

**Worked reading.**

The key habit is to name the state, action, reward, transition, policy, and value object before writing an update.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. small tabular MDPs.
2. neural value functions.
3. preference-optimized language policies.

Non-examples:

1. static regression.
2. uncontrolled simulation traces.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

### 8.4 DQN stabilization

**Purpose.** DQN stabilization focuses on replay buffers and target networks. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$V^\pi(s)=\sum_a\pi(a\mid s)\sum_{s'}P(s'\mid s,a)\left(r(s,a,s')+\gamma V^\pi(s')\right).$$

**Operational definition.**

Function approximation replaces tables with parameterized models so the agent can generalize across large state spaces.

**Worked reading.**

Deep RL is powerful because neural networks share statistical strength, but unstable because approximate bootstrapping can amplify errors.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. DQN target networks.
2. experience replay.
3. critic networks.

Non-examples:

1. exact dynamic programming in a tiny known MDP.
2. memorizing every state-action value in a table.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

### 8.5 Representation learning for policies and critics

**Purpose.** Representation learning for policies and critics focuses on how neural networks change the math. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$Q^*(s,a)=\sum_{s'}P(s'\mid s,a)\left(r(s,a,s')+\gamma\max_{a'}Q^*(s',a')\right).$$

**Operational definition.**

This concept is part of the bridge from sequential probability models to practical RL algorithms.

**Worked reading.**

The key habit is to name the state, action, reward, transition, policy, and value object before writing an update.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. small tabular MDPs.
2. neural value functions.
3. preference-optimized language policies.

Non-examples:

1. static regression.
2. uncontrolled simulation traces.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

## 9. Policy Gradients

Policy Gradients is part of the core mathematical path from Markov chains to modern AI agents. The emphasis is on the object definitions and update equations a learner must be able to inspect in code.

### 9.1 Policy objective

**Purpose.** Policy objective focuses on maximizing $J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[G_0]$. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$G_t=\sum_{k=0}^{\infty}\gamma^kR_{t+k+1},\qquad 0\le \gamma<1.$$

**Operational definition.**

Policy methods optimize the action distribution directly. This is essential when actions are continuous, structured, or generated token by token.

**Worked reading.**

The score-function identity converts a derivative of an expectation into an expectation of $\nabla_\theta\log\pi_\theta(a\mid s)$ times a return-like signal.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. REINFORCE.
2. actor-critic.
3. PPO for RLHF.

Non-examples:

1. choosing $\arg\max_a Q(s,a)$ from a tiny action table.
2. behavior cloning without reward feedback.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

### 9.2 Log-derivative trick

**Purpose.** Log-derivative trick focuses on turning trajectory probabilities into score functions. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$V^\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s],\qquad Q^\pi(s,a)=\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a].$$

**Operational definition.**

This concept is part of the bridge from sequential probability models to practical RL algorithms.

**Worked reading.**

The key habit is to name the state, action, reward, transition, policy, and value object before writing an update.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. small tabular MDPs.
2. neural value functions.
3. preference-optimized language policies.

Non-examples:

1. static regression.
2. uncontrolled simulation traces.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

### 9.3 Policy gradient theorem

**Purpose.** Policy gradient theorem focuses on why gradients can use $Q^\pi(s,a)$. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$V^\pi(s)=\sum_a\pi(a\mid s)\sum_{s'}P(s'\mid s,a)\left(r(s,a,s')+\gamma V^\pi(s')\right).$$

**Operational definition.**

Policy methods optimize the action distribution directly. This is essential when actions are continuous, structured, or generated token by token.

**Worked reading.**

The score-function identity converts a derivative of an expectation into an expectation of $\nabla_\theta\log\pi_\theta(a\mid s)$ times a return-like signal.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. REINFORCE.
2. actor-critic.
3. PPO for RLHF.

Non-examples:

1. choosing $\arg\max_a Q(s,a)$ from a tiny action table.
2. behavior cloning without reward feedback.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

### 9.4 Baselines and variance reduction

**Purpose.** Baselines and variance reduction focuses on why subtracting $b(s)$ is unbiased. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$Q^*(s,a)=\sum_{s'}P(s'\mid s,a)\left(r(s,a,s')+\gamma\max_{a'}Q^*(s',a')\right).$$

**Operational definition.**

This concept is part of the bridge from sequential probability models to practical RL algorithms.

**Worked reading.**

The key habit is to name the state, action, reward, transition, policy, and value object before writing an update.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. small tabular MDPs.
2. neural value functions.
3. preference-optimized language policies.

Non-examples:

1. static regression.
2. uncontrolled simulation traces.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

### 9.5 Entropy regularization

**Purpose.** Entropy regularization focuses on why stochastic policies are encouraged. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$\nabla_\theta J(\theta)=\mathbb{E}_{\pi_\theta}\left[\nabla_\theta\log\pi_\theta(A_t\mid S_t)A^{\pi_\theta}(S_t,A_t)\right].$$

**Operational definition.**

This concept is part of the bridge from sequential probability models to practical RL algorithms.

**Worked reading.**

The key habit is to name the state, action, reward, transition, policy, and value object before writing an update.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. small tabular MDPs.
2. neural value functions.
3. preference-optimized language policies.

Non-examples:

1. static regression.
2. uncontrolled simulation traces.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

## 10. Actor-Critic PPO and RLHF

Actor-Critic PPO and RLHF is part of the core mathematical path from Markov chains to modern AI agents. The emphasis is on the object definitions and update equations a learner must be able to inspect in code.

### 10.1 Actor-critic decomposition

**Purpose.** Actor-critic decomposition focuses on policy and value learning together. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$V^\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s],\qquad Q^\pi(s,a)=\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a].$$

**Operational definition.**

Policy methods optimize the action distribution directly. This is essential when actions are continuous, structured, or generated token by token.

**Worked reading.**

The score-function identity converts a derivative of an expectation into an expectation of $\nabla_\theta\log\pi_\theta(a\mid s)$ times a return-like signal.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. REINFORCE.
2. actor-critic.
3. PPO for RLHF.

Non-examples:

1. choosing $\arg\max_a Q(s,a)$ from a tiny action table.
2. behavior cloning without reward feedback.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

### 10.2 Generalized advantage estimation

**Purpose.** Generalized advantage estimation focuses on the $\lambda$ tradeoff for advantages. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$V^\pi(s)=\sum_a\pi(a\mid s)\sum_{s'}P(s'\mid s,a)\left(r(s,a,s')+\gamma V^\pi(s')\right).$$

**Operational definition.**

Value functions summarize future consequences. Advantage functions compare an action with the policy's average behavior at the same state.

**Worked reading.**

Occupancy measures explain why RL gradients weight states by how often the current policy visits them.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. $V^\pi$.
2. $Q^\pi$.
3. $A^\pi$.

Non-examples:

1. instant reward only.
2. a metric computed on states never reached by the policy.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

### 10.3 Trust regions and KL control

**Purpose.** Trust regions and KL control focuses on limiting policy movement. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$Q^*(s,a)=\sum_{s'}P(s'\mid s,a)\left(r(s,a,s')+\gamma\max_{a'}Q^*(s',a')\right).$$

**Operational definition.**

Control algorithms learn how to choose actions, not just how to evaluate a fixed policy.

**Worked reading.**

SARSA uses the action actually sampled by the behavior policy; Q-learning uses a greedy target and is therefore off-policy.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. tabular gridworld control.
2. DQN-style value learning.
3. epsilon-greedy exploration.

Non-examples:

1. estimating $V^\pi$ only.
2. planning with a perfect model and no samples.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

### 10.4 PPO clipped surrogate objective

**Purpose.** PPO clipped surrogate objective focuses on a practical trust-region approximation. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$\nabla_\theta J(\theta)=\mathbb{E}_{\pi_\theta}\left[\nabla_\theta\log\pi_\theta(A_t\mid S_t)A^{\pi_\theta}(S_t,A_t)\right].$$

**Operational definition.**

Policy methods optimize the action distribution directly. This is essential when actions are continuous, structured, or generated token by token.

**Worked reading.**

The score-function identity converts a derivative of an expectation into an expectation of $\nabla_\theta\log\pi_\theta(a\mid s)$ times a return-like signal.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. REINFORCE.
2. actor-critic.
3. PPO for RLHF.

Non-examples:

1. choosing $\arg\max_a Q(s,a)$ from a tiny action table.
2. behavior cloning without reward feedback.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

### 10.5 Reward modeling and preference optimization

**Purpose.** Reward modeling and preference optimization focuses on the RLHF bridge to language models. This is not optional vocabulary: it determines which variables are random, which distribution creates the data, and which recursive target is valid.

$$L^{\mathrm{CLIP}}(\theta)=\mathbb{E}\left[\min(r_t(\theta)\hat{A}_t,\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t)\right].$$

**Operational definition.**

This concept is part of the bridge from sequential probability models to practical RL algorithms.

**Worked reading.**

The key habit is to name the state, action, reward, transition, policy, and value object before writing an update.

| Object | Mathematical role | ML interpretation |
| --- | --- | --- |
| $\mathcal{S}$ | state space | task context, simulator state, dialogue state |
| $\mathcal{A}$ | action space | moves, controls, generated tokens, tool choices |
| $P(s'\mid s,a)$ | transition kernel | environment dynamics or next-context distribution |
| $r(s,a,s')$ | reward function | scalar training signal, preference score, task score |
| $\pi(a\mid s)$ | policy | behavior rule or neural action distribution |
| $V^\pi,Q^\pi$ | value functions | estimates of future performance |

Examples:

1. small tabular MDPs.
2. neural value functions.
3. preference-optimized language policies.

Non-examples:

1. static regression.
2. uncontrolled simulation traces.

**Derivation habit.**

1. State whether the policy is fixed, greedy-improved, or being optimized.
2. Write the return or Bellman target before writing code.
3. Identify whether the target is sampled, model-based, bootstrapped, or exact.
4. Check whether the data are on-policy or off-policy.
5. Mask terminal states so that value is not propagated beyond the episode.

**Implementation lens.**

In a small tabular MDP, every object can be printed: transition matrices, reward vectors, value arrays, and greedy actions. In a deep RL system, the same objects exist but are represented by neural networks, replay buffers, sampled rollouts, and approximate losses. The names should not change just because the implementation becomes larger.

For LLM work, the state is usually a prompt plus generated prefix, the action is the next token or structured action, and the policy is an autoregressive distribution. The reward may arrive only after a full response, which means the return is sequence-level even though actions are token-level.

Useful checks:

- Does the transition or sample contain the next state?
- Is the update using $V$, $Q$, an advantage, or a reward-only signal?
- Is the current policy also the data-collection policy?
- Does discounting express time preference or just numerical convenience?
- Is the reward model being optimized beyond its trustworthy region?

A common failure mode is to copy an update rule while silently changing the sampling assumption. For example, SARSA and Q-learning can look nearly identical in code, but one updates from the action actually taken and the other updates from a greedy next action. That distinction changes both the mathematics and the behavior.

The notebook version of this idea keeps the environment small enough to inspect by hand. If a concept is unclear, compute it first in the chain MDP and only then move to neural approximators.

## 11. Common Mistakes

| # | Mistake | Why it is wrong | Fix |
| --- | --- | --- | --- |
| 1 | Confusing rewards with returns | A reward is local; a return is accumulated over time. | Always write $G_t$ before deriving an update target. |
| 2 | Ignoring the data distribution shift | Changing the policy changes which states are visited. | Name whether the data are on-policy or off-policy. |
| 3 | Treating Bellman equations as supervised labels | Bellman targets contain estimates and bootstrapping. | Track target networks, stop-gradient choices, or tabular guarantees. |
| 4 | Using Q-learning for every problem | Continuous action spaces and large state spaces often need different methods. | Choose value-based, policy-gradient, or actor-critic methods from the action and data structure. |
| 5 | Forgetting exploration | A greedy policy may never see better actions. | Use explicit exploration or uncertainty-aware data collection. |
| 6 | Trusting average reward without variance | RL estimates are noisy and seed-sensitive. | Report confidence intervals, seeds, and learning curves. |
| 7 | Mixing offline and online assumptions | Logged data may not cover actions needed by the learned policy. | Check coverage and use conservative offline RL when needed. |
| 8 | Over-optimizing a reward model | The policy may exploit reward model errors. | Use KL control, held-out preference evaluation, and adversarial tests. |
| 9 | Calling PPO a magic stabilizer | PPO still depends on advantage quality, clipping, normalization, and KL monitoring. | Audit ratios, advantages, entropy, KL, and value loss. |
| 10 | Forgetting terminal states | Bootstrapping through terminal states creates false future value. | Mask terminal transitions in TD targets. |

## 12. Exercises

1. (*) Solve a Bellman policy-evaluation system for a three-state chain.
   - (a) Name the random variables.
   - (b) Write the target equation.
   - (c) Compute the numeric result.
   - (d) Explain what the result means for an agent or LLM policy.

2. (*) Run three value-iteration backups and track the sup-norm change.
   - (a) Name the random variables.
   - (b) Write the target equation.
   - (c) Compute the numeric result.
   - (d) Explain what the result means for an agent or LLM policy.

3. (*) Compute a Monte Carlo return and compare it with a TD target.
   - (a) Name the random variables.
   - (b) Write the target equation.
   - (c) Compute the numeric result.
   - (d) Explain what the result means for an agent or LLM policy.

4. (**) Apply one SARSA update and one Q-learning update to the same transition.
   - (a) Name the random variables.
   - (b) Write the target equation.
   - (c) Compute the numeric result.
   - (d) Explain what the result means for an agent or LLM policy.

5. (**) Compute an epsilon-greedy action distribution.
   - (a) Name the random variables.
   - (b) Write the target equation.
   - (c) Compute the numeric result.
   - (d) Explain what the result means for an agent or LLM policy.

6. (**) Estimate a policy-gradient direction for a two-action softmax policy.
   - (a) Name the random variables.
   - (b) Write the target equation.
   - (c) Compute the numeric result.
   - (d) Explain what the result means for an agent or LLM policy.

7. (**) Compute generalized advantages from rewards and value estimates.
   - (a) Name the random variables.
   - (b) Write the target equation.
   - (c) Compute the numeric result.
   - (d) Explain what the result means for an agent or LLM policy.

8. (***) Evaluate the PPO clipped surrogate for positive and negative advantages.
   - (a) Name the random variables.
   - (b) Write the target equation.
   - (c) Compute the numeric result.
   - (d) Explain what the result means for an agent or LLM policy.

9. (***) Fit a Bradley-Terry reward-model probability for preference data.
   - (a) Name the random variables.
   - (b) Write the target equation.
   - (c) Compute the numeric result.
   - (d) Explain what the result means for an agent or LLM policy.

10. (***) Compute a DPO-style preference loss and explain the KL-control intuition.
   - (a) Name the random variables.
   - (b) Write the target equation.
   - (c) Compute the numeric result.
   - (d) Explain what the result means for an agent or LLM policy.

## 13. Why This Matters for AI

| Concept | AI impact |
| --- | --- |
| MDPs | Provide the mathematical contract for agents, simulators, robotics tasks, games, and dialogue policies. |
| Bellman equations | Turn long-horizon objectives into one-step recursive learning targets. |
| TD learning | Explains bootstrapping, credit assignment, and value-learning targets used in deep RL. |
| Q-learning | Powers value-based control and explains why replay and target networks stabilize DQN-style systems. |
| Policy gradients | Give the gradient estimator behind REINFORCE, actor-critic, PPO, and many RLHF implementations. |
| Advantage estimation | Reduces variance and makes policy updates more sample-efficient. |
| KL regularization | Keeps a learned policy close to a reference model in RLHF and safe fine-tuning. |
| Reward modeling | Connects human preferences to scalar optimization, while exposing reward hacking risks. |

## 14. Conceptual Bridge

The backward bridge is probability and Markov chains. A Markov chain has transitions but no actions. An MDP adds actions, rewards, and optimization. Once actions enter the process, the learner must reason about both inference and control.

The forward bridge is alignment and interactive systems. RLHF, DPO, preference models, online experiments, and agentic tool-use loops all reuse RL ideas: reward signals, policies, KL constraints, distribution shift, and evaluation under feedback.

```text
+-------------------+      +------------------------+      +----------------------+
| Markov chains     | ---> | Markov decision process | ---> | policy optimization  |
| transition only   |      | actions and rewards     |      | RLHF, agents, games  |
+-------------------+      +------------------------+      +----------------------+
```

The most important practical lesson is that RL is not just an optimizer. It is a complete data-generating loop. When a policy changes, the future dataset changes. That is why careful RL work always audits rewards, policies, value estimates, exploration, and evaluation together.

## References

- Sutton and Barto. Reinforcement Learning: An Introduction, 2nd ed.. https://incompleteideas.net/book/the-book.html
- OpenAI. Spinning Up: Key Concepts in RL. https://spinningup.openai.com/en/latest/spinningup/rl_intro.html
- OpenAI. Spinning Up: Policy Optimization. https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html
- Mnih et al.. Human-level control through deep reinforcement learning. https://www.nature.com/articles/nature14236
- Schulman et al.. Proximal Policy Optimization Algorithms. https://arxiv.org/abs/1707.06347
- Christiano et al.. Deep reinforcement learning from human preferences. https://arxiv.org/abs/1706.03741
- Rafailov et al.. Direct Preference Optimization. https://arxiv.org/abs/2305.18290
