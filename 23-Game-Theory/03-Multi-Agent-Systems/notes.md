[Back to Curriculum](../../README.md) | [Previous: Minimax Theorem](../02-Minimax-Theorem/notes.md) | [Next: Adversarial Game Theory](../04-Adversarial-Game-Theory/notes.md)

---

# Multi-Agent Systems

> _"When many learners share an environment, every policy becomes part of someone else's data distribution."_

## Overview

Multi-agent systems study strategic learning when multiple agents act, adapt, communicate, and optimize in a shared environment.

Game theory is the part of the curriculum that studies adaptive decision makers. It asks what happens when each model, user, attacker, defender, or agent optimizes while anticipating the choices of others.

This section is written in LaTeX Markdown. Inline mathematics uses `$...$`, and display equations use `$$...$$`. The notes emphasize strategy, payoff, best response, equilibrium, exploitability, and adversarial adaptation.

## Prerequisites

- [Markov Chains](../../06-Probability-Theory/07-Markov-Chains/notes.md)
- [Bellman Equations](../../14-Math-for-Specific-Models/06-Reinforcement-Learning/notes.md)
- [Nash Equilibria](../01-Nash-Equilibria/notes.md)
- [Causal Inference](../../22-Causal-Inference/01-Structural-Causal-Models/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Executable demonstrations for multi-agent systems |
| [exercises.ipynb](exercises.ipynb) | Graded practice for multi-agent systems |

## Learning Objectives

After completing this section, you will be able to:

- Define Markov games using states, joint actions, transition kernels, rewards, and discounting
- Compute simple joint-action transitions and agent-specific value functions
- Explain why independent learning creates nonstationary data for every other learner
- Relate Nash policies to equilibrium concepts in stochastic games
- Simulate fictitious play and interpret empirical strategy trajectories
- Compare cooperative, competitive, and mixed-motive multi-agent settings
- Analyze communication, conventions, and credit assignment in team games
- Connect multi-agent learning dynamics to self-play and LLM-agent orchestration
- Use welfare and fairness criteria without confusing them with equilibrium
- Identify when partial observability changes the mathematical model

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 many learners sharing one environment](#11-many-learners-sharing-one-environment)
  - [1.2 nonstationarity](#12-nonstationarity)
  - [1.3 cooperation vs competition](#13-cooperation-vs-competition)
  - [1.4 communication and coordination](#14-communication-and-coordination)
  - [1.5 emergent behavior in AI systems](#15-emergent-behavior-in-ai-systems)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 agent set $N$](#21-agent-set)
  - [2.2 joint action $\mathbf{a}$](#22-joint-action)
  - [2.3 reward vector $\mathbf{r}$](#23-reward-vector)
  - [2.4 Markov game](#24-markov-game)
  - [2.5 joint policy $\boldsymbol{\pi}$](#25-joint-policy)
- [3. Stochastic and Markov Games](#3-stochastic-and-markov-games)
  - [3.1 state transitions](#31-state-transitions)
  - [3.2 value functions for agents](#32-value-functions-for-agents)
  - [3.3 Nash policies](#33-nash-policies)
  - [3.4 cooperative team games](#34-cooperative-team-games)
  - [3.5 partially observed settings preview](#35-partially-observed-settings-preview)
- [4. Learning Dynamics](#4-learning-dynamics)
  - [4.1 independent learners](#41-independent-learners)
  - [4.2 fictitious play](#42-fictitious-play)
  - [4.3 no-regret learning](#43-noregret-learning)
  - [4.4 policy gradients in games](#44-policy-gradients-in-games)
  - [4.5 equilibrium selection](#45-equilibrium-selection)
- [5. Coordination and Communication](#5-coordination-and-communication)
  - [5.1 common-payoff games](#51-commonpayoff-games)
  - [5.2 conventions](#52-conventions)
  - [5.3 communication protocols](#53-communication-protocols)
  - [5.4 credit assignment](#54-credit-assignment)
  - [5.5 social welfare and fairness](#55-social-welfare-and-fairness)
- [6. AI Applications](#6-ai-applications)
  - [6.1 multi-agent RL](#61-multiagent-rl)
  - [6.2 self-play systems](#62-selfplay-systems)
  - [6.3 tool-using LLM swarms](#63-toolusing-llm-swarms)
  - [6.4 market-style model routing](#64-marketstyle-model-routing)
  - [6.5 cooperative safety and oversight](#65-cooperative-safety-and-oversight)
- [7. Common Mistakes](#7-common-mistakes)
- [8. Exercises](#8-exercises)
- [9. Why This Matters for AI](#9-why-this-matters-for-ai)
- [10. Conceptual Bridge](#10-conceptual-bridge)
- [References](#references)

---

## 1. Intuition

Intuition develops the part of multi-agent systems specified by the approved Chapter 23 table of contents. The treatment is game-theoretic, not merely an optimization recipe.

### 1.1 many learners sharing one environment

Many learners sharing one environment belongs to the canonical scope of Multi-Agent Systems. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is Markov games, joint actions, multi-agent value functions, nonstationarity, learning dynamics, coordination, and AI-agent systems. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\mathcal{G}=(N,\mathcal{S},(A_i)_{i\in N},P,(r_i)_{i\in N},\gamma).$$

The formula gives the mathematical handle for many learners sharing one environment. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Many-agent learning means every learner's policy is part of the environment seen by the others.

**Worked reading.**

If agent 1 changes its policy, agent 2's data distribution changes even when the physical simulator is unchanged. That is the core nonstationarity of multi-agent learning.

Three examples of many learners sharing one environment:

1. Self-play agents improving by training against earlier or current versions.
2. LLM tool agents changing each other's context and options.
3. A routing marketplace where traffic shifts after one provider changes quality.

Two non-examples clarify the boundary:

1. A single-agent RL problem with a fixed transition kernel.
2. Batch supervised learning on immutable labels.

Proof or verification habit for many learners sharing one environment:

Analyze the joint policy trajectory, not only individual losses.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, many learners sharing one environment is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Agentic LLM systems make multi-agent math practical: prompts, tools, memory, and policies interact in a shared state.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using many learners sharing one environment responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Ask which part of another agent's behavior enters this agent's observation or reward.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Many learners sharing one environment gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 1.2 nonstationarity

Nonstationarity belongs to the canonical scope of Multi-Agent Systems. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is Markov games, joint actions, multi-agent value functions, nonstationarity, learning dynamics, coordination, and AI-agent systems. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\mathbf{a}_t=(a_{1,t},\ldots,a_{n,t}).$$

The formula gives the mathematical handle for nonstationarity. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Learning dynamics study how strategies move over time, not just where equilibrium points are located.

**Worked reading.**

In fictitious play, each player tracks empirical frequencies of the opponent's past actions and best-responds to those beliefs.

Three examples of nonstationarity:

1. Rock-paper-scissors empirical play approaching the mixed region.
2. Independent Q-learners chasing each other's changing policies.
3. GAN gradients rotating around a saddle-like point.

Two non-examples clarify the boundary:

1. A static equilibrium certificate.
2. A supervised learner trained against an immutable dataset.

Proof or verification habit for nonstationarity:

Analyze updates as a dynamical system: fixed points, cycles, regret, and exploitability are different diagnostics.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, nonstationarity is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Many AI failures are dynamic failures: the target moves while the learner is trying to fit it.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using nonstationarity responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Plot trajectories or regret; do not infer convergence from one snapshot.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Nonstationarity gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 1.3 cooperation vs competition

Cooperation vs competition belongs to the canonical scope of Multi-Agent Systems. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is Markov games, joint actions, multi-agent value functions, nonstationarity, learning dynamics, coordination, and AI-agent systems. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$V_i^{\boldsymbol{\pi}}(s)=\mathbb{E}\left[\sum_{t=0}^{\infty}\gamma^t r_i(s_t,\mathbf{a}_t)\mid s_0=s\right].$$

The formula gives the mathematical handle for cooperation vs competition. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Cooperation and competition are payoff-structure choices. Cooperative games align rewards; competitive games put rewards in conflict; mixed-motive games do both.

**Worked reading.**

A common-payoff team has $r_1=\cdots=r_n$. A zero-sum game has $r_1=-r_2$. Most deployed multi-agent systems sit between these extremes.

Three examples of cooperation vs competition:

1. A team of tool agents sharing a task reward.
2. A self-play opponent trained to expose weaknesses.
3. A market of model providers competing while still serving user welfare.

Two non-examples clarify the boundary:

1. Calling agents cooperative because they are in the same codebase.
2. Calling a game competitive because agents are different processes.

Proof or verification habit for cooperation vs competition:

Classify the payoff relation before selecting an equilibrium or learning method.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, cooperation vs competition is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

The same algorithm can look safe or unsafe depending on whether rewards create cooperation, competition, or collusion.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using cooperation vs competition responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Write the reward vector, not just the environment reward.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Cooperation vs competition gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 1.4 communication and coordination

Communication and coordination belongs to the canonical scope of Multi-Agent Systems. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is Markov games, joint actions, multi-agent value functions, nonstationarity, learning dynamics, coordination, and AI-agent systems. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\boldsymbol{\pi}^* \text{ is Nash if } V_i^{\pi_i^*,\boldsymbol{\pi}_{-i}^*}(s)\ge V_i^{\pi_i,\boldsymbol{\pi}_{-i}^*}(s).$$

The formula gives the mathematical handle for communication and coordination. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Coordination games contain multiple stable outcomes, so the mathematical problem is not only existence but selection.

**Worked reading.**

If two agents both prefer choosing the same protocol, both $(A,A)$ and $(B,B)$ can be equilibria. Which one appears may depend on initialization, communication, history, or focal points.

Three examples of communication and coordination:

1. LLM agents agree on a tool-call schema.
2. Distributed learners converge to a shared convention for labels.
3. A team of models selects the same plan representation before acting.

Two non-examples clarify the boundary:

1. A zero-sum contest where one player's gain is the other's loss.
2. A single model choosing a format without another agent needing to match it.

Proof or verification habit for communication and coordination:

Equilibrium verification is easy; equilibrium selection is the hard part. Show each matched profile is stable, then analyze basins, signals, or welfare.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, communication and coordination is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Coordination failures are common in agentic systems because technically correct local policies can still fail to align interfaces.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using communication and coordination responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Ask whether agents need the same convention, and whether the convention is observable before action.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Communication and coordination gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 1.5 emergent behavior in AI systems

Emergent behavior in ai systems belongs to the canonical scope of Multi-Agent Systems. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is Markov games, joint actions, multi-agent value functions, nonstationarity, learning dynamics, coordination, and AI-agent systems. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\mathcal{G}=(N,\mathcal{S},(A_i)_{i\in N},P,(r_i)_{i\in N},\gamma).$$

The formula gives the mathematical handle for emergent behavior in ai systems. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Many-agent learning means every learner's policy is part of the environment seen by the others.

**Worked reading.**

If agent 1 changes its policy, agent 2's data distribution changes even when the physical simulator is unchanged. That is the core nonstationarity of multi-agent learning.

Three examples of emergent behavior in ai systems:

1. Self-play agents improving by training against earlier or current versions.
2. LLM tool agents changing each other's context and options.
3. A routing marketplace where traffic shifts after one provider changes quality.

Two non-examples clarify the boundary:

1. A single-agent RL problem with a fixed transition kernel.
2. Batch supervised learning on immutable labels.

Proof or verification habit for emergent behavior in ai systems:

Analyze the joint policy trajectory, not only individual losses.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, emergent behavior in ai systems is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Agentic LLM systems make multi-agent math practical: prompts, tools, memory, and policies interact in a shared state.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using emergent behavior in ai systems responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Ask which part of another agent's behavior enters this agent's observation or reward.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Emergent behavior in ai systems gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

## 2. Formal Definitions

Formal Definitions develops the part of multi-agent systems specified by the approved Chapter 23 table of contents. The treatment is game-theoretic, not merely an optimization recipe.

### 2.1 agent set $N$

Agent set $n$ belongs to the canonical scope of Multi-Agent Systems. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is Markov games, joint actions, multi-agent value functions, nonstationarity, learning dynamics, coordination, and AI-agent systems. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\mathbf{a}_t=(a_{1,t},\ldots,a_{n,t}).$$

The formula gives the mathematical handle for agent set $n$. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Players, actions, and payoffs define the interface of a game. If any one of them is vague, the equilibrium claim is usually vague too.

**Worked reading.**

A payoff matrix is a compact table: rows are one player's actions, columns are another player's actions, and entries are utilities or losses induced by the joint action.

Three examples of agent set $n$:

1. A row action chooses a defense, while a column action chooses an attack family.
2. An agent set lists every model or tool-using process that can affect reward.
3. A utility function converts accuracy, safety, latency, and cost into strategic incentives.

Two non-examples clarify the boundary:

1. A metric with no actor who optimizes it.
2. An action that is impossible in deployment but included for convenience.

Proof or verification habit for agent set $n$:

Before proving anything, audit the model specification: every allowed action must map to a payoff for every player.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, agent set $n$ is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Payoff design is AI system design. The game will faithfully optimize the incentives it is given, including bad incentives.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using agent set $n$ responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Can you name each player, enumerate or parameterize its actions, and compute its payoff from a joint action?

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Agent set $n$ gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 2.2 joint action $\mathbf{a}$

Joint action $\mathbf{a}$ belongs to the canonical scope of Multi-Agent Systems. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is Markov games, joint actions, multi-agent value functions, nonstationarity, learning dynamics, coordination, and AI-agent systems. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$V_i^{\boldsymbol{\pi}}(s)=\mathbb{E}\left[\sum_{t=0}^{\infty}\gamma^t r_i(s_t,\mathbf{a}_t)\mid s_0=s\right].$$

The formula gives the mathematical handle for joint action $\mathbf{a}$. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

A Markov game extends an MDP by replacing one action and one reward with joint actions and agent-specific rewards.

**Worked reading.**

At state $s$, agents sample a joint action $\mathbf{a}$, the environment transitions by $P(s'\mid s,\mathbf{a})$, and each agent receives $r_i(s,\mathbf{a})$.

Three examples of joint action $\mathbf{a}$:

1. Two dialogue agents sharing a tool environment.
2. Robot teams with shared state and individual rewards.
3. Self-play systems where the opponent policy is part of the transition distribution.

Two non-examples clarify the boundary:

1. A single-agent MDP with a fixed environment.
2. A static normal-form game with no state.

Proof or verification habit for joint action $\mathbf{a}$:

Bellman-style reasoning still applies, but values are indexed by both agent and joint policy.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, joint action $\mathbf{a}$ is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Multi-agent RL inherits all MDP difficulty and adds strategic nonstationarity.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using joint action $\mathbf{a}$ responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Ask whether rewards are common, opposed, or mixed; the answer changes the solution concept.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Joint action $\mathbf{a}$ gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 2.3 reward vector $\mathbf{r}$

Reward vector $\mathbf{r}$ belongs to the canonical scope of Multi-Agent Systems. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is Markov games, joint actions, multi-agent value functions, nonstationarity, learning dynamics, coordination, and AI-agent systems. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\boldsymbol{\pi}^* \text{ is Nash if } V_i^{\pi_i^*,\boldsymbol{\pi}_{-i}^*}(s)\ge V_i^{\pi_i,\boldsymbol{\pi}_{-i}^*}(s).$$

The formula gives the mathematical handle for reward vector $\mathbf{r}$. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

A Markov game extends an MDP by replacing one action and one reward with joint actions and agent-specific rewards.

**Worked reading.**

At state $s$, agents sample a joint action $\mathbf{a}$, the environment transitions by $P(s'\mid s,\mathbf{a})$, and each agent receives $r_i(s,\mathbf{a})$.

Three examples of reward vector $\mathbf{r}$:

1. Two dialogue agents sharing a tool environment.
2. Robot teams with shared state and individual rewards.
3. Self-play systems where the opponent policy is part of the transition distribution.

Two non-examples clarify the boundary:

1. A single-agent MDP with a fixed environment.
2. A static normal-form game with no state.

Proof or verification habit for reward vector $\mathbf{r}$:

Bellman-style reasoning still applies, but values are indexed by both agent and joint policy.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, reward vector $\mathbf{r}$ is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Multi-agent RL inherits all MDP difficulty and adds strategic nonstationarity.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using reward vector $\mathbf{r}$ responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Ask whether rewards are common, opposed, or mixed; the answer changes the solution concept.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Reward vector $\mathbf{r}$ gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 2.4 Markov game

Markov game belongs to the canonical scope of Multi-Agent Systems. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is Markov games, joint actions, multi-agent value functions, nonstationarity, learning dynamics, coordination, and AI-agent systems. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\mathcal{G}=(N,\mathcal{S},(A_i)_{i\in N},P,(r_i)_{i\in N},\gamma).$$

The formula gives the mathematical handle for markov game. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

A Markov game extends an MDP by replacing one action and one reward with joint actions and agent-specific rewards.

**Worked reading.**

At state $s$, agents sample a joint action $\mathbf{a}$, the environment transitions by $P(s'\mid s,\mathbf{a})$, and each agent receives $r_i(s,\mathbf{a})$.

Three examples of markov game:

1. Two dialogue agents sharing a tool environment.
2. Robot teams with shared state and individual rewards.
3. Self-play systems where the opponent policy is part of the transition distribution.

Two non-examples clarify the boundary:

1. A single-agent MDP with a fixed environment.
2. A static normal-form game with no state.

Proof or verification habit for markov game:

Bellman-style reasoning still applies, but values are indexed by both agent and joint policy.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, markov game is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Multi-agent RL inherits all MDP difficulty and adds strategic nonstationarity.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using markov game responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Ask whether rewards are common, opposed, or mixed; the answer changes the solution concept.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Markov game gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 2.5 joint policy $\boldsymbol{\pi}$

Joint policy $\boldsymbol{\pi}$ belongs to the canonical scope of Multi-Agent Systems. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is Markov games, joint actions, multi-agent value functions, nonstationarity, learning dynamics, coordination, and AI-agent systems. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\mathbf{a}_t=(a_{1,t},\ldots,a_{n,t}).$$

The formula gives the mathematical handle for joint policy $\boldsymbol{\pi}$. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

A Markov game extends an MDP by replacing one action and one reward with joint actions and agent-specific rewards.

**Worked reading.**

At state $s$, agents sample a joint action $\mathbf{a}$, the environment transitions by $P(s'\mid s,\mathbf{a})$, and each agent receives $r_i(s,\mathbf{a})$.

Three examples of joint policy $\boldsymbol{\pi}$:

1. Two dialogue agents sharing a tool environment.
2. Robot teams with shared state and individual rewards.
3. Self-play systems where the opponent policy is part of the transition distribution.

Two non-examples clarify the boundary:

1. A single-agent MDP with a fixed environment.
2. A static normal-form game with no state.

Proof or verification habit for joint policy $\boldsymbol{\pi}$:

Bellman-style reasoning still applies, but values are indexed by both agent and joint policy.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, joint policy $\boldsymbol{\pi}$ is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Multi-agent RL inherits all MDP difficulty and adds strategic nonstationarity.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using joint policy $\boldsymbol{\pi}$ responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Ask whether rewards are common, opposed, or mixed; the answer changes the solution concept.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Joint policy $\boldsymbol{\pi}$ gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

## 3. Stochastic and Markov Games

Stochastic and Markov Games develops the part of multi-agent systems specified by the approved Chapter 23 table of contents. The treatment is game-theoretic, not merely an optimization recipe.

### 3.1 state transitions

State transitions belongs to the canonical scope of Multi-Agent Systems. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is Markov games, joint actions, multi-agent value functions, nonstationarity, learning dynamics, coordination, and AI-agent systems. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$V_i^{\boldsymbol{\pi}}(s)=\mathbb{E}\left[\sum_{t=0}^{\infty}\gamma^t r_i(s_t,\mathbf{a}_t)\mid s_0=s\right].$$

The formula gives the mathematical handle for state transitions. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

A Markov game extends an MDP by replacing one action and one reward with joint actions and agent-specific rewards.

**Worked reading.**

At state $s$, agents sample a joint action $\mathbf{a}$, the environment transitions by $P(s'\mid s,\mathbf{a})$, and each agent receives $r_i(s,\mathbf{a})$.

Three examples of state transitions:

1. Two dialogue agents sharing a tool environment.
2. Robot teams with shared state and individual rewards.
3. Self-play systems where the opponent policy is part of the transition distribution.

Two non-examples clarify the boundary:

1. A single-agent MDP with a fixed environment.
2. A static normal-form game with no state.

Proof or verification habit for state transitions:

Bellman-style reasoning still applies, but values are indexed by both agent and joint policy.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, state transitions is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Multi-agent RL inherits all MDP difficulty and adds strategic nonstationarity.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using state transitions responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Ask whether rewards are common, opposed, or mixed; the answer changes the solution concept.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. State transitions gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 3.2 value functions for agents

Value functions for agents belongs to the canonical scope of Multi-Agent Systems. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is Markov games, joint actions, multi-agent value functions, nonstationarity, learning dynamics, coordination, and AI-agent systems. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\boldsymbol{\pi}^* \text{ is Nash if } V_i^{\pi_i^*,\boldsymbol{\pi}_{-i}^*}(s)\ge V_i^{\pi_i,\boldsymbol{\pi}_{-i}^*}(s).$$

The formula gives the mathematical handle for value functions for agents. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

A Markov game extends an MDP by replacing one action and one reward with joint actions and agent-specific rewards.

**Worked reading.**

At state $s$, agents sample a joint action $\mathbf{a}$, the environment transitions by $P(s'\mid s,\mathbf{a})$, and each agent receives $r_i(s,\mathbf{a})$.

Three examples of value functions for agents:

1. Two dialogue agents sharing a tool environment.
2. Robot teams with shared state and individual rewards.
3. Self-play systems where the opponent policy is part of the transition distribution.

Two non-examples clarify the boundary:

1. A single-agent MDP with a fixed environment.
2. A static normal-form game with no state.

Proof or verification habit for value functions for agents:

Bellman-style reasoning still applies, but values are indexed by both agent and joint policy.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, value functions for agents is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Multi-agent RL inherits all MDP difficulty and adds strategic nonstationarity.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using value functions for agents responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Ask whether rewards are common, opposed, or mixed; the answer changes the solution concept.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Value functions for agents gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 3.3 Nash policies

Nash policies belongs to the canonical scope of Multi-Agent Systems. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is Markov games, joint actions, multi-agent value functions, nonstationarity, learning dynamics, coordination, and AI-agent systems. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\mathcal{G}=(N,\mathcal{S},(A_i)_{i\in N},P,(r_i)_{i\in N},\gamma).$$

The formula gives the mathematical handle for nash policies. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

A Nash equilibrium is a profile of strategies where no player can improve by changing its own strategy while all other strategies remain fixed.

**Worked reading.**

In the prisoner's dilemma payoff convention, mutual defection can be a Nash equilibrium even when mutual cooperation is better for both players. This is the central warning: stability and desirability are different properties.

Three examples of nash policies:

1. A self-play policy pair where neither side has a profitable unilateral exploit.
2. A GAN fixed point where the generator distribution matches data and the discriminator cannot improve classification.
3. A routing market where no model provider benefits from changing only its bid.

Two non-examples clarify the boundary:

1. A high-welfare outcome with a profitable unilateral deviation.
2. A training checkpoint with low loss but a large best-response exploit.

Proof or verification habit for nash policies:

The proof is a universal deviation check: for each player $i$, hold $\pi_{-i}$ fixed and show $u_i(\pi_i^*,\pi_{-i}^*)\ge u_i(\pi_i,\pi_{-i}^*)$ for all allowed $\pi_i$.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, nash policies is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

For AI agents, Nash is a stability diagnostic. It does not guarantee safety, alignment, fairness, or global efficiency unless those objectives are encoded in the game.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using nash policies responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Ask: if one deployed model, user, or attacker changed behavior alone, would it gain?

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Nash policies gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 3.4 cooperative team games

Cooperative team games belongs to the canonical scope of Multi-Agent Systems. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is Markov games, joint actions, multi-agent value functions, nonstationarity, learning dynamics, coordination, and AI-agent systems. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\mathbf{a}_t=(a_{1,t},\ldots,a_{n,t}).$$

The formula gives the mathematical handle for cooperative team games. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Cooperation and competition are payoff-structure choices. Cooperative games align rewards; competitive games put rewards in conflict; mixed-motive games do both.

**Worked reading.**

A common-payoff team has $r_1=\cdots=r_n$. A zero-sum game has $r_1=-r_2$. Most deployed multi-agent systems sit between these extremes.

Three examples of cooperative team games:

1. A team of tool agents sharing a task reward.
2. A self-play opponent trained to expose weaknesses.
3. A market of model providers competing while still serving user welfare.

Two non-examples clarify the boundary:

1. Calling agents cooperative because they are in the same codebase.
2. Calling a game competitive because agents are different processes.

Proof or verification habit for cooperative team games:

Classify the payoff relation before selecting an equilibrium or learning method.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, cooperative team games is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

The same algorithm can look safe or unsafe depending on whether rewards create cooperation, competition, or collusion.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using cooperative team games responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Write the reward vector, not just the environment reward.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Cooperative team games gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 3.5 partially observed settings preview

Partially observed settings preview belongs to the canonical scope of Multi-Agent Systems. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is Markov games, joint actions, multi-agent value functions, nonstationarity, learning dynamics, coordination, and AI-agent systems. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$V_i^{\boldsymbol{\pi}}(s)=\mathbb{E}\left[\sum_{t=0}^{\infty}\gamma^t r_i(s_t,\mathbf{a}_t)\mid s_0=s\right].$$

The formula gives the mathematical handle for partially observed settings preview. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Partial observability means agents condition decisions on observations rather than the full state.

**Worked reading.**

A policy becomes $\pi_i(a_i\mid o_i)$ instead of $\pi_i(a_i\mid s)$, and beliefs or histories may be needed to act well.

Three examples of partially observed settings preview:

1. Agents with different tool logs.
2. A defender that sees alerts but not the attacker's full plan.
3. A dialogue agent that sees conversation text but not hidden user intent.

Two non-examples clarify the boundary:

1. A fully observed Markov game.
2. A static matrix game.

Proof or verification habit for partially observed settings preview:

The proof habit is to specify observation functions and information sets before writing policies.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, partially observed settings preview is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Many AI security and coordination problems are partially observed because intent, hidden prompts, and private tools are not directly visible.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using partially observed settings preview responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: State what each agent observes at decision time.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Partially observed settings preview gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

## 4. Learning Dynamics

Learning Dynamics develops the part of multi-agent systems specified by the approved Chapter 23 table of contents. The treatment is game-theoretic, not merely an optimization recipe.

### 4.1 independent learners

Independent learners belongs to the canonical scope of Multi-Agent Systems. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is Markov games, joint actions, multi-agent value functions, nonstationarity, learning dynamics, coordination, and AI-agent systems. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\boldsymbol{\pi}^* \text{ is Nash if } V_i^{\pi_i^*,\boldsymbol{\pi}_{-i}^*}(s)\ge V_i^{\pi_i,\boldsymbol{\pi}_{-i}^*}(s).$$

The formula gives the mathematical handle for independent learners. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Learning dynamics study how strategies move over time, not just where equilibrium points are located.

**Worked reading.**

In fictitious play, each player tracks empirical frequencies of the opponent's past actions and best-responds to those beliefs.

Three examples of independent learners:

1. Rock-paper-scissors empirical play approaching the mixed region.
2. Independent Q-learners chasing each other's changing policies.
3. GAN gradients rotating around a saddle-like point.

Two non-examples clarify the boundary:

1. A static equilibrium certificate.
2. A supervised learner trained against an immutable dataset.

Proof or verification habit for independent learners:

Analyze updates as a dynamical system: fixed points, cycles, regret, and exploitability are different diagnostics.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, independent learners is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Many AI failures are dynamic failures: the target moves while the learner is trying to fit it.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using independent learners responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Plot trajectories or regret; do not infer convergence from one snapshot.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Independent learners gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 4.2 fictitious play

Fictitious play belongs to the canonical scope of Multi-Agent Systems. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is Markov games, joint actions, multi-agent value functions, nonstationarity, learning dynamics, coordination, and AI-agent systems. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\mathcal{G}=(N,\mathcal{S},(A_i)_{i\in N},P,(r_i)_{i\in N},\gamma).$$

The formula gives the mathematical handle for fictitious play. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Learning dynamics study how strategies move over time, not just where equilibrium points are located.

**Worked reading.**

In fictitious play, each player tracks empirical frequencies of the opponent's past actions and best-responds to those beliefs.

Three examples of fictitious play:

1. Rock-paper-scissors empirical play approaching the mixed region.
2. Independent Q-learners chasing each other's changing policies.
3. GAN gradients rotating around a saddle-like point.

Two non-examples clarify the boundary:

1. A static equilibrium certificate.
2. A supervised learner trained against an immutable dataset.

Proof or verification habit for fictitious play:

Analyze updates as a dynamical system: fixed points, cycles, regret, and exploitability are different diagnostics.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, fictitious play is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Many AI failures are dynamic failures: the target moves while the learner is trying to fit it.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using fictitious play responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Plot trajectories or regret; do not infer convergence from one snapshot.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Fictitious play gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 4.3 no-regret learning

No-regret learning belongs to the canonical scope of Multi-Agent Systems. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is Markov games, joint actions, multi-agent value functions, nonstationarity, learning dynamics, coordination, and AI-agent systems. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\mathbf{a}_t=(a_{1,t},\ldots,a_{n,t}).$$

The formula gives the mathematical handle for no-regret learning. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

No-regret learning turns repeated play into approximate equilibrium guarantees by making average regret small.

**Worked reading.**

If both players in a zero-sum game have average regret at most $\epsilon$, the average strategies are $O(\epsilon)$-approximate minimax strategies.

Three examples of no-regret learning:

1. Multiplicative weights for action probabilities.
2. Self-play policies averaged over training.
3. Exploitability curves used to track poker or board-game agents.

Two non-examples clarify the boundary:

1. A decreasing supervised loss curve with no opponent model.
2. A single final policy checkpoint without averaging or regret accounting.

Proof or verification habit for no-regret learning:

The proof decomposes the average payoff gap into the row player's regret plus the column player's regret.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, no-regret learning is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

This is why practical game-playing systems track exploitability and regret-like quantities instead of only reward.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using no-regret learning responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Report whether the guarantee applies to last iterate, averaged iterate, or best checkpoint.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. No-regret learning gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 4.4 policy gradients in games

Policy gradients in games belongs to the canonical scope of Multi-Agent Systems. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is Markov games, joint actions, multi-agent value functions, nonstationarity, learning dynamics, coordination, and AI-agent systems. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$V_i^{\boldsymbol{\pi}}(s)=\mathbb{E}\left[\sum_{t=0}^{\infty}\gamma^t r_i(s_t,\mathbf{a}_t)\mid s_0=s\right].$$

The formula gives the mathematical handle for policy gradients in games. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Learning dynamics study how strategies move over time, not just where equilibrium points are located.

**Worked reading.**

In fictitious play, each player tracks empirical frequencies of the opponent's past actions and best-responds to those beliefs.

Three examples of policy gradients in games:

1. Rock-paper-scissors empirical play approaching the mixed region.
2. Independent Q-learners chasing each other's changing policies.
3. GAN gradients rotating around a saddle-like point.

Two non-examples clarify the boundary:

1. A static equilibrium certificate.
2. A supervised learner trained against an immutable dataset.

Proof or verification habit for policy gradients in games:

Analyze updates as a dynamical system: fixed points, cycles, regret, and exploitability are different diagnostics.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, policy gradients in games is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Many AI failures are dynamic failures: the target moves while the learner is trying to fit it.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using policy gradients in games responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Plot trajectories or regret; do not infer convergence from one snapshot.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Policy gradients in games gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 4.5 equilibrium selection

Equilibrium selection belongs to the canonical scope of Multi-Agent Systems. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is Markov games, joint actions, multi-agent value functions, nonstationarity, learning dynamics, coordination, and AI-agent systems. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\boldsymbol{\pi}^* \text{ is Nash if } V_i^{\pi_i^*,\boldsymbol{\pi}_{-i}^*}(s)\ge V_i^{\pi_i,\boldsymbol{\pi}_{-i}^*}(s).$$

The formula gives the mathematical handle for equilibrium selection. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Equilibrium selection asks which equilibrium appears when several are mathematically possible.

**Worked reading.**

In a coordination game with two stable conventions, initialization, communication, history, or payoff-dominance can determine the selected convention.

Three examples of equilibrium selection:

1. Two agents converging to the same API schema.
2. Self-play selecting one opening strategy among many stable ones.
3. A market standard emerging from repeated routing choices.

Two non-examples clarify the boundary:

1. The proof that at least one equilibrium exists.
2. A claim that all equilibria are equally safe.

Proof or verification habit for equilibrium selection:

First verify the candidate equilibria, then study basin of attraction or selection criterion.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, equilibrium selection is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

In AI systems, multiple stable behaviors can differ sharply in safety and usefulness.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using equilibrium selection responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Report which equilibrium is selected and why that selection mechanism is credible.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Equilibrium selection gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

## 5. Coordination and Communication

Coordination and Communication develops the part of multi-agent systems specified by the approved Chapter 23 table of contents. The treatment is game-theoretic, not merely an optimization recipe.

### 5.1 common-payoff games

Common-payoff games belongs to the canonical scope of Multi-Agent Systems. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is Markov games, joint actions, multi-agent value functions, nonstationarity, learning dynamics, coordination, and AI-agent systems. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\mathcal{G}=(N,\mathcal{S},(A_i)_{i\in N},P,(r_i)_{i\in N},\gamma).$$

The formula gives the mathematical handle for common-payoff games. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Coordination games contain multiple stable outcomes, so the mathematical problem is not only existence but selection.

**Worked reading.**

If two agents both prefer choosing the same protocol, both $(A,A)$ and $(B,B)$ can be equilibria. Which one appears may depend on initialization, communication, history, or focal points.

Three examples of common-payoff games:

1. LLM agents agree on a tool-call schema.
2. Distributed learners converge to a shared convention for labels.
3. A team of models selects the same plan representation before acting.

Two non-examples clarify the boundary:

1. A zero-sum contest where one player's gain is the other's loss.
2. A single model choosing a format without another agent needing to match it.

Proof or verification habit for common-payoff games:

Equilibrium verification is easy; equilibrium selection is the hard part. Show each matched profile is stable, then analyze basins, signals, or welfare.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, common-payoff games is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Coordination failures are common in agentic systems because technically correct local policies can still fail to align interfaces.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using common-payoff games responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Ask whether agents need the same convention, and whether the convention is observable before action.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Common-payoff games gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 5.2 conventions

Conventions belongs to the canonical scope of Multi-Agent Systems. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is Markov games, joint actions, multi-agent value functions, nonstationarity, learning dynamics, coordination, and AI-agent systems. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\mathbf{a}_t=(a_{1,t},\ldots,a_{n,t}).$$

The formula gives the mathematical handle for conventions. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Coordination games contain multiple stable outcomes, so the mathematical problem is not only existence but selection.

**Worked reading.**

If two agents both prefer choosing the same protocol, both $(A,A)$ and $(B,B)$ can be equilibria. Which one appears may depend on initialization, communication, history, or focal points.

Three examples of conventions:

1. LLM agents agree on a tool-call schema.
2. Distributed learners converge to a shared convention for labels.
3. A team of models selects the same plan representation before acting.

Two non-examples clarify the boundary:

1. A zero-sum contest where one player's gain is the other's loss.
2. A single model choosing a format without another agent needing to match it.

Proof or verification habit for conventions:

Equilibrium verification is easy; equilibrium selection is the hard part. Show each matched profile is stable, then analyze basins, signals, or welfare.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, conventions is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Coordination failures are common in agentic systems because technically correct local policies can still fail to align interfaces.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using conventions responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Ask whether agents need the same convention, and whether the convention is observable before action.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Conventions gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 5.3 communication protocols

Communication protocols belongs to the canonical scope of Multi-Agent Systems. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is Markov games, joint actions, multi-agent value functions, nonstationarity, learning dynamics, coordination, and AI-agent systems. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$V_i^{\boldsymbol{\pi}}(s)=\mathbb{E}\left[\sum_{t=0}^{\infty}\gamma^t r_i(s_t,\mathbf{a}_t)\mid s_0=s\right].$$

The formula gives the mathematical handle for communication protocols. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Communication changes the information structure of a game; credit assignment changes how global outcomes are mapped back to individual actions.

**Worked reading.**

In a common-payoff team, all agents may receive the same reward, but each still needs enough signal to know which local decision helped.

Three examples of communication protocols:

1. Agents exchange intermediate plans before tool use.
2. A debate system where messages reveal evidence.
3. A cooperative safety monitor assigns responsibility to specialized agents.

Two non-examples clarify the boundary:

1. A hidden side channel not included in the game model.
2. A global score used as if it directly explains each agent's contribution.

Proof or verification habit for communication protocols:

Model messages as actions, observations, or signals, then analyze how they alter feasible strategies and incentives.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, communication protocols is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

LLM-agent systems often fail at interfaces before they fail at individual reasoning, so communication is a mathematical object, not an implementation detail.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using communication protocols responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Specify who observes each message and when.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Communication protocols gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 5.4 credit assignment

Credit assignment belongs to the canonical scope of Multi-Agent Systems. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is Markov games, joint actions, multi-agent value functions, nonstationarity, learning dynamics, coordination, and AI-agent systems. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\boldsymbol{\pi}^* \text{ is Nash if } V_i^{\pi_i^*,\boldsymbol{\pi}_{-i}^*}(s)\ge V_i^{\pi_i,\boldsymbol{\pi}_{-i}^*}(s).$$

The formula gives the mathematical handle for credit assignment. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Communication changes the information structure of a game; credit assignment changes how global outcomes are mapped back to individual actions.

**Worked reading.**

In a common-payoff team, all agents may receive the same reward, but each still needs enough signal to know which local decision helped.

Three examples of credit assignment:

1. Agents exchange intermediate plans before tool use.
2. A debate system where messages reveal evidence.
3. A cooperative safety monitor assigns responsibility to specialized agents.

Two non-examples clarify the boundary:

1. A hidden side channel not included in the game model.
2. A global score used as if it directly explains each agent's contribution.

Proof or verification habit for credit assignment:

Model messages as actions, observations, or signals, then analyze how they alter feasible strategies and incentives.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, credit assignment is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

LLM-agent systems often fail at interfaces before they fail at individual reasoning, so communication is a mathematical object, not an implementation detail.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using credit assignment responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Specify who observes each message and when.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Credit assignment gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 5.5 social welfare and fairness

Social welfare and fairness belongs to the canonical scope of Multi-Agent Systems. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is Markov games, joint actions, multi-agent value functions, nonstationarity, learning dynamics, coordination, and AI-agent systems. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\mathcal{G}=(N,\mathcal{S},(A_i)_{i\in N},P,(r_i)_{i\in N},\gamma).$$

The formula gives the mathematical handle for social welfare and fairness. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Pareto and welfare criteria evaluate outcomes across players; equilibrium evaluates unilateral incentives.

**Worked reading.**

An outcome is Pareto inefficient if another feasible outcome makes at least one player better off and no player worse off. A Nash equilibrium can fail this test.

Three examples of social welfare and fairness:

1. Mutual cooperation in a prisoner's dilemma improves both players but may be unstable.
2. A routing policy raises total quality but gives one provider incentive to deviate.
3. A safety policy improves social welfare but reduces one actor's private payoff.

Two non-examples clarify the boundary:

1. A unilateral deviation check.
2. A fairness claim without specifying the social objective.

Proof or verification habit for social welfare and fairness:

Separate the two predicates: first test deviations for equilibrium, then compare feasible outcomes for welfare.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, social welfare and fairness is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

AI alignment often lives in the gap between private incentive and social objective, so this distinction is not philosophical decoration.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using social welfare and fairness responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: If an equilibrium is bad, changing incentives or constraints is usually required; wishing for cooperation is not a proof.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Social welfare and fairness gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

## 6. AI Applications

AI Applications develops the part of multi-agent systems specified by the approved Chapter 23 table of contents. The treatment is game-theoretic, not merely an optimization recipe.

### 6.1 multi-agent RL

Multi-agent rl belongs to the canonical scope of Multi-Agent Systems. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is Markov games, joint actions, multi-agent value functions, nonstationarity, learning dynamics, coordination, and AI-agent systems. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\mathbf{a}_t=(a_{1,t},\ldots,a_{n,t}).$$

The formula gives the mathematical handle for multi-agent rl. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Many-agent learning means every learner's policy is part of the environment seen by the others.

**Worked reading.**

If agent 1 changes its policy, agent 2's data distribution changes even when the physical simulator is unchanged. That is the core nonstationarity of multi-agent learning.

Three examples of multi-agent rl:

1. Self-play agents improving by training against earlier or current versions.
2. LLM tool agents changing each other's context and options.
3. A routing marketplace where traffic shifts after one provider changes quality.

Two non-examples clarify the boundary:

1. A single-agent RL problem with a fixed transition kernel.
2. Batch supervised learning on immutable labels.

Proof or verification habit for multi-agent rl:

Analyze the joint policy trajectory, not only individual losses.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, multi-agent rl is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Agentic LLM systems make multi-agent math practical: prompts, tools, memory, and policies interact in a shared state.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using multi-agent rl responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Ask which part of another agent's behavior enters this agent's observation or reward.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Multi-agent rl gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 6.2 self-play systems

Self-play systems belongs to the canonical scope of Multi-Agent Systems. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is Markov games, joint actions, multi-agent value functions, nonstationarity, learning dynamics, coordination, and AI-agent systems. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$V_i^{\boldsymbol{\pi}}(s)=\mathbb{E}\left[\sum_{t=0}^{\infty}\gamma^t r_i(s_t,\mathbf{a}_t)\mid s_0=s\right].$$

The formula gives the mathematical handle for self-play systems. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Many-agent learning means every learner's policy is part of the environment seen by the others.

**Worked reading.**

If agent 1 changes its policy, agent 2's data distribution changes even when the physical simulator is unchanged. That is the core nonstationarity of multi-agent learning.

Three examples of self-play systems:

1. Self-play agents improving by training against earlier or current versions.
2. LLM tool agents changing each other's context and options.
3. A routing marketplace where traffic shifts after one provider changes quality.

Two non-examples clarify the boundary:

1. A single-agent RL problem with a fixed transition kernel.
2. Batch supervised learning on immutable labels.

Proof or verification habit for self-play systems:

Analyze the joint policy trajectory, not only individual losses.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, self-play systems is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Agentic LLM systems make multi-agent math practical: prompts, tools, memory, and policies interact in a shared state.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using self-play systems responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Ask which part of another agent's behavior enters this agent's observation or reward.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Self-play systems gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 6.3 tool-using LLM swarms

Tool-using llm swarms belongs to the canonical scope of Multi-Agent Systems. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is Markov games, joint actions, multi-agent value functions, nonstationarity, learning dynamics, coordination, and AI-agent systems. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\boldsymbol{\pi}^* \text{ is Nash if } V_i^{\pi_i^*,\boldsymbol{\pi}_{-i}^*}(s)\ge V_i^{\pi_i,\boldsymbol{\pi}_{-i}^*}(s).$$

The formula gives the mathematical handle for tool-using llm swarms. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Generative, evaluation, and deployment games arise when model behavior changes in response to the measurement or defense mechanism.

**Worked reading.**

In a GAN, the discriminator improves its classifier while the generator improves samples to fool it. In red-team evaluation, the attacker improves examples after seeing failures of the defense.

Three examples of tool-using llm swarms:

1. GAN generator-discriminator training.
2. Jailbreak discovery against a deployed policy layer.
3. Benchmark gaming where systems optimize for the public metric instead of the intended task.

Two non-examples clarify the boundary:

1. One-time evaluation on a frozen hidden test set.
2. A content filter measured only against historical prompts.

Proof or verification habit for tool-using llm swarms:

The mathematical proof obligation is to identify the adaptive loop and the payoff each side optimizes.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, tool-using llm swarms is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Many LLM safety and evaluation failures are game failures: optimizing the metric changes the population of attempts.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using tool-using llm swarms responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Ask who can observe the metric, adapt to it, and benefit from adaptation.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Tool-using llm swarms gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 6.4 market-style model routing

Market-style model routing belongs to the canonical scope of Multi-Agent Systems. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is Markov games, joint actions, multi-agent value functions, nonstationarity, learning dynamics, coordination, and AI-agent systems. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\mathcal{G}=(N,\mathcal{S},(A_i)_{i\in N},P,(r_i)_{i\in N},\gamma).$$

The formula gives the mathematical handle for market-style model routing. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Generative, evaluation, and deployment games arise when model behavior changes in response to the measurement or defense mechanism.

**Worked reading.**

In a GAN, the discriminator improves its classifier while the generator improves samples to fool it. In red-team evaluation, the attacker improves examples after seeing failures of the defense.

Three examples of market-style model routing:

1. GAN generator-discriminator training.
2. Jailbreak discovery against a deployed policy layer.
3. Benchmark gaming where systems optimize for the public metric instead of the intended task.

Two non-examples clarify the boundary:

1. One-time evaluation on a frozen hidden test set.
2. A content filter measured only against historical prompts.

Proof or verification habit for market-style model routing:

The mathematical proof obligation is to identify the adaptive loop and the payoff each side optimizes.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, market-style model routing is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Many LLM safety and evaluation failures are game failures: optimizing the metric changes the population of attempts.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using market-style model routing responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Ask who can observe the metric, adapt to it, and benefit from adaptation.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Market-style model routing gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 6.5 cooperative safety and oversight

Cooperative safety and oversight belongs to the canonical scope of Multi-Agent Systems. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is Markov games, joint actions, multi-agent value functions, nonstationarity, learning dynamics, coordination, and AI-agent systems. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\mathbf{a}_t=(a_{1,t},\ldots,a_{n,t}).$$

The formula gives the mathematical handle for cooperative safety and oversight. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Communication changes the information structure of a game; credit assignment changes how global outcomes are mapped back to individual actions.

**Worked reading.**

In a common-payoff team, all agents may receive the same reward, but each still needs enough signal to know which local decision helped.

Three examples of cooperative safety and oversight:

1. Agents exchange intermediate plans before tool use.
2. A debate system where messages reveal evidence.
3. A cooperative safety monitor assigns responsibility to specialized agents.

Two non-examples clarify the boundary:

1. A hidden side channel not included in the game model.
2. A global score used as if it directly explains each agent's contribution.

Proof or verification habit for cooperative safety and oversight:

Model messages as actions, observations, or signals, then analyze how they alter feasible strategies and incentives.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, cooperative safety and oversight is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

LLM-agent systems often fail at interfaces before they fail at individual reasoning, so communication is a mathematical object, not an implementation detail.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using cooperative safety and oversight responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Specify who observes each message and when.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Cooperative safety and oversight gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

## 7. Common Mistakes

| # | Mistake | Why It Is Wrong | Fix |
| --- | --- | --- | --- |
| 1 | Treating equilibrium as social optimality | A Nash equilibrium can be inefficient or unfair. | Compare equilibrium outcomes with Pareto and welfare criteria. |
| 2 | Checking only one player's incentive | Equilibrium requires every player to lack profitable unilateral deviation. | Compute best responses for all players. |
| 3 | Ignoring mixed strategies | Some finite games have no pure equilibrium. | Use probability distributions over actions and the indifference principle. |
| 4 | Applying minimax to non-zero-sum games blindly | Minimax value is a zero-sum guarantee, not a general welfare solution. | State whether payoffs are strictly opposed before using minimax. |
| 5 | Confusing learning convergence with equilibrium | A learning process can cycle, diverge, or converge to a non-equilibrium behavior. | Track regret, exploitability, and stationarity separately. |
| 6 | Forgetting that other agents adapt | In multi-agent systems, each learner changes the data distribution of the others. | Model policies jointly and monitor nonstationarity. |
| 7 | Using average-case metrics against adaptive attackers | An adaptive opponent targets the worst exploitable gap. | Define threat sets and robust objectives. |
| 8 | Equating red teaming with complete security | Red-team examples are samples, not proofs against all attacks. | Use adaptive evaluation and explicit threat models. |
| 9 | Treating GAN instability as ordinary optimization only | GANs are games whose gradients can rotate instead of descend. | Analyze generator and discriminator objectives jointly. |
| 10 | Letting game abstractions erase values | Payoff design determines incentives and side effects. | Audit utility functions, constraints, and welfare implications. |

## 8. Exercises

1. (*) Work through a game-theory task for multi-agent systems.
   - (a) State the players, actions, and payoffs.
   - (b) Compute or characterize best responses.
   - (c) Decide whether the proposed joint strategy is stable.
   - (d) Interpret the result for an AI, LLM, or adversarial system.

2. (*) Work through a game-theory task for multi-agent systems.
   - (a) State the players, actions, and payoffs.
   - (b) Compute or characterize best responses.
   - (c) Decide whether the proposed joint strategy is stable.
   - (d) Interpret the result for an AI, LLM, or adversarial system.

3. (*) Work through a game-theory task for multi-agent systems.
   - (a) State the players, actions, and payoffs.
   - (b) Compute or characterize best responses.
   - (c) Decide whether the proposed joint strategy is stable.
   - (d) Interpret the result for an AI, LLM, or adversarial system.

4. (**) Work through a game-theory task for multi-agent systems.
   - (a) State the players, actions, and payoffs.
   - (b) Compute or characterize best responses.
   - (c) Decide whether the proposed joint strategy is stable.
   - (d) Interpret the result for an AI, LLM, or adversarial system.

5. (**) Work through a game-theory task for multi-agent systems.
   - (a) State the players, actions, and payoffs.
   - (b) Compute or characterize best responses.
   - (c) Decide whether the proposed joint strategy is stable.
   - (d) Interpret the result for an AI, LLM, or adversarial system.

6. (**) Work through a game-theory task for multi-agent systems.
   - (a) State the players, actions, and payoffs.
   - (b) Compute or characterize best responses.
   - (c) Decide whether the proposed joint strategy is stable.
   - (d) Interpret the result for an AI, LLM, or adversarial system.

7. (***) Work through a game-theory task for multi-agent systems.
   - (a) State the players, actions, and payoffs.
   - (b) Compute or characterize best responses.
   - (c) Decide whether the proposed joint strategy is stable.
   - (d) Interpret the result for an AI, LLM, or adversarial system.

8. (***) Work through a game-theory task for multi-agent systems.
   - (a) State the players, actions, and payoffs.
   - (b) Compute or characterize best responses.
   - (c) Decide whether the proposed joint strategy is stable.
   - (d) Interpret the result for an AI, LLM, or adversarial system.

9. (***) Work through a game-theory task for multi-agent systems.
   - (a) State the players, actions, and payoffs.
   - (b) Compute or characterize best responses.
   - (c) Decide whether the proposed joint strategy is stable.
   - (d) Interpret the result for an AI, LLM, or adversarial system.

10. (***) Work through a game-theory task for multi-agent systems.
   - (a) State the players, actions, and payoffs.
   - (b) Compute or characterize best responses.
   - (c) Decide whether the proposed joint strategy is stable.
   - (d) Interpret the result for an AI, LLM, or adversarial system.

## 9. Why This Matters for AI

| Concept | AI Impact |
| --- | --- |
| Best response | Explains how users, attackers, or agents adapt to a model policy |
| Nash equilibrium | Defines strategic stability for GANs, self-play, routing, and agent systems |
| Mixed strategy | Motivates randomized defenses, stochastic policies, and exploration |
| Minimax value | Formalizes robust worst-case guarantees |
| Exploitability | Measures how far a policy is from strategic stability |
| No-regret learning | Connects repeated play to approximate equilibrium |
| Security game | Models limited defensive resources against adaptive threats |
| Payoff design | Shows why objective misspecification creates strategic side effects |

## 10. Conceptual Bridge

Multi-Agent Systems follows causal inference because interventions often change incentives. Chapter 22 asks what changes when an action is taken. Chapter 23 asks what happens when other agents see that action, learn from it, and respond strategically.

The backward bridge is intervention. A policy change can have a causal effect, but if users or attackers adapt, the effect becomes part of a game. The forward bridge is measure theory: later probability foundations make the stochastic strategies, repeated games, and distributional assumptions more rigorous.

```text
+--------------------------------------------------------------+
| Chapter 22: intervention and causal mechanisms               |
| Chapter 23: strategic adaptation and adversarial objectives   |
| Chapter 24: rigorous probability and measure foundations      |
+--------------------------------------------------------------+
```

## References

- Shoham and Leyton-Brown. Multiagent Systems. https://www.masfoundations.org/toc.pdf
- Littman. Markov games as a framework for multi-agent reinforcement learning. https://www.cs.rutgers.edu/~mlittman/papers/ml94-final.pdf
- Nisan et al.. Algorithmic Game Theory. https://doi.org/10.1017/CBO9780511800481
- Cesa-Bianchi and Lugosi. Prediction, Learning, and Games. https://www.cambridge.org/core/books/prediction-learning-and-games/30E375A151AD4A73012C9BA075E6C482
