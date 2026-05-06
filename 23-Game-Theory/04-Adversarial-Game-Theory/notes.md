[Back to Curriculum](../../README.md) | [Previous: Multi-Agent Systems](../03-Multi-Agent-Systems/notes.md) | [Next: Sigma Algebras](../../24-Measure-Theory/01-Sigma-Algebras/notes.md)

---

# Adversarial Game Theory

> _"Security begins when the opponent is modeled as adaptive, not random."_

## Overview

Adversarial game theory studies attacker-defender interaction, robust objectives, strategic adaptation, and security games for AI systems.

Game theory is the part of the curriculum that studies adaptive decision makers. It asks what happens when each model, user, attacker, defender, or agent optimizes while anticipating the choices of others.

This section is written in LaTeX Markdown. Inline mathematics uses `$...$`, and display equations use `$$...$$`. The notes emphasize strategy, payoff, best response, equilibrium, exploitability, and adversarial adaptation.

## Prerequisites

- [Minimax Theorem](../02-Minimax-Theorem/notes.md)
- [Robustness and Distribution Shift](../../17-Evaluation-and-Reliability/03-Robustness-and-Distribution-Shift/notes.md)
- [Policy and Guardrails](../../18-Alignment-and-Safety/04-Policy-and-Guardrails/notes.md)
- [Model Serving and Inference Optimization](../../19-Production-ML-and-MLOps/04-Model-Serving-and-Inference-Optimization/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Executable demonstrations for adversarial game theory |
| [exercises.ipynb](exercises.ipynb) | Graded practice for adversarial game theory |

## Learning Objectives

After completing this section, you will be able to:

- Define attacker-defender games with actions, utilities, losses, and threat sets
- Write robust-risk objectives as nested minimization and maximization problems
- Explain perturbation sets for adversarial examples and PGD-style inner maximization
- Compute simple attacker best responses against defensive allocations
- Distinguish simultaneous, Stackelberg, and repeated security-game timing
- Use randomization and deception as strategic defensive tools
- Model GANs, red-team loops, benchmark gaming, and reward hacking as adaptive games
- Connect adversarial training to robustness under specified threat models
- Explain model extraction, poisoning, and jailbreak pressure as strategic adaptation
- Design evaluation protocols that account for adaptive opponents

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 attacker-defender thinking](#11-attackerdefender-thinking)
  - [1.2 threat models](#12-threat-models)
  - [1.3 strategic adaptation](#13-strategic-adaptation)
  - [1.4 robust vs average-case performance](#14-robust-vs-averagecase-performance)
  - [1.5 adversarial AI as game design](#15-adversarial-ai-as-game-design)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 attacker action $a_A$](#21-attacker-action)
  - [2.2 defender action $a_D$](#22-defender-action)
  - [2.3 utility and loss functions](#23-utility-and-loss-functions)
  - [2.4 threat set $\mathcal{S}$](#24-threat-set)
  - [2.5 security game and Stackelberg preview](#25-security-game-and-stackelberg-preview)
- [3. Adversarial Examples and Robust Optimization](#3-adversarial-examples-and-robust-optimization)
  - [3.1 perturbation sets](#31-perturbation-sets)
  - [3.2 inner maximization](#32-inner-maximization)
  - [3.3 outer minimization](#33-outer-minimization)
  - [3.4 PGD attack preview](#34-pgd-attack-preview)
  - [3.5 robust risk](#35-robust-risk)
- [4. Strategic Security Games](#4-strategic-security-games)
  - [4.1 defender resource allocation](#41-defender-resource-allocation)
  - [4.2 attacker best response](#42-attacker-best-response)
  - [4.3 Stackelberg equilibrium](#43-stackelberg-equilibrium)
  - [4.4 deception and randomization](#44-deception-and-randomization)
  - [4.5 audit and monitoring](#45-audit-and-monitoring)
- [5. Generative and Evaluation Games](#5-generative-and-evaluation-games)
  - [5.1 GAN discriminator-generator game](#51-gan-discriminatorgenerator-game)
  - [5.2 red-team blue-team loops](#52-redteam-blueteam-loops)
  - [5.3 benchmark gaming](#53-benchmark-gaming)
  - [5.4 reward hacking](#54-reward-hacking)
  - [5.5 adaptive evaluation](#55-adaptive-evaluation)
- [6. AI Applications](#6-ai-applications)
  - [6.1 jailbreak defenses](#61-jailbreak-defenses)
  - [6.2 adversarial training](#62-adversarial-training)
  - [6.3 model extraction and poisoning](#63-model-extraction-and-poisoning)
  - [6.4 robust retrieval and tool gates](#64-robust-retrieval-and-tool-gates)
  - [6.5 governance under adaptive opponents](#65-governance-under-adaptive-opponents)
- [7. Common Mistakes](#7-common-mistakes)
- [8. Exercises](#8-exercises)
- [9. Why This Matters for AI](#9-why-this-matters-for-ai)
- [10. Conceptual Bridge](#10-conceptual-bridge)
- [References](#references)

---

## 1. Intuition

Intuition develops the part of adversarial game theory specified by the approved Chapter 23 table of contents. The treatment is game-theoretic, not merely an optimization recipe.

### 1.1 attacker-defender thinking

Attacker-defender thinking belongs to the canonical scope of Adversarial Game Theory. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is attacker-defender games, threat sets, robust optimization, Stackelberg security games, adversarial examples, and adaptive evaluation. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$a_A\in A_A,\qquad a_D\in A_D.$$

The formula gives the mathematical handle for attacker-defender thinking. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Strategic interaction begins when another decision maker can react to the policy being studied. Stability means the modeled behavior remains defensible after that reaction is allowed.

**Worked reading.**

Start with one proposed joint behavior. Freeze everyone except one player, compute that player's best alternative, then repeat for every player. If a profitable switch exists, the behavior is not strategically stable.

Three examples of attacker-defender thinking:

1. A guardrail remains effective after attackers see examples of blocked prompts.
2. A model-routing policy remains attractive after providers update prices.
3. A self-play policy cannot be easily exploited by a newly trained opponent.

Two non-examples clarify the boundary:

1. A high average score on a fixed dataset.
2. A local minimum of one model's loss with no opponent.

Proof or verification habit for attacker-defender thinking:

The verification habit is adversarial: search for profitable deviations rather than only confirming the proposed behavior works in the original scenario.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, attacker-defender thinking is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

This is the mathematical shift from offline ML to deployed AI systems where users, competitors, and automated attacks learn from the model.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using attacker-defender thinking responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: State the adaptation channel: what can the other side observe, change, and optimize?

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Attacker-defender thinking gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 1.2 threat models

Threat models belongs to the canonical scope of Adversarial Game Theory. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is attacker-defender games, threat sets, robust optimization, Stackelberg security games, adversarial examples, and adaptive evaluation. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$R_{\mathrm{rob}}(\theta)=\mathbb{E}_{(\mathbf{x},y)}\left[\max_{\boldsymbol{\delta}\in\mathcal{S}}\mathcal{L}(f_\theta(\mathbf{x}+\boldsymbol{\delta}),y)\right].$$

The formula gives the mathematical handle for threat models. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

A threat model defines the attacker's allowed moves. Robust optimization then trains or evaluates against the worst allowed move.

**Worked reading.**

For an $\ell_\infty$ perturbation set, PGD repeatedly steps in the gradient-sign direction and projects back into the allowed box.

Three examples of threat models:

1. Image perturbations bounded by a norm.
2. Prompt transformations allowed by a jailbreak policy.
3. Retrieval poisoning constrained by an index-insertion budget.

Two non-examples clarify the boundary:

1. Any attack the modeler can imagine but has not formalized.
2. Random corruption treated as adaptive attack.

Proof or verification habit for threat models:

The nested objective is proved meaningful only after the feasible attack set is stated. The inner maximum is over that set, not over all possible bad events.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, threat models is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Adversarial training improves robustness to the modeled threat, not to every strategic behavior.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using threat models responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Write the set $\mathcal{S}$ before writing the max.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Threat models gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 1.3 strategic adaptation

Strategic adaptation belongs to the canonical scope of Adversarial Game Theory. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is attacker-defender games, threat sets, robust optimization, Stackelberg security games, adversarial examples, and adaptive evaluation. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$a_A^*(a_D)\in\arg\max_{a_A}u_A(a_A,a_D).$$

The formula gives the mathematical handle for strategic adaptation. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Strategic interaction begins when another decision maker can react to the policy being studied. Stability means the modeled behavior remains defensible after that reaction is allowed.

**Worked reading.**

Start with one proposed joint behavior. Freeze everyone except one player, compute that player's best alternative, then repeat for every player. If a profitable switch exists, the behavior is not strategically stable.

Three examples of strategic adaptation:

1. A guardrail remains effective after attackers see examples of blocked prompts.
2. A model-routing policy remains attractive after providers update prices.
3. A self-play policy cannot be easily exploited by a newly trained opponent.

Two non-examples clarify the boundary:

1. A high average score on a fixed dataset.
2. A local minimum of one model's loss with no opponent.

Proof or verification habit for strategic adaptation:

The verification habit is adversarial: search for profitable deviations rather than only confirming the proposed behavior works in the original scenario.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, strategic adaptation is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

This is the mathematical shift from offline ML to deployed AI systems where users, competitors, and automated attacks learn from the model.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using strategic adaptation responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: State the adaptation channel: what can the other side observe, change, and optimize?

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Strategic adaptation gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 1.4 robust vs average-case performance

Robust vs average-case performance belongs to the canonical scope of Adversarial Game Theory. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is attacker-defender games, threat sets, robust optimization, Stackelberg security games, adversarial examples, and adaptive evaluation. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\min_G\max_D \mathbb{E}_{\mathbf{x}\sim p_{\mathrm{data}}}\log D(\mathbf{x})+\mathbb{E}_{\mathbf{z}\sim p_{\mathbf{z}}}\log(1-D(G(\mathbf{z}))).$$

The formula gives the mathematical handle for robust vs average-case performance. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

A threat model defines the attacker's allowed moves. Robust optimization then trains or evaluates against the worst allowed move.

**Worked reading.**

For an $\ell_\infty$ perturbation set, PGD repeatedly steps in the gradient-sign direction and projects back into the allowed box.

Three examples of robust vs average-case performance:

1. Image perturbations bounded by a norm.
2. Prompt transformations allowed by a jailbreak policy.
3. Retrieval poisoning constrained by an index-insertion budget.

Two non-examples clarify the boundary:

1. Any attack the modeler can imagine but has not formalized.
2. Random corruption treated as adaptive attack.

Proof or verification habit for robust vs average-case performance:

The nested objective is proved meaningful only after the feasible attack set is stated. The inner maximum is over that set, not over all possible bad events.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, robust vs average-case performance is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Adversarial training improves robustness to the modeled threat, not to every strategic behavior.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using robust vs average-case performance responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Write the set $\mathcal{S}$ before writing the max.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Robust vs average-case performance gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 1.5 adversarial AI as game design

Adversarial ai as game design belongs to the canonical scope of Adversarial Game Theory. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is attacker-defender games, threat sets, robust optimization, Stackelberg security games, adversarial examples, and adaptive evaluation. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$a_A\in A_A,\qquad a_D\in A_D.$$

The formula gives the mathematical handle for adversarial ai as game design. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Strategic interaction begins when another decision maker can react to the policy being studied. Stability means the modeled behavior remains defensible after that reaction is allowed.

**Worked reading.**

Start with one proposed joint behavior. Freeze everyone except one player, compute that player's best alternative, then repeat for every player. If a profitable switch exists, the behavior is not strategically stable.

Three examples of adversarial ai as game design:

1. A guardrail remains effective after attackers see examples of blocked prompts.
2. A model-routing policy remains attractive after providers update prices.
3. A self-play policy cannot be easily exploited by a newly trained opponent.

Two non-examples clarify the boundary:

1. A high average score on a fixed dataset.
2. A local minimum of one model's loss with no opponent.

Proof or verification habit for adversarial ai as game design:

The verification habit is adversarial: search for profitable deviations rather than only confirming the proposed behavior works in the original scenario.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, adversarial ai as game design is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

This is the mathematical shift from offline ML to deployed AI systems where users, competitors, and automated attacks learn from the model.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using adversarial ai as game design responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: State the adaptation channel: what can the other side observe, change, and optimize?

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Adversarial ai as game design gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

## 2. Formal Definitions

Formal Definitions develops the part of adversarial game theory specified by the approved Chapter 23 table of contents. The treatment is game-theoretic, not merely an optimization recipe.

### 2.1 attacker action $a_A$

Attacker action $a_a$ belongs to the canonical scope of Adversarial Game Theory. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is attacker-defender games, threat sets, robust optimization, Stackelberg security games, adversarial examples, and adaptive evaluation. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$R_{\mathrm{rob}}(\theta)=\mathbb{E}_{(\mathbf{x},y)}\left[\max_{\boldsymbol{\delta}\in\mathcal{S}}\mathcal{L}(f_\theta(\mathbf{x}+\boldsymbol{\delta}),y)\right].$$

The formula gives the mathematical handle for attacker action $a_a$. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

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

Three examples of attacker action $a_a$:

1. A row action chooses a defense, while a column action chooses an attack family.
2. An agent set lists every model or tool-using process that can affect reward.
3. A utility function converts accuracy, safety, latency, and cost into strategic incentives.

Two non-examples clarify the boundary:

1. A metric with no actor who optimizes it.
2. An action that is impossible in deployment but included for convenience.

Proof or verification habit for attacker action $a_a$:

Before proving anything, audit the model specification: every allowed action must map to a payoff for every player.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, attacker action $a_a$ is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Payoff design is AI system design. The game will faithfully optimize the incentives it is given, including bad incentives.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using attacker action $a_a$ responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Can you name each player, enumerate or parameterize its actions, and compute its payoff from a joint action?

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Attacker action $a_a$ gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 2.2 defender action $a_D$

Defender action $a_d$ belongs to the canonical scope of Adversarial Game Theory. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is attacker-defender games, threat sets, robust optimization, Stackelberg security games, adversarial examples, and adaptive evaluation. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$a_A^*(a_D)\in\arg\max_{a_A}u_A(a_A,a_D).$$

The formula gives the mathematical handle for defender action $a_d$. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

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

Three examples of defender action $a_d$:

1. A row action chooses a defense, while a column action chooses an attack family.
2. An agent set lists every model or tool-using process that can affect reward.
3. A utility function converts accuracy, safety, latency, and cost into strategic incentives.

Two non-examples clarify the boundary:

1. A metric with no actor who optimizes it.
2. An action that is impossible in deployment but included for convenience.

Proof or verification habit for defender action $a_d$:

Before proving anything, audit the model specification: every allowed action must map to a payoff for every player.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, defender action $a_d$ is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Payoff design is AI system design. The game will faithfully optimize the incentives it is given, including bad incentives.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using defender action $a_d$ responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Can you name each player, enumerate or parameterize its actions, and compute its payoff from a joint action?

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Defender action $a_d$ gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 2.3 utility and loss functions

Utility and loss functions belongs to the canonical scope of Adversarial Game Theory. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is attacker-defender games, threat sets, robust optimization, Stackelberg security games, adversarial examples, and adaptive evaluation. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\min_G\max_D \mathbb{E}_{\mathbf{x}\sim p_{\mathrm{data}}}\log D(\mathbf{x})+\mathbb{E}_{\mathbf{z}\sim p_{\mathbf{z}}}\log(1-D(G(\mathbf{z}))).$$

The formula gives the mathematical handle for utility and loss functions. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

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

Three examples of utility and loss functions:

1. A row action chooses a defense, while a column action chooses an attack family.
2. An agent set lists every model or tool-using process that can affect reward.
3. A utility function converts accuracy, safety, latency, and cost into strategic incentives.

Two non-examples clarify the boundary:

1. A metric with no actor who optimizes it.
2. An action that is impossible in deployment but included for convenience.

Proof or verification habit for utility and loss functions:

Before proving anything, audit the model specification: every allowed action must map to a payoff for every player.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, utility and loss functions is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Payoff design is AI system design. The game will faithfully optimize the incentives it is given, including bad incentives.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using utility and loss functions responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Can you name each player, enumerate or parameterize its actions, and compute its payoff from a joint action?

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Utility and loss functions gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 2.4 threat set $\mathcal{S}$

Threat set $\mathcal{s}$ belongs to the canonical scope of Adversarial Game Theory. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is attacker-defender games, threat sets, robust optimization, Stackelberg security games, adversarial examples, and adaptive evaluation. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$a_A\in A_A,\qquad a_D\in A_D.$$

The formula gives the mathematical handle for threat set $\mathcal{s}$. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

A threat model defines the attacker's allowed moves. Robust optimization then trains or evaluates against the worst allowed move.

**Worked reading.**

For an $\ell_\infty$ perturbation set, PGD repeatedly steps in the gradient-sign direction and projects back into the allowed box.

Three examples of threat set $\mathcal{s}$:

1. Image perturbations bounded by a norm.
2. Prompt transformations allowed by a jailbreak policy.
3. Retrieval poisoning constrained by an index-insertion budget.

Two non-examples clarify the boundary:

1. Any attack the modeler can imagine but has not formalized.
2. Random corruption treated as adaptive attack.

Proof or verification habit for threat set $\mathcal{s}$:

The nested objective is proved meaningful only after the feasible attack set is stated. The inner maximum is over that set, not over all possible bad events.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, threat set $\mathcal{s}$ is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Adversarial training improves robustness to the modeled threat, not to every strategic behavior.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using threat set $\mathcal{s}$ responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Write the set $\mathcal{S}$ before writing the max.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Threat set $\mathcal{s}$ gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 2.5 security game and Stackelberg preview

Security game and stackelberg preview belongs to the canonical scope of Adversarial Game Theory. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is attacker-defender games, threat sets, robust optimization, Stackelberg security games, adversarial examples, and adaptive evaluation. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$R_{\mathrm{rob}}(\theta)=\mathbb{E}_{(\mathbf{x},y)}\left[\max_{\boldsymbol{\delta}\in\mathcal{S}}\mathcal{L}(f_\theta(\mathbf{x}+\boldsymbol{\delta}),y)\right].$$

The formula gives the mathematical handle for security game and stackelberg preview. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Security games often have timing: a defender commits to a randomized allocation, then an attacker chooses a best response.

**Worked reading.**

A defender with two monitors and three targets chooses coverage probabilities; the attacker chooses the target with highest expected utility after observing the commitment rule.

Three examples of security game and stackelberg preview:

1. Random audits over model outputs.
2. Rate-limit allocation over API endpoints.
3. Canary documents placed to detect extraction.

Two non-examples clarify the boundary:

1. A simultaneous zero-sum matrix game with no commitment.
2. A fixed checklist that attackers cannot observe or learn from.

Proof or verification habit for security game and stackelberg preview:

Stackelberg analysis proves optimal commitment by solving the follower's best-response constraints inside the leader's optimization.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, security game and stackelberg preview is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

For AI security, commitment and observability matter because attackers often adapt after seeing public defenses.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using security game and stackelberg preview responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: State what the attacker knows about the defense.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Security game and stackelberg preview gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

## 3. Adversarial Examples and Robust Optimization

Adversarial Examples and Robust Optimization develops the part of adversarial game theory specified by the approved Chapter 23 table of contents. The treatment is game-theoretic, not merely an optimization recipe.

### 3.1 perturbation sets

Perturbation sets belongs to the canonical scope of Adversarial Game Theory. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is attacker-defender games, threat sets, robust optimization, Stackelberg security games, adversarial examples, and adaptive evaluation. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$a_A^*(a_D)\in\arg\max_{a_A}u_A(a_A,a_D).$$

The formula gives the mathematical handle for perturbation sets. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

A threat model defines the attacker's allowed moves. Robust optimization then trains or evaluates against the worst allowed move.

**Worked reading.**

For an $\ell_\infty$ perturbation set, PGD repeatedly steps in the gradient-sign direction and projects back into the allowed box.

Three examples of perturbation sets:

1. Image perturbations bounded by a norm.
2. Prompt transformations allowed by a jailbreak policy.
3. Retrieval poisoning constrained by an index-insertion budget.

Two non-examples clarify the boundary:

1. Any attack the modeler can imagine but has not formalized.
2. Random corruption treated as adaptive attack.

Proof or verification habit for perturbation sets:

The nested objective is proved meaningful only after the feasible attack set is stated. The inner maximum is over that set, not over all possible bad events.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, perturbation sets is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Adversarial training improves robustness to the modeled threat, not to every strategic behavior.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using perturbation sets responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Write the set $\mathcal{S}$ before writing the max.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Perturbation sets gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 3.2 inner maximization

Inner maximization belongs to the canonical scope of Adversarial Game Theory. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is attacker-defender games, threat sets, robust optimization, Stackelberg security games, adversarial examples, and adaptive evaluation. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\min_G\max_D \mathbb{E}_{\mathbf{x}\sim p_{\mathrm{data}}}\log D(\mathbf{x})+\mathbb{E}_{\mathbf{z}\sim p_{\mathbf{z}}}\log(1-D(G(\mathbf{z}))).$$

The formula gives the mathematical handle for inner maximization. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

A threat model defines the attacker's allowed moves. Robust optimization then trains or evaluates against the worst allowed move.

**Worked reading.**

For an $\ell_\infty$ perturbation set, PGD repeatedly steps in the gradient-sign direction and projects back into the allowed box.

Three examples of inner maximization:

1. Image perturbations bounded by a norm.
2. Prompt transformations allowed by a jailbreak policy.
3. Retrieval poisoning constrained by an index-insertion budget.

Two non-examples clarify the boundary:

1. Any attack the modeler can imagine but has not formalized.
2. Random corruption treated as adaptive attack.

Proof or verification habit for inner maximization:

The nested objective is proved meaningful only after the feasible attack set is stated. The inner maximum is over that set, not over all possible bad events.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, inner maximization is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Adversarial training improves robustness to the modeled threat, not to every strategic behavior.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using inner maximization responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Write the set $\mathcal{S}$ before writing the max.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Inner maximization gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 3.3 outer minimization

Outer minimization belongs to the canonical scope of Adversarial Game Theory. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is attacker-defender games, threat sets, robust optimization, Stackelberg security games, adversarial examples, and adaptive evaluation. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$a_A\in A_A,\qquad a_D\in A_D.$$

The formula gives the mathematical handle for outer minimization. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

A threat model defines the attacker's allowed moves. Robust optimization then trains or evaluates against the worst allowed move.

**Worked reading.**

For an $\ell_\infty$ perturbation set, PGD repeatedly steps in the gradient-sign direction and projects back into the allowed box.

Three examples of outer minimization:

1. Image perturbations bounded by a norm.
2. Prompt transformations allowed by a jailbreak policy.
3. Retrieval poisoning constrained by an index-insertion budget.

Two non-examples clarify the boundary:

1. Any attack the modeler can imagine but has not formalized.
2. Random corruption treated as adaptive attack.

Proof or verification habit for outer minimization:

The nested objective is proved meaningful only after the feasible attack set is stated. The inner maximum is over that set, not over all possible bad events.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, outer minimization is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Adversarial training improves robustness to the modeled threat, not to every strategic behavior.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using outer minimization responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Write the set $\mathcal{S}$ before writing the max.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Outer minimization gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 3.4 PGD attack preview

Pgd attack preview belongs to the canonical scope of Adversarial Game Theory. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is attacker-defender games, threat sets, robust optimization, Stackelberg security games, adversarial examples, and adaptive evaluation. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$R_{\mathrm{rob}}(\theta)=\mathbb{E}_{(\mathbf{x},y)}\left[\max_{\boldsymbol{\delta}\in\mathcal{S}}\mathcal{L}(f_\theta(\mathbf{x}+\boldsymbol{\delta}),y)\right].$$

The formula gives the mathematical handle for pgd attack preview. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

A threat model defines the attacker's allowed moves. Robust optimization then trains or evaluates against the worst allowed move.

**Worked reading.**

For an $\ell_\infty$ perturbation set, PGD repeatedly steps in the gradient-sign direction and projects back into the allowed box.

Three examples of pgd attack preview:

1. Image perturbations bounded by a norm.
2. Prompt transformations allowed by a jailbreak policy.
3. Retrieval poisoning constrained by an index-insertion budget.

Two non-examples clarify the boundary:

1. Any attack the modeler can imagine but has not formalized.
2. Random corruption treated as adaptive attack.

Proof or verification habit for pgd attack preview:

The nested objective is proved meaningful only after the feasible attack set is stated. The inner maximum is over that set, not over all possible bad events.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, pgd attack preview is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Adversarial training improves robustness to the modeled threat, not to every strategic behavior.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using pgd attack preview responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Write the set $\mathcal{S}$ before writing the max.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Pgd attack preview gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 3.5 robust risk

Robust risk belongs to the canonical scope of Adversarial Game Theory. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is attacker-defender games, threat sets, robust optimization, Stackelberg security games, adversarial examples, and adaptive evaluation. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$a_A^*(a_D)\in\arg\max_{a_A}u_A(a_A,a_D).$$

The formula gives the mathematical handle for robust risk. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

A threat model defines the attacker's allowed moves. Robust optimization then trains or evaluates against the worst allowed move.

**Worked reading.**

For an $\ell_\infty$ perturbation set, PGD repeatedly steps in the gradient-sign direction and projects back into the allowed box.

Three examples of robust risk:

1. Image perturbations bounded by a norm.
2. Prompt transformations allowed by a jailbreak policy.
3. Retrieval poisoning constrained by an index-insertion budget.

Two non-examples clarify the boundary:

1. Any attack the modeler can imagine but has not formalized.
2. Random corruption treated as adaptive attack.

Proof or verification habit for robust risk:

The nested objective is proved meaningful only after the feasible attack set is stated. The inner maximum is over that set, not over all possible bad events.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, robust risk is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Adversarial training improves robustness to the modeled threat, not to every strategic behavior.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using robust risk responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Write the set $\mathcal{S}$ before writing the max.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Robust risk gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

## 4. Strategic Security Games

Strategic Security Games develops the part of adversarial game theory specified by the approved Chapter 23 table of contents. The treatment is game-theoretic, not merely an optimization recipe.

### 4.1 defender resource allocation

Defender resource allocation belongs to the canonical scope of Adversarial Game Theory. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is attacker-defender games, threat sets, robust optimization, Stackelberg security games, adversarial examples, and adaptive evaluation. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\min_G\max_D \mathbb{E}_{\mathbf{x}\sim p_{\mathrm{data}}}\log D(\mathbf{x})+\mathbb{E}_{\mathbf{z}\sim p_{\mathbf{z}}}\log(1-D(G(\mathbf{z}))).$$

The formula gives the mathematical handle for defender resource allocation. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Security games often have timing: a defender commits to a randomized allocation, then an attacker chooses a best response.

**Worked reading.**

A defender with two monitors and three targets chooses coverage probabilities; the attacker chooses the target with highest expected utility after observing the commitment rule.

Three examples of defender resource allocation:

1. Random audits over model outputs.
2. Rate-limit allocation over API endpoints.
3. Canary documents placed to detect extraction.

Two non-examples clarify the boundary:

1. A simultaneous zero-sum matrix game with no commitment.
2. A fixed checklist that attackers cannot observe or learn from.

Proof or verification habit for defender resource allocation:

Stackelberg analysis proves optimal commitment by solving the follower's best-response constraints inside the leader's optimization.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, defender resource allocation is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

For AI security, commitment and observability matter because attackers often adapt after seeing public defenses.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using defender resource allocation responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: State what the attacker knows about the defense.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Defender resource allocation gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 4.2 attacker best response

Attacker best response belongs to the canonical scope of Adversarial Game Theory. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is attacker-defender games, threat sets, robust optimization, Stackelberg security games, adversarial examples, and adaptive evaluation. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$a_A\in A_A,\qquad a_D\in A_D.$$

The formula gives the mathematical handle for attacker best response. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

A best response is an action or policy that maximizes one player's payoff while the other players' strategies are held fixed.

**Worked reading.**

For a payoff matrix $A$, if the column player chooses column $j$, the row player's best responses are the rows attaining $\max_i A_{ij}$. In mixed play, the best response maximizes expected payoff against the opponent's distribution.

Three examples of attacker best response:

1. A discriminator chooses the classifier update that most separates generated and real samples.
2. An attacker chooses the prompt family with highest bypass rate against a fixed guardrail.
3. A retrieval system chooses the route with highest utility against the current user distribution.

Two non-examples clarify the boundary:

1. The globally highest payoff cell when the opponent is not fixed.
2. A socially preferred action that is not payoff-maximizing for the player.

Proof or verification habit for attacker best response:

To prove a response is best, compare it to every allowed unilateral deviation under the same opponent strategy.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, attacker best response is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Best-response thinking is how exploitability is measured: ask what an adaptive user, attacker, or agent could gain by switching strategy alone.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using attacker best response responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Never call an outcome stable until every player has passed the same best-response check.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Attacker best response gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 4.3 Stackelberg equilibrium

Stackelberg equilibrium belongs to the canonical scope of Adversarial Game Theory. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is attacker-defender games, threat sets, robust optimization, Stackelberg security games, adversarial examples, and adaptive evaluation. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$R_{\mathrm{rob}}(\theta)=\mathbb{E}_{(\mathbf{x},y)}\left[\max_{\boldsymbol{\delta}\in\mathcal{S}}\mathcal{L}(f_\theta(\mathbf{x}+\boldsymbol{\delta}),y)\right].$$

The formula gives the mathematical handle for stackelberg equilibrium. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

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

Three examples of stackelberg equilibrium:

1. A self-play policy pair where neither side has a profitable unilateral exploit.
2. A GAN fixed point where the generator distribution matches data and the discriminator cannot improve classification.
3. A routing market where no model provider benefits from changing only its bid.

Two non-examples clarify the boundary:

1. A high-welfare outcome with a profitable unilateral deviation.
2. A training checkpoint with low loss but a large best-response exploit.

Proof or verification habit for stackelberg equilibrium:

The proof is a universal deviation check: for each player $i$, hold $\pi_{-i}$ fixed and show $u_i(\pi_i^*,\pi_{-i}^*)\ge u_i(\pi_i,\pi_{-i}^*)$ for all allowed $\pi_i$.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, stackelberg equilibrium is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

For AI agents, Nash is a stability diagnostic. It does not guarantee safety, alignment, fairness, or global efficiency unless those objectives are encoded in the game.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using stackelberg equilibrium responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Ask: if one deployed model, user, or attacker changed behavior alone, would it gain?

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Stackelberg equilibrium gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 4.4 deception and randomization

Deception and randomization belongs to the canonical scope of Adversarial Game Theory. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is attacker-defender games, threat sets, robust optimization, Stackelberg security games, adversarial examples, and adaptive evaluation. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$a_A^*(a_D)\in\arg\max_{a_A}u_A(a_A,a_D).$$

The formula gives the mathematical handle for deception and randomization. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Security games often have timing: a defender commits to a randomized allocation, then an attacker chooses a best response.

**Worked reading.**

A defender with two monitors and three targets chooses coverage probabilities; the attacker chooses the target with highest expected utility after observing the commitment rule.

Three examples of deception and randomization:

1. Random audits over model outputs.
2. Rate-limit allocation over API endpoints.
3. Canary documents placed to detect extraction.

Two non-examples clarify the boundary:

1. A simultaneous zero-sum matrix game with no commitment.
2. A fixed checklist that attackers cannot observe or learn from.

Proof or verification habit for deception and randomization:

Stackelberg analysis proves optimal commitment by solving the follower's best-response constraints inside the leader's optimization.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, deception and randomization is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

For AI security, commitment and observability matter because attackers often adapt after seeing public defenses.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using deception and randomization responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: State what the attacker knows about the defense.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Deception and randomization gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 4.5 audit and monitoring

Audit and monitoring belongs to the canonical scope of Adversarial Game Theory. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is attacker-defender games, threat sets, robust optimization, Stackelberg security games, adversarial examples, and adaptive evaluation. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\min_G\max_D \mathbb{E}_{\mathbf{x}\sim p_{\mathrm{data}}}\log D(\mathbf{x})+\mathbb{E}_{\mathbf{z}\sim p_{\mathbf{z}}}\log(1-D(G(\mathbf{z}))).$$

The formula gives the mathematical handle for audit and monitoring. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Security games often have timing: a defender commits to a randomized allocation, then an attacker chooses a best response.

**Worked reading.**

A defender with two monitors and three targets chooses coverage probabilities; the attacker chooses the target with highest expected utility after observing the commitment rule.

Three examples of audit and monitoring:

1. Random audits over model outputs.
2. Rate-limit allocation over API endpoints.
3. Canary documents placed to detect extraction.

Two non-examples clarify the boundary:

1. A simultaneous zero-sum matrix game with no commitment.
2. A fixed checklist that attackers cannot observe or learn from.

Proof or verification habit for audit and monitoring:

Stackelberg analysis proves optimal commitment by solving the follower's best-response constraints inside the leader's optimization.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, audit and monitoring is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

For AI security, commitment and observability matter because attackers often adapt after seeing public defenses.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using audit and monitoring responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: State what the attacker knows about the defense.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Audit and monitoring gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

## 5. Generative and Evaluation Games

Generative and Evaluation Games develops the part of adversarial game theory specified by the approved Chapter 23 table of contents. The treatment is game-theoretic, not merely an optimization recipe.

### 5.1 GAN discriminator-generator game

Gan discriminator-generator game belongs to the canonical scope of Adversarial Game Theory. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is attacker-defender games, threat sets, robust optimization, Stackelberg security games, adversarial examples, and adaptive evaluation. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$a_A\in A_A,\qquad a_D\in A_D.$$

The formula gives the mathematical handle for gan discriminator-generator game. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

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

Three examples of gan discriminator-generator game:

1. GAN generator-discriminator training.
2. Jailbreak discovery against a deployed policy layer.
3. Benchmark gaming where systems optimize for the public metric instead of the intended task.

Two non-examples clarify the boundary:

1. One-time evaluation on a frozen hidden test set.
2. A content filter measured only against historical prompts.

Proof or verification habit for gan discriminator-generator game:

The mathematical proof obligation is to identify the adaptive loop and the payoff each side optimizes.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, gan discriminator-generator game is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Many LLM safety and evaluation failures are game failures: optimizing the metric changes the population of attempts.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using gan discriminator-generator game responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Ask who can observe the metric, adapt to it, and benefit from adaptation.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Gan discriminator-generator game gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 5.2 red-team blue-team loops

Red-team blue-team loops belongs to the canonical scope of Adversarial Game Theory. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is attacker-defender games, threat sets, robust optimization, Stackelberg security games, adversarial examples, and adaptive evaluation. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$R_{\mathrm{rob}}(\theta)=\mathbb{E}_{(\mathbf{x},y)}\left[\max_{\boldsymbol{\delta}\in\mathcal{S}}\mathcal{L}(f_\theta(\mathbf{x}+\boldsymbol{\delta}),y)\right].$$

The formula gives the mathematical handle for red-team blue-team loops. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

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

Three examples of red-team blue-team loops:

1. GAN generator-discriminator training.
2. Jailbreak discovery against a deployed policy layer.
3. Benchmark gaming where systems optimize for the public metric instead of the intended task.

Two non-examples clarify the boundary:

1. One-time evaluation on a frozen hidden test set.
2. A content filter measured only against historical prompts.

Proof or verification habit for red-team blue-team loops:

The mathematical proof obligation is to identify the adaptive loop and the payoff each side optimizes.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, red-team blue-team loops is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Many LLM safety and evaluation failures are game failures: optimizing the metric changes the population of attempts.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using red-team blue-team loops responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Ask who can observe the metric, adapt to it, and benefit from adaptation.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Red-team blue-team loops gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 5.3 benchmark gaming

Benchmark gaming belongs to the canonical scope of Adversarial Game Theory. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is attacker-defender games, threat sets, robust optimization, Stackelberg security games, adversarial examples, and adaptive evaluation. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$a_A^*(a_D)\in\arg\max_{a_A}u_A(a_A,a_D).$$

The formula gives the mathematical handle for benchmark gaming. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

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

Three examples of benchmark gaming:

1. GAN generator-discriminator training.
2. Jailbreak discovery against a deployed policy layer.
3. Benchmark gaming where systems optimize for the public metric instead of the intended task.

Two non-examples clarify the boundary:

1. One-time evaluation on a frozen hidden test set.
2. A content filter measured only against historical prompts.

Proof or verification habit for benchmark gaming:

The mathematical proof obligation is to identify the adaptive loop and the payoff each side optimizes.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, benchmark gaming is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Many LLM safety and evaluation failures are game failures: optimizing the metric changes the population of attempts.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using benchmark gaming responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Ask who can observe the metric, adapt to it, and benefit from adaptation.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Benchmark gaming gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 5.4 reward hacking

Reward hacking belongs to the canonical scope of Adversarial Game Theory. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is attacker-defender games, threat sets, robust optimization, Stackelberg security games, adversarial examples, and adaptive evaluation. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\min_G\max_D \mathbb{E}_{\mathbf{x}\sim p_{\mathrm{data}}}\log D(\mathbf{x})+\mathbb{E}_{\mathbf{z}\sim p_{\mathbf{z}}}\log(1-D(G(\mathbf{z}))).$$

The formula gives the mathematical handle for reward hacking. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

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

Three examples of reward hacking:

1. GAN generator-discriminator training.
2. Jailbreak discovery against a deployed policy layer.
3. Benchmark gaming where systems optimize for the public metric instead of the intended task.

Two non-examples clarify the boundary:

1. One-time evaluation on a frozen hidden test set.
2. A content filter measured only against historical prompts.

Proof or verification habit for reward hacking:

The mathematical proof obligation is to identify the adaptive loop and the payoff each side optimizes.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, reward hacking is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Many LLM safety and evaluation failures are game failures: optimizing the metric changes the population of attempts.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using reward hacking responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Ask who can observe the metric, adapt to it, and benefit from adaptation.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Reward hacking gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 5.5 adaptive evaluation

Adaptive evaluation belongs to the canonical scope of Adversarial Game Theory. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is attacker-defender games, threat sets, robust optimization, Stackelberg security games, adversarial examples, and adaptive evaluation. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$a_A\in A_A,\qquad a_D\in A_D.$$

The formula gives the mathematical handle for adaptive evaluation. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

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

Three examples of adaptive evaluation:

1. GAN generator-discriminator training.
2. Jailbreak discovery against a deployed policy layer.
3. Benchmark gaming where systems optimize for the public metric instead of the intended task.

Two non-examples clarify the boundary:

1. One-time evaluation on a frozen hidden test set.
2. A content filter measured only against historical prompts.

Proof or verification habit for adaptive evaluation:

The mathematical proof obligation is to identify the adaptive loop and the payoff each side optimizes.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, adaptive evaluation is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Many LLM safety and evaluation failures are game failures: optimizing the metric changes the population of attempts.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using adaptive evaluation responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Ask who can observe the metric, adapt to it, and benefit from adaptation.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Adaptive evaluation gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

## 6. AI Applications

AI Applications develops the part of adversarial game theory specified by the approved Chapter 23 table of contents. The treatment is game-theoretic, not merely an optimization recipe.

### 6.1 jailbreak defenses

Jailbreak defenses belongs to the canonical scope of Adversarial Game Theory. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is attacker-defender games, threat sets, robust optimization, Stackelberg security games, adversarial examples, and adaptive evaluation. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$R_{\mathrm{rob}}(\theta)=\mathbb{E}_{(\mathbf{x},y)}\left[\max_{\boldsymbol{\delta}\in\mathcal{S}}\mathcal{L}(f_\theta(\mathbf{x}+\boldsymbol{\delta}),y)\right].$$

The formula gives the mathematical handle for jailbreak defenses. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

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

Three examples of jailbreak defenses:

1. GAN generator-discriminator training.
2. Jailbreak discovery against a deployed policy layer.
3. Benchmark gaming where systems optimize for the public metric instead of the intended task.

Two non-examples clarify the boundary:

1. One-time evaluation on a frozen hidden test set.
2. A content filter measured only against historical prompts.

Proof or verification habit for jailbreak defenses:

The mathematical proof obligation is to identify the adaptive loop and the payoff each side optimizes.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, jailbreak defenses is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Many LLM safety and evaluation failures are game failures: optimizing the metric changes the population of attempts.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using jailbreak defenses responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Ask who can observe the metric, adapt to it, and benefit from adaptation.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Jailbreak defenses gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 6.2 adversarial training

Adversarial training belongs to the canonical scope of Adversarial Game Theory. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is attacker-defender games, threat sets, robust optimization, Stackelberg security games, adversarial examples, and adaptive evaluation. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$a_A^*(a_D)\in\arg\max_{a_A}u_A(a_A,a_D).$$

The formula gives the mathematical handle for adversarial training. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Adversarial training and governance both treat the opponent as adaptive rather than as a fixed noise source.

**Worked reading.**

Training solves an approximate inner attack problem, then updates the model on those attacks. Governance designs rules and monitoring under the expectation that actors respond strategically.

Three examples of adversarial training:

1. PGD adversarial training.
2. Adaptive jailbreak evaluation.
3. Policy rules that anticipate model providers optimizing around metrics.

Two non-examples clarify the boundary:

1. A static checklist.
2. One red-team run treated as exhaustive.

Proof or verification habit for adversarial training:

The argument must connect the adaptation model to the defense or policy mechanism.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, adversarial training is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Robust AI governance needs game-theoretic assumptions because rules create incentives.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using adversarial training responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Specify the adaptive opponent, not only the defense.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Adversarial training gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 6.3 model extraction and poisoning

Model extraction and poisoning belongs to the canonical scope of Adversarial Game Theory. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is attacker-defender games, threat sets, robust optimization, Stackelberg security games, adversarial examples, and adaptive evaluation. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\min_G\max_D \mathbb{E}_{\mathbf{x}\sim p_{\mathrm{data}}}\log D(\mathbf{x})+\mathbb{E}_{\mathbf{z}\sim p_{\mathbf{z}}}\log(1-D(G(\mathbf{z}))).$$

The formula gives the mathematical handle for model extraction and poisoning. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

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

Three examples of model extraction and poisoning:

1. GAN generator-discriminator training.
2. Jailbreak discovery against a deployed policy layer.
3. Benchmark gaming where systems optimize for the public metric instead of the intended task.

Two non-examples clarify the boundary:

1. One-time evaluation on a frozen hidden test set.
2. A content filter measured only against historical prompts.

Proof or verification habit for model extraction and poisoning:

The mathematical proof obligation is to identify the adaptive loop and the payoff each side optimizes.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, model extraction and poisoning is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Many LLM safety and evaluation failures are game failures: optimizing the metric changes the population of attempts.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using model extraction and poisoning responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Ask who can observe the metric, adapt to it, and benefit from adaptation.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Model extraction and poisoning gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 6.4 robust retrieval and tool gates

Robust retrieval and tool gates belongs to the canonical scope of Adversarial Game Theory. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is attacker-defender games, threat sets, robust optimization, Stackelberg security games, adversarial examples, and adaptive evaluation. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$a_A\in A_A,\qquad a_D\in A_D.$$

The formula gives the mathematical handle for robust retrieval and tool gates. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

A threat model defines the attacker's allowed moves. Robust optimization then trains or evaluates against the worst allowed move.

**Worked reading.**

For an $\ell_\infty$ perturbation set, PGD repeatedly steps in the gradient-sign direction and projects back into the allowed box.

Three examples of robust retrieval and tool gates:

1. Image perturbations bounded by a norm.
2. Prompt transformations allowed by a jailbreak policy.
3. Retrieval poisoning constrained by an index-insertion budget.

Two non-examples clarify the boundary:

1. Any attack the modeler can imagine but has not formalized.
2. Random corruption treated as adaptive attack.

Proof or verification habit for robust retrieval and tool gates:

The nested objective is proved meaningful only after the feasible attack set is stated. The inner maximum is over that set, not over all possible bad events.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, robust retrieval and tool gates is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Adversarial training improves robustness to the modeled threat, not to every strategic behavior.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using robust retrieval and tool gates responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Write the set $\mathcal{S}$ before writing the max.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Robust retrieval and tool gates gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 6.5 governance under adaptive opponents

Governance under adaptive opponents belongs to the canonical scope of Adversarial Game Theory. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is attacker-defender games, threat sets, robust optimization, Stackelberg security games, adversarial examples, and adaptive evaluation. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$R_{\mathrm{rob}}(\theta)=\mathbb{E}_{(\mathbf{x},y)}\left[\max_{\boldsymbol{\delta}\in\mathcal{S}}\mathcal{L}(f_\theta(\mathbf{x}+\boldsymbol{\delta}),y)\right].$$

The formula gives the mathematical handle for governance under adaptive opponents. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Adversarial training and governance both treat the opponent as adaptive rather than as a fixed noise source.

**Worked reading.**

Training solves an approximate inner attack problem, then updates the model on those attacks. Governance designs rules and monitoring under the expectation that actors respond strategically.

Three examples of governance under adaptive opponents:

1. PGD adversarial training.
2. Adaptive jailbreak evaluation.
3. Policy rules that anticipate model providers optimizing around metrics.

Two non-examples clarify the boundary:

1. A static checklist.
2. One red-team run treated as exhaustive.

Proof or verification habit for governance under adaptive opponents:

The argument must connect the adaptation model to the defense or policy mechanism.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, governance under adaptive opponents is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Robust AI governance needs game-theoretic assumptions because rules create incentives.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using governance under adaptive opponents responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Specify the adaptive opponent, not only the defense.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Governance under adaptive opponents gives the language to reason about that pressure.

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

1. (*) Work through a game-theory task for adversarial game theory.
   - (a) State the players, actions, and payoffs.
   - (b) Compute or characterize best responses.
   - (c) Decide whether the proposed joint strategy is stable.
   - (d) Interpret the result for an AI, LLM, or adversarial system.

2. (*) Work through a game-theory task for adversarial game theory.
   - (a) State the players, actions, and payoffs.
   - (b) Compute or characterize best responses.
   - (c) Decide whether the proposed joint strategy is stable.
   - (d) Interpret the result for an AI, LLM, or adversarial system.

3. (*) Work through a game-theory task for adversarial game theory.
   - (a) State the players, actions, and payoffs.
   - (b) Compute or characterize best responses.
   - (c) Decide whether the proposed joint strategy is stable.
   - (d) Interpret the result for an AI, LLM, or adversarial system.

4. (**) Work through a game-theory task for adversarial game theory.
   - (a) State the players, actions, and payoffs.
   - (b) Compute or characterize best responses.
   - (c) Decide whether the proposed joint strategy is stable.
   - (d) Interpret the result for an AI, LLM, or adversarial system.

5. (**) Work through a game-theory task for adversarial game theory.
   - (a) State the players, actions, and payoffs.
   - (b) Compute or characterize best responses.
   - (c) Decide whether the proposed joint strategy is stable.
   - (d) Interpret the result for an AI, LLM, or adversarial system.

6. (**) Work through a game-theory task for adversarial game theory.
   - (a) State the players, actions, and payoffs.
   - (b) Compute or characterize best responses.
   - (c) Decide whether the proposed joint strategy is stable.
   - (d) Interpret the result for an AI, LLM, or adversarial system.

7. (***) Work through a game-theory task for adversarial game theory.
   - (a) State the players, actions, and payoffs.
   - (b) Compute or characterize best responses.
   - (c) Decide whether the proposed joint strategy is stable.
   - (d) Interpret the result for an AI, LLM, or adversarial system.

8. (***) Work through a game-theory task for adversarial game theory.
   - (a) State the players, actions, and payoffs.
   - (b) Compute or characterize best responses.
   - (c) Decide whether the proposed joint strategy is stable.
   - (d) Interpret the result for an AI, LLM, or adversarial system.

9. (***) Work through a game-theory task for adversarial game theory.
   - (a) State the players, actions, and payoffs.
   - (b) Compute or characterize best responses.
   - (c) Decide whether the proposed joint strategy is stable.
   - (d) Interpret the result for an AI, LLM, or adversarial system.

10. (***) Work through a game-theory task for adversarial game theory.
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

Adversarial Game Theory follows causal inference because interventions often change incentives. Chapter 22 asks what changes when an action is taken. Chapter 23 asks what happens when other agents see that action, learn from it, and respond strategically.

The backward bridge is intervention. A policy change can have a causal effect, but if users or attackers adapt, the effect becomes part of a game. The forward bridge is measure theory: later probability foundations make the stochastic strategies, repeated games, and distributional assumptions more rigorous.

```text
+--------------------------------------------------------------+
| Chapter 22: intervention and causal mechanisms               |
| Chapter 23: strategic adaptation and adversarial objectives   |
| Chapter 24: rigorous probability and measure foundations      |
+--------------------------------------------------------------+
```

## References

- Madry et al.. Towards Deep Learning Models Resistant to Adversarial Attacks. https://arxiv.org/abs/1706.06083
- Goodfellow et al.. Generative Adversarial Nets. https://arxiv.org/abs/1406.2661
- Nisan et al.. Algorithmic Game Theory. https://doi.org/10.1017/CBO9780511800481
- Brown and Sandholm. Superhuman AI for multiplayer poker. https://www.science.org/doi/10.1126/science.aay2400
