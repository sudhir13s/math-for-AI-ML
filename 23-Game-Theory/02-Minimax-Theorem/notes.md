[Back to Curriculum](../../README.md) | [Previous: Nash Equilibria](../01-Nash-Equilibria/notes.md) | [Next: Multi-Agent Systems](../03-Multi-Agent-Systems/notes.md)

---

# Minimax Theorem

> _"In a zero-sum game, optimal defense and optimal attack meet at the value."_

## Overview

The minimax theorem explains why mixed strategies can equalize worst-case guarantees in finite zero-sum games.

Game theory is the part of the curriculum that studies adaptive decision makers. It asks what happens when each model, user, attacker, defender, or agent optimizes while anticipating the choices of others.

This section is written in LaTeX Markdown. Inline mathematics uses `$...$`, and display equations use `$$...$$`. The notes emphasize strategy, payoff, best response, equilibrium, exploitability, and adversarial adaptation.

## Prerequisites

- [Convex Optimization](../../08-Optimization/01-Convex-Optimization/notes.md)
- [Constrained Optimization](../../08-Optimization/04-Constrained-Optimization/notes.md)
- [Nash Equilibria](../01-Nash-Equilibria/notes.md)
- [Adversarial Robustness Preview](../../17-Evaluation-and-Reliability/03-Robustness-and-Distribution-Shift/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Executable demonstrations for minimax theorem |
| [exercises.ipynb](exercises.ipynb) | Graded practice for minimax theorem |

## Learning Objectives

After completing this section, you will be able to:

- Define finite two-player zero-sum matrix games and their mixed-strategy payoffs
- Compute maximin and minimax values for small payoff matrices
- Recognize pure saddle points and explain when mixed strategies are required
- Write the row-player and column-player linear programs for a zero-sum game
- State von Neumann's minimax theorem for finite games
- Explain the LP-duality proof idea behind equality of maximin and minimax values
- Use exploitability to diagnose approximate zero-sum equilibria
- Simulate no-regret dynamics as an approximation route to minimax play
- Translate minimax objectives into robust ML and adversarial training losses
- Distinguish finite zero-sum minimax from general-sum Nash analysis

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 adversarial decision-making](#11-adversarial-decisionmaking)
  - [1.2 worst-case loss](#12-worstcase-loss)
  - [1.3 maximin vs minimax](#13-maximin-vs-minimax)
  - [1.4 zero-sum games](#14-zerosum-games)
  - [1.5 why minimax appears in robust ML](#15-why-minimax-appears-in-robust-ml)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 two-player zero-sum game](#21-twoplayer-zerosum-game)
  - [2.2 payoff matrix $A$](#22-payoff-matrix)
  - [2.3 mixed strategies $\mathbf{p},\mathbf{q}$](#23-mixed-strategies)
  - [2.4 game value $v$](#24-game-value)
  - [2.5 saddle point](#25-saddle-point)
- [3. Matrix Games](#3-matrix-games)
  - [3.1 pure saddle points](#31-pure-saddle-points)
  - [3.2 mixed strategies](#32-mixed-strategies)
  - [3.3 row player LP](#33-row-player-lp)
  - [3.4 column player dual](#34-column-player-dual)
  - [3.5 rock-paper-scissors](#35-rockpaperscissors)
- [4. Minimax Theorem](#4-minimax-theorem)
  - [4.1 theorem statement](#41-theorem-statement)
  - [4.2 geometric simplex intuition](#42-geometric-simplex-intuition)
  - [4.3 LP duality proof sketch](#43-lp-duality-proof-sketch)
  - [4.4 relation to Nash equilibrium](#44-relation-to-nash-equilibrium)
  - [4.5 finite vs continuous games preview](#45-finite-vs-continuous-games-preview)
- [5. Algorithms and Approximation](#5-algorithms-and-approximation)
  - [5.1 linear programming solution](#51-linear-programming-solution)
  - [5.2 no-regret dynamics](#52-noregret-dynamics)
  - [5.3 multiplicative weights preview](#53-multiplicative-weights-preview)
  - [5.4 exploitability](#54-exploitability)
  - [5.5 approximate equilibrium](#55-approximate-equilibrium)
- [6. AI Applications](#6-ai-applications)
  - [6.1 adversarial robustness objective](#61-adversarial-robustness-objective)
  - [6.2 GAN minimax loss](#62-gan-minimax-loss)
  - [6.3 worst-case decoding and evaluation](#63-worstcase-decoding-and-evaluation)
  - [6.4 Yao-style lower bounds preview](#64-yaostyle-lower-bounds-preview)
  - [6.5 robust policy learning](#65-robust-policy-learning)
- [7. Common Mistakes](#7-common-mistakes)
- [8. Exercises](#8-exercises)
- [9. Why This Matters for AI](#9-why-this-matters-for-ai)
- [10. Conceptual Bridge](#10-conceptual-bridge)
- [References](#references)

---

## 1. Intuition

Intuition develops the part of minimax theorem specified by the approved Chapter 23 table of contents. The treatment is game-theoretic, not merely an optimization recipe.

### 1.1 adversarial decision-making

Adversarial decision-making belongs to the canonical scope of Minimax Theorem. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is zero-sum matrix games, maximin and minimax values, saddle points, LP duality, no-regret approximation, and robust AI objectives. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$v^- = \max_{\mathbf{p}\in\Delta_m}\min_{\mathbf{q}\in\Delta_n}\mathbf{p}^\top A\mathbf{q}.$$

The formula gives the mathematical handle for adversarial decision-making. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

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

Three examples of adversarial decision-making:

1. A guardrail remains effective after attackers see examples of blocked prompts.
2. A model-routing policy remains attractive after providers update prices.
3. A self-play policy cannot be easily exploited by a newly trained opponent.

Two non-examples clarify the boundary:

1. A high average score on a fixed dataset.
2. A local minimum of one model's loss with no opponent.

Proof or verification habit for adversarial decision-making:

The verification habit is adversarial: search for profitable deviations rather than only confirming the proposed behavior works in the original scenario.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, adversarial decision-making is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

This is the mathematical shift from offline ML to deployed AI systems where users, competitors, and automated attacks learn from the model.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using adversarial decision-making responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: State the adaptation channel: what can the other side observe, change, and optimize?

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Adversarial decision-making gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 1.2 worst-case loss

Worst-case loss belongs to the canonical scope of Minimax Theorem. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is zero-sum matrix games, maximin and minimax values, saddle points, LP duality, no-regret approximation, and robust AI objectives. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$v^+ = \min_{\mathbf{q}\in\Delta_n}\max_{\mathbf{p}\in\Delta_m}\mathbf{p}^\top A\mathbf{q}.$$

The formula gives the mathematical handle for worst-case loss. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Minimax reasoning chooses a strategy by its guaranteed performance against the strongest opponent response in a zero-sum game.

**Worked reading.**

The row player computes $\max_p\min_q p^\top A q$ while the column player computes $\min_q\max_p p^\top A q$. The minimax theorem says these values agree for finite zero-sum games.

Three examples of worst-case loss:

1. Robust classification against bounded perturbations.
2. A discriminator maximizing the generator's loss in a simplified GAN objective.
3. Worst-case evaluation where the tester chooses the hardest valid case.

Two non-examples clarify the boundary:

1. Average validation loss over a fixed dataset.
2. General-sum bargaining where both players can gain together.

Proof or verification habit for worst-case loss:

The LP-duality proof writes each player's guarantee as a linear program; strong duality equates the two optimal values.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, worst-case loss is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Minimax is the mathematical backbone of adversarial robustness, but only relative to the stated threat model.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using worst-case loss responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Check zero-sum structure before importing minimax conclusions.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Worst-case loss gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 1.3 maximin vs minimax

Maximin vs minimax belongs to the canonical scope of Minimax Theorem. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is zero-sum matrix games, maximin and minimax values, saddle points, LP duality, no-regret approximation, and robust AI objectives. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\max_{\mathbf{p}\in\Delta_m}\min_j (A^\top\mathbf{p})_j = \min_{\mathbf{q}\in\Delta_n}\max_i (A\mathbf{q})_i.$$

The formula gives the mathematical handle for maximin vs minimax. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Minimax reasoning chooses a strategy by its guaranteed performance against the strongest opponent response in a zero-sum game.

**Worked reading.**

The row player computes $\max_p\min_q p^\top A q$ while the column player computes $\min_q\max_p p^\top A q$. The minimax theorem says these values agree for finite zero-sum games.

Three examples of maximin vs minimax:

1. Robust classification against bounded perturbations.
2. A discriminator maximizing the generator's loss in a simplified GAN objective.
3. Worst-case evaluation where the tester chooses the hardest valid case.

Two non-examples clarify the boundary:

1. Average validation loss over a fixed dataset.
2. General-sum bargaining where both players can gain together.

Proof or verification habit for maximin vs minimax:

The LP-duality proof writes each player's guarantee as a linear program; strong duality equates the two optimal values.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, maximin vs minimax is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Minimax is the mathematical backbone of adversarial robustness, but only relative to the stated threat model.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using maximin vs minimax responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Check zero-sum structure before importing minimax conclusions.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Maximin vs minimax gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 1.4 zero-sum games

Zero-sum games belongs to the canonical scope of Minimax Theorem. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is zero-sum matrix games, maximin and minimax values, saddle points, LP duality, no-regret approximation, and robust AI objectives. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\min_\theta \max_{\boldsymbol{\delta}\in\mathcal{S}} \mathcal{L}(f_\theta(\mathbf{x}+\boldsymbol{\delta}),y).$$

The formula gives the mathematical handle for zero-sum games. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Minimax reasoning chooses a strategy by its guaranteed performance against the strongest opponent response in a zero-sum game.

**Worked reading.**

The row player computes $\max_p\min_q p^\top A q$ while the column player computes $\min_q\max_p p^\top A q$. The minimax theorem says these values agree for finite zero-sum games.

Three examples of zero-sum games:

1. Robust classification against bounded perturbations.
2. A discriminator maximizing the generator's loss in a simplified GAN objective.
3. Worst-case evaluation where the tester chooses the hardest valid case.

Two non-examples clarify the boundary:

1. Average validation loss over a fixed dataset.
2. General-sum bargaining where both players can gain together.

Proof or verification habit for zero-sum games:

The LP-duality proof writes each player's guarantee as a linear program; strong duality equates the two optimal values.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, zero-sum games is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Minimax is the mathematical backbone of adversarial robustness, but only relative to the stated threat model.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using zero-sum games responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Check zero-sum structure before importing minimax conclusions.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Zero-sum games gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 1.5 why minimax appears in robust ML

Why minimax appears in robust ml belongs to the canonical scope of Minimax Theorem. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is zero-sum matrix games, maximin and minimax values, saddle points, LP duality, no-regret approximation, and robust AI objectives. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$v^- = \max_{\mathbf{p}\in\Delta_m}\min_{\mathbf{q}\in\Delta_n}\mathbf{p}^\top A\mathbf{q}.$$

The formula gives the mathematical handle for why minimax appears in robust ml. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Minimax reasoning chooses a strategy by its guaranteed performance against the strongest opponent response in a zero-sum game.

**Worked reading.**

The row player computes $\max_p\min_q p^\top A q$ while the column player computes $\min_q\max_p p^\top A q$. The minimax theorem says these values agree for finite zero-sum games.

Three examples of why minimax appears in robust ml:

1. Robust classification against bounded perturbations.
2. A discriminator maximizing the generator's loss in a simplified GAN objective.
3. Worst-case evaluation where the tester chooses the hardest valid case.

Two non-examples clarify the boundary:

1. Average validation loss over a fixed dataset.
2. General-sum bargaining where both players can gain together.

Proof or verification habit for why minimax appears in robust ml:

The LP-duality proof writes each player's guarantee as a linear program; strong duality equates the two optimal values.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, why minimax appears in robust ml is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Minimax is the mathematical backbone of adversarial robustness, but only relative to the stated threat model.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using why minimax appears in robust ml responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Check zero-sum structure before importing minimax conclusions.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Why minimax appears in robust ml gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

## 2. Formal Definitions

Formal Definitions develops the part of minimax theorem specified by the approved Chapter 23 table of contents. The treatment is game-theoretic, not merely an optimization recipe.

### 2.1 two-player zero-sum game

Two-player zero-sum game belongs to the canonical scope of Minimax Theorem. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is zero-sum matrix games, maximin and minimax values, saddle points, LP duality, no-regret approximation, and robust AI objectives. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$v^+ = \min_{\mathbf{q}\in\Delta_n}\max_{\mathbf{p}\in\Delta_m}\mathbf{p}^\top A\mathbf{q}.$$

The formula gives the mathematical handle for two-player zero-sum game. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Minimax reasoning chooses a strategy by its guaranteed performance against the strongest opponent response in a zero-sum game.

**Worked reading.**

The row player computes $\max_p\min_q p^\top A q$ while the column player computes $\min_q\max_p p^\top A q$. The minimax theorem says these values agree for finite zero-sum games.

Three examples of two-player zero-sum game:

1. Robust classification against bounded perturbations.
2. A discriminator maximizing the generator's loss in a simplified GAN objective.
3. Worst-case evaluation where the tester chooses the hardest valid case.

Two non-examples clarify the boundary:

1. Average validation loss over a fixed dataset.
2. General-sum bargaining where both players can gain together.

Proof or verification habit for two-player zero-sum game:

The LP-duality proof writes each player's guarantee as a linear program; strong duality equates the two optimal values.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, two-player zero-sum game is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Minimax is the mathematical backbone of adversarial robustness, but only relative to the stated threat model.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using two-player zero-sum game responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Check zero-sum structure before importing minimax conclusions.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Two-player zero-sum game gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 2.2 payoff matrix $A$

Payoff matrix $a$ belongs to the canonical scope of Minimax Theorem. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is zero-sum matrix games, maximin and minimax values, saddle points, LP duality, no-regret approximation, and robust AI objectives. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\max_{\mathbf{p}\in\Delta_m}\min_j (A^\top\mathbf{p})_j = \min_{\mathbf{q}\in\Delta_n}\max_i (A\mathbf{q})_i.$$

The formula gives the mathematical handle for payoff matrix $a$. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

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

Three examples of payoff matrix $a$:

1. A row action chooses a defense, while a column action chooses an attack family.
2. An agent set lists every model or tool-using process that can affect reward.
3. A utility function converts accuracy, safety, latency, and cost into strategic incentives.

Two non-examples clarify the boundary:

1. A metric with no actor who optimizes it.
2. An action that is impossible in deployment but included for convenience.

Proof or verification habit for payoff matrix $a$:

Before proving anything, audit the model specification: every allowed action must map to a payoff for every player.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, payoff matrix $a$ is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Payoff design is AI system design. The game will faithfully optimize the incentives it is given, including bad incentives.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using payoff matrix $a$ responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Can you name each player, enumerate or parameterize its actions, and compute its payoff from a joint action?

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Payoff matrix $a$ gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 2.3 mixed strategies $\mathbf{p},\mathbf{q}$

Mixed strategies $\mathbf{p},\mathbf{q}$ belongs to the canonical scope of Minimax Theorem. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is zero-sum matrix games, maximin and minimax values, saddle points, LP duality, no-regret approximation, and robust AI objectives. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\min_\theta \max_{\boldsymbol{\delta}\in\mathcal{S}} \mathcal{L}(f_\theta(\mathbf{x}+\boldsymbol{\delta}),y).$$

The formula gives the mathematical handle for mixed strategies $\mathbf{p},\mathbf{q}$. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

A mixed strategy is a probability distribution over actions. In equilibrium, actions used with positive probability must usually give the same expected payoff; otherwise probability can move to the better action.

**Worked reading.**

In matching pennies, the row player is indifferent only when the column player randomizes heads and tails equally. The same calculation makes the column player indifferent, giving the $1/2,1/2$ equilibrium.

Three examples of mixed strategies $\mathbf{p},\mathbf{q}$:

1. Randomized audits that make attackers uncertain.
2. Stochastic decoding policies that prevent deterministic exploitation.
3. Exploration policies in self-play where pure repetition would be exploited.

Two non-examples clarify the boundary:

1. Adding noise after choosing a deterministic losing action.
2. A distribution that assigns probability to an action with strictly lower payoff while another supported action is better.

Proof or verification habit for mixed strategies $\mathbf{p},\mathbf{q}$:

Set expected payoffs of supported actions equal, solve for probabilities, then verify unsupported actions are not better.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, mixed strategies $\mathbf{p},\mathbf{q}$ is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Mixed strategies explain why robust systems often randomize: predictability can be a vulnerability when opponents adapt.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using mixed strategies $\mathbf{p},\mathbf{q}$ responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Check both support equality and off-support inequalities.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Mixed strategies $\mathbf{p},\mathbf{q}$ gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 2.4 game value $v$

Game value $v$ belongs to the canonical scope of Minimax Theorem. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is zero-sum matrix games, maximin and minimax values, saddle points, LP duality, no-regret approximation, and robust AI objectives. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$v^- = \max_{\mathbf{p}\in\Delta_m}\min_{\mathbf{q}\in\Delta_n}\mathbf{p}^\top A\mathbf{q}.$$

The formula gives the mathematical handle for game value $v$. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

The game value is the payoff both sides can guarantee in a finite zero-sum game when they use optimal mixed strategies.

**Worked reading.**

The row player's guarantee is the lower value; the column player's guarantee is the upper value. The minimax theorem states these agree in finite zero-sum games.

Three examples of game value $v$:

1. The value of rock-paper-scissors is zero.
2. A robust classifier's worst-case loss is the value of a specified attack game.
3. Yao-style lower bounds swap randomized algorithms and input distributions under minimax reasoning.

Two non-examples clarify the boundary:

1. A general-sum welfare score.
2. A last-iterate training reward.

Proof or verification habit for game value $v$:

The finite proof can be read through LP duality: the column player's dual program certifies the same scalar value as the row player's primal program.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, game value $v$ is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

The value is useful because it is a guarantee, not a hope about average-case behavior.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using game value $v$ responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Check the assumptions: finite action sets or the right compactness, convexity, and continuity conditions for extensions.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Game value $v$ gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 2.5 saddle point

Saddle point belongs to the canonical scope of Minimax Theorem. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is zero-sum matrix games, maximin and minimax values, saddle points, LP duality, no-regret approximation, and robust AI objectives. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$v^+ = \min_{\mathbf{q}\in\Delta_n}\max_{\mathbf{p}\in\Delta_m}\mathbf{p}^\top A\mathbf{q}.$$

The formula gives the mathematical handle for saddle point. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Minimax reasoning chooses a strategy by its guaranteed performance against the strongest opponent response in a zero-sum game.

**Worked reading.**

The row player computes $\max_p\min_q p^\top A q$ while the column player computes $\min_q\max_p p^\top A q$. The minimax theorem says these values agree for finite zero-sum games.

Three examples of saddle point:

1. Robust classification against bounded perturbations.
2. A discriminator maximizing the generator's loss in a simplified GAN objective.
3. Worst-case evaluation where the tester chooses the hardest valid case.

Two non-examples clarify the boundary:

1. Average validation loss over a fixed dataset.
2. General-sum bargaining where both players can gain together.

Proof or verification habit for saddle point:

The LP-duality proof writes each player's guarantee as a linear program; strong duality equates the two optimal values.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, saddle point is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Minimax is the mathematical backbone of adversarial robustness, but only relative to the stated threat model.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using saddle point responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Check zero-sum structure before importing minimax conclusions.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Saddle point gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

## 3. Matrix Games

Matrix Games develops the part of minimax theorem specified by the approved Chapter 23 table of contents. The treatment is game-theoretic, not merely an optimization recipe.

### 3.1 pure saddle points

Pure saddle points belongs to the canonical scope of Minimax Theorem. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is zero-sum matrix games, maximin and minimax values, saddle points, LP duality, no-regret approximation, and robust AI objectives. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\max_{\mathbf{p}\in\Delta_m}\min_j (A^\top\mathbf{p})_j = \min_{\mathbf{q}\in\Delta_n}\max_i (A\mathbf{q})_i.$$

The formula gives the mathematical handle for pure saddle points. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Minimax reasoning chooses a strategy by its guaranteed performance against the strongest opponent response in a zero-sum game.

**Worked reading.**

The row player computes $\max_p\min_q p^\top A q$ while the column player computes $\min_q\max_p p^\top A q$. The minimax theorem says these values agree for finite zero-sum games.

Three examples of pure saddle points:

1. Robust classification against bounded perturbations.
2. A discriminator maximizing the generator's loss in a simplified GAN objective.
3. Worst-case evaluation where the tester chooses the hardest valid case.

Two non-examples clarify the boundary:

1. Average validation loss over a fixed dataset.
2. General-sum bargaining where both players can gain together.

Proof or verification habit for pure saddle points:

The LP-duality proof writes each player's guarantee as a linear program; strong duality equates the two optimal values.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, pure saddle points is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Minimax is the mathematical backbone of adversarial robustness, but only relative to the stated threat model.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using pure saddle points responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Check zero-sum structure before importing minimax conclusions.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Pure saddle points gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 3.2 mixed strategies

Mixed strategies belongs to the canonical scope of Minimax Theorem. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is zero-sum matrix games, maximin and minimax values, saddle points, LP duality, no-regret approximation, and robust AI objectives. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\min_\theta \max_{\boldsymbol{\delta}\in\mathcal{S}} \mathcal{L}(f_\theta(\mathbf{x}+\boldsymbol{\delta}),y).$$

The formula gives the mathematical handle for mixed strategies. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

A mixed strategy is a probability distribution over actions. In equilibrium, actions used with positive probability must usually give the same expected payoff; otherwise probability can move to the better action.

**Worked reading.**

In matching pennies, the row player is indifferent only when the column player randomizes heads and tails equally. The same calculation makes the column player indifferent, giving the $1/2,1/2$ equilibrium.

Three examples of mixed strategies:

1. Randomized audits that make attackers uncertain.
2. Stochastic decoding policies that prevent deterministic exploitation.
3. Exploration policies in self-play where pure repetition would be exploited.

Two non-examples clarify the boundary:

1. Adding noise after choosing a deterministic losing action.
2. A distribution that assigns probability to an action with strictly lower payoff while another supported action is better.

Proof or verification habit for mixed strategies:

Set expected payoffs of supported actions equal, solve for probabilities, then verify unsupported actions are not better.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, mixed strategies is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Mixed strategies explain why robust systems often randomize: predictability can be a vulnerability when opponents adapt.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using mixed strategies responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Check both support equality and off-support inequalities.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Mixed strategies gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 3.3 row player LP

Row player lp belongs to the canonical scope of Minimax Theorem. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is zero-sum matrix games, maximin and minimax values, saddle points, LP duality, no-regret approximation, and robust AI objectives. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$v^- = \max_{\mathbf{p}\in\Delta_m}\min_{\mathbf{q}\in\Delta_n}\mathbf{p}^\top A\mathbf{q}.$$

The formula gives the mathematical handle for row player lp. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Finite zero-sum games can be solved as primal-dual linear programs over probability simplices and value variables.

**Worked reading.**

The row player maximizes $v$ subject to every column giving expected payoff at least $v$. The column player minimizes $w$ subject to every row giving expected payoff at most $w$.

Three examples of row player lp:

1. Solving rock-paper-scissors as an LP.
2. Computing an adversarial test mixture with linear constraints.
3. Finding a randomized defense allocation under budget constraints.

Two non-examples clarify the boundary:

1. Using an unconstrained optimizer that ignores probabilities summing to one.
2. Applying LP duality to a nonlinear continuous attack without convexity assumptions.

Proof or verification habit for row player lp:

The dual variables are the opponent's mixed strategy; complementary slackness explains why supported actions tie in payoff.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, row player lp is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

LP form makes game values auditable: constraints correspond directly to opponent deviations.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using row player lp responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Inspect primal feasibility, dual feasibility, and matching objective values.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Row player lp gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 3.4 column player dual

Column player dual belongs to the canonical scope of Minimax Theorem. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is zero-sum matrix games, maximin and minimax values, saddle points, LP duality, no-regret approximation, and robust AI objectives. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$v^+ = \min_{\mathbf{q}\in\Delta_n}\max_{\mathbf{p}\in\Delta_m}\mathbf{p}^\top A\mathbf{q}.$$

The formula gives the mathematical handle for column player dual. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

The game value is the payoff both sides can guarantee in a finite zero-sum game when they use optimal mixed strategies.

**Worked reading.**

The row player's guarantee is the lower value; the column player's guarantee is the upper value. The minimax theorem states these agree in finite zero-sum games.

Three examples of column player dual:

1. The value of rock-paper-scissors is zero.
2. A robust classifier's worst-case loss is the value of a specified attack game.
3. Yao-style lower bounds swap randomized algorithms and input distributions under minimax reasoning.

Two non-examples clarify the boundary:

1. A general-sum welfare score.
2. A last-iterate training reward.

Proof or verification habit for column player dual:

The finite proof can be read through LP duality: the column player's dual program certifies the same scalar value as the row player's primal program.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, column player dual is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

The value is useful because it is a guarantee, not a hope about average-case behavior.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using column player dual responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Check the assumptions: finite action sets or the right compactness, convexity, and continuity conditions for extensions.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Column player dual gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 3.5 rock-paper-scissors

Rock-paper-scissors belongs to the canonical scope of Minimax Theorem. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is zero-sum matrix games, maximin and minimax values, saddle points, LP duality, no-regret approximation, and robust AI objectives. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\max_{\mathbf{p}\in\Delta_m}\min_j (A^\top\mathbf{p})_j = \min_{\mathbf{q}\in\Delta_n}\max_i (A\mathbf{q})_i.$$

The formula gives the mathematical handle for rock-paper-scissors. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Rock-paper-scissors is the canonical finite zero-sum game with no pure saddle point and a uniform mixed equilibrium.

**Worked reading.**

Each pure action is beaten by another pure action, so deterministic play is exploitable. Uniform mixing makes the opponent indifferent among all pure actions.

Three examples of rock-paper-scissors:

1. A toy model of cyclic best responses.
2. A clean demonstration of exploitability.
3. A notebook benchmark for fictitious play or no-regret updates.

Two non-examples clarify the boundary:

1. A coordination game.
2. A game with a dominant strategy.

Proof or verification habit for rock-paper-scissors:

Set the expected payoff of rock, paper, and scissors equal; the only symmetric solution is the uniform distribution.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, rock-paper-scissors is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

RPS is small, but its cycling behavior mirrors larger adversarial training dynamics.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using rock-paper-scissors responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: If a learner puts too much mass on one action, compute the opponent's exploiting action.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Rock-paper-scissors gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

## 4. Minimax Theorem

Minimax Theorem develops the part of minimax theorem specified by the approved Chapter 23 table of contents. The treatment is game-theoretic, not merely an optimization recipe.

### 4.1 theorem statement

Theorem statement belongs to the canonical scope of Minimax Theorem. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is zero-sum matrix games, maximin and minimax values, saddle points, LP duality, no-regret approximation, and robust AI objectives. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\min_\theta \max_{\boldsymbol{\delta}\in\mathcal{S}} \mathcal{L}(f_\theta(\mathbf{x}+\boldsymbol{\delta}),y).$$

The formula gives the mathematical handle for theorem statement. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

The game value is the payoff both sides can guarantee in a finite zero-sum game when they use optimal mixed strategies.

**Worked reading.**

The row player's guarantee is the lower value; the column player's guarantee is the upper value. The minimax theorem states these agree in finite zero-sum games.

Three examples of theorem statement:

1. The value of rock-paper-scissors is zero.
2. A robust classifier's worst-case loss is the value of a specified attack game.
3. Yao-style lower bounds swap randomized algorithms and input distributions under minimax reasoning.

Two non-examples clarify the boundary:

1. A general-sum welfare score.
2. A last-iterate training reward.

Proof or verification habit for theorem statement:

The finite proof can be read through LP duality: the column player's dual program certifies the same scalar value as the row player's primal program.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, theorem statement is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

The value is useful because it is a guarantee, not a hope about average-case behavior.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using theorem statement responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Check the assumptions: finite action sets or the right compactness, convexity, and continuity conditions for extensions.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Theorem statement gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 4.2 geometric simplex intuition

Geometric simplex intuition belongs to the canonical scope of Minimax Theorem. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is zero-sum matrix games, maximin and minimax values, saddle points, LP duality, no-regret approximation, and robust AI objectives. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$v^- = \max_{\mathbf{p}\in\Delta_m}\min_{\mathbf{q}\in\Delta_n}\mathbf{p}^\top A\mathbf{q}.$$

The formula gives the mathematical handle for geometric simplex intuition. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

A mixed strategy is a probability distribution over actions. In equilibrium, actions used with positive probability must usually give the same expected payoff; otherwise probability can move to the better action.

**Worked reading.**

In matching pennies, the row player is indifferent only when the column player randomizes heads and tails equally. The same calculation makes the column player indifferent, giving the $1/2,1/2$ equilibrium.

Three examples of geometric simplex intuition:

1. Randomized audits that make attackers uncertain.
2. Stochastic decoding policies that prevent deterministic exploitation.
3. Exploration policies in self-play where pure repetition would be exploited.

Two non-examples clarify the boundary:

1. Adding noise after choosing a deterministic losing action.
2. A distribution that assigns probability to an action with strictly lower payoff while another supported action is better.

Proof or verification habit for geometric simplex intuition:

Set expected payoffs of supported actions equal, solve for probabilities, then verify unsupported actions are not better.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, geometric simplex intuition is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Mixed strategies explain why robust systems often randomize: predictability can be a vulnerability when opponents adapt.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using geometric simplex intuition responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Check both support equality and off-support inequalities.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Geometric simplex intuition gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 4.3 LP duality proof sketch

Lp duality proof sketch belongs to the canonical scope of Minimax Theorem. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is zero-sum matrix games, maximin and minimax values, saddle points, LP duality, no-regret approximation, and robust AI objectives. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$v^+ = \min_{\mathbf{q}\in\Delta_n}\max_{\mathbf{p}\in\Delta_m}\mathbf{p}^\top A\mathbf{q}.$$

The formula gives the mathematical handle for lp duality proof sketch. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Finite zero-sum games can be solved as primal-dual linear programs over probability simplices and value variables.

**Worked reading.**

The row player maximizes $v$ subject to every column giving expected payoff at least $v$. The column player minimizes $w$ subject to every row giving expected payoff at most $w$.

Three examples of lp duality proof sketch:

1. Solving rock-paper-scissors as an LP.
2. Computing an adversarial test mixture with linear constraints.
3. Finding a randomized defense allocation under budget constraints.

Two non-examples clarify the boundary:

1. Using an unconstrained optimizer that ignores probabilities summing to one.
2. Applying LP duality to a nonlinear continuous attack without convexity assumptions.

Proof or verification habit for lp duality proof sketch:

The dual variables are the opponent's mixed strategy; complementary slackness explains why supported actions tie in payoff.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, lp duality proof sketch is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

LP form makes game values auditable: constraints correspond directly to opponent deviations.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using lp duality proof sketch responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Inspect primal feasibility, dual feasibility, and matching objective values.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Lp duality proof sketch gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 4.4 relation to Nash equilibrium

Relation to nash equilibrium belongs to the canonical scope of Minimax Theorem. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is zero-sum matrix games, maximin and minimax values, saddle points, LP duality, no-regret approximation, and robust AI objectives. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\max_{\mathbf{p}\in\Delta_m}\min_j (A^\top\mathbf{p})_j = \min_{\mathbf{q}\in\Delta_n}\max_i (A\mathbf{q})_i.$$

The formula gives the mathematical handle for relation to nash equilibrium. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

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

Three examples of relation to nash equilibrium:

1. A self-play policy pair where neither side has a profitable unilateral exploit.
2. A GAN fixed point where the generator distribution matches data and the discriminator cannot improve classification.
3. A routing market where no model provider benefits from changing only its bid.

Two non-examples clarify the boundary:

1. A high-welfare outcome with a profitable unilateral deviation.
2. A training checkpoint with low loss but a large best-response exploit.

Proof or verification habit for relation to nash equilibrium:

The proof is a universal deviation check: for each player $i$, hold $\pi_{-i}$ fixed and show $u_i(\pi_i^*,\pi_{-i}^*)\ge u_i(\pi_i,\pi_{-i}^*)$ for all allowed $\pi_i$.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, relation to nash equilibrium is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

For AI agents, Nash is a stability diagnostic. It does not guarantee safety, alignment, fairness, or global efficiency unless those objectives are encoded in the game.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using relation to nash equilibrium responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Ask: if one deployed model, user, or attacker changed behavior alone, would it gain?

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Relation to nash equilibrium gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 4.5 finite vs continuous games preview

Finite vs continuous games preview belongs to the canonical scope of Minimax Theorem. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is zero-sum matrix games, maximin and minimax values, saddle points, LP duality, no-regret approximation, and robust AI objectives. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\min_\theta \max_{\boldsymbol{\delta}\in\mathcal{S}} \mathcal{L}(f_\theta(\mathbf{x}+\boldsymbol{\delta}),y).$$

The formula gives the mathematical handle for finite vs continuous games preview. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

The game value is the payoff both sides can guarantee in a finite zero-sum game when they use optimal mixed strategies.

**Worked reading.**

The row player's guarantee is the lower value; the column player's guarantee is the upper value. The minimax theorem states these agree in finite zero-sum games.

Three examples of finite vs continuous games preview:

1. The value of rock-paper-scissors is zero.
2. A robust classifier's worst-case loss is the value of a specified attack game.
3. Yao-style lower bounds swap randomized algorithms and input distributions under minimax reasoning.

Two non-examples clarify the boundary:

1. A general-sum welfare score.
2. A last-iterate training reward.

Proof or verification habit for finite vs continuous games preview:

The finite proof can be read through LP duality: the column player's dual program certifies the same scalar value as the row player's primal program.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, finite vs continuous games preview is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

The value is useful because it is a guarantee, not a hope about average-case behavior.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using finite vs continuous games preview responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Check the assumptions: finite action sets or the right compactness, convexity, and continuity conditions for extensions.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Finite vs continuous games preview gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

## 5. Algorithms and Approximation

Algorithms and Approximation develops the part of minimax theorem specified by the approved Chapter 23 table of contents. The treatment is game-theoretic, not merely an optimization recipe.

### 5.1 linear programming solution

Linear programming solution belongs to the canonical scope of Minimax Theorem. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is zero-sum matrix games, maximin and minimax values, saddle points, LP duality, no-regret approximation, and robust AI objectives. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$v^- = \max_{\mathbf{p}\in\Delta_m}\min_{\mathbf{q}\in\Delta_n}\mathbf{p}^\top A\mathbf{q}.$$

The formula gives the mathematical handle for linear programming solution. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Finite zero-sum games can be solved as primal-dual linear programs over probability simplices and value variables.

**Worked reading.**

The row player maximizes $v$ subject to every column giving expected payoff at least $v$. The column player minimizes $w$ subject to every row giving expected payoff at most $w$.

Three examples of linear programming solution:

1. Solving rock-paper-scissors as an LP.
2. Computing an adversarial test mixture with linear constraints.
3. Finding a randomized defense allocation under budget constraints.

Two non-examples clarify the boundary:

1. Using an unconstrained optimizer that ignores probabilities summing to one.
2. Applying LP duality to a nonlinear continuous attack without convexity assumptions.

Proof or verification habit for linear programming solution:

The dual variables are the opponent's mixed strategy; complementary slackness explains why supported actions tie in payoff.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, linear programming solution is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

LP form makes game values auditable: constraints correspond directly to opponent deviations.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using linear programming solution responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Inspect primal feasibility, dual feasibility, and matching objective values.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Linear programming solution gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 5.2 no-regret dynamics

No-regret dynamics belongs to the canonical scope of Minimax Theorem. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is zero-sum matrix games, maximin and minimax values, saddle points, LP duality, no-regret approximation, and robust AI objectives. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$v^+ = \min_{\mathbf{q}\in\Delta_n}\max_{\mathbf{p}\in\Delta_m}\mathbf{p}^\top A\mathbf{q}.$$

The formula gives the mathematical handle for no-regret dynamics. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

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

Three examples of no-regret dynamics:

1. Multiplicative weights for action probabilities.
2. Self-play policies averaged over training.
3. Exploitability curves used to track poker or board-game agents.

Two non-examples clarify the boundary:

1. A decreasing supervised loss curve with no opponent model.
2. A single final policy checkpoint without averaging or regret accounting.

Proof or verification habit for no-regret dynamics:

The proof decomposes the average payoff gap into the row player's regret plus the column player's regret.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, no-regret dynamics is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

This is why practical game-playing systems track exploitability and regret-like quantities instead of only reward.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using no-regret dynamics responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Report whether the guarantee applies to last iterate, averaged iterate, or best checkpoint.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. No-regret dynamics gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 5.3 multiplicative weights preview

Multiplicative weights preview belongs to the canonical scope of Minimax Theorem. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is zero-sum matrix games, maximin and minimax values, saddle points, LP duality, no-regret approximation, and robust AI objectives. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\max_{\mathbf{p}\in\Delta_m}\min_j (A^\top\mathbf{p})_j = \min_{\mathbf{q}\in\Delta_n}\max_i (A\mathbf{q})_i.$$

The formula gives the mathematical handle for multiplicative weights preview. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

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

Three examples of multiplicative weights preview:

1. Multiplicative weights for action probabilities.
2. Self-play policies averaged over training.
3. Exploitability curves used to track poker or board-game agents.

Two non-examples clarify the boundary:

1. A decreasing supervised loss curve with no opponent model.
2. A single final policy checkpoint without averaging or regret accounting.

Proof or verification habit for multiplicative weights preview:

The proof decomposes the average payoff gap into the row player's regret plus the column player's regret.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, multiplicative weights preview is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

This is why practical game-playing systems track exploitability and regret-like quantities instead of only reward.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using multiplicative weights preview responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Report whether the guarantee applies to last iterate, averaged iterate, or best checkpoint.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Multiplicative weights preview gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 5.4 exploitability

Exploitability belongs to the canonical scope of Minimax Theorem. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is zero-sum matrix games, maximin and minimax values, saddle points, LP duality, no-regret approximation, and robust AI objectives. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\min_\theta \max_{\boldsymbol{\delta}\in\mathcal{S}} \mathcal{L}(f_\theta(\mathbf{x}+\boldsymbol{\delta}),y).$$

The formula gives the mathematical handle for exploitability. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

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

Three examples of exploitability:

1. Multiplicative weights for action probabilities.
2. Self-play policies averaged over training.
3. Exploitability curves used to track poker or board-game agents.

Two non-examples clarify the boundary:

1. A decreasing supervised loss curve with no opponent model.
2. A single final policy checkpoint without averaging or regret accounting.

Proof or verification habit for exploitability:

The proof decomposes the average payoff gap into the row player's regret plus the column player's regret.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, exploitability is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

This is why practical game-playing systems track exploitability and regret-like quantities instead of only reward.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using exploitability responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Report whether the guarantee applies to last iterate, averaged iterate, or best checkpoint.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Exploitability gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 5.5 approximate equilibrium

Approximate equilibrium belongs to the canonical scope of Minimax Theorem. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is zero-sum matrix games, maximin and minimax values, saddle points, LP duality, no-regret approximation, and robust AI objectives. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$v^- = \max_{\mathbf{p}\in\Delta_m}\min_{\mathbf{q}\in\Delta_n}\mathbf{p}^\top A\mathbf{q}.$$

The formula gives the mathematical handle for approximate equilibrium. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

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

Three examples of approximate equilibrium:

1. A self-play policy pair where neither side has a profitable unilateral exploit.
2. A GAN fixed point where the generator distribution matches data and the discriminator cannot improve classification.
3. A routing market where no model provider benefits from changing only its bid.

Two non-examples clarify the boundary:

1. A high-welfare outcome with a profitable unilateral deviation.
2. A training checkpoint with low loss but a large best-response exploit.

Proof or verification habit for approximate equilibrium:

The proof is a universal deviation check: for each player $i$, hold $\pi_{-i}$ fixed and show $u_i(\pi_i^*,\pi_{-i}^*)\ge u_i(\pi_i,\pi_{-i}^*)$ for all allowed $\pi_i$.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, approximate equilibrium is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

For AI agents, Nash is a stability diagnostic. It does not guarantee safety, alignment, fairness, or global efficiency unless those objectives are encoded in the game.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using approximate equilibrium responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Ask: if one deployed model, user, or attacker changed behavior alone, would it gain?

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Approximate equilibrium gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

## 6. AI Applications

AI Applications develops the part of minimax theorem specified by the approved Chapter 23 table of contents. The treatment is game-theoretic, not merely an optimization recipe.

### 6.1 adversarial robustness objective

Adversarial robustness objective belongs to the canonical scope of Minimax Theorem. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is zero-sum matrix games, maximin and minimax values, saddle points, LP duality, no-regret approximation, and robust AI objectives. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$v^+ = \min_{\mathbf{q}\in\Delta_n}\max_{\mathbf{p}\in\Delta_m}\mathbf{p}^\top A\mathbf{q}.$$

The formula gives the mathematical handle for adversarial robustness objective. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

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

Three examples of adversarial robustness objective:

1. Image perturbations bounded by a norm.
2. Prompt transformations allowed by a jailbreak policy.
3. Retrieval poisoning constrained by an index-insertion budget.

Two non-examples clarify the boundary:

1. Any attack the modeler can imagine but has not formalized.
2. Random corruption treated as adaptive attack.

Proof or verification habit for adversarial robustness objective:

The nested objective is proved meaningful only after the feasible attack set is stated. The inner maximum is over that set, not over all possible bad events.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, adversarial robustness objective is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Adversarial training improves robustness to the modeled threat, not to every strategic behavior.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using adversarial robustness objective responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Write the set $\mathcal{S}$ before writing the max.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Adversarial robustness objective gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 6.2 GAN minimax loss

Gan minimax loss belongs to the canonical scope of Minimax Theorem. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is zero-sum matrix games, maximin and minimax values, saddle points, LP duality, no-regret approximation, and robust AI objectives. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\max_{\mathbf{p}\in\Delta_m}\min_j (A^\top\mathbf{p})_j = \min_{\mathbf{q}\in\Delta_n}\max_i (A\mathbf{q})_i.$$

The formula gives the mathematical handle for gan minimax loss. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Minimax reasoning chooses a strategy by its guaranteed performance against the strongest opponent response in a zero-sum game.

**Worked reading.**

The row player computes $\max_p\min_q p^\top A q$ while the column player computes $\min_q\max_p p^\top A q$. The minimax theorem says these values agree for finite zero-sum games.

Three examples of gan minimax loss:

1. Robust classification against bounded perturbations.
2. A discriminator maximizing the generator's loss in a simplified GAN objective.
3. Worst-case evaluation where the tester chooses the hardest valid case.

Two non-examples clarify the boundary:

1. Average validation loss over a fixed dataset.
2. General-sum bargaining where both players can gain together.

Proof or verification habit for gan minimax loss:

The LP-duality proof writes each player's guarantee as a linear program; strong duality equates the two optimal values.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, gan minimax loss is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Minimax is the mathematical backbone of adversarial robustness, but only relative to the stated threat model.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using gan minimax loss responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Check zero-sum structure before importing minimax conclusions.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Gan minimax loss gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 6.3 worst-case decoding and evaluation

Worst-case decoding and evaluation belongs to the canonical scope of Minimax Theorem. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is zero-sum matrix games, maximin and minimax values, saddle points, LP duality, no-regret approximation, and robust AI objectives. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\min_\theta \max_{\boldsymbol{\delta}\in\mathcal{S}} \mathcal{L}(f_\theta(\mathbf{x}+\boldsymbol{\delta}),y).$$

The formula gives the mathematical handle for worst-case decoding and evaluation. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Minimax reasoning chooses a strategy by its guaranteed performance against the strongest opponent response in a zero-sum game.

**Worked reading.**

The row player computes $\max_p\min_q p^\top A q$ while the column player computes $\min_q\max_p p^\top A q$. The minimax theorem says these values agree for finite zero-sum games.

Three examples of worst-case decoding and evaluation:

1. Robust classification against bounded perturbations.
2. A discriminator maximizing the generator's loss in a simplified GAN objective.
3. Worst-case evaluation where the tester chooses the hardest valid case.

Two non-examples clarify the boundary:

1. Average validation loss over a fixed dataset.
2. General-sum bargaining where both players can gain together.

Proof or verification habit for worst-case decoding and evaluation:

The LP-duality proof writes each player's guarantee as a linear program; strong duality equates the two optimal values.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, worst-case decoding and evaluation is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Minimax is the mathematical backbone of adversarial robustness, but only relative to the stated threat model.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using worst-case decoding and evaluation responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Check zero-sum structure before importing minimax conclusions.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Worst-case decoding and evaluation gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 6.4 Yao-style lower bounds preview

Yao-style lower bounds preview belongs to the canonical scope of Minimax Theorem. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is zero-sum matrix games, maximin and minimax values, saddle points, LP duality, no-regret approximation, and robust AI objectives. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$v^- = \max_{\mathbf{p}\in\Delta_m}\min_{\mathbf{q}\in\Delta_n}\mathbf{p}^\top A\mathbf{q}.$$

The formula gives the mathematical handle for yao-style lower bounds preview. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

The game value is the payoff both sides can guarantee in a finite zero-sum game when they use optimal mixed strategies.

**Worked reading.**

The row player's guarantee is the lower value; the column player's guarantee is the upper value. The minimax theorem states these agree in finite zero-sum games.

Three examples of yao-style lower bounds preview:

1. The value of rock-paper-scissors is zero.
2. A robust classifier's worst-case loss is the value of a specified attack game.
3. Yao-style lower bounds swap randomized algorithms and input distributions under minimax reasoning.

Two non-examples clarify the boundary:

1. A general-sum welfare score.
2. A last-iterate training reward.

Proof or verification habit for yao-style lower bounds preview:

The finite proof can be read through LP duality: the column player's dual program certifies the same scalar value as the row player's primal program.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, yao-style lower bounds preview is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

The value is useful because it is a guarantee, not a hope about average-case behavior.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using yao-style lower bounds preview responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Check the assumptions: finite action sets or the right compactness, convexity, and continuity conditions for extensions.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Yao-style lower bounds preview gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 6.5 robust policy learning

Robust policy learning belongs to the canonical scope of Minimax Theorem. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is zero-sum matrix games, maximin and minimax values, saddle points, LP duality, no-regret approximation, and robust AI objectives. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$v^+ = \min_{\mathbf{q}\in\Delta_n}\max_{\mathbf{p}\in\Delta_m}\mathbf{p}^\top A\mathbf{q}.$$

The formula gives the mathematical handle for robust policy learning. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

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

Three examples of robust policy learning:

1. Image perturbations bounded by a norm.
2. Prompt transformations allowed by a jailbreak policy.
3. Retrieval poisoning constrained by an index-insertion budget.

Two non-examples clarify the boundary:

1. Any attack the modeler can imagine but has not formalized.
2. Random corruption treated as adaptive attack.

Proof or verification habit for robust policy learning:

The nested objective is proved meaningful only after the feasible attack set is stated. The inner maximum is over that set, not over all possible bad events.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, robust policy learning is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Adversarial training improves robustness to the modeled threat, not to every strategic behavior.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using robust policy learning responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Write the set $\mathcal{S}$ before writing the max.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Robust policy learning gives the language to reason about that pressure.

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

1. (*) Work through a game-theory task for minimax theorem.
   - (a) State the players, actions, and payoffs.
   - (b) Compute or characterize best responses.
   - (c) Decide whether the proposed joint strategy is stable.
   - (d) Interpret the result for an AI, LLM, or adversarial system.

2. (*) Work through a game-theory task for minimax theorem.
   - (a) State the players, actions, and payoffs.
   - (b) Compute or characterize best responses.
   - (c) Decide whether the proposed joint strategy is stable.
   - (d) Interpret the result for an AI, LLM, or adversarial system.

3. (*) Work through a game-theory task for minimax theorem.
   - (a) State the players, actions, and payoffs.
   - (b) Compute or characterize best responses.
   - (c) Decide whether the proposed joint strategy is stable.
   - (d) Interpret the result for an AI, LLM, or adversarial system.

4. (**) Work through a game-theory task for minimax theorem.
   - (a) State the players, actions, and payoffs.
   - (b) Compute or characterize best responses.
   - (c) Decide whether the proposed joint strategy is stable.
   - (d) Interpret the result for an AI, LLM, or adversarial system.

5. (**) Work through a game-theory task for minimax theorem.
   - (a) State the players, actions, and payoffs.
   - (b) Compute or characterize best responses.
   - (c) Decide whether the proposed joint strategy is stable.
   - (d) Interpret the result for an AI, LLM, or adversarial system.

6. (**) Work through a game-theory task for minimax theorem.
   - (a) State the players, actions, and payoffs.
   - (b) Compute or characterize best responses.
   - (c) Decide whether the proposed joint strategy is stable.
   - (d) Interpret the result for an AI, LLM, or adversarial system.

7. (***) Work through a game-theory task for minimax theorem.
   - (a) State the players, actions, and payoffs.
   - (b) Compute or characterize best responses.
   - (c) Decide whether the proposed joint strategy is stable.
   - (d) Interpret the result for an AI, LLM, or adversarial system.

8. (***) Work through a game-theory task for minimax theorem.
   - (a) State the players, actions, and payoffs.
   - (b) Compute or characterize best responses.
   - (c) Decide whether the proposed joint strategy is stable.
   - (d) Interpret the result for an AI, LLM, or adversarial system.

9. (***) Work through a game-theory task for minimax theorem.
   - (a) State the players, actions, and payoffs.
   - (b) Compute or characterize best responses.
   - (c) Decide whether the proposed joint strategy is stable.
   - (d) Interpret the result for an AI, LLM, or adversarial system.

10. (***) Work through a game-theory task for minimax theorem.
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

Minimax Theorem follows causal inference because interventions often change incentives. Chapter 22 asks what changes when an action is taken. Chapter 23 asks what happens when other agents see that action, learn from it, and respond strategically.

The backward bridge is intervention. A policy change can have a causal effect, but if users or attackers adapt, the effect becomes part of a game. The forward bridge is measure theory: later probability foundations make the stochastic strategies, repeated games, and distributional assumptions more rigorous.

```text
+--------------------------------------------------------------+
| Chapter 22: intervention and causal mechanisms               |
| Chapter 23: strategic adaptation and adversarial objectives   |
| Chapter 24: rigorous probability and measure foundations      |
+--------------------------------------------------------------+
```

## References

- von Neumann and Morgenstern. Theory of Games and Economic Behavior. https://press.princeton.edu/books/paperback/9780691130613/theory-of-games-and-economic-behavior
- Nisan et al.. Algorithmic Game Theory. https://doi.org/10.1017/CBO9780511800481
- Goodfellow et al.. Generative Adversarial Nets. https://arxiv.org/abs/1406.2661
- Madry et al.. Towards Deep Learning Models Resistant to Adversarial Attacks. https://arxiv.org/abs/1706.06083
