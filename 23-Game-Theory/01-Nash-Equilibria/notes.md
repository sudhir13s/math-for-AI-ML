[Back to Curriculum](../../README.md) | [Previous: Causal Discovery](../../22-Causal-Inference/04-Causal-Discovery/notes.md) | [Next: Minimax Theorem](../02-Minimax-Theorem/notes.md)

---

# Nash Equilibria

> _"Equilibrium is not harmony; it is the absence of profitable unilateral change."_

## Overview

Nash equilibrium formalizes strategic stability: each agent's policy is optimal against the policies of the others.

Game theory is the part of the curriculum that studies adaptive decision makers. It asks what happens when each model, user, attacker, defender, or agent optimizes while anticipating the choices of others.

This section is written in LaTeX Markdown. Inline mathematics uses `$...$`, and display equations use `$$...$$`. The notes emphasize strategy, payoff, best response, equilibrium, exploitability, and adversarial adaptation.

## Prerequisites

- [Optimization Landscape](../../08-Optimization/06-Optimization-Landscape/notes.md)
- [Probability Measure Spaces Preview](../../06-Probability-Theory/03-Joint-Distributions/notes.md)
- [Causal Discovery](../../22-Causal-Inference/04-Causal-Discovery/notes.md)
- [Generative Models](../../14-Math-for-Specific-Models/05-Generative-Models/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Executable demonstrations for nash equilibria |
| [exercises.ipynb](exercises.ipynb) | Graded practice for nash equilibria |

## Learning Objectives

After completing this section, you will be able to:

- Define finite normal-form games with players, actions, payoffs, and information assumptions
- Compute best-response correspondences from a payoff table
- Identify pure Nash equilibria and distinguish them from Pareto-optimal outcomes
- Derive mixed equilibria in two-action games using the indifference principle
- Explain why randomization changes strategic stability in games with no pure equilibrium
- State the finite-game Nash existence theorem and the fixed-point idea behind it
- Recognize when support enumeration or Lemke-Howson style methods are appropriate
- Connect Nash reasoning to GANs, self-play, routing markets, and LLM tool agents
- Measure equilibrium failure using unilateral deviations and exploitability
- Separate strategic stability from social welfare, fairness, and safety

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 strategic stability](#11-strategic-stability)
  - [1.2 best responses](#12-best-responses)
  - [1.3 prisoner's dilemma and coordination](#13-prisoners-dilemma-and-coordination)
  - [1.4 equilibrium vs optimum](#14-equilibrium-vs-optimum)
  - [1.5 why Nash matters for GANs and agents](#15-why-nash-matters-for-gans-and-agents)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 normal-form game](#21-normalform-game)
  - [2.2 players actions and payoffs](#22-players-actions-and-payoffs)
  - [2.3 pure strategy](#23-pure-strategy)
  - [2.4 mixed strategy $\boldsymbol{\pi}_i$](#24-mixed-strategy)
  - [2.5 Nash equilibrium](#25-nash-equilibrium)
- [3. Pure-Strategy Equilibria](#3-purestrategy-equilibria)
  - [3.1 best-response tables](#31-bestresponse-tables)
  - [3.2 dominant strategies](#32-dominant-strategies)
  - [3.3 coordination games](#33-coordination-games)
  - [3.4 Pareto inefficiency](#34-pareto-inefficiency)
  - [3.5 no-pure-equilibrium examples](#35-nopureequilibrium-examples)
- [4. Mixed-Strategy Equilibria](#4-mixedstrategy-equilibria)
  - [4.1 probability simplex](#41-probability-simplex)
  - [4.2 indifference principle](#42-indifference-principle)
  - [4.3 matching pennies](#43-matching-pennies)
  - [4.4 support enumeration preview](#44-support-enumeration-preview)
  - [4.5 entropy and stochastic policies](#45-entropy-and-stochastic-policies)
- [5. Existence and Computation](#5-existence-and-computation)
  - [5.1 Nash existence theorem](#51-nash-existence-theorem)
  - [5.2 fixed-point intuition](#52-fixedpoint-intuition)
  - [5.3 Lemke-Howson preview](#53-lemkehowson-preview)
  - [5.4 correlated equilibrium preview](#54-correlated-equilibrium-preview)
  - [5.5 computational hardness](#55-computational-hardness)
- [6. AI Applications](#6-ai-applications)
  - [6.1 GAN training dynamics](#61-gan-training-dynamics)
  - [6.2 self-play](#62-selfplay)
  - [6.3 model routing competition](#63-model-routing-competition)
  - [6.4 tool-use equilibria](#64-tooluse-equilibria)
  - [6.5 equilibrium failure in nonstationary systems](#65-equilibrium-failure-in-nonstationary-systems)
- [7. Common Mistakes](#7-common-mistakes)
- [8. Exercises](#8-exercises)
- [9. Why This Matters for AI](#9-why-this-matters-for-ai)
- [10. Conceptual Bridge](#10-conceptual-bridge)
- [References](#references)

---

## 1. Intuition

Intuition develops the part of nash equilibria specified by the approved Chapter 23 table of contents. The treatment is game-theoretic, not merely an optimization recipe.

### 1.1 strategic stability

Strategic stability belongs to the canonical scope of Nash Equilibria. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is normal-form games, pure and mixed strategies, best responses, Nash equilibria, existence, computation, and AI equilibrium failures. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$G=(N,(A_i)_{i\in N},(u_i)_{i\in N}).$$

The formula gives the mathematical handle for strategic stability. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

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

Three examples of strategic stability:

1. A guardrail remains effective after attackers see examples of blocked prompts.
2. A model-routing policy remains attractive after providers update prices.
3. A self-play policy cannot be easily exploited by a newly trained opponent.

Two non-examples clarify the boundary:

1. A high average score on a fixed dataset.
2. A local minimum of one model's loss with no opponent.

Proof or verification habit for strategic stability:

The verification habit is adversarial: search for profitable deviations rather than only confirming the proposed behavior works in the original scenario.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, strategic stability is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

This is the mathematical shift from offline ML to deployed AI systems where users, competitors, and automated attacks learn from the model.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using strategic stability responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: State the adaptation channel: what can the other side observe, change, and optimize?

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Strategic stability gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 1.2 best responses

Best responses belongs to the canonical scope of Nash Equilibria. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is normal-form games, pure and mixed strategies, best responses, Nash equilibria, existence, computation, and AI equilibrium failures. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$u_i(a_i,a_{-i}) \ge u_i(a_i',a_{-i}) \quad \forall a_i'\in A_i.$$

The formula gives the mathematical handle for best responses. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

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

Three examples of best responses:

1. A discriminator chooses the classifier update that most separates generated and real samples.
2. An attacker chooses the prompt family with highest bypass rate against a fixed guardrail.
3. A retrieval system chooses the route with highest utility against the current user distribution.

Two non-examples clarify the boundary:

1. The globally highest payoff cell when the opponent is not fixed.
2. A socially preferred action that is not payoff-maximizing for the player.

Proof or verification habit for best responses:

To prove a response is best, compare it to every allowed unilateral deviation under the same opponent strategy.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, best responses is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Best-response thinking is how exploitability is measured: ask what an adaptive user, attacker, or agent could gain by switching strategy alone.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using best responses responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Never call an outcome stable until every player has passed the same best-response check.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Best responses gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 1.3 prisoner's dilemma and coordination

Prisoner's dilemma and coordination belongs to the canonical scope of Nash Equilibria. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is normal-form games, pure and mixed strategies, best responses, Nash equilibria, existence, computation, and AI equilibrium failures. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\boldsymbol{\pi}_i\in\Delta(A_i),\qquad \sum_{a_i\in A_i}\pi_i(a_i)=1.$$

The formula gives the mathematical handle for prisoner's dilemma and coordination. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

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

Three examples of prisoner's dilemma and coordination:

1. LLM agents agree on a tool-call schema.
2. Distributed learners converge to a shared convention for labels.
3. A team of models selects the same plan representation before acting.

Two non-examples clarify the boundary:

1. A zero-sum contest where one player's gain is the other's loss.
2. A single model choosing a format without another agent needing to match it.

Proof or verification habit for prisoner's dilemma and coordination:

Equilibrium verification is easy; equilibrium selection is the hard part. Show each matched profile is stable, then analyze basins, signals, or welfare.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, prisoner's dilemma and coordination is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Coordination failures are common in agentic systems because technically correct local policies can still fail to align interfaces.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using prisoner's dilemma and coordination responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Ask whether agents need the same convention, and whether the convention is observable before action.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Prisoner's dilemma and coordination gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 1.4 equilibrium vs optimum

Equilibrium vs optimum belongs to the canonical scope of Nash Equilibria. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is normal-form games, pure and mixed strategies, best responses, Nash equilibria, existence, computation, and AI equilibrium failures. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$u_i(\boldsymbol{\pi}_i^*,\boldsymbol{\pi}_{-i}^*) \ge u_i(\boldsymbol{\pi}_i,\boldsymbol{\pi}_{-i}^*) \quad \forall \boldsymbol{\pi}_i.$$

The formula gives the mathematical handle for equilibrium vs optimum. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

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

Three examples of equilibrium vs optimum:

1. A self-play policy pair where neither side has a profitable unilateral exploit.
2. A GAN fixed point where the generator distribution matches data and the discriminator cannot improve classification.
3. A routing market where no model provider benefits from changing only its bid.

Two non-examples clarify the boundary:

1. A high-welfare outcome with a profitable unilateral deviation.
2. A training checkpoint with low loss but a large best-response exploit.

Proof or verification habit for equilibrium vs optimum:

The proof is a universal deviation check: for each player $i$, hold $\pi_{-i}$ fixed and show $u_i(\pi_i^*,\pi_{-i}^*)\ge u_i(\pi_i,\pi_{-i}^*)$ for all allowed $\pi_i$.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, equilibrium vs optimum is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

For AI agents, Nash is a stability diagnostic. It does not guarantee safety, alignment, fairness, or global efficiency unless those objectives are encoded in the game.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using equilibrium vs optimum responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Ask: if one deployed model, user, or attacker changed behavior alone, would it gain?

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Equilibrium vs optimum gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 1.5 why Nash matters for GANs and agents

Why nash matters for gans and agents belongs to the canonical scope of Nash Equilibria. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is normal-form games, pure and mixed strategies, best responses, Nash equilibria, existence, computation, and AI equilibrium failures. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$G=(N,(A_i)_{i\in N},(u_i)_{i\in N}).$$

The formula gives the mathematical handle for why nash matters for gans and agents. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

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

Three examples of why nash matters for gans and agents:

1. A self-play policy pair where neither side has a profitable unilateral exploit.
2. A GAN fixed point where the generator distribution matches data and the discriminator cannot improve classification.
3. A routing market where no model provider benefits from changing only its bid.

Two non-examples clarify the boundary:

1. A high-welfare outcome with a profitable unilateral deviation.
2. A training checkpoint with low loss but a large best-response exploit.

Proof or verification habit for why nash matters for gans and agents:

The proof is a universal deviation check: for each player $i$, hold $\pi_{-i}$ fixed and show $u_i(\pi_i^*,\pi_{-i}^*)\ge u_i(\pi_i,\pi_{-i}^*)$ for all allowed $\pi_i$.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, why nash matters for gans and agents is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

For AI agents, Nash is a stability diagnostic. It does not guarantee safety, alignment, fairness, or global efficiency unless those objectives are encoded in the game.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using why nash matters for gans and agents responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Ask: if one deployed model, user, or attacker changed behavior alone, would it gain?

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Why nash matters for gans and agents gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

## 2. Formal Definitions

Formal Definitions develops the part of nash equilibria specified by the approved Chapter 23 table of contents. The treatment is game-theoretic, not merely an optimization recipe.

### 2.1 normal-form game

Normal-form game belongs to the canonical scope of Nash Equilibria. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is normal-form games, pure and mixed strategies, best responses, Nash equilibria, existence, computation, and AI equilibrium failures. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$u_i(a_i,a_{-i}) \ge u_i(a_i',a_{-i}) \quad \forall a_i'\in A_i.$$

The formula gives the mathematical handle for normal-form game. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

A normal-form game freezes timing and information into one payoff table. The table is useful when each player chooses an action once, or when a larger system can be locally summarized by a simultaneous strategic choice.

**Worked reading.**

For two players with actions $A_1=\{U,D\}$ and $A_2=\{L,R\}$, a payoff table assigns a pair $(u_1,u_2)$ to each of the four cells. A candidate outcome is one cell; a candidate mixed profile is a distribution over rows and columns.

Three examples of normal-form game:

1. A model router and a model provider choose prices simultaneously.
2. A generator and discriminator pick local update directions in one training round.
3. A defender and attacker pick a guardrail and a prompt class before observing the exact prompt.

Two non-examples clarify the boundary:

1. A single gradient step with no opponent.
2. A sequential decision process where the second mover observes the first action.

Proof or verification habit for normal-form game:

The proof habit is bookkeeping: verify the tuple contains all players, action sets, and payoff maps before asking about equilibrium.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, normal-form game is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Normal-form abstraction is the small table behind larger AI systems. It lets us ask which local incentives a training loop or deployment mechanism creates.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using normal-form game responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: If timing, hidden information, or state transitions matter, normal form is a model reduction rather than the full game.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Normal-form game gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 2.2 players actions and payoffs

Players actions and payoffs belongs to the canonical scope of Nash Equilibria. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is normal-form games, pure and mixed strategies, best responses, Nash equilibria, existence, computation, and AI equilibrium failures. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\boldsymbol{\pi}_i\in\Delta(A_i),\qquad \sum_{a_i\in A_i}\pi_i(a_i)=1.$$

The formula gives the mathematical handle for players actions and payoffs. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

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

Three examples of players actions and payoffs:

1. A row action chooses a defense, while a column action chooses an attack family.
2. An agent set lists every model or tool-using process that can affect reward.
3. A utility function converts accuracy, safety, latency, and cost into strategic incentives.

Two non-examples clarify the boundary:

1. A metric with no actor who optimizes it.
2. An action that is impossible in deployment but included for convenience.

Proof or verification habit for players actions and payoffs:

Before proving anything, audit the model specification: every allowed action must map to a payoff for every player.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, players actions and payoffs is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Payoff design is AI system design. The game will faithfully optimize the incentives it is given, including bad incentives.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using players actions and payoffs responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Can you name each player, enumerate or parameterize its actions, and compute its payoff from a joint action?

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Players actions and payoffs gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 2.3 pure strategy

Pure strategy belongs to the canonical scope of Nash Equilibria. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is normal-form games, pure and mixed strategies, best responses, Nash equilibria, existence, computation, and AI equilibrium failures. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$u_i(\boldsymbol{\pi}_i^*,\boldsymbol{\pi}_{-i}^*) \ge u_i(\boldsymbol{\pi}_i,\boldsymbol{\pi}_{-i}^*) \quad \forall \boldsymbol{\pi}_i.$$

The formula gives the mathematical handle for pure strategy. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

A pure strategy chooses one action deterministically. Best-response tables mark which pure actions are optimal against each opponent action.

**Worked reading.**

For every column, highlight the row entries with maximal row payoff; for every row, highlight the column entries with maximal column payoff. A cell highlighted for both players is a pure Nash equilibrium.

Three examples of pure strategy:

1. A deterministic guardrail mode.
2. A fixed model route.
3. A single action chosen by each player in a coordination game.

Two non-examples clarify the boundary:

1. A probability distribution over actions.
2. A randomized audit policy.

Proof or verification habit for pure strategy:

The proof is finite enumeration: compare every row within a column and every column within a row.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, pure strategy is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Pure-strategy analysis is the fastest sanity check before moving to mixed strategies or dynamic learning.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using pure strategy responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: If no cell is jointly best-response highlighted, search for mixed equilibria instead of forcing a pure one.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Pure strategy gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 2.4 mixed strategy $\boldsymbol{\pi}_i$

Mixed strategy $\boldsymbol{\pi}_i$ belongs to the canonical scope of Nash Equilibria. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is normal-form games, pure and mixed strategies, best responses, Nash equilibria, existence, computation, and AI equilibrium failures. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$G=(N,(A_i)_{i\in N},(u_i)_{i\in N}).$$

The formula gives the mathematical handle for mixed strategy $\boldsymbol{\pi}_i$. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

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

Three examples of mixed strategy $\boldsymbol{\pi}_i$:

1. Randomized audits that make attackers uncertain.
2. Stochastic decoding policies that prevent deterministic exploitation.
3. Exploration policies in self-play where pure repetition would be exploited.

Two non-examples clarify the boundary:

1. Adding noise after choosing a deterministic losing action.
2. A distribution that assigns probability to an action with strictly lower payoff while another supported action is better.

Proof or verification habit for mixed strategy $\boldsymbol{\pi}_i$:

Set expected payoffs of supported actions equal, solve for probabilities, then verify unsupported actions are not better.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, mixed strategy $\boldsymbol{\pi}_i$ is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Mixed strategies explain why robust systems often randomize: predictability can be a vulnerability when opponents adapt.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using mixed strategy $\boldsymbol{\pi}_i$ responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Check both support equality and off-support inequalities.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Mixed strategy $\boldsymbol{\pi}_i$ gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 2.5 Nash equilibrium

Nash equilibrium belongs to the canonical scope of Nash Equilibria. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is normal-form games, pure and mixed strategies, best responses, Nash equilibria, existence, computation, and AI equilibrium failures. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$u_i(a_i,a_{-i}) \ge u_i(a_i',a_{-i}) \quad \forall a_i'\in A_i.$$

The formula gives the mathematical handle for nash equilibrium. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

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

Three examples of nash equilibrium:

1. A self-play policy pair where neither side has a profitable unilateral exploit.
2. A GAN fixed point where the generator distribution matches data and the discriminator cannot improve classification.
3. A routing market where no model provider benefits from changing only its bid.

Two non-examples clarify the boundary:

1. A high-welfare outcome with a profitable unilateral deviation.
2. A training checkpoint with low loss but a large best-response exploit.

Proof or verification habit for nash equilibrium:

The proof is a universal deviation check: for each player $i$, hold $\pi_{-i}$ fixed and show $u_i(\pi_i^*,\pi_{-i}^*)\ge u_i(\pi_i,\pi_{-i}^*)$ for all allowed $\pi_i$.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, nash equilibrium is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

For AI agents, Nash is a stability diagnostic. It does not guarantee safety, alignment, fairness, or global efficiency unless those objectives are encoded in the game.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using nash equilibrium responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Ask: if one deployed model, user, or attacker changed behavior alone, would it gain?

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Nash equilibrium gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

## 3. Pure-Strategy Equilibria

Pure-Strategy Equilibria develops the part of nash equilibria specified by the approved Chapter 23 table of contents. The treatment is game-theoretic, not merely an optimization recipe.

### 3.1 best-response tables

Best-response tables belongs to the canonical scope of Nash Equilibria. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is normal-form games, pure and mixed strategies, best responses, Nash equilibria, existence, computation, and AI equilibrium failures. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\boldsymbol{\pi}_i\in\Delta(A_i),\qquad \sum_{a_i\in A_i}\pi_i(a_i)=1.$$

The formula gives the mathematical handle for best-response tables. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

A pure strategy chooses one action deterministically. Best-response tables mark which pure actions are optimal against each opponent action.

**Worked reading.**

For every column, highlight the row entries with maximal row payoff; for every row, highlight the column entries with maximal column payoff. A cell highlighted for both players is a pure Nash equilibrium.

Three examples of best-response tables:

1. A deterministic guardrail mode.
2. A fixed model route.
3. A single action chosen by each player in a coordination game.

Two non-examples clarify the boundary:

1. A probability distribution over actions.
2. A randomized audit policy.

Proof or verification habit for best-response tables:

The proof is finite enumeration: compare every row within a column and every column within a row.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, best-response tables is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Pure-strategy analysis is the fastest sanity check before moving to mixed strategies or dynamic learning.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using best-response tables responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: If no cell is jointly best-response highlighted, search for mixed equilibria instead of forcing a pure one.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Best-response tables gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 3.2 dominant strategies

Dominant strategies belongs to the canonical scope of Nash Equilibria. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is normal-form games, pure and mixed strategies, best responses, Nash equilibria, existence, computation, and AI equilibrium failures. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$u_i(\boldsymbol{\pi}_i^*,\boldsymbol{\pi}_{-i}^*) \ge u_i(\boldsymbol{\pi}_i,\boldsymbol{\pi}_{-i}^*) \quad \forall \boldsymbol{\pi}_i.$$

The formula gives the mathematical handle for dominant strategies. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

A dominant strategy is best regardless of the other players' actions. It is stronger than being a best response to one particular opponent strategy.

**Worked reading.**

If action $D$ gives player 1 at least as much payoff as action $C$ for every opponent action, and strictly more for one opponent action, then $D$ weakly dominates $C$.

Three examples of dominant strategies:

1. Always rejecting a malicious input class when the false-positive cost is explicitly lower than the exploit cost.
2. A bid strategy that wins under every competitor bid in a simplified auction.
3. A safe fallback tool that dominates risky tool use under every audited state.

Two non-examples clarify the boundary:

1. An action that is best only against the current opponent.
2. An action with higher average payoff but lower payoff in some opponent response.

Proof or verification habit for dominant strategies:

Prove dominance by comparing payoffs row-by-row or column-by-column across every opponent action.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, dominant strategies is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Dominance is rare in rich AI systems, but when present it simplifies analysis before searching for equilibria.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using dominant strategies responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Do not infer dominance from one cell or from expected payoff under one distribution.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Dominant strategies gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 3.3 coordination games

Coordination games belongs to the canonical scope of Nash Equilibria. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is normal-form games, pure and mixed strategies, best responses, Nash equilibria, existence, computation, and AI equilibrium failures. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$G=(N,(A_i)_{i\in N},(u_i)_{i\in N}).$$

The formula gives the mathematical handle for coordination games. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

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

Three examples of coordination games:

1. LLM agents agree on a tool-call schema.
2. Distributed learners converge to a shared convention for labels.
3. A team of models selects the same plan representation before acting.

Two non-examples clarify the boundary:

1. A zero-sum contest where one player's gain is the other's loss.
2. A single model choosing a format without another agent needing to match it.

Proof or verification habit for coordination games:

Equilibrium verification is easy; equilibrium selection is the hard part. Show each matched profile is stable, then analyze basins, signals, or welfare.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, coordination games is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Coordination failures are common in agentic systems because technically correct local policies can still fail to align interfaces.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using coordination games responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Ask whether agents need the same convention, and whether the convention is observable before action.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Coordination games gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 3.4 Pareto inefficiency

Pareto inefficiency belongs to the canonical scope of Nash Equilibria. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is normal-form games, pure and mixed strategies, best responses, Nash equilibria, existence, computation, and AI equilibrium failures. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$u_i(a_i,a_{-i}) \ge u_i(a_i',a_{-i}) \quad \forall a_i'\in A_i.$$

The formula gives the mathematical handle for pareto inefficiency. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

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

Three examples of pareto inefficiency:

1. Mutual cooperation in a prisoner's dilemma improves both players but may be unstable.
2. A routing policy raises total quality but gives one provider incentive to deviate.
3. A safety policy improves social welfare but reduces one actor's private payoff.

Two non-examples clarify the boundary:

1. A unilateral deviation check.
2. A fairness claim without specifying the social objective.

Proof or verification habit for pareto inefficiency:

Separate the two predicates: first test deviations for equilibrium, then compare feasible outcomes for welfare.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, pareto inefficiency is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

AI alignment often lives in the gap between private incentive and social objective, so this distinction is not philosophical decoration.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using pareto inefficiency responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: If an equilibrium is bad, changing incentives or constraints is usually required; wishing for cooperation is not a proof.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Pareto inefficiency gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 3.5 no-pure-equilibrium examples

No-pure-equilibrium examples belongs to the canonical scope of Nash Equilibria. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is normal-form games, pure and mixed strategies, best responses, Nash equilibria, existence, computation, and AI equilibrium failures. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\boldsymbol{\pi}_i\in\Delta(A_i),\qquad \sum_{a_i\in A_i}\pi_i(a_i)=1.$$

The formula gives the mathematical handle for no-pure-equilibrium examples. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

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

Three examples of no-pure-equilibrium examples:

1. A self-play policy pair where neither side has a profitable unilateral exploit.
2. A GAN fixed point where the generator distribution matches data and the discriminator cannot improve classification.
3. A routing market where no model provider benefits from changing only its bid.

Two non-examples clarify the boundary:

1. A high-welfare outcome with a profitable unilateral deviation.
2. A training checkpoint with low loss but a large best-response exploit.

Proof or verification habit for no-pure-equilibrium examples:

The proof is a universal deviation check: for each player $i$, hold $\pi_{-i}$ fixed and show $u_i(\pi_i^*,\pi_{-i}^*)\ge u_i(\pi_i,\pi_{-i}^*)$ for all allowed $\pi_i$.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, no-pure-equilibrium examples is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

For AI agents, Nash is a stability diagnostic. It does not guarantee safety, alignment, fairness, or global efficiency unless those objectives are encoded in the game.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using no-pure-equilibrium examples responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Ask: if one deployed model, user, or attacker changed behavior alone, would it gain?

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. No-pure-equilibrium examples gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

## 4. Mixed-Strategy Equilibria

Mixed-Strategy Equilibria develops the part of nash equilibria specified by the approved Chapter 23 table of contents. The treatment is game-theoretic, not merely an optimization recipe.

### 4.1 probability simplex

Probability simplex belongs to the canonical scope of Nash Equilibria. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is normal-form games, pure and mixed strategies, best responses, Nash equilibria, existence, computation, and AI equilibrium failures. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$u_i(\boldsymbol{\pi}_i^*,\boldsymbol{\pi}_{-i}^*) \ge u_i(\boldsymbol{\pi}_i,\boldsymbol{\pi}_{-i}^*) \quad \forall \boldsymbol{\pi}_i.$$

The formula gives the mathematical handle for probability simplex. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

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

Three examples of probability simplex:

1. Randomized audits that make attackers uncertain.
2. Stochastic decoding policies that prevent deterministic exploitation.
3. Exploration policies in self-play where pure repetition would be exploited.

Two non-examples clarify the boundary:

1. Adding noise after choosing a deterministic losing action.
2. A distribution that assigns probability to an action with strictly lower payoff while another supported action is better.

Proof or verification habit for probability simplex:

Set expected payoffs of supported actions equal, solve for probabilities, then verify unsupported actions are not better.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, probability simplex is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Mixed strategies explain why robust systems often randomize: predictability can be a vulnerability when opponents adapt.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using probability simplex responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Check both support equality and off-support inequalities.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Probability simplex gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 4.2 indifference principle

Indifference principle belongs to the canonical scope of Nash Equilibria. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is normal-form games, pure and mixed strategies, best responses, Nash equilibria, existence, computation, and AI equilibrium failures. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$G=(N,(A_i)_{i\in N},(u_i)_{i\in N}).$$

The formula gives the mathematical handle for indifference principle. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

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

Three examples of indifference principle:

1. Randomized audits that make attackers uncertain.
2. Stochastic decoding policies that prevent deterministic exploitation.
3. Exploration policies in self-play where pure repetition would be exploited.

Two non-examples clarify the boundary:

1. Adding noise after choosing a deterministic losing action.
2. A distribution that assigns probability to an action with strictly lower payoff while another supported action is better.

Proof or verification habit for indifference principle:

Set expected payoffs of supported actions equal, solve for probabilities, then verify unsupported actions are not better.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, indifference principle is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Mixed strategies explain why robust systems often randomize: predictability can be a vulnerability when opponents adapt.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using indifference principle responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Check both support equality and off-support inequalities.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Indifference principle gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 4.3 matching pennies

Matching pennies belongs to the canonical scope of Nash Equilibria. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is normal-form games, pure and mixed strategies, best responses, Nash equilibria, existence, computation, and AI equilibrium failures. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$u_i(a_i,a_{-i}) \ge u_i(a_i',a_{-i}) \quad \forall a_i'\in A_i.$$

The formula gives the mathematical handle for matching pennies. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

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

Three examples of matching pennies:

1. Randomized audits that make attackers uncertain.
2. Stochastic decoding policies that prevent deterministic exploitation.
3. Exploration policies in self-play where pure repetition would be exploited.

Two non-examples clarify the boundary:

1. Adding noise after choosing a deterministic losing action.
2. A distribution that assigns probability to an action with strictly lower payoff while another supported action is better.

Proof or verification habit for matching pennies:

Set expected payoffs of supported actions equal, solve for probabilities, then verify unsupported actions are not better.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, matching pennies is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Mixed strategies explain why robust systems often randomize: predictability can be a vulnerability when opponents adapt.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using matching pennies responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Check both support equality and off-support inequalities.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Matching pennies gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 4.4 support enumeration preview

Support enumeration preview belongs to the canonical scope of Nash Equilibria. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is normal-form games, pure and mixed strategies, best responses, Nash equilibria, existence, computation, and AI equilibrium failures. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\boldsymbol{\pi}_i\in\Delta(A_i),\qquad \sum_{a_i\in A_i}\pi_i(a_i)=1.$$

The formula gives the mathematical handle for support enumeration preview. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

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

Three examples of support enumeration preview:

1. Randomized audits that make attackers uncertain.
2. Stochastic decoding policies that prevent deterministic exploitation.
3. Exploration policies in self-play where pure repetition would be exploited.

Two non-examples clarify the boundary:

1. Adding noise after choosing a deterministic losing action.
2. A distribution that assigns probability to an action with strictly lower payoff while another supported action is better.

Proof or verification habit for support enumeration preview:

Set expected payoffs of supported actions equal, solve for probabilities, then verify unsupported actions are not better.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, support enumeration preview is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Mixed strategies explain why robust systems often randomize: predictability can be a vulnerability when opponents adapt.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using support enumeration preview responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Check both support equality and off-support inequalities.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Support enumeration preview gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 4.5 entropy and stochastic policies

Entropy and stochastic policies belongs to the canonical scope of Nash Equilibria. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is normal-form games, pure and mixed strategies, best responses, Nash equilibria, existence, computation, and AI equilibrium failures. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$u_i(\boldsymbol{\pi}_i^*,\boldsymbol{\pi}_{-i}^*) \ge u_i(\boldsymbol{\pi}_i,\boldsymbol{\pi}_{-i}^*) \quad \forall \boldsymbol{\pi}_i.$$

The formula gives the mathematical handle for entropy and stochastic policies. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Entropy measures how spread out a mixed strategy or stochastic policy is. Strategic entropy can make behavior less predictable to opponents.

**Worked reading.**

A deterministic policy on two actions has entropy $0$; a uniform policy has entropy $\log 2$. In a game, adding entropy changes both exploration and exploitability.

Three examples of entropy and stochastic policies:

1. Entropy-regularized self-play.
2. Randomized security audits.
3. Stochastic decoding that avoids always exposing the same response pattern.

Two non-examples clarify the boundary:

1. Noise added without considering payoffs.
2. Randomness that violates constraints.

Proof or verification habit for entropy and stochastic policies:

Check whether the entropy term appears in the payoff, the learning algorithm, or only in the modeler's description.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, entropy and stochastic policies is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Many AI policies are stochastic, but game theory asks whether that stochasticity improves strategic robustness or just hides deterministic weakness.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using entropy and stochastic policies responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Compare expected payoff and exploitability as entropy changes.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Entropy and stochastic policies gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

## 5. Existence and Computation

Existence and Computation develops the part of nash equilibria specified by the approved Chapter 23 table of contents. The treatment is game-theoretic, not merely an optimization recipe.

### 5.1 Nash existence theorem

Nash existence theorem belongs to the canonical scope of Nash Equilibria. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is normal-form games, pure and mixed strategies, best responses, Nash equilibria, existence, computation, and AI equilibrium failures. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$G=(N,(A_i)_{i\in N},(u_i)_{i\in N}).$$

The formula gives the mathematical handle for nash existence theorem. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

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

Three examples of nash existence theorem:

1. A self-play policy pair where neither side has a profitable unilateral exploit.
2. A GAN fixed point where the generator distribution matches data and the discriminator cannot improve classification.
3. A routing market where no model provider benefits from changing only its bid.

Two non-examples clarify the boundary:

1. A high-welfare outcome with a profitable unilateral deviation.
2. A training checkpoint with low loss but a large best-response exploit.

Proof or verification habit for nash existence theorem:

The proof is a universal deviation check: for each player $i$, hold $\pi_{-i}$ fixed and show $u_i(\pi_i^*,\pi_{-i}^*)\ge u_i(\pi_i,\pi_{-i}^*)$ for all allowed $\pi_i$.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, nash existence theorem is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

For AI agents, Nash is a stability diagnostic. It does not guarantee safety, alignment, fairness, or global efficiency unless those objectives are encoded in the game.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using nash existence theorem responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Ask: if one deployed model, user, or attacker changed behavior alone, would it gain?

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Nash existence theorem gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 5.2 fixed-point intuition

Fixed-point intuition belongs to the canonical scope of Nash Equilibria. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is normal-form games, pure and mixed strategies, best responses, Nash equilibria, existence, computation, and AI equilibrium failures. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$u_i(a_i,a_{-i}) \ge u_i(a_i',a_{-i}) \quad \forall a_i'\in A_i.$$

The formula gives the mathematical handle for fixed-point intuition. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Existence theorems show that under finite action sets and mixed strategies, at least one equilibrium exists even when no pure equilibrium exists.

**Worked reading.**

The best-response map sends a mixed profile to the set of best responses. Fixed-point theorems guarantee a profile that is consistent with its own best-response set.

Three examples of fixed-point intuition:

1. Matching pennies has no pure equilibrium but has a mixed equilibrium.
2. Rock-paper-scissors has the uniform mixed equilibrium.
3. Finite routing games can have mixed equilibria even when deterministic routing cycles.

Two non-examples clarify the boundary:

1. A convergence guarantee for gradient descent.
2. A guarantee that equilibrium is unique.

Proof or verification habit for fixed-point intuition:

The proof idea is not to enumerate all games. It builds a continuous or set-valued response map over compact convex simplices and applies a fixed-point theorem.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, fixed-point intuition is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

This is why equilibrium can be a mathematically well-defined target even when training dynamics are unstable.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using fixed-point intuition responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Remember: existence is not computation, and computation is not deployment safety.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Fixed-point intuition gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 5.3 Lemke-Howson preview

Lemke-howson preview belongs to the canonical scope of Nash Equilibria. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is normal-form games, pure and mixed strategies, best responses, Nash equilibria, existence, computation, and AI equilibrium failures. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\boldsymbol{\pi}_i\in\Delta(A_i),\qquad \sum_{a_i\in A_i}\pi_i(a_i)=1.$$

The formula gives the mathematical handle for lemke-howson preview. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Computing equilibria is an algorithmic problem with its own complexity, approximation, and representation issues.

**Worked reading.**

Support enumeration guesses which actions receive positive probability, solves indifference equations, and checks inequalities. Lemke-Howson gives a pivoting method for two-player games. Correlated equilibrium enlarges the solution concept using signals.

Three examples of lemke-howson preview:

1. A small two-player game solved by support enumeration.
2. A traffic-routing game where correlated signals improve coordination.
3. A large multi-agent benchmark where exact Nash search is computationally unrealistic.

Two non-examples clarify the boundary:

1. Assuming an equilibrium solver scales because the theorem says an equilibrium exists.
2. Confusing a correlated equilibrium with independent mixed strategies.

Proof or verification habit for lemke-howson preview:

The proof obligation moves from math existence to certificate checking: payoffs, supports, probabilities, and deviation inequalities must all be validated.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, lemke-howson preview is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

AI systems often use approximate or learned equilibria because exact game solving is too expensive at model scale.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using lemke-howson preview responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Always report the approximation notion and residual deviation gain.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Lemke-howson preview gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 5.4 correlated equilibrium preview

Correlated equilibrium preview belongs to the canonical scope of Nash Equilibria. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is normal-form games, pure and mixed strategies, best responses, Nash equilibria, existence, computation, and AI equilibrium failures. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$u_i(\boldsymbol{\pi}_i^*,\boldsymbol{\pi}_{-i}^*) \ge u_i(\boldsymbol{\pi}_i,\boldsymbol{\pi}_{-i}^*) \quad \forall \boldsymbol{\pi}_i.$$

The formula gives the mathematical handle for correlated equilibrium preview. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

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

Three examples of correlated equilibrium preview:

1. A self-play policy pair where neither side has a profitable unilateral exploit.
2. A GAN fixed point where the generator distribution matches data and the discriminator cannot improve classification.
3. A routing market where no model provider benefits from changing only its bid.

Two non-examples clarify the boundary:

1. A high-welfare outcome with a profitable unilateral deviation.
2. A training checkpoint with low loss but a large best-response exploit.

Proof or verification habit for correlated equilibrium preview:

The proof is a universal deviation check: for each player $i$, hold $\pi_{-i}$ fixed and show $u_i(\pi_i^*,\pi_{-i}^*)\ge u_i(\pi_i,\pi_{-i}^*)$ for all allowed $\pi_i$.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, correlated equilibrium preview is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

For AI agents, Nash is a stability diagnostic. It does not guarantee safety, alignment, fairness, or global efficiency unless those objectives are encoded in the game.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using correlated equilibrium preview responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Ask: if one deployed model, user, or attacker changed behavior alone, would it gain?

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Correlated equilibrium preview gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 5.5 computational hardness

Computational hardness belongs to the canonical scope of Nash Equilibria. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is normal-form games, pure and mixed strategies, best responses, Nash equilibria, existence, computation, and AI equilibrium failures. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$G=(N,(A_i)_{i\in N},(u_i)_{i\in N}).$$

The formula gives the mathematical handle for computational hardness. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

| Game object | Meaning | AI interpretation |
| --- | --- | --- |
| Player | Decision maker with an objective | Model, user, attacker, defender, generator, evaluator, tool-using agent |
| Action | Choice available to a player | Prompt, route, attack, defense, bid, policy update, generated sample |
| Strategy | Rule or distribution over actions | Stochastic policy, decoding policy, defense randomization, routing policy |
| Payoff | Utility or negative loss | Accuracy, reward, cost, safety score, exploitability, compute budget |
| Equilibrium | Stable joint behavior | No agent can improve by changing alone under the stated game |

**Operational definition.**

Computing equilibria is an algorithmic problem with its own complexity, approximation, and representation issues.

**Worked reading.**

Support enumeration guesses which actions receive positive probability, solves indifference equations, and checks inequalities. Lemke-Howson gives a pivoting method for two-player games. Correlated equilibrium enlarges the solution concept using signals.

Three examples of computational hardness:

1. A small two-player game solved by support enumeration.
2. A traffic-routing game where correlated signals improve coordination.
3. A large multi-agent benchmark where exact Nash search is computationally unrealistic.

Two non-examples clarify the boundary:

1. Assuming an equilibrium solver scales because the theorem says an equilibrium exists.
2. Confusing a correlated equilibrium with independent mixed strategies.

Proof or verification habit for computational hardness:

The proof obligation moves from math existence to certificate checking: payoffs, supports, probabilities, and deviation inequalities must all be validated.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, computational hardness is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

AI systems often use approximate or learned equilibria because exact game solving is too expensive at model scale.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using computational hardness responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Always report the approximation notion and residual deviation gain.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Computational hardness gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

## 6. AI Applications

AI Applications develops the part of nash equilibria specified by the approved Chapter 23 table of contents. The treatment is game-theoretic, not merely an optimization recipe.

### 6.1 GAN training dynamics

Gan training dynamics belongs to the canonical scope of Nash Equilibria. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is normal-form games, pure and mixed strategies, best responses, Nash equilibria, existence, computation, and AI equilibrium failures. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$u_i(a_i,a_{-i}) \ge u_i(a_i',a_{-i}) \quad \forall a_i'\in A_i.$$

The formula gives the mathematical handle for gan training dynamics. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

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

Three examples of gan training dynamics:

1. GAN generator-discriminator training.
2. Jailbreak discovery against a deployed policy layer.
3. Benchmark gaming where systems optimize for the public metric instead of the intended task.

Two non-examples clarify the boundary:

1. One-time evaluation on a frozen hidden test set.
2. A content filter measured only against historical prompts.

Proof or verification habit for gan training dynamics:

The mathematical proof obligation is to identify the adaptive loop and the payoff each side optimizes.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, gan training dynamics is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Many LLM safety and evaluation failures are game failures: optimizing the metric changes the population of attempts.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using gan training dynamics responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Ask who can observe the metric, adapt to it, and benefit from adaptation.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Gan training dynamics gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 6.2 self-play

Self-play belongs to the canonical scope of Nash Equilibria. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is normal-form games, pure and mixed strategies, best responses, Nash equilibria, existence, computation, and AI equilibrium failures. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$\boldsymbol{\pi}_i\in\Delta(A_i),\qquad \sum_{a_i\in A_i}\pi_i(a_i)=1.$$

The formula gives the mathematical handle for self-play. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

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

Three examples of self-play:

1. Self-play agents improving by training against earlier or current versions.
2. LLM tool agents changing each other's context and options.
3. A routing marketplace where traffic shifts after one provider changes quality.

Two non-examples clarify the boundary:

1. A single-agent RL problem with a fixed transition kernel.
2. Batch supervised learning on immutable labels.

Proof or verification habit for self-play:

Analyze the joint policy trajectory, not only individual losses.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, self-play is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Agentic LLM systems make multi-agent math practical: prompts, tools, memory, and policies interact in a shared state.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using self-play responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Ask which part of another agent's behavior enters this agent's observation or reward.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Self-play gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 6.3 model routing competition

Model routing competition belongs to the canonical scope of Nash Equilibria. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is normal-form games, pure and mixed strategies, best responses, Nash equilibria, existence, computation, and AI equilibrium failures. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$u_i(\boldsymbol{\pi}_i^*,\boldsymbol{\pi}_{-i}^*) \ge u_i(\boldsymbol{\pi}_i,\boldsymbol{\pi}_{-i}^*) \quad \forall \boldsymbol{\pi}_i.$$

The formula gives the mathematical handle for model routing competition. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

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

Three examples of model routing competition:

1. GAN generator-discriminator training.
2. Jailbreak discovery against a deployed policy layer.
3. Benchmark gaming where systems optimize for the public metric instead of the intended task.

Two non-examples clarify the boundary:

1. One-time evaluation on a frozen hidden test set.
2. A content filter measured only against historical prompts.

Proof or verification habit for model routing competition:

The mathematical proof obligation is to identify the adaptive loop and the payoff each side optimizes.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, model routing competition is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Many LLM safety and evaluation failures are game failures: optimizing the metric changes the population of attempts.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using model routing competition responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Ask who can observe the metric, adapt to it, and benefit from adaptation.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Model routing competition gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 6.4 tool-use equilibria

Tool-use equilibria belongs to the canonical scope of Nash Equilibria. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is normal-form games, pure and mixed strategies, best responses, Nash equilibria, existence, computation, and AI equilibrium failures. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$G=(N,(A_i)_{i\in N},(u_i)_{i\in N}).$$

The formula gives the mathematical handle for tool-use equilibria. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

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

Three examples of tool-use equilibria:

1. GAN generator-discriminator training.
2. Jailbreak discovery against a deployed policy layer.
3. Benchmark gaming where systems optimize for the public metric instead of the intended task.

Two non-examples clarify the boundary:

1. One-time evaluation on a frozen hidden test set.
2. A content filter measured only against historical prompts.

Proof or verification habit for tool-use equilibria:

The mathematical proof obligation is to identify the adaptive loop and the payoff each side optimizes.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, tool-use equilibria is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

Many LLM safety and evaluation failures are game failures: optimizing the metric changes the population of attempts.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using tool-use equilibria responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Ask who can observe the metric, adapt to it, and benefit from adaptation.

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Tool-use equilibria gives the language to reason about that pressure.

A final diagnostic question is whether a decision remains good after another agent learns from it. If not, the analysis needs game theory, not just prediction, causality, or optimization.

| Diagnostic question | Game-theoretic discipline it tests |
| --- | --- |
| Who can respond? | Player modeling |
| What can they change? | Action space |
| What do they want? | Payoff design |
| Can one side commit first? | Stackelberg structure |
| Is the worst case important? | Minimax or robust objective |

### 6.5 equilibrium failure in nonstationary systems

Equilibrium failure in nonstationary systems belongs to the canonical scope of Nash Equilibria. The central object is not a single optimizer but a system of decision makers whose objectives interact.

For this subsection, the working scope is normal-form games, pure and mixed strategies, best responses, Nash equilibria, existence, computation, and AI equilibrium failures. We use players, action sets, strategies, payoffs, and response rules. The key question is whether a proposed behavior is stable when another agent adapts.

$$u_i(a_i,a_{-i}) \ge u_i(a_i',a_{-i}) \quad \forall a_i'\in A_i.$$

The formula gives the mathematical handle for equilibrium failure in nonstationary systems. In game theory, this expression should always be read with the opponent's decision rule in mind. A policy that is optimal in isolation may be exploitable once another player observes and responds to it.

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

Three examples of equilibrium failure in nonstationary systems:

1. A self-play policy pair where neither side has a profitable unilateral exploit.
2. A GAN fixed point where the generator distribution matches data and the discriminator cannot improve classification.
3. A routing market where no model provider benefits from changing only its bid.

Two non-examples clarify the boundary:

1. A high-welfare outcome with a profitable unilateral deviation.
2. A training checkpoint with low loss but a large best-response exploit.

Proof or verification habit for equilibrium failure in nonstationary systems:

The proof is a universal deviation check: for each player $i$, hold $\pi_{-i}$ fixed and show $u_i(\pi_i^*,\pi_{-i}^*)\ge u_i(\pi_i,\pi_{-i}^*)$ for all allowed $\pi_i$.

```text
single-agent optimization:    choose theta to minimize L(theta)
game-theoretic optimization:  choose pi_i while others choose pi_-i
adversarial objective:        choose defense against best attack
multi-agent learning:         policies change the environment itself
```

In AI systems, equilibrium failure in nonstationary systems is useful because modern models are deployed into adaptive environments: users learn prompt tricks, attackers search for failures, evaluators change rubrics, and other agents compete for resources.

For AI agents, Nash is a stability diagnostic. It does not guarantee safety, alignment, fairness, or global efficiency unless those objectives are encoded in the game.

Notebook implementation will use small synthetic payoff matrices and learning dynamics. This keeps the mathematics executable while avoiding external datasets or heavyweight game solvers.

Checklist for using equilibrium failure in nonstationary systems responsibly:

- State the players and their objectives.
- State the action spaces and information structure.
- Decide whether the game is zero-sum, general-sum, cooperative, or adversarial.
- Identify pure, mixed, or policy strategies.
- Compute best responses or exploitability before claiming stability.
- Separate equilibrium analysis from welfare analysis.
- Explain what changes if opponents adapt.

Local diagnostic: Ask: if one deployed model, user, or attacker changed behavior alone, would it gain?

This chapter follows Chapter 22 by adding strategic adaptation. Causal inference asks what happens when we intervene. Game theory asks what happens when other decision makers anticipate or respond to that intervention.

Modern AI makes the distinction practical. A deployed model can be optimized against by users, attackers, competitors, automated evaluators, and other models. Equilibrium failure in nonstationary systems gives the language to reason about that pressure.

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

1. (*) Work through a game-theory task for nash equilibria.
   - (a) State the players, actions, and payoffs.
   - (b) Compute or characterize best responses.
   - (c) Decide whether the proposed joint strategy is stable.
   - (d) Interpret the result for an AI, LLM, or adversarial system.

2. (*) Work through a game-theory task for nash equilibria.
   - (a) State the players, actions, and payoffs.
   - (b) Compute or characterize best responses.
   - (c) Decide whether the proposed joint strategy is stable.
   - (d) Interpret the result for an AI, LLM, or adversarial system.

3. (*) Work through a game-theory task for nash equilibria.
   - (a) State the players, actions, and payoffs.
   - (b) Compute or characterize best responses.
   - (c) Decide whether the proposed joint strategy is stable.
   - (d) Interpret the result for an AI, LLM, or adversarial system.

4. (**) Work through a game-theory task for nash equilibria.
   - (a) State the players, actions, and payoffs.
   - (b) Compute or characterize best responses.
   - (c) Decide whether the proposed joint strategy is stable.
   - (d) Interpret the result for an AI, LLM, or adversarial system.

5. (**) Work through a game-theory task for nash equilibria.
   - (a) State the players, actions, and payoffs.
   - (b) Compute or characterize best responses.
   - (c) Decide whether the proposed joint strategy is stable.
   - (d) Interpret the result for an AI, LLM, or adversarial system.

6. (**) Work through a game-theory task for nash equilibria.
   - (a) State the players, actions, and payoffs.
   - (b) Compute or characterize best responses.
   - (c) Decide whether the proposed joint strategy is stable.
   - (d) Interpret the result for an AI, LLM, or adversarial system.

7. (***) Work through a game-theory task for nash equilibria.
   - (a) State the players, actions, and payoffs.
   - (b) Compute or characterize best responses.
   - (c) Decide whether the proposed joint strategy is stable.
   - (d) Interpret the result for an AI, LLM, or adversarial system.

8. (***) Work through a game-theory task for nash equilibria.
   - (a) State the players, actions, and payoffs.
   - (b) Compute or characterize best responses.
   - (c) Decide whether the proposed joint strategy is stable.
   - (d) Interpret the result for an AI, LLM, or adversarial system.

9. (***) Work through a game-theory task for nash equilibria.
   - (a) State the players, actions, and payoffs.
   - (b) Compute or characterize best responses.
   - (c) Decide whether the proposed joint strategy is stable.
   - (d) Interpret the result for an AI, LLM, or adversarial system.

10. (***) Work through a game-theory task for nash equilibria.
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

Nash Equilibria follows causal inference because interventions often change incentives. Chapter 22 asks what changes when an action is taken. Chapter 23 asks what happens when other agents see that action, learn from it, and respond strategically.

The backward bridge is intervention. A policy change can have a causal effect, but if users or attackers adapt, the effect becomes part of a game. The forward bridge is measure theory: later probability foundations make the stochastic strategies, repeated games, and distributional assumptions more rigorous.

```text
+--------------------------------------------------------------+
| Chapter 22: intervention and causal mechanisms               |
| Chapter 23: strategic adaptation and adversarial objectives   |
| Chapter 24: rigorous probability and measure foundations      |
+--------------------------------------------------------------+
```

## References

- Nash. Equilibrium Points in N-Person Games. https://pmc.ncbi.nlm.nih.gov/articles/PMC1063129/
- Osborne and Rubinstein. A Course in Game Theory. https://mitpress.mit.edu/9780262650403/a-course-in-game-theory/
- Shoham and Leyton-Brown. Multiagent Systems. https://www.masfoundations.org/toc.pdf
- Goodfellow et al.. Generative Adversarial Nets. https://arxiv.org/abs/1406.2661
