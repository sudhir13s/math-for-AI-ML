[<- Back to Graph Theory](../README.md) | [Next: Graph Algorithms ->](../07-Graph-Algorithms/notes.md)

---

# Random Graphs

> _"In the random graph, order emerges from chaos - the giant component appears suddenly, like a phase transition in physics, when the average degree crosses one."_
> - Bela Bollobas

## Overview

Random graphs are probability distributions over graph-valued random variables. They provide the mathematical framework for reasoning about networks whose structure is not fully known or arises from stochastic processes - exactly the situation we face with the internet, social networks, protein interaction maps, and the implicit graphs inside neural networks.

This section develops four canonical random graph models - Erdos-Renyi $G(n,p)$, Watts-Strogatz small-world, Barabasi-Albert scale-free, and the Stochastic Block Model - from their definitions through their phase transitions, spectral properties, and connections to modern machine learning. We close with **graphons**, the measure-theoretic limit objects that unify all dense random graph models and form the mathematical foundation of graph neural network universality theory.

The central theme is **emergence**: how global structure (giant components, communities, power-law degree distributions) arises from simple local rules, and how this emergence can be detected, quantified, and exploited by ML algorithms.

## Prerequisites

- Probability theory: expectation, variance, concentration inequalities - [Probability Foundations](../../07-Probability-Statistics/01-Probability-Foundations/notes.md)
- Graph Laplacians and spectral graph theory - [Spectral Graph Theory](../04-Spectral-Graph-Theory/notes.md)
- Graph Neural Networks (for the ML applications) - [Graph Neural Networks](../05-Graph-Neural-Networks/notes.md)
- Basic combinatorics: binomial coefficients, Stirling's approximation

## Companion Notebooks

| Notebook                           | Description                                                                 |
| ---------------------------------- | --------------------------------------------------------------------------- |
| [theory.ipynb](theory.ipynb)       | Interactive simulations of all four models, phase transitions, spectral laws |
| [exercises.ipynb](exercises.ipynb) | 8 graded problems from threshold functions to graphon estimation             |

## Learning Objectives

After completing this section, you will:

1. Define and sample from $G(n,p)$, $G(n,m)$, Watts-Strogatz, Barabasi-Albert, and SBM
2. State the giant component phase transition theorem and prove the critical threshold $p = 1/n$
3. Derive the Poisson degree distribution of $G(n,p)$ in the sparse regime
4. Compute clustering coefficient and average path length for small-world graphs
5. Derive the power-law degree distribution via mean-field preferential attachment
6. State the Kesten-Stigum threshold for community detection in the SBM
7. Apply Wigner's semicircle law and Davis-Kahan to spectral clustering of random graphs
8. Define graphons, the cut metric, and explain how graphons arise as limits of dense graph sequences
9. Connect random graph theory to GNN benchmark design, graph generation, and LLM attention structure
10. Identify and fix the most common mistakes in applying random graph models

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 What Is a Random Graph?](#11-what-is-a-random-graph)
  - [1.2 Why Random Graphs Matter for AI](#12-why-random-graphs-matter-for-ai)
  - [1.3 The Four Canonical Models](#13-the-four-canonical-models)
  - [1.4 Historical Timeline: 1959-2026](#14-historical-timeline-19592026)
  - [1.5 Phase Transitions: The Central Metaphor](#15-phase-transitions-the-central-metaphor)
- [2. Probability on Graphs: Formal Setup](#2-probability-on-graphs-formal-setup)
  - [2.1 Graph Probability Spaces](#21-graph-probability-spaces)
  - [2.2 With High Probability (w.h.p.)](#22-with-high-probability-whp)
  - [2.3 Monotone Properties and Threshold Functions](#23-monotone-properties-and-threshold-functions)
  - [2.4 First and Second Moment Methods](#24-first-and-second-moment-methods)
- [3. Erdos-Renyi Model](#3-erdos-renyi-model)
  - [3.1 Definitions: G(n,p) and G(n,m)](#31-definitions-gnp-and-gnm)
  - [3.2 Degree Distribution: Poisson Limit](#32-degree-distribution-poisson-limit)
  - [3.3 Giant Component Phase Transition](#33-giant-component-phase-transition)
  - [3.4 Connectivity Threshold](#34-connectivity-threshold)
  - [3.5 Subgraph Counts and Triangle Thresholds](#35-subgraph-counts-and-triangle-thresholds)
  - [3.6 Diameter and Distances](#36-diameter-and-distances)
  - [3.7 Spectral Properties of G(n,p)](#37-spectral-properties-of-gnp)
- [4. Watts-Strogatz Small-World Model](#4-watts-strogatz-small-world-model)
  - [4.1 Construction and Parameters](#41-construction-and-parameters)
  - [4.2 Clustering Coefficient](#42-clustering-coefficient)
  - [4.3 Average Path Length](#43-average-path-length)
  - [4.4 The Small-World Regime](#44-the-small-world-regime)
  - [4.5 Navigability and Kleinberg's Grid](#45-navigability-and-kleinbergs-grid)
  - [4.6 Social Network Evidence](#46-social-network-evidence)
- [5. Barabasi-Albert Scale-Free Model](#5-barabasi-albert-scale-free-model)
  - [5.1 Preferential Attachment](#51-preferential-attachment)
  - [5.2 Power-Law Degree Distribution](#52-power-law-degree-distribution)
  - [5.3 Mean-Field Analysis](#53-mean-field-analysis)
  - [5.4 Robustness and Fragility](#54-robustness-and-fragility)
  - [5.5 Scale-Free Networks in the Wild](#55-scale-free-networks-in-the-wild)
- [6. Stochastic Block Model](#6-stochastic-block-model)
  - [6.1 Definition and Parameters](#61-definition-and-parameters)
  - [6.2 Community Detection: Information-Theoretic Limits](#62-community-detection-information-theoretic-limits)
  - [6.3 Degree-Corrected SBM](#63-degree-corrected-sbm)
  - [6.4 SBM as GNN Benchmark Generator](#64-sbm-as-gnn-benchmark-generator)
- [7. Spectral Properties of Random Graphs](#7-spectral-properties-of-random-graphs)
  - [7.1 Wigner's Semicircle Law](#71-wigners-semicircle-law)
  - [7.2 Spectral Gap and Community Detection](#72-spectral-gap-and-community-detection)
  - [7.3 Davis-Kahan Theorem and Perturbation Bounds](#73-davis-kahan-theorem-and-perturbation-bounds)
  - [7.4 Laplacian Spectrum of G(n,p)](#74-laplacian-spectrum-of-gnp)
- [8. Graphons: The Infinite-Size Limit](#8-graphons-the-infinite-size-limit)
  - [8.1 Dense Graph Sequences and the Cut Metric](#81-dense-graph-sequences-and-the-cut-metric)
  - [8.2 Graphon Definition and Sampling](#82-graphon-definition-and-sampling)
  - [8.3 Homomorphism Densities and Graph Parameters](#83-homomorphism-densities-and-graph-parameters)
  - [8.4 Graphon Neural Networks](#84-graphon-neural-networks)
- [9. Applications in Machine Learning](#9-applications-in-machine-learning)
  - [9.1 GraphWorld: Benchmark Generation from SBM](#91-graphworld-benchmark-generation-from-sbm)
  - [9.2 Graph Generation: GRAN, GDSS, DiGress](#92-graph-generation-gran-gdss-digress)
  - [9.3 LLM Attention as a Random Graph Process](#93-llm-attention-as-a-random-graph-process)
  - [9.4 Lottery Ticket Hypothesis and Sparse Subgraphs](#94-lottery-ticket-hypothesis-and-sparse-subgraphs)
  - [9.5 Social Network Analysis at Scale](#95-social-network-analysis-at-scale)
- [10. Common Mistakes](#10-common-mistakes)
- [11. Exercises](#11-exercises)
- [12. Why This Matters for AI (2026 Perspective)](#12-why-this-matters-for-ai-2026-perspective)
- [13. Conceptual Bridge](#13-conceptual-bridge)

---

## 1. Intuition

### 1.1 What Is a Random Graph?

A **random graph** is not a single graph but a *probability distribution* over graphs. When we write $G \sim G(n, p)$, we mean: take $n$ labeled vertices; independently include each of the $\binom{n}{2}$ possible edges with probability $p$. The specific graph $G$ is a sample from this distribution.

This is a powerful abstraction. Real-world networks - social graphs, citation networks, protein-protein interaction maps, the web - cannot be written down in closed form. They arise from complex processes involving millions of actors. Random graph models capture the *statistical regularity* of these networks without requiring knowledge of every individual edge.

**Key insight:** Many properties of real-world networks can be characterized by just a few parameters (average degree, clustering coefficient, community structure), and random graph models are the mathematical objects that interpolate between these summary statistics and the full combinatorial structure of the graph.

For AI and ML, random graphs matter in at least three ways:

1. **Benchmark generation**: We generate synthetic graphs from random models to test GNN algorithms across controlled conditions (the GraphWorld framework uses SBM).
2. **Graph generation models**: GRAN, GDSS, and DiGress learn to sample from distributions over graphs - essentially learning a graphon.
3. **Network analysis**: Training data graphs (social networks, citation networks, knowledge graphs) have structure that random graph models help characterize and exploit.

### 1.2 Why Random Graphs Matter for AI

The connection between random graphs and modern AI is deeper than benchmark generation:

**Transformer attention is a random graph.** In a language model with $n$ tokens and sparse attention, the attention pattern defines a random-looking bipartite graph. The connectivity of this graph determines information flow - whether distant tokens can influence each other and how quickly information propagates. Results from random graph theory (connectivity thresholds, small-world phenomena) directly predict when transformers can or cannot capture long-range dependencies.

**GNN expressiveness depends on graph structure.** The WL hierarchy and GIN's expressiveness guarantees depend on what graphs the model will see. If training graphs are sampled from an SBM, the GNN faces a specific community detection problem that has known information-theoretic limits - limits that bound what no GNN can achieve regardless of architecture.

**Overparameterized networks are sparse graphs.** The lottery ticket hypothesis says that dense networks contain sparse subnetworks (tickets) that train just as well. These sparse subnetworks have random-graph-like structure: they appear near the percolation threshold of the dense network.

**Graphons = graph neural network limits.** The mathematical theory of graphons (limit objects of dense graph sequences) is the same theory that gives GNNs their universality results. A GNN that is continuous in the cut metric topology of graphons is provably expressive on all graphs in the same graphon equivalence class.

### 1.3 The Four Canonical Models

```
THE FOUR CANONICAL RANDOM GRAPH MODELS
========================================================================

  Model           Parameters          Key property
  ---------------------------------------------------------------------
  Erdos-Renyi     n, p (or n, m)      Phase transition at p = 1/n
  G(n,p)          Independent edges   Poisson degree distribution
                                      Thresholds for all monotone props

  Watts-Strogatz  n, k, \beta             High clustering + short paths
  Small-World     Ring lattice +      Models social/neural networks
                  random rewiring     Navigability (Kleinberg)

  Barabasi-Albert n, m_0, m            Power-law degree distribution
  Scale-Free      Preferential        Hubs, robustness to random failure
                  attachment          Most real-world networks

  Stochastic      k, n_1,...,n_k, B     Community structure
  Block Model     Block membership    Kesten-Stigum threshold
  (SBM)           + probability matrix Benchmark for GNN node classif.

========================================================================
```

Each model captures one dominant feature of real-world networks:

- **ER** captures mathematical tractability and threshold phenomena
- **WS** captures the small-world property (short paths, high clustering)
- **BA** captures heterogeneity (a few highly-connected hubs, many low-degree nodes)
- **SBM** captures community structure (dense intra-community, sparse inter-community)

Real networks typically exhibit several of these properties simultaneously. Hybrid models (e.g., SBM with degree correction, or preferential attachment with community structure) better match empirical data.

### 1.4 Historical Timeline: 1959-2026

```
RANDOM GRAPH THEORY: 1959-2026
========================================================================

  1959  Erdos & Renyi introduce G(n,m); first random graph paper
  1960  Erdos & Renyi prove giant component phase transition
  1961  Erdos & Renyi prove connectivity threshold p = log(n)/n
  1984  Bollobas writes first comprehensive textbook on random graphs
  1995  Chung & Lu: random graphs with given expected degrees
  1998  Watts & Strogatz: small-world networks (Nature, ~36,000 cites)
  1999  Barabasi & Albert: scale-free networks (Science, ~50,000 cites)
  2001  Lovasz & Szegedy begin graphon theory (published 2006)
  2001  Girvan & Newman: modularity and community detection
  2007  Lovasz & Szegedy: limits of dense graph sequences (JCTB)
  2011  Mossel, Neeman, Sly: Kesten-Stigum threshold (stochastic proof)
  2014  Abbe & Sandon: exact recovery threshold for SBM
  2015  You, Ying, Leskovec: GraphRNN graph generation
  2018  Keriven & Peyre: graphon neural networks
  2020  GraphWorld: benchmark generation via SBM
  2022  Rusch, Bronstein, Mishra: gradient over-squashing theory
  2023  DiGress: discrete diffusion for graph generation
  2024  Graphon attention transformers (sparse attention limits)
  2026  Graphon-based GNN universality: completeness results

========================================================================
```

### 1.5 Phase Transitions: The Central Metaphor

The most striking feature of random graphs is the **phase transition**: a sudden, dramatic change in global structure as a parameter crosses a critical threshold. This is not merely a quantitative change but a qualitative one - the graph transitions from one phase (many small components) to another (one giant component plus small components).

**Physical analogy:** In ferromagnetism, a material transitions from disordered (many small magnetic domains) to ordered (one aligned domain) as temperature drops below the Curie point. The mathematics is identical: both are percolation phase transitions.

**The critical exponent phenomenon:** Near the critical point $p_c = 1/n$, the giant component has size $\Theta(n^{2/3})$ - a polynomial intermediate between the subcritical $O(\log n)$ and supercritical $\Theta(n)$ regimes. This $n^{2/3}$ scaling is a universal critical exponent that appears in many other combinatorial and physical phase transitions.

**For AI:** Phase transitions in random graphs correspond to phase transitions in learning. A GNN trained to detect community structure in SBMs undergoes a computational phase transition at the Kesten-Stigum threshold - above it, polynomial-time algorithms succeed; below it, no efficient algorithm can (conditional on computational hardness conjectures). This is the rigorous mathematical statement of why some graph learning problems are fundamentally hard.

---

## 2. Probability on Graphs: Formal Setup

### 2.1 Graph Probability Spaces

Let $\mathcal{G}_n$ denote the set of all labeled graphs on vertex set $[n] = \{1, 2, \ldots, n\}$. A **random graph model** is a probability measure $\mu$ on $\mathcal{G}_n$.

**Definition (Random Graph):** A random graph $G$ on $n$ vertices is a random variable taking values in $\mathcal{G}_n$, defined on some probability space $(\Omega, \mathcal{F}, \mathbb{P})$.

The key examples:

**$G(n,p)$** (Erdos-Renyi): Each edge $\{u,v\} \in \binom{[n]}{2}$ is included independently with probability $p$. The probability of a specific graph $g$ with $m$ edges is:
$$\mathbb{P}[G = g] = p^m (1-p)^{\binom{n}{2} - m}$$

**$G(n,m)$** (fixed-edge): Uniform over all graphs with exactly $m$ edges:
$$\mathbb{P}[G = g] = \binom{\binom{n}{2}}{m}^{-1} \cdot \mathbf{1}[|E(g)| = m]$$

**Equivalence:** $G(n,p)$ and $G(n,m)$ with $m = p\binom{n}{2}$ have the same asymptotic behavior for most properties. More precisely, $G(n,p)$ conditioned on having exactly $m$ edges is $G(n,m)$.

**Graph properties as events:** A *graph property* $\mathcal{P}$ is a set of graphs closed under isomorphism (it depends only on structure, not labeling). We study the event $\{G \in \mathcal{P}\}$ and its probability as $n \to \infty$.

### 2.2 With High Probability (w.h.p.)

**Definition (w.h.p.):** We say $G(n,p)$ satisfies property $\mathcal{P}$ **with high probability** (w.h.p.) if:
$$\lim_{n \to \infty} \mathbb{P}[G(n,p) \in \mathcal{P}] = 1$$

This is distinct from "always" - there will always be rare samples that violate the property. But in the limit, the measure of the failure set goes to zero.

**Notation conventions:**
- $f(n) = o(g(n))$: $f/g \to 0$
- $f(n) = \omega(g(n))$: $f/g \to \infty$  
- $f(n) = \Theta(g(n))$: $c_1 g \le f \le c_2 g$ for constants $c_1, c_2 > 0$
- $f(n) \sim g(n)$: $f/g \to 1$

**Example (Isolated vertices):** In $G(n,p)$ with $p = c/n$, the probability that vertex $v$ is isolated is $(1-p)^{n-1} \sim e^{-c}$. The expected number of isolated vertices is $n \cdot e^{-c}$. For $c = \ln n$, this expected number is $1$, and the actual number of isolated vertices is 0 w.h.p.

**Markov's inequality (First Moment Method):** If $X \ge 0$ and $\mathbb{E}[X] \to 0$, then $X = 0$ w.h.p. (since $\mathbb{P}[X \ge 1] \le \mathbb{E}[X] \to 0$). This proves upper bounds on the probability of properties.

**Second moment method:** If $\mathbb{E}[X^2] / \mathbb{E}[X]^2 \to 1$, then $X > 0$ w.h.p. (Paley-Zygmund inequality: $\mathbb{P}[X > 0] \ge \mathbb{E}[X]^2 / \mathbb{E}[X^2]$). This proves lower bounds.

### 2.3 Monotone Properties and Threshold Functions

**Definition (Monotone property):** A graph property $\mathcal{P}$ is **monotone increasing** if whenever $G \in \mathcal{P}$ and $G' \supset G$ (additional edges), then $G' \in \mathcal{P}$. Examples: connectivity, containing a triangle, having a giant component.

**Theorem (Bollobas-Thomason, 1987):** Every non-trivial monotone graph property has a **threshold function** $p^*(n)$ such that:
$$\lim_{n \to \infty} \mathbb{P}[G(n,p) \in \mathcal{P}] = \begin{cases} 0 & \text{if } p/p^* \to 0 \\ 1 & \text{if } p/p^* \to \infty \end{cases}$$

The proof uses the **noise sensitivity / sharp threshold** theory: monotone properties exhibit sharp transitions (the window where $\mathbb{P}[\mathcal{P}]$ goes from near-0 to near-1 is $o(p^*)$ wide) due to the influence of edges being approximately equal.

**For AI:** Sharp thresholds are the theoretical explanation for why GNN performance can change dramatically as graph density or community strength crosses a critical value. A model that performs at 90% accuracy might drop to random guessing when a graph parameter decreases by a factor of 2.

### 2.4 First and Second Moment Methods

These are the two workhorses for proving w.h.p. statements.

**First Moment (Upper Bound):**

To show $\mathcal{P}$ holds w.h.p., find a "bad event" $B$ and show $\mathbb{E}[\mathbf{1}_B] \to 0$:
$$\mathbb{P}[B] = \mathbb{E}[\mathbf{1}_B] \to 0$$

**Second Moment (Lower Bound):**

To show $X > 0$ w.h.p. (where $X = \sum_i X_i$ counts copies of some structure):

**Paley-Zygmund inequality:** $\mathbb{P}[X > 0] \ge \frac{\mathbb{E}[X]^2}{\mathbb{E}[X^2]}$

Expanding: $\mathbb{E}[X^2] = \sum_{i,j} \mathbb{E}[X_i X_j]$, so we need to bound correlations between pairs of copies.

**Example (Triangles):** Let $X$ = number of triangles in $G(n,p)$.

- $\mathbb{E}[X] = \binom{n}{3} p^3 \sim \frac{n^3 p^3}{6}$
- For $p = c/n$: $\mathbb{E}[X] \sim c^3/6$

The triangle threshold is $p^* = n^{-1}$: for $p \ll 1/n$, no triangles w.h.p.; for $p \gg 1/n$, triangles exist w.h.p.

---

## 3. Erdos-Renyi Model

### 3.1 Definitions: G(n,p) and G(n,m)

The Erdos-Renyi model is the simplest and most mathematically tractable random graph model. Its beauty lies in the complete independence of edges, which makes exact calculations feasible.

**Definition $G(n,p)$:** The random graph on $[n]$ where each edge is independently present with probability $p \in [0,1]$.

**Statistics:**
- $\mathbb{E}[|E|] = \binom{n}{2} p \approx n^2 p / 2$
- $\text{Var}[|E|] = \binom{n}{2} p(1-p)$
- $\mathbb{E}[\deg(v)] = (n-1)p \approx np$ for each vertex $v$

**Definition $G(n,m)$:** The random graph on $[n]$ drawn uniformly from all graphs with exactly $m$ edges.

**Regime classification** (by average degree $c = (n-1)p \approx np$):

| Regime | $p$ range | $c$ range | Largest component |
|--------|-----------|-----------|-------------------|
| Subcritical | $p < 1/n$ | $c < 1$ | $O(\log n)$ |
| Critical | $p = 1/n$ | $c = 1$ | $\Theta(n^{2/3})$ |
| Supercritical | $p > 1/n$ | $c > 1$ | $\Theta(n)$ |
| Connected | $p \ge \ln(n)/n$ | $c \ge \ln n$ | $n$ (whole graph) |

**For AI:** When analyzing message passing in sparse GNNs on ER graphs, the regime determines whether information can flow globally. Below criticality ($c < 1$), most nodes are in isolated small components - the GNN can only see local structure. Above criticality, there's a giant component enabling global information flow.

### 3.2 Degree Distribution: Poisson Limit

**Theorem (Poisson Degree Distribution):** In $G(n,p)$ with $p = c/n$, the degree of a fixed vertex $v$ satisfies:
$$\deg(v) \sim \text{Binomial}(n-1, c/n) \xrightarrow{d} \text{Poisson}(c) \text{ as } n \to \infty$$

**Proof sketch:** $\deg(v) = \sum_{u \neq v} X_{uv}$ where $X_{uv} \sim \text{Bernoulli}(c/n)$ are i.i.d. The sum of $n-1$ independent Bernoullis with success probability $c/n$ converges to Poisson($c$) by the law of small numbers (Poisson limit theorem).

**Generating function approach:** The probability generating function of $\text{Binomial}(n-1, c/n)$ is:
$$G_{B}(z) = \left(1 - \frac{c}{n} + \frac{cz}{n}\right)^{n-1} \to e^{c(z-1)} = G_{\text{Poi}(c)}(z)$$

**Consequences of Poisson degree distribution:**

1. **Exponential tail**: $\mathbb{P}[\deg(v) = k] = e^{-c} c^k / k!$ - degrees are concentrated near $c$
2. **Max degree**: $\max_v \deg(v) \sim \frac{\log n}{\log \log n}$ (sublinear in $n$)
3. **No hubs**: Unlike scale-free networks, ER graphs have no nodes with $\Omega(\sqrt{n})$ degree

**Contrast with scale-free:** Power-law degree distributions $\mathbb{P}[\deg = k] \propto k^{-\gamma}$ have polynomial tails and allow hubs with degree $\Theta(n^{1/(\gamma-1)})$. This is why real networks (preferential attachment) look so different from ER.

### 3.3 Giant Component Phase Transition

This is the central theorem of random graph theory - one of the most beautiful results in all of combinatorics.

**Theorem (Erdos-Renyi, 1960):** Let $c > 0$ be a constant and $p = c/n$. Let $L_1(G)$ denote the size of the largest connected component of $G \sim G(n,p)$.

1. **Subcritical** ($c < 1$): $L_1 / \ln n \to 1/I(c)$ w.h.p., where $I(c) = c - 1 - \ln c > 0$. In particular, $L_1 = O(\log n)$.

2. **Critical** ($c = 1$): $L_1 / n^{2/3} \to \Theta(1)$ in distribution (scaling limit is Brownian motion related).

3. **Supercritical** ($c > 1$): $L_1 / n \to \beta(c)$ w.h.p., where $\beta(c)$ is the unique solution in $(0,1)$ to:
$$\beta = 1 - e^{-c\beta}$$

The second-largest component has size $O(\log n)$.

**The survival probability $\beta(c)$:** The equation $\beta = 1 - e^{-c\beta}$ has a probabilistic interpretation. Think of each vertex independently joining the giant component with probability $\beta$. A vertex joins if it has at least one neighbor that also joins. If neighbors join with probability $\beta$, and there are $\text{Poisson}(c)$ neighbors, the probability of having at least one joining neighbor is $1 - e^{-c\beta}$ - hence the fixed-point equation.

**Derivation via branching processes:** A key technique is to compare component exploration with a branching process. Starting from vertex $v$, reveal its neighbors one by one. Each neighbor independently has $\text{Poisson}(c)$ additional neighbors (in the limit). This is a Galton-Watson branching process with offspring distribution $\text{Poisson}(c)$.

A Galton-Watson process with mean offspring $c$ survives (generates an infinite tree) with probability $\beta$ satisfying $\beta = 1 - e^{-c\beta}$. Below $c = 1$ (mean offspring $< 1$), the process dies out almost surely - hence subcritical. Above $c = 1$, there's a positive probability of survival - hence the giant component.

**Size of the giant component:**

| $c$ | $\beta(c)$ (fraction in giant) |
|-----|-------------------------------|
| 0.5 | 0 (subcritical) |
| 1.0 | 0 (critical, but $n^{2/3}$ scaling) |
| 1.5 | 0.583 |
| 2.0 | 0.797 |
| 3.0 | 0.940 |
| 5.0 | 0.993 |

**For AI:** GNN expressiveness on random graphs undergoes a similar phase transition. In the subcritical regime, the WL algorithm (and GINs) see disconnected local neighborhoods - they cannot distinguish non-isomorphic components. In the supercritical regime, the global component provides rich structural information.

### 3.4 Connectivity Threshold

**Theorem (Erdos-Renyi, 1961):** Let $p = (\ln n + c) / n$ for a constant $c \in \mathbb{R}$. Then:
$$\lim_{n \to \infty} \mathbb{P}[G(n,p) \text{ is connected}] = e^{-e^{-c}}$$

In particular:
- If $p \ll \ln(n)/n$: $G$ is disconnected w.h.p.
- If $p = \ln(n)/n$: $G$ is connected with probability $e^{-1} \approx 0.368$
- If $p \gg \ln(n)/n$: $G$ is connected w.h.p.

**Proof sketch (upper bound via first moment):**

Let $X_k$ = number of components of size $k \le n/2$. The bottleneck is $k=1$ (isolated vertices). A vertex $v$ is isolated iff none of its $n-1$ potential edges are present:
$$\mathbb{P}[v \text{ isolated}] = (1-p)^{n-1} = \left(1 - \frac{\ln n + c}{n}\right)^{n-1} \to e^{-(\ln n + c)} = \frac{e^{-c}}{n}$$

Expected isolated vertices: $\mathbb{E}[X_1] = n \cdot \frac{e^{-c}}{n} = e^{-c}$.

By a Poisson approximation argument, $X_1 \to \text{Poisson}(e^{-c})$ in distribution, so:
$$\mathbb{P}[X_1 = 0] \to e^{-e^{-c}}$$

Connectivity fails iff $X_1 > 0$ or there's a larger isolated component. One shows larger components disappear before isolated vertices, so connectivity threshold $=$ isolated vertex disappearance threshold.

**Sharp threshold:** The window is $p = (\ln n + \omega(1))/n$ - any diverging $\omega$ suffices. This is an unusually sharp threshold (polynomial-width thresholds are more common).

### 3.5 Subgraph Counts and Triangle Thresholds

**Definition:** A **subgraph count** for a fixed graph $H$ is $X_H = $ number of labeled copies of $H$ in $G(n,p)$.

**Expectation:**
$$\mathbb{E}[X_H] = \frac{n!}{(n - |V(H)|)!} \cdot \frac{1}{|\text{Aut}(H)|} \cdot p^{|E(H)|}$$

For dense subgraphs, this simplifies to $\Theta(n^{|V(H)|} p^{|E(H)|})$.

**Threshold for $H$:** By the first and second moment methods, $X_H > 0$ w.h.p. iff $p \gg n^{-1/m(H)}$, where:
$$m(H) = \max_{H' \subseteq H, |V(H')| \ge 1} \frac{|E(H')|}{|V(H')|}$$
is the **maximum 2-core density** of $H$.

**Triangle threshold:** For $H = K_3$ (triangle), $m(H) = 3/3 = 1$, so threshold is $p^* = n^{-1}$.

| Subgraph $H$ | $|V|$ | $|E|$ | $m(H)$ | Threshold $p^*$ |
|-------------|-------|-------|---------|----------------|
| Edge | 2 | 1 | 1/2 | $n^{-2}$ |
| Path $P_3$ | 3 | 2 | 2/3 | $n^{-3/2}$ |
| Triangle $K_3$ | 3 | 3 | 1 | $n^{-1}$ |
| 4-clique $K_4$ | 4 | 6 | 3/2 | $n^{-2/3}$ |
| 5-clique $K_5$ | 5 | 10 | 2 | $n^{-1/2}$ |

**Janson's inequality:** Provides exponential concentration for subgraph counts when $\mathbb{E}[X_H]$ is large:
$$\mathbb{P}[X_H = 0] \le \exp\left(-\frac{\mathbb{E}[X_H]^2}{2\Delta}\right)$$
where $\Delta = \sum_{H' \sim H''} \mathbb{P}[H' \cup H'' \subseteq G]$ sums over pairs sharing at least one edge.

### 3.6 Diameter and Distances

**Theorem:** In $G(n,p)$ with $c = np > 1$ (supercritical), the diameter of the giant component is:
$$\text{diam} \sim \frac{\log n}{\log(c)}$$
w.h.p. More precisely, $\text{diam} = (1 + o(1)) \log n / \log c$.

**Why?** The giant component behaves like a random tree with branching factor $c$. Starting from any node, the neighborhood of radius $r$ has size $\sim c^r$. It reaches $n$ when $c^r \approx n$, i.e., $r \approx \log n / \log c$.

**Characteristic path length:** This $O(\log n)$ diameter is the mathematical explanation for the **six degrees of separation** phenomenon. With $n = 10^9$ (Facebook) and $c \approx 100$ (100 friends), the diameter is $\log(10^9) / \log(100) \approx 4.5$ - indeed about 4-6 hops.

**Contrast with small-world:** In the ring lattice (before rewiring), diameter is $n/(2k) = \Omega(n)$ - linear. Watts-Strogatz introduces just $\beta$ fraction of random rewirings to drop this to $O(\log n)$ while maintaining high clustering.

### 3.7 Spectral Properties of G(n,p)

**Expected adjacency matrix:** $\mathbb{E}[\mathbf{A}] = p(\mathbf{1}\mathbf{1}^\top - \mathbf{I}_n)$, a rank-1 perturbation of $-p\mathbf{I}$.

**Spectral decomposition:** Write $\mathbf{A} = \mathbb{E}[\mathbf{A}] + (\mathbf{A} - \mathbb{E}[\mathbf{A}])$. The fluctuation matrix $\mathbf{W} = \mathbf{A} - \mathbb{E}[\mathbf{A}]$ is a Wigner matrix (symmetric, zero-diagonal, i.i.d. entries above diagonal).

**Theorem (Furedi-Komlos, 1981):** For $G(n,p)$ with $p$ constant, the eigenvalues of $\mathbf{A}$ satisfy:
- $\lambda_1 \sim np$ (outlier, corresponding to the average degree direction $\mathbf{1}/\sqrt{n}$)
- $\lambda_2, \ldots, \lambda_n$ are in $[-(2+\epsilon)\sqrt{np(1-p)}, (2+\epsilon)\sqrt{np(1-p)}]$ w.h.p.

The bulk eigenvalues follow Wigner's semicircle law with radius $2\sqrt{np(1-p)}$.

**For spectral clustering:** The spectral gap $\lambda_1 - \lambda_2 \approx np - 2\sqrt{np}$ determines how easily we can detect the leading eigenvector (which encodes the community structure in SBM).

---

## 4. Watts-Strogatz Small-World Model

### 4.1 Construction and Parameters

The Watts-Strogatz (WS) model was introduced in 1998 to explain a paradox: real-world networks (social, neural, power grid) have both high clustering AND short path lengths. Regular lattices have high clustering but long paths; ER random graphs have short paths but low clustering. WS interpolates between them.

**Construction (Watts-Strogatz, 1998):**

1. Start with a **ring lattice**: $n$ nodes arranged in a circle, each connected to $k$ nearest neighbors ($k/2$ on each side). Assume $n \gg k \gg \ln n$.
2. **Rewire**: For each edge $(i, j)$ with $j > i$ in the ring, with probability $\beta$ replace $j$ with a uniformly random node $j' \ne i$ (avoiding duplicate edges).

**Parameters:**
- $n$: number of nodes
- $k$: initial degree (even integer, typically $k \in \{4, 6, 10\}$)
- $\beta \in [0,1]$: rewiring probability
  - $\beta = 0$: regular ring lattice (high clustering, long paths)
  - $\beta = 1$: approximately ER random graph (low clustering, short paths)
  - $\beta \approx 0.01$-$0.1$: **small-world regime** (high clustering, short paths!)

**For AI:** Neural network connection graphs and attention patterns often exhibit small-world structure. The feedforward layers of transformers have short path lengths (like random graphs) while local attention heads maintain clustered connections (like lattices). Understanding this structure informs efficient attention design.

### 4.2 Clustering Coefficient

**Definition:** The **local clustering coefficient** of vertex $v$ with degree $d_v$ is:
$$C_v = \frac{|\{(u,w) \in E : u,w \in N(v)\}|}{\binom{d_v}{2}}$$
the fraction of $v$'s possible neighbor-pairs that are also connected.

**Global clustering coefficient:** $C = \frac{1}{n} \sum_v C_v$ (average over vertices).

**Ring lattice ($\beta = 0$):** Each vertex has $k$ neighbors (the $k/2$ on each side of the ring). Neighbors of $v$ include all nodes within distance $k/2$. A pair of $v$'s neighbors $(u,w)$ are connected iff they are within distance $k/2$ of each other.

For large $k$, the fraction of connected neighbor pairs is approximately:
$$C_{\text{lattice}} = \frac{3(k-2)}{4(k-1)} \approx \frac{3}{4} \text{ for large } k$$

**WS model:** For small $\beta$, the clustering coefficient is approximately:
$$C(\beta) \approx C(0) \cdot (1 - \beta)^3 = \frac{3(k-2)}{4(k-1)} (1 - \beta)^3$$

This is because a triangle involving $v$ requires all three edges to survive rewiring, and each edge survives with probability $(1-\beta)$.

**Random graph ($\beta = 1$):** For ER with $p \approx k/n$:
$$C_{\text{random}} \approx k/n \ll 1$$

**Key observation:** For small $\beta$ (say $\beta = 0.05$), $C(\beta) \approx 0.75 \cdot (0.95)^3 \approx 0.64$ - still very high, close to the lattice value.

### 4.3 Average Path Length

**Ring lattice ($\beta = 0$):** The shortest path between two nodes separated by $r$ positions in the ring is $\lceil r / (k/2) \rceil$. The average path length is approximately $n/(2k)$, which grows linearly with $n$.

**WS model ($0 < \beta < 1$):** The average path length drops dramatically with rewiring. Even a tiny $\beta$ introduces long-range shortcuts that drastically reduce path lengths.

**Heuristic analysis (Newman-Watts):** The typical inter-component distance after rewiring is:
$$L(\beta) \approx \frac{n}{k} \cdot f(nk\beta/2)$$

where $f(u) \sim \log(u)/u$ for large $u$. For $\beta \gg 2/(nk)$, the path length transitions from $\Theta(n)$ to $\Theta(\log n)$.

**Numerical example** ($n = 1000$, $k = 10$):

| $\beta$ | $C(\beta)$ | $L(\beta)$ |
|---------|-----------|-----------|
| 0 | 0.667 | 50 |
| 0.001 | 0.660 | 20 |
| 0.01 | 0.640 | 8 |
| 0.1 | 0.528 | 5 |
| 0.5 | 0.195 | 4 |
| 1.0 | 0.010 | 3 |

The small-world regime ($\beta \approx 0.01$-$0.05$) achieves high $C$ and low $L$ simultaneously.

### 4.4 The Small-World Regime

**Watts-Strogatz property:** A graph is said to have the **small-world property** if:
1. $C \gg C_{\text{random}} = k/n$ (much more clustered than ER)
2. $L \approx L_{\text{random}} = \log(n)/\log(k)$ (path length comparable to ER)

These two conditions are simultaneously satisfied for WS with $\beta \in [\Omega(1/(nk)), O(1)]$.

**Empirical validation:**

Watts and Strogatz validated the model against three real networks:

| Network | $n$ | $k$ | $C_{\text{actual}}$ | $C_{\text{random}}$ | $L_{\text{actual}}$ | $L_{\text{random}}$ |
|---------|-----|-----|---------------------|---------------------|---------------------|---------------------|
| Film actors | 225,226 | 61 | 0.79 | 0.00027 | 3.65 | 2.99 |
| Power grid (W. US) | 4,941 | 2.67 | 0.080 | 0.005 | 18.7 | 12.4 |
| C. elegans neural | 282 | 14 | 0.28 | 0.05 | 2.65 | 2.25 |

All three networks have high clustering (much above ER random graph level) but short average path lengths (comparable to ER). This is the hallmark of small-world structure.

### 4.5 Navigability and Kleinberg's Grid

Watts and Strogatz showed that small-world graphs have short paths, but Kleinberg (2000) asked: can nodes *find* these short paths using only local information?

**Kleinberg's model:** Start with a 2D grid of $n = k \times k$ nodes. Each node $v$ has edges to all grid neighbors within distance $r$ (local structure) plus one long-range link to node $u$ chosen with probability proportional to $d(v,u)^{-s}$.

**Theorem (Kleinberg, 2000):**
- If $s = 2$ (exponent matches grid dimension): greedy routing finds paths of length $O((\log n)^2)$ w.h.p.
- If $s \ne 2$: any decentralized algorithm requires $\Omega(n^{\delta})$ steps for some $\delta > 0$.

**Interpretation:** The power-law exponent $s = d$ (where $d$ is the grid dimension) uniquely enables efficient decentralized navigation. Real social networks approximate this condition, explaining how people can navigate social networks in few steps even with only local knowledge.

**For AI:** This connects to hierarchical attention in transformers. FlashAttention-2 and related methods achieve efficient attention by exploiting the fact that attention scores decay with token distance - a continuous analog of Kleinberg's inverse power-law long-range links.

### 4.6 Social Network Evidence

Small-world structure has been empirically validated across many domains:

- **Social networks:** Milgram's 1967 experiment found ~6 hops between strangers in the US; Facebook (2016) found average path length 3.57 for 1.6 billion users.
- **Neural connectomes:** C. elegans (302 neurons), mouse visual cortex, human brain fMRI networks all show high clustering and short paths.
- **Internet topology:** ASN-level and router-level graphs exhibit small-world properties.
- **Citation networks:** Academic citation graphs have $C \approx 0.3$-$0.5$, well above the ER baseline.

**Limitations:** WS does not generate power-law degree distributions. Real networks often exhibit BOTH small-world AND scale-free properties - requiring hybrid models.

---

## 5. Barabasi-Albert Scale-Free Model

### 5.1 Preferential Attachment

The Barabasi-Albert (BA) model was introduced in 1999 to explain a universal observation: degree distributions in real-world networks follow a power law $P(k) \propto k^{-\gamma}$ with $\gamma \approx 2$-$3$. This is incompatible with the exponential tails of ER's Poisson distribution.

The explanation: networks grow over time, and new nodes prefer to attach to high-degree nodes. This "rich get richer" mechanism generates hubs.

**Construction (Barabasi-Albert):**

1. Start with $m_0 \ge m$ nodes with arbitrary connections.
2. At each time step $t = m_0+1, m_0+2, \ldots, n$:
   - Add one new node $v_t$
   - Connect $v_t$ to $m$ existing nodes, chosen independently with probability proportional to their current degree:
$$\Pi(v_t \to u) = \frac{\deg(u)}{\sum_{w} \deg(w)}$$

The denominator $\sum_w \deg(w) = 2|E|$ (handshaking lemma).

**Why "preferential attachment"?** This mimics real-world network growth:
- Web pages link to popular pages (already have many in-links)
- Scientists cite highly-cited papers
- Airports add routes to hubs

**For AI:** GNN training graphs from large knowledge bases (Wikidata, Freebase) exhibit scale-free degree distributions. GNN aggregation functions must handle this heterogeneity - a degree-10 node and a degree-1000 node require different treatment.

### 5.2 Power-Law Degree Distribution

**Theorem (Barabasi-Albert, 1999; rigorous: Bollobas et al., 2001):** For $G \sim \text{BA}(n, m)$, as $n \to \infty$, the degree distribution satisfies:
$$\mathbb{P}[\deg(v) = k] \to \frac{2m(m+1)}{k(k+1)(k+2)} \sim \frac{2m^2}{k^3} \text{ for large } k$$

This is a **power law** with exponent $\gamma = 3$:
$$P(k) \propto k^{-3}$$

**Scale-free:** A distribution is called **scale-free** if $P(k) \propto k^{-\gamma}$ for some $\gamma > 1$. Scale-free means the distribution looks the same at all scales (self-similar under rescaling).

**Moments:** For $P(k) \propto k^{-\gamma}$:
- $\mathbb{E}[k] < \infty$ iff $\gamma > 2$
- $\mathbb{E}[k^2] < \infty$ iff $\gamma > 3$

For BA with $\gamma = 3$: mean degree is finite, but variance diverges. This has profound implications:
- **No effective epidemic threshold**: diseases spread to the whole network for any transmission rate $>0$
- **Robustness to random failure**: removing random nodes rarely disconnects the graph (most nodes have low degree)
- **Vulnerability to targeted attack**: removing the few hubs disconnects the graph immediately

### 5.3 Mean-Field Analysis

**Mean-field derivation:** Let $k_i(t)$ be the degree of node $i$ at time $t$. Treating $k_i$ as a continuous variable and taking expectations:
$$\frac{\partial k_i}{\partial t} = m \cdot \Pi(i) = m \cdot \frac{k_i}{\sum_j k_j}$$

Since $\sum_j k_j = 2mt$ (each new node adds $m$ edges, contributing $2m$ to total degree sum):
$$\frac{\partial k_i}{\partial t} = \frac{k_i}{2t}$$

**Solution:** With initial condition $k_i(t_i) = m$ (node $i$ added at time $t_i$ with $m$ initial edges):
$$k_i(t) = m \sqrt{t / t_i}$$

**Degree distribution from mean-field:** Node $i$ has degree $> k$ at time $t$ iff $k_i(t) > k$, i.e., $t_i < m^2 t / k^2$. Since nodes are added uniformly at rate 1 per step:
$$\mathbb{P}[k_i(t) > k] = \mathbb{P}[t_i < m^2 t / k^2] = \frac{m^2 t / k^2}{t} = \frac{m^2}{k^2}$$

Therefore:
$$P(k) = -\frac{d}{dk}\mathbb{P}[k_i(t) > k] = \frac{2m^2}{k^3}$$

This reproduces the power law $P(k) \propto k^{-3}$, consistent with the rigorous result.

**Hub formation:** The oldest nodes grow fastest (degree $\propto \sqrt{t/t_i}$). Node $i$ added at time $t_i = 1$ has degree $\sim m\sqrt{n}$ at time $n$ - a hub with $\Theta(\sqrt{n})$ degree.

### 5.4 Robustness and Fragility

**Random failure:** If each node is independently removed with probability $q$, what is the critical $q^*$ above which the giant component disappears?

For a network with degree distribution $P(k)$, the critical fraction for percolation is determined by the **Molloy-Reed criterion**:
$$\frac{\mathbb{E}[k^2]}{\mathbb{E}[k]} > 2$$

For scale-free networks with $\gamma \le 3$ (including BA with $\gamma = 3$): $\mathbb{E}[k^2] = \infty$, so this criterion always holds. No matter how many nodes are randomly removed, a giant component persists (in infinite networks). In finite BA networks, the giant component persists for $q$ close to 1.

**Targeted attack:** If the highest-degree nodes are removed first, the network is highly vulnerable. Removing even 5% of the highest-degree nodes can destroy the giant component of a BA network.

**Implication for AI:** Neural network pruning that removes random weights (random failure) is much safer than pruning by magnitude (targeted attack) - magnitude-based pruning could inadvertently target the "hubs" of the implicit neural network graph.

### 5.5 Scale-Free Networks in the Wild

**Evidence for scale-free:** Many networks show approximate power-law degree distributions:
- World Wide Web: $\gamma_{\text{in}} \approx 2.1$, $\gamma_{\text{out}} \approx 2.7$
- Internet (AS-level): $\gamma \approx 2.1$
- Protein-protein interaction: $\gamma \approx 2.4$
- Actor collaboration: $\gamma \approx 2.3$

**Controversy (2019):** Broido & Clauset (2019) argued that true power laws are rare - most claimed scale-free networks fit log-normal or other heavy-tailed distributions better. The debate continues, but the practical point stands: real networks have much heavier degree tails than ER Poisson.

**GNN implications:** Degree-heterogeneous graphs require careful normalization in GCN-style aggregation. GraphSAGE's neighborhood sampling provides approximate uniformity; PNA (Principal Neighbourhood Aggregation) uses multiple aggregators (mean, max, std) to handle degree heterogeneity robustly.

---

## 6. Stochastic Block Model

### 6.1 Definition and Parameters

The Stochastic Block Model (SBM) is the canonical random graph model for **community structure**. It is the workhorse of GNN benchmark generation and the model for which information-theoretic limits of community detection are fully understood.

**Definition (SBM):** Given parameters:
- $n$: number of nodes
- $k$: number of communities/blocks
- $\sigma: [n] \to [k]$: community assignment vector
- $B \in [0,1]^{k \times k}$: symmetric block probability matrix (B_{rs} = probability of edge between community $r$ and $s$)

The SBM generates a random graph $G \sim \text{SBM}(n, k, \sigma, B)$ by independently including each edge $(u,v)$ with probability $B_{\sigma(u), \sigma(v)}$.

**Symmetric SBM (planted partition):** All communities have equal size $n/k$; $B_{rr} = p$ (within-community) and $B_{rs} = q$ for $r \ne s$ (between-community), with $p > q$.

**Sparse SBM:** $p = a/n$, $q = b/n$ with $a > b$ constants. This is the regime studied in information-theoretic limits.

**Why SBM matters for AI:** Node classification on real graphs (citation networks, social networks) essentially asks: given an observed graph $G$ with node features, recover the community labels $\sigma$. SBM provides the ground truth for studying when this is possible and what algorithms can achieve it.

### 6.2 Community Detection: Information-Theoretic Limits

**Three regimes for symmetric 2-block SBM** ($k = 2$, equal communities):

With $p = a/n$, $q = b/n$:

1. **Exact recovery** ($a > b$ large enough): Recover $\sigma$ exactly (up to global permutation) with high probability. Threshold: $\sqrt{a} - \sqrt{b} > \sqrt{2}$.

2. **Weak recovery / detection**: Identify communities better than chance (correlated with ground truth). Threshold (**Kesten-Stigum**): $(a-b)^2 > 2(a+b)$.

3. **Impossible**: Below the Kesten-Stigum threshold, no algorithm can do better than random guessing.

**Kesten-Stigum threshold:** The SNR condition $(a-b)^2 > 2(a+b)$ can be rewritten as:
$$\frac{(a-b)^2}{2(a+b)} > 1$$

The left side is the signal-to-noise ratio: $(a-b)$ measures community signal, $\sqrt{a+b}$ measures noise (average degree). When SNR $< 1$, the noise overwhelms the signal.

**Spectral interpretation:** The second eigenvalue of the adjacency matrix satisfies $\lambda_2 \approx (a-b)/n \cdot n/2 = (a-b)/2$. The spectral radius of the noise (Wigner semicircle) is $2\sqrt{(a+b)/2}$. The signal eigenvalue emerges from the noise bulk iff $(a-b)/2 > 2\sqrt{(a+b)/2}$, i.e., $(a-b)^2 > 8(a+b)/2 = 4(a+b)$ - slightly different from KS due to different normalization, but same qualitative conclusion.

**For GNNs:** GNNs trained on SBM graphs face exactly this detection problem. Below the Kesten-Stigum threshold, no GNN - regardless of depth, width, or architecture - can reliably classify nodes into communities. This is the hard limit on what graph learning can achieve.

**$k$-block SBM:** For $k$ communities, the threshold generalizes. The second eigenvalue of $B$ determines the computational threshold.

### 6.3 Degree-Corrected SBM

Real-world networks have heterogeneous degree distributions within communities. The DC-SBM adds degree parameters:

**Definition (DC-SBM):** Node $v$ has a degree parameter $\theta_v > 0$. The probability of edge $(u,v)$ is:
$$\mathbb{P}[(u,v) \in E] = \theta_u \theta_v B_{\sigma(u), \sigma(v)}$$

Under DC-SBM, the expected degree of node $v$ is $\theta_v \sum_u \theta_u B_{\sigma(v), \sigma(u)}$. Choosing $\theta_v \propto k^{-\alpha}$ for some power law generates a scale-free SBM combining community structure with hub-spoke topology.

**Why DC-SBM matters:** Citation networks and social networks have communities AND degree heterogeneity. Standard spectral clustering fails on DC-SBM because high-degree nodes dominate the leading eigenvectors. Regularized spectral clustering or normalized Laplacian-based methods are needed.

### 6.4 SBM as GNN Benchmark Generator

**GraphWorld (2022):** A Google framework that generates GNN benchmarks by sampling SBM parameters from a distribution over:
- Number of communities $k \in [2, 20]$
- Community sizes (balanced or Zipf-distributed)
- Within-block density $p$
- Between-block density $q$

By sampling $(a,b)$ pairs across and below the Kesten-Stigum threshold, GraphWorld generates benchmarks that are systematically hard or easy for GNNs.

**Result:** Different GNN architectures excel in different SBM regimes:
- GCN: best for homophilic SBM (dense intra-community)
- GraphSAGE: robust across densities
- GIN: best near the Kesten-Stigum threshold (maximally expressive)
- MixHop: best for heterophilic SBM (dense inter-community)

This validates the theoretical prediction that no single GNN architecture dominates all graph types.

---

## 7. Spectral Properties of Random Graphs

### 7.1 Wigner's Semicircle Law

Wigner's semicircle law is a fundamental result in random matrix theory that describes the bulk eigenvalue distribution of random symmetric matrices.

**Setup:** Let $W_n = \frac{1}{\sqrt{n}} M_n$ where $M_n$ is a real symmetric $n \times n$ random matrix with:
- Diagonal entries $M_{ii} = 0$
- Off-diagonal entries $M_{ij} = M_{ji}$ i.i.d. with mean 0 and variance $\sigma^2$

**Theorem (Wigner, 1955):** The empirical spectral distribution of $W_n$:
$$\mu_n = \frac{1}{n} \sum_{i=1}^n \delta_{\lambda_i(W_n)}$$
converges weakly in probability to the semicircle law:
$$\mu_{sc}(dx) = \frac{1}{2\pi\sigma^2}\sqrt{4\sigma^2 - x^2} \cdot \mathbf{1}_{|x| \le 2\sigma} dx$$

**Support:** $[-2\sigma, 2\sigma]$ - all eigenvalues of $W_n$ lie in this interval asymptotically.

**For $G(n,p)$:** The adjacency matrix $A$ centered and scaled: $W = (A - p\mathbf{1}\mathbf{1}^\top) / \sqrt{np(1-p)}$ has bulk eigenvalues following the semicircle law on $[-2, 2]$ (with $\sigma = 1$).

The **outlier eigenvalue** $\lambda_1 \approx np$ comes from the mean matrix $p\mathbf{1}\mathbf{1}^\top$ - it pokes out of the bulk.

**For SBM:** The adjacency matrix of an SBM decomposes as $A = \bar{A} + W$ where $\bar{A} = \mathbb{E}[A]$ is the block-structured mean matrix (rank $k$) and $W$ is a Wigner-like noise matrix. The $k$ outlier eigenvalues of $\bar{A}$ emerge from the semicircle bulk and encode the community structure, provided the signal-to-noise ratio exceeds the Kesten-Stigum threshold.

### 7.2 Spectral Gap and Community Detection

**Spectral algorithm for SBM:**

1. Compute the leading $k$ eigenvectors of $A$ (or normalized Laplacian $L_{sym}$)
2. Form the $n \times k$ embedding matrix $U$
3. Run $k$-means on rows of $U$ to recover community assignments

**Why does this work?** For the planted partition SBM with $p = a/n$ and $q = b/n$:
- Leading eigenvalue: $\lambda_1 \approx (a + (k-1)b)/(2k) \cdot n/n = (a + (k-1)b)/2$ ... (using $k$-block symmetric SBM)
- Second eigenvalue: $\lambda_2 \approx (a-b)/2$ (signal eigenvalue)
- Bulk edge: $\lambda_{\text{bulk}} \approx \sqrt{(a+b)/2}$

Signal eigenvalue separates from bulk iff $|(a-b)/2| > \sqrt{(a+b)/2}$, which is exactly the Kesten-Stigum condition.

**Davis-Kahan bound (next subsection):** If signal and noise eigenvalues are separated, the eigenvectors of $A$ are close to those of $\mathbb{E}[A]$, enabling accurate community recovery.

### 7.3 Davis-Kahan Theorem and Perturbation Bounds

**Theorem (Davis-Kahan, 1970):** Let $A = \bar{A} + W$ where $\bar{A}$ is symmetric with eigenvalue $\lambda$ and eigenvector $\mathbf{u}$, and $W$ is a perturbation with $\|W\|_{op} \le \delta$. Let $\delta'$ be the gap between $\lambda$ and all other eigenvalues of $\bar{A}$. Then the corresponding eigenvector $\hat{\mathbf{u}}$ of $A$ satisfies:
$$\sin\angle(\hat{\mathbf{u}}, \mathbf{u}) \le \frac{\delta}{\delta' - \delta}$$

**Application to SBM:** For the 2-block SBM:
- Signal gap: $\delta' = \lambda_2(\bar{A}) - \lambda_3(\bar{A}) = (a-b)/2$
- Noise: $\|W\|_{op} \le 2\sqrt{(a+b)/2} \cdot (1 + o(1))$ (Wigner semicircle radius)
- Angle error: $\sin\angle(\hat{\mathbf{u}}_2, \mathbf{u}_2) \le \frac{2\sqrt{(a+b)/2}}{(a-b)/2 - 2\sqrt{(a+b)/2}}$

This is $o(1)$ iff $(a-b)/2 > 2\sqrt{(a+b)/2}$, i.e., the Kesten-Stigum condition holds. When it holds, the estimated eigenvector is close to the ground-truth community indicator vector, and $k$-means succeeds.

**For GNNs:** Davis-Kahan explains why GNNs with spectral-style message passing (GCN) can do community detection above the threshold but fail below it. It also shows that adding node features (which contribute additional signal to $\bar{A}$) can push GNNs above the threshold even when the graph structure alone is insufficient.

### 7.4 Laplacian Spectrum of G(n,p)

**Normalized Laplacian:** $L_{sym} = D^{-1/2}(D-A)D^{-1/2} = I - D^{-1/2}AD^{-1/2}$.

For $G(n,p)$ with $p$ constant:
- Eigenvalues of $L_{sym}$ lie in $[0, 2]$ (always)
- $\lambda_1(L_{sym}) = 0$ always (corresponding to $\mathbf{1}/\sqrt{n}$)
- For $p$ fixed: $\lambda_n(L_{sym}) \to 2$, $\lambda_2(L_{sym}) \to 1$ - the bulk concentrates near 1

**Algebraic connectivity (Fiedler value):** $\lambda_2(L)$ (unnormalized Laplacian) is the **Fiedler value**, which controls:
- Convergence rate of random walks (mixing time $\propto 1/\lambda_2$)
- Robustness (edge connectivity $\ge \lambda_2$)
- Cheeger constant approximation: $\lambda_2/2 \le h(G) \le \sqrt{2\lambda_2}$

For $G(n,p)$ with $p = c/n$, $c > 1$: $\lambda_2(L) \approx c - 2\sqrt{c} + O(1/n)$ for the giant component.

---

## 8. Graphons: The Infinite-Size Limit

### 8.1 Dense Graph Sequences and the Cut Metric

As graph size $n \to \infty$, what is the "limit" of a sequence of dense graphs? This question leads to graphon theory - the measure-theoretic framework that unifies all dense random graph models.

**Dense graph sequence:** A sequence $G_n$ of graphs with $|V(G_n)| = n$ and $|E(G_n)| = \Theta(n^2)$ (constant edge density).

**Cut distance:** The cut norm between two symmetric functions $f, g: [0,1]^2 \to \mathbb{R}$ is:
$$\|f - g\|_{\square} = \sup_{S, T \subseteq [0,1]} \left| \int_{S \times T} (f(x,y) - g(x,y)) \, dx \, dy \right|$$

The **cut metric** between graphs $G$ and $H$ on the same vertex set is $d_\square(G, H) = \|W_G - W_H\|_\square$ where $W_G(x,y)$ is the step function encoding the adjacency of $G$.

After quotienting by relabelings (graph isomorphisms), we get the **cut distance** $\delta_\square(G, H)$.

### 8.2 Graphon Definition and Sampling

**Definition (Graphon):** A **graphon** is a symmetric measurable function $W: [0,1]^2 \to [0,1]$.

**Intuition:** Think of $W(x,y)$ as the "connection probability" between two abstract node types $x$ and $y$, where node types are drawn uniformly from $[0,1]$.

**Graphon sampling:** To sample an $n$-node graph from graphon $W$:
1. Draw $\xi_1, \ldots, \xi_n \sim \text{Uniform}[0,1]$ i.i.d. (latent node types)
2. Include edge $(i,j)$ with probability $W(\xi_i, \xi_j)$, independently

**Examples of graphons and their corresponding random graph models:**

| Graphon $W(x,y)$ | Corresponding model |
|-----------------|---------------------|
| $W(x,y) = p$ (constant) | Erdos-Renyi $G(n,p)$ |
| $W(x,y) = \mathbf{1}[x+y > 1]$ | Half-graph |
| $W(x,y) = \mathbf{1}[\lfloor kx \rfloor = \lfloor ky \rfloor] \cdot p + (1 - \mathbf{1}[\ldots]) \cdot q$ | $k$-block SBM |
| $W(x,y) = x \cdot y$ | Product graphon (Chung-Lu style) |
| $W(x,y) = \min(x,y)$ | Threshold model |

**Lovasz-Szegedy theorem (2006):** Every convergent sequence of dense graphs (in the cut metric) converges to a graphon. Conversely, every graphon arises as the limit of a graph sequence. The space of graphons with the cut metric is compact.

**For AI:** Graphons are the natural limiting objects for graph neural networks. A GNN $f$ maps graphs to outputs. If $f$ is *continuous in the cut metric*, then $f(G_n) \to f(W)$ whenever $G_n \to W$ in cut distance. This is precisely the condition for a GNN to be "robust" to graph size changes.

### 8.3 Homomorphism Densities and Graph Parameters

**Homomorphism density:** For graphs $F$ and $G$, the homomorphism density is:
$$t(F, G) = \frac{\text{hom}(F, G)}{|V(G)|^{|V(F)|}}$$
where $\text{hom}(F, G)$ counts graph homomorphisms $F \to G$.

**Examples:**
- $t(K_2, G)$ = edge density
- $t(K_3, G)$ = triangle density
- $t(C_4, G)$ = 4-cycle density

**Graphon version:** For a graphon $W$ and graph $F$ with $k$ vertices:
$$t(F, W) = \int_{[0,1]^k} \prod_{(i,j) \in E(F)} W(x_i, x_j) \, d\mathbf{x}$$

**Theorem (Lovasz-Szegedy):** Two graphons $W$ and $W'$ are equivalent (isomorphic in the graphon sense) iff $t(F, W) = t(F, W')$ for all graphs $F$.

**Graph parameters from homomorphism densities:**
- Triangle density: $t(K_3, W) = \int W(x,y)W(y,z)W(x,z) \, dx\, dy\, dz$
- Clustering coefficient: related to $t(K_3, W) / t(P_2, W)^{3/2}$

**For ML:** Substructure counting (homomorphism density estimation) is a key primitive for graph feature extraction. Ring GNNs and structural GNNs compute approximate homomorphism densities; this is one way to understand what graph structure GNNs learn.

### 8.4 Graphon Neural Networks

**Definition (Graphon Neural Network, Keriven & Peyre, 2019):** A graphon neural network is a sequence of functions $f_n$ (on $n$-node graphs) that converge to a limiting function $f_W$ on graphons. Formally, $f_n$ is a GNN architecture such that as $G_n \to W$ in cut metric, $f_n(G_n) \to f_W(W)$.

**Key result:** GCN-style message passing with the propagation operator $T_W h(x) = \int_0^1 W(x,y) h(y) \, dy$ defines a graphon neural network. The discrete GCN is a finite approximation.

**Universality of graphon NNs:** Maron et al. (2019) show that equivariant graph networks are universal approximators on graphons - any graphon property that is measurable can be approximated to arbitrary accuracy.

**Limitation:** Universality on graphons is density-dependent. For SPARSE graph sequences (edge density $\to 0$), graphons become trivial (the zero graphon), and a different limit theory (graphexes, local limits) is needed.

> **Forward reference:** Graphon theory connects directly to functional analysis - the operator $T_W h(x) = \int W(x,y)h(y) \, dy$ is an integral operator on $L^2[0,1]$. Spectral theory of compact operators (Hilbert-Schmidt theorem) governs its eigenvalue decomposition. -> Full treatment: [Functional Analysis](../../12-Functional-Analysis/notes.md)

---

## 9. Applications in Machine Learning

### 9.1 GraphWorld: Benchmark Generation from SBM

**Problem:** GNN papers often report results on 3-4 standard benchmarks (Cora, Citeseer, OGBN-Arxiv). These benchmarks may not represent the diversity of graph structures encountered in practice.

**GraphWorld (Palowitch et al., 2022):** A benchmark generation framework that:
1. Parameterizes SBM space $(k, n, p, q, \text{features})$
2. Samples thousands of graph instances across this parameter space
3. Evaluates GNN architectures across the full parameter landscape

**Key findings:**
- No single GNN architecture dominates across all SBM parameters
- GCN is best on homophilic dense graphs; GIN is best near the detection threshold
- The Kesten-Stigum threshold accurately predicts when ALL GNNs fail
- Node feature quality (signal-to-noise ratio in features) often matters more than graph structure

**For practitioners:** When evaluating a GNN, generate benchmarks from an SBM sweep to characterize the algorithm's operating regime, rather than relying on a few fixed benchmarks.

### 9.2 Graph Generation: GRAN, GDSS, DiGress

Graph generation models learn to sample new graphs from a distribution. Framed probabilistically: learn a distribution $p_\theta(G)$ that matches a target distribution $p^*(G)$.

**GRAN (Graph Recurrent Attention Networks, 2019):** Generates graphs node-by-node, at each step attending to previously generated nodes. The attention mechanism implicitly models preferential attachment - recently added high-degree nodes attract more attention, reproducing scale-free structure.

**GDSS (Jo et al., 2022):** Score-based diffusion model for graphs. Joint diffusion over node features and edge features. Samples new graphs by reversing a diffusion process that gradually adds noise. The score function implicitly learns the SBM-like block structure of training graphs.

**DiGress (Vignac et al., 2022):** Discrete diffusion - adds/removes edges following a Markov process. Denoising model is a graph transformer that learns to predict the original edge from the noised version. DiGress can generate molecular graphs and large social networks by learning implicit graphon structure.

**Random graph theory connection:** Graph generation is essentially graphon estimation. Given samples from $p^*(G)$ (real graphs), estimate the underlying graphon $W^*$ such that sampling from $W^*$ approximates $p^*$. The approximation quality is measured by the cut metric $d_\square$.

### 9.3 LLM Attention as a Random Graph Process

**Attention graph:** In a transformer with $L$ layers and $H$ heads, the attention pattern at layer $\ell$, head $h$ defines a complete directed graph on tokens where edge weight $(i,j)$ is the attention score $\alpha^{(\ell,h)}_{ij}$.

**Sparse attention = random graph:** Sparse attention mechanisms (Longformer, BigBird, Reformer) explicitly sparsify the attention graph, keeping only $O(n)$ edges rather than $O(n^2)$. The sparsification pattern is often random or pseudo-random.

**Random graph analysis of attention:**
- **Connectivity**: Is the sparse attention graph connected? If not, information cannot flow between disconnected components. ER theory predicts connectivity iff expected degree $\ge \ln n$.
- **Small-world**: BigBird combines local window attention (ring lattice) with random global tokens (rewiring) and special tokens (hubs). This exactly matches the Watts-Strogatz construction!
- **Expander properties**: Expander graphs (high spectral gap) are the best sparse graphs for information flow. Ramanujan graphs achieve the optimal spectral gap - this motivates expander-based sparse attention.

**Result:** The random graph structure of sparse attention patterns determines the theoretical expressiveness of the transformer. An attention graph that is disconnected or has large diameter cannot capture long-range dependencies regardless of the weight values.

### 9.4 Lottery Ticket Hypothesis and Sparse Subgraphs

**Lottery Ticket Hypothesis (Frankle & Carlin, 2019):** A randomly initialized dense network contains sparse subnetworks ("winning tickets") that, when trained in isolation from the same initialization, achieve comparable accuracy to the dense network.

**Random graph framing:** Think of the neural network as a random graph where:
- Nodes = neurons
- Edges = weights (including the weight value)
- Sparsification = edge removal

The winning ticket is a sparse subgraph that retains the connectivity and flow properties of the dense graph. Random graph percolation theory describes when sparse subgraphs retain giant component connectivity.

**Percolation interpretation:** If we randomly retain each edge with probability $\rho$ (magnitude-based pruning approximation), the network retains its giant component iff $\rho > \rho_c$, where $\rho_c$ is the percolation threshold. For ER-like neural network graphs, $\rho_c \approx 1/(np_0)$ where $p_0$ is the original edge density.

**Practical implication:** Networks can be pruned to 90%+ sparsity (i.e., $\rho \approx 0.1$) while retaining performance, consistent with the existence of a giant percolating subgraph at such densities for typical neural network width/depth ratios.

### 9.5 Social Network Analysis at Scale

**Community detection at scale:** For billion-node graphs (Facebook, Twitter), exact SBM community detection is computationally infeasible. In practice:
- **Louvain algorithm**: Greedy modularity maximization, $O(n \log n)$
- **Label propagation**: Message-passing on the graph, converges in $O(\text{diam})$ steps
- **GraphSAGE + semi-supervised**: Use a few labeled nodes (community labels) to train GNN

**Random graph models as null models:** When analyzing a real social network, we ask: "Is the observed community structure more than what we'd expect by chance?" We compare to an ER null model with the same degree sequence (configuration model) and test whether the observed modularity exceeds the null expectation.

**Link prediction:** Predicting missing edges $(u,v)$ in a social graph using random graph models. Under ER: $\mathbb{P}[(u,v) \in E] = p$ (same for all pairs). Under SBM: $\mathbb{P}[(u,v) \in E] = B_{\sigma(u), \sigma(v)}$ - within-community edges are more likely. GNNs for link prediction learn to approximate this block-structured probability.

---

## 10. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
|---|---------|---------------|-----|
| 1 | Confusing $G(n,p)$ and $G(n,m)$ | $G(n,p)$ has random edge count; $G(n,m)$ has exactly $m$ edges - they agree asymptotically but differ in finite samples | Use $G(n,p)$ when you want independence; $G(n,m)$ when you want exact count control |
| 2 | Assuming ER is a good null model for all real graphs | ER lacks clustering, hubs, and communities - major features of real networks | Use configuration model (fixed degree sequence) or SBM as null model |
| 3 | Saying Barabasi-Albert generates power law with any exponent | BA always gives $\gamma = 3$; only extensions (fitness, rewiring) change $\gamma$ | Use generalized preferential attachment $\Pi(v) \propto \deg(v)^\alpha$ for $\gamma = 1 + 1/(\alpha - 1)$ |
| 4 | Confusing clustering coefficient with transitivity | Local CC averages over vertices; global transitivity = $3 \times$ triangles / paths of length 2. They differ when degree distribution is heterogeneous | Use transitivity for global property; local CC for per-node property |
| 5 | Thinking the Kesten-Stigum threshold is a computational limit | KS is an information-theoretic limit - it bounds what ANY algorithm can do, not just efficient ones. The computational hardness (SDP vs AMP) is a separate question | Distinguish between information-theoretic and computational thresholds |
| 6 | Applying graphon theory to sparse graphs | Graphons describe the limit of DENSE sequences (constant edge density). Sparse graphs ($m = O(n)$) have trivial graphon limits (the zero function) | Use local weak limits (Benjamini-Schramm) or graphex theory for sparse sequences |
| 7 | Equating "scale-free" with "power law" | Scale-free means the DEGREE distribution is a power law. Many other distributions (log-normal, Pareto) look similar. Broido & Clauset (2019) show many claimed scale-free networks don't pass rigorous power-law tests | Use maximum likelihood to fit and compare competing distributions (power law, log-normal, exponential) |
| 8 | Ignoring the giant component when computing path lengths | Average path length on a disconnected graph is undefined (infinite) or meaningless without restricting to the giant component | Always compute path lengths within the largest connected component |
| 9 | Assuming WS small-world is scale-free | WS generates Poisson-like degree distributions (each node rewires independently). It has small-world property but NOT scale-free | Combine WS with preferential attachment for small-world + scale-free |
| 10 | Using the adjacency matrix spectrum directly for SBM clustering when graph is sparse | For sparse SBM, the adjacency matrix eigenvalues don't separate cleanly (semicircle radius comparable to signal). Use regularized Laplacian or Bethe-Hessian instead | Replace $A$ with $L_{rw} = D^{-1}A$ or Bethe-Hessian $H(\rho) = (\rho^2-1)I - \rho A + D$ |
| 11 | Treating the WS model as a generative process for new nodes | WS is defined on a fixed set of $n$ nodes with rewiring - it does not define how to add new nodes. It's not a growing network model | Use BA for growing network models; use WS for fixed-size networks |
| 12 | Forgetting that graphon equivalence classes ignore node labels | Two graphons $W$ and $W'$ are equivalent if one is a measure-preserving relabeling of the other. Most graph statistics depend only on the graphon equivalence class | Work with homomorphism densities (relabeling-invariant) when comparing graphons |

---

## 11. Exercises

**Exercise 1** * - Phase Transition Simulation

Simulate the Erdos-Renyi phase transition computationally. For $n = 2000$ nodes and $p = c/n$ with $c \in [0.5, 3.0]$:

(a) Generate $G(n,p)$ for 20 values of $c$ linearly spaced in $[0.5, 3.0]$.

(b) For each graph, compute the size of the largest connected component $L_1$ and the second largest $L_2$.

(c) Plot $L_1/n$ and $L_2/n$ as functions of $c$. Identify the phase transition point visually.

(d) Overlay the theoretical prediction $\beta(c)$ satisfying $\beta = 1 - e^{-c\beta}$. Compute $\beta(c)$ numerically (Newton's method or bisection) for each $c$.

(e) Compute and plot $L_2/n$. What happens to the second-largest component at criticality? Explain from branching process theory.

---

**Exercise 2** * - Degree Distribution Analysis

(a) Generate $G(n,p)$ with $n = 10000$ and $p = 5/n$. Compute the empirical degree distribution and overlay the theoretical $\text{Poisson}(5)$ PMF.

(b) Generate a BA graph with $n = 10000$ and $m = 3$. Compute the empirical degree distribution on a log-log scale and fit a power law using linear regression on the tail ($k \ge 10$). Report the estimated exponent $\hat{\gamma}$.

(c) Compare the two distributions using a Q-Q plot. What are the key structural differences?

(d) Compute the maximum degree in each model. Derive theoretically why $\max_v \deg(v) = \Theta(\log n / \log \log n)$ for ER and $\Theta(\sqrt{n})$ for BA.

---

**Exercise 3** * - Small-World Analysis

Implement the Watts-Strogatz model from scratch.

(a) Build the ring lattice: $n = 500$ nodes, each connected to $k = 10$ nearest neighbors.

(b) For $\beta \in \{0, 0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0\}$, rewire each edge independently with probability $\beta$. Compute $C(\beta)$ and $L(\beta)$ for each value.

(c) Plot normalized clustering coefficient $C(\beta)/C(0)$ and normalized average path length $L(\beta)/L(0)$ on the same plot (both on log scale for $\beta$). Identify the small-world regime.

(d) Verify that $C(\beta) \approx C(0)(1-\beta)^3$ for small $\beta$. What does this formula say about how clustering degrades with rewiring?

---

**Exercise 4** ** - Stochastic Block Model and Community Detection

(a) Sample an SBM with $n = 500$, $k = 2$ equal communities, $p = a/n$, $q = b/n$. Use $(a,b) = (20, 5)$ (above Kesten-Stigum threshold).

(b) Apply spectral clustering: compute the second eigenvector of the adjacency matrix, threshold at 0 (positive = community 1, negative = community 2), and compute accuracy (fraction of correctly classified nodes, up to community permutation).

(c) Repeat with $(a,b) = (10, 5)$ (near threshold) and $(a,b) = (6, 4)$ (below threshold). How does accuracy vary?

(d) The Kesten-Stigum threshold for 2-block SBM is $(a-b)^2 = 2(a+b)$. Verify your experimental results are consistent with this threshold.

(e) *** Implement the belief propagation algorithm (AMP / approximate message passing) for the 2-block SBM and compare accuracy to spectral clustering near the threshold.

---

**Exercise 5** ** - Wigner's Semicircle Law

(a) Generate a Wigner matrix $W_n = (M + M^\top) / (2\sqrt{n})$ where $M$ has i.i.d. $\text{Normal}(0,1)$ entries, for $n \in \{50, 200, 500, 1000\}$.

(b) Plot the empirical spectral distribution (histogram of eigenvalues) for each $n$. Overlay the theoretical semicircle density $\frac{2}{\pi}\sqrt{4 - x^2}$ for $|x| \le 2$.

(c) Now set $W_n = A/\sqrt{np(1-p)}$ where $A$ is the centered adjacency matrix of $G(n,p)$ with $p = 0.3$, $n = 1000$. Plot the empirical spectral distribution. Identify the outlier eigenvalue corresponding to the average degree.

(d) For the SBM with 2 communities ($n = 1000$, $a = 15$, $b = 3$): plot the eigenvalue spectrum of $A$. Identify which eigenvalues encode community structure and which are bulk noise.

---

**Exercise 6** ** - Graphon Estimation

(a) Generate a sequence of SBM graphs with $n \in \{100, 500, 2000\}$, $k = 3$ communities, and block matrix $B = \begin{pmatrix} 0.8 & 0.1 & 0.1 \\ 0.1 & 0.7 & 0.2 \\ 0.1 & 0.2 & 0.6 \end{pmatrix}$.

(b) For each graph, sort vertices by community label (oracle information) and display the sorted adjacency matrix as a heatmap. Does it converge to the block graphon as $n$ grows?

(c) Without oracle labels: apply spectral clustering to estimate community labels, then display the sorted (estimated) adjacency matrix. Measure the cut distance $d_\square$ between the estimated and true graphon.

(d) Implement the "histogram graphon estimator": divide $[0,1]^2$ into $k^2$ bins and estimate $W$ by averaging edges within each bin. Compute the $L^2$ error $\|W_{\text{est}} - W^*\|_{L^2}$.

---

**Exercise 7** *** - Giant Component Critical Window

This exercise studies the fine-grained behavior near the critical point $p = 1/n$.

(a) For $p = (1 + \lambda n^{-1/3})/n$ with $\lambda \in [-3, 3]$ and $n \in \{500, 2000, 8000\}$, compute $L_1 / n^{2/3}$ for 50 trials each. Plot the distribution of $L_1/n^{2/3}$ vs $\lambda$.

(b) Observe that the distribution has a universal shape (independent of $n$ for large $n$). This is the **Brownian excursion** limit - the critical window scaling is $n^{2/3}$ for the component size and $n^{1/3}$ for the window width.

(c) Compute the mean and standard deviation of $L_1 / n^{2/3}$ as functions of $\lambda$. Show that the mean crosses 0 near $\lambda = 0$ but the transition is smooth (not a jump) at finite $n$.

(d) Compare to the predicted infinite-$n$ limit: for $\lambda > 0$, $\mathbb{E}[L_1/n] \to \beta(1 + \lambda n^{-1/3}) \approx \lambda^2 n^{-1/3} \cdot C$ for some constant $C$. Verify this scaling.

---

**Exercise 8** *** - Preferential Attachment Dynamics

(a) Implement the Barabasi-Albert preferential attachment model for $n = 5000$ nodes with $m = 2$ edges per new node. Use the efficient alias method or linear scan for sampling.

(b) After generation, fit the degree distribution tail to a power law $P(k) = C k^{-\gamma}$ using maximum likelihood estimation on $k \ge k_{\min}$ for suitable $k_{\min}$.

(c) Track the degree of each node over time as the network grows. Plot $k_i(t)$ for nodes added at times $t_i \in \{10, 100, 500, 1000\}$. Verify the mean-field prediction $k_i(t) \approx m\sqrt{t/t_i}$.

(d) Implement "fitness-based" preferential attachment: $\Pi(v) \propto \eta_v \deg(v)$ where $\eta_v \sim \text{Uniform}[0,1]$ is a fixed fitness. Compare the resulting degree distribution to standard BA. Does the power-law exponent change?

(e) Simulate a targeted attack: iteratively remove the highest-degree node. Plot the size of the giant component as a function of fraction of nodes removed. Compare to random removal. What fraction of nodes must be targeted to destroy the giant component?

---

## 12. Why This Matters for AI (2026 Perspective)

| Concept | Impact on AI/ML |
|---------|----------------|
| **ER Phase Transition** | Connectivity threshold determines information flow in GNNs on sparse graphs; GCN cannot aggregate across disconnected components, giving a hard limit on performance |
| **Poisson Degree Distribution** | Most GNN benchmark graphs (Cora, Citeseer) have near-Poisson degrees; GCN's symmetric normalization is optimal for Poisson-degree graphs but suboptimal for scale-free |
| **Kesten-Stigum Threshold** | Hard information-theoretic limit on community detection; no GNN can exceed this on SBM data regardless of architecture, depth, or width |
| **Scale-Free Networks** | Real knowledge graphs (Wikidata, Freebase) are scale-free; hub nodes with $\Theta(\sqrt{n})$ degree dominate GCN aggregation - require degree-aware normalization or degree-bucketing |
| **Small-World Structure** | Transformer attention graphs have small-world properties; BigBird/Longformer mimic WS construction (local window + random long-range); designing optimal sparse attention = finding optimal WS parameters |
| **Graphons** | Theoretical foundation for GNN universality; GNNs that are continuous in cut metric topology can generalize across graph sizes; graphon theory predicts which graph properties a GNN can and cannot learn |
| **Graphon Neural Networks** | First provably universal GNN framework; used to prove that message-passing GNNs cannot distinguish non-isomorphic graphs with identical WL certificates |
| **Spectral Gap** | Controls convergence rate of GNN over-smoothing (energy decays at rate $\lambda_2$); higher Fiedler value -> faster smoothing -> shallower optimal depth |
| **Davis-Kahan** | Explains why spectral GNNs (GCN) succeed above community detection threshold and fail below; same theorem underlies learning theory for graph classification |
| **Wigner Semicircle** | Bulk eigenvalue distribution of noise in graph learning; signal from community structure must exceed semicircle radius to be learnable - same condition as Kesten-Stigum |
| **Configuration Model** | Null model for testing GNN hypotheses: does the GNN learn graph structure, or just degree sequence? Test by evaluating on configuration model graphs with same degree sequence |
| **Random Graph Generation** | DiGress, GDSS, GRAN all learn distributions over graphs - effectively learning graphons. Quality measured by cut distance between learned and true graphon |

---

## 13. Conceptual Bridge

### Backward: What This Builds On

Random graph theory synthesizes several branches of mathematics developed in earlier sections:

**From Spectral Graph Theory (04):** The Laplacian eigenvalues $\lambda_2, \ldots, \lambda_n$ studied there take on probabilistic meaning here - for random graphs, they become random variables following the Wigner semicircle law. The spectral gap $\lambda_2$ that controls mixing time in deterministic graphs now becomes a random quantity whose distribution determines community detectability.

**From Probability Theory (07-Probability-Statistics):** All threshold results (giant component, connectivity, community detection) use first and second moment methods, Chernoff bounds, and the Poisson limit theorem. Branching process theory is the probabilistic tool that gives the exact threshold equation $\beta = 1 - e^{-c\beta}$.

**From Graph Neural Networks (05):** The SBM community detection problem IS the node classification problem that GNNs solve in practice. The Kesten-Stigum threshold gives the hard limit on what any GNN can achieve, while Davis-Kahan shows why spectral GNNs succeed above this threshold.

### Forward: What This Enables

**Functional Analysis (12):** The graphon operator $T_W h(x) = \int_0^1 W(x,y)h(y)\,dy$ is a Hilbert-Schmidt operator on $L^2[0,1]$. Its spectral theory - the Hilbert-Schmidt theorem giving a countable orthonormal eigenfunction expansion - is the infinite-dimensional generalization of the adjacency matrix eigendecomposition. This is the full treatment of graphon operators.

**Graph Algorithms (07):** Random graph models motivate efficient algorithms: spectral clustering (from SBM theory), Louvain community detection (from modularity theory), and link prediction (from random graph null models). The average-case complexity of graph problems is analyzed using random graph models.

**Information Theory (09):** The Kesten-Stigum threshold is an information-theoretic limit - it follows from a channel capacity argument. The mutual information between the community labels $\sigma$ and the observed graph $G$ is zero below the threshold, making recovery impossible regardless of computation. This connects random graphs to channel coding theory.

```
POSITION IN CURRICULUM
========================================================================

  04 Spectral Graph Theory
       |  Laplacians, eigenvalues, Cheeger inequality
       |
       +--> 05 Graph Neural Networks
       |         GCN, GAT, GIN, MPNN, expressiveness
       |
       +--> 06 Random Graphs  <=== YOU ARE HERE
                |
                +- Erdos-Renyi: phase transitions, thresholds
                +- Watts-Strogatz: small-world, navigation
                +- Barabasi-Albert: scale-free, preferential attachment
                +- SBM: communities, Kesten-Stigum threshold
                +- Spectral theory: semicircle law, Davis-Kahan
                +- Graphons: infinite limits, universality
                       |
                       v
                07 Graph Algorithms
                       |
                       v
                12 Functional Analysis <-- graphon operators T_W
                       Hilbert-Schmidt theory, L^2 spectral decomposition

========================================================================
```

The central insight of this section - that random graph models are not merely toy examples but rigorous frameworks for understanding real network behavior - carries forward into every domain where graphs appear. In machine learning, understanding when community structure is detectable, when information can flow across a sparse graph, and when a graph distribution can be learned from samples are fundamental questions with precise mathematical answers, and those answers come from random graph theory.

---

*[<- Back to Graph Theory](../README.md) | [Next: Graph Algorithms ->](../07-Graph-Algorithms/notes.md)*

---

## Appendix A: Branching Process Theory

### A.1 Galton-Watson Branching Processes

A **Galton-Watson process** is a model of population growth where each individual independently produces offspring according to a fixed offspring distribution $\{p_k\}_{k \ge 0}$.

**Formal definition:** Let $Z_0 = 1$ (a single ancestor). At generation $t$, if $Z_t = z$, each of the $z$ individuals produces offspring independently according to $\{p_k\}$:
$$Z_{t+1} = \sum_{i=1}^{Z_t} \xi_i^{(t)}$$
where $\xi_i^{(t)} \sim \{p_k\}$ i.i.d.

**Mean offspring:** $\mu = \sum_k k p_k = \mathbb{E}[\xi]$.

**Probability generating function (PGF):** $\phi(s) = \sum_k p_k s^k = \mathbb{E}[s^\xi]$.

**Extinction probability:** The probability of ultimate extinction $q = \lim_{t \to \infty} \mathbb{P}[Z_t = 0]$ satisfies $q = \phi(q)$.

**Theorem (Extinction):**
- If $\mu \le 1$: $q = 1$ (certain extinction)
- If $\mu > 1$: $q < 1$ (positive survival probability $= 1 - q$)

**Connection to ER:** For $G(n,p)$ with $p = c/n$, exploring the component of a vertex proceeds like a branching process where each node has $\text{Binomial}(n-k, c/n) \approx \text{Poisson}(c)$ offspring (unexplored neighbors). The giant component probability $\beta$ satisfies:
$$1 - \beta = q = e^{-c\beta} = \phi_{\text{Poisson}(c)}(1 - \beta)$$
exactly the fixed-point equation $\beta = 1 - e^{-c\beta}$.

### A.2 Multi-Type Branching Processes

For the SBM with $k$ communities, the local neighborhood exploration is a **multi-type branching process** where the type of an individual is its community label.

**Offspring matrix:** $M_{rs}$ = expected number of type-$s$ offspring from a type-$r$ parent.

For the symmetric 2-block SBM with $p = a/n$, $q = b/n$:
$$M = \frac{1}{2}\begin{pmatrix} a & b \\ b & a \end{pmatrix}$$

**Survival theorem:** The multi-type branching process survives iff $\rho(M) > 1$, where $\rho(M)$ is the spectral radius.

Eigenvalues of $M$: $\lambda_+ = (a+b)/2$, $\lambda_- = (a-b)/2$.

- Giant component exists iff $\lambda_+ > 1$, i.e., $a + b > 2$ (average degree condition).
- Community detection possible iff the "community eigenvalue" $\lambda_- = (a-b)/2 > 1$ ... but this is not quite right. The actual condition involves the non-backtracking matrix.

**Non-backtracking operator:** The correct spectral condition for SBM community detection uses the **non-backtracking (Hashimoto) operator** $B$ on directed edges. The eigenvalue of $B$ corresponding to community structure is $(a-b)/2$, and the bulk spectral radius is $\sqrt{(a+b)/2}$. Community detection is possible iff:
$$\frac{a-b}{2} > \sqrt{\frac{a+b}{2}}$$
i.e., $(a-b)^2 > 2(a+b)$ - precisely the Kesten-Stigum threshold.

### A.3 Critical Branching Processes

At criticality $\mu = 1$ (Poisson offspring with $c = 1$), the branching process has a different behavior:

**Yaglom's theorem:** Conditioned on survival to generation $t$, the population $Z_t / t$ converges in distribution to an Exponential(1) random variable:
$$\mathbb{P}[Z_t / t > x \mid Z_t > 0] \to e^{-x}$$

**For the ER critical window:** At $p = 1/n$, the largest component has size $\Theta(n^{2/3})$ and there are $\Theta(n^{1/3})$ such components. The component size distribution follows Yaglom's theorem for the critical Poisson branching process, but rescaled by $n^{2/3}$.

---

## Appendix B: Configuration Model

### B.1 Definition

The **configuration model** generates a random graph with a prescribed degree sequence $(d_1, d_2, \ldots, d_n)$.

**Construction:**
1. Give vertex $v$ exactly $d_v$ "half-edges" (stubs)
2. Pair up all $2m = \sum_v d_v$ half-edges uniformly at random
3. Each pairing creates an edge

**Result:** A random multigraph (may have self-loops and multi-edges) with the given degree sequence.

**Properties:**
- For degree sequences with bounded maximum degree: the number of self-loops and multi-edges is $O(1)$ - a simple graph w.h.p.
- Conditioned on simplicity: uniform over all simple graphs with the given degree sequence

**Why use it?** The configuration model is the correct null model for testing graph hypotheses. Instead of comparing to ER (wrong degree distribution), compare to configuration model (same degree distribution, no other structure). If a property (e.g., high clustering) exceeds what configuration model predicts, it's genuinely structural, not just a degree artifact.

### B.2 Giant Component in Configuration Model

For the configuration model with degree distribution $P(k)$:

**Molloy-Reed criterion:** A giant component exists iff:
$$\sum_k k(k-2) P(k) > 0 \iff \frac{\mathbb{E}[k^2]}{\mathbb{E}[k]} > 2$$

**Interpretation:** The excess degree distribution $q_k = (k+1)P(k+1)/\mathbb{E}[k]$ governs the branching factor of the exploration process. The process is supercritical (giant component exists) iff the mean of $q_k$ exceeds 1, which is $\mathbb{E}[k(k-1)]/\mathbb{E}[k] > 1$.

**For scale-free networks:** With $P(k) \propto k^{-\gamma}$, $\mathbb{E}[k^2]$ diverges when $\gamma \le 3$. Hence the Molloy-Reed criterion is always satisfied for BA-style scale-free networks - a giant component exists for arbitrarily sparse scale-free graphs.

**For ER:** $P(k) = e^{-c} c^k / k!$ (Poisson), so $\mathbb{E}[k^2] = c^2 + c$ and $\mathbb{E}[k] = c$. Molloy-Reed: $(c^2 + c)/c > 2 \Leftrightarrow c + 1 > 2 \Leftrightarrow c > 1$ - exactly the ER threshold.

### B.3 Clustering in Configuration Model

For the configuration model:
$$C_{\text{conf}} = \frac{(\mathbb{E}[k^2] - \mathbb{E}[k])^2}{n \cdot \mathbb{E}[k]^3} \to 0$$
as $n \to \infty$ (for fixed degree distribution). The configuration model has vanishing clustering coefficient - it's locally tree-like.

**Real networks vs. configuration model:** Comparing observed clustering to $C_{\text{conf}} \approx 0$ tests for genuine clustering beyond degree effects. Small-world networks have $C \gg C_{\text{conf}}$ - they have clustering not explained by degree distribution alone.

---

## Appendix C: Percolation Theory

### C.1 Bond Percolation on Graphs

**Bond percolation:** Given a graph $G$ and probability $\rho \in [0,1]$, independently retain each edge with probability $\rho$. Let $G_\rho$ denote the resulting random subgraph.

**Site percolation:** Independently retain each vertex with probability $\rho$.

**Critical probability:** The percolation threshold $\rho_c$ is the infimum of $\rho$ for which $G_\rho$ has an infinite component (on infinite graphs) or a giant component of size $\Theta(n)$ (on finite graphs).

**On the integer lattice $\mathbb{Z}^d$:** Exact thresholds:
- $d = 1$: $\rho_c = 1$ (must keep all edges)
- $d = 2$ (square lattice): $\rho_c = 1/2$ exactly (by self-duality)
- $d \ge 2$: $\rho_c < 1$; harder to compute exactly

**On ER graphs:** Bond percolation on $G(n,p_0)$ with retention probability $\rho$ gives $G(n, \rho p_0)$. The percolation threshold is $\rho_c = 1/(np_0)$, so the giant component survives iff $\rho > 1/(np_0)$, i.e., $\rho np_0 > 1$.

### C.2 Connection to Neural Network Pruning

Neural network weight pruning is isomorphic to bond percolation:
- Dense network graph $G$ (neurons = nodes, weights = edges)
- Pruning mask $m_{ij} \in \{0,1\}$ with $\mathbb{P}[m_{ij} = 1] = \rho$ (retention probability)
- Pruned network $G_\rho$ must retain computational connectivity

For the network to maintain its computational capacity, it must retain a giant component. The percolation threshold $\rho_c$ gives the minimum retention rate.

**Structured pruning:** Head pruning in transformers (removing entire attention heads) is site percolation on the attention head graph. Magnitude pruning selects edges by weight magnitude - approximately bond percolation with $\rho = $ fraction of weights retained.

**Lottery ticket connection:** A winning lottery ticket is precisely a giant percolating subgraph that retains the "signal paths" of the original network. The existence of such subgraphs at high sparsity ($\rho \approx 0.1$) is guaranteed by percolation theory for sufficiently wide networks.

### C.3 Expanders and Optimal Percolation

**Expander graph:** A $d$-regular graph $G$ on $n$ nodes with spectral gap $\lambda(G) = d - \lambda_2(A_G)$. Large spectral gap means fast mixing and high robustness.

**Percolation on expanders:** For a $d$-regular expander, the percolation threshold is $\rho_c = 1/(d - \lambda(G)/d)^{-1} \approx 1/(d - 1)$ for bounded-degree expanders. The giant component after percolation at $\rho > \rho_c$ has size $\ge (1 - \epsilon)n$ for small $\epsilon$.

**Implication for sparse attention:** Expander-based sparse attention (using Ramanujan graphs with spectral gap $\approx d - 2\sqrt{d-1}$) achieves:
- $O(n)$ edges (efficient)
- $O(\log n)$ diameter (short paths)
- Maximum spectral gap (optimal information flow)
- Robust to random edge removal (good percolation threshold)

This is why expanders are theoretically optimal sparse attention patterns, even if not used in practice due to implementation complexity.

---

## Appendix D: Threshold Functions - Complete Table

| Property $\mathcal{P}$ | Threshold $p^*(n)$ | Window width | Notes |
|------------------------|-------------------|-------------|-------|
| Contains an edge | $n^{-2}$ | $\Theta(p^*)$ | First property to appear |
| Contains a triangle | $n^{-1}$ | $\Theta(p^*)$ | $K_3$ threshold |
| Contains $K_4$ | $n^{-2/3}$ | $\Theta(p^*)$ | $m(K_4) = 3/2$ |
| Contains $K_r$ | $n^{-2/(r-1)}$ | $\Theta(p^*)$ | $m(K_r) = (r-1)/2$ |
| Giant component | $1/n$ | $\Theta(1/(n))$ | Phase transition |
| Connectivity | $\ln(n)/n$ | $1/n$ (sharp!) | Very sharp threshold |
| Diameter $\le 2$ | $\sqrt{\ln(n)/n}$ | $\Theta(p^*)$ | |
| Contains Hamiltonian cycle | $\ln(n)/n$ | $1/n$ | Same as connectivity! |
| Planarity loss | $1/n$ | $\Theta(1/n)$ | Planarity threshold |
| Chromatic number $> k$ | Problem-dependent | Varies | Open for exact $k$ |

**Sharp vs. coarse thresholds:**

A threshold $p^*(n)$ is **sharp** if the transition from probability 0 to probability 1 occurs in a window of width $o(p^*)$. Connectivity and Hamiltonian cycles have sharp thresholds (window width $O(1/n) \ll \ln(n)/n$).

A threshold is **coarse** if the window is $\Theta(p^*)$. Most subgraph appearance thresholds are coarse - the probability transitions from $\epsilon$ to $1-\epsilon$ over a multiplicative constant change in $p$.

**Friedgut's theorem (1999):** Every monotone property has a sharp threshold OR can be reduced to a "small" property. Most natural properties (connectivity, $k$-colorability) have sharp thresholds. This is the technical statement of the Bollobas-Thomason heuristic.

---

## Appendix E: Random Geometric Graphs

### E.1 Definition

A **random geometric graph** $G(n, r)$ is constructed by:
1. Placing $n$ nodes uniformly at random in $[0,1]^2$ (or another metric space)
2. Connecting two nodes iff their Euclidean distance is $\le r$

**Properties:**
- **Soft threshold for connectivity:** $r^* = \sqrt{\ln n / (\pi n)}$ - same $\ln n / n$ scaling as ER
- **High clustering:** Nodes close to each other share many common neighbors (geometric constraint) - $C \to 1$ as $r \to 0$ with $nr^2$ fixed
- **Bounded degree distribution:** All degrees in $[0, \pi n r^2]$
- **No long-range edges:** Maximum edge length $= r$, giving large diameter for small $r$

**For AI:** Random geometric graphs model sensor networks, robotic swarms, and spatial point processes. They also model attention in vision transformers where tokens correspond to image patches at geometric positions - nearby patches should have high attention, distant patches low.

### E.2 Comparison to Other Models

| Property | ER | WS | BA | Geometric |
|----------|----|----|----|-----------| 
| Degree dist. | Poisson | Poisson-like | Power law | Bounded |
| Clustering | Low | High | Low | Very high |
| Path length | $O(\log n)$ | $O(\log n)$ | $O(\log n / \log \log n)$ | $O(1/r)$ |
| Spatial | No | No | No | Yes |
| Community struct. | None | None | None | Implicit (spatial) |

### E.3 Connection to Kernel Methods

The random geometric graph adjacency matrix is a kernel matrix:
$$A_{ij} = \mathbf{1}[\|x_i - x_j\|_2 \le r] = k(x_i, x_j)$$
with kernel $k(x,y) = \mathbf{1}[\|x-y\| \le r]$ (indicator kernel).

More generally, **kernel random graphs** use $A_{ij} \sim \text{Bernoulli}(k(x_i, x_j))$ for a kernel function $k: \mathcal{X}^2 \to [0,1]$. This is a graphon with $W(x,y) = k(x,y)$ for node types $x,y \in \mathcal{X}$.

**For attention:** The softmax attention matrix $\text{softmax}(QK^\top/\sqrt{d})$ is a random kernel matrix where the "positions" are query/key vectors. Random graph theory for kernel random graphs directly applies to study information flow in attention layers.

---

## Appendix F: Network Motifs and Subgraph Statistics

### F.1 Network Motifs

**Definition:** A **network motif** is a subgraph pattern that occurs significantly more often in a real network than in random graphs with the same degree sequence (configuration model null).

**Common motifs:**
- **Feedforward loop** (3-node DAG): overrepresented in gene regulatory networks
- **Bifan** (2->2->2 bipartite): common in neural circuits
- **Clique** ($K_3$, $K_4$): overrepresented in social networks

**Detection:** For each candidate subgraph $H$, compare $t(H, G_{\text{real}})$ (density in real graph) to $\mathbb{E}[t(H, G_{\text{config}})]$ (expected density under null model). Z-score $> 2$: motif. Z-score $< -2$: anti-motif.

**For GNNs:** Motif counting is a proxy for what GNNs learn. Standard MPNNs (GCN, GraphSAGE) can count triangles but not 4-cycles. Higher-order GNNs (OSAN, subgraph GNNs) can count richer motif sets. The motif profile of a graph determines which GNN architecture is most suitable.

### F.2 Triangle Counting at Scale

**Exact triangle count:** For dense graphs, $T = \frac{1}{6}\text{tr}(A^3)$ using matrix multiplication in $O(n^{2.37})$ (matrix multiplication exponent). For sparse graphs ($m = O(n)$), algorithms run in $O(m^{3/2})$.

**Approximate counting:** For massive graphs ($n = 10^9$), use streaming algorithms or random sampling:
- **Wedge sampling:** Count triangles by sampling paths of length 2 and checking closure
- **DOULION:** Sample edges independently with probability $\rho$; count triangles in subgraph; scale by $1/\rho^3$

**Expected triangle count in $G(n,p)$:** $\mathbb{E}[T] = \binom{n}{3} p^3 \approx n^3 p^3 / 6$.

**Concentration:** For $np^3 n^2 \to \infty$, by Azuma-Hoeffding (Lipschitz condition), $T$ concentrates around its mean with standard deviation $O(\text{mean}/\sqrt{n})$.

---

## Appendix G: Information-Theoretic Limits

### G.1 Mutual Information and Detection

**The detection problem:** Given $G \sim \text{SBM}(n, 2, \sigma, p, q)$, recover $\sigma$ (up to global flip).

**Impossible regime:** Below the Kesten-Stigum threshold, the mutual information $I(\sigma; G) = 0$ in the $n \to \infty$ limit. This means the graph $G$ contains literally no information about the community labels that could be extracted by any algorithm.

**Formal statement:** For $p = a/n$, $q = b/n$, $(a-b)^2 \le 2(a+b)$:
$$\lim_{n \to \infty} I(\sigma; G) / n = 0$$

The $/ n$ normalization is because both $\sigma$ and $G$ grow with $n$; the mutual information per node goes to 0.

**Above threshold:** For $(a-b)^2 > 2(a+b)$:
$$\lim_{n \to \infty} I(\sigma; G) / n = h_b\left(\frac{1+\alpha^*}{2}\right) - h_b\left(\frac{1}{2}\right) > 0$$
where $\alpha^*$ is the Bayes-optimal overlap and $h_b$ is binary entropy.

### G.2 Exact Recovery Threshold

For the exact recovery problem (recover $\sigma$ exactly, not just correlate with it), the threshold is higher:

**Theorem (Abbe-Sandon, 2015):** For the 2-block SBM with $p = a \ln(n)/n$, $q = b \ln(n)/n$:
$$\text{Exact recovery is possible w.h.p.} \iff \left(\sqrt{a} - \sqrt{b}\right)^2 > 2$$

This is the CHI-squared divergence condition between the Poisson distributions with means $a$ and $b$.

**The three thresholds:**
1. **Impossible**: $(a-b)^2 \le 2(a+b)$ - no algorithm can detect communities
2. **Weak recovery**: $(a-b)^2 > 2(a+b)$ - algorithms exist to partially recover
3. **Exact recovery**: $(\sqrt{a} - \sqrt{b})^2 > 2$ - algorithms exist for exact recovery

These thresholds become relevant for GNN practitioners when designing tasks: is the community detection problem in your benchmark achievable at all? Which threshold regime does it fall in?

---

## Appendix H: Practical Algorithms

### H.1 Spectral Clustering Pipeline for SBM

```
INPUT: Adjacency matrix A of SBM graph with k communities
OUTPUT: Community assignments \sigma

1. REGULARIZE: Compute A_reg = A + \tau/n * 11^T (small ridge)
   (Prevents leading eigenvector being dominated by high-degree nodes)

2. NORMALIZE: Compute L_sym = D^{-1/2} A_reg D^{-1/2}

3. EIGENVECTORS: Compute top k eigenvectors U \in R^{n\timesk} of L_sym

4. NORMALIZE ROWS: Let U_row = U / ||u_i||_2 (spherical projection)

5. k-MEANS: Run k-means on rows of U_row, get cluster labels \sigma

6. RETURN: \sigma (up to global permutation)

COMPLEXITY: O(n*k/\epsilon^2) for sparse graphs (k power iterations)
ACCURACY: Achieves min error rate above Kesten-Stigum threshold
```

### H.2 Louvain Community Detection

The Louvain algorithm maximizes **modularity** $Q$, a measure of community quality:
$$Q = \frac{1}{2m} \sum_{ij} \left[ A_{ij} - \frac{d_i d_j}{2m} \right] \delta(\sigma_i, \sigma_j)$$

**Phase 1 (local optimization):** For each node $v$, move $v$ to the neighboring community that maximizes $\Delta Q$. Repeat until no improvement.

**Phase 2 (aggregation):** Collapse each community into a supernode. Edge weights between supernodes = total edges between original communities. Apply Phase 1 to the collapsed graph.

**Repeat** until no further improvement.

**Complexity:** $O(n \log n)$ empirically - the fastest practical community detection algorithm.

**Limitation:** Modularity has a resolution limit: communities smaller than $\sqrt{m}$ edges may not be detected. For fine-grained community structure, use alternatives (Infomap, hierarchical spectral methods).

### H.3 Fast Giant Component Detection

```python
def giant_component_fraction(A, n):
    """Compute giant component size via BFS."""
    visited = [False] * n
    max_component = 0
    
    for start in range(n):
        if visited[start]:
            continue
        # BFS from start
        queue = [start]
        visited[start] = True
        component_size = 0
        while queue:
            v = queue.pop(0)
            component_size += 1
            for u in range(n):
                if A[v][u] and not visited[u]:
                    visited[u] = True
                    queue.append(u)
        max_component = max(max_component, component_size)
    
    return max_component / n
```

For sparse adjacency lists (CSR format), BFS runs in $O(n + m)$ - linear in the graph size.

---

*End of 06 Random Graphs notes.*

---

## Appendix I: Advanced Topics in Random Graphs

### I.1 Local Weak Convergence (Benjamini-Schramm)

For **sparse** graph sequences where edge density $\to 0$, graphon theory breaks down. The correct limit theory uses **local weak convergence**.

**Definition (Benjamini-Schramm limit):** A sequence of graphs $G_n$ converges in the local weak sense to a random rooted graph $(G, \rho)$ if for every rooted graph $(H, v)$ and every $r \ge 0$:
$$\frac{1}{n} |\{u \in V(G_n) : B_r(G_n, u) \cong (H, v)\}| \to \mathbb{P}[B_r(G, \rho) \cong (H, v)]$$
where $B_r(G, u)$ is the ball of radius $r$ around $u$ in $G$.

**For $G(n, c/n)$:** The local weak limit is the Galton-Watson tree with Poisson($c$) offspring distribution - an infinite random tree. This confirms the "locally tree-like" structure of sparse ER graphs.

**For BA networks:** The local weak limit involves correlated degree distributions due to preferential attachment - the limiting tree is not a Galton-Watson tree but a Polya urn tree.

**For SBM:** The local weak limit is a multi-type Galton-Watson tree where the types are community labels. This is the probabilistic foundation of the belief propagation algorithm for community detection.

### I.2 Random Regular Graphs

**Definition:** A $d$-regular random graph on $n$ vertices is a graph chosen uniformly at random among all $d$-regular simple graphs on $n$ vertices.

**Construction (configuration model):** Give each vertex $d$ stubs, pair uniformly at random, condition on simplicity.

**Spectral properties:** For $d$-regular random graphs, the eigenvalues are in $[-(2-\epsilon)\sqrt{d-1}, d]$ w.h.p. The Alon-Boppana bound says $\lambda_2 \ge 2\sqrt{d-1} - o(1)$ for any $d$-regular graph. A **Ramanujan graph** achieves equality: $\lambda_2 = 2\sqrt{d-1}$.

**Alon conjecture (proved by Friedman, 2008):** Almost all $d$-regular random graphs are Ramanujan:
$$\lambda_2(G) \le 2\sqrt{d-1} + \epsilon$$
for any fixed $\epsilon > 0$.

**For AI:** Expanders (near-Ramanujan regular graphs) provide the optimal sparse attention pattern:
- $d$ edges per node (linear total edges)
- Diameter $O(\log n)$ (short paths)
- Spectral gap $\approx d - 2\sqrt{d-1}$ (fast mixing)

### I.3 Random Hypergraphs

Real-world networks often have **higher-order interactions** - a chemistry reaction involves multiple molecules simultaneously. Hypergraphs capture this.

**Definition:** A hypergraph $H = (V, E)$ where edges $e \in E$ can contain any number of vertices.

**Random $k$-uniform hypergraph $G^k(n,p)$:** Each $k$-subset of $V$ is an edge independently with probability $p$.

**Phase transition:** The giant component threshold for $G^k(n,p)$ occurs at $p = 1/\binom{n-1}{k-1}$ (average degree 1). The analysis uses a multi-type branching process.

**For AI:** Simplicial complex neural networks (SCNNs) and topological data analysis (TDA) use higher-order graph structure. Random simplicial complexes (random clique complexes of ER graphs) model the topological structure of neural network activation spaces.

### I.4 Dynamic Random Graphs

Real networks evolve over time. Several models capture network dynamics:

**Forest Fire model (Leskovec et al., 2007):** A new node contacts a random "ambassador" and copies some of its links. Exhibits densification (average degree increases over time) and shrinking diameter - both observed in real growing networks.

**Copying model:** Each new node copies $m$ existing edges from a random source node. This generates power-law degree distributions similar to BA but with different exponent depending on copy probability.

**Edge dynamics:** Instead of just adding edges, allow edge rewiring, deletion, and creation. Models temporal networks (who talked to whom at time $t$). The temporal analog of clustering coefficient and path length requires time-respecting paths.

**For AI:** Training data graphs (social networks, citation graphs) evolve over time. Distribution shift in graph ML often comes from temporal evolution of the underlying random graph model. A GNN trained on a 2019 citation graph may fail on a 2024 citation graph if the graph generative process has changed.

---

## Appendix J: Mathematical Proofs

### J.1 Proof: Giant Component Threshold (Upper Bound)

We prove: for $c < 1$, all components have size $O(\log n)$ w.h.p.

**Proof:** Let $X_k$ = number of connected components of size exactly $k$ in $G(n,p)$ with $p = c/n$.

The probability that a specific set $S$ of $k$ vertices forms a component involves:
1. The internal graph on $S$ being connected: probability $\ge (1-p)^{k(k-1)/2} \cdot [\text{connectivity}]$
2. No edges from $S$ to $V \setminus S$: probability $(1-p)^{k(n-k)}$

By union bound:
$$\mathbb{E}[X_k] \le \binom{n}{k} \cdot k^{k-2} p^{k-1} \cdot (1-p)^{k(n-k)}$$

using Cayley's formula ($k^{k-2}$ labeled trees on $k$ vertices, each connected tree is a spanning tree of its component).

For $p = c/n$ and $k \le K \log n$:
$$\mathbb{E}[X_k] \le \frac{n^k}{k!} \cdot k^{k-2} \cdot (c/n)^{k-1} \cdot e^{-ck(n-k)/n}$$

$$\approx \frac{c^{k-1} k^{k-2}}{k! \cdot n} \cdot e^{-ck(1 - k/n)}$$

$$\lesssim \frac{1}{n} \cdot \frac{(ce^{1-c})^k}{k^{3/2}}$$

(using Stirling: $k! \approx k^k e^{-k} \sqrt{2\pi k}$, $k^{k-2}/k! \approx e^k k^{-2}/\sqrt{2\pi k}$)

For $c < 1$: $ce^{1-c} < 1$ (since $f(c) = ce^{1-c}$ has $f(1) = 1$, $f'(1) = 0$, $f''(1) = -1 < 0$ - it's a maximum at $c = 1$). Let $\alpha = ce^{1-c} < 1$.

Summing over $k = 2, 3, \ldots, \infty$:
$$\sum_k \mathbb{E}[X_k] \lesssim \frac{1}{n} \sum_k \alpha^k k^{-3/2} = \frac{C}{n} \to 0$$

By Markov's inequality, the total number of vertices in components of size $\ge 2$ is $O(1) = o(n)$, so the maximum component size is $O(\log n)$. $\square$

### J.2 Proof: Poisson Degree Distribution

**Claim:** For $G(n,p)$ with $p = c/n$, $\deg(v) \xrightarrow{d} \text{Poisson}(c)$.

**Proof via PGF:** The degree $\deg(v) = \sum_{u \neq v} X_{uv}$ where $X_{uv} \sim \text{Bernoulli}(c/n)$ i.i.d.

PGF of $\deg(v)$:
$$G_{\deg}(s) = \mathbb{E}[s^{\deg(v)}] = \prod_{u \neq v} \mathbb{E}[s^{X_{uv}}] = \left(1 - \frac{c}{n} + \frac{cs}{n}\right)^{n-1}$$

Taking $n \to \infty$:
$$\left(1 + \frac{c(s-1)}{n}\right)^{n-1} \to e^{c(s-1)} = \sum_{k=0}^\infty \frac{e^{-c} c^k}{k!} s^k$$

which is the PGF of $\text{Poisson}(c)$.

Since PGF convergence (for $|s| \le 1$) implies distributional convergence (by continuity theorem for generating functions), $\deg(v) \xrightarrow{d} \text{Poisson}(c)$. $\square$

**Multivariate extension:** For distinct vertices $v_1, \ldots, v_m$, their degrees are **jointly** Poisson in the limit, with joint PGF:
$$\mathbb{E}\left[\prod_{i=1}^m s_i^{\deg(v_i)}\right] \to e^{\sum_i c_i(s_i - 1)}$$
where $c_i = c$ for all $i$. But the joint distribution is NOT independent Poisson (edges between $v_i$ and $v_j$ contribute to both $\deg(v_i)$ and $\deg(v_j)$). For fixed $m$, the correlation between degrees of $v_1$ and $v_2$ is $p = c/n \to 0$, so they are asymptotically independent.

### J.3 Proof sketch: Davis-Kahan Theorem

**Simplified version:** Let $A = \bar{A} + W$ where $\bar{A}$ has eigenvalue $\bar{\lambda}$ and unit eigenvector $\bar{u}$. Let $\hat{u}$ be the unit eigenvector of $A$ corresponding to eigenvalue $\hat{\lambda}$ nearest to $\bar{\lambda}$. Let $\delta = \min_{\mu \in \sigma(\bar{A}) \setminus \{\bar{\lambda}\}} |\bar{\lambda} - \mu|$ (gap to other eigenvalues).

**Claim:** $\sin\angle(\hat{u}, \bar{u}) \le \|W\|_{op} / (\delta - \|W\|_{op})$ (assuming $\delta > \|W\|_{op}$).

**Proof:** Decompose $\hat{u} = \alpha \bar{u} + \beta v$ where $v \perp \bar{u}$, $|\alpha|^2 + |\beta|^2 = 1$, and $\sin\angle(\hat{u}, \bar{u}) = |\beta|$.

From $A\hat{u} = \hat{\lambda}\hat{u}$: $\bar{A}\hat{u} + W\hat{u} = \hat{\lambda}\hat{u}$.

Project onto $\bar{u}^\perp$: $P_{\bar{u}^\perp} \bar{A} P_{\bar{u}^\perp} (\beta v) + P_{\bar{u}^\perp} W \hat{u} = \hat{\lambda} \beta v$.

All eigenvalues of $P_{\bar{u}^\perp} \bar{A} P_{\bar{u}^\perp}$ are at distance $\ge \delta$ from $\hat{\lambda}$ (which is close to $\bar{\lambda}$):
$$\|(P_{\bar{u}^\perp} \bar{A} P_{\bar{u}^\perp} - \hat{\lambda} I)^{-1}\|_{op} \le \frac{1}{\delta - |\hat{\lambda} - \bar{\lambda}|}$$

Therefore:
$$|\beta| = \|P_{\bar{u}^\perp}(P_{\bar{u}^\perp} \bar{A} P_{\bar{u}^\perp} - \hat{\lambda} I)^{-1} P_{\bar{u}^\perp} W \hat{u}\| \le \frac{\|W\|_{op}}{\delta - \|W\|_{op}}$$

This gives the Davis-Kahan bound. $\square$

---

## Appendix K: Notation Reference

| Symbol | Meaning | First appears |
|--------|---------|--------------|
| $G(n,p)$ | Erdos-Renyi random graph, $n$ vertices, edge prob $p$ | 3.1 |
| $G(n,m)$ | ER graph with exactly $m$ edges | 3.1 |
| $L_1(G)$ | Size of largest connected component | 3.3 |
| $\beta(c)$ | Giant component fraction, $\beta = 1 - e^{-c\beta}$ | 3.3 |
| $C_v$ | Local clustering coefficient of vertex $v$ | 4.2 |
| $L$ | Average path length | 4.3 |
| $\Pi(v)$ | Preferential attachment probability of vertex $v$ | 5.1 |
| $\text{SBM}(n, k, \sigma, B)$ | Stochastic Block Model | 6.1 |
| $B \in [0,1]^{k\times k}$ | SBM block probability matrix | 6.1 |
| $\sigma: [n] \to [k]$ | Community assignment | 6.1 |
| $\rho(M)$ | Spectral radius of matrix $M$ | App. A.2 |
| $\mu_{sc}$ | Wigner semicircle measure | 7.1 |
| $d_\square(G, H)$ | Cut metric between graphs $G$ and $H$ | 8.1 |
| $W: [0,1]^2 \to [0,1]$ | Graphon | 8.2 |
| $t(F, W)$ | Homomorphism density of $F$ in $W$ | 8.3 |
| $T_W$ | Graphon integral operator, $T_Wh(x) = \int W(x,y)h(y)dy$ | 8.4 |
| $\text{w.h.p.}$ | With high probability (probability $\to 1$) | 2.2 |
| $o(\cdot), O(\cdot), \Theta(\cdot), \omega(\cdot)$ | Asymptotic notation | 2.2 |
| $I(c) = c - 1 - \ln c$ | Large deviation function for ER | 3.3 |
| $\phi(s)$ | Probability generating function | App. A.1 |
| $q_k$ | Excess degree distribution | App. B.2 |

---

*[<- Back to Graph Theory](../README.md) | [Next: Graph Algorithms ->](../07-Graph-Algorithms/notes.md)*

---

## Appendix L: Worked Examples

### L.1 Computing the Giant Component Fraction

**Problem:** For $G(n,p)$ with $p = 2.5/n$, find the theoretical fraction of vertices in the giant component.

**Solution:** We need $\beta$ satisfying $\beta = 1 - e^{-2.5\beta}$.

Define $f(\beta) = 1 - e^{-2.5\beta} - \beta$. We need $f(\beta) = 0$.

- $f(0) = 0$ (trivial solution - subcritical regime would have this as the only root)
- $f'(\beta) = 2.5 e^{-2.5\beta} - 1$; $f'(0) = 1.5 > 0$ (slope at 0 is positive)
- $f$ is concave for $\beta > 0$, so there is one non-trivial root

**Newton's method:** Starting from $\beta_0 = 0.7$:
- $f(0.7) = 1 - e^{-1.75} - 0.7 = 1 - 0.1738 - 0.7 = 0.1262$
- $f'(0.7) = 2.5 \cdot 0.1738 - 1 = -0.5655$
- $\beta_1 = 0.7 - 0.1262/(-0.5655) = 0.7 + 0.2232 = 0.923$ (overshoot, try smaller)

Using bisection between 0.7 and 0.95:
- $f(0.8) = 1 - e^{-2} - 0.8 = 1 - 0.1353 - 0.8 = 0.0647 > 0$
- $f(0.85) = 1 - e^{-2.125} - 0.85 = 1 - 0.1194 - 0.85 = 0.0306 > 0$
- $f(0.88) \approx 1 - 0.1108 - 0.88 = 0.0092 > 0$
- $f(0.89) \approx 1 - 0.1084 - 0.89 = 0.0016 > 0$
- $f(0.895) \approx 1 - 0.1072 - 0.895 = -0.0022 < 0$

So $\beta \approx 0.892$ - about 89.2% of vertices are in the giant component.

**Interpretation:** At average degree 2.5, the network is well into the supercritical regime. The vast majority of vertices are connected in one giant component, leaving only about 10.8% in small satellite components.

### L.2 Kesten-Stigum Threshold Calculation

**Problem:** For the 2-block SBM with $a = 15$ and $b = 4$, determine:
(a) Is community detection possible?
(b) Can we achieve exact recovery?

**Solution:**

**(a) Detection (weak recovery):** Kesten-Stigum threshold: $(a-b)^2 > 2(a+b)$

$(15-4)^2 = 11^2 = 121$

$2(15+4) = 2 \cdot 19 = 38$

$121 > 38$ OK - **Detection is possible.**

SNR $= (a-b)^2 / (2(a+b)) = 121/38 \approx 3.18 \gg 1$.

**(b) Exact recovery:** Need $(\sqrt{a} - \sqrt{b})^2 > 2$ (scaled threshold for logarithmic degree regime).

$\sqrt{15} \approx 3.873$, $\sqrt{4} = 2$.

$(\sqrt{15} - \sqrt{4})^2 = (3.873 - 2)^2 = (1.873)^2 = 3.51 > 2$ OK - **Exact recovery is achievable.**

**Interpretation:** With $a = 15$ and $b = 4$, we have a relatively easy community detection problem - both detection and exact recovery are achievable. Spectral clustering should work well here.

### L.3 Small-World Parameter Calculation

**Problem:** Design a WS graph on $n = 1000$ nodes that approximates the C. elegans neural connectome: $C \approx 0.28$, $L \approx 2.65$, $k \approx 14$.

**Solution:**

**Step 1:** Initial ring lattice has $C(0) = 3(k-2)/(4(k-1)) = 3 \cdot 12/(4 \cdot 13) = 36/52 \approx 0.692$.

**Step 2:** Target clustering $C(\beta) \approx 0.28$. Using $C(\beta) \approx C(0)(1-\beta)^3$:
$$0.28 \approx 0.692 (1-\beta)^3$$
$$(1-\beta)^3 \approx 0.404$$
$$1-\beta \approx 0.739, \quad \beta \approx 0.261$$

**Step 3:** Verify path length at $\beta = 0.261$. Using the WS approximation $L(\beta) \approx \frac{n}{k} \cdot f(nk\beta/2)$ where $f(u) \approx \frac{\log u}{u}$ for large $u$:

$nk\beta/2 = 1000 \cdot 14 \cdot 0.261 / 2 = 1827$

$L \approx (1000/14) \cdot \log(1827)/1827 \approx 71.4 \cdot 7.51/1827 \approx 0.29$

This is too small - the approximation breaks down at large $\beta$. Numerically, $L(\beta = 0.26) \approx 3.5$ for $n=1000$, $k=14$, which is reasonably close to the target $L = 2.65$.

**Conclusion:** WS parameters $(n=1000, k=14, \beta \approx 0.26)$ produce a graph close to C. elegans in both clustering and path length, validating the small-world model.

### L.4 Graphon Estimation from Data

**Problem:** Given a graph $G$ with 100 nodes known to be sampled from a 3-block SBM, estimate the graphon.

**Procedure:**

1. **Run spectral clustering** with $k=3$ to get estimated community labels $\hat{\sigma}$.

2. **Sort vertices** by $\hat{\sigma}$: reorder adjacency matrix so community 1 nodes come first, then community 2, then community 3.

3. **Estimate block probabilities:** For communities $r, s$:
$$\hat{B}_{rs} = \frac{|\{(u,v) \in E : \hat{\sigma}(u) = r, \hat{\sigma}(v) = s\}|}{|\{(u,v) : \hat{\sigma}(u) = r, \hat{\sigma}(v) = s\}|}$$

4. **Construct step-function graphon:**
$$\hat{W}(x, y) = \hat{B}_{\lceil 3x \rceil, \lceil 3y \rceil}$$
(piecewise constant on $[0,1]^2$ divided into $3 \times 3$ blocks)

5. **Evaluate quality:** Compute cut distance $d_\square(\hat{W}, W^*)$ where $W^*$ is the true graphon. The cut distance converges to 0 as $n \to \infty$.

**Error bound:** For a $k$-block SBM, the best graphon estimator achieves error $O(k/\sqrt{n})$ in cut distance (minimax optimal). With $n = 100$ and $k = 3$, expected cut distance $\approx 3/10 = 0.3$ - substantial estimation uncertainty.

---

## Appendix M: Connections to Statistical Physics

### M.1 Random Graphs and Spin Glasses

The Stochastic Block Model community detection problem is isomorphic to the **ferromagnetic Ising model** on the graph:

- Community labels $\sigma_v \in \{+1, -1\}$ correspond to spin states
- Edges within communities are "ferromagnetic" (prefer aligned spins)
- Edges between communities are "antiferromagnetic" (prefer anti-aligned)

The Kesten-Stigum threshold corresponds to the **Nishimori temperature** in the disordered Ising model - the temperature at which the magnetization (overlap with ground truth) first becomes nonzero.

**Belief propagation = Cavity method:** The belief propagation algorithm for community detection in SBM is the Bethe approximation applied to the Ising model. Near the KS threshold, BP is asymptotically optimal - no polynomial-time algorithm can do better.

### M.2 Percolation and Statistical Mechanics

Bond percolation on $\mathbb{Z}^2$ is a model in statistical mechanics. Its critical phenomena:
- **Order parameter:** $P_\infty(\rho) = \mathbb{P}[\text{vertex is in infinite component}]$
- **Critical exponent:** $P_\infty(\rho) \sim (\rho - \rho_c)^\beta$ with $\beta = 5/36$ (exact, 2D)
- **Correlation length:** $\xi(\rho) \sim |\rho - \rho_c|^{-\nu}$ with $\nu = 4/3$ (exact, 2D)

For ER graphs: $P_\infty(c) = \beta(c)$ with critical exponent 1 (mean-field universality class, since ER has "infinite dimension").

**Conformal invariance (2D percolation):** Near criticality, 2D percolation is conformally invariant - the scaling limit is described by SLE (Schramm-Loewner Evolution). This deep connection between combinatorics and complex analysis has no direct ML application yet, but illustrates the richness of the field.

### M.3 Free Energy and the Replica Method

The **replica method** is a non-rigorous but powerful physics technique for computing expectations over random graphs.

For the SBM community detection problem, the free energy (log-partition function) per vertex is:
$$f = \lim_{n \to \infty} \frac{1}{n} \mathbb{E}[\log Z(\sigma, G)]$$

where $Z = \sum_\sigma e^{H(\sigma, G)}$ and $H$ is the log-likelihood of the community assignment.

The replica calculation predicts the exact phase diagram of the SBM - both the KS threshold and the exact recovery threshold - and was later made rigorous using interpolation methods (Guerra, Talagrand).

**ML connection:** Free energy calculations in statistical physics are the rigorous foundation of variational inference in machine learning. The Bethe free energy (used in belief propagation) is the physics analog of the ELBO (evidence lower bound) in variational autoencoders.

---

*End of 06-Random-Graphs notes.*

---

## Appendix N: Extended Exercise Solutions

### N.1 Solution Notes for Exercise 1 (Phase Transition)

**Key implementation details:**

The fixed-point equation $\beta = 1 - e^{-c\beta}$ can be solved numerically via Newton's method:
$$\beta_{k+1} = \beta_k - \frac{\beta_k - 1 + e^{-c\beta_k}}{1 - ce^{-c\beta_k}}$$

Starting from $\beta_0 = 1 - e^{-c}$ (one step of Picard iteration), Newton's method converges in 5-10 iterations for $c \in [1, 5]$.

For $c \le 1$: the only fixed point is $\beta = 0$ (return 0).
For $c > 1$: there are two fixed points (0 and the positive root); return the positive root.

**Expected results table:**

| $c$ | Theoretical $\beta(c)$ | Simulated $L_1/n$ (variance) |
|-----|----------------------|------------------------------|
| 0.5 | 0 | $O(\log n / n) \approx 0.004$ |
| 0.8 | 0 | $O(\log n / n) \approx 0.006$ |
| 1.0 | 0 (but $n^{-1/3}$ scaling) | $\approx 0.15 \cdot n^{-1/3}$ |
| 1.5 | 0.583 | $0.583 \pm 0.015$ |
| 2.0 | 0.797 | $0.797 \pm 0.008$ |
| 2.5 | 0.892 | $0.892 \pm 0.005$ |
| 3.0 | 0.940 | $0.940 \pm 0.003$ |

**Why $L_2$ peaks at criticality:** The second-largest component size peaks at $c = 1$ (size $\sim n^{2/3}$) because this is where the giant component is just forming - there are many large components competing. For $c > 1$, the giant component absorbs everything, leaving only $O(\log n)$ satellites.

### N.2 Solution Notes for Exercise 4 (Community Detection)

**Spectral algorithm implementation:**

```python
def spectral_community_detection(A, k=2):
    """Spectral clustering for SBM community detection."""
    n = A.shape[0]
    # Compute degree matrix and normalized Laplacian
    d = A.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d + 1e-10))
    L_sym = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
    
    # Eigendecomposition (smallest eigenvalues = community structure)
    eigvals, eigvecs = np.linalg.eigh(L_sym)
    
    # Second eigenvector (first non-trivial)
    u2 = eigvecs[:, 1]
    
    # Threshold at 0
    labels = (u2 > 0).astype(int)
    return labels

def community_accuracy(labels_est, labels_true):
    """Fraction correct, up to global flip."""
    acc1 = np.mean(labels_est == labels_true)
    acc2 = np.mean(labels_est == (1 - labels_true))
    return max(acc1, acc2)
```

**Expected accuracy table:**

| $(a, b)$ | SNR = $(a-b)^2/(2(a+b))$ | Regime | Expected accuracy |
|----------|--------------------------|--------|------------------|
| $(20, 5)$ | $225/50 = 4.5$ | Well above KS | 0.95-0.99 |
| $(10, 5)$ | $25/30 = 0.83$ | Below KS | $\approx 0.5$ (random) |
| $(6, 4)$ | $4/20 = 0.2$ | Below KS | $\approx 0.5$ (random) |
| $(8, 3)$ | $25/22 = 1.14$ | Just above KS | 0.55-0.65 |

### N.3 Solution Notes for Exercise 8 (Preferential Attachment)

**Efficient alias sampling for preferential attachment:**

The naive approach (linear scan to sample proportional to degree) is $O(n)$ per edge, giving $O(n^2)$ total. For large $n$, use the **alias method**:

1. Maintain a list of $(d_v, v)$ pairs for all existing nodes
2. Use the alias method to sample from this distribution in $O(1)$ per sample
3. After adding new edges, update the alias table in $O(m)$ per step

Alternatively, use the **Vose-Walker** alias method which supports updates efficiently.

**Simpler $O(n \log n)$ approach:** Maintain a binary indexed tree (Fenwick tree) over cumulative degrees. Sampling is $O(\log n)$ per edge, updating is $O(\log n)$ per edge, giving $O(nm \log n)$ total.

**Power law fit:** Maximum likelihood estimator for power-law exponent on data $k_1, \ldots, k_m$ with $k_i \ge k_{\min}$:
$$\hat{\gamma} = 1 + m \left[\sum_{i=1}^m \ln\frac{k_i}{k_{\min} - 0.5}\right]^{-1}$$

For BA with $m = 2$ and $k_{\min} = 10$: expect $\hat{\gamma} \approx 3.0 \pm 0.2$ for $n = 10000$.

---

## Appendix O: Further Reading

### O.1 Textbooks

1. **Bollobas, B.** (2001). *Random Graphs* (2nd ed.). Cambridge University Press.
   - The definitive mathematical treatment; covers ER theory exhaustively.

2. **Janson, S., Luczak, T., Rucinski, A.** (2000). *Random Graphs*. Wiley.
   - More accessible than Bollobas; covers first and second moment methods thoroughly.

3. **Lovasz, L.** (2012). *Large Networks and Graph Limits*. AMS.
   - The definitive treatment of graphon theory by its creator.

4. **Durrett, R.** (2007). *Random Graph Dynamics*. Cambridge University Press.
   - Dynamical random graphs and their applications to epidemics and social networks.

### O.2 Foundational Papers

1. **Erdos, P., Renyi, A.** (1960). On the evolution of random graphs. *Magyar Tud. Akad. Mat. Kutato Int. Kozl.*, 5, 17-61.

2. **Watts, D.J., Strogatz, S.H.** (1998). Collective dynamics of 'small-world' networks. *Nature*, 393, 440-442.

3. **Barabasi, A.-L., Albert, R.** (1999). Emergence of scaling in random networks. *Science*, 286, 509-512.

4. **Lovasz, L., Szegedy, B.** (2006). Limits of dense graph sequences. *J. Combin. Theory Ser. B*, 96, 933-957.

5. **Mossel, E., Neeman, J., Sly, A.** (2015). Reconstruction and estimation in the planted partition model. *Probab. Theory Related Fields*, 162, 431-461.

6. **Abbe, E., Sandon, C.** (2015). Community detection in the stochastic block models by spectral methods. *arXiv:1503.00609*.

### O.3 ML-Specific Papers

1. **Palowitch, J. et al.** (2022). GraphWorld: Fake Graphs Bring Real Insights for GNNs. *KDD 2022*.

2. **Keriven, N., Peyre, G.** (2019). Universal Invariant and Equivariant Graph Neural Networks. *NeurIPS 2019*.

3. **Vignac, C. et al.** (2022). DiGress: Discrete Denoising diffusion for graph generation. *ICLR 2023*.

4. **Rusch, T.K., Bronstein, M.M., Mishra, S.** (2023). A survey on oversmoothing in graph neural networks. *arXiv:2303.10993*.

5. **Broido, A.D., Clauset, A.** (2019). Scale-free networks are rare. *Nature Communications*, 10, 1017.

---

*[<- Back to Graph Theory](../README.md) | [Next: Graph Algorithms ->](../07-Graph-Algorithms/notes.md)*

---

## Appendix P: Summary of Key Inequalities

### P.1 Concentration Inequalities for Random Graphs

**Markov's inequality:** For non-negative $X$:
$$\mathbb{P}[X \ge t] \le \frac{\mathbb{E}[X]}{t}$$

**Chebyshev's inequality:**
$$\mathbb{P}[|X - \mathbb{E}[X]| \ge t] \le \frac{\text{Var}(X)}{t^2}$$

**Chernoff bound (Poisson random variables):** For $X \sim \text{Poisson}(\mu)$:
$$\mathbb{P}[X \ge (1+\delta)\mu] \le e^{-\mu \delta^2 / (2+\delta)}$$
$$\mathbb{P}[X \le (1-\delta)\mu] \le e^{-\mu \delta^2 / 2}$$

**Paley-Zygmund inequality:** For non-negative $X$ with finite second moment:
$$\mathbb{P}[X > 0] \ge \frac{(\mathbb{E}[X])^2}{\mathbb{E}[X^2]}$$

**Azuma-Hoeffding (Martingale concentration):** If $(X_0, X_1, \ldots, X_m)$ is a martingale with $|X_k - X_{k-1}| \le c_k$:
$$\mathbb{P}[|X_m - X_0| \ge t] \le 2\exp\left(-\frac{t^2}{2\sum_k c_k^2}\right)$$

For random graph properties: reveal edges one at a time. Each edge revelation changes the property value by at most $c_k$ (Lipschitz constant of the property). Azuma-Hoeffding gives concentration of order $\sqrt{m} = O(n)$ around the mean.

**McDiarmid's inequality (Bounded differences):** If $f(G)$ changes by at most $c$ when one edge is added/removed:
$$\mathbb{P}[|f(G) - \mathbb{E}[f(G)]| \ge t] \le 2\exp\left(-\frac{t^2}{2\binom{n}{2}c^2}\right)$$

Applications: triangle count ($c = O(n)$ per edge change), largest clique ($c = 1$), chromatic number ($c = 1$).

### P.2 Spectral Inequalities

**Perron-Frobenius:** For non-negative irreducible $A$, $\lambda_1(A) > 0$ and the leading eigenvector has all positive entries.

**Courant-Fischer (Min-Max):** The $k$-th eigenvalue of symmetric $A$ satisfies:
$$\lambda_k(A) = \min_{\dim V = n-k+1} \max_{\mathbf{x} \in V, \|\mathbf{x}\|=1} \mathbf{x}^\top A \mathbf{x}$$

**Weyl's theorem:** For symmetric $A$ and $E$:
$$|\lambda_k(A+E) - \lambda_k(A)| \le \|E\|_{op} \quad \forall k$$

**Bauer-Fike:** If $A$ is diagonalizable with condition number $\kappa$, eigenvalue perturbations are bounded by $\kappa \|E\|$. For symmetric $A$: $\kappa = 1$ (always), recovering Weyl's theorem.

**Cheeger inequality:** For graph Laplacian $L$, the Cheeger constant $h(G) = \min_{S: |S| \le n/2} |E(S, \bar{S})| / |S|$ satisfies:
$$\frac{\lambda_2}{2} \le h(G) \le \sqrt{2\lambda_2}$$

This is the fundamental connection between algebraic (spectral gap) and geometric (cut quality) properties of graphs.

---

*[<- Back to Graph Theory](../README.md) | [Next: Graph Algorithms ->](../07-Graph-Algorithms/notes.md)*

---

## Appendix Q: Quick Reference Card

```
RANDOM GRAPH QUICK REFERENCE
========================================================================

  ERDOS-RENYI G(n,p)
  ------------------
  - Average degree:   np
  - Degree dist:      Poisson(np) as n->\infty
  - Giant component:  exists iff p > 1/n  ->  fraction \beta(c) w/ c=np
  - Connectivity:     w.h.p. iff p \geq ln(n)/n
  - Diameter:         ~ log(n)/log(np) when connected
  - Clustering:       ~ p -> 0  (locally tree-like)

  WATTS-STROGATZ (n, k, \beta)
  -------------------------
  - Degree:            \approx k (Poisson-like spread from rewiring)
  - Clustering:        C(0)(1-\beta)^3  where C(0) \approx 3/4 for large k
  - Path length:       O(log n) for any \beta > 0
  - Small-world:       \beta \in [1/(nk), 0.3] gives C\ggC_ER and L\approxL_ER

  BARABASI-ALBERT (n, m)
  -----------------------
  - Degree dist:       P(k) ~ 2m^2/k^3  (power law \gamma=3)
  - Max degree:        ~ m\sqrtn  (oldest node)
  - Diameter:          ~ log(n)/log(log(n))
  - Clustering:        C ~ (log n)^2/n -> 0  (no clustering!)

  STOCHASTIC BLOCK MODEL SBM(n,k,\sigma,B)
  -------------------------------------
  - KS threshold:      (a-b)^2 = 2(a+b)  for 2-block, p=a/n, q=b/n
  - Exact recovery:    (\sqrta - \sqrtb)^2 = 2   for log-degree regime
  - Spectral alg:      works above KS threshold
  - GNN limit:         No GNN beats KS threshold on SBM data

  GRAPHONS W:[0,1]^2->[0,1]
  -------------------------
  - Dense graph limit: every dense G-sequence converges to a graphon
  - Sampling:          draw \xi^i~U[0,1]; edge (i,j) w.p. W(\xi^i,\xi_j)
  - ER graphon:        W(x,y) = p  (constant)
  - SBM graphon:       W(x,y) = B_{\lceilkx\rceil,\lceilky\rceil}  (step function)
  - Operator:          T_W h(x) = \int_0^1 W(x,y)h(y)dy  (Hilbert-Schmidt)

========================================================================
```
