[← Back to Information Theory](../README.md) | [Next: KL Divergence →](../02-KL-Divergence/notes.md)

---

# Entropy

> _"Information is the resolution of uncertainty."_ — Claude Shannon

## Overview

Entropy is the central quantity of information theory — a precise, mathematical measure of uncertainty, surprise, and information content. Introduced by Claude Shannon in his landmark 1948 paper *A Mathematical Theory of Communication*, entropy transformed our understanding of communication, computation, and inference. Where nineteenth-century thermodynamics defined entropy as a measure of physical disorder, Shannon showed that the same mathematical structure governs the information carried by any probabilistic source, from telegraph signals to protein sequences to transformer language models.

For machine learning practitioners, entropy is ubiquitous. The cross-entropy loss that trains every classifier is entropy in disguise. The perplexity that evaluates every language model is an entropy estimate. The information gain that splits every decision tree is an entropy difference. The soft policy entropy bonus in soft actor-critic prevents premature convergence. Entropy appears not as an abstract concept but as a concrete, computable quantity that governs the fundamental limits of compression, inference, and learning.

This section builds entropy from first principles: self-information (§2), the Shannon entropy formula and its standard properties (§3), joint and conditional entropy with the chain rule (§4), the maximum entropy principle (§5), source coding and data compression (§6), entropy rates of stochastic processes including language models (§7), differential entropy for continuous variables (§8), and Rényi and Tsallis generalizations (§9). Every concept is developed mathematically and connected to its role in modern AI systems.

**Scope note.** This section is the canonical home for entropy. The closely related quantities — KL divergence (§02), mutual information (§03), cross-entropy loss (§04), and Fisher information (§05) — each have their own dedicated sections in this chapter. Where they appear here, they appear as brief previews with forward references.

## Prerequisites

- **Probability distributions** — PMFs, PDFs, expectations — [Chapter 6: Probability Theory](../../06-Probability-Theory/README.md)
- **Logarithm rules** — $\log(ab) = \log a + \log b$, $\log(1/p) = -\log p$ — [Chapter 1](../../01-Mathematical-Foundations/README.md)
- **Expected values** — $\mathbb{E}[f(X)] = \sum_x p(x) f(x)$ — [Chapter 6 §02](../../06-Probability-Theory/02-Random-Variables/notes.md)
- **Jensen's inequality** — for convex $\phi$: $\phi(\mathbb{E}[X]) \le \mathbb{E}[\phi(X)]$ — [Chapter 8 §01](../../08-Optimization/01-Convex-Optimization/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Interactive derivations: entropy computations, MaxEnt proofs, entropy rate of Markov chains, perplexity, differential entropy |
| [exercises.ipynb](exercises.ipynb) | 8 graded exercises from binary entropy to MaxEnt RL entropy bonus |

## Learning Objectives

After completing this section, you will:

1. Define self-information $I(x) = -\log p(x)$ and explain why the logarithm is the unique choice satisfying the additivity axiom
2. Compute Shannon entropy $H(X)$ for discrete distributions and verify the bounds $0 \le H(X) \le \log |\mathcal{X}|$
3. Prove that entropy is concave in the probability vector $\mathbf{p}$ and derive the maximum-entropy distribution for given constraints
4. Compute joint entropy $H(X,Y)$, conditional entropy $H(X \mid Y)$, and verify the chain rule
5. Apply the maximum entropy principle to derive the Gaussian, uniform, and exponential distributions from moment constraints
6. State Shannon's source coding theorem and construct a Huffman code for a given source
7. Compute the entropy rate of a Markov chain and connect it to language model perplexity
8. Define differential entropy $h(X)$, compute it for Gaussian and uniform distributions, and identify its key differences from discrete entropy
9. Parametrize the Rényi entropy family $H_\alpha(X)$ and identify the limiting cases (Hartley, Shannon, min-entropy)
10. Explain how entropy appears in decision trees, MaxEnt RL (SAC), confidence calibration, and LLM perplexity evaluation

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 What Is Entropy?](#11-what-is-entropy)
  - [1.2 Why Entropy Matters for AI](#12-why-entropy-matters-for-ai)
  - [1.3 Historical Timeline](#13-historical-timeline)
  - [1.4 Units and the Base Question](#14-units-and-the-base-question)
- [2. Self-Information and Shannon Entropy](#2-self-information-and-shannon-entropy)
  - [2.1 Self-Information](#21-self-information)
  - [2.2 Shannon Entropy: Definition](#22-shannon-entropy-definition)
  - [2.3 Computing Entropy for Standard Distributions](#23-computing-entropy-for-standard-distributions)
  - [2.4 Non-Examples and Edge Cases](#24-non-examples-and-edge-cases)
- [3. Properties of Shannon Entropy](#3-properties-of-shannon-entropy)
  - [3.1 Non-Negativity and Boundedness](#31-non-negativity-and-boundedness)
  - [3.2 Concavity](#32-concavity)
  - [3.3 Subadditivity](#33-subadditivity)
  - [3.4 Invariance Under Bijections](#34-invariance-under-bijections)
  - [3.5 Continuity](#35-continuity)
- [4. Joint Entropy, Conditional Entropy, and the Chain Rule](#4-joint-entropy-conditional-entropy-and-the-chain-rule)
  - [4.1 Joint Entropy](#41-joint-entropy)
  - [4.2 Conditional Entropy](#42-conditional-entropy)
  - [4.3 Chain Rule for Entropy](#43-chain-rule-for-entropy)
  - [4.4 Venn Diagram of Information Quantities](#44-venn-diagram-of-information-quantities)
- [5. Maximum Entropy Principle](#5-maximum-entropy-principle)
  - [5.1 MaxEnt Statement](#51-maxent-statement)
  - [5.2 MaxEnt Derivation via Lagrange Multipliers](#52-maxent-derivation-via-lagrange-multipliers)
  - [5.3 Exponential Family from MaxEnt](#53-exponential-family-from-maxent)
  - [5.4 MaxEnt in Machine Learning](#54-maxent-in-machine-learning)
- [6. Source Coding and Data Compression](#6-source-coding-and-data-compression)
  - [6.1 Shannon's Source Coding Theorem](#61-shannons-source-coding-theorem)
  - [6.2 Prefix Codes and Kraft's Inequality](#62-prefix-codes-and-krafts-inequality)
  - [6.3 Huffman Coding](#63-huffman-coding)
  - [6.4 Range Coding in LLM Inference](#64-range-coding-in-llm-inference)
- [7. Entropy Rate of Stochastic Processes](#7-entropy-rate-of-stochastic-processes)
  - [7.1 Entropy Rate: Definition](#71-entropy-rate-definition)
  - [7.2 Entropy Rate of Markov Chains](#72-entropy-rate-of-markov-chains)
  - [7.3 Perplexity as Entropy Rate Estimate](#73-perplexity-as-entropy-rate-estimate)
- [8. Differential Entropy](#8-differential-entropy)
  - [8.1 Definition and Subtleties](#81-definition-and-subtleties)
  - [8.2 Differential Entropy of Standard Distributions](#82-differential-entropy-of-standard-distributions)
  - [8.3 Maximum Differential Entropy: The Gaussian](#83-maximum-differential-entropy-the-gaussian)
- [9. Rényi Entropy and Generalizations](#9-rényi-entropy-and-generalizations)
  - [9.1 Rényi Entropy](#91-rényi-entropy)
  - [9.2 Min-Entropy and Security](#92-min-entropy-and-security)
  - [9.3 Tsallis Entropy](#93-tsallis-entropy)
- [10. Applications in Machine Learning](#10-applications-in-machine-learning)
  - [10.1 Decision Tree Splits](#101-decision-tree-splits)
  - [10.2 Entropy Regularization in RL](#102-entropy-regularization-in-rl)
  - [10.3 Confidence Calibration](#103-confidence-calibration)
  - [10.4 Preview: Cross-Entropy, KL, Mutual Information, Fisher Information](#104-preview-cross-entropy-kl-mutual-information-fisher-information)
- [11. Common Mistakes](#11-common-mistakes)
- [12. Exercises](#12-exercises)
- [13. Why This Matters for AI (2026 Perspective)](#13-why-this-matters-for-ai-2026-perspective)
- [14. Conceptual Bridge](#14-conceptual-bridge)

---

## 1. Intuition

### 1.1 What Is Entropy?

Imagine you are handed a message before reading it. How surprised will you be? If the message is from a source that always sends the same symbol — say, a machine that outputs `A` with probability 1 — you learn nothing; no surprise is possible. If the source sends `A` or `B` with equal probability $\frac{1}{2}$, you learn exactly one bit of information when the symbol arrives. If the source can send any of 256 characters with equal probability, you learn 8 bits per symbol.

Entropy quantifies this intuition precisely. It is the **average amount of information** (or equivalently, the average surprise) produced by a probabilistic source. Rare events are more surprising than common ones — learning that a 1-in-a-million event occurred tells you far more than learning that a coin came up heads. The entropy of a distribution is the expected value of this surprise, averaged over all possible outcomes.

Three equivalent ways to think about entropy:

1. **Uncertainty before the fact.** $H(X)$ measures how uncertain you are about $X$ before observing it. A uniform distribution over $n$ outcomes has maximum entropy $\log n$; a deterministic distribution has entropy $0$.

2. **Information gained after the fact.** $H(X)$ measures how much information (in bits, nats, or hartleys) you gain on average when you observe $X$. This is why entropy equals the minimum average code length for transmitting $X$.

3. **Compression limit.** $H(X)$ bits per symbol is the theoretical minimum average code length for losslessly encoding a source with distribution $p$. No code can do better, and Huffman coding comes within 1 bit of this bound.

The formal definition makes all three views precise:

$$H(X) = -\sum_{x \in \mathcal{X}} p(x) \log p(x) = \mathbb{E}[-\log p(X)]$$

The minus sign appears because $\log p(x) \le 0$ for $p(x) \in (0,1]$, and we want entropy to be non-negative.

**Intuitive examples.** A fair coin: $H = \log 2 = 1$ bit. A biased coin ($p = 0.9$): $H \approx 0.469$ bits — less uncertainty because heads is much more likely. A fair 6-sided die: $H = \log_2 6 \approx 2.585$ bits. A fair deck of 52 cards: $H = \log_2 52 \approx 5.7$ bits.

**The key insight for AI.** Language is a stochastic source. Each token in a sequence is drawn from a distribution over the vocabulary. The entropy of this distribution measures how uncertain (or creative) the language model is. A model that always predicts the next token with certainty has zero entropy and is not a useful language model. A model with moderate entropy is generating diverse, informative text.

### 1.2 Why Entropy Matters for AI

Entropy is not merely a theoretical construct — it is load-bearing mathematics in the most common ML operations:

**Cross-entropy loss.** The most common training objective in classification and language modeling is:

$$\mathcal{L} = -\frac{1}{n}\sum_{i=1}^n \log p_{\boldsymbol{\theta}}(y^{(i)} \mid \mathbf{x}^{(i)})$$

This is the empirical estimate of the cross-entropy $H(p_{\mathrm{data}}, p_{\boldsymbol{\theta}})$. Minimizing it minimizes the KL divergence from the data distribution to the model distribution. Every GPT, LLaMA, Claude, and Gemini model is trained by minimizing cross-entropy. The connection to entropy is developed fully in [§04 Cross-Entropy](../04-Cross-Entropy/notes.md).

**Perplexity.** The standard metric for evaluating language models is:

$$\operatorname{PPL} = \exp\!\left(-\frac{1}{T}\sum_{t=1}^T \log p(x_t \mid x_{<t})\right)$$

This is $2^H$ (in nats, $e^H$) — the exponentiated per-token entropy of the model's predictions. A perplexity of 10 means the model is, on average, as uncertain as a uniform distribution over 10 tokens. Lower perplexity = lower entropy = better compression = better language model.

**Decision trees.** The information gain criterion for splitting a node is:

$$\Delta H = H(Y) - H(Y \mid X_j)$$

This chooses the feature $X_j$ that maximally reduces the entropy of the target $Y$. ID3, C4.5, and the CART (entropy variant) all use this criterion.

**Entropy regularization.** Soft Actor-Critic (SAC) and other maximum-entropy RL algorithms add $\alpha H(\pi(\cdot \mid s_t))$ to the reward, encouraging the policy to maintain high entropy (explore diverse actions). This prevents premature collapse to deterministic policies and substantially improves sample efficiency.

**Confidence calibration.** The entropy of a model's output distribution $H(p_{\boldsymbol{\theta}}(\cdot \mid \mathbf{x}))$ is a measure of predictive uncertainty. High-entropy outputs signal that the model is unsure; low-entropy outputs signal confidence. Temperature scaling adjusts this entropy calibration.

### 1.3 Historical Timeline

```
HISTORY OF ENTROPY
════════════════════════════════════════════════════════════════════════

  1865  Rudolf Clausius coins "entropy" for thermodynamics (disorder)
  1877  Ludwig Boltzmann: S = k log W (statistical mechanics)
  1929  Leo Szilard: information and thermodynamic entropy connected
  1948  Claude Shannon: H(X) = -∑ p log p (information theory born)
        "A Mathematical Theory of Communication", Bell System Tech. J.
  1951  Kullback & Leibler: KL divergence as entropy-relative measure
  1959  Peter Elias, David Huffman: optimal prefix codes
  1961  Alfréd Rényi: generalized entropies H_α
  1976  Edwin Jaynes: MaxEnt as Bayesian inference framework
  1991  Thomas Cover & Joy Thomas: "Elements of Information Theory"
        (still the standard graduate textbook)
  2000s Arithmetic coding / range coding for LLM compression
  2018+ Perplexity as universal LLM benchmark; entropy loss as
        training objective for every major language model
  2018  SAC (Haarnoja et al.): maximum-entropy RL goes mainstream
  2022+ LLM inference with speculative decoding; token entropy
        as early-exit signal; entropy-based uncertainty quantification

════════════════════════════════════════════════════════════════════════
```

### 1.4 Units and the Base Question

The formula $H(X) = -\sum_x p(x) \log p(x)$ depends on the base of the logarithm:

| Base | Unit | Symbol | Conversion |
| --- | --- | --- | --- |
| $\log_2$ | bit (binary digit) | b | 1 nat = $\log_2 e \approx 1.443$ bits |
| $\ln = \log_e$ | nat (natural unit) | nat | 1 bit = $\ln 2 \approx 0.693$ nats |
| $\log_{10}$ | hartley (dit) | H | rarely used in ML |

**Which base does ML use?**

- **Theory and proofs:** nats ($\ln$) — calculus is cleaner, no constant factors in derivatives
- **Compression and coding theory:** bits ($\log_2$) — matches binary representation
- **Perplexity computation:** nats in code, reported as base-$e$ exponent, but "bits per token" uses base-2

In this section, $\log$ denotes the natural logarithm unless otherwise specified. When working with bits, we write $\log_2$. The choice of base scales entropy by a constant factor and does not affect any mathematical property or ordering of distributions.

**For AI:** PyTorch's `nn.CrossEntropyLoss` and `F.cross_entropy` use natural logarithm internally (bits are not relevant for gradient computation). Perplexity is reported as $e^{\bar{\ell}}$ where $\bar{\ell}$ is the average negative log-likelihood in nats. The distinction matters when comparing reported perplexity values across papers — verify whether they use $e$ or $2$ as the base.

---

## 2. Self-Information and Shannon Entropy

### 2.1 Self-Information

Before defining entropy, we need to define the information content of a single outcome.

**Definition (Self-Information).** The **self-information** (or **surprise**) of an event $\{X = x\}$ with probability $p(x)$ is:

$$I(x) = -\log p(x) = \log \frac{1}{p(x)}$$

Self-information is the unique function satisfying three natural axioms:

1. **Monotonicity.** $I(x)$ is a decreasing function of $p(x)$: rarer events carry more information.
2. **Non-negativity.** $I(x) \ge 0$ for all $p(x) \in (0,1]$.
3. **Additivity for independent events.** If $X \perp\!\!\!\perp Y$, then $I(x,y) = I(x) + I(y)$.

The third axiom forces the logarithm. If $f(p \cdot q) = f(p) + f(q)$ for all $p, q \in (0,1]$ and $f$ is monotone decreasing, then $f(p) = -c \log p$ for some constant $c > 0$. The constant $c$ sets the unit; choosing $c = 1/\ln 2$ gives bits.

**Examples:**
- Fair coin, outcome heads: $I(\text{heads}) = -\log_2(1/2) = 1$ bit
- Fair die, outcome 6: $I(6) = -\log_2(1/6) \approx 2.585$ bits
- English letter 'e' (frequency $\approx 12.7\%$): $I(\text{e}) \approx 2.98$ bits
- English letter 'z' (frequency $\approx 0.07\%$): $I(\text{z}) \approx 10.8$ bits
- Token `" the"` in GPT (very common): low self-information, e.g. $\approx 2$–$4$ nats
- A rare technical token: high self-information

**The surprise interpretation.** If your language model assigns probability $p = 0.001$ to the next token, and that token occurs, the model is "surprised" by $-\ln(0.001) \approx 6.9$ nats. If $p = 0.9$ and the token occurs, surprise is $-\ln(0.9) \approx 0.105$ nats. The negative log-likelihood of a prediction is literally how surprised the model is.

### 2.2 Shannon Entropy: Definition

**Definition (Shannon Entropy).** Let $X$ be a discrete random variable with probability mass function $p: \mathcal{X} \to [0,1]$. The **Shannon entropy** of $X$ is:

$$H(X) = -\sum_{x \in \mathcal{X}} p(x) \log p(x)$$

with the convention $0 \log 0 = 0$ (since $\lim_{p \to 0^+} p \log p = 0$).

Equivalently, entropy is the expected self-information:

$$H(X) = \mathbb{E}[-\log p(X)] = \mathbb{E}[I(X)]$$

This is a key identity: entropy is the **expected surprise** of the distribution. A high-entropy distribution is one where you are, on average, very surprised by outcomes. A low-entropy distribution is one where outcomes are mostly predictable.

**The convention $0 \log 0 = 0$.** This is justified by continuity: $\lim_{p \to 0^+} p \log p = 0$. Outcomes with probability zero do not contribute to entropy, which makes sense — an impossible outcome carries infinite self-information, but it never occurs, so its contribution to the expected surprise is zero.

**Entropy depends only on $\mathbf{p}$.** $H(X)$ depends only on the probability vector $(p(x_1), \ldots, p(x_n))$, not on the actual values $x_1, \ldots, x_n$. Renaming outcomes does not change entropy.

**Functional notation.** When we want to emphasize dependence on the probability vector, we write $H(\mathbf{p})$ or $H(p_1, \ldots, p_n)$ where $p_i = p(x_i)$.

### 2.3 Computing Entropy for Standard Distributions

**Bernoulli distribution.** Let $X \sim \operatorname{Bern}(p)$: $P(X=1) = p$, $P(X=0) = 1-p$.

$$H(X) = -p\log p - (1-p)\log(1-p) =: h(p)$$

This is the **binary entropy function** $h(p)$. Key values:
- $h(0) = h(1) = 0$ (deterministic — no uncertainty)
- $h(1/2) = \log 2 = 1$ bit (maximum — fair coin)
- $h(0.1) \approx 0.469$ bits; $h(0.01) \approx 0.081$ bits

The function $h(p)$ is symmetric about $p = 1/2$, concave, and achieves its unique maximum at $p = 1/2$.

**Categorical distribution.** Let $X \sim \operatorname{Cat}(\mathbf{p})$ with $|\mathcal{X}| = n$ outcomes.

$$H(X) = -\sum_{i=1}^n p_i \log p_i$$

Special cases:
- Uniform: $p_i = 1/n$ for all $i$ → $H = \log n$ (maximum)
- Deterministic: $p_k = 1$ for some $k$ → $H = 0$ (minimum)

**Geometric distribution.** Let $X \sim \operatorname{Geom}(p)$: $P(X=k) = (1-p)^{k-1}p$ for $k \ge 1$.

$$H(X) = \frac{-(1-p)\log(1-p) - p\log p}{p} = \frac{h(p)}{p}$$

For small $p$ (rare successes), $H \approx \frac{1}{p}(\log \frac{1}{p} - 1)$ → large entropy (many trials needed).

**Poisson distribution.** Let $X \sim \operatorname{Poisson}(\lambda)$. No closed form, but:

$$H(X) = \lambda(1 - \log \lambda) + e^{-\lambda}\sum_{k=0}^\infty \frac{\lambda^k \log k!}{k!}$$

For large $\lambda$: $H(X) \approx \frac{1}{2}\log(2\pi e\lambda)$.

**Uniform distribution over $n$ elements.** $H = \log n$ — the unique distribution achieving the upper bound. This is one reason uniform distributions arise in MaxEnt problems.

**For AI — Token Distribution Entropy.** In language model inference, each forward pass produces a categorical distribution over $|\mathcal{V}|$ vocabulary tokens (often $|\mathcal{V}| \approx 32{,}000$–$128{,}000$). The entropy of this output distribution is:
- Low when the model is confident (e.g., completing `"The capital of France is _"`)
- High when the model is uncertain (e.g., completing `"The best approach to _"`)

Monitoring token entropy during inference is used for uncertainty quantification, early exit, and speculative decoding.

### 2.4 Non-Examples and Edge Cases

**Non-example 1: Entropy cannot be negative (for discrete distributions).** Since $p(x) \in [0,1]$ and $-\log p(x) \ge 0$, each term $-p(x)\log p(x) \ge 0$, so $H(X) \ge 0$ always. *(Differential entropy for continuous distributions can be negative — see §8.)*

**Non-example 2: High-cardinality does not guarantee high entropy.** A distribution over $10^6$ outcomes with $p(x_1) = 0.9999$ and equal probability for the remaining outcomes has very low entropy despite having many possible values. Entropy measures the shape of the distribution, not just its support size.

**Non-example 3: Equal entropy does not mean equal distributions.** Two distributions can have the same entropy but completely different shapes. For example, $\mathbf{p} = (0.5, 0.5)$ and $\mathbf{q} = (0.9, 0.05, 0.05)$ both have entropy $1$ bit, but they are very different distributions. *(To compare distributions, use KL divergence — see §02.)*

**Non-example 4: $H(X) = 0$ does not mean $X$ is constant.** It means $X$ is **almost surely constant** — i.e., some outcome has probability 1 and all others have probability 0. $H(X) = 0 \iff X$ is deterministic.

**Edge case: Infinite alphabet.** For countably infinite $\mathcal{X}$, $H(X)$ can be infinite. For example, $P(X = k) = c/k^2$ has finite entropy, while $P(X = k) \propto 1/k$ does not (harmonic series). In practice, vocabulary sizes are finite so this edge case does not arise in LLMs.

**Edge case: $p(x) = 0$.** By convention, $0 \cdot \log 0 = 0$. This is the correct limit and ensures that adding probability-zero outcomes to a distribution does not change its entropy.

---

## 3. Properties of Shannon Entropy

### 3.1 Non-Negativity and Boundedness

**Theorem.** For any discrete random variable $X$ with $|\mathcal{X}| = n$:

$$0 \le H(X) \le \log n$$

**Lower bound ($H \ge 0$).** Each term $-p(x)\log p(x) \ge 0$ since $p(x) \in [0,1]$ implies $\log p(x) \le 0$. Equality $H = 0$ holds iff $p(x) \in \{0,1\}$ for all $x$, i.e., $X$ is deterministic.

**Upper bound ($H \le \log n$).** This follows from the **log-sum inequality** or from the **non-negativity of KL divergence**. Let $u(x) = 1/n$ be the uniform distribution. Then:

$$D_{\mathrm{KL}}(p \| u) = \sum_x p(x)\log \frac{p(x)}{u(x)} = \log n - H(X) \ge 0$$

The last inequality holds because KL divergence is always non-negative (proven in §02). Therefore $H(X) \le \log n$. Equality holds iff $p = u$, i.e., $X$ is uniform.

**Implications for ML:**
- A language model's output entropy over $|\mathcal{V}|$ tokens is bounded above by $\log |\mathcal{V}|$. A model outputting the uniform distribution has maximum entropy and zero confidence.
- Entropy approaching $0$ means the model is becoming deterministic — either very confident (good) or overfit to a degenerate output (bad, e.g., mode collapse in generation).

### 3.2 Concavity

**Theorem.** The entropy function $H(\mathbf{p})$ is **strictly concave** on the probability simplex $\Delta_n = \{\mathbf{p} \in \mathbb{R}^n : p_i \ge 0, \sum_i p_i = 1\}$.

**Proof.** We must show that for any $\mathbf{p}, \mathbf{q} \in \Delta_n$ and $\lambda \in [0,1]$:

$$H(\lambda\mathbf{p} + (1-\lambda)\mathbf{q}) \ge \lambda H(\mathbf{p}) + (1-\lambda) H(\mathbf{q})$$

This follows because $f(t) = -t\log t$ is concave on $(0,\infty)$ (since $f''(t) = -1/t < 0$), and $H(\mathbf{p}) = \sum_i f(p_i)$. A sum of concave functions is concave.

Strict concavity: equality holds iff $\mathbf{p} = \mathbf{q}$.

**Consequence: Mixing increases entropy.** If a mixture source outputs $X$ from distribution $p$ with probability $\lambda$ and from distribution $q$ with probability $1-\lambda$, the entropy of the mixture is at least as large as the weighted average of the individual entropies. Intuitively, mixing introduces additional uncertainty about which sub-source generated the output.

**For ML:** The concavity of entropy means that the **maximum entropy distribution** over a convex constraint set is unique. This underpins the MaxEnt principle (§5). It also means that gradient-based optimization of entropy (as in SAC) is well-behaved — there is a unique maximum.

### 3.3 Subadditivity

**Theorem (Subadditivity of Entropy).** For any two jointly distributed random variables $X$, $Y$:

$$H(X, Y) \le H(X) + H(Y)$$

with equality if and only if $X \perp\!\!\!\perp Y$.

**Proof.** Using the definition of joint entropy:

$$H(X) + H(Y) - H(X,Y) = \sum_{x,y} p(x,y)\log\frac{p(x,y)}{p(x)p(y)} = D_{\mathrm{KL}}(p(x,y) \| p(x)p(y))$$

Since $D_{\mathrm{KL}} \ge 0$, we have $H(X,Y) \le H(X) + H(Y)$. Equality holds iff $p(x,y) = p(x)p(y)$ for all $x,y$, i.e., $X \perp\!\!\!\perp Y$.

**Intuition.** The joint entropy is the total uncertainty about the pair $(X,Y)$. If $X$ and $Y$ share information (are dependent), then knowing $X$ reduces uncertainty about $Y$, so the total joint uncertainty is less than the sum of individual uncertainties.

**The quantity $H(X) + H(Y) - H(X,Y) = I(X;Y)$** is the **mutual information** — the amount of information $X$ and $Y$ share. This is the subject of [§03 Mutual Information](../03-Mutual-Information/notes.md). Subadditivity is equivalent to $I(X;Y) \ge 0$.

**For ML:** In multi-task learning, the subadditivity of entropy bounds how much information the combined tasks share. In representation learning (e.g., InfoNCE objectives in contrastive learning), maximizing mutual information between views is equivalent to reducing the gap in $H(X) + H(Y) - H(X,Y)$.

### 3.4 Invariance Under Bijections

**Theorem.** If $f: \mathcal{X} \to \mathcal{Y}$ is a bijection (one-to-one and onto), then $H(f(X)) = H(X)$.

**Proof.** A bijection preserves the probability values: $p_{f(X)}(y) = p_X(f^{-1}(y))$. Since entropy depends only on the multiset of probability values, it is unchanged.

**Non-bijective case.** If $f$ is not injective (many-to-one), then $H(f(X)) \le H(X)$ — collapsing outcomes reduces entropy. This is the **data processing inequality** for functions of $X$: processing cannot increase entropy.

**For ML:** Lossless tokenization is a bijection on the original character sequence, so the entropy of the token-level distribution is determined by the character-level entropy plus the log of the average number of characters per token. This connects token-level perplexity to character-level perplexity.

### 3.5 Continuity

**Theorem.** $H: \Delta_n \to \mathbb{R}_{\ge 0}$ is a continuous function of $\mathbf{p}$.

The proof uses the fact that $-t \log t$ is continuous on $[0,1]$ (with value $0$ at $t=0$ by convention), and a finite sum of continuous functions is continuous.

**Fano's inequality.** A quantitative version of continuity: if $X$ and $\hat{X} = g(Y)$ are such that $P(\hat{X} \ne X) = p_e$, then:

$$H(X \mid Y) \le h(p_e) + p_e \log(|\mathcal{X}| - 1)$$

This bounds the conditional entropy (residual uncertainty) given the probability of error. Fano's inequality is a fundamental tool for proving converse channel capacity theorems, and it is covered in depth in [§03 Mutual Information](../03-Mutual-Information/notes.md).

---

## 4. Joint Entropy, Conditional Entropy, and the Chain Rule

### 4.1 Joint Entropy

**Definition (Joint Entropy).** For jointly distributed random variables $X$ and $Y$ with joint PMF $p(x,y)$:

$$H(X, Y) = -\sum_{x \in \mathcal{X}}\sum_{y \in \mathcal{Y}} p(x,y) \log p(x,y)$$

Joint entropy measures the total uncertainty about the pair $(X,Y)$.

**Properties:**
- $H(X,Y) \ge \max(H(X), H(Y))$ — the joint entropy is at least as large as either marginal
- $H(X,Y) \le H(X) + H(Y)$ — subadditivity (proven above)
- $H(X,Y) = H(X) + H(Y)$ iff $X \perp\!\!\!\perp Y$
- $H(X,Y) = H(X)$ iff $Y = f(X)$ for some deterministic function $f$ — if $Y$ is determined by $X$, knowing $X$ gives you $Y$ for free

**Generalization.** For $n$ jointly distributed variables $X_1, \ldots, X_n$:

$$H(X_1, \ldots, X_n) = -\sum_{x_1, \ldots, x_n} p(x_1, \ldots, x_n) \log p(x_1, \ldots, x_n)$$

By subadditivity: $H(X_1, \ldots, X_n) \le \sum_{i=1}^n H(X_i)$ with equality iff all $X_i$ are mutually independent.

**For AI:** In a language model with context $x_1, \ldots, x_{T-1}$, the joint entropy $H(x_1, \ldots, x_T)$ is the total information in a sequence of $T$ tokens. The entropy rate (§7) is $H(x_1,\ldots,x_T)/T$ per token — the per-token information content in the limit.

### 4.2 Conditional Entropy

**Definition (Conditional Entropy).** The **conditional entropy** of $X$ given $Y$ is:

$$H(X \mid Y) = \sum_{y \in \mathcal{Y}} p(y) H(X \mid Y = y) = -\sum_{x,y} p(x,y)\log p(x \mid y)$$

This is the **expected entropy of $X$ after observing $Y$** — the residual uncertainty about $X$ that remains after knowing $Y$.

**Key identity:**

$$H(X \mid Y) = H(X, Y) - H(Y)$$

**Proof:**
$$H(X, Y) - H(Y) = -\sum_{x,y}p(x,y)\log p(x,y) + \sum_y p(y)\log p(y)$$
$$= -\sum_{x,y}p(x,y)[\log p(x,y) - \log p(y)] = -\sum_{x,y}p(x,y)\log\frac{p(x,y)}{p(y)}$$
$$= -\sum_{x,y}p(x,y)\log p(x \mid y) = H(X \mid Y)$$

**Properties:**
- $0 \le H(X \mid Y) \le H(X)$ — conditioning never increases entropy
- $H(X \mid Y) = H(X)$ iff $X \perp\!\!\!\perp Y$ — knowing $Y$ tells you nothing about $X$
- $H(X \mid Y) = 0$ iff $X$ is a deterministic function of $Y$ — knowing $Y$ determines $X$ completely
- $H(X \mid X) = 0$ — knowing $X$ eliminates all uncertainty about $X$

**Important subtlety:** $H(X \mid Y) \ne H(X \mid Y = y)$ in general. The conditional entropy $H(X \mid Y)$ is an expectation over $y$; the quantity $H(X \mid Y = y)$ is the entropy of the conditional distribution $p(\cdot \mid Y=y)$ for a specific value $y$.

**For AI — Next-Token Prediction:** In a language model, $H(x_t \mid x_1, \ldots, x_{t-1})$ is the conditional entropy of the $t$-th token given the context. This is exactly the per-token uncertainty that the model must compress. The average of this over $t$ is the entropy rate. The model's prediction $p_{\boldsymbol{\theta}}(x_t \mid x_{<t})$ attempts to approximate this conditional distribution.

### 4.3 Chain Rule for Entropy

**Theorem (Chain Rule for Entropy).**

$$H(X_1, X_2, \ldots, X_n) = \sum_{i=1}^n H(X_i \mid X_1, \ldots, X_{i-1})$$

with the convention $H(X_1 \mid X_0) = H(X_1)$ (no conditioning for the first term).

**Proof.** By induction using $H(X_1, X_2) = H(X_1) + H(X_2 \mid X_1)$, which follows from the identity $H(X \mid Y) = H(X,Y) - H(Y)$. The general case follows by repeatedly applying this identity.

**Special cases:**
- $n=2$: $H(X,Y) = H(X) + H(Y \mid X) = H(Y) + H(X \mid Y)$
- i.i.d. case: all $X_i$ i.i.d., so $H(X_i \mid X_{<i}) = H(X_1)$ and $H(X_1,\ldots,X_n) = n \cdot H(X_1)$

**For AI — Language Model Factorization:** Every autoregressive language model uses the chain rule for entropy (or rather, for probability via the chain rule of probability):

$$\log p(x_1, \ldots, x_T) = \sum_{t=1}^T \log p(x_t \mid x_1, \ldots, x_{t-1})$$

The average negative log-likelihood $-\frac{1}{T}\log p(x_1,\ldots,x_T)$ is the empirical estimate of the entropy rate $\frac{1}{T}H(x_1,\ldots,x_T)$, which converges to the true entropy rate of the language distribution as $T \to \infty$.

### 4.4 Venn Diagram of Information Quantities

The relationships among entropy quantities can be visualized using a Venn diagram where the area of each region represents a quantity:

```
VENN DIAGRAM OF INFORMATION QUANTITIES
════════════════════════════════════════════════════════════════════════

  ┌──────────────────────────────────────────┐
  │          H(X,Y)  (joint entropy)         │
  │  ┌───────────────────┐                   │
  │  │      H(X)         │                   │
  │  │  ┌────────┬───────┤───────────────┐   │
  │  │  │ H(X|Y) │ I(X;Y)│   H(Y|X)     │   │
  │  │  │        │       │               │   │
  │  └──┴────────┴───────┴───────────────┘   │
  │             H(Y)                         │
  └──────────────────────────────────────────┘

  H(X)     = H(X|Y) + I(X;Y)      [X's total entropy]
  H(Y)     = H(Y|X) + I(X;Y)      [Y's total entropy]
  H(X,Y)   = H(X) + H(Y|X)        [chain rule]
           = H(Y) + H(X|Y)        [chain rule, reversed]
           = H(X) + H(Y) - I(X;Y) [via mutual information]
  I(X;Y)   = H(X) + H(Y) - H(X,Y) [mutual information]
           = H(X) - H(X|Y)        [= reduction in uncertainty]
           = H(Y) - H(Y|X)        [symmetric]

════════════════════════════════════════════════════════════════════════
```

The quantity $I(X;Y)$ in the overlap is the **mutual information** — the information shared between $X$ and $Y$. When $X \perp\!\!\!\perp Y$, the circles do not overlap: $I(X;Y) = 0$. This quantity is the subject of [§03 Mutual Information](../03-Mutual-Information/notes.md).

---

## 5. Maximum Entropy Principle

### 5.1 MaxEnt Statement

The **maximum entropy principle** (Jaynes, 1957) states:

> Among all probability distributions consistent with given constraints (known information), choose the one with maximum entropy.

The MaxEnt distribution is the "most uninformative" distribution given what you know — it makes no assumptions beyond the specified constraints. This is not a heuristic but a theorem: the MaxEnt distribution is the unique distribution consistent with the constraints that arises as the uniform distribution over microstate descriptions given macroscopic constraints (Jaynes' maximum entropy formalism).

**Why maximum entropy?**
- Any distribution with lower entropy is implicitly asserting additional structure not in the constraints
- The MaxEnt distribution is the unique one that is invariant under all symmetries of the constraint set
- It minimizes the KL divergence from the uniform distribution (proof in §02)

**Examples of MaxEnt distributions:**
- Constraint: $X \in \{1,\ldots,n\}$ → MaxEnt: uniform distribution $\operatorname{Uniform}\{1,\ldots,n\}$
- Constraint: $X \ge 0$, $\mathbb{E}[X] = \mu$ → MaxEnt: exponential distribution $\operatorname{Exp}(1/\mu)$
- Constraint: $\mathbb{E}[X] = \mu$, $\operatorname{Var}(X) = \sigma^2$ → MaxEnt: Gaussian $\mathcal{N}(\mu, \sigma^2)$

### 5.2 MaxEnt Derivation via Lagrange Multipliers

**Problem.** Maximize $H(\mathbf{p}) = -\sum_{x} p(x)\log p(x)$ subject to:
1. $\sum_x p(x) = 1$ (normalization)
2. $\mathbb{E}[f_k(X)] = \alpha_k$ for $k = 1, \ldots, m$ (moment constraints)

**Method: Lagrange multipliers.** Form the Lagrangian:

$$\mathcal{L}(\mathbf{p}, \lambda_0, \boldsymbol{\lambda}) = -\sum_x p(x)\log p(x) - \lambda_0\!\left(\sum_x p(x) - 1\right) - \sum_k \lambda_k \!\left(\sum_x p(x)f_k(x) - \alpha_k\right)$$

Setting $\partial \mathcal{L}/\partial p(x) = 0$:

$$-\log p(x) - 1 - \lambda_0 - \sum_k \lambda_k f_k(x) = 0$$

Solving:

$$p^*(x) = \exp\!\left(-1 - \lambda_0 - \sum_k \lambda_k f_k(x)\right) = \frac{1}{Z(\boldsymbol{\lambda})} \exp\!\left(-\sum_k \lambda_k f_k(x)\right)$$

where $Z(\boldsymbol{\lambda}) = \sum_x \exp(-\sum_k \lambda_k f_k(x))$ is the **partition function** (normalization constant). The $\lambda_k$ are chosen to satisfy the moment constraints $\mathbb{E}_{p^*}[f_k(X)] = \alpha_k$.

**Result.** The MaxEnt distribution given moment constraints $\mathbb{E}[f_k(X)] = \alpha_k$ is always of the exponential family form:

$$\boxed{p^*(x) = \frac{1}{Z(\boldsymbol{\lambda})} \exp\!\left(-\sum_k \lambda_k f_k(x)\right)}$$

**Worked example — Gaussian as MaxEnt.** Maximize $h(X)$ subject to $\mathbb{E}[X] = \mu$ and $\mathbb{E}[X^2] = \sigma^2 + \mu^2$. The constraints are $f_1(x) = x$ and $f_2(x) = x^2$. The MaxEnt distribution is:

$$p^*(x) \propto \exp(-\lambda_1 x - \lambda_2 x^2)$$

Completing the square: $p^*(x) \propto \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$, which is exactly $\mathcal{N}(\mu, \sigma^2)$.

**Worked example — Uniform as MaxEnt.** With no moment constraints (only normalization), the MaxEnt distribution is uniform: $p^*(x) = 1/n$. Any other distribution would assign higher probability to some outcomes than others, asserting structure not in the constraints.

### 5.3 Exponential Family from MaxEnt

The MaxEnt derivation shows that the exponential family:

$$p(x; \boldsymbol{\eta}) = h(x) \exp(\boldsymbol{\eta}^\top \mathbf{T}(x) - A(\boldsymbol{\eta}))$$

arises naturally whenever we impose moment constraints on the sufficient statistics $\mathbf{T}(x)$. Conversely, every exponential family distribution is the MaxEnt distribution for some set of moment constraints.

| Distribution | Constraints | Natural parameter |
| --- | --- | --- |
| Uniform$(0,1)$ | $X \in [0,1]$, no moments | None |
| Exponential$(\lambda)$ | $X \ge 0$, $\mathbb{E}[X] = 1/\lambda$ | $\eta = -\lambda$ |
| Gaussian$(\mu, \sigma^2)$ | $\mathbb{E}[X] = \mu$, $\operatorname{Var}(X) = \sigma^2$ | $\eta = (\mu/\sigma^2, -1/2\sigma^2)$ |
| Poisson$(\lambda)$ | $X \in \mathbb{N}_0$, $\mathbb{E}[X] = \lambda$ | $\eta = \log\lambda$ |
| Bernoulli$(p)$ | $X \in \{0,1\}$, $\mathbb{E}[X] = p$ | $\eta = \log(p/(1-p))$ |

**Boltzmann distribution.** In statistical physics, the MaxEnt distribution under fixed mean energy $\mathbb{E}[E(x)] = U$ is the **Boltzmann distribution** $p(x) \propto e^{-\beta E(x)}$ where $\beta = 1/(kT)$ is the inverse temperature. Shannon entropy is formally identical to the Gibbs-Boltzmann thermodynamic entropy.

### 5.4 MaxEnt in Machine Learning

**Temperature in language model sampling.** The softmax operation in LLMs:

$$p_\tau(x) = \frac{e^{z_x / \tau}}{\sum_{x'} e^{z_{x'}/\tau}}$$

is the MaxEnt distribution under fixed mean logit $\mathbb{E}[z_X] = \bar{z}$, where $\tau$ is the temperature. High $\tau \to 1$ (uniform, maximum entropy); low $\tau \to 0$ (argmax, minimum entropy). This is why temperature is the natural control knob for generation diversity.

**MaxEnt language models.** Before neural LLMs, MaxEnt models were the dominant NLP approach. They model $p(y \mid \mathbf{x}) \propto \exp(\boldsymbol{\lambda}^\top \mathbf{f}(\mathbf{x}, y))$ where $\mathbf{f}$ is a feature vector. This is logistic regression with hand-crafted features — still the MaxEnt distribution given feature expectations.

**Entropy regularization in optimization.** Adding $-\tau H(\mathbf{p})$ as a regularizer to an optimization problem (mirror descent, natural gradient) corresponds to constraining the solution to stay close to the MaxEnt (uniform) distribution. Softmax normalization in attention can be viewed as MaxEnt regularization of the attention weights.

---

## 6. Source Coding and Data Compression

### 6.1 Shannon's Source Coding Theorem

The most fundamental theorem of information theory connects entropy to the minimum number of bits needed to describe a random variable.

**Setup.** We have a source producing i.i.d. symbols $X_1, X_2, \ldots$ from distribution $p$ over alphabet $\mathcal{X}$. A **code** $C: \mathcal{X} \to \{0,1\}^*$ assigns a binary string to each symbol. The **average code length** is $\bar{L} = \sum_x p(x) |C(x)|$ where $|C(x)|$ is the length of the codeword for $x$.

**Theorem (Shannon 1948).** For any uniquely decodable code $C$:

$$\bar{L} \ge H_2(X)$$

where $H_2(X) = -\sum_x p(x)\log_2 p(x)$ is the entropy in bits. Furthermore, there exists a uniquely decodable code achieving:

$$\bar{L} < H_2(X) + 1$$

The lower bound says: no code can do better than entropy. The upper bound says: entropy is achievable within 1 bit per symbol. The gap of 1 bit can be made arbitrarily small by coding blocks of $n$ symbols (achieving $H_2(X) + 1/n$ per symbol).

**Proof of lower bound.** By **Kraft's inequality** (§6.2), any uniquely decodable code satisfies $\sum_x 2^{-l_x} \le 1$ where $l_x = |C(x)|$. Define $q(x) = 2^{-l_x} / Z$ as a probability distribution. Then:

$$\bar{L} - H_2(X) = \sum_x p(x)l_x + \sum_x p(x)\log_2 p(x)$$
$$= \sum_x p(x)(-\log_2 2^{-l_x} + \log_2 p(x)) = \sum_x p(x)\log_2\frac{p(x)}{2^{-l_x}}$$
$$\ge \sum_x p(x)\log_2 p(x) \cdot Z = D_{\mathrm{KL}}(p \| q) + \log_2 Z \ge 0$$

using $\log_2 Z \ge 0$ (since $Z \ge 1$ by Kraft's inequality for decodable codes) and $D_{\mathrm{KL}} \ge 0$.

**Shannon code.** Assign code length $l_x = \lceil -\log_2 p(x) \rceil$. This achieves $l_x < -\log_2 p(x) + 1$, giving $\bar{L} < H_2(X) + 1$.

**For AI:** Every neural network that outputs log-probabilities implicitly defines a code. The cross-entropy loss $-\log p_{\boldsymbol{\theta}}(y)$ is the code length of outcome $y$ under the model's code. Minimizing cross-entropy minimizes the expected code length — i.e., the model is learning to compress the data.

### 6.2 Prefix Codes and Kraft's Inequality

A **prefix code** (or instantaneous code) is a code where no codeword is a prefix of another. Prefix codes are desirable because they can be decoded without reading ahead (instantaneous decoding).

**Theorem (Kraft's Inequality).** A prefix code with codeword lengths $l_1, \ldots, l_n$ over a binary alphabet exists if and only if:

$$\sum_{i=1}^n 2^{-l_i} \le 1$$

The inequality is tight for **complete** prefix codes (every binary string has a unique prefix that is a codeword).

**Intuition.** Each codeword of length $l$ "uses up" $2^{-l}$ of the binary tree's budget. Kraft's inequality says you cannot exceed the available budget. The integer lengths $l_x = \lceil -\log_2 p(x) \rceil$ satisfy Kraft's inequality since $\sum_x 2^{-\lceil -\log_2 p(x)\rceil} \le \sum_x p(x) = 1$.

**Connection to entropy.** The optimal code length for outcome $x$ is $l_x^* = -\log_2 p(x)$ (generally not an integer). Rounding up to $\lceil -\log_2 p(x) \rceil$ gives the Shannon code. The entropy $H_2(X)$ is the minimum average length achievable by any prefix code.

### 6.3 Huffman Coding

**Huffman's algorithm** (1952) constructs the optimal prefix code for a given distribution $p$ — the code that minimizes $\bar{L}$ exactly (over integer code lengths).

**Algorithm:**
1. Create a leaf node for each symbol with weight $p(x)$
2. While more than one node remains: combine the two lowest-weight nodes into a parent node with weight equal to their sum; assign bit `0` to one child, `1` to the other
3. Read off codewords from the resulting binary tree

**Example.** Distribution: $p(A) = 0.5, p(B) = 0.25, p(C) = 0.125, p(D) = 0.125$.

```
HUFFMAN CODE CONSTRUCTION
════════════════════════════════════════════════════════════════════════

  Symbols:  A(0.5)  B(0.25)  C(0.125)  D(0.125)

  Step 1: Combine C and D → CD(0.25)
  Step 2: Combine B and CD → BCD(0.5)
  Step 3: Combine A and BCD → root(1.0)

  Tree:          root(1.0)
                /         \
           A(0.5)        BCD(0.5)
              0          /      1
                     B(0.25)  CD(0.25)
                     0       /        1
                         C(0.125)  D(0.125)
                         0             1

  Codewords: A=0, B=10, C=110, D=111

  Average length: 0.5×1 + 0.25×2 + 0.125×3 + 0.125×3 = 1.75 bits
  Entropy: H = 0.5log2+0.25log4+0.125log8+0.125log8 = 1.75 bits (exact!)

════════════════════════════════════════════════════════════════════════
```

This distribution has $p(x) = 2^{-l_x}$ exactly (a dyadic distribution), so Huffman coding achieves the entropy exactly with no rounding loss.

**Huffman optimality.** Huffman coding achieves the minimum average code length over all prefix codes for the given distribution. The average length satisfies $H_2(X) \le \bar{L} < H_2(X) + 1$.

### 6.4 Range Coding in LLM Inference

Modern LLM inference systems use **arithmetic coding** (and its approximation **range coding**) to losslessly compress the model's output token sequence to the model's predicted entropy per token.

**Key idea.** Arithmetic coding encodes a sequence of symbols by narrowing an interval $[0,1)$ based on the predicted probabilities of each symbol. The final interval can be encoded in $H_2(x_1,\ldots,x_T) + 2$ bits — within 2 bits of the entropy of the sequence.

**Perplexity and compression.** A language model with perplexity $\operatorname{PPL}$ can compress text to approximately $\log_2 \operatorname{PPL}$ bits per token. A model with $\operatorname{PPL} = 10$ compresses to ~3.32 bits/token; with $\operatorname{PPL} = 3$ to ~1.58 bits/token. This is why perplexity is a measure of compression efficiency: lower perplexity = better compressor.

**For AI:** The equivalence between learning (cross-entropy minimization), prediction (perplexity), and compression (code length) is one of the deepest connections in information theory for ML. Chinchilla scaling laws can be derived from the information-theoretic view of language modeling as optimal compression.

---

## 7. Entropy Rate of Stochastic Processes

### 7.1 Entropy Rate: Definition

For a sequence of random variables $X_1, X_2, \ldots$, the **entropy rate** measures the per-symbol entropy in the limit:

**Definition.** The entropy rate of a stochastic process $\{X_t\}$ is:

$$\mathcal{H} = \lim_{n \to \infty} \frac{1}{n} H(X_1, \ldots, X_n)$$

when the limit exists.

**Equivalent formulation.** For a stationary process, the entropy rate also equals:

$$\mathcal{H} = \lim_{n \to \infty} H(X_n \mid X_1, \ldots, X_{n-1})$$

the conditional entropy of the next symbol given an infinitely long past. This limit exists and equals the previous one for stationary processes (by the AEP — Asymptotic Equipartition Property).

**For i.i.d. processes:** $\mathcal{H} = H(X_1)$ — each symbol contributes its full individual entropy.

**For deterministic processes:** $\mathcal{H} = 0$ — once you know the rule, you can predict every future symbol.

**For natural language:** The entropy rate of English is estimated at ~1–1.5 bits per character (Shannon estimated it at 0.6–1.3 bits/character through human prediction experiments). This is much less than $\log_2 26 \approx 4.7$ bits/character (uniform over letters), reflecting the strong statistical regularities in language.

### 7.2 Entropy Rate of Markov Chains

For a **stationary ergodic Markov chain** with state space $\mathcal{X}$, transition matrix $P$ (where $P_{xy} = P(X_{t+1}=y \mid X_t=x)$), and stationary distribution $\boldsymbol{\mu}$ satisfying $\boldsymbol{\mu}^\top P = \boldsymbol{\mu}^\top$:

$$\boxed{\mathcal{H} = -\sum_{x \in \mathcal{X}} \mu_x \sum_{y \in \mathcal{X}} P_{xy} \log P_{xy} = \mathbb{E}_{\mu}\left[H(X_{t+1} \mid X_t = \cdot)\right]}$$

**Derivation.** For a stationary Markov chain, $H(X_n \mid X_1,\ldots,X_{n-1}) = H(X_n \mid X_{n-1})$ (Markov property). In stationarity, this equals:

$$H(X_2 \mid X_1) = -\sum_{x,y} p(x,y)\log p(y \mid x) = -\sum_x \mu_x \sum_y P_{xy}\log P_{xy}$$

**Example — Bigram language model.** A bigram language model is a first-order Markov chain over the vocabulary $\mathcal{V}$. Its entropy rate is:

$$\mathcal{H}_{\text{bigram}} = -\sum_{w \in \mathcal{V}} \mu_w \sum_{w' \in \mathcal{V}} P_{ww'}\log P_{ww'}$$

where $\mu_w$ is the stationary token frequency and $P_{ww'}$ is the bigram probability.

**Worked example.** Two-state Markov chain: states $\{0,1\}$, transition matrix $P = \begin{pmatrix} 1-\alpha & \alpha \\ \beta & 1-\beta \end{pmatrix}$.

Stationary distribution: $\mu_0 = \beta/(\alpha+\beta)$, $\mu_1 = \alpha/(\alpha+\beta)$.

$$\mathcal{H} = \frac{\beta \cdot h(\alpha) + \alpha \cdot h(\beta)}{\alpha+\beta}$$

where $h(p) = -p\log p - (1-p)\log(1-p)$ is the binary entropy function.

When $\alpha = \beta = 1/2$ (memoryless): $\mathcal{H} = h(1/2) = 1$ bit (matches i.i.d.).  
When $\alpha = \beta = 0.01$ (sticky): $\mathcal{H} \approx h(0.01) \approx 0.08$ bits — very predictable.

### 7.3 Perplexity as Entropy Rate Estimate

**Definition (Perplexity).** Given a language model $p_{\boldsymbol{\theta}}$ and a test sequence $x_1, \ldots, x_T$:

$$\operatorname{PPL}(x_1,\ldots,x_T) = \exp\!\left(-\frac{1}{T}\sum_{t=1}^T \log p_{\boldsymbol{\theta}}(x_t \mid x_{<t})\right)$$

(Using natural log; for base-2 bits, replace $\exp$ with $2^{(\cdot)}$.)

**Connection to entropy rate.** By the ergodic theorem, as $T \to \infty$:

$$-\frac{1}{T}\sum_{t=1}^T \log p_{\boldsymbol{\theta}}(x_t \mid x_{<t}) \xrightarrow{a.s.} \mathcal{H}(p_{\mathrm{true}}, p_{\boldsymbol{\theta}}) = \text{cross-entropy rate}$$

If $p_{\boldsymbol{\theta}} = p_{\mathrm{true}}$ (perfect model), this equals the true entropy rate $\mathcal{H}$, so $\operatorname{PPL} = e^{\mathcal{H}}$ — a perfect model's perplexity equals the exponentiated entropy rate of the language.

**Interpretation table:**

| Perplexity | Bits/token | Interpretation |
| --- | --- | --- |
| $e^1 \approx 2.7$ | 1 bit | Near-perfect prediction; approximately 2 equally likely choices per token |
| $e^2 \approx 7.4$ | 2.9 bits | Good model; ~7 equally likely choices |
| $e^3 \approx 20$ | 4.3 bits | Moderate model |
| $e^{\ln 32000} = 32000$ | 15 bits | Uniform distribution over 32k-token vocabulary; random model |

**For AI — Benchmark context (2026):** State-of-the-art LLMs on standard benchmarks:
- GPT-4, Claude 3.5+: WikiText-103 PPL $\sim 3$–$5$ (nats base)
- GPT-2 (1.5B): WikiText-103 PPL $\approx 18$ (bits base: $\log_2 18 \approx 4.2$ bits/token)
- A random model over 50k vocabulary: PPL = 50,000

**Cross-entropy rate vs true entropy rate.** The perplexity of a model $p_{\boldsymbol{\theta}}$ is always at least $e^{\mathcal{H}}$ (the perplexity of a perfect model), because $H(p_{\mathrm{true}}, p_{\boldsymbol{\theta}}) = \mathcal{H} + D_{\mathrm{KL}}(p_{\mathrm{true}} \| p_{\boldsymbol{\theta}}) \ge \mathcal{H}$. Improving a language model reduces the KL gap to the true distribution, pushing perplexity toward the fundamental entropy rate limit.

---

## 8. Differential Entropy

### 8.1 Definition and Subtleties

For continuous random variables, the analogue of Shannon entropy is **differential entropy**.

**Definition.** Let $X$ be a continuous random variable with probability density function $p(x)$. The **differential entropy** is:

$$h(X) = -\int_{-\infty}^\infty p(x) \log p(x)\, dx = \mathbb{E}[-\log p(X)]$$

**Key differences from discrete entropy:**

1. **Can be negative.** Unlike $H(X) \ge 0$, differential entropy satisfies no such lower bound. For example, $\mathcal{U}(0, 1/2)$ has $h = \log(1/2) = -\log 2 < 0$. A very concentrated distribution has $h \to -\infty$.

2. **Not invariant under bijections.** Under a bijection $Y = f(X)$:
   $$h(Y) = h(X) + \mathbb{E}[\log |f'(X)|]$$
   The Jacobian term appears. This means differential entropy depends on the scale of the variable.

3. **Not a probability.** $p(x)$ can exceed 1 for densities, so $-\log p(x)$ can be negative for individual points.

4. **Limit of discrete entropy.** For a discrete approximation of $X$ with bin width $\Delta$: $H(X_\Delta) \approx h(X) - \log \Delta$. As $\Delta \to 0$, the discrete entropy diverges (infinite precision requires infinite bits), but the **differential entropy $h(X)$** is the finite part of this divergence.

**Despite these subtleties**, differential entropy is meaningful for comparing distributions of the same type and for measuring the "entropy-like" uncertainty of continuous variables.

### 8.2 Differential Entropy of Standard Distributions

**Uniform distribution** $X \sim \mathcal{U}(a, b)$, $p(x) = \frac{1}{b-a}$:

$$h(X) = -\int_a^b \frac{1}{b-a}\log\frac{1}{b-a}\,dx = \log(b-a)$$

Note: $h < 0$ when $b - a < 1$; $h > 0$ when $b - a > 1$.

**Gaussian distribution** $X \sim \mathcal{N}(\mu, \sigma^2)$:

$$h(X) = \frac{1}{2}\ln(2\pi e \sigma^2)$$

**Derivation:**
$$h(X) = -\mathbb{E}\!\left[\log\frac{1}{\sqrt{2\pi\sigma^2}}e^{-(X-\mu)^2/2\sigma^2}\right] = \frac{1}{2}\log(2\pi\sigma^2) + \frac{\mathbb{E}[(X-\mu)^2]}{2\sigma^2} = \frac{1}{2}\log(2\pi\sigma^2) + \frac{1}{2} = \frac{1}{2}\log(2\pi e\sigma^2)$$

**Multivariate Gaussian** $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$:

$$h(\mathbf{X}) = \frac{1}{2}\log\det(2\pi e \Sigma) = \frac{n}{2}\log(2\pi e) + \frac{1}{2}\log\det(\Sigma)$$

The differential entropy of a multivariate Gaussian depends on the covariance matrix only through its determinant $\det(\Sigma) = \prod_i \lambda_i$ (product of eigenvalues). A degenerate covariance ($\det = 0$) gives $h = -\infty$.

**Laplace distribution** $X \sim \operatorname{Laplace}(\mu, b)$, $p(x) = \frac{1}{2b}e^{-|x-\mu|/b}$:

$$h(X) = 1 + \log(2b)$$

**Exponential distribution** $X \sim \operatorname{Exp}(\lambda)$, $p(x) = \lambda e^{-\lambda x}$:

$$h(X) = 1 - \log\lambda$$

**For AI — VAE latent space.** The VAE (Variational Autoencoder) encodes inputs into a distribution $q_\phi(\mathbf{z} \mid \mathbf{x}) \approx \mathcal{N}(\boldsymbol{\mu}_\phi, \Sigma_\phi)$. The ELBO involves the differential entropy of this distribution:

$$\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x} \mid \mathbf{z})] - D_{\mathrm{KL}}(q_\phi(\mathbf{z} \mid \mathbf{x}) \| p(\mathbf{z}))$$

The KL term equals $h(\mathcal{N}(\mathbf{0}, I)) - h(q_\phi(\mathbf{z} \mid \mathbf{x})) + \text{const}$, so optimizing the ELBO maximizes the differential entropy of the encoder distribution subject to staying close to the prior.

### 8.3 Maximum Differential Entropy: The Gaussian

**Theorem.** Among all continuous distributions with fixed variance $\sigma^2$, the Gaussian $\mathcal{N}(\mu, \sigma^2)$ maximizes the differential entropy:

$$h(X) \le \frac{1}{2}\log(2\pi e \sigma^2)$$

with equality iff $X \sim \mathcal{N}(\mu, \sigma^2)$.

**Proof.** Let $p$ be any distribution with variance $\sigma^2$ and let $g = \mathcal{N}(\mu, \sigma^2)$ be the Gaussian. Then:

$$0 \le D_{\mathrm{KL}}(p \| g) = \int p(x)\log\frac{p(x)}{g(x)}\,dx = -h(p) + \mathbb{E}_p[-\log g(X)]$$

Now $-\log g(x) = \frac{1}{2}\log(2\pi\sigma^2) + \frac{(x-\mu)^2}{2\sigma^2}$, so:

$$\mathbb{E}_p[-\log g(X)] = \frac{1}{2}\log(2\pi\sigma^2) + \frac{\mathbb{E}_p[(X-\mu)^2]}{2\sigma^2} = \frac{1}{2}\log(2\pi\sigma^2) + \frac{1}{2} = \frac{1}{2}\log(2\pi e\sigma^2)$$

Therefore $0 \le D_{\mathrm{KL}}(p \| g) = -h(p) + \frac{1}{2}\log(2\pi e\sigma^2)$, giving $h(p) \le \frac{1}{2}\log(2\pi e\sigma^2) = h(g)$.

**Implication.** The Gaussian is the MaxEnt distribution for continuous variables with fixed variance. This justifies the Gaussian prior in Bayesian models: it encodes only the constraint on variance and makes no additional assumptions. It also explains why many natural phenomena follow Gaussian distributions — they are consistent with a variance constraint and maximal uncertainty otherwise.

---

## 9. Rényi Entropy and Generalizations

### 9.1 Rényi Entropy

Shannon entropy is one member of a parametric family introduced by Alfréd Rényi in 1961.

**Definition (Rényi Entropy).** For $\alpha > 0$, $\alpha \ne 1$, the **Rényi entropy of order $\alpha$** is:

$$H_\alpha(X) = \frac{1}{1-\alpha}\log\!\left(\sum_{x \in \mathcal{X}} p(x)^\alpha\right)$$

**Limiting cases:**

| Order $\alpha$ | Name | Formula |
| --- | --- | --- |
| $\alpha \to 0$ | Hartley entropy (max-entropy) | $H_0 = \log \lvert\operatorname{supp}(p)\rvert$ |
| $\alpha \to 1$ | Shannon entropy | $H_1 = H(X) = -\sum_x p(x)\log p(x)$ |
| $\alpha = 2$ | Collision entropy | $H_2 = -\log \sum_x p(x)^2 = -\log P(\text{collision})$ |
| $\alpha \to \infty$ | Min-entropy | $H_\infty = -\log \max_x p(x)$ |

**Ordering property.** Rényi entropy is a decreasing function of $\alpha$:

$$H_0(X) \ge H_1(X) \ge H_2(X) \ge \cdots \ge H_\infty(X)$$

All orders coincide for the uniform distribution.

**Shannon entropy as the $\alpha \to 1$ limit.** Using L'Hôpital's rule:

$$\lim_{\alpha \to 1} H_\alpha(X) = \lim_{\alpha \to 1} \frac{\frac{d}{d\alpha}\log\sum_x p^\alpha}{\frac{d}{d\alpha}(1-\alpha)} = -\frac{\sum_x p(x)\log p(x)}{\sum_x p(x)} = -\sum_x p(x)\log p(x) = H(X)$$

**Additivity.** Rényi entropy satisfies $H_\alpha(X,Y) = H_\alpha(X) + H_\alpha(Y)$ when $X \perp\!\!\!\perp Y$, for all $\alpha$. Shannon entropy satisfies a stronger additivity: the chain rule.

### 9.2 Min-Entropy and Security

**Definition.** The **min-entropy** is:

$$H_\infty(X) = -\log \max_{x \in \mathcal{X}} p(x)$$

Min-entropy is the worst-case uncertainty — it measures how concentrated the distribution is at its most likely outcome.

**Security interpretation.** $2^{-H_\infty(X)}$ is the **guessing advantage** — the probability that an adversary can correctly guess $X$ in one try by always guessing the most likely outcome. A cryptographic key $X$ with high min-entropy is hard to guess; one with low min-entropy (concentrated on a few values) is insecure.

**Connection to ML.** In adversarial robustness, min-entropy of the model's output distribution bounds the probability that an adversary can predict the top-1 output without information about the input. Models with collapsed output distributions (mode seeking) have low min-entropy and are more vulnerable to certain attacks.

**Relationship to other quantities:**
- $H_\infty(X) \le H(X) \le H_0(X) = \log |\operatorname{supp}(p)|$
- For uniform distribution: all Rényi entropies equal $\log n$

### 9.3 Tsallis Entropy

The **Tsallis entropy** of order $q$ (Tsallis 1988) is:

$$S_q(X) = \frac{1 - \sum_x p(x)^q}{q-1}$$

with $S_1(X) = H(X)$ (Shannon entropy) as the $q \to 1$ limit.

**Differences from Rényi:**
- Tsallis entropy is **not additive** for independent variables: $S_q(X,Y) = S_q(X) + S_q(Y) + (1-q)S_q(X)S_q(Y)$
- Maximized by the $q$-exponential distribution $p(x) \propto [1-(1-q)E(x)]^{1/(1-q)}$ under energy constraints

**$q$-exponential family and ML.** Tsallis entropy's maximizer is the $q$-Gaussian, a heavy-tailed generalization. Setting $q < 1$ gives distributions with bounded support; $q > 1$ gives power-law tails. The $q$-softmax:

$$p_q(x) \propto [1 + (1-q)(z_x - z_{\max})]^{1/(1-q)}$$

is used in some modern attention variants and sparse softmax approaches as an alternative to the standard exponential softmax, giving exactly-zero probabilities to low-scoring tokens rather than near-zero.

---

## 10. Applications in Machine Learning

### 10.1 Decision Tree Splits

Decision trees recursively split the training data to reduce prediction uncertainty. **Information gain** is the entropy-based splitting criterion:

$$\text{IG}(Y; X_j) = H(Y) - H(Y \mid X_j) = I(Y; X_j)$$

This is the reduction in uncertainty about the target $Y$ achieved by knowing feature $X_j$. The ID3 and C4.5 algorithms choose the split with maximum information gain at each node.

**Binary information gain.** For a binary split $X_j \le \theta$ vs $X_j > \theta$ producing left subset of size $n_L$ and right subset of size $n_R$:

$$H(Y \mid X_j) = \frac{n_L}{n}H(Y_L) + \frac{n_R}{n}H(Y_R)$$

$$\text{IG} = H(Y) - \frac{n_L}{n}H(Y_L) - \frac{n_R}{n}H(Y_R)$$

where $H(Y_L)$, $H(Y_R)$ are the class entropies in the left and right partitions.

**Gini vs entropy.** The Gini impurity $G = 1 - \sum_c p_c^2$ is often used as an approximation to entropy (the first-order Taylor approximation of $-p\log p$). For binary classification, $G \approx 2p(1-p)$ while $H = h(p)$. Both yield similar splits in practice.

**Gradient boosted trees.** XGBoost and LightGBM use second-order Taylor expansions of the loss function to compute split gains, but the connection to entropy remains: the gain measures uncertainty reduction in the residual distribution.

**Random forests.** The diversity of trees in a random forest is controlled by the randomized feature selection, which limits the information gain at each split. Higher randomization → higher entropy of the ensemble → better generalization.

### 10.2 Entropy Regularization in RL

Standard RL seeks a policy $\pi$ that maximizes expected cumulative reward. **Maximum entropy RL** (Ziebart et al., 2008; Haarnoja et al., 2018) adds an entropy bonus:

$$J_{\text{MaxEnt}}(\pi) = \mathbb{E}\!\left[\sum_{t=0}^\infty \gamma^t \left(r(s_t, a_t) + \alpha H(\pi(\cdot \mid s_t))\right)\right]$$

where $\alpha > 0$ is the temperature parameter controlling the trade-off between reward and entropy.

**Why entropy regularization helps:**
1. **Exploration:** High-entropy policies try diverse actions, avoiding premature convergence to suboptimal deterministic policies.
2. **Robustness:** Maintaining entropy provides a hedge against environment uncertainty and distributional shift.
3. **Multiple modes:** Standard RL tends to find a single optimal policy; MaxEnt RL discovers the full manifold of near-optimal behaviors.

**Soft Actor-Critic (SAC, Haarnoja et al. 2018).** The dominant maximum entropy RL algorithm for continuous control. Key equations:

$$\pi^* = \arg\max_\pi J_{\text{MaxEnt}}(\pi) \iff \pi^*(a \mid s) \propto \exp(Q^{\pi^*}(s,a)/\alpha)$$

The optimal policy is the Boltzmann/softmax policy over Q-values — exactly the MaxEnt distribution over actions under fixed mean Q-value constraint. SAC is used in robotics, LLM RLHF exploration stages, and game AI.

**Entropy in RLHF.** In RLHF (Reinforcement Learning from Human Feedback) for LLMs, a KL penalty term enforces that the fine-tuned policy $\pi_{\boldsymbol{\theta}}$ stays close to the reference policy $\pi_{\text{ref}}$:

$$J(\boldsymbol{\theta}) = \mathbb{E}_\pi[r(x,y)] - \beta D_{\mathrm{KL}}(\pi_{\boldsymbol{\theta}} \| \pi_{\text{ref}})$$

The KL penalty is equivalent to an entropy regularizer relative to $\pi_{\text{ref}}$. Minimizing this keeps the output distribution high-entropy (close to the reference), preventing the model from collapsing to reward-hacking degenerate outputs.

### 10.3 Confidence Calibration

A model is **calibrated** if its predicted probability for an event equals the empirical frequency of that event. The entropy of the model's output distribution is a key calibration signal:

**Predictive entropy** $H(p_{\boldsymbol{\theta}}(\cdot \mid \mathbf{x}))$ decomposes into two components:
1. **Aleatoric uncertainty** — inherent randomness in the data (irreducible)
2. **Epistemic uncertainty** — uncertainty from lack of data or model misspecification (reducible)

**Temperature scaling** (Guo et al., 2017): The most effective post-hoc calibration method. Fit a single scalar $T > 0$ such that $p_T(y \mid \mathbf{x}) = \operatorname{softmax}(\mathbf{z}/T)$ is calibrated. $T > 1$ increases entropy (reduces overconfidence); $T < 1$ decreases entropy (sharpens predictions).

**Entropy-based uncertainty detection:**
- Out-of-distribution inputs tend to produce high-entropy output distributions (model is uncertain)
- In-distribution inputs the model knows well produce low-entropy outputs
- Threshold $H(p_{\boldsymbol{\theta}}(\cdot \mid \mathbf{x})) > \tau$ for anomaly detection

**For LLM generation:** Entropy of the output distribution at each step can signal when the model is about to hallucinate (high entropy) vs. when it is confident (low entropy). Token-level entropy is used in speculative decoding, early exit, and uncertainty-guided sampling.

### 10.4 Preview: Cross-Entropy, KL, Mutual Information, Fisher Information

This chapter (§09) develops four closely related concepts that each deserve a full section:

> **Cross-Entropy** $H(p,q) = -\sum_x p(x)\log q(x)$
>
> Cross-entropy is the expected code length under code $q$ when the true distribution is $p$. It decomposes as $H(p,q) = H(p) + D_{\mathrm{KL}}(p \| q)$: the sum of the true entropy and the KL divergence. The cross-entropy loss in ML trains models to minimize the KL divergence between the data distribution and the model. Minimizing cross-entropy $\equiv$ minimizing KL divergence $\equiv$ maximum likelihood estimation.
>
> → _Full treatment: [§04 Cross-Entropy](../04-Cross-Entropy/notes.md)_

> **KL Divergence** $D_{\mathrm{KL}}(p \| q) = \sum_x p(x)\log\frac{p(x)}{q(x)}$
>
> KL divergence measures how much $q$ fails to match $p$. It is non-negative ($D_{\mathrm{KL}} \ge 0$, with equality iff $p = q$), asymmetric ($D_{\mathrm{KL}}(p\|q) \ne D_{\mathrm{KL}}(q\|p)$ in general), and appears everywhere: VAE regularizer, knowledge distillation, RLHF KL penalty, variational inference.
>
> → _Full treatment: [§02 KL Divergence](../02-KL-Divergence/notes.md)_

> **Mutual Information** $I(X;Y) = H(X) - H(X \mid Y) = H(X) + H(Y) - H(X,Y)$
>
> Mutual information measures the statistical dependence between $X$ and $Y$ — how much knowing $Y$ reduces uncertainty about $X$. It is symmetric, non-negative, and zero iff $X \perp\!\!\!\perp Y$. Key applications: InfoNCE contrastive loss (SimCLR, CLIP), information bottleneck in DNNs, feature selection.
>
> → _Full treatment: [§03 Mutual Information](../03-Mutual-Information/notes.md)_

> **Fisher Information** $\mathcal{F}(\theta) = \mathbb{E}\!\left[\left(\frac{\partial}{\partial\theta}\log p(X;\theta)\right)^2\right]$
>
> Fisher information measures how much a random variable $X$ tells you about the parameter $\theta$ generating it. The Cramér-Rao bound says no unbiased estimator can have variance less than $1/\mathcal{F}(\theta)$. The natural gradient in optimization uses the Fisher information matrix as a Riemannian metric, leading to K-FAC and Shampoo.
>
> → _Full treatment: [§05 Fisher Information](../05-Fisher-Information/notes.md)_

---

## 11. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
| --- | --- | --- | --- |
| 1 | Using $\log$ base 2 vs $\ln$ inconsistently in the same derivation | Mixing bases introduces a factor of $\ln 2 \approx 0.693$ that invalidates gradient computations | Always use natural log ($\ln$) for analysis and derivatives; convert to bits only for reporting |
| 2 | Claiming $H(X) < 0$ for a discrete distribution | Discrete entropy is always $\ge 0$ | Check: $0 \le p(x) \le 1 \Rightarrow -\log p(x) \ge 0 \Rightarrow H(X) \ge 0$; negative entropy only arises for differential entropy |
| 3 | Omitting the convention $0\log 0 = 0$ | Outcomes with $p(x) = 0$ would contribute $0 \times (-\infty)$, which is undefined | Always apply the convention; outcomes with zero probability do not contribute to entropy |
| 4 | Confusing $H(X \mid Y)$ with $H(X \mid Y=y)$ | $H(X \mid Y)$ is the expectation of $H(X \mid Y=y)$ over $y \sim p(y)$; they differ unless $H(X \mid Y=y)$ is constant in $y$ | Write $H(X \mid Y) = \mathbb{E}_Y[H(X \mid Y=y)] = \sum_y p(y)H(X \mid Y=y)$ |
| 5 | Confusing cross-entropy with entropy | $H(p,q) \ne H(p)$ in general; $H(p,q) = H(p)$ only when $p = q$ | Remember $H(p,q) = H(p) + D_{\mathrm{KL}}(p\|q) \ge H(p)$ |
| 6 | Assuming "higher entropy = better model" | High-entropy output distributions mean uncertain predictions; a perfect model should have low entropy at high-frequency tokens | Perplexity = exponentiated per-token entropy; the model's entropy should match the data distribution's entropy |
| 7 | Thinking differential entropy is always positive | $h(X) = \log(b-a)$ for $\mathcal{U}(a,b)$ is negative when $b - a < 1$ | Differential entropy has no lower bound; it can be $-\infty$ for degenerate distributions |
| 8 | Interpreting differential entropy as a probability | $h(X)$ is not a probability; $p(x)$ can exceed 1 for densities | Differential entropy is a real number measuring uncertainty, not a probability |
| 9 | Confusing entropy with diversity | A distribution can have high entropy but concentrate near a few values if there are many near-uniform values | Entropy measures uncertainty, not diversity of interesting outputs; check the full distribution shape |
| 10 | Applying Shannon's source coding theorem to non-i.i.d. sources | The theorem applies to i.i.d. sequences; for correlated sources, the entropy rate $\mathcal{H}$ (not $H(X_1)$) is the relevant bound | Use entropy rate $\mathcal{H} = \lim \frac{1}{n}H(X_1,\ldots,X_n)$ for dependent sources |
| 11 | Confusing $H(X,Y)$ with $H(X) \cdot H(Y)$ | Joint entropy is not the product; entropy is additive (not multiplicative) for independent variables | For independent $X, Y$: $H(X,Y) = H(X) + H(Y)$ (sum, not product) |
| 12 | Using Rényi entropy for all purposes | Different Rényi orders emphasize different parts of the distribution; Shannon entropy ($\alpha=1$) is the unique one satisfying the chain rule | Choose $\alpha$ based on the application: $\alpha=\infty$ for security, $\alpha=1$ for compression/learning, $\alpha=2$ for collision probability |

---

## 12. Exercises

**Exercise 1 ★ — Binary Entropy Function and Distributions**

(a) Compute $H(X)$ in bits for: (i) a fair coin ($p=0.5$), (ii) a biased coin ($p=0.1$), (iii) a fair 6-sided die, (iv) the distribution $(0.5, 0.25, 0.125, 0.125)$.

(b) Plot the binary entropy function $h(p) = -p\log_2 p - (1-p)\log_2(1-p)$ for $p \in [0,1]$. Verify $h(0) = h(1) = 0$ and $h(1/2) = 1$ bit. Identify the unique maximum.

(c) A language model outputs a distribution over a vocabulary of $n = 32{,}000$ tokens. What is the maximum possible entropy in nats? What is the entropy if the distribution is uniform? What if the top-1 token has probability 0.95 and all others are uniform?

**Exercise 2 ★ — Maximum Entropy via Lagrange Multipliers**

(a) Show that the uniform distribution $p_i = 1/n$ maximizes $H(\mathbf{p}) = -\sum_i p_i \log p_i$ subject to $\sum_i p_i = 1$, $p_i \ge 0$.

(b) Find the maximum entropy distribution over $\mathbb{R}_{\ge 0}$ subject to $\mathbb{E}[X] = \mu > 0$. Show this is the exponential distribution $\operatorname{Exp}(1/\mu)$.

(c) Find the maximum entropy distribution over $\mathbb{R}$ subject to $\mathbb{E}[X] = \mu$ and $\operatorname{Var}(X) = \sigma^2$. Show this is $\mathcal{N}(\mu, \sigma^2)$.

**Exercise 3 ★ — Joint and Conditional Entropy**

(a) Given joint distribution $p(X,Y)$ with table: $p(0,0)=0.3$, $p(0,1)=0.1$, $p(1,0)=0.2$, $p(1,1)=0.4$. Compute $H(X)$, $H(Y)$, $H(X,Y)$, $H(X \mid Y)$, $H(Y \mid X)$.

(b) Verify the chain rule: $H(X,Y) = H(X) + H(Y \mid X) = H(Y) + H(X \mid Y)$.

(c) Compute the mutual information $I(X;Y) = H(X) + H(Y) - H(X,Y)$. Verify $I(X;Y) \ge 0$.

(d) Modify the joint distribution so that $X \perp\!\!\!\perp Y$. Verify that $I(X;Y) = 0$ and $H(X,Y) = H(X) + H(Y)$.

**Exercise 4 ★★ — Entropy Rate of a Markov Chain**

Consider a 3-state Markov chain with states $\{A, B, C\}$ and transition matrix:
$$P = \begin{pmatrix} 0.7 & 0.2 & 0.1 \\ 0.3 & 0.4 & 0.3 \\ 0.1 & 0.3 & 0.6 \end{pmatrix}$$

(a) Find the stationary distribution $\boldsymbol{\mu}$ satisfying $\boldsymbol{\mu}^\top P = \boldsymbol{\mu}^\top$ and $\sum_i \mu_i = 1$.

(b) Compute the entropy rate $\mathcal{H} = -\sum_{x} \mu_x \sum_{y} P_{xy}\log P_{xy}$ in nats.

(c) Compare to the entropy rate if all states were independent: $H(X_1) = H(\boldsymbol{\mu})$. Is the Markov chain entropy rate higher or lower than the i.i.d. rate? Explain intuitively.

(d) Compute the perplexity $\operatorname{PPL} = e^{\mathcal{H}}$. If this Markov chain were a language model, what would this perplexity tell you?

**Exercise 5 ★★ — Shannon Source Coding and Huffman Code**

(a) Given source distribution $p(A) = 0.4, p(B) = 0.3, p(C) = 0.2, p(D) = 0.1$:
  - Compute $H(X)$ in bits.
  - Construct the optimal Huffman code. Draw the code tree.
  - Compute the average code length $\bar{L}$. Verify $H(X) \le \bar{L} < H(X)+1$.

(b) Write a function `huffman_encode(probs)` that returns a dict mapping symbols to codewords. Verify on the distribution above.

(c) For a distribution where $p(x_1) = 1/2^k$ for all $k = 1,\ldots,n-1$ and $p(x_n) = 1/2^{n-1}$ (geometric): show that the Huffman code achieves $\bar{L} = H(X)$ exactly (a dyadic distribution).

**Exercise 6 ★★ — Differential Entropy and Gaussian MaxEnt**

(a) Compute the differential entropy $h(X)$ for: (i) $\mathcal{U}(0,1)$, (ii) $\mathcal{U}(0,2)$, (iii) $\mathcal{N}(0,1)$, (iv) $\mathcal{N}(0,4)$, (v) $\operatorname{Exp}(1)$.

(b) Show that $h(aX) = h(X) + \log |a|$ for any scalar $a \ne 0$. What happens to entropy when you scale a Gaussian by $a = 2$?

(c) Prove directly (without citing the MaxEnt theorem) that $h(\mathcal{N}(0,\sigma^2)) \ge h(X)$ for any $X$ with $\operatorname{Var}(X) = \sigma^2$ using the non-negativity of $D_{\mathrm{KL}}(p \| g)$ where $g = \mathcal{N}(0,\sigma^2)$.

**Exercise 7 ★★★ — Rényi Entropy Family**

(a) Implement a function `renyi_entropy(p, alpha)` that computes $H_\alpha(X)$ for $\alpha \ne 1$ and falls back to Shannon entropy for $\alpha = 1$ (handle the limit numerically).

(b) For the categorical distribution $p = (0.5, 0.3, 0.15, 0.05)$, compute $H_\alpha$ for $\alpha \in \{0, 0.5, 1, 2, 5, \infty\}$. Plot $H_\alpha$ as a function of $\alpha$ and verify the ordering $H_0 \ge H_1 \ge H_2 \ge H_\infty$.

(c) Compute the min-entropy $H_\infty = -\log \max_x p(x)$ directly. Verify it matches your `renyi_entropy(p, alpha)` as $\alpha \to \infty$.

(d) What is the guessing advantage for this distribution? (The probability that an adversary guessing optimally in one shot succeeds.)

**Exercise 8 ★★★ — MaxEnt RL: Entropy Bonus in Policy Optimization**

(a) Implement a tabular policy gradient for a simple 5-action bandit problem with reward vector $\mathbf{r} = (0.8, 0.6, 0.4, 0.2, 0.0)$. Train for 500 steps with learning rate $\eta = 0.1$.

(b) Add an entropy bonus: objective = $\mathbb{E}_\pi[r] + \alpha H(\pi)$ for $\alpha \in \{0, 0.1, 0.5, 1.0\}$. Train each for 500 steps.

(c) Plot the policy entropy over training for each $\alpha$. Show that higher $\alpha$ maintains higher entropy and finds the softmax-optimal policy $\pi^*(a) \propto e^{r(a)/\alpha}$ rather than the greedy argmax.

(d) Verify that as $\alpha \to 0$, the MaxEnt RL solution approaches the greedy policy; as $\alpha \to \infty$, it approaches the uniform policy.

---

## 13. Why This Matters for AI (2026 Perspective)

| Concept | AI Impact |
| --- | --- |
| Shannon entropy | Foundation of cross-entropy loss; used in training every LLM, classifier, and generative model |
| Perplexity = $e^{\mathcal{H}}$ | Universal language model benchmark; lower perplexity = better compression = better model |
| Maximum entropy principle | Justifies softmax temperature, Gaussian priors, exponential family likelihoods in all Bayesian ML |
| Entropy rate of Markov chains | Connects LLM training loss to the fundamental information-theoretic limit of language compression |
| Source coding (Huffman, arithmetic) | LLM inference compression; range coding for GPU-efficient inference; Chinchilla scaling via compression |
| Conditional entropy $H(Y \mid X)$ | Minimum irreducible error in any supervised learning problem; Bayes error rate bound |
| Entropy regularization (MaxEnt RL) | SAC for robotics, game AI, RLHF exploration; prevents reward hacking; used in Gemini/Grok fine-tuning |
| Temperature scaling | Post-hoc calibration for classification; diverse vs. greedy decoding in LLMs; top-$p$ sampling |
| Rényi / min-entropy | Security of cryptographic tokens; private inference; differential privacy (Rényi DP) |
| Tsallis / sparse softmax | $q$-softmax gives exact sparsity, used in sparse transformer attention variants |
| Differential entropy of Gaussian | VAE ELBO; diffusion model score matching; optimal quantization in quantization-aware training |
| Binary entropy function $h(p)$ | VC dimension and generalization bounds; PAC-Bayes theory; Fano's inequality in lower bounds |
| Chain rule for entropy | Autoregressive factorization: every GPT/LLaMA/Claude uses this identity to factorize $p(x_1,\ldots,x_T)$ |
| Decision tree information gain | Gradient-boosted trees (XGBoost, LightGBM): the core split criterion for tabular ML |

---

## 14. Conceptual Bridge

**Looking backward.** Entropy builds directly on probability theory (Chapter 6) and statistics (Chapter 7). The fundamental input is a probability distribution $p$ over an alphabet $\mathcal{X}$ — the core object of Chapter 6. The convention $H(X) = \mathbb{E}[-\log p(X)]$ uses the expectation operator. The maximum entropy principle invokes Lagrange multipliers from optimization (Chapter 8). The source coding theorem connects to combinatorics and counting (Chapter 1). Everything in this section is built from probability theory and the logarithm.

**Looking forward within Chapter 9.** Entropy is the foundation from which the other §09 concepts branch:
- **KL divergence** (§02) = $H(p,q) - H(p)$ — the extra cost of using code $q$ when truth is $p$; also proved non-negative from $H(X) \le \log |\mathcal{X}|$
- **Mutual information** (§03) = $H(X) - H(X \mid Y)$ — the reduction in $X$'s entropy from observing $Y$
- **Cross-entropy** (§04) = $H(p,q)$ — the training loss for classification and language modeling
- **Fisher information** (§05) = the local curvature of the KL divergence; a second-order entropy quantity

**Looking forward to advanced topics.** The entropy rate connects to ergodic theory and the Asymptotic Equipartition Property (AEP), which underpins all of lossless data compression. Rényi entropy connects to hypothesis testing and Rényi differential privacy. Differential entropy connects to rate-distortion theory (how much distortion is unavoidable at a given bit rate) and Gaussian channels (capacity $C = \frac{1}{2}\log(1 + \text{SNR})$). These are all covered in advanced information theory courses (Cover & Thomas, 2006).

```
CURRICULUM POSITION
════════════════════════════════════════════════════════════════════════

  Chapter 6: Probability Theory
  └── Distributions, expectations, independence
           │
           ▼
  Chapter 7: Statistics
  └── MLE, MAP, sufficient statistics, exponential family
           │
           ▼
  Chapter 8: Optimization
  └── Lagrange multipliers for MaxEnt derivations
           │
           ▼
  ┌─────────────────────────────────────────────────┐
  │  Chapter 9: Information Theory                  │
  │                                                 │
  │  § 01  ENTROPY  ◄── You are here                │
  │         │                                       │
  │         ├──► §02 KL Divergence                  │
  │         │     (H(p,q) = H(p) + D_KL)            │
  │         │                                       │
  │         ├──► §03 Mutual Information              │
  │         │     (I(X;Y) = H(X) - H(X|Y))          │
  │         │                                       │
  │         ├──► §04 Cross-Entropy                  │
  │         │     (training loss for all LLMs)       │
  │         │                                       │
  │         └──► §05 Fisher Information             │
  │               (geometry of the KL ball)         │
  └─────────────────────────────────────────────────┘
           │
           ▼
  Chapter 10: Numerical Methods
  └── Computing entropy stably (log-sum-exp trick),
      perplexity computation, information geometry

════════════════════════════════════════════════════════════════════════
```

**The central unifying theme.** Every quantity in information theory can be expressed in terms of entropy. KL divergence, mutual information, cross-entropy, Fisher information, channel capacity — all reduce to sums of $-p \log p$ terms. Mastering entropy means mastering the mathematical language that unifies probabilistic inference, data compression, optimal coding, and modern machine learning.

---

[← Back to Information Theory](../README.md) | [Next: KL Divergence →](../02-KL-Divergence/notes.md)

---

## Appendix A: Axiomatic Characterization of Entropy

Shannon (1948) derived entropy from a small set of natural axioms. Understanding these axioms illuminates why $H = -\sum p\log p$ is the *unique* reasonable measure of uncertainty.

**The Khinchin Axioms (1953 formulation):**

Let $H_n(p_1, \ldots, p_n)$ be a measure of uncertainty for a distribution $(p_1,\ldots,p_n)$. We require:

1. **Continuity.** $H_n(p_1,\ldots,p_n)$ is continuous in all arguments.

2. **Symmetry.** $H_n$ is invariant under any permutation of $(p_1,\ldots,p_n)$: the labeling of outcomes does not matter.

3. **Maximum.** $H_n(p_1,\ldots,p_n) \le H_n(1/n, \ldots, 1/n)$: the uniform distribution maximizes uncertainty.

4. **Branching / Chainability.** Adding a zero-probability outcome does not change uncertainty:
   $$H_n(p_1,\ldots,p_n) = H_{n+1}(p_1,\ldots,p_n,0)$$
   And a two-stage experiment is consistent with the joint:
   $$H_2(p, 1-p) + p \cdot H_n(q_1,\ldots,q_n) = H_{n+1}(pq_1,\ldots,pq_n, 1-p)$$

**Theorem (Khinchin 1953, Shannon 1948).** The *unique* family of functions satisfying these axioms is:

$$H_n(p_1, \ldots, p_n) = -c \sum_{i=1}^n p_i \log p_i$$

for some constant $c > 0$ (which sets the unit: $c = 1/\ln 2$ gives bits).

**Proof sketch.** From Axiom 4 applied inductively, $H$ must satisfy:

$$H_n\!\left(\frac{1}{n},\ldots,\frac{1}{n}\right) = A(n)$$

where $A(n)$ is some function satisfying $A(mn) = A(m) + A(n)$ (additivity) by the branching axiom applied to two uniform distributions. By continuity, the unique solution is $A(n) = c\log n$. Extending to rational probabilities by the branching axiom and then to all probabilities by continuity gives the unique formula $H = -c\sum p_i \log p_i$.

**Why this matters.** The axiomatic derivation shows that there is no alternative: if you want a measure of uncertainty that is symmetric, continuous, maximal for uniform distributions, and consistent with hierarchical decomposition of outcomes, you *must* use Shannon entropy. Any other formula would violate at least one of these natural requirements.

**The role of continuity.** Without the continuity axiom, many pathological functions satisfy the remaining axioms. Continuity ensures that nearly-certain events have nearly-zero entropy — a tiny probability cannot create a large information contribution.

**Variants.** Dropping the symmetry axiom gives directed entropies (measuring asymmetric uncertainty in sequential decisions). Replacing the maximum axiom with a different normalization gives Rényi entropies. The Khinchin axioms are therefore the minimal set characterizing Shannon entropy specifically.

---

## Appendix B: Asymptotic Equipartition Property (AEP)

The **Asymptotic Equipartition Property** is the information-theoretic analogue of the Law of Large Numbers. It is the mathematical foundation of all lossless compression and channel coding.

**Theorem (AEP, Shannon 1948).** Let $X_1, X_2, \ldots$ be i.i.d. random variables with distribution $p$ and entropy $H(X)$. Then:

$$-\frac{1}{n}\log p(X_1, X_2, \ldots, X_n) \xrightarrow{P} H(X) \quad \text{as } n \to \infty$$

**Proof.** By the law of large numbers applied to $-\log p(X_i)$:

$$-\frac{1}{n}\log p(X_1,\ldots,X_n) = -\frac{1}{n}\sum_{i=1}^n \log p(X_i) \xrightarrow{P} \mathbb{E}[-\log p(X)] = H(X)$$

**Interpretation.** The AEP says that for large $n$, all long sequences have approximately the same probability: roughly $2^{-nH}$ (in bits). The set of "typical sequences" — those with probability $\approx 2^{-nH(X)}$ — has total probability approaching 1.

**Typical Set.** Define the **typical set** $\mathcal{T}_\epsilon^{(n)}$ as:

$$\mathcal{T}_\epsilon^{(n)} = \left\{(x_1,\ldots,x_n) : \left|-\frac{1}{n}\log p(x_1,\ldots,x_n) - H(X)\right| < \epsilon\right\}$$

**Properties of the typical set:**
1. $P\!\left((X_1,\ldots,X_n) \in \mathcal{T}_\epsilon^{(n)}\right) > 1 - \epsilon$ for large enough $n$
2. $\lvert \mathcal{T}_\epsilon^{(n)}\rvert \le 2^{n(H(X)+\epsilon)}$ (not too many typical sequences)
3. $\lvert \mathcal{T}_\epsilon^{(n)}\rvert \ge (1-\epsilon)2^{n(H(X)-\epsilon)}$ (not too few)

**Consequence for compression.** To compress $n$ symbols losslessly:
- Assign short codewords to sequences in $\mathcal{T}_\epsilon^{(n)}$: need $\approx nH(X)$ bits to index them
- Assign a flag + enumerate for atypical sequences: needed with vanishing probability

This achieves an average code rate of $H(X) + \epsilon$ bits per symbol, converging to $H(X)$ as $\epsilon \to 0$, $n \to \infty$. This proves the achievability direction of Shannon's source coding theorem for i.i.d. sources.

**AEP for Markov chains.** The AEP generalizes to stationary ergodic processes:

$$-\frac{1}{n}\log p(X_1,\ldots,X_n) \xrightarrow{a.s.} \mathcal{H}$$

where $\mathcal{H}$ is the entropy rate. This is the **Shannon-McMillan-Breiman theorem**. For language models, it says that the per-token loss converges almost surely to the entropy rate of the true language distribution.

**For AI — Training Loss Convergence.** The training cross-entropy loss of an LLM:

$$\hat{\mathcal{H}}_T = -\frac{1}{T}\sum_{t=1}^T \log p_{\boldsymbol{\theta}}(x_t \mid x_{<t})$$

converges (by the ergodic theorem) to $H(p_{\mathrm{true}}, p_{\boldsymbol{\theta}})$ — the cross-entropy rate. When the model is perfect, this equals the true entropy rate of the language, which is the theoretical minimum achievable loss. The AEP tells us that this convergence is almost sure, not just in probability.

---

## Appendix C: Entropy and Channel Capacity

Shannon's information theory has two fundamental theorems: the source coding theorem (Appendix B and §6) and the **channel coding theorem** (noisy channel coding). While channel capacity is beyond the scope of this section (it is a mutual information concept, covered in §03), we briefly preview the connection to entropy.

**Binary Symmetric Channel (BSC).** A BSC with crossover probability $p$ flips each transmitted bit with probability $p$. The maximum rate of reliable communication (channel capacity) is:

$$C = 1 - h(p) \text{ bits per channel use}$$

where $h(p)$ is the binary entropy function. This is the entropy of the output minus the entropy of the noise: you can reliably communicate $1 - h(p)$ bits per channel use even though the channel is noisy.

**Entropy of noise limits communication.** The intuition: the channel introduces $h(p)$ bits of noise entropy per use. Of the $\log_2 2 = 1$ bit per channel use "capacity," $h(p)$ bits are consumed by noise uncertainty, leaving $1 - h(p)$ for reliable information. At $p = 1/2$ (complete randomness): $C = 0$. At $p = 0$ or $p = 1$ (deterministic): $C = 1$.

**Shannon's Channel Coding Theorem (preview).** For any rate $R < C$, there exists a code of rate $R$ bits per channel use that achieves arbitrarily small error probability. Conversely, for any rate $R > C$, the error probability is bounded away from zero. The channel capacity is therefore a sharp threshold — another fundamental limit set by entropy.

Full treatment in: Cover & Thomas (2006), *Elements of Information Theory*, Chapter 7.

---

## Appendix D: Entropy in Quantum Information

Quantum information theory generalizes Shannon entropy to quantum states.

**Von Neumann entropy.** For a density matrix $\rho$ (the quantum analogue of a probability distribution):

$$S(\rho) = -\operatorname{tr}(\rho \log \rho) = -\sum_i \lambda_i \log \lambda_i$$

where $\lambda_i$ are the eigenvalues of $\rho$. For a pure state ($\rho^2 = \rho$, rank 1): $S = 0$. For the maximally mixed state ($\rho = I/n$): $S = \log n$.

**Connection to Shannon entropy.** Von Neumann entropy reduces to Shannon entropy when $\rho$ is diagonal: $\rho = \operatorname{diag}(p_1, \ldots, p_n)$ gives $S(\rho) = H(p_1,\ldots,p_n)$.

**Entanglement entropy.** For a bipartite quantum system $AB$ in state $\rho_{AB}$, the entanglement entropy $S(\rho_A)$ (entropy of the reduced density matrix of subsystem $A$) measures quantum entanglement between $A$ and $B$. When $A$ and $B$ are maximally entangled, $S(\rho_A) = \log d$ where $d$ is the dimension. This is a purely quantum phenomenon with no classical analogue.

**For AI — Quantum ML.** While quantum computing for ML remains largely theoretical (2026), the von Neumann entropy is used in:
- Analyzing the information geometry of quantum circuits
- Quantum kernel methods (quantum feature maps via density matrices)
- Understanding the entanglement structure of tensor network representations of neural networks

---

## Appendix E: Entropy Estimation from Data

In practice, the distribution $p$ is unknown and must be estimated from a finite sample $x_1, \ldots, x_n$.

**Plug-in (naive) estimator.** Use empirical frequencies $\hat{p}(x) = \frac{\text{count}(x)}{n}$:

$$\hat{H}_{\text{naive}} = -\sum_x \hat{p}(x)\log \hat{p}(x)$$

**Bias.** The naive estimator is **biased downward**:

$$\mathbb{E}[\hat{H}_{\text{naive}}] = H(X) - \frac{|\mathcal{X}| - 1}{2n} + O(1/n^2)$$

The bias is $-\frac{|\mathcal{X}|-1}{2n}$ (first-order correction from Miller & Madow, 1955). For large alphabets (e.g., vocabulary size $|\mathcal{V}| = 50{,}000$) with small samples, this bias can be severe.

**Miller-Madow correction.** Add $\frac{\hat{m}-1}{2n}$ where $\hat{m}$ is the number of distinct symbols observed:

$$\hat{H}_{\text{MM}} = \hat{H}_{\text{naive}} + \frac{\hat{m}-1}{2n}$$

**Bayesian estimators.** With a Dirichlet prior $\operatorname{Dir}(\boldsymbol{\alpha})$ on the probability vector, the posterior mean entropy can be computed via the posterior predictive distribution. The **NSB estimator** (Nemenman-Shafee-Bialek, 2002) is a mixture of Dirichlet priors that gives near-optimal entropy estimates for undersampled alphabets.

**Neural entropy estimation.** For continuous random variables, entropy can be estimated via:
- **k-nearest-neighbor estimators** (Kozachenko-Leonenko, 1987): $\hat{h} = \log n - \psi(k) + \log c_d + \frac{d}{n}\sum_i \log r_{k,i}$ where $r_{k,i}$ is the distance to the $k$-th nearest neighbor
- **MINE** (Mutual Information Neural Estimation): variational lower bounds on mutual information via neural networks, which can be repurposed for entropy estimation

**For LLM evaluation.** Estimating the entropy rate of natural language from finite text corpora is done by fitting a language model and measuring its perplexity. The model's perplexity upper bounds $e^{\mathcal{H}}$; as the model improves, the bound tightens. Shannon's original estimate of English entropy used human prediction experiments, not language models.

---

## Appendix F: Log-Sum-Exp and Numerical Stability

Computing entropy in practice requires stable numerical computation of $\sum_x p(x)\log p(x)$.

**The challenge.** In log-space (when you have $\log p(x)$ not $p(x)$ directly), computing entropy requires:

$$H = -\sum_x e^{\ell_x} \ell_x \quad \text{where } \ell_x = \log p(x)$$

If any $\ell_x$ is very negative (small probability), $e^{\ell_x}$ can underflow to 0. If any $\ell_x \approx 0$ (large probability), it can overflow in intermediate steps.

**Numerical entropy computation in PyTorch:**

```python
import torch
import torch.nn.functional as F

def stable_entropy(logits):
    """Compute entropy of softmax distribution from logits."""
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)  # H = -sum(p log p)
    return entropy

# Equivalent: F.cross_entropy(logits, probs) where probs = softmax(logits)
# This uses the log-sum-exp trick internally via log_softmax
```

**Log-sum-exp trick.** To compute $\log\sum_i e^{a_i}$ stably:

$$\log\sum_i e^{a_i} = M + \log\sum_i e^{a_i - M} \quad \text{where } M = \max_i a_i$$

The shifted sum $\sum_i e^{a_i - M}$ has the largest term equal to 1 ($e^0 = 1$), preventing both overflow (largest term is $e^0$) and underflow (other terms are in $(0,1]$).

**Entropy via log-sum-exp:**

$$H(X) = -\sum_x p(x)\log p(x) = \log n - \frac{1}{n}\sum_x D_{\mathrm{KL}}(\delta_x \| p)$$

For neural network outputs $\mathbf{z}$, the entropy of $\operatorname{softmax}(\mathbf{z}/\tau)$ is computed stably using log-softmax:

$$H = -\sum_x \operatorname{softmax}(z_x)\log\operatorname{softmax}(z_x) = -\sum_x p_x(z_x/\tau - \log Z)$$

where $Z = \sum_{x'} e^{z_{x'}/\tau}$ and the computation is done entirely in log-space.

**For AI — Perplexity Computation.** Computing perplexity from a sequence of log-probabilities:

```python
def perplexity(log_probs):
    """log_probs: tensor of shape (T,) with log p(x_t | x_{<t})"""
    return torch.exp(-log_probs.mean())
```

Numerically stable because averaging log-probabilities avoids the product $\prod_t p(x_t)$ which underflows exponentially in $T$.

---

## Appendix G: Entropy in Coding Theory and Compression Algorithms

### G.1 Entropy Coding in Practice

Modern lossless compression combines two components:
1. **Statistical model:** Estimates $p(x_t \mid x_{<t})$ — a probability model (can be adaptive, arithmetic, or neural)
2. **Entropy coder:** Converts probability estimates to bits — arithmetic coding or range coding

The theoretical compression rate is $H(X)$ bits per symbol. The gap from practice comes from:
- Imperfect probability model: adds $D_{\mathrm{KL}}(p_{\mathrm{true}} \| p_{\mathrm{model}})$ bits
- Integer rounding in prefix codes: adds $\le 1$ bit per symbol (avoided by arithmetic coding)
- Block-wise vs. symbol-wise coding: arithmetic coding handles this

### G.2 LLM-Based Compression

A striking recent result: large language models are powerful compressors. By using an LLM as the probability model in arithmetic coding:

- **Chinchilla 70B** achieves ~0.9 bits/byte on text (vs. ~2 bits/byte for gzip, ~1.8 bits/byte for zstd)
- This is near the estimated entropy rate of English (~0.6–1.0 bits/char = ~1–1.5 bits/byte counting spaces/punctuation)

**Deletang et al. (2023), "Language Modeling is Compression."** This paper establishes:
1. A predictor (LLM) can be transformed into a lossless compressor (arithmetic coding on its outputs)
2. A compressor can be transformed into a predictor (context tree weighting, Lempel-Ziv)
3. Compression performance equals prediction performance: lower perplexity = higher compression ratio

**Implication for scaling laws.** Chinchilla scaling laws (Hoffmann et al., 2022) can be derived from information-theoretic principles: the optimal compute allocation follows from minimizing the cross-entropy loss subject to FLOPs and data constraints, which is equivalent to maximizing compression ratio at fixed compute budget.

### G.3 Brotli, Zstandard, and Neural Compression

Modern production compressors (Brotli in HTTP, Zstandard/zstd in databases) use:
- **Context mixing:** Combine multiple probability models via a meta-predictor (similar to ensemble learning)
- **ANS (Asymmetric Numeral Systems):** Modern replacement for arithmetic coding; same compression ratio, faster decode; used in Apple, Meta, NVIDIA compression stacks

Neural network-based learned compression (BPG, WebP2, neural image codecs):
- Train encoder-decoder pair to minimize $\mathbb{E}[D(x, \hat{x})] + \lambda H(Z)$ where $D$ is distortion and $H(Z)$ is the entropy of the compressed representation
- The entropy term $H(Z)$ is approximated by entropy coding the quantized latent $Z$ with an entropy model; this is exactly Shannon entropy in the rate-distortion objective

---

## Appendix H: Entropy in Cryptography and Privacy

**Shannon entropy and secrecy.** Shannon (1949) introduced the concept of **perfect secrecy**: a cipher $C$ is perfectly secret if $I(P; C) = 0$, i.e., the ciphertext is statistically independent of the plaintext. The one-time pad achieves perfect secrecy: if the key $K$ is uniform and independent, then $H(P \mid C) = H(P)$ — knowing the ciphertext tells you nothing.

**Entropy and passwords.** Password strength is often measured in "bits of entropy" — but this means the log of the number of equally likely passwords in the password space, which equals $H_0$ (Hartley entropy / support size) in log₂, not Shannon entropy. A strong password has high Hartley entropy (many possible values) but may have low min-entropy (if certain patterns are more common).

**Differential privacy and Rényi entropy.** Differential privacy (DP) protects individual records in a dataset. The standard $(\epsilon, \delta)$-DP has been largely replaced by:

- **Rényi DP (Mironov 2017):** $M$ is $(\alpha, \epsilon)$-Rényi DP if $D_\alpha(M(\mathcal{D}) \| M(\mathcal{D}')) \le \epsilon$ where $D_\alpha$ is the Rényi divergence. Composing $T$ Rényi-DP mechanisms gives $(T\epsilon)$-Rényi DP — much tighter than standard DP composition.
- **Privacy amplification by sampling:** Random subsampling of the dataset amplifies privacy; the analysis uses Rényi divergence properties.

**For LLM training with DP:** DP-SGD (Abadi et al., 2016) clips and noises gradients per sample. The privacy accounting uses Rényi DP moments for tight composition. Understanding Rényi entropy is necessary to analyze the privacy guarantees of differentially private language model training.

---

## Appendix I: Connection to Statistical Mechanics

Shannon did not derive entropy from scratch — he was informed by thermodynamic entropy. The connection is deep and precise.

**Boltzmann entropy.** For a physical system with $W$ equally likely microstates corresponding to a given macrostate:

$$S_{\text{Boltzmann}} = k_B \ln W$$

where $k_B$ is Boltzmann's constant. This is Hartley entropy (Shannon entropy with uniform distribution) scaled by $k_B$.

**Gibbs entropy.** For a system with probability distribution $p_i$ over microstates:

$$S_{\text{Gibbs}} = -k_B \sum_i p_i \ln p_i$$

This is exactly Shannon entropy scaled by $k_B$. Shannon was aware of Gibbs' work; his advisor John von Neumann reportedly told him to call the new quantity "entropy" because "nobody knows what entropy really is, so in a debate you will always have the advantage."

**Partition function = normalizing constant.** The partition function $Z = \sum_x e^{-\beta E(x)}$ in statistical mechanics is the same normalizing constant that appears in MaxEnt probability models (§5.2). Temperature $T = 1/(\beta k_B)$ in physics corresponds to the temperature parameter $\tau$ in softmax: high temperature = high entropy.

**Free energy and ELBO.** The Helmholtz free energy $F = U - TS = \mathbb{E}[E] - T\cdot S$ is the variational objective in statistical mechanics. The ELBO in variational inference is:

$$\mathcal{L} = \mathbb{E}_q[\log p(\mathbf{x}, \mathbf{z})] - \mathbb{E}_q[\log q(\mathbf{z})] = \mathbb{E}_q[\log p(\mathbf{x} \mid \mathbf{z})] - D_{\mathrm{KL}}(q \| p)$$

Setting $U = -\mathbb{E}_q[\log p(\mathbf{x} \mid \mathbf{z})]$ (reconstruction energy) and $S = H(q(\mathbf{z} \mid \mathbf{x}))$ (encoder entropy):

$$\mathcal{L} = -U + TS \propto -(U - TS) = -F$$

Maximizing the ELBO is equivalent to minimizing the variational free energy — the same variational principle as statistical mechanics. The VAE is literally learning the statistical mechanics of the data distribution.

---

## Appendix J: Further Reading and References

### Canonical Textbooks

1. **Cover & Thomas (2006)** — *Elements of Information Theory*, 2nd ed.
   The definitive graduate textbook. Chapters 1-3 cover everything in this section rigorously.
   URL: https://www.wiley.com/en-us/Elements+of+Information+Theory%2C+2nd+Edition-p-9780471241959

2. **MacKay (2003)** — *Information Theory, Inference, and Learning Algorithms*.
   Free online; excellent for ML practitioners. Chapter 1-8 cover entropy and coding.
   URL: http://www.inference.org.uk/itila/

3. **Shannon (1948)** — *A Mathematical Theory of Communication*.
   The original paper. Remarkably readable.

4. **Jaynes (2003)** — *Probability Theory: The Logic of Science*.
   The MaxEnt perspective; Bayesian interpretation of entropy.

### Papers for AI/ML Context

5. **Haarnoja et al. (2018)** — *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*. arXiv:1801.01290.

6. **Deletang et al. (2023)** — *Language Modeling is Compression*. ICLR 2024.

7. **Mironov (2017)** — *Rényi Differential Privacy*. arXiv:1702.07476.

8. **Guo et al. (2017)** — *On Calibration of Modern Neural Networks*. ICML 2017.

9. **Nemenman, Shafee & Bialek (2002)** — *Entropy and Inference, Revisited*. NeurIPS 2001.

### Online Resources

- MIT OpenCourseWare 6.441: Information Theory lectures — covers all material in this chapter
- Stanford EE376A: Information Theory — excellent problem sets on entropy and coding

---

## Appendix K: Worked Examples and Computations

### K.1 Entropy of English Text

**Letter-level entropy of English.** The distribution of English letters has been estimated from large corpora. Using Brown corpus frequencies:

| Letters | $p(x)$ | $-\log_2 p(x)$ | $p(x)\log_2 p(x)$ |
| --- | --- | --- | --- |
| e | 0.1270 | 2.977 | 0.378 |
| t | 0.0906 | 3.464 | 0.314 |
| a | 0.0817 | 3.615 | 0.295 |
| ... | ... | ... | ... |
| z | 0.0007 | 10.48 | 0.007 |

The letter-level entropy of English (26 letters, case-insensitive) is approximately:

$$H(\text{letters}) \approx 4.18 \text{ bits/letter}$$

This is much less than $\log_2 26 \approx 4.70$ bits (uniform), reflecting that some letters (e, t, a) are far more common than others (q, z, x).

**Including spaces and punctuation:** $H \approx 4.3$–$4.5$ bits/character.

**Word-level entropy:** With a 10,000-word vocabulary, a rough estimate gives $H \approx 9$–$10$ bits/word.

**Shannon's estimate of English entropy rate:** Shannon (1951) estimated the entropy rate at approximately 1.0–1.3 bits/character using a human prediction experiment. More recent estimates with n-gram models give 1.5–2 bits/character; with modern LLMs the estimate decreases as models capture more long-range structure.

**Perplexity of a uniform letter model:** $2^{4.18} \approx 18$ — a model knowing only letter frequencies is as uncertain as a uniform distribution over 18 letters.

**LLM perplexity comparison.** GPT-4 on English text achieves approximately $e^{1.8} \approx 6$ in bits/token (roughly 1.5 bits/character). This means the model is, on average, as uncertain as a uniform distribution over about 6 equally likely next tokens — vastly better than a letter-frequency model.

### K.2 Entropy Computation for Joint Distributions

**Example.** Two correlated binary random variables $X$ (weather: sunny/rainy) and $Y$ (activity: indoor/outdoor):

| | $Y=0$ (indoor) | $Y=1$ (outdoor) | $p(X)$ |
| --- | --- | --- | --- |
| $X=0$ (rainy) | 0.40 | 0.05 | 0.45 |
| $X=1$ (sunny) | 0.10 | 0.45 | 0.55 |
| $p(Y)$ | 0.50 | 0.50 | 1.00 |

**Computations (in nats):**

$$H(X) = -0.45\ln 0.45 - 0.55\ln 0.55 \approx 0.688 \text{ nats}$$

$$H(Y) = -0.50\ln 0.50 - 0.50\ln 0.50 = \ln 2 \approx 0.693 \text{ nats}$$

$$H(X,Y) = -0.40\ln 0.40 - 0.05\ln 0.05 - 0.10\ln 0.10 - 0.45\ln 0.45 \approx 1.032 \text{ nats}$$

$$H(X \mid Y) = H(X,Y) - H(Y) \approx 1.032 - 0.693 = 0.339 \text{ nats}$$

$$H(Y \mid X) = H(X,Y) - H(X) \approx 1.032 - 0.688 = 0.344 \text{ nats}$$

$$I(X;Y) = H(X) + H(Y) - H(X,Y) \approx 0.688 + 0.693 - 1.032 = 0.349 \text{ nats}$$

**Verification:** $H(X,Y) = 1.032 \le H(X) + H(Y) = 1.381$ ✓ (subadditivity)  
$H(X \mid Y) = 0.339 \le H(X) = 0.688$ ✓ (conditioning reduces entropy)  
$I(X;Y) = 0.349 \ge 0$ ✓ (mutual information non-negative)

If $X \perp\!\!\!\perp Y$: $H(X,Y) = H(X) + H(Y) = 1.381$; the actual $H(X,Y) = 1.032 < 1.381$ confirms dependence.

### K.3 Computing Entropy Rate of a Markov Chain Numerically

**Two-state chain.** States: $\{0,1\}$. Transition matrix:
$$P = \begin{pmatrix} 0.8 & 0.2 \\ 0.3 & 0.7 \end{pmatrix}$$

**Step 1: Stationary distribution.** Solve $\boldsymbol{\mu}^\top P = \boldsymbol{\mu}^\top$:
- $\mu_0 \cdot 0.8 + \mu_1 \cdot 0.3 = \mu_0$
- $\mu_0 + \mu_1 = 1$
- Solution: $\mu_0 = 3/5 = 0.6$, $\mu_1 = 2/5 = 0.4$

**Step 2: Per-state conditional entropies:**
- $H(X_2 \mid X_1 = 0) = h(0.2) = -0.2\ln 0.2 - 0.8\ln 0.8 \approx 0.500$ nats
- $H(X_2 \mid X_1 = 1) = h(0.3) = -0.3\ln 0.3 - 0.7\ln 0.7 \approx 0.611$ nats

**Step 3: Entropy rate:**
$$\mathcal{H} = 0.6 \times 0.500 + 0.4 \times 0.611 = 0.300 + 0.244 = 0.544 \text{ nats}$$

**Comparison to i.i.d.:** If each state were drawn independently from $\boldsymbol{\mu} = (0.6, 0.4)$:
$$H(X_1) = -0.6\ln 0.6 - 0.4\ln 0.4 \approx 0.673 \text{ nats}$$

The Markov chain has lower entropy rate ($0.544 < 0.673$) because there is temporal structure: state 0 tends to follow state 0, state 1 tends to follow state 1.

**Perplexity:** $e^{0.544} \approx 1.72$ — the Markov chain is approximately as uncertain as a uniform distribution over 1.72 symbols, a very predictable process.

### K.4 Maximum Entropy Distributions: A Summary Table

| Support | Constraints | MaxEnt distribution | Entropy |
| --- | --- | --- | --- |
| $\{1,\ldots,n\}$ (discrete) | None | Uniform $1/n$ | $\ln n$ |
| $\{0,1\}$ (binary) | $\mathbb{E}[X] = p$ | Bernoulli$(p)$ | $h(p)$ |
| $\{0,1,2,\ldots\}$ (integers) | $\mathbb{E}[X] = \mu$ | Geometric (modified) | — |
| $\mathbb{R}_{\ge 0}$ (non-negative reals) | $\mathbb{E}[X] = \mu$ | Exponential$(1/\mu)$ | $1 + \ln\mu$ |
| $[a,b]$ (bounded interval) | None | Uniform$[a,b]$ | $\ln(b-a)$ |
| $\mathbb{R}$ (all reals) | $\mathbb{E}[X]=\mu$, $\operatorname{Var}(X)=\sigma^2$ | $\mathcal{N}(\mu,\sigma^2)$ | $\frac{1}{2}\ln(2\pi e\sigma^2)$ |
| $\mathbb{R}^d$ (multivariate) | $\mathbb{E}[\mathbf{X}]=\boldsymbol{\mu}$, $\operatorname{Cov}(\mathbf{X})=\Sigma$ | $\mathcal{N}(\boldsymbol{\mu},\Sigma)$ | $\frac{1}{2}\ln\det(2\pi e\Sigma)$ |
| $\mathbb{R}$ with $\mathbb{E}[\lvert X\rvert]=b$ | Sign symmetric | Laplace$(0,b)$ | $1+\ln(2b)$ |

This table is a fundamental reference: the Gaussian arises from constraining only the first two moments, nothing more. Assuming any other continuous distribution under a variance constraint is asserting additional structure not supported by the constraints.

---

## Appendix L: Entropy Regularization: Mathematical Details

### L.1 Mirror Descent and Entropy

**Regularized optimization.** Consider minimizing $f(\mathbf{p})$ subject to $\mathbf{p} \in \Delta_n$ (simplex). Adding entropy regularization:

$$\min_{\mathbf{p} \in \Delta_n} f(\mathbf{p}) - \tau H(\mathbf{p})$$

The term $-\tau H(\mathbf{p})$ penalizes distributions far from uniform. This is equivalent to the **negative entropy regularizer** in mirror descent.

**Mirror descent update.** The mirror descent algorithm with entropy mirror map gives the update:

$$p_{t+1}(x) \propto p_t(x) \cdot \exp(-\eta_t \nabla_{p_t} f(p_t)(x))$$

This is exactly the **multiplicative weights** algorithm — the natural gradient descent on the simplex with entropy as the Bregman potential.

**Softmax as MaxEnt regularization.** The softmax attention mechanism:

$$\alpha_i = \frac{e^{q_i}}{\sum_j e^{q_j}} = \arg\max_{\mathbf{p} \in \Delta_n}\left[\sum_i p_i q_i + H(\mathbf{p})\right]$$

is the unique distribution that maximizes the linear objective $\sum_i p_i q_i$ (expected score) subject to entropy regularization. Without entropy regularization, the optimal distribution would be the argmax (hardmax); the entropy term smooths this to softmax.

### L.2 The Information Bottleneck

The **Information Bottleneck** (Tishby, Pereira & Bialek, 1999) provides an information-theoretic framework for representation learning:

$$\min_{p(Z \mid X)} I(X; Z) - \beta I(Z; Y)$$

Find a compression $Z$ of input $X$ that:
- Minimizes mutual information $I(X;Z)$ (compress $X$, reduce information stored in $Z$)
- Maximizes mutual information $I(Z;Y)$ (preserve information about label $Y$)

The parameter $\beta$ trades off compression vs. prediction. At $\beta = 0$: compress maximally (throw away all information). At $\beta \to \infty$: preserve all information about $Y$ (no compression).

**Connection to entropy.** The IB objective is a function of entropy quantities:
$$I(X;Z) = H(Z) - H(Z \mid X) = H(X) - H(X \mid Z)$$

The IB Lagrangian can be solved analytically when $X, Y, Z$ are jointly Gaussian: $Z = AX + \boldsymbol{\epsilon}$ is optimal for Gaussian data.

**Deep learning as IB.** Tishby & Schwartz-Ziv (2017) proposed that deep neural networks implicitly solve the IB: early layers maximize $I(Z;Y)$ (fitting), while later layers minimize $I(X;Z)$ (compression/generalization). This remains an active research area.

### L.3 Entropy and Generalization in PAC Learning

The PAC (Probably Approximately Correct) framework uses entropy to bound generalization:

**Occam's Razor Bound.** For a hypothesis class $\mathcal{H}$ and any distribution $P_\mathcal{H}$ over hypotheses, the expected generalization error satisfies:

$$\mathbb{E}_{h \sim P_{\mathcal{H}}}[\text{gen-error}(h)] \le \sqrt{\frac{D_{\mathrm{KL}}(P_{\mathcal{H}} \| Q_{\mathcal{H}}) + \ln(1/\delta)}{2n}}$$

where $Q_\mathcal{H}$ is a prior over hypotheses and $n$ is the number of training samples. This is the **PAC-Bayes bound** (McAllester 1999), and the KL divergence term $D_{\mathrm{KL}}(P_{\mathcal{H}} \| Q_\mathcal{H})$ penalizes complexity of the learned distribution relative to the prior.

**Mutual information bound (Xu & Raginsky 2017).** The generalization gap is bounded by:

$$|\text{gen-error}| \le \sqrt{\frac{I(W; S)}{2n}}$$

where $W$ are the model weights and $S$ is the training dataset. Mutual information $I(W;S)$ measures how much the weights "remember" the training data — a form of entropy-based overfitting measure.

These connections show that entropy and information theory provide rigorous foundations for understanding why deep learning generalizes — not just heuristic explanations.


---

## Appendix M: Entropy in Generative Models

### M.1 Diffusion Models and Entropy

**Denoising diffusion models** (Ho et al., 2020; Song et al., 2020) define a forward process that gradually adds Gaussian noise:

$$q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}\,\mathbf{x}_{t-1}, \beta_t I)$$

The forward process converges to $\mathcal{N}(\mathbf{0}, I)$ — the maximum differential entropy distribution under unit variance. The total entropy increases monotonically along the forward process:

$$h(\mathbf{x}_0) \le h(\mathbf{x}_1) \le \cdots \le h(\mathbf{x}_T) = \frac{d}{2}\ln(2\pi e)$$

where $d$ is the dimension. The reverse process (denoising) decreases entropy back to $h(\mathbf{x}_0)$.

**ELBO for diffusion.** The training objective is a variational lower bound on $\log p_{\boldsymbol{\theta}}(\mathbf{x}_0)$ that can be written as a sum of KL divergences:

$$\mathcal{L}_{\text{ELBO}} = \underbrace{D_{\mathrm{KL}}(q(\mathbf{x}_T \mid \mathbf{x}_0) \| p(\mathbf{x}_T))}_{\text{prior matching}} + \sum_{t>1}\underbrace{D_{\mathrm{KL}}(q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) \| p_{\boldsymbol{\theta}}(\mathbf{x}_{t-1} \mid \mathbf{x}_t))}_{\text{denoising step}} - \underbrace{\log p_{\boldsymbol{\theta}}(\mathbf{x}_0 \mid \mathbf{x}_1)}_{\text{reconstruction}}$$

Each KL divergence is the entropy gap between the true reverse process and the learned reverse process.

### M.2 Normalizing Flows and Entropy

**Normalizing flows** learn a bijection $f_{\boldsymbol{\theta}}: \mathbb{R}^d \to \mathbb{R}^d$ from a simple distribution $p_Z$ to the data distribution $p_X$. The change-of-variables formula gives:

$$h(\mathbf{X}) = h(\mathbf{Z}) + \mathbb{E}[\log |\det J_{f_{\boldsymbol{\theta}}}(\mathbf{Z})|]$$

where $J_{f_{\boldsymbol{\theta}}}$ is the Jacobian of $f_{\boldsymbol{\theta}}$. The Jacobian term adds entropy based on how much the flow expands or contracts volumes. Training maximizes $\log p_{\boldsymbol{\theta}}(\mathbf{x}) = \log p_Z(f^{-1}(\mathbf{x})) + \log |\det J_{f^{-1}}(\mathbf{x})|$ — a combination of the base entropy and the volume scaling.

### M.3 GANs and Entropy

**Generative Adversarial Networks** (Goodfellow et al., 2014) train a generator $G$ and discriminator $D$. The optimal generator minimizes the **Jensen-Shannon divergence**:

$$\mathrm{JSD}(p_{\mathrm{data}} \| p_G) = \frac{1}{2}D_{\mathrm{KL}}\!\left(p_{\mathrm{data}} \| \frac{p_{\mathrm{data}}+p_G}{2}\right) + \frac{1}{2}D_{\mathrm{KL}}\!\left(p_G \| \frac{p_{\mathrm{data}}+p_G}{2}\right)$$

The JSD is a symmetrized, bounded version of KL divergence. It equals zero iff $p_G = p_{\mathrm{data}}$.

**Mode collapse = entropy collapse.** A GAN suffering from mode collapse has a generator with low entropy: $H(p_G) \ll H(p_{\mathrm{data}})$. The generator is only producing a small number of distinct outputs. This is an entropy problem: the generator has learned a low-entropy distribution. Solutions include entropy regularization of the generator (minibatch discrimination, Unrolled GAN, etc.).

---

## Appendix N: Han's Inequality and Shearer's Lemma

### N.1 Han's Inequality

**Theorem (Han's Inequality, 1978).** For jointly distributed random variables $X_1, \ldots, X_n$:

$$H(X_1, \ldots, X_n) \le \frac{1}{n-1} \sum_{i=1}^n H(X_1, \ldots, X_{i-1}, X_{i+1}, \ldots, X_n)$$

In words: the joint entropy is no more than the average of the $(n-1)$-subset joint entropies.

**Proof sketch.** Define $H_{-i} = H(X_1,\ldots,X_n) - H(X_i \mid X_{-i})$ where $X_{-i}$ denotes all variables except $X_i$. Summing over $i$ and using the chain rule gives the result.

**For ML.** Han's inequality is used to prove submodularity of entropy: the function $S \mapsto H(X_S)$ (where $X_S = (X_i : i \in S)$ is a subset of variables) is **submodular**:

$$H(X_{S \cup T}) + H(X_{S \cap T}) \le H(X_S) + H(X_T)$$

Submodularity is exploited in feature selection algorithms (greedy feature selection maximizing information gain is approximately optimal due to submodularity), sensor placement, and document summarization.

### N.2 Shearer's Lemma

**Lemma (Shearer 1986).** Let $\mathcal{F}$ be a collection of subsets of $\{1,\ldots,n\}$ such that each element $i$ appears in at least $k$ sets in $\mathcal{F}$. Then:

$$k \cdot H(X_1,\ldots,X_n) \le \sum_{S \in \mathcal{F}} H(X_S)$$

Shearer's lemma is a generalization of subadditivity and is used in combinatorics to prove counting inequalities. In information theory, it provides tight bounds on joint entropy from marginal entropies.

**For ML — Federated Learning.** In federated learning with $n$ clients, each holding data subset $\mathcal{D}_i$, Shearer's lemma bounds the amount of information the combined model can learn from the distributed data relative to what each client knows individually.

---

## Appendix O: Summary of Key Entropy Inequalities

```
ENTROPY INEQUALITY REFERENCE
════════════════════════════════════════════════════════════════════════

  DISCRETE ENTROPY:

  Non-negativity:       0 ≤ H(X) ≤ log |X|
  MaxEnt:               H(X) = log n  ⟺  X ~ Uniform
  Conditioning:         H(X|Y) ≤ H(X)  with equality ⟺  X⊥Y
  Subadditivity:        H(X,Y) ≤ H(X) + H(Y)  with equality ⟺  X⊥Y
  Chain rule:           H(X,Y) = H(X) + H(Y|X)
  Chain rule (n vars):  H(X₁,...,Xₙ) = Σᵢ H(Xᵢ | X₁,...,Xᵢ₋₁)
  Data processing:      H(f(X)) ≤ H(X)  (bijection: equality)
  Han's inequality:     H(X₁,...,Xₙ) ≤ (1/(n-1)) Σᵢ H(X₋ᵢ)

  DIFFERENTIAL ENTROPY:

  Scale:                h(aX) = h(X) + log|a|
  Gaussian MaxEnt:      h(X) ≤ ½ log(2πe σ²)  given Var(X) = σ²
  Multivariate MaxEnt:  h(X) ≤ ½ log det(2πe Σ)  given Cov(X) = Σ
  Conditioning:         h(X|Y) ≤ h(X)  (differential version)

  RÉNYI ENTROPY:

  Ordering:             H₀(X) ≥ H₁(X) ≥ H₂(X) ≥ ... ≥ H∞(X)
  Shannon limit:        lim_{α→1} Hα(X) = H(X)
  Min-entropy:          H∞(X) = -log max_x p(x)
  Guessing:             P(guess X correctly) = 2^{-H∞(X)}

  KL DIVERGENCE (preview):

  Non-negativity:       DKL(p‖q) ≥ 0  with equality ⟺  p = q
  Cross-entropy:        H(p,q) = H(p) + DKL(p‖q) ≥ H(p)
  MaxEnt via KL:        H(X) ≤ log n  ⟺  DKL(p ‖ uniform) ≥ 0

════════════════════════════════════════════════════════════════════════
```

This inequality table is a key reference for the entire information theory chapter. Every inequality here will be used in §02 (KL divergence derivations), §03 (mutual information bounds), §04 (cross-entropy loss analysis), and §05 (Fisher information and Cramér-Rao).


---

## Appendix P: Implementation Recipes

### P.1 Entropy Computation — Production Patterns

```python
import numpy as np
from scipy.special import entr, rel_entr

# Pattern 1: From probability vector (numerically stable via scipy)
def entropy_from_probs(p):
    """H(X) in nats. entr(p) = -p*log(p) with 0*log(0)=0 convention."""
    return np.sum(entr(p))

# Pattern 2: From logits (e.g., neural network outputs)
def entropy_from_logits(logits, temperature=1.0):
    """H(softmax(logits/T)) in nats. Numerically stable."""
    logits = logits / temperature
    log_Z = np.log(np.sum(np.exp(logits - logits.max()))) + logits.max()
    log_probs = logits - log_Z  # log softmax
    probs = np.exp(log_probs)
    return -np.sum(probs * log_probs)

# Pattern 3: Conditional entropy H(X|Y) from joint table
def conditional_entropy(joint):
    """joint[i,j] = p(X=i, Y=j). Returns H(X|Y) in nats."""
    py = joint.sum(axis=0)  # p(Y)
    hxy = np.sum(entr(joint))  # H(X,Y)
    hy = np.sum(entr(py))     # H(Y)
    return hxy - hy  # chain rule

# Pattern 4: Entropy rate of Markov chain
def markov_entropy_rate(P):
    """P: transition matrix. Returns entropy rate in nats."""
    n = P.shape[0]
    # Stationary distribution via eigenvector
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    mu = np.real(eigvecs[:, idx])
    mu = mu / mu.sum()
    # Entropy rate = sum_i mu_i * H(P[i,:])
    return np.sum(mu * np.array([np.sum(entr(P[i, :])) for i in range(n)]))

# Pattern 5: Perplexity from log-probabilities
def perplexity(log_probs):
    """log_probs: array of log p(x_t | x_{<t}). Returns PPL."""
    return np.exp(-np.mean(log_probs))

# Pattern 6: Rényi entropy
def renyi_entropy(p, alpha):
    """Rényi entropy of order alpha in nats."""
    if np.isclose(alpha, 1.0):
        return np.sum(entr(p))  # Shannon limit
    if np.isinf(alpha):
        return -np.log(np.max(p))  # Min-entropy
    if np.isclose(alpha, 0.0):
        return np.log(np.sum(p > 0))  # Hartley
    return (1.0 / (1.0 - alpha)) * np.log(np.sum(p ** alpha))
```

### P.2 PyTorch One-Liners

```python
import torch
import torch.nn.functional as F

# Entropy from logits (vectorized over batch)
def batch_entropy(logits, dim=-1):
    """Returns entropy H(softmax(logits)) for each sample. Shape: [batch]"""
    log_p = F.log_softmax(logits, dim=dim)
    p = log_p.exp()
    return -(p * log_p).sum(dim=dim)

# Cross-entropy loss (= H(p_data, p_model))
loss = F.cross_entropy(logits, targets)  # = -log p_model(y_true)

# KL divergence: D_KL(p || q) where p, q are probability vectors
kl = F.kl_div(q.log(), p, reduction='batchmean')  # Note: F.kl_div(log_q, p)

# Per-token perplexity
log_probs = F.log_softmax(logits, dim=-1)
token_nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
ppl = token_nll.mean().exp()
```

### P.3 Common Pitfalls in Code

```python
# WRONG: Does not handle p=0 (nan from log(0))
H_wrong = -np.sum(p * np.log(p))  # nan when p[i] == 0

# CORRECT: Use scipy.special.entr or add epsilon
from scipy.special import entr
H_correct = np.sum(entr(p))  # handles p=0 → 0

# WRONG: Mixing log bases
H_bits = -np.sum(p * np.log2(p))   # bits
H_nats = -np.sum(p * np.log(p))    # nats
# Cross-entropy with mixed bases will give wrong numerical values

# WRONG: Computing PPL from probability products (underflows)
ppl_wrong = np.prod(probs) ** (-1.0/T)  # underflows for T > ~100

# CORRECT: Compute in log space
ppl_correct = np.exp(-np.mean(np.log(probs)))  # stable
```

---

## Appendix Q: Entropy in Transformer Attention

### Q.1 Attention Entropy as a Diagnostic

Each attention head in a transformer computes:

$$\alpha_{ij} = \frac{e^{q_i^\top k_j / \sqrt{d_k}}}{\sum_{j'} e^{q_i^\top k_{j'} / \sqrt{d_k}}}$$

The **attention entropy** for query $i$ in head $h$ is:

$$H_h(i) = -\sum_j \alpha_{ij} \log \alpha_{ij}$$

This ranges from $0$ (attending to exactly one position — argmax attention) to $\log T$ (attending uniformly to all $T$ positions).

**Attention entropy diagnostics:**
- **Low entropy heads** (sharp, local attention): typically handle syntactic dependencies (subject-verb agreement, coreference)
- **High entropy heads** (diffuse, global attention): often handle semantic averaging or positional patterns
- **Entropy collapse:** when fine-tuning collapses all heads to very low entropy, the model loses diversity and may overfit
- **Dead heads:** entropy near $\log T$ (uniform) indicates a head that does not attend meaningfully; often pruned safely

**For interpretability.** Mechanistic interpretability studies (Olsson et al., 2022 — "induction heads"; Elhage et al., 2021 — "Mathematical Framework for Transformer Circuits") characterize attention heads by their entropy patterns. High-entropy heads tend to be backup circuits; low-entropy heads implement specific algorithms.

### Q.2 Temperature and Entropy in Decoding

During LLM generation, three common sampling strategies adjust the output entropy:

| Strategy | Formula | Effect on entropy |
| --- | --- | --- |
| Greedy | $x = \arg\max_x p(x)$ | $H = 0$ (deterministic) |
| Temperature | $p_\tau(x) \propto e^{z_x/\tau}$ | $H \propto \tau$; higher $\tau$ → higher $H$ |
| Top-$k$ | Restrict to top-$k$ tokens, renormalize | $H \le \log k$ |
| Top-$p$ (nucleus) | Restrict to smallest set with $\sum p \ge p_{\text{thresh}}$ | $H$ varies with context |
| Min-$p$ | Restrict to tokens with $p(x) \ge p_{\min} \cdot \max_x p(x)$ | Adaptive entropy bound |

**The key insight:** All sampling strategies are entropy control mechanisms. They interpolate between $H = 0$ (deterministic, no creativity) and $H = H_{\max}$ (random sampling from the full distribution). Good generation requires entropy calibration: enough uncertainty to be interesting, not so much as to be incoherent.

**Entropy-adaptive sampling.** Recent work (Entropix, 2024) uses the token-level entropy and varentropy (variance of entropy across the vocabulary) to dynamically adjust sampling strategy:
- Low entropy + low varentropy → greedy (model is confident)
- High entropy + high varentropy → sample with high temperature (genuine uncertainty)
- Low entropy + high varentropy → sample carefully (multiple plausible branches)


---

## Appendix R: Varentropy and Higher-Order Entropy Statistics

**Varentropy** (variance of the entropy) is a recently popularized concept in LLM decoding:

$$\operatorname{Varentropy}(X) = \operatorname{Var}[-\log p(X)] = \mathbb{E}[(\log p(X))^2] - (H(X))^2$$

While $H(X) = \mathbb{E}[-\log p(X)]$ measures average surprise, varentropy measures the spread of the surprise distribution. A high-entropy, low-varentropy distribution is one where all outcomes have nearly equal probability (uniform-like). A high-entropy, high-varentropy distribution has a few very likely outcomes and many very unlikely ones — a different shape that calls for different sampling strategies.

**Third and fourth moments.** The full distribution of $-\log p(X)$ (called the "information spectrum" or "entropy density") characterizes how information is distributed across outcomes. The skewness and kurtosis of this distribution are used in information spectrum analysis and Rényi entropy comparisons.

**For LLMs.** The entropy and varentropy of the output distribution at each generation step provide a 2D signal:

```
ENTROPY-VARENTROPY DECISION SPACE
════════════════════════════════════════════════════════════════════════

  High varentropy │ Uncertain (branching)    │ Noisy (sample broadly)
                  │ → careful sampling       │ → high temperature
                  ├──────────────────────────┤
  Low varentropy  │ Confident (clear choice) │ Uniform (stuck)
                  │ → greedy / low temp      │ → restart / probe
                  └──────────────────────────┘
                   Low entropy              High entropy

════════════════════════════════════════════════════════════════════════
```

**Entropix sampling** (Xjdr et al., 2024) implements entropy-adaptive sampling using both metrics. It represents an application of higher-order information theory to practical LLM inference quality improvement.

---

## Appendix S: Notation Quick Reference

| Symbol | Meaning | Definition |
| --- | --- | --- |
| $I(x)$ | Self-information | $-\log p(x)$ |
| $H(X)$ | Shannon entropy | $-\sum_x p(x)\log p(x)$ |
| $H(\mathbf{p})$ | Entropy of distribution | Same as $H(X)$, emphasis on $\mathbf{p}$ |
| $h(p)$ | Binary entropy function | $-p\log p - (1-p)\log(1-p)$ |
| $H(X,Y)$ | Joint entropy | $-\sum_{x,y}p(x,y)\log p(x,y)$ |
| $H(X \mid Y)$ | Conditional entropy | $H(X,Y) - H(Y)$ |
| $\mathcal{H}$ | Entropy rate | $\lim_{n\to\infty}\frac{1}{n}H(X_1,\ldots,X_n)$ |
| $h(X)$ | Differential entropy | $-\int p(x)\log p(x)\,dx$ |
| $H_\alpha(X)$ | Rényi entropy | $\frac{1}{1-\alpha}\log\sum_x p(x)^\alpha$ |
| $H_\infty(X)$ | Min-entropy | $-\log\max_x p(x)$ |
| $S_q(X)$ | Tsallis entropy | $\frac{1-\sum_x p(x)^q}{q-1}$ |
| $\operatorname{PPL}$ | Perplexity | $\exp(-\frac{1}{T}\sum_t \log p(x_t \mid x_{<t}))$ |
| $I(X;Y)$ | Mutual information | $H(X) + H(Y) - H(X,Y)$ (preview: §03) |
| $D_{\mathrm{KL}}(p\|q)$ | KL divergence | $\sum_x p(x)\log\frac{p(x)}{q(x)}$ (preview: §02) |
| $H(p,q)$ | Cross-entropy | $H(p) + D_{\mathrm{KL}}(p\|q)$ (preview: §04) |




