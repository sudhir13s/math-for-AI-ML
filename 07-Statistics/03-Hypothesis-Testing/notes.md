[← Back to Chapter 7: Statistics](../README.md) | [Next: Bayesian Inference →](../04-Bayesian-Inference/notes.md)

---

# Hypothesis Testing

> _"To call in the statistician after the experiment is done may be no more than asking him to perform a post-mortem examination: he may be able to say what the experiment died of."_
> — Sir Ronald A. Fisher, *Presidential Address to the First Indian Statistical Congress* (1938)

## Overview

Hypothesis testing is the art of making principled decisions from data under uncertainty. Where estimation theory (§02) asks "what is the value of this parameter?", hypothesis testing asks a sharper question: "is this parameter consistent with a specific claim?" This inversion — from continuous estimation to binary decision — is the formal machinery behind every A/B experiment that ships a product feature, every clinical trial that approves a drug, and every benchmark comparison that claims one model outperforms another.

The discipline has two intertwined origins. Ronald Fisher developed the **p-value framework** in the 1920s: compute how surprising the data would be if the null hypothesis were true, and report that probability as evidence. Jerzy Neyman and Egon Pearson developed the **decision-theoretic framework** in 1933: pre-commit to a decision rule with controlled error rates before seeing the data. Modern practice blends both — using p-values as a continuous measure of evidence while respecting the Neyman-Pearson discipline of pre-specified significance levels, power analysis, and sample size planning.

For AI and ML, hypothesis testing has never been more important. Every benchmark leaderboard is an implicit multiple-comparison experiment susceptible to false-discovery inflation. Every online A/B test deployed at scale faces the sequential testing problem. Every data pipeline needs distributional drift detection. This section builds the complete framework: from the formal definition of a test statistic through the Neyman-Pearson lemma, classical t/χ²/F tests, likelihood ratio tests, multiple testing correction, nonparametric methods, and the sequential A/B testing infrastructure that powers modern ML deployment.

## Prerequisites

- **Confidence intervals and asymptotic normality of MLE** — [§02 Estimation Theory](../02-Estimation-Theory/notes.md)
- **Sampling distributions** (t, χ², F distributions) — [Ch6 §02 Common Distributions](../../06-Probability-Theory/02-Common-Distributions/notes.md)
- **Law of large numbers and CLT** — [Ch6 §05 Limit Theorems](../../06-Probability-Theory/05-Limit-Theorems/notes.md)
- **Expectation and variance** — [Ch6 §04 Expectation and Moments](../../06-Probability-Theory/04-Expectation-and-Moments/notes.md)
- **Likelihood functions** (log-likelihood, score function) — [§02 §4–§5](../02-Estimation-Theory/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Interactive derivations: t-tests, chi-squared tests, power curves, NP lemma, GLRT, Bonferroni/BH correction, permutation tests, KS drift detection, sequential SPRT |
| [exercises.ipynb](exercises.ipynb) | 8 graded exercises from one-sample t-tests through sequential A/B testing and KS-based LLM drift detection |

## Learning Objectives

After completing this section, you will:

- State the formal definition of a statistical hypothesis and distinguish simple from composite hypotheses
- Define a test statistic, rejection region, and p-value and explain what each does and does not mean
- Quantify Type I error (α), Type II error (β), and power (1−β), and explain why they cannot all be minimised simultaneously
- Derive sample size requirements from desired α, β, and effect size (Cohen's d)
- Derive and apply one-sample t-test, two-sample Welch t-test, chi-squared goodness-of-fit, and one-way ANOVA
- State and prove the Neyman-Pearson lemma and identify when a UMP test exists
- Apply Wilks' theorem to construct generalized likelihood ratio tests for composite hypotheses
- Explain the multiple testing problem and apply Bonferroni, Holm, and Benjamini-Hochberg corrections
- Implement permutation tests, Wilcoxon rank tests, and KS tests for nonparametric inference
- Design a sequential A/B test using SPRT and explain why it avoids the peeking problem
- Identify statistical pitfalls in NLP benchmark comparisons and LLM evaluation leaderboards

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 The Core Question: Evidence or Noise?](#11-the-core-question-evidence-or-noise)
  - [1.2 Two Schools of Thought](#12-two-schools-of-thought)
  - [1.3 Historical Timeline](#13-historical-timeline)
  - [1.4 Why Hypothesis Testing Matters for AI](#14-why-hypothesis-testing-matters-for-ai)
- [2. The Formal Framework](#2-the-formal-framework)
  - [2.1 Hypotheses: Null and Alternative](#21-hypotheses-null-and-alternative)
  - [2.2 Test Statistics and Sampling Distributions](#22-test-statistics-and-sampling-distributions)
  - [2.3 Rejection Regions and Critical Values](#23-rejection-regions-and-critical-values)
  - [2.4 The p-Value](#24-the-p-value)
  - [2.5 Duality: Tests and Confidence Intervals](#25-duality-tests-and-confidence-intervals)
- [3. Errors, Power, and Sample Size](#3-errors-power-and-sample-size)
  - [3.1 Type I and Type II Errors](#31-type-i-and-type-ii-errors)
  - [3.2 The Power Function](#32-the-power-function)
  - [3.3 Effect Size](#33-effect-size)
  - [3.4 Sample Size Calculation](#34-sample-size-calculation)
  - [3.5 ROC Analogy](#35-roc-analogy)
- [4. Classical Parametric Tests](#4-classical-parametric-tests)
  - [4.1 The Z-Test](#41-the-z-test)
  - [4.2 Student's t-Test](#42-students-t-test)
  - [4.3 The Chi-Squared Test](#43-the-chi-squared-test)
  - [4.4 The F-Test and ANOVA](#44-the-f-test-and-anova)
  - [4.5 Which Test When](#45-which-test-when)
- [5. Likelihood Ratio Tests and UMP Tests](#5-likelihood-ratio-tests-and-ump-tests)
  - [5.1 The Neyman-Pearson Lemma](#51-the-neyman-pearson-lemma)
  - [5.2 Uniformly Most Powerful Tests](#52-uniformly-most-powerful-tests)
  - [5.3 The Generalized Likelihood Ratio Test](#53-the-generalized-likelihood-ratio-test)
  - [5.4 Score and Wald Tests](#54-score-and-wald-tests)
- [6. Multiple Testing](#6-multiple-testing)
  - [6.1 The Multiple Testing Problem](#61-the-multiple-testing-problem)
  - [6.2 Bonferroni and Holm Corrections](#62-bonferroni-and-holm-corrections)
  - [6.3 False Discovery Rate](#63-false-discovery-rate)
  - [6.4 NLP Benchmark Comparisons](#64-nlp-benchmark-comparisons)
  - [6.5 Bayesian Alternative Preview](#65-bayesian-alternative-preview)
- [7. Nonparametric Tests](#7-nonparametric-tests)
  - [7.1 Why Nonparametric?](#71-why-nonparametric)
  - [7.2 Permutation and Randomization Tests](#72-permutation-and-randomization-tests)
  - [7.3 Rank-Based Tests](#73-rank-based-tests)
  - [7.4 Kolmogorov-Smirnov Test](#74-kolmogorov-smirnov-test)
  - [7.5 Bootstrap Hypothesis Tests](#75-bootstrap-hypothesis-tests)
- [8. A/B Testing and ML Evaluation](#8-ab-testing-and-ml-evaluation)
  - [8.1 The A/B Testing Framework](#81-the-ab-testing-framework)
  - [8.2 Sequential A/B Testing](#82-sequential-ab-testing)
  - [8.3 Model Comparison Tests](#83-model-comparison-tests)
  - [8.4 Data Drift Detection](#84-data-drift-detection)
  - [8.5 LLM Evaluation and Leaderboards](#85-llm-evaluation-and-leaderboards)
- [9. Common Mistakes](#9-common-mistakes)
- [10. Exercises](#10-exercises)
- [11. Why This Matters for AI (2026 Perspective)](#11-why-this-matters-for-ai-2026-perspective)
- [12. Conceptual Bridge](#12-conceptual-bridge)

---

## 1. Intuition

### 1.1 The Core Question: Evidence or Noise?

Imagine you flip a coin 100 times and observe 63 heads. Is the coin biased, or is 63 just a chance fluctuation from a fair coin? You cannot answer this by staring at the number 63. You need a framework that asks: **how often would a fair coin produce 63 or more heads in 100 flips?** If the answer is "1 in 1,000 times", you have strong evidence for bias. If the answer is "1 in 5 times", the result is easily explained by chance.

This is the essence of hypothesis testing: quantify how surprising the data would be if the "nothing interesting happened" explanation were true. The "nothing interesting happened" explanation is the **null hypothesis** $H_0$. The alternative — something systematic is going on — is the **alternative hypothesis** $H_1$.

**The court-room analogy** is exact and instructive. In criminal law, the null hypothesis is innocence ($H_0$: defendant is innocent). The prosecution must present evidence so overwhelming that innocence becomes implausible. The defendant is never "proven innocent" — the court simply fails to accumulate enough evidence to reject $H_0$. Similarly, in statistics, we never "prove" the null hypothesis true; we can only fail to reject it. The asymmetry is deliberate: falsely convicting an innocent person (Type I error) is considered worse than failing to convict a guilty one (Type II error), so we set the bar for conviction (rejection) very high.

**For AI:** Every time you report "model A achieves 87.3% accuracy vs. model B's 86.1% — a statistically significant improvement at p < 0.05", you are running a hypothesis test. The null hypothesis is $H_0: \mu_A = \mu_B$ (no real difference). The question is whether the 1.2% gap is real signal or sampling noise from a finite test set.

### 1.2 Two Schools of Thought

Modern statistical testing is a marriage of two incompatible philosophies that practitioners blend without always realising it.

**Fisher's approach (1925):** Compute the **p-value** — the probability of observing data at least as extreme as what was obtained, under $H_0$. Report it as a continuous measure of evidence against $H_0$. Never pre-specify $H_1$. Never pre-specify a decision threshold. The p-value is just one piece of evidence to weigh alongside domain knowledge and replication. Fisher rejected the idea of a fixed significance threshold as "absurdly academic".

**Neyman-Pearson approach (1933):** Pre-specify both $H_0$ and $H_1$, a significance level $\alpha$ (Type I error rate), and a desired power $1 - \beta$ (sensitivity to $H_1$). Compute the **most powerful test** for those hypotheses. Make a binary decision: reject or do not reject. The p-value is irrelevant — what matters is whether $T > c_\alpha$. This framework optimises long-run decision quality across many repeated experiments.

**What practitioners actually do:** Use the Neyman-Pearson machinery (pre-specify $\alpha$, compute a test statistic, check whether $p < \alpha$) while interpreting the p-value in Fisher's spirit (as a continuous measure of evidence). This hybrid is coherent enough for most purposes but creates confusions — particularly the widespread misinterpretation of p-values as "the probability that $H_0$ is true" (which is Bayesian thinking, belonging to neither school).

```
FISHER vs. NEYMAN-PEARSON COMPARISON
════════════════════════════════════════════════════════════════════════

  Property           │ Fisher                  │ Neyman-Pearson
  ───────────────────┼─────────────────────────┼──────────────────────
  Goal               │ Measure evidence        │ Make optimal decision
  Pre-specify H₁?    │ No                      │ Yes
  Pre-specify α?     │ No                      │ Yes (before seeing data)
  Output             │ p-value (continuous)    │ Reject / Do not reject
  Power              │ Not part of framework   │ Central to design
  Philosophical base │ Inductive reasoning     │ Long-run frequency
  Use case           │ Exploratory science     │ Industrial quality control

════════════════════════════════════════════════════════════════════════
```

### 1.3 Historical Timeline

| Year | Contributor | Contribution |
|------|-------------|--------------|
| 1710 | John Arbuthnot | First known significance test (sex ratio at birth) |
| 1900 | Karl Pearson | Chi-squared goodness-of-fit test |
| 1908 | William Gosset ("Student") | t-distribution for small samples |
| 1922 | Ronald Fisher | Formalises likelihood, degrees of freedom |
| 1925 | Ronald Fisher | *Statistical Methods for Research Workers* — p-values, F-test, ANOVA |
| 1933 | Neyman & Pearson | Power, UMP tests, Neyman-Pearson lemma |
| 1943 | Abraham Wald | Sequential probability ratio test (SPRT) |
| 1951 | Wald | Statistical decision theory |
| 1979 | Bonferroni correction | Widely adopted for multiple testing |
| 1995 | Benjamini & Hochberg | False discovery rate (FDR) — transformative for genomics and ML |
| 2005 | Ioannidis | "Why Most Published Research Findings Are False" — catalyses replication crisis |
| 2016 | ASA statement | Formal warning against p-value misuse |
| 2019 | *Nature* editorial | 800+ scientists call for retiring "statistical significance" |
| 2022+ | Always-valid p-values | Ramdaset al. — sequential testing for online A/B experiments |

### 1.4 Why Hypothesis Testing Matters for AI

**Model evaluation:** Reporting a test accuracy without a confidence interval or significance test is meaningless for model comparison. Is 87.3% vs. 86.1% real? On 1000 test examples at 0.5 base rate, a 1.2% difference corresponds to a two-proportion z-test with $p \approx 0.15$ — not significant. On 10,000 examples, the same gap gives $p \approx 0.002$. Sample size is everything.

**A/B testing at scale:** Tech companies run thousands of simultaneous A/B experiments. Each one is a two-sample hypothesis test. The infrastructure problem is: how do you test without pre-committing to a fixed sample size (you want to stop early if the effect is clear), while controlling false discovery rate across simultaneous tests?

**Data drift detection:** Production ML systems degrade when the input distribution changes. Detecting this is a two-sample test: is the distribution of today's features statistically different from training data? The Kolmogorov-Smirnov test, Maximum Mean Discrepancy, and Population Stability Index are all hypothesis tests under the hood.

**LLM benchmark evaluation:** The 2024-2026 era of LLM leaderboards (MMLU, HumanEval, BIG-Bench, LMSYS Arena) suffers from massive multiple-comparison inflation. If you test 100 models on 50 benchmarks, you expect 250 false discoveries at $\alpha = 0.05$ even if no model truly differs. Proper evaluation requires FDR correction and bootstrap confidence intervals.

**Causal inference for RLHF:** When measuring whether RLHF improves output quality, you need a randomised controlled design and a proper two-sample test. Confounded comparisons (different prompts, different raters) can produce entirely spurious "improvements".


---

## 2. The Formal Framework

### 2.1 Hypotheses: Null and Alternative

**Definition (Statistical Hypothesis).** A *statistical hypothesis* is a claim about the parameter $\theta$ of a probability model $\{P_\theta : \theta \in \Omega\}$. Formally, a hypothesis specifies a subset $\Theta_0 \subseteq \Omega$:

$$H_0: \theta \in \Theta_0 \quad \text{vs.} \quad H_1: \theta \in \Theta_1 = \Omega \setminus \Theta_0$$

**Simple vs. composite hypotheses:**
- A **simple hypothesis** pins $\theta$ to a single value: $H_0: \theta = \theta_0$.
- A **composite hypothesis** specifies a range: $H_0: \theta \leq \theta_0$ or $H_0: \theta \neq \theta_0$.

**One-sided vs. two-sided tests:**
- **One-sided** (directional): $H_1: \theta > \theta_0$ or $H_1: \theta < \theta_0$. Use when the direction of the effect is theoretically specified in advance.
- **Two-sided** (non-directional): $H_1: \theta \neq \theta_0$. Use when any deviation from $\theta_0$ is of interest, or when the direction is unknown.

**The asymmetry between $H_0$ and $H_1$:** The null hypothesis is the "default" — the claim we assume true unless data provide sufficient evidence against it. This asymmetry has important consequences:
- We control the probability of falsely rejecting $H_0$ (Type I error).
- We do **not** automatically control the probability of falsely accepting $H_0$ (Type II error) — that requires separate power analysis.
- "Fail to reject $H_0$" is NOT the same as "accept $H_0$". Absence of evidence is not evidence of absence.

**Standard examples:**

| Setting | $H_0$ | $H_1$ |
|---------|-------|-------|
| Coin fairness | $p = 0.5$ | $p \neq 0.5$ |
| New drug effectiveness | $\mu_{\text{treatment}} = \mu_{\text{control}}$ | $\mu_{\text{treatment}} > \mu_{\text{control}}$ |
| Model improvement | $\text{acc}_A = \text{acc}_B$ | $\text{acc}_A \neq \text{acc}_B$ |
| Feature distribution shift | $F_{\text{new}} = F_{\text{train}}$ | $F_{\text{new}} \neq F_{\text{train}}$ |
| Independence in contingency table | Variables independent | Variables associated |

### 2.2 Test Statistics and Sampling Distributions

**Definition (Test Statistic).** A *test statistic* $T = T(X_1, \ldots, X_n)$ is a function of the data that summarises the evidence against $H_0$. A good test statistic:
1. Has a **known distribution** under $H_0$ (enables exact p-value computation).
2. Takes **extreme values** when $H_1$ is true (enables detection).

The **sampling distribution** of $T$ under $H_0$ is central. For example:
- If $X_1, \ldots, X_n \overset{iid}{\sim} \mathcal{N}(\mu, \sigma^2)$ with $\sigma$ known, then under $H_0: \mu = \mu_0$:

$$Z = \frac{\bar{X} - \mu_0}{\sigma / \sqrt{n}} \sim \mathcal{N}(0, 1)$$

- If $\sigma$ is unknown, replacing it with the sample standard deviation $S$ introduces extra variability:

$$T = \frac{\bar{X} - \mu_0}{S / \sqrt{n}} \sim t_{n-1}$$

The shift from $\mathcal{N}(0,1)$ to $t_{n-1}$ is not a minor detail — for small $n$, the t-distribution has much heavier tails, making it much harder to reject $H_0$ unless the evidence is very strong.

**Standardisation principle:** Most test statistics are of the form:

$$T = \frac{\text{Estimator} - \text{Null value}}{\text{Standard error of estimator}}$$

This form ensures $T$ is dimensionless and has a tractable distribution under $H_0$.

### 2.3 Rejection Regions and Critical Values

**Definition (Rejection Region).** For a test of size $\alpha$, the *rejection region* $\mathcal{R}_\alpha$ is a subset of the sample space such that:

$$P_{H_0}(T \in \mathcal{R}_\alpha) = \alpha$$

For a two-sided test of a Gaussian mean with known $\sigma$:

$$\mathcal{R}_\alpha = \{z : \lvert z \rvert > z_{\alpha/2}\}$$

where $z_{\alpha/2}$ is the $(1 - \alpha/2)$ quantile of $\mathcal{N}(0, 1)$. At $\alpha = 0.05$, $z_{0.025} = 1.96$.

**Critical value:** The boundary $c_\alpha$ such that $P_{H_0}(T > c_\alpha) = \alpha$ (one-sided) or $P_{H_0}(\lvert T \rvert > c_\alpha) = \alpha$ (two-sided). The test rejects $H_0$ iff $T > c_\alpha$ (or $\lvert T \rvert > c_\alpha$).

**Exact vs. approximate rejection regions:** 
- For normal populations with known $\sigma$: exact (z-test).
- For normal populations with unknown $\sigma$: exact (t-test, using t distribution).
- For non-normal populations, large $n$: approximate (CLT makes $Z$ approximately standard normal).
- For small $n$, non-normal: nonparametric tests (Section 7).

### 2.4 The p-Value

**Definition (p-Value).** The *p-value* of a test with statistic $T = t_{\text{obs}}$ is:

$$p = P_{H_0}(T \geq t_{\text{obs}})$$

for a one-sided test, or

$$p = P_{H_0}(\lvert T \rvert \geq \lvert t_{\text{obs}} \rvert)$$

for a two-sided test. Equivalently, $p$ is the smallest significance level $\alpha$ at which the observed data would lead to rejection of $H_0$.

**Key properties of p-values:**

1. **Under $H_0$, $p \sim \mathcal{U}(0,1)$.** This is a fundamental result: if $H_0$ is true and the test is exact, the p-value is uniformly distributed. This enables calibration checks.
2. **Under $H_1$, $p$ is stochastically smaller** — it tends toward 0 as sample size grows or effect size increases.
3. **$p$ is a random variable.** Running the same experiment twice will give different p-values. The p-value quantifies how surprising the specific data are, not how true or false $H_0$ is.

**The six most important p-value misinterpretations:**

| Misinterpretation | Why it's wrong | Correct statement |
|---|---|---|
| "$p$ = probability $H_0$ is true" | Frequentist $p$ makes no probability claim about $H_0$ | $p$ = prob of data this extreme under $H_0$ |
| "$1-p$ = probability $H_1$ is true" | Same error | Not a probability about hypotheses |
| "$p < 0.05$ means effect is large" | $p$ conflates effect size with sample size | Report effect size separately |
| "$p > 0.05$ means no effect" | Absence of evidence ≠ evidence of absence | Report power and CI |
| "$p < 0.05$ means the finding replicates" | Single-study p is unreliable | Need replication studies |
| "We found $p = 0.049$, thus significant" | Arbitrary threshold; $p = 0.051$ is equally evidential | Report exact $p$; don't dichotomize |

### 2.5 Duality: Tests and Confidence Intervals

There is an exact correspondence between hypothesis tests and confidence intervals — a fact that is both theoretically beautiful and practically useful.

**The Inversion Principle:** Given a size-$\alpha$ test for $H_0: \theta = \theta_0$, the $(1-\alpha)$ confidence interval for $\theta$ is:

$$\text{CI}_{1-\alpha} = \{\theta_0 : H_0 \text{ is not rejected by the size-}\alpha \text{ test}\}$$

Conversely, the size-$\alpha$ test rejects $H_0: \theta = \theta_0$ if and only if $\theta_0 \notin \text{CI}_{1-\alpha}$.

**Concrete example:** The 95% CI for a Gaussian mean with known $\sigma$ is:

$$\text{CI}_{0.95} = \left[\bar{X} - 1.96 \frac{\sigma}{\sqrt{n}},\; \bar{X} + 1.96 \frac{\sigma}{\sqrt{n}}\right]$$

The corresponding z-test rejects $H_0: \mu = \mu_0$ at $\alpha = 0.05$ iff $\mu_0$ falls outside this interval — exactly the inversion principle.

> **Recall:** Confidence intervals were derived in [§02 Estimation Theory](../02-Estimation-Theory/notes.md#7-confidence-intervals). The CI for $\mu$ was constructed by pivoting on the standard normal. Here, we see that same CI is the set of null values we would fail to reject.

**Practical implication:** Reporting a CI is strictly more informative than reporting a p-value. The CI tells you the effect size and uncertainty; the p-value alone only tells you whether a point null is rejected. Always prefer CIs over p-values where possible.


---

## 3. Errors, Power, and Sample Size

### 3.1 Type I and Type II Errors

Any binary decision procedure applied to random data will sometimes make mistakes. There are exactly two ways to err:

**Definition (Type I Error).** Rejecting $H_0$ when $H_0$ is true. Also called a **false positive**. Probability = $\alpha$ (the significance level).

**Definition (Type II Error).** Failing to reject $H_0$ when $H_1$ is true. Also called a **false negative**. Probability = $\beta$ (depends on the specific alternative $\theta \in \Theta_1$).

```
ERROR TYPE TABLE
════════════════════════════════════════════════════════════════════════

                      │  H₀ True             │  H₁ True
  ────────────────────┼──────────────────────┼──────────────────────
  Reject H₀           │  Type I Error (α)    │  CORRECT (Power 1-β)
  Do not reject H₀    │  CORRECT (1-α)       │  Type II Error (β)

  Analogy:            │  Convict innocent    │  Free the guilty
  Medical test:       │  False positive      │  False negative

════════════════════════════════════════════════════════════════════════
```

**The fundamental trade-off:** For a fixed sample size $n$, decreasing $\alpha$ (requiring stronger evidence to reject) increases $\beta$ (making it harder to detect real effects). To decrease both simultaneously, you must increase $n$.

**Conventional thresholds (and their limitations):**
- $\alpha = 0.05$: Fisher's suggestion from 1925, now a near-universal convention despite having no theoretical justification.
- $\alpha = 0.01$: More stringent; used in physics and genomics.
- $\alpha = 0.005$: Proposed by Benjamin et al. (2018) as a new standard to reduce false discoveries.
- **For AI deployment decisions:** The appropriate $\alpha$ depends on the cost of each error type. Deploying a harmful model is a Type II error; rejecting a good model is a Type I error. These costs are application-specific.

### 3.2 The Power Function

**Definition (Power Function).** The *power function* of a test is:

$$\pi(\theta) = P_\theta(\text{reject } H_0) = P_\theta(T \in \mathcal{R}_\alpha)$$

evaluated at every $\theta \in \Omega$.

Key properties of a well-designed power function:
- $\pi(\theta) = \alpha$ for $\theta \in \Theta_0$ (the test has correct size).
- $\pi(\theta) \to 1$ as $\theta$ moves far from $\Theta_0$ (the test is consistent).
- $\pi(\theta)$ is large for $\theta \in \Theta_1$ of practical interest (the test is powerful).

**Power at a specific alternative:** For the one-sample z-test with $H_0: \mu = \mu_0$ vs. $H_1: \mu = \mu_1 > \mu_0$:

$$\pi(\mu_1) = P_{\mu_1}\left(Z > z_\alpha\right) = P\left(\mathcal{N}(0,1) > z_\alpha - \frac{(\mu_1 - \mu_0)\sqrt{n}}{\sigma}\right) = 1 - \Phi\left(z_\alpha - \frac{(\mu_1 - \mu_0)\sqrt{n}}{\sigma}\right)$$

This formula reveals exactly how power depends on:
- **Effect size** $\delta = (\mu_1 - \mu_0)/\sigma$: larger effect → higher power.
- **Sample size** $n$: larger $n$ → higher power (as $\sqrt{n}$).
- **Significance level** $\alpha$: larger $\alpha$ → higher power (but more Type I errors).

**Minimum detectable effect (MDE):** The smallest $\delta$ such that $\pi(\mu_0 + \delta\sigma) \geq 1 - \beta_{\text{target}}$. Solving for $\delta$:

$$\delta_{\min} = \frac{(z_\alpha + z_\beta)\sigma}{\sqrt{n}}$$

where $z_\beta = \Phi^{-1}(1-\beta)$. At $\alpha = 0.05$, $\beta = 0.20$: $z_{0.05} + z_{0.20} = 1.645 + 0.842 = 2.487$.

### 3.3 Effect Size

**The problem with raw differences:** A mean difference of 2 points on an exam is huge if the standard deviation is 1, but negligible if it is 100. Effect sizes standardise the comparison.

**Cohen's d** (for means):
$$d = \frac{\mu_1 - \mu_2}{\sigma_{\text{pooled}}} \quad \text{where} \quad \sigma_{\text{pooled}} = \sqrt{\frac{(n_1-1)\sigma_1^2 + (n_2-1)\sigma_2^2}{n_1+n_2-2}}$$

Benchmarks (Cohen 1988): $d = 0.2$ small, $d = 0.5$ medium, $d = 0.8$ large.

**Cohen's h** (for proportions):
$$h = 2\arcsin\!\sqrt{p_1} - 2\arcsin\!\sqrt{p_2}$$

The arcsin transform stabilises variance. Benchmarks: $h = 0.2$ small, $h = 0.5$ medium, $h = 0.8$ large.

**Cramér's V** (for contingency tables with $\chi^2$):
$$V = \sqrt{\frac{\chi^2}{n \cdot \min(r-1, c-1)}}$$

where $r, c$ are the numbers of rows and columns. $V \in [0, 1]$ with 0 = no association.

**For AI:** When comparing two models' accuracies, report Cohen's h for proportions. A 1% absolute accuracy gain with $h = 0.03$ is small and may not justify the deployment cost; with $h = 0.20$ it is meaningful. Effect size is always reported alongside p-value in rigorous ML papers.

### 3.4 Sample Size Calculation

Solving the power equation for $n$ gives the required sample size to detect effect size $\delta$ with power $1-\beta$ at level $\alpha$:

$$n = \left(\frac{z_\alpha + z_\beta}{\delta}\right)^2$$

(one-sample, one-sided). For two-sided tests replace $z_\alpha$ with $z_{\alpha/2}$.

**Two-sample comparison of means** (equal group sizes):

$$n_{\text{per group}} = \frac{2(z_{\alpha/2} + z_\beta)^2}{\delta^2}$$

where $\delta = (\mu_1 - \mu_2)/\sigma$.

**Two-proportion z-test** (comparing accuracy rates $p_1$ vs. $p_2$):

$$n = \frac{(z_{\alpha/2}\sqrt{2\bar{p}(1-\bar{p})} + z_\beta\sqrt{p_1(1-p_1) + p_2(1-p_2)})^2}{(p_1 - p_2)^2}$$

where $\bar{p} = (p_1+p_2)/2$.

**Worked example:** You want to detect a 2% accuracy improvement (from 85% to 87%) with 80% power at $\alpha = 0.05$.
- $\bar{p} = 0.86$, $z_{0.025} = 1.96$, $z_{0.20} = 0.842$.
- $n \approx \frac{(1.96\sqrt{2(0.86)(0.14)} + 0.842\sqrt{0.85(0.15)+0.87(0.13)})^2}{(0.02)^2} \approx 3{,}600$ per group.

This reveals why benchmark comparisons on small test sets are inconclusive: on 1,000 examples per group, the same test has power $\approx 30\%$.

### 3.5 ROC Analogy

The error rate trade-off in hypothesis testing is structurally identical to the ROC curve in binary classification:

| Hypothesis Testing | Binary Classification |
|---|---|
| Significance level $\alpha$ | False positive rate (FPR) |
| Power $1 - \beta$ | True positive rate (TPR) / Recall |
| Critical value $c_\alpha$ | Classification threshold |
| Type I error | False positive |
| Type II error | False negative |
| Reject region | Predicted positive region |

In both settings, you trace out a curve by varying the threshold (critical value / classification threshold), and the curve represents the complete trade-off between sensitivity and specificity. The AUC of a classifier measures the same thing as the integrated power function of a test: how well the score separates the two classes.

**For AI:** The ROC analogy makes hypothesis testing intuitive for ML practitioners. Choosing $\alpha = 0.05$ is exactly like choosing a classification threshold to achieve 5% FPR. Power analysis is like calculating recall at that threshold.


---

## 4. Classical Parametric Tests

### 4.1 The Z-Test

**Setting:** $X_1, \ldots, X_n \overset{iid}{\sim} \mathcal{N}(\mu, \sigma^2)$ with $\sigma^2$ **known**. Test $H_0: \mu = \mu_0$.

**Test statistic:**
$$Z = \frac{\bar{X} - \mu_0}{\sigma / \sqrt{n}} \overset{H_0}{\sim} \mathcal{N}(0, 1)$$

**Rejection regions and p-values:**
- Two-sided ($H_1: \mu \neq \mu_0$): reject iff $\lvert Z \rvert > z_{\alpha/2}$; $p = 2(1 - \Phi(\lvert Z \rvert))$.
- Upper-tailed ($H_1: \mu > \mu_0$): reject iff $Z > z_\alpha$; $p = 1 - \Phi(Z)$.
- Lower-tailed ($H_1: \mu < \mu_0$): reject iff $Z < -z_\alpha$; $p = \Phi(Z)$.

**Two-sample z-test for proportions:** Compare $p_1$ (proportion in group 1) vs. $p_2$ (group 2).

$$Z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}(1-\hat{p})(1/n_1 + 1/n_2)}} \overset{H_0}{\overset{\text{approx}}{\sim}} \mathcal{N}(0,1)$$

where $\hat{p} = (n_1\hat{p}_1 + n_2\hat{p}_2)/(n_1+n_2)$ is the pooled proportion. This is the standard test for A/B experiments comparing click-through rates or model accuracy.

**Validity:** Requires $n_1\hat{p}(1-\hat{p}) \geq 5$ and $n_2\hat{p}(1-\hat{p}) \geq 5$. For rare events or small samples, use Fisher's exact test.

### 4.2 Student's t-Test

The t-test is the workhorse of applied statistics: it handles the realistic case where $\sigma$ is unknown.

**One-sample t-test:** $X_1, \ldots, X_n \overset{iid}{\sim} \mathcal{N}(\mu, \sigma^2)$, $\sigma$ unknown. Test $H_0: \mu = \mu_0$.

$$T = \frac{\bar{X} - \mu_0}{S / \sqrt{n}} \overset{H_0}{\sim} t_{n-1}$$

where $S^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2$ is the sample variance. Reject at level $\alpha$ iff $\lvert T \rvert > t_{n-1, \alpha/2}$.

**Gosset's insight:** Why $t_{n-1}$ and not $\mathcal{N}(0,1)$? Because $S$ is estimated from data, not known. Substituting $S$ for $\sigma$ introduces additional randomness. The t-distribution has heavier tails to account for this extra uncertainty. As $n \to \infty$, $t_{n-1} \to \mathcal{N}(0,1)$.

**Paired t-test:** When observations come in natural pairs (before/after measurements, matched subjects), compute differences $D_i = X_i - Y_i$ and apply the one-sample t-test to $D_1, \ldots, D_n$. This removes between-pair variability and dramatically increases power.

**Two-sample Welch t-test:** Compare means from two independent groups with possibly unequal variances ($\sigma_1^2 \neq \sigma_2^2$):

$$T = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{S_1^2/n_1 + S_2^2/n_2}} \overset{H_0}{\approx} t_\nu$$

where the Welch-Satterthwaite degrees of freedom are:

$$\nu = \frac{(S_1^2/n_1 + S_2^2/n_2)^2}{(S_1^2/n_1)^2/(n_1-1) + (S_2^2/n_2)^2/(n_2-1)}$$

Always use the Welch t-test (not the pooled t-test) unless you have strong prior evidence that $\sigma_1 = \sigma_2$. The pooled t-test's assumption of equal variances is rarely justified and can inflate Type I error badly.

**Robustness:** The t-test is remarkably robust to non-normality for $n \geq 30$ by the CLT. For small $n$ with strongly skewed or heavy-tailed distributions, use nonparametric alternatives (Wilcoxon signed-rank or Mann-Whitney).

**For AI:** Use the paired t-test when comparing two models evaluated on the same test examples (paired observations). Use Welch's t-test when comparing models evaluated on different test sets.

### 4.3 The Chi-Squared Test

**Goodness-of-fit test:** Observed counts $O_1, \ldots, O_k$ from $n$ observations, expected counts $E_i = np_{i0}$ under $H_0$.

$$\chi^2 = \sum_{i=1}^k \frac{(O_i - E_i)^2}{E_i} \overset{H_0}{\approx} \chi^2_{k-1}$$

Valid when $E_i \geq 5$ for all $i$. The $\chi^2$ approximation improves with $n$.

**Test of independence:** A $r \times c$ contingency table with counts $O_{ij}$. Under $H_0$ (row and column variables independent):

$$E_{ij} = \frac{R_i C_j}{n}, \quad \chi^2 = \sum_{i=1}^r\sum_{j=1}^c \frac{(O_{ij} - E_{ij})^2}{E_{ij}} \overset{H_0}{\approx} \chi^2_{(r-1)(c-1)}$$

**Worked example:** A model is evaluated on 4 topic categories. Observed errors: [12, 8, 25, 5]. Under $H_0$ (equal error rate): $E_i = 50/4 = 12.5$ each. $\chi^2 = (12-12.5)^2/12.5 + \ldots = 16.08$, df = 3, $p = 0.001$. Strong evidence the error rate varies by topic.

**For AI:** Chi-squared tests are natural for:
- Testing whether a language model's errors are uniformly distributed across categories.
- Testing whether a tokenizer's vocabulary coverage is uniform across languages.
- Detecting systematic biases in model outputs (contingency table: output category vs. demographic group).

### 4.4 The F-Test and ANOVA

**F-test for two variances:** $H_0: \sigma_1^2 = \sigma_2^2$.

$$F = \frac{S_1^2}{S_2^2} \overset{H_0}{\sim} F_{n_1-1, n_2-1}$$

Rarely used directly; arises naturally inside ANOVA.

**One-way ANOVA:** Compare means across $k$ groups with $n_j$ observations in group $j$. Total $N = \sum n_j$.

Decompose total variance into between-group and within-group components:

$$\underbrace{\sum_{j=1}^k \sum_{i=1}^{n_j} (X_{ij} - \bar{X})^2}_{\text{SST}} = \underbrace{\sum_{j=1}^k n_j(\bar{X}_j - \bar{X})^2}_{\text{SSB}} + \underbrace{\sum_{j=1}^k \sum_{i=1}^{n_j} (X_{ij} - \bar{X}_j)^2}_{\text{SSW}}$$

**Test statistic:**
$$F = \frac{\text{SSB}/(k-1)}{\text{SSW}/(N-k)} = \frac{\text{MSB}}{\text{MSW}} \overset{H_0}{\sim} F_{k-1, N-k}$$

Reject $H_0: \mu_1 = \cdots = \mu_k$ when $F > F_{k-1, N-k, \alpha}$.

**Post-hoc tests:** A significant ANOVA F-test says "at least one mean differs" but not which ones. Post-hoc comparisons (Tukey HSD, Bonferroni-corrected t-tests) identify the specific differences while controlling FWER.

**ANOVA assumptions:** Normality within groups, equal variances (homoscedasticity), independence. Use Welch's ANOVA or Kruskal-Wallis test when homoscedasticity fails.

### 4.5 Which Test When

```
TEST SELECTION FLOWCHART
════════════════════════════════════════════════════════════════════════

  How many groups?
  ├── One group
  │   ├── Normal / large n → One-sample t-test or z-test
  │   └── Non-normal, small n → Wilcoxon signed-rank
  ├── Two groups
  │   ├── Paired observations?
  │   │   ├── Yes: Normal → Paired t-test
  │   │   └── Yes: Non-normal → Wilcoxon signed-rank
  │   └── Independent observations?
  │       ├── Normal: Welch two-sample t-test
  │       └── Non-normal: Mann-Whitney U test
  └── Three or more groups
      ├── Normal, equal variances → One-way ANOVA
      ├── Normal, unequal variances → Welch's ANOVA
      └── Non-normal → Kruskal-Wallis test

  Categorical / count data?
  ├── One sample, counts vs. expected → Chi-squared GoF
  ├── Two categorical variables → Chi-squared independence
  └── Small expected counts (< 5) → Fisher's exact test

════════════════════════════════════════════════════════════════════════
```


---

## 5. Likelihood Ratio Tests and UMP Tests

### 5.1 The Neyman-Pearson Lemma

The Neyman-Pearson lemma answers a fundamental question: among all tests with size at most $\alpha$, which one maximises power at a specific alternative $\theta_1$?

**Theorem (Neyman-Pearson, 1933).** Consider testing $H_0: \theta = \theta_0$ vs. $H_1: \theta = \theta_1$ (both simple). The most powerful size-$\alpha$ test rejects $H_0$ when:

$$\Lambda(\mathbf{x}) = \frac{p(\mathbf{x} \mid \theta_1)}{p(\mathbf{x} \mid \theta_0)} > k_\alpha$$

where the constant $k_\alpha$ is chosen so that $P_{\theta_0}(\Lambda > k_\alpha) = \alpha$.

**Proof sketch:** Let $\phi^*$ be the likelihood ratio test and $\phi$ any other test with $\mathbb{E}_{\theta_0}[\phi] \leq \alpha$. We want to show $\mathbb{E}_{\theta_1}[\phi^*] \geq \mathbb{E}_{\theta_1}[\phi]$.

By construction of $\phi^*$: $(\phi^*(\mathbf{x}) - \phi(\mathbf{x}))(p(\mathbf{x}|\theta_1) - k_\alpha p(\mathbf{x}|\theta_0)) \geq 0$ for all $\mathbf{x}$ (both factors have the same sign). Therefore:

$$\mathbb{E}_{\theta_1}[\phi^*] - \mathbb{E}_{\theta_1}[\phi] \geq k_\alpha(\mathbb{E}_{\theta_0}[\phi^*] - \mathbb{E}_{\theta_0}[\phi]) \geq 0$$

since $\mathbb{E}_{\theta_0}[\phi^*] = \alpha \geq \mathbb{E}_{\theta_0}[\phi]$. $\square$

**Intuition:** The likelihood ratio $\Lambda(\mathbf{x})$ ranks data points by how much more likely they are under $H_1$ than $H_0$. Including the most $H_1$-likely data points in the rejection region maximises power. No other region of the same size can do better.

**Example — Gaussian mean:** Testing $H_0: \mu = 0$ vs. $H_1: \mu = 1$ with known $\sigma = 1$, $n$ observations.

$$\Lambda(\mathbf{x}) = \frac{\prod \mathcal{N}(x_i; 1, 1)}{\prod \mathcal{N}(x_i; 0, 1)} = \exp\!\left(\sum x_i - \frac{n}{2}\right)$$

Rejecting when $\Lambda > k$ is equivalent to rejecting when $\bar{X} > c$ for some threshold $c$. The NP lemma tells us the z-test is the most powerful test for this specific $H_1$.

### 5.2 Uniformly Most Powerful Tests

The NP lemma gives the most powerful test against a single specific alternative. Can we find a test that is simultaneously most powerful against all alternatives in $\Theta_1$?

**Definition (UMP Test).** A size-$\alpha$ test $\phi^*$ is *uniformly most powerful* (UMP) if for every other size-$\alpha$ test $\phi$ and every $\theta \in \Theta_1$:
$$\pi_{\phi^*}(\theta) \geq \pi_\phi(\theta)$$

**Monotone Likelihood Ratio (MLR):** A family $\{p(\mathbf{x}|\theta)\}$ has the MLR property in statistic $T(\mathbf{x})$ if for $\theta_1 > \theta_2$, the ratio $p(\mathbf{x}|\theta_1)/p(\mathbf{x}|\theta_2)$ is a non-decreasing function of $T(\mathbf{x})$.

**Theorem (Karlin-Rubin).** If the family has MLR in $T$, then for $H_0: \theta \leq \theta_0$ vs. $H_1: \theta > \theta_0$, the test that rejects when $T > c_\alpha$ is UMP.

**Exponential families have MLR:** The natural exponential family $p(\mathbf{x}|\eta) \propto \exp(\eta T(\mathbf{x}) - A(\eta))$ has MLR in the sufficient statistic $T(\mathbf{x})$. This means UMP tests exist for one-sided hypotheses about natural parameters of: Gaussian (mean), Bernoulli (logit), Poisson (log-rate), Exponential (rate), Gamma.

**When UMP tests do NOT exist:** For two-sided alternatives $H_1: \theta \neq \theta_0$, UMP tests generally do not exist. The best we can do is a UMP *unbiased* test (UMPU), which has power $\geq \alpha$ everywhere in $\Theta_1$.

### 5.3 The Generalized Likelihood Ratio Test

For composite hypotheses involving multiple parameters, the Neyman-Pearson approach does not directly apply. The GLRT provides a general-purpose solution.

**Definition (GLRT).** The *generalised likelihood ratio* is:

$$\Lambda(\mathbf{x}) = \frac{\sup_{\theta \in \Theta_0} \mathcal{L}(\theta)}{\sup_{\theta \in \Omega} \mathcal{L}(\theta)} = \frac{\mathcal{L}(\hat{\theta}_0)}{\mathcal{L}(\hat{\theta}_{\text{MLE}})}$$

where $\hat{\theta}_0$ is the restricted MLE (constrained to $\Theta_0$) and $\hat{\theta}_{\text{MLE}}$ is the unrestricted MLE.

Note $0 \leq \Lambda \leq 1$. Small $\Lambda$ means the constrained model fits much worse than the unconstrained model — evidence against $H_0$.

**Wilks' Theorem.** Under $H_0$ and regularity conditions, as $n \to \infty$:

$$-2\log\Lambda(\mathbf{x}) \overset{d}{\to} \chi^2_k$$

where $k = \dim(\Omega) - \dim(\Theta_0)$ is the number of constraints imposed by $H_0$.

**Proof idea:** Taylor-expand $\log \mathcal{L}(\hat{\theta}_0)$ around $\hat{\theta}_{\text{MLE}}$. The second-order term yields a quadratic form in $(\hat{\theta}_0 - \hat{\theta}_{\text{MLE}})$ scaled by the Fisher information. By asymptotic normality of MLE (§02), this quadratic form is $\chi^2_k$. $\square$

**Example:** Testing $H_0: \mu = 0, \sigma^2 = 1$ in a Gaussian model — 2 constraints, so $-2\log\Lambda \sim \chi^2_2$.

**For AI:** Wilks' theorem underlies model comparison via likelihood. Any time you compare a restricted neural architecture (fewer parameters) to a full model using their log-likelihoods, you are implicitly using a GLRT. The $\chi^2$ approximation provides a principled p-value.

### 5.4 Score and Wald Tests

The GLRT requires fitting both the restricted and unrestricted models. Two alternatives — the score test and the Wald test — each require fitting only one model. Together with the GLRT, they form the **trinity of asymptotic tests**, all asymptotically equivalent under $H_0$ and local alternatives.

**Wald Test:** Fit the unrestricted MLE $\hat{\theta}$ and check if it is far from $\Theta_0$.

$$W = (\hat{\theta} - \theta_0)^\top \hat{I}(\hat{\theta})(\hat{\theta} - \theta_0) \overset{H_0}{\to} \chi^2_k$$

where $\hat{I}(\hat{\theta})$ is the observed Fisher information at the MLE. For scalar $\theta$: $W = (\hat{\theta} - \theta_0)^2 / \widehat{\operatorname{Var}}(\hat{\theta})$, which is the square of a z-score.

**Score (Rao) Test:** Fit only the restricted MLE $\hat{\theta}_0$ and check if the score function (gradient of log-likelihood) is far from zero there.

$$S = \mathbf{s}(\hat{\theta}_0)^\top I(\hat{\theta}_0)^{-1} \mathbf{s}(\hat{\theta}_0) \overset{H_0}{\to} \chi^2_k$$

where $\mathbf{s}(\theta) = \nabla_\theta \log \mathcal{L}(\theta)$ is the score. Under $H_0$, the score should be near zero; a large score indicates the null constraint is straining the model.

```
TRINITY OF ASYMPTOTIC TESTS
════════════════════════════════════════════════════════════════════════

  Test       │ Fits model under │ Statistic              │ Geometric intuition
  ───────────┼──────────────────┼────────────────────────┼────────────────────
  Wald       │ H₁ (unrestr.)    │ Distance from θ̂ to Θ₀  │ How far is MLE from H₀?
  Score      │ H₀ (restr.)      │ Gradient at θ̂₀         │ Is restricted fit stable?
  LRT (GLRT) │ Both             │ Ratio of likelihoods    │ How much does H₀ cost?

  All three → χ²_k under H₀, with same asymptotic power under H₁

════════════════════════════════════════════════════════════════════════
```

**When they differ:** For small $n$, the three tests can give different p-values. The LRT is generally most accurate; the Wald test can be anti-conservative (over-rejects) for parameters near boundaries. The score test is preferred when fitting the unconstrained model is computationally expensive.

**For AI:** The Wald test is used to test whether individual neural network weights are significantly different from zero (a form of pruning criterion). The score test is used in online learning to detect if the current gradient is significantly non-zero (an adaptive stopping criterion).


---

## 6. Multiple Testing

### 6.1 The Multiple Testing Problem

Conduct $m$ independent hypothesis tests, each at level $\alpha$. If all $m$ null hypotheses are true, what is the probability of making at least one false rejection?

$$P(\text{at least one false positive}) = 1 - (1-\alpha)^m$$

For $m = 20$ tests at $\alpha = 0.05$: $1 - 0.95^{20} \approx 0.64$. You expect about one false discovery just by chance, even if nothing is real. This is the **multiple testing problem** — the fundamental challenge underlying the replication crisis in science and the benchmark arms race in ML.

**The error metrics:**

| Metric | Definition | Controls |
|---|---|---|
| Per-comparison error rate (PCER) | $\alpha$ per test | Nothing about joint errors |
| Family-wise error rate (FWER) | $P(\geq 1$ false rejection$)$ | Strict; few false positives |
| False discovery rate (FDR) | $\mathbb{E}[\text{FP} / \max(\text{total rejections}, 1)]$ | Balanced; allows some false positives |
| False discovery proportion (FDP) | Actual FP/total rejections | Random variable |

**m₀ and m₁:** Of $m$ tests, let $m_0$ be the number of true nulls and $m_1 = m - m_0$ be the number of true alternatives. Define $V$ = false positives, $S$ = true positives, $R = V + S$ = total rejections. Then FWER $= P(V \geq 1)$ and FDR $= \mathbb{E}[V/R]$ (with $V/R = 0$ if $R = 0$).

### 6.2 Bonferroni and Holm Corrections

**Bonferroni correction:** Test each hypothesis at level $\alpha/m$. By union bound:
$$P(V \geq 1) \leq m \cdot \frac{\alpha}{m} = \alpha$$

This guarantees FWER $\leq \alpha$ regardless of the dependency structure between tests.

**Procedure:** Compute p-values $p_1, \ldots, p_m$. Reject $H_{0i}$ if $p_i < \alpha/m$.

**Conservative when tests are positively correlated:** If tests share the same data, the union bound is loose. Bonferroni wastes power in such settings.

**Holm-Bonferroni step-down procedure (1979):**
1. Order p-values: $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$.
2. Find the smallest $j$ such that $p_{(j)} > \alpha / (m - j + 1)$.
3. Reject $H_{0,(1)}, \ldots, H_{0,(j-1)}$.

**Claim:** Holm controls FWER at level $\alpha$ and is uniformly more powerful than Bonferroni — it never rejects fewer hypotheses.

**Proof sketch:** The key step: if $H_{0,(1)}, \ldots, H_{0,(k)}$ are all true nulls (worst case for false positives), Holm's threshold for the $i$-th ordered p-value is $\alpha/(m-i+1) \geq \alpha/m$, which controls each step-wise rejection probability by the Bonferroni argument. $\square$

**Šidák correction:** For independent tests, the exact threshold is $1 - (1-\alpha)^{1/m}$ (slightly larger than $\alpha/m$, hence slightly more powerful).

### 6.3 False Discovery Rate

**The Benjamini-Hochberg (BH) procedure (1995):**

Given p-values $p_{(1)} \leq \cdots \leq p_{(m)}$ ordered from smallest to largest:

1. Find $k = \max\!\left\{i : p_{(i)} \leq \frac{i}{m} \alpha\right\}$.
2. Reject $H_{0,(1)}, \ldots, H_{0,(k)}$.

If no such $k$ exists, reject nothing.

**Theorem (Benjamini-Hochberg).** Under independence (or positive dependence, PRDS), BH controls FDR at level $\frac{m_0}{m}\alpha \leq \alpha$.

**Proof idea (Storey, 2002):** Write $\mathbb{E}[V/R] = \sum_{i \in \mathcal{H}_0} P(\text{reject } H_i \mid H_i \text{ true}) \cdot \mathbb{E}[1/R \mid \text{reject }H_i]$. Under independence and the BH threshold, each term is bounded by $\alpha/m$, giving $\mathbb{E}[V/R] \leq m_0\alpha/m \leq \alpha$. $\square$

**BH vs. Bonferroni comparison:**

| Property | Bonferroni | BH |
|---|---|---|
| Controls | FWER | FDR |
| Stringency | Very strict | Moderate |
| Power at large $m$ | Very low | Much higher |
| False positives allowed | None (probabilistically) | Some (controlled on average) |
| Best for | Few tests, each critical | Many tests, some FP acceptable |

**q-values:** For each rejected hypothesis, the q-value $q_i$ is the minimum FDR level at which $H_i$ would be rejected. Analogous to p-value for FDR control. Introduced by Storey (2002).

**For AI:** In genomics (the original motivation for BH), researchers test 20,000 gene expression differences — Bonferroni would require $p < 0.0000025$. In ML, testing 100 models across 50 benchmarks creates 5,000 comparisons — FDR control via BH is the appropriate framework.

### 6.4 NLP Benchmark Comparisons

**The leaderboard problem (2024-2026):** The major LLM leaderboards (MMLU, HellaSwag, HumanEval, GSM8K, LMSYS Arena, LiveBench) face severe multiple testing issues:

1. **Model selection bias:** Model developers report best results across many runs, architectures, and prompting strategies. This is implicit p-hacking at the model level.
2. **Benchmark contamination:** Test sets get into training data over time. Reported improvements may reflect memorisation rather than generalisation.
3. **Multiple comparisons across benchmarks:** A model scoring highest on 3 of 10 benchmarks is not necessarily best — with 100 models and 10 benchmarks, 50 false "wins" are expected by chance at $\alpha = 0.05$.
4. **Non-stationary test sets:** Rolling evaluation windows mean the effective sample size is unclear.

**Rigorous evaluation practices:**
- Report bootstrap CIs (Section 7.5) on aggregate scores.
- Apply BH correction when comparing $m$ models.
- Use held-out evaluation sets not seen during model selection.
- Report McNemar's test for paired model comparisons on the same instances.
- Require pre-registration of evaluation protocols before model training.

**Significance thresholds for benchmarks:** At $m = 100$ comparisons, BH at $\alpha = 0.05$ requires $p_{(k)} \leq 0.05k/100$. For the top-ranked model to be significantly different from the second, you typically need $n \geq 5,000$ test examples per benchmark.

### 6.5 Bayesian Alternative Preview

Classical multiple testing corrections (Bonferroni, BH) are explicitly frequentist: they control long-run error rates without asking "what is the probability that $H_i$ is true?" The Bayesian framework offers a fundamentally different approach.

> **Preview: Bayesian Model Comparison**
>
> Given observed data $\mathbf{x}$, the **Bayes factor** for $H_0$ vs. $H_1$ is:
> $$B_{01} = \frac{P(\mathbf{x} \mid H_0)}{P(\mathbf{x} \mid H_1)} = \frac{\int p(\mathbf{x}|\theta)p(\theta|H_0)d\theta}{\int p(\mathbf{x}|\theta)p(\theta|H_1)d\theta}$$
> The Bayes factor naturally accounts for model complexity (Occam's razor) and provides a direct measure of evidence. In the multiple testing setting, Bayesian methods control the posterior expected FDR by placing a prior on the proportion of true nulls $\pi_0$.
>
> → _Full treatment: [§04 Bayesian Inference](../04-Bayesian-Inference/notes.md)_


---

## 7. Nonparametric Tests

### 7.1 Why Nonparametric?

Classical tests (t, F, z) assume the data follow a specific parametric family (usually Gaussian). What if:
- The data are ordinal (rankings, Likert scales)?
- The sample size is small ($n < 15$) and normality is implausible?
- The data contain extreme outliers that violate distributional assumptions?
- You want an exact test without large-sample approximations?

**Nonparametric tests** make no (or minimal) distributional assumptions. The trade-off: they are typically less powerful than their parametric counterparts when the parametric assumptions hold, but more robust when those assumptions fail.

**Distribution-free vs. nonparametric:** A test is *distribution-free* if its null distribution is the same regardless of the data distribution. Permutation tests and rank tests are distribution-free. A test is *nonparametric* in the sense that it estimates a non-finite-dimensional quantity. The terms are often used interchangeably.

### 7.2 Permutation and Randomization Tests

**Motivation:** If $H_0: F_1 = F_2$ (two groups have the same distribution), then under $H_0$, the group labels are exchangeable. We can compute the null distribution of any test statistic exactly by enumerating all $\binom{n_1+n_2}{n_1}$ label permutations.

**Algorithm (two-sample permutation test):**
1. Compute the observed test statistic $T_{\text{obs}}$ (e.g., difference in means $\bar{X}_1 - \bar{X}_2$).
2. For $b = 1, \ldots, B$: randomly permute the combined group labels; recompute $T^{(b)}$.
3. Estimate p-value: $p = (\#\{b : T^{(b)} \geq T_{\text{obs}}\} + 1) / (B + 1)$.

**Properties:**
- **Exact** (not asymptotic) when all permutations are enumerated.
- **Valid** for any test statistic, no distributional assumptions.
- **Computationally expensive** for large samples (use $B \approx 10{,}000$ Monte Carlo permutations).
- **Any statistic:** Unlike t-tests, permutation tests work for medians, trimmed means, Gini coefficients, AUC, or custom ML metrics.

**For AI:** When comparing two LLMs on a shared benchmark, a permutation test on per-example score differences avoids all distributional assumptions. With $n = 500$ test examples, a permutation test with $B = 10{,}000$ has better calibration than a t-test.

### 7.3 Rank-Based Tests

**Mann-Whitney U test (Wilcoxon rank-sum):** Two-sample test. Combine and rank all $n_1 + n_2$ observations. Let $W$ = sum of ranks in group 1. Under $H_0$: $\mathbb{E}[W] = n_1(n_1+n_2+1)/2$.

$$U = W - \frac{n_1(n_1+1)}{2}, \quad Z = \frac{U - n_1n_2/2}{\sqrt{n_1n_2(n_1+n_2+1)/12}} \overset{H_0}{\approx} \mathcal{N}(0,1)$$

**AUC connection:** The Mann-Whitney U statistic has a beautiful probabilistic interpretation:

$$\hat{U} = \frac{1}{n_1 n_2}\sum_{i=1}^{n_1}\sum_{j=1}^{n_2} \mathbf{1}[X_i > Y_j]$$

This is exactly the **empirical AUC** — the probability that a random draw from group 1 exceeds a random draw from group 2. A Mann-Whitney test is equivalent to testing whether AUC $= 0.5$. This unifies hypothesis testing with classifier evaluation.

**Wilcoxon signed-rank test:** Paired two-sample test. Compute differences $D_i = X_i - Y_i$. Rank the $\lvert D_i \rvert$. $W^+ = $ sum of ranks of positive differences. Under $H_0: \mathbb{E}[W^+] = n(n+1)/4$.

**Kruskal-Wallis test:** Extension of Mann-Whitney to $k \geq 2$ groups. Rank all $N$ observations jointly; the test statistic is based on the between-group variability of ranks. Under $H_0$: approximately $\chi^2_{k-1}$.

### 7.4 Kolmogorov-Smirnov Test

**One-sample KS test:** Test whether a sample comes from a specified distribution $F_0$.

$$D_n = \sup_x \lvert F_n(x) - F_0(x) \rvert$$

where $F_n$ is the empirical CDF. Under $H_0$, $D_n$ has a known distribution (Kolmogorov distribution) independent of $F_0$.

**Two-sample KS test:** Test whether two samples share the same distribution.

$$D_{n,m} = \sup_x \lvert F_n(x) - G_m(x) \rvert$$

Under $H_0$: $\sqrt{\frac{nm}{n+m}} D_{n,m} \overset{d}{\to} K$ where $K$ is the Kolmogorov distribution. Reject for large $D_{n,m}$.

**Properties:**
- Sensitive to differences in **location, scale, and shape** — not just means.
- **Consistent** against all continuous alternative distributions.
- Less powerful than t-test against pure location shifts (it wastes power on shape/scale).
- CDF-based, so naturally handles multivariate data via joint ECDFs (though the asymptotic distribution changes).

**For AI — Data drift detection:** The two-sample KS test is the most widely used drift detector in production ML:

```
DRIFT DETECTION PIPELINE
════════════════════════════════════════════════════════════════════════

  Training data distribution: F_train(x)
  Production batch (daily):   F_prod(x)

  For each feature j:
    Compute D_j = sup_x |F_train(x_j) - F_prod(x_j)|
    Compute p_j = KS test p-value
    Apply BH correction across all features

  Alert if: ∃ j with q_j < 0.05 (BH-adjusted)
  Report: Which features drifted and by how much

════════════════════════════════════════════════════════════════════════
```

**Limitations:** KS tests features marginally (one at a time). For multivariate drift, use Maximum Mean Discrepancy (MMD) or domain classifier-based tests.

### 7.5 Bootstrap Hypothesis Tests

The bootstrap (Efron 1979) provides a general method for constructing null distributions without parametric assumptions. Reviewed in §02 for CIs; here we use it for testing.

**Bootstrap test for two-sample means ($H_0: \mu_1 = \mu_2$):**
1. Compute $T_{\text{obs}} = \bar{X}_1 - \bar{X}_2$.
2. Shift both samples to have equal means: $\tilde{X}_i = X_i - \bar{X}_1 + \bar{X}$, $\tilde{Y}_j = Y_j - \bar{Y} + \bar{X}$ (where $\bar{X}$ is the pooled mean). Now $H_0$ holds exactly in the shifted data.
3. Draw bootstrap samples from $\tilde{X}$ and $\tilde{Y}$, compute $T^{(b)} = \bar{X}^{(b)} - \bar{Y}^{(b)}$.
4. $p = P(T^{(b)} \geq T_{\text{obs}})$.

**Bootstrap for complex statistics:** The t-test requires normality for exact validity. Bootstrap tests work for any statistic: median differences, correlation coefficients, AUC, BLEU scores, F1 scores — anything you can compute on resampled data.

**For AI:** Bootstrap CI and tests are standard for NLP evaluation. When comparing BLEU or ROUGE scores, a paired bootstrap test (sampling test-set instances) is the gold standard, as used by Koehn (2004) and standard in MT evaluation.

---

## 8. A/B Testing and ML Evaluation

### 8.1 The A/B Testing Framework

An A/B test is a randomised controlled experiment comparing two (or more) versions of a system. The framework:

1. **Define the primary metric** (CTR, revenue per user, model accuracy).
2. **Define guardrail metrics** (latency, crash rate, user retention) that must not degrade.
3. **Pre-specify $\alpha$, $\beta$, and MDE** (minimum detectable effect) before data collection.
4. **Randomise units** (users, sessions, requests) to treatment and control.
5. **Run until the pre-specified sample size is reached** (or sequential stopping criterion is met).
6. **Analyse with the appropriate test** and report effect size + CI.

**Unit of randomisation:** The choice of randomisation unit is critical.
- **User-level:** Each user sees only one variant. Avoids within-user interference. Used for UI changes.
- **Session-level:** Users can see both variants in different sessions. Higher statistical power but potentially biased.
- **Request-level:** Each request is independently assigned. Maximum power; appropriate for stateless ML inference.

**The experimental design matters more than the test:** Even the perfect hypothesis test cannot salvage a poorly designed experiment. Survivorship bias, Novelty effects, and Simpson's paradox are design problems, not statistical ones.

### 8.2 Sequential A/B Testing

**The peeking problem:** If you check p-values daily and stop when $p < 0.05$, you have not run an $\alpha = 0.05$ test. You have run a **repeated testing procedure** with inflated Type I error. For $\alpha = 0.05$, peeking 5 times inflates the actual error rate to $\approx 0.14$; peeking indefinitely drives Type I error to 1.

**Sequential Probability Ratio Test (SPRT, Wald 1943):** A test with no fixed sample size that guarantees both $P(\text{false positive}) \leq \alpha$ and $P(\text{false negative}) \leq \beta$, while stopping as early as possible.

Given observations $x_1, x_2, \ldots$, compute the log likelihood ratio:

$$\ell_n = \log \frac{\prod_{i=1}^n p(x_i \mid H_1)}{\prod_{i=1}^n p(x_i \mid H_0)} = \sum_{i=1}^n \log \frac{p(x_i \mid H_1)}{p(x_i \mid H_0)}$$

**Decision rule:** At each step:
- If $\ell_n \geq \log(1-\beta)/\alpha$: **reject $H_0$** (accept $H_1$).
- If $\ell_n \leq \log\beta/(1-\alpha)$: **accept $H_0$**.
- Otherwise: **continue sampling**.

**Wald's bounds:** These thresholds guarantee $P(\text{false positive}) \leq \alpha$ and $P(\text{false negative}) \leq \beta$, with minimal expected sample size compared to fixed-$n$ tests.

**Mixture Sequential Ratio Test (mSPRT):** An extension by Johari et al. (2022) that uses a mixture distribution over $H_1$, producing "always-valid p-values" — p-values that can be checked at any time without inflating error rates. This is the theoretical foundation for modern continuous A/B testing platforms (Spotify, Netflix, Booking.com).

**For AI:** The standard "wait N days, then look at p-value" A/B protocol is inefficient. Sequential testing with mSPRT or anytime-valid confidence sequences allows early stopping when effects are clear, reducing the cost of failed experiments by 30-50%.

### 8.3 Model Comparison Tests

**Paired t-test on accuracy:** Compare model A and B on the same $n$ test examples. For each example $i$, record whether model A was correct ($a_i \in \{0,1\}$) and model B correct ($b_i \in \{0,1\}$). Compute differences $D_i = a_i - b_i$ and apply a one-sample t-test to $D_1, \ldots, D_n$.

**McNemar's test:** More appropriate for binary outcomes. Contingency table of (correct/incorrect) pairs:

|  | B correct | B incorrect |
|---|---|---|
| A correct | $n_{11}$ | $n_{10}$ |
| A incorrect | $n_{01}$ | $n_{00}$ |

The test statistic $\chi^2 = (n_{10} - n_{01})^2 / (n_{10} + n_{01}) \sim \chi^2_1$ under $H_0$ (models have equal accuracy). Only discordant pairs ($n_{10}$, $n_{01}$) contribute — concordant pairs carry no information about which model is better.

**Diebold-Mariano test:** For comparing two forecasters. Test $H_0: \mathbb{E}[d_t] = 0$ where $d_t = L(e_{1t}) - L(e_{2t})$ is the loss differential at time $t$. Uses a HAC-robust variance estimator to handle serial correlation in $d_t$.

### 8.4 Data Drift Detection

**Covariate shift:** The input distribution $P(X)$ changes between training and deployment, but the conditional $P(Y|X)$ remains stable. This is the most common drift type in production ML.

**Concept drift:** The relationship $P(Y|X)$ changes. Harder to detect without labels.

**Statistical tests for drift:**

| Test | Detects | Suitable for |
|---|---|---|
| KS test (per feature) | Distributional shift | Continuous features, univariate |
| Chi-squared (per feature) | Distributional shift | Categorical features |
| MMD | Multivariate shift | High-dimensional features |
| LSDD | Local shift | Detecting where distributions differ |
| PSI | Magnitude of shift | Production monitoring, tabular data |

**Population Stability Index (PSI):** A practitioner-favourite drift metric:

$$\text{PSI} = \sum_{b=1}^B (p_{\text{prod},b} - p_{\text{train},b}) \log\frac{p_{\text{prod},b}}{p_{\text{train},b}}$$

where $b$ indexes histogram bins. PSI < 0.1: no drift; 0.1–0.25: moderate drift; > 0.25: significant drift requiring retraining. Structurally equivalent to a symmetrised KL divergence.

### 8.5 LLM Evaluation and Leaderboards

**Current best practices (2025-2026) for rigorous LLM evaluation:**

1. **Bootstrap confidence intervals on aggregate scores:** Sample test instances with replacement $B = 1{,}000$ times, compute benchmark score each time. Report median ± 95% CI.

2. **McNemar's test for pairwise comparisons:** For two LLMs on the same benchmark, use McNemar's test (paired binary outcomes) rather than an unpaired proportion test.

3. **BH-corrected comparisons across benchmarks:** When reporting "Model X outperforms Model Y on $k$ benchmarks", apply BH at $\alpha = 0.05$ and report the q-values.

4. **Effect sizes, not just p-values:** Report Cohen's h (for accuracy differences), or normalised score differences, alongside p-values.

5. **Power analysis for benchmark design:** A new benchmark should be designed with sufficient items ($n$) to detect a 1% accuracy difference with 80% power. For $\alpha = 0.05$, $\beta = 0.20$, this requires $n \approx 7{,}200$ per model comparison.

6. **Chatbot Arena / ELO ratings:** LMSYS Arena uses pairwise preference data to estimate ELO ratings. The uncertainty in ELO estimates should be reported as CIs derived from bootstrap resampling of preference pairs.


---

## 9. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
|---|---------|----------------|-----|
| 1 | Interpreting p-value as $P(H_0 \text{ is true})$ | p-value is a frequency, not a posterior probability | p = P(data this extreme \| H₀ true); use Bayes factor for posterior |
| 2 | Claiming "no effect" from $p > 0.05$ | Absence of evidence ≠ evidence of absence; may be underpowered | Report power and 95% CI; use equivalence testing |
| 3 | Running many tests without correction | FWER inflates to near 1; produces spurious discoveries | Apply Bonferroni (few tests) or BH (many tests) |
| 4 | Peeking at data repeatedly and stopping at $p < 0.05$ | Actual Type I error rate far exceeds $\alpha$ | Use sequential tests (SPRT, mSPRT) or pre-register fixed $n$ |
| 5 | Confusing statistical and practical significance | Large $n$ can make trivial effects significant | Always report effect size (Cohen's d/h) alongside p-value |
| 6 | HARKing: Hypothesising After Results Known | Converts exploratory analysis to confirmatory; p-values invalid | Pre-register hypotheses; treat post-hoc analysis as exploratory |
| 7 | Using pooled t-test when variances differ | Can inflate Type I error dramatically | Default to Welch's t-test; test variance equality only if motivated |
| 8 | Applying chi-squared with small expected counts | Chi-squared approximation fails; invalid p-values | Use Fisher's exact test when any $E_{ij} < 5$ |
| 9 | Ignoring paired structure | Discards within-pair correlation; wastes power | Use paired t-test or Wilcoxon signed-rank for paired data |
| 10 | Not checking normality for small samples | t-test assumptions violated; p-values inaccurate | For $n < 30$ with skewed data, use nonparametric or bootstrap test |
| 11 | Reporting only "p < 0.05, significant" | Loses information; invites binary thinking | Report exact p, effect size, CI, and power |
| 12 | One-tailed test chosen after seeing data direction | Halves the p-value post-hoc; inflates Type I error | Pre-register test direction or use two-tailed by default |

---

## 10. Exercises

**Exercise 1 ★ — One-Sample t-Test from Scratch**

A language model's token latency (ms) is measured on 20 requests: mean = 47.3 ms, sample std = 8.1 ms. The SLA requires mean latency ≤ 45 ms.

(a) State $H_0$ and $H_1$ precisely. Is this one-sided or two-sided?
(b) Compute the t-statistic and degrees of freedom.
(c) Find the critical value at $\alpha = 0.05$.
(d) Compute the exact p-value.
(e) State your conclusion in plain English.

**Exercise 2 ★ — Chi-Squared Goodness-of-Fit**

A text classifier should distribute predictions uniformly across 5 categories. On 500 test examples, observed counts are [87, 113, 95, 102, 103].

(a) State $H_0$ and compute expected counts.
(b) Compute the $\chi^2$ statistic.
(c) Find the p-value (df = 4, $\chi^2_{4, 0.05} = 9.49$).
(d) Is the distribution significantly non-uniform at $\alpha = 0.05$?
(e) Compute Cramér's V and interpret the effect size.

**Exercise 3 ★ — Power and Sample Size**

You want to detect that model A has a higher accuracy than model B ($p_A = 0.88$, $p_B = 0.85$) with 80% power at $\alpha = 0.05$.

(a) Compute Cohen's h for this effect.
(b) Derive the required sample size per group for a two-proportion z-test.
(c) What is the power if you can only collect $n = 2{,}000$ per group?
(d) Plot the power curve as a function of $n$ (from 500 to 5000).
(e) What sample size gives 95% power?

**Exercise 4 ★★ — Neyman-Pearson Lemma for Exponential Distribution**

Let $X_1, \ldots, X_n \overset{iid}{\sim} \text{Exp}(\lambda)$. Test $H_0: \lambda = \lambda_0$ vs. $H_1: \lambda = \lambda_1 > \lambda_0$.

(a) Write the likelihood ratio $\Lambda(\mathbf{x}) = \mathcal{L}(\lambda_1)/\mathcal{L}(\lambda_0)$.
(b) Show that rejecting when $\Lambda > k$ is equivalent to rejecting when $\bar{X} < c$ for some $c$.
(c) Find $c$ in terms of $\alpha$, $\lambda_0$, and $n$ using the fact that $2\lambda_0 n\bar{X} \sim \chi^2_{2n}$.
(d) Verify that this test has the correct size $\alpha = 0.05$ for $\lambda_0 = 1$, $n = 10$.
(e) Is this test UMP for all $\lambda_1 > \lambda_0$? Justify using the MLR property.

**Exercise 5 ★★ — Multiple Testing Correction**

In an NLP evaluation, 50 hypothesis tests are conducted (comparing a new model to baseline on 50 benchmarks). The raw p-values are generated synthetically.

(a) Simulate 45 true nulls (p-values ~ Uniform[0,1]) and 5 true alternatives (p-values ~ Beta(0.2, 1)).
(b) Count discoveries with no correction at $\alpha = 0.05$.
(c) Apply Bonferroni correction and count discoveries.
(d) Apply BH correction and count discoveries.
(e) Across 1000 simulation replications, estimate the empirical FWER and FDR for each method. Plot the results.

**Exercise 6 ★★ — Permutation Test for Two-Sample Means**

Two LLMs (A and B) are evaluated on 30 shared test prompts. Model A scores: drawn from $\mathcal{N}(0.72, 0.1^2)$. Model B scores: drawn from $\mathcal{N}(0.68, 0.1^2)$.

(a) Compute the observed mean difference.
(b) Implement a permutation test with $B = 10{,}000$ permutations.
(c) Compute the permutation p-value.
(d) Compare to a Welch t-test p-value on the same data.
(e) Repeat 500 times and compare the empirical Type I error rates of both tests under $H_0$.

**Exercise 7 ★★★ — Sequential A/B Test with SPRT**

Two model variants A and B are tested on streaming requests. $H_0: p_A = p_B = 0.80$ vs. $H_1: p_A = 0.85, p_B = 0.80$. Set $\alpha = 0.05$, $\beta = 0.20$.

(a) Derive the log-likelihood ratio $\ell_n$ for Bernoulli outcomes.
(b) Compute the Wald stopping boundaries $A = (1-\beta)/\alpha$ and $B = \beta/(1-\alpha)$.
(c) Simulate the sequential process until stopping or $n = 2{,}000$. Plot $\ell_n$ vs. $n$ with the boundaries.
(d) Compare the expected stopping time under $H_0$ and $H_1$.
(e) Estimate the empirical Type I error rate over 1000 simulated experiments where $H_0$ is true. Verify it is $\leq \alpha = 0.05$.

**Exercise 8 ★★★ — KS-Based Data Drift Detector for LLM Features**

Build a drift detection system for an LLM serving system. Reference distribution: sentence embedding norms $\sim \mathcal{N}(10, 1^2)$. Production batches vary.

(a) Simulate a reference dataset of 5,000 embeddings and three daily production batches: no drift, moderate drift ($\mu = 10.5$), severe drift ($\mu = 12$).
(b) Apply the two-sample KS test to each batch vs. reference.
(c) Apply BH correction across the 3 batch comparisons.
(d) Implement a sliding window detector: alert if last 3 consecutive days all have KS $p < 0.1$.
(e) Compare KS vs. a t-test drift detector: which is more sensitive to scale changes? Demonstrate with a scenario where the mean is unchanged but $\sigma = 1.5$.

---

## 11. Why This Matters for AI (2026 Perspective)

| Concept | AI / LLM Application | Impact |
|---|---|---|
| p-values and significance | Model comparison on benchmark leaderboards | Prevents claiming spurious improvements; requires $n \geq 5{,}000$ per comparison |
| Power analysis | Benchmark design; A/B experiment sizing | Determines minimum test set size to detect meaningful improvements |
| Type I / II error trade-off | Deployment gates (safety vs. capability) | Conservative α (0.01) for safety tests; liberal α (0.1) for early exploration |
| Multiple testing correction | Simultaneous evaluation across benchmarks | BH correction required when testing ≥ 10 benchmarks |
| Welch t-test | Comparing model variants on different test sets | Default for unpaired, unequal-variance model comparisons |
| McNemar's test | Paired model comparison on shared test examples | Most powerful paired comparison for binary accuracy |
| GLRT / Wilks' theorem | Comparing nested model architectures by NLL | $\chi^2_k$ test on difference in log-likelihoods; model selection |
| Wald test | Pruning significance of neural network weights | Test if weight significantly differs from zero before pruning |
| BH FDR correction | Multi-benchmark leaderboards (MMLU, HumanEval, etc.) | Controls false discovery rate across hundreds of simultaneous comparisons |
| Permutation test | LLM evaluation on custom metrics (BLEU, ROUGE, win rate) | Exact calibration without distributional assumptions |
| KS test | Production ML monitoring; data drift detection | Feature-level drift detection; triggers retraining pipeline |
| SPRT / mSPRT | Online A/B testing at scale (Spotify, Netflix, deployment) | Reduces experiment duration by 30-50% vs. fixed-n tests |
| Sequential testing | LLM RLHF reward model evaluation | Valid early stopping during human preference collection |
| PSI (Population Stability Index) | Model monitoring dashboards | Industry-standard drift metric for tabular features |
| Bootstrap hypothesis tests | Evaluation with small test sets | Valid inference without normality; standard in MT evaluation |

---

## 12. Conceptual Bridge

### Looking Back: Estimation Theory (§02)

Hypothesis testing builds directly on estimation theory (§02). The estimators derived there — sample mean $\bar{X}$, MLE $\hat{\theta}$, sample variance $S^2$ — reappear as the building blocks of every test statistic. The confidence interval duality (Section 2.5) makes this connection explicit: a confidence interval *is* the set of parameter values we would fail to reject, and the test *is* an inversion of the CI procedure.

The asymptotic normality of MLE (§02 §8) is the theoretical engine behind the Wald test and the asymptotic validity of the z-test for large samples. Fisher information (§02 §4) enters hypothesis testing through the score test and through the Cramér-Rao bound's role in characterising optimal tests.

Confidence intervals (§02 §7) and hypothesis tests are dual constructions: every confidence interval corresponds to a test, and every test corresponds to a confidence interval. Reporting CIs is strictly more informative, because CIs communicate effect size and precision, not just a binary reject/don't-reject decision.

### Looking Forward: Bayesian Inference (§04)

Section §04 provides the Bayesian counterpart to every major concept in this section:

| Frequentist (§03) | Bayesian (§04) |
|---|---|
| p-value | Posterior probability $P(H_0 \mid \mathbf{x})$ |
| Significance test | Bayes factor $B_{01}$ |
| Confidence interval | Credible interval |
| FWER / FDR control | Prior on proportion of true nulls $\pi_0$ |
| Point null $H_0: \theta = \theta_0$ | Spike-and-slab prior centred at $\theta_0$ |

The philosophical divide is deep: frequentists refuse to assign probabilities to hypotheses (hypotheses are fixed; data are random). Bayesians treat parameters and hypotheses as random variables with prior distributions. The Bayesian framework provides a natural solution to multiple testing (the prior on $\pi_0$ automatically corrects for multiplicity), but requires specification of that prior — a potential source of subjectivity.

For AI practitioners, the practical choice is often dictated by computational constraints and domain norms. Frequentist tests are fast and require no prior specification; Bayesian methods provide richer inference at the cost of prior elicitation and posterior computation.

### Looking Further Forward: Regression (§06)

The F-test derived in Section 4.4 reappears in §06 as the overall F-test for regression significance. The t-test for individual regression coefficients ($H_0: \beta_j = 0$) is a direct application of the Wald test from Section 5.4. The multiple testing problem reappears when testing many coefficients simultaneously in high-dimensional regression — LASSO regularisation can be seen as an implicit multiple testing correction that shrinks small coefficients to zero.

```
POSITION IN CURRICULUM
════════════════════════════════════════════════════════════════════════

  §02 ESTIMATION THEORY
    MLE, Fisher info, CIs, asymptotic normality
           │
           ▼ (test statistics are functions of estimators)
  §03 HYPOTHESIS TESTING  ◄── YOU ARE HERE
    p-values, power, t/χ²/F tests, LRT, multiple testing,
    nonparametric tests, A/B testing, sequential tests
           │                         │
           ▼                         ▼
  §04 BAYESIAN INFERENCE    §06 REGRESSION ANALYSIS
  (Bayes factors, posterior    (F-test, t-tests on
   probability of hypotheses)   regression coefficients)
           │
           ▼
  Ch8 OPTIMISATION
  (RLHF experiment design,
   model selection, early stopping)

════════════════════════════════════════════════════════════════════════
```

Hypothesis testing is the formal language of scientific comparison. Every claim that "model A is better than model B", every statement that "this feature is significant", every assertion that "the distribution shifted" — all of these are hypothesis tests, whether or not they are recognised as such. Making these tests explicit, pre-specified, and properly corrected is the difference between rigorous science and post-hoc storytelling.

---

## Appendix A: Key Distributions in Hypothesis Testing

| Distribution | PDF / PMF | Key role in testing |
|---|---|---|
| $\mathcal{N}(0,1)$ | $(2\pi)^{-1/2}e^{-z^2/2}$ | Z-test null distribution |
| $t_{n-1}$ | $\propto (1+t^2/(n-1))^{-n/2}$ | t-test null distribution |
| $\chi^2_k$ | $\propto x^{k/2-1}e^{-x/2}$ | Chi-squared, GLRT, Wald, Score test |
| $F_{k,m}$ | $\propto x^{k/2-1}(1+kx/m)^{-(k+m)/2}$ | F-test, ANOVA |
| $\text{Kolmogorov}$ | $\sum_k (-1)^{k-1}e^{-2k^2t^2}$ | KS test null distribution |

**Relationships:**
- $Z^2 \sim \chi^2_1$
- $t_n^2 \sim F_{1,n}$ (square of t is F with df 1 in numerator)
- $\chi^2_k / k \cdot \chi^2_m / m = F_{k,m}$ (ratio of chi-squared variables / their df)
- $-2\log\Lambda \overset{d}{\to} \chi^2_k$ (Wilks' theorem)

## Appendix B: Critical Values Reference

| Test | $\alpha = 0.10$ | $\alpha = 0.05$ | $\alpha = 0.01$ |
|---|---|---|---|
| $\mathcal{N}(0,1)$ (two-sided) | $\pm 1.645$ | $\pm 1.960$ | $\pm 2.576$ |
| $t_{30}$ (two-sided) | $\pm 1.697$ | $\pm 2.042$ | $\pm 2.750$ |
| $\chi^2_1$ | 2.706 | 3.841 | 6.635 |
| $\chi^2_5$ | 9.236 | 11.070 | 15.086 |
| $\chi^2_{10}$ | 15.987 | 18.307 | 23.209 |
| $F_{1,30}$ | 2.881 | 4.171 | 7.562 |
| $F_{3,30}$ | 2.276 | 2.922 | 4.510 |

## Appendix C: Statistical Testing in Python

```python
from scipy import stats
import numpy as np

# One-sample t-test
t_stat, p_val = stats.ttest_1samp(data, popmean=mu0)

# Welch two-sample t-test
t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)

# Paired t-test
t_stat, p_val = stats.ttest_rel(before, after)

# Chi-squared goodness-of-fit
chi2, p_val = stats.chisquare(observed, expected)

# Chi-squared test of independence
chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)

# One-way ANOVA
f_stat, p_val = stats.f_oneway(group1, group2, group3)

# Mann-Whitney U test
u_stat, p_val = stats.mannwhitneyu(x, y, alternative='two-sided')

# Kolmogorov-Smirnov two-sample test
ks_stat, p_val = stats.ks_2samp(sample1, sample2)

# Wilcoxon signed-rank test
w_stat, p_val = stats.wilcoxon(differences)
```

## Appendix D: Glossary

| Term | Definition |
|---|---|
| **p-value** | $P_{H_0}(T \geq t_{\text{obs}})$; probability of data this extreme under $H_0$ |
| **Size** | $\sup_{\theta \in \Theta_0} P_\theta(\text{reject})$; actual Type I error rate |
| **Level** | Upper bound on size; test of level $\alpha$ has size $\leq \alpha$ |
| **Power** | $P_{H_1}(\text{reject})$; probability of correctly detecting the alternative |
| **Consistent test** | Power $\to 1$ as $n \to \infty$ for all $\theta \in \Theta_1$ |
| **UMP test** | Uniformly most powerful; maximises power at every $\theta \in \Theta_1$ |
| **FWER** | Family-wise error rate; probability of at least one false rejection |
| **FDR** | False discovery rate; expected proportion of false rejections among all rejections |
| **SPRT** | Sequential probability ratio test; optimal sequential test (Wald 1943) |
| **mSPRT** | Mixture SPRT; produces always-valid p-values for continuous monitoring |
| **MLR** | Monotone likelihood ratio; condition guaranteeing existence of UMP tests |


## Appendix E: Proof of the Bonferroni Inequality

**Lemma (Bonferroni).** Let $A_1, \ldots, A_m$ be events. Then:
$$P\!\left(\bigcup_{i=1}^m A_i\right) \leq \sum_{i=1}^m P(A_i)$$

**Proof:** By inclusion-exclusion and the fact that all higher-order intersection terms are non-negative:
$$P\!\left(\bigcup_{i=1}^m A_i\right) = \sum P(A_i) - \sum_{i<j} P(A_i \cap A_j) + \cdots \leq \sum_{i=1}^m P(A_i). \quad \square$$

**Application to FWER:** Let $A_i = \{\text{falsely reject } H_{0i}\}$. If each test has size $\alpha/m$, then $P(A_i) \leq \alpha/m$ and:

$$\text{FWER} = P\!\left(\bigcup_{i=1}^{m_0} A_i\right) \leq \sum_{i=1}^{m_0} P(A_i) \leq m_0 \cdot \frac{\alpha}{m} \leq m \cdot \frac{\alpha}{m} = \alpha$$

The Bonferroni correction is conservative because the inequality is tight only when the $A_i$ are mutually exclusive — which is the worst case for the union bound.

**Simes' inequality (1986):** For independent tests, the probability that any $p_{(i)} \leq i\alpha/m$ (BH threshold) is exactly $\alpha$ under the complete null. This is sharper than Bonferroni and is the basis of the BH procedure's validity proof.

---

## Appendix F: Derivation of t-Distribution

**Setup:** $X_1, \ldots, X_n \overset{iid}{\sim} \mathcal{N}(\mu, \sigma^2)$. Show that $T = (\bar{X} - \mu)/(S/\sqrt{n}) \sim t_{n-1}$.

**Step 1:** $\bar{X} \sim \mathcal{N}(\mu, \sigma^2/n)$ so $\sqrt{n}(\bar{X}-\mu)/\sigma \sim \mathcal{N}(0,1)$.

**Step 2:** $(n-1)S^2/\sigma^2 \sim \chi^2_{n-1}$ (Cochran's theorem; requires $\bar{X} \perp S^2$ which holds for Gaussian data).

**Step 3:** $\bar{X}$ and $S^2$ are independent (also Cochran).

**Step 4:** By definition of the t-distribution, if $Z \sim \mathcal{N}(0,1)$ and $V \sim \chi^2_k$ are independent, then $T = Z/\sqrt{V/k} \sim t_k$. Apply with $k = n-1$:

$$T = \frac{(\bar{X}-\mu)/(\sigma/\sqrt{n})}{\sqrt{(n-1)S^2/(\sigma^2(n-1))}} = \frac{\bar{X}-\mu}{S/\sqrt{n}} \sim t_{n-1}. \quad \square$$

**Why t has heavier tails than Normal:** The denominator $S/\sqrt{n}$ is random. On lucky samples, $S$ is small, making $T$ large. On unlucky samples, $S$ is large, making $T$ small. This extra randomness spreads the distribution's tails. As $n \to \infty$, $S \to \sigma$ by LLN, and $t_n \to \mathcal{N}(0,1)$.

---

## Appendix G: Power Analysis — Detailed Derivations

**One-sample z-test power derivation:**

Under $H_1: \mu = \mu_1$, the test statistic $Z = (\bar{X} - \mu_0)/(\sigma/\sqrt{n})$ has distribution:
$$Z \sim \mathcal{N}\!\left(\frac{(\mu_1-\mu_0)\sqrt{n}}{\sigma}, 1\right) = \mathcal{N}(\delta\sqrt{n}, 1)$$

where $\delta = (\mu_1 - \mu_0)/\sigma$ is Cohen's d.

Two-sided rejection region: $\lvert Z \rvert > z_{\alpha/2}$. Power:

$$\pi(\mu_1) = P_{\mu_1}(Z > z_{\alpha/2}) + P_{\mu_1}(Z < -z_{\alpha/2})$$

For $\mu_1 > \mu_0$ (so $\delta > 0$), the second term is negligible, giving:

$$\pi(\mu_1) \approx 1 - \Phi(z_{\alpha/2} - \delta\sqrt{n})$$

Setting $\pi(\mu_1) = 1 - \beta$ (desired power):

$$z_{\alpha/2} - \delta\sqrt{n} = -z_\beta \implies n = \left(\frac{z_{\alpha/2} + z_\beta}{\delta}\right)^2$$

**Power table for two-sample test ($\alpha = 0.05$, $\beta = 0.20$):**

| Cohen's d | n per group |
|---|---|
| 0.20 (small) | 393 |
| 0.50 (medium) | 64 |
| 0.80 (large) | 26 |
| 1.00 (very large) | 17 |

**For accuracy comparisons (proportions, $\alpha = 0.05$, $\beta = 0.20$):**

| Accuracy gap | Baseline | n per group |
|---|---|---|
| 0.5% | 85% | ~28,000 |
| 1.0% | 85% | ~7,200 |
| 2.0% | 85% | ~1,800 |
| 5.0% | 85% | ~310 |

These numbers explain why ML benchmark evaluations are so often underpowered: a 5% absolute improvement requires only 310 examples per model, but a 1% improvement requires 7,200 — yet many benchmarks have 1,000-3,000 examples total.

---

## Appendix H: Exact Permutation Distribution

For a two-sample test with $n_1 = n_2 = n/2$ observations, there are $\binom{n}{n/2}$ possible label assignments under $H_0$. For $n = 20$: $\binom{20}{10} = 184{,}756$ permutations — feasible to enumerate exactly. For $n = 100$: $\binom{100}{50} \approx 10^{29}$ — use Monte Carlo with $B = 10{,}000$ permutations.

**Exactness:** The permutation p-value $\hat{p} = |\{b : T^{(b)} \geq T_{\text{obs}}\}| / B$ is an unbiased estimate of the true permutation p-value. Adding 1 to numerator and denominator (standard practice) ensures $\hat{p} > 0$ and conservatism.

**Validity without normality:** The permutation test is exactly valid for any test statistic, any sample size, and any continuous distribution. The only assumption is exchangeability under $H_0$ — which is guaranteed by randomisation in designed experiments.

---

## Appendix I: Benjamini-Hochberg Procedure — Step-by-Step

**Input:** p-values $p_1, \ldots, p_m$ (unordered); target FDR level $q$.

**Algorithm:**
1. Sort: $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$.
2. For each $i$ from $m$ down to 1: check if $p_{(i)} \leq \frac{i \cdot q}{m}$.
3. Let $k = \max\{i : p_{(i)} \leq iq/m\}$ (or $k = 0$ if no such $i$ exists).
4. Reject $H_{(1)}, \ldots, H_{(k)}$.

**Example:** $m = 10$ tests, $q = 0.05$. Sorted p-values: 0.001, 0.008, 0.039, 0.041, 0.042, 0.060, 0.074, 0.205, 0.396, 0.950.

BH thresholds ($iq/m$): 0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050.

| $i$ | $p_{(i)}$ | $iq/m$ | $p_{(i)} \leq iq/m$? |
|---|---|---|---|
| 1 | 0.001 | 0.005 | Yes |
| 2 | 0.008 | 0.010 | Yes |
| 3 | 0.039 | 0.015 | **No** |
| 4 | 0.041 | 0.020 | No |

Working from bottom: $p_{(4)} = 0.041 > 0.020$, $p_{(3)} = 0.039 > 0.015$, $p_{(2)} = 0.008 \leq 0.010$ → $k = 2$. Reject hypotheses 1 and 2.

Bonferroni would require $p \leq 0.005$ — only hypothesis 1 would be rejected. BH is more powerful.

---

## Appendix J: Sequential Testing and the Optional Stopping Problem

**The optional stopping theorem (Doob):** For a martingale $\{M_n\}$ and stopping time $\tau$: $\mathbb{E}[M_\tau] = \mathbb{E}[M_0]$ under mild conditions. Under $H_0$, the likelihood ratio $\Lambda_n = \prod p(x_i|H_1)/p(x_i|H_0)$ is a martingale, so $\mathbb{E}[\Lambda_\tau] = 1$.

**Why peeking inflates error:** If you peek at p-values and stop whenever $p < \alpha$, you are effectively running a random walk and stopping when it first crosses a boundary. The boundary crossings are more frequent than the fixed-$n$ analysis assumes, inflating the Type I error.

**Ville's inequality:** For a non-negative martingale $\{M_n\}$ with $M_0 = 1$ and any stopping time $\tau$:
$$P\!\left(\sup_{n \leq \tau} M_n \geq 1/\alpha\right) \leq \alpha$$

This is the key inequality behind always-valid p-values: if you stop when $\Lambda_n \geq 1/\alpha$ (equivalently when p-value $\leq \alpha$), the false positive rate is controlled at $\alpha$ regardless of when you stop.

**E-values:** A recent (2020+) framework replaces p-values with *e-values* $E \geq 0$ satisfying $\mathbb{E}_{H_0}[E] \leq 1$. E-values can be combined multiplicatively across observations and across experiments, and Ville's inequality guarantees $P_{H_0}(E \geq 1/\alpha) \leq \alpha$ at any stopping time. E-values are the natural language for sequential testing and meta-analysis.

---

## Appendix K: Worked Examples — Common Tests

### K.1 One-Sample t-Test

**Problem:** A new LLM fine-tune is tested on 15 reasoning problems. Mean score = 72.3, sample std = 8.7. Baseline score = 68.0. Is the improvement significant at $\alpha = 0.05$?

**Solution:**
- $H_0: \mu = 68$, $H_1: \mu > 68$ (one-sided, improvement was predicted).
- $T = (72.3 - 68)/(8.7/\sqrt{15}) = 4.3/2.247 = 1.913$.
- Critical value: $t_{14, 0.05} = 1.761$.
- $T = 1.913 > 1.761$: reject $H_0$.
- p-value $= P(t_{14} > 1.913) \approx 0.038$.
- **Conclusion:** The fine-tune shows a statistically significant improvement ($p = 0.038$, one-sided $t_{14}$).
- **Effect size:** Cohen's d $= (72.3 - 68)/8.7 = 0.49$ (medium effect).

### K.2 Two-Proportion Z-Test for A/B Test

**Problem:** A chat interface is tested: control group ($n_1 = 5{,}000$) has 12% click-through rate; treatment group ($n_2 = 5{,}000$) has 13.5% CTR. Is the improvement significant?

**Solution:**
- $H_0: p_1 = p_2$, $H_1: p_1 \neq p_2$ (two-sided, pre-specified).
- Pooled $\hat{p} = (600 + 675)/10{,}000 = 0.1275$.
- $\text{SE} = \sqrt{0.1275(1-0.1275)(1/5000 + 1/5000)} = \sqrt{0.1275 \cdot 0.8725 \cdot 0.0004} = 0.00667$.
- $Z = (0.135 - 0.120)/0.00667 = 2.25$.
- p-value $= 2(1 - \Phi(2.25)) = 2(0.0122) = 0.024 < 0.05$: significant.
- **Effect size:** Cohen's $h = 2\arcsin\sqrt{0.135} - 2\arcsin\sqrt{0.120} = 0.043$ (very small).
- **Decision:** Statistically significant but effect is tiny. Consider cost of deployment vs. 1.5% CTR gain.

### K.3 Chi-Squared Test of Independence

**Problem:** Test whether LLM output quality (good/bad) is independent of prompt language (English/French/Spanish/German). Contingency table:

|  | English | French | Spanish | German |
|---|---|---|---|---|
| Good | 420 | 310 | 290 | 180 |
| Bad | 80 | 90 | 110 | 70 |

**Solution:**
- $H_0$: quality independent of language.
- Row totals: 1200, 350; Column totals: 500, 400, 400, 250. Grand total: 1550.
- $E_{11} = 1200 \times 500/1550 = 387.1$; compute all 8 expected values.
- $\chi^2 = \sum (O-E)^2/E \approx 23.4$, df $= (2-1)(4-1) = 3$.
- $p = P(\chi^2_3 > 23.4) < 0.0001$: strong evidence of language dependence.
- Cramér's V $= \sqrt{23.4/(1550 \cdot 1)} = 0.123$ (small to medium effect).
- **Conclusion:** Quality differs significantly across languages; Spanish and German have notably higher error rates.

---

## Appendix L: Further Reading

### Core Textbooks

1. **Lehmann & Romano — *Testing Statistical Hypotheses* (3rd ed., 2005):** The definitive theoretical reference. Covers NP lemma, UMP tests, unbiasedness, invariance, and asymptotic theory with full proofs. Essential for anyone wanting the complete frequentist theory.

2. **Casella & Berger — *Statistical Inference* (2nd ed., 2001):** Chapters 8-9 cover hypothesis testing at the graduate textbook level. Excellent balance of theory and computation.

3. **Wasserman — *All of Statistics* (2004):** Compressed, modern treatment with connections to ML. Chapters 10-14 cover testing, p-values, and multiple testing.

4. **Efron & Hastie — *Computer Age Statistical Inference* (2016):** Covers bootstrap, FDR, empirical Bayes, and algorithmic inference. Free PDF from Stanford.

### ML-Specific References

5. **Dror et al. — "Deep Dominance: How to Properly Compare Deep Neural Models" (ACL 2019):** Comprehensive study of hypothesis tests for NLP model comparison. Advocates for bootstrap and permutation tests over t-tests.

6. **Demšar — "Statistical Comparisons of Classifiers over Multiple Datasets" (JMLR, 2006):** Recommends Friedman test + Nemenyi post-hoc for comparing multiple classifiers across multiple datasets.

7. **Johari et al. — "Peeking at A/B Tests" (KDD 2017):** The original paper on always-valid p-values and mSPRT for online A/B testing.

8. **Ramdas et al. — "Testing Exchangeability: Fork-Convex Hulls, Supermartingales and e-Processes" (2022):** Modern framework for e-values and anytime-valid inference.

9. **Liao et al. — "Are Emergent Abilities of Large Language Models a Mirage?" (NeurIPS 2023):** Demonstrates that many claimed LLM phase transitions are statistical artifacts of discontinuous metrics + multiple comparisons.

10. **Koehn — "Statistical Significance Tests for Machine Translation Evaluation" (EMNLP 2004):** The canonical reference for bootstrap resampling in MT evaluation. Introduced paired bootstrap testing to NLP.


---

## Appendix M: Advanced Topics in Hypothesis Testing

### M.1 Composite Hypotheses and Nuisance Parameters

Many practical testing problems involve **nuisance parameters** — parameters that appear in the model but are not the focus of the test. For example, in the two-sample t-test, the common variance $\sigma^2$ (or the two separate variances in Welch's test) are nuisance parameters when the hypothesis concerns the difference in means.

**Problem:** If $H_0: \mu_1 = \mu_2$ with unknown $\sigma_1, \sigma_2$ (Behrens-Fisher problem), there is no exact test. The Welch t-test provides an approximate solution via the Satterthwaite degrees of freedom approximation.

**Conditional tests:** One approach is to condition on sufficient statistics for the nuisance parameters. Fisher's exact test conditions on the row and column marginals of a contingency table — the marginals are ancillary for the association parameter of interest.

**Profile likelihood:** Replace nuisance parameters by their profile MLEs. The profile likelihood ratio test then has the same $\chi^2$ asymptotic distribution as the full GLRT.

### M.2 Equivalence Testing and Non-Inferiority Tests

Classical hypothesis testing asks: "is there an effect?" But in ML deployment, the question is often reversed: "is the new model **at least as good as** the old one?" This requires **equivalence testing** or **non-inferiority testing**.

**TOST (Two One-Sided Tests):** To test that $\lvert \mu_1 - \mu_2 \rvert < \Delta$ (practically equivalent):
1. Test $H_{01}: \mu_1 - \mu_2 \geq \Delta$ at level $\alpha$ (one-sided).
2. Test $H_{02}: \mu_1 - \mu_2 \leq -\Delta$ at level $\alpha$ (one-sided).
3. Conclude equivalence if both are rejected.

The equivalence margin $\Delta$ must be pre-specified based on domain knowledge (e.g., "a difference of less than 0.5% accuracy is practically irrelevant").

**Non-inferiority test:** Show that the new model is not worse than baseline by more than $\Delta$:
$$H_0: \mu_{\text{new}} < \mu_{\text{baseline}} - \Delta \quad \text{vs.} \quad H_1: \mu_{\text{new}} \geq \mu_{\text{baseline}} - \Delta$$

Both frameworks are essential for responsible ML deployment: before retiring a production model, verify the replacement is not inferior beyond an acceptable margin.

### M.3 Multiple Testing in Modern Machine Learning

**Neural architecture search (NAS):** Testing thousands of architectural variants involves extreme multiple comparisons. Without FDR correction, reported improvements are largely artifacts. Proper NAS evaluation requires:
- Held-out final evaluation (not the search objective).
- BH correction across all tried architectures.
- Multiple random seeds per architecture.

**Hyperparameter tuning:** Grid search over $k$ hyperparameters creates implicit multiple comparisons. Bayesian optimization with proper uncertainty quantification (Gaussian processes) naturally avoids this by reasoning about the distribution over hyperparameter performance rather than making independent comparisons.

**Neural network weight testing:** Magnitude pruning implicitly tests whether each weight is significantly different from zero. The formal version is a Wald test $W_j = \hat{\theta}_j^2 / \widehat{\text{Var}}(\hat{\theta}_j)$, where $\widehat{\text{Var}}$ comes from the Fisher information matrix. Applying BH correction gives a principled sparse pruning criterion. This connects to the lottery ticket hypothesis: a subnetwork survives iff its weights are statistically distinguishable from zero.

### M.4 Causal Inference and Hypothesis Testing

Standard hypothesis testing establishes **association** ($P(Y|X) \neq P(Y)$) but not **causation** ($\text{do}(X = x)$ changes $P(Y)$). The connection:

**Randomised experiments:** When treatments are randomly assigned (RCT), the two-sample t-test or Wilcoxon test on outcomes provides valid causal inference. Randomisation eliminates confounding, so association implies causation.

**Observational studies:** Without randomisation, a significant test only shows association. Causal inference requires additional assumptions (instrumental variables, regression discontinuity, difference-in-differences) and sensitivity analysis.

**RLHF and causal testing:** When evaluating whether RLHF improves a model, the "treatment" (RLHF fine-tuning) must be applied to otherwise identical models. Comparing a fine-tuned model to a different base model conflates the RLHF effect with base model differences.

---

## Appendix N: Pitfalls in Benchmark Evaluation — Extended Analysis

### N.1 The Evaluation Overfitting Problem

**Adaptive data analysis:** Every time a benchmark is used to select a model or tune hyperparameters, the benchmark becomes part of the training signal. The final evaluation on the same benchmark is biased upward.

**Holdout sets:** The standard remedy is a held-out test set that is never used for model selection. In practice, LLM benchmark contamination makes this extremely difficult — web-scraped training data often contains benchmark questions and answers.

**Differential privacy approach:** Dwork et al. (2015) proved that if researchers are allowed at most $k$ adaptive queries to a holdout set of size $n$, they can answer up to $O(n^{2/3}k^{1/3})$ queries with valid statistical guarantees. This puts a hard limit on the number of models that can be compared on a single benchmark before results become meaningless.

### N.2 The Multiple Metrics Problem

When a model is evaluated on 50 metrics (BLEU, ROUGE-1, ROUGE-2, ROUGE-L, BERTScore, METEOR, ...) and reported as "best on 30 of 50", this is not a well-defined test result. The correct approach:

1. **Pre-specify the primary metric** before evaluation.
2. Report secondary metrics as exploratory with FDR-corrected p-values.
3. Use a composite score (average normalised ranking across metrics) as the primary outcome.

### N.3 Model Size and Benchmark Artefacts

Many apparent improvements in LLM evaluations are confounded with model size. Larger models score higher on essentially every benchmark not because of the specific training choices being evaluated, but because of the additional parameters. Proper evaluation must control for (or fix) model size.

**Scaling law adjustments:** When comparing models of different sizes, use scaling law predictions (Chinchilla) to normalise scores to a common compute budget. A model that achieves $s$ score at $C$ FLOPs is better than one achieving $s$ score at $10C$ FLOPs, even if their raw scores are identical.

---

## Appendix O: Practice Problems

**Problem O.1:** Show that the chi-squared goodness-of-fit statistic $\chi^2 = \sum (O_i - E_i)^2/E_i$ can be written as $\chi^2 = \sum O_i^2/E_i - n$. Use this to show that $\chi^2 = 0$ iff $O_i = E_i$ for all $i$.

**Problem O.2:** A coin is flipped 1000 times and 520 heads observed. Compute the p-value for $H_0: p = 0.5$ vs. $H_1: p \neq 0.5$. Is the coin significantly biased at $\alpha = 0.05$? What is the 95% CI for $p$? Verify the CI/test duality.

**Problem O.3:** Two classifiers are evaluated on 200 test examples. Classifier A is correct on 156, B on 148. They agree on 130 correct and 18 incorrect predictions. Set up and perform McNemar's test. Compare to a naive two-proportion z-test on the aggregate counts.

**Problem O.4:** Prove that for the Mann-Whitney U statistic, $\mathbb{E}[U/n_1 n_2] = P(X > Y)$ where $X$ and $Y$ are independent draws from the two populations. Conclude that $U/(n_1 n_2) = \hat{U}$ is an unbiased estimator of $P(X > Y)$, and that under $H_0$, $P(X > Y) = 0.5$.

**Problem O.5:** Consider $m = 100$ independent tests where the true null proportion is $\pi_0 = 0.8$. Under BH at $q = 0.05$, derive the expected number of true and false discoveries as a function of the effect size under $H_1$. Plot the expected FDP (false discovery proportion) as a function of $\pi_0$.

**Problem O.6:** Implement the SPRT for comparing two Bernoulli distributions with $p_0 = 0.5$ vs. $p_1 = 0.6$. Run the test 10,000 times under $H_0$ (true $p = 0.5$) and 10,000 times under $H_1$ (true $p = 0.6$). Report: (a) empirical Type I error rate, (b) empirical Type II error rate, (c) distribution of stopping times under $H_0$ and $H_1$.

**Problem O.7:** You have 5,000 features in a fraud detection model. After retraining on new data, 847 features show distributional shift with raw p-value < 0.05 from KS tests. (a) How many false discoveries do you expect under $H_0$? (b) Apply BH at $q = 0.05$ and report the number of discoveries. (c) Which features would be worth investigating for model update?

**Problem O.8:** A researcher claims that their new prompt engineering technique improves GPT-4 on MMLU from 86.4% to 87.1% (score on 14,000 questions). Perform a formal hypothesis test and compute the 95% CI. Is this improvement practically significant? What is the effect size (Cohen's h)?

---

## Appendix P: Connection to Information Theory

Hypothesis testing has deep connections to information theory. These connections illuminate why certain tests are optimal and provide a geometric view of the testing problem.

### P.1 KL Divergence and Test Power

The **Chernoff information** between two distributions $P$ and $Q$ is:
$$C(P, Q) = -\min_{0 \leq s \leq 1} \log \int p(\mathbf{x})^s q(\mathbf{x})^{1-s} d\mathbf{x}$$

For $n$ independent observations, the optimal Type II error rate (at fixed Type I error) decays exponentially as $e^{-nC(P,Q)}$. Chernoff information determines the fundamental limit on how fast hypothesis testing errors vanish with sample size.

**Special case:** For one-sided tests with simple $H_0$ and $H_1$, the Stein's lemma states:
$$\beta_n^* \approx e^{-n D_{\mathrm{KL}}(P \| Q)}$$

where $D_{\mathrm{KL}}(P \| Q)$ is the KL divergence from $Q$ to $P$. More KL divergence → test errors vanish faster. This is why KL divergence is the natural measure of "distance" between distributions in testing.

### P.2 Sufficient Statistics and Data Processing

The **Data Processing Inequality** states that processing data (applying a function $f$) cannot increase information:
$$I(X; Y) \geq I(f(X); Y)$$

In hypothesis testing: a test statistic $T(\mathbf{x})$ is a deterministic function of $\mathbf{x}$. By the DPI, $T$ cannot be more informative about $\theta$ than $\mathbf{x}$ itself. Equality holds when $T$ is a **sufficient statistic** — when $T$ captures all the information in the data about $\theta$.

This gives an information-theoretic characterisation of sufficient statistics: $T$ is sufficient for $\theta$ iff $I(T; \theta) = I(\mathbf{x}; \theta)$, i.e., no information about $\theta$ is lost by replacing $\mathbf{x}$ with $T(\mathbf{x})$. A test based on a sufficient statistic is just as powerful as one based on the full data.

### P.3 Minimum Description Length and Hypothesis Selection

**MDL (Minimum Description Length):** The MDL principle selects the model that provides the shortest description of the data. For hypothesis testing:
- The null model $H_0$ has description length $L(H_0) + L(\mathbf{x}|H_0)$.
- The alternative $H_1$ has description length $L(H_1) + L(\mathbf{x}|H_1)$.

Reject $H_0$ if the alternative provides a shorter total description. This is equivalent to GLRT when $H_0$ and $H_1$ are parametric models of different complexity — the MDL penalty for the more complex model plays the role of the chi-squared df in Wilks' theorem.

**Connection to Bayes factors:** If priors on $H_0$ and $H_1$ are encoded as prefix codes, the Bayes factor equals $2^{L(H_0) - L(H_1)}$ where $L$ includes both model complexity and data description length. MDL, Bayes factors, and GLRT are three facets of the same information-theoretic principle: model complexity must be penalised when comparing models of different complexity.


---

## Appendix Q: Numerical Examples — Power Curves

### Q.1 Power Curve for One-Sample t-Test

For a one-sample t-test ($H_0: \mu = 0$, $\sigma = 1$ known, $n = 25$, $\alpha = 0.05$):

| True $\mu$ | Cohen's d | Power |
|---|---|---|
| 0.0 | 0.00 | 0.050 (= $\alpha$) |
| 0.1 | 0.10 | 0.098 |
| 0.2 | 0.20 | 0.212 |
| 0.3 | 0.30 | 0.398 |
| 0.4 | 0.40 | 0.594 |
| 0.5 | 0.50 | 0.761 |
| 0.6 | 0.60 | 0.876 |
| 0.8 | 0.80 | 0.977 |
| 1.0 | 1.00 | 0.998 |

Observation: 80% power requires $d \approx 0.57$, i.e., true mean $\approx 0.57\sigma$ away from null.

### Q.2 Effect of Sample Size on Power

For a two-sample t-test detecting $d = 0.5$ (medium effect), $\alpha = 0.05$:

| $n$ per group | Power |
|---|---|
| 20 | 0.338 |
| 40 | 0.521 |
| 64 | 0.700 |
| 100 | 0.851 |
| 150 | 0.940 |
| 200 | 0.978 |
| 300 | 0.997 |

The "required $n$" of 64 achieves 70% power (not 80% — this is a common mistake in power calculation formulas; exact values depend on using the t-distribution vs. normal approximation).

### Q.3 Multiple Testing Power Comparison

For $m = 50$ tests, 10 true alternatives with $d = 0.5$, $n = 100$ per comparison:

| Method | FWER | FDR | Expected discoveries | Expected true disc. |
|---|---|---|---|---|
| No correction | ~1.00 | ~0.30 | 12 | 8.5 |
| Bonferroni | 0.05 | ~0.01 | 6.5 | 6.4 |
| Holm | 0.05 | ~0.02 | 6.8 | 6.7 |
| BH ($q=0.05$) | ~0.35 | 0.05 | 9.3 | 8.9 |

BH makes approximately 2.7 more true discoveries than Bonferroni at the cost of slightly higher FWER.

---

## Appendix R: Connection to Decision Theory

### R.1 Minimax Hypothesis Testing

Hypothesis testing can be formulated as a decision problem. Let:
- $\ell_{ij}$ = loss of decision $d_i$ when $H_j$ is true.
- Standard 0-1 loss: $\ell_{00} = \ell_{11} = 0$ (correct decision), $\ell_{10} = 1$ (reject when $H_0$ true), $\ell_{01} = 1$ (fail to reject when $H_1$ true).

The **Bayes risk** of a test $\phi$ with prior $(\pi_0, \pi_1)$ on the hypotheses:
$$r(\phi, \pi) = \pi_0 \mathbb{E}_{H_0}[\ell_{10}\phi(\mathbf{x})] + \pi_1 \mathbb{E}_{H_1}[\ell_{01}(1-\phi(\mathbf{x}))] = \pi_0 \alpha + \pi_1 \beta$$

Minimising Bayes risk gives the likelihood ratio test (NP lemma generalised to Bayesian setting): reject when $p(\mathbf{x}|H_1)/p(\mathbf{x}|H_0) > \pi_0 \ell_{10}/(\pi_1 \ell_{01})$.

The **minimax test** minimises the maximum risk over all priors:
$$\phi^* = \arg\min_\phi \max_\pi r(\phi, \pi)$$

For 0-1 loss, the minimax test is the one that equalises the power function at $\theta_0$ and $\theta_1$ — equivalently, the Bayes test under the least favourable prior.

### R.2 Asymptotic Relative Efficiency

How much more data does test A need compared to test B to achieve the same power? The **asymptotic relative efficiency** (ARE) of B relative to A is:

$$\text{ARE}(B, A) = \lim_{n \to \infty} \frac{n_A}{n_B}$$

where $n_A, n_B$ are the sample sizes needed for both tests to achieve power $\pi(\theta)$.

**Pitman efficiency** computes ARE for tests against local alternatives $\theta_n = \theta_0 + c/\sqrt{n}$:

$$\text{ARE}(B, A) = \frac{e_B}{e_A}$$

where $e_T = [(\partial/\partial\theta)\mathbb{E}_{H_1}[T]]^2 / \operatorname{Var}_{H_0}(T)$ is the efficiency of test statistic $T$.

**Key results:**
- Wilcoxon vs. t-test for Gaussian data: $\text{ARE} = 3/\pi \approx 0.955$. Wilcoxon loses only 4.5% efficiency.
- Wilcoxon vs. t-test for heavy-tailed data: $\text{ARE} > 1$. Wilcoxon can be substantially more efficient.
- Minimum ARE of Wilcoxon vs. t-test (over all symmetric distributions): $0.864$ — Wilcoxon never needs more than 16% more data.

This remarkable result (Hodges-Lehmann, 1956) justifies using Wilcoxon as a default nonparametric test: you sacrifice at most 16% efficiency relative to t-test in the best case for t, while potentially gaining large efficiency for non-normal distributions.

### R.3 Sensitivity Analysis for Robust Testing

In observational studies, test validity depends on unverifiable assumptions. **Sensitivity analysis** asks: how strong would an unmeasured confounder need to be to explain the observed effect?

**Rosenbaum's sensitivity parameter $\Gamma$:** For a matched pairs study, $\Gamma$ is the odds ratio of treatment assignment that an unmeasured binary confounder could induce. A result is "significant at $\Gamma$-sensitivity level" if it remains significant even when allowing for a confounder with odds ratio $\Gamma$.

Report sensitivity: "Our finding remains significant at the $\Gamma = 2$ level, meaning an unmeasured confounder would need to double the odds of treatment to explain away the effect."

For AI experiments, sensitivity analysis is essential when comparing models across different data pipelines, hardware, or evaluation setups — all of which are potential confounders.


---

## Appendix S: Statistical Testing Checklist for ML Practitioners

Before reporting any hypothesis test in a paper or technical document, verify the following:

### S.1 Pre-Analysis Checklist

- [ ] **Hypothesis pre-specified:** $H_0$, $H_1$, and the primary metric were stated before data collection or model training.
- [ ] **Test choice pre-specified:** The specific test (t-test, McNemar, permutation, etc.) was chosen based on the study design, not on which test gives a lower p-value.
- [ ] **Sample size justified:** Power analysis was performed and the required $n$ was collected (or power at the actual $n$ is reported).
- [ ] **Significance level stated:** $\alpha$ was pre-specified (typically 0.05; consider 0.01 for high-stakes claims).
- [ ] **Multiple comparisons planned:** If multiple tests are planned, the correction method (Bonferroni/BH) was pre-specified.

### S.2 Analysis Checklist

- [ ] **Assumptions verified:**
  - Normality (for t-test): checked via Shapiro-Wilk or Q-Q plot for $n < 50$.
  - Homoscedasticity (for pooled t-test/ANOVA): checked via Levene's test; use Welch if violated.
  - Independence: observations are not clustered, repeated, or time-dependent.
- [ ] **Correct test applied:** Paired data → paired test. Small expected counts → Fisher's exact. Non-normal small $n$ → nonparametric.
- [ ] **Effect size computed:** Cohen's d/h/f or Cramér's V reported alongside p-value.
- [ ] **Confidence interval reported:** 95% CI for the effect size (not just the p-value).

### S.3 Reporting Checklist

- [ ] **Exact p-value reported:** Not just "p < 0.05" but the exact value (e.g., $p = 0.032$).
- [ ] **Test statistic and df reported:** "$t(38) = 2.14$, $p = 0.039$" is complete; "$p < 0.05$" is not.
- [ ] **Effect size and CI reported:** "$d = 0.68$ (95% CI: [0.22, 1.14])".
- [ ] **Sample size reported:** $n$ per group for two-sample tests.
- [ ] **Multiple testing correction applied:** Which method and at what level.
- [ ] **No HARK:** Post-hoc analyses are clearly labelled as exploratory.

### S.4 Interpretation Checklist

- [ ] **Statistical vs. practical significance distinguished:** A significant result with $d = 0.05$ may not justify deployment cost.
- [ ] **Null result properly qualified:** "We failed to find evidence of X" not "We showed X does not exist". Power and MDE are reported.
- [ ] **Replication recommended:** A single significant result (especially $p \approx 0.04$) should be confirmed in an independent replication.

---

## Appendix T: Quick Reference — Test Statistics and Null Distributions

### One-Sample Tests

| Test | Statistic | Null Distribution | When to Use |
|---|---|---|---|
| Z-test | $Z = (\bar{X}-\mu_0)/(\sigma/\sqrt{n})$ | $\mathcal{N}(0,1)$ | Normal, $\sigma$ known |
| One-sample t | $T = (\bar{X}-\mu_0)/(S/\sqrt{n})$ | $t_{n-1}$ | Normal, $\sigma$ unknown |
| Sign test | $B = \#\{X_i > \mu_0\}$ | $\text{Binomial}(n, 0.5)$ | Any continuous, robust |
| Wilcoxon signed-rank | $W^+ = \sum_{D_i>0} R_i$ | Wilcoxon distribution | Symmetric, non-normal |
| Chi-squared GoF | $\sum (O_i-E_i)^2/E_i$ | $\chi^2_{k-1}$ | Count data |

### Two-Sample Tests (Independent)

| Test | Statistic | Null Distribution | When to Use |
|---|---|---|---|
| Welch t | See §4.2 | $t_\nu$ (Satterthwaite) | Normal, unequal var |
| Pooled t | $T = (\bar{X}_1-\bar{X}_2)/(S_p\sqrt{1/n_1+1/n_2})$ | $t_{n_1+n_2-2}$ | Normal, equal var |
| Z-test (proportions) | See §4.1 | $\mathcal{N}(0,1)$ | Large $n$, proportions |
| Mann-Whitney | $U$ statistic | Wilcoxon/Normal approx | Non-normal |
| KS test | $D_{n,m}$ | Kolmogorov dist | Any continuous |
| Permutation | Any statistic | Empirical (permuted) | Any statistic, any dist |

### Two-Sample Tests (Paired)

| Test | When to Use |
|---|---|
| Paired t-test | Normal differences |
| Wilcoxon signed-rank | Non-normal differences |
| Sign test | Ordinal or non-symmetric |
| McNemar | Binary outcomes (accuracy) |
| Permutation | Any paired statistic |

### $k$-Sample Tests

| Test | Null distribution | When to Use |
|---|---|---|
| One-way ANOVA | $F_{k-1, N-k}$ | Normal, equal variances |
| Welch ANOVA | $F$ (adjusted df) | Normal, unequal variances |
| Kruskal-Wallis | $\chi^2_{k-1}$ | Non-normal |
| Friedman | $\chi^2_{k-1}$ | Repeated measures |


---

## Appendix U: Extended Worked Examples — Machine Learning Scenarios

### U.1 McNemar's Test for LLM Comparison

**Setting:** Two LLMs (Gemini Pro and GPT-4o) are evaluated on 1,200 coding problems. For each problem, each model either passes or fails the test suite.

**Data:**
- Both pass: $n_{11} = 820$
- GPT-4o passes, Gemini fails: $n_{10} = 95$
- Gemini passes, GPT-4o fails: $n_{01} = 68$
- Both fail: $n_{00} = 217$

**McNemar's test:**
$$\chi^2 = \frac{(n_{10} - n_{01})^2}{n_{10} + n_{01}} = \frac{(95 - 68)^2}{95 + 68} = \frac{729}{163} = 4.47$$

$df = 1$, $\chi^2_{1, 0.05} = 3.84$. Since $4.47 > 3.84$: reject $H_0$ at $\alpha = 0.05$.

GPT-4o accuracy: $(820 + 95)/1200 = 76.25\%$. Gemini accuracy: $(820 + 68)/1200 = 74.00\%$. The 2.25% gap is statistically significant ($p = 0.034$).

**If we had naively used a two-proportion z-test:**
$$Z = \frac{0.7625 - 0.7400}{\sqrt{\bar{p}(1-\bar{p})(1/1200+1/1200)}} = \frac{0.0225}{\sqrt{0.7513 \cdot 0.2487 \cdot 0.00167}} = \frac{0.0225}{0.01766} = 1.27$$

$p = 0.20$: not significant! The z-test ignores the correlation between paired responses. McNemar correctly uses only the $163$ discordant pairs, which concentrate all the information about the performance difference.

### U.2 Bootstrap Confidence Interval for BLEU Score Comparison

**Setting:** Two MT systems are evaluated on 500 test sentences. System A achieves BLEU = 28.3, System B achieves BLEU = 26.8. Is the 1.5 BLEU point difference significant?

**Algorithm (paired bootstrap test):**
1. For $b = 1, \ldots, 10{,}000$:
   - Sample 500 sentence pairs with replacement (same indices for both systems).
   - Compute BLEU(A$^{(b)}$) - BLEU(B$^{(b)}$) on the resampled set.
2. Estimate $p = P(\text{BLEU}(A^{(b)}) - \text{BLEU}(B^{(b)}) \leq 0)$ — the fraction of bootstrap replicates where B is better.

This is the Koehn (2004) paired bootstrap test, the standard for MT evaluation.

**Why not t-test?** BLEU is a corpus-level metric (not an average of per-sentence scores), so the CLT does not directly apply. Bootstrap resampling over sentences respects the actual data-generating process.

### U.3 SPRT for Online Evaluation

**Setting:** A chat assistant is being A/B tested. Primary metric: thumbs-up rate. Control: $p_0 = 0.72$. Treatment hypothesis: $p_1 = 0.75$. Target: $\alpha = 0.05$, $\beta = 0.10$.

**Wald boundaries:**
- Upper boundary: $A = \log((1-\beta)/\alpha) = \log(0.90/0.05) = \log(18) = 2.890$.
- Lower boundary: $B = \log(\beta/(1-\alpha)) = \log(0.10/0.95) = \log(0.105) = -2.254$.

**Log-likelihood ratio increment per observation:**

For a thumbs-up ($x = 1$): $\delta_1 = \log(p_1/p_0) = \log(0.75/0.72) = 0.040$.
For a thumbs-down ($x = 0$): $\delta_0 = \log((1-p_1)/(1-p_0)) = \log(0.25/0.28) = -0.113$.

**Expected stopping times:**
- Under $H_1$ ($p = 0.75$): $\mathbb{E}[\tau] \approx (B \cdot \pi_1 + A \cdot \pi_0) / \mathbb{E}_1[\delta]$ ≈ $(-2.254 \cdot 0.10 + 2.890 \cdot 0.90) / (0.75\delta_1 + 0.25\delta_0) \approx 2.396/0.002 \approx 1{,}200$ observations.
- Fixed-$n$ test for same $\alpha, \beta$: approximately $2{,}100$ observations.

SPRT requires ~43% fewer observations in this scenario by stopping early when evidence accumulates quickly.

### U.4 KS-Based Feature Drift Alert

**Setting:** An NLP model processes document embeddings. Reference distribution of document lengths (tokens): $\mathcal{N}(256, 80^2)$ fitted on 50,000 training documents. Daily monitoring with 1,000 production documents.

**Drift events:**
- **Week 1 (no drift):** $\bar{X}_{\text{prod}} = 258$, $S_{\text{prod}} = 82$. KS statistic $D = 0.021$, $p = 0.73$. No alert.
- **Week 2 (mean shift):** $\bar{X}_{\text{prod}} = 310$, $S_{\text{prod}} = 85$. $D = 0.118$, $p = 0.00003$. **Alert: mean drift**.
- **Week 3 (variance shift only):** $\bar{X}_{\text{prod}} = 257$, $S_{\text{prod}} = 150$. $D = 0.043$, $p = 0.12$. t-test: $p = 0.84$ (no mean shift). **KS detects variance drift that t-test misses**.

This demonstrates the key advantage of KS over t-test for drift detection: KS is sensitive to any distributional change (mean, variance, shape), while t-test only detects mean shifts.

---

## Appendix V: Historical Notes

### V.1 The Lady Tasting Tea

Fisher's canonical example (1935): A lady claims she can tell whether tea or milk was poured first. Fisher designs an experiment: 8 cups, 4 with tea first, 4 with milk first, presented in random order. The lady must identify the 4 tea-first cups.

Under $H_0$ (random guessing): $P(\text{all 4 correct}) = 1/\binom{8}{4} = 1/70 = 0.014$.

This tiny experiment — 8 cups, 1 run — is sufficient to achieve $p = 0.014$ if the lady guesses perfectly. Fisher's point: careful experimental design can yield strong statistical conclusions from minimal data.

**For AI:** The same logic applies to benchmark construction. A cleverly designed benchmark where random performance is exactly 25% (4-choice multiple choice) and human performance is 90% has high discriminating power. MMLU was designed with this principle.

### V.2 Gosset and the Brewery

William Sealy Gosset derived the t-distribution in 1908 while working as a statistician for Guinness Brewery. Guinness had small-batch experiments (barley yields, hop compositions) where $n$ was typically 3-10. The existing large-sample theory (requiring normality and known $\sigma$) was useless. Gosset published under the pseudonym "Student" because Guinness forbade employees from publishing (for fear of revealing industrial methods).

The t-test is thus directly connected to the practical problem of drawing conclusions from small samples — exactly the problem faced by ML researchers evaluating expensive models on small benchmark sets.

### V.3 Neyman-Pearson and the Cigarette Industry

Jerzy Neyman and Egon Pearson developed their framework in the 1930s, partly motivated by quality control in manufacturing (testing whether a batch of products meets specifications). The framework is explicitly about **decisions**, not inference: you must ship or reject a batch based on a sample inspection. This decision-theoretic framing became the dominant paradigm in industrial statistics.

The cigarette industry later (1950s-70s) exploited the p-value/significance framework to manufacture doubt about cancer studies — repeatedly pointing out that individual studies did not achieve $p < 0.05$ while ignoring the overwhelming weight of evidence across hundreds of studies. This historical episode motivates modern emphasis on effect sizes, meta-analysis, and replication over single-study p-values.


---

## Appendix W: Common Distributions — Moments and Quantiles

### W.1 Standard Normal

$$Z \sim \mathcal{N}(0, 1): \quad f(z) = \frac{1}{\sqrt{2\pi}}e^{-z^2/2}$$

Key quantiles: $z_{0.10} = 1.282$, $z_{0.05} = 1.645$, $z_{0.025} = 1.960$, $z_{0.01} = 2.326$, $z_{0.005} = 2.576$.

### W.2 Student's t-Distribution

$$T \sim t_\nu: \quad f(t) = \frac{\Gamma((\nu+1)/2)}{\sqrt{\nu\pi}\,\Gamma(\nu/2)}\left(1+\frac{t^2}{\nu}\right)^{-(\nu+1)/2}$$

$\mathbb{E}[T] = 0$, $\operatorname{Var}(T) = \nu/(\nu-2)$ for $\nu > 2$. Approaches $\mathcal{N}(0,1)$ as $\nu \to \infty$.

| df | $t_{\nu, 0.025}$ | $t_{\nu, 0.005}$ |
|---|---|---|
| 5 | 2.571 | 4.032 |
| 10 | 2.228 | 3.169 |
| 20 | 2.086 | 2.845 |
| 30 | 2.042 | 2.750 |
| 60 | 2.000 | 2.660 |
| $\infty$ | 1.960 | 2.576 |

### W.3 Chi-Squared Distribution

$$V \sim \chi^2_k: \quad f(v) = \frac{v^{k/2-1}e^{-v/2}}{2^{k/2}\Gamma(k/2)}, \quad v > 0$$

$\mathbb{E}[V] = k$, $\operatorname{Var}(V) = 2k$. Sum of $k$ independent squared standard normals.

### W.4 F-Distribution

$$F \sim F_{k_1, k_2}: \quad F = \frac{\chi^2_{k_1}/k_1}{\chi^2_{k_2}/k_2}$$

Used in ANOVA and comparing nested model likelihoods. $F_{1, \nu} = t_\nu^2$.

---

## Appendix X: Summary of Key Theorems

**Theorem 1 (Neyman-Pearson Lemma).** For simple $H_0$ vs. $H_1$, the most powerful size-$\alpha$ test rejects when $\mathcal{L}(\theta_1)/\mathcal{L}(\theta_0) > k_\alpha$.

**Theorem 2 (Wilks' Theorem).** Under regularity conditions, $-2\log\Lambda \overset{d}{\to} \chi^2_k$ as $n \to \infty$, where $k$ is the number of equality constraints in $H_0$.

**Theorem 3 (Benjamini-Hochberg).** The BH procedure at level $q$ controls $\text{FDR} \leq q \cdot m_0/m \leq q$ under independence (and PRDS).

**Theorem 4 (Kolmogorov-Smirnov).** For continuous $F_0$, $\sqrt{n} D_n \overset{d}{\to} \sup_t \lvert B(F_0(t)) \rvert$ where $B$ is a Brownian bridge.

**Theorem 5 (Wald, SPRT).** The SPRT with boundaries $A$ and $B$ satisfies $P_{H_0}(\text{reject}) \leq \alpha = B/(1-A)$ and $P_{H_1}(\text{accept } H_0) \leq \beta = A/(1-B)$ (approximately). The SPRT minimises expected sample size among all tests with the same error bounds.

**Theorem 6 (Hodges-Lehmann).** The ARE of the Wilcoxon signed-rank test relative to the t-test satisfies $\text{ARE} \geq 3/\pi \approx 0.864$ for all symmetric continuous distributions, with equality for the logistic distribution. The Wilcoxon test is never less than 86.4% as efficient as the t-test.

**Theorem 7 (Pitman-Koopmans).** For exponential families, one-sided tests of the natural parameter are UMP: reject when $T(\mathbf{x}) > c_\alpha$ for the sufficient statistic $T$.

**Theorem 8 (Equivalence of Trinity Tests).** Under $H_0$ and contiguous alternatives, the Wald, Score, and Likelihood Ratio tests are all asymptotically equivalent: they have the same asymptotic size $\alpha$ and the same asymptotic power function against local alternatives.

---

*This section is part of the [Math for LLMs curriculum](../../README.md).
Previous: [§02 Estimation Theory](../02-Estimation-Theory/notes.md) | Next: [§04 Bayesian Inference](../04-Bayesian-Inference/notes.md)*


---

## Appendix Y: Statistical Software and Implementation Notes

### Y.1 SciPy Reference for Common Tests

```python
from scipy import stats
import numpy as np

# ── One-sample tests ──────────────────────────────────────────────────
# Z-test (manually, since scipy has no z-test function)
z = (xbar - mu0) / (sigma / np.sqrt(n))
p_two = 2 * (1 - stats.norm.cdf(abs(z)))

# One-sample t-test
t, p = stats.ttest_1samp(x, popmean=mu0)

# Wilcoxon signed-rank test
w, p = stats.wilcoxon(x - mu0)

# ── Two-sample tests ──────────────────────────────────────────────────
# Welch's t-test (ALWAYS use equal_var=False unless you have strong reason)
t, p = stats.ttest_ind(x, y, equal_var=False)

# Paired t-test
t, p = stats.ttest_rel(x, y)

# Mann-Whitney U
u, p = stats.mannwhitneyu(x, y, alternative='two-sided')

# Two-sample KS test
d, p = stats.ks_2samp(x, y)

# Permutation test (scipy >= 1.8)
result = stats.permutation_test((x, y),
    statistic=lambda a, b: a.mean() - b.mean(),
    n_resamples=10_000, alternative='two-sided')
p = result.pvalue

# ── Multi-sample tests ────────────────────────────────────────────────
# One-way ANOVA
f, p = stats.f_oneway(group1, group2, group3)

# Kruskal-Wallis
h, p = stats.kruskal(group1, group2, group3)

# ── Categorical tests ─────────────────────────────────────────────────
# Chi-squared goodness-of-fit
chi2, p = stats.chisquare(observed, f_exp=expected)

# Chi-squared test of independence
chi2, p, dof, expected = stats.chi2_contingency(table)

# McNemar's test (statsmodels)
from statsmodels.stats.contingency_tables import mcnemar
result = mcnemar([[n11, n10], [n01, n00]])
p = result.pvalue
```

### Y.2 Multiple Testing Correction

```python
from statsmodels.stats.multitest import multipletests

# Bonferroni, Holm, BH corrections
reject, p_adj, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')
reject, p_adj, _, _ = multipletests(p_values, alpha=0.05, method='holm')
reject, p_adj, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

# method options: 'bonferroni', 'holm', 'fdr_bh', 'fdr_by', 'sidak',
#                 'holm-sidak', 'simes-hochberg', 'hommel'
```

### Y.3 Power Analysis

```python
from statsmodels.stats.power import (
    TTestIndPower, TTestOneSamplePower, NormalIndPower
)

# Required sample size for two-sample t-test
analysis = TTestIndPower()
n = analysis.solve_power(effect_size=0.5, alpha=0.05, power=0.80,
                          ratio=1.0, alternative='two-sided')

# Power at fixed n
power = analysis.power(effect_size=0.5, nobs1=64, alpha=0.05,
                        ratio=1.0, alternative='two-sided')

# For proportions
from statsmodels.stats.proportion import proportion_effectsize, zt_ind_solve_power
h = proportion_effectsize(0.87, 0.85)  # Cohen's h
n = zt_ind_solve_power(effect_size=h, alpha=0.05, power=0.80)
```

### Y.4 Numerical Tips

- Always set `np.random.seed(42)` before generating synthetic data for reproducibility.
- For exact p-values from t-distribution: `p = 2 * stats.t.sf(abs(t_stat), df=df)` (two-sided).
- For chi-squared p-value: `p = stats.chi2.sf(chi2_stat, df=k-1)`.
- KS test is sensitive to sample size — even tiny real differences are "significant" at large $n$. Always report KS statistic $D$ alongside p-value.
- For bootstrap tests, use at least $B = 9{,}999$ permutations (so that $p = 1/10000$ is achievable).

