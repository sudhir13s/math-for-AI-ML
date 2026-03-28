# Time Series

[← Back to Chapter 7: Statistics](../README.md) | [Next: Regression Analysis →](../06-Regression-Analysis/notes.md)

---

> _"All models are wrong, but some are useful."_
> - George E. P. Box

## Overview

Time-series analysis studies data whose order is part of the signal.
A shuffled bag of observations may support estimation,
classification,
or regression,
but it cannot answer the central questions of forecasting:
what tends to happen next,
how uncertainty grows with horizon,
which patterns are persistent,
and which changes signal a regime break.
When measurements arrive sequentially,
dependence across time is no longer a nuisance.
It is the mathematical structure that determines what can be predicted,
what can be controlled,
and what can go wrong.

This matters immediately in machine learning.
Model-serving latency,
token streams,
sensor logs,
demand curves,
traffic traces,
financial sequences,
robot trajectories,
and hidden-state evolution are all time-indexed.
ARIMA-style forecasting is still a strong baseline for many operational tasks.
Kalman filtering remains the canonical method for linear-Gaussian tracking and sensor fusion.
Spectral analysis explains periodicity,
multi-scale structure,
and why Fourier features and state-space models keep reappearing in modern sequence modeling.
Even when frontier systems use transformers,
structured state-space models,
or foundation models for forecasting,
the classical mathematics of autocorrelation,
stationarity,
innovation,
and filtering still determines what those systems are trying to capture.

This section develops time series as the statistics chapter's canonical home for temporally dependent data.
We will not re-teach the general theory of stochastic processes from [Chapter 6, Stochastic Processes](../../06-Probability-Theory/06-Stochastic-Processes/notes.md),
and we will not re-teach Bayesian posterior machinery from [Bayesian Inference](../04-Bayesian-Inference/notes.md).
Instead,
we focus on the statistical viewpoint:
observed sequences,
diagnostics,
linear dependence models,
forecasting,
seasonality,
state-space representation,
Kalman filtering,
spectral methods,
and the practical modeling decisions that connect classical time-series analysis to modern AI systems.

## Prerequisites

- **Expectation, variance, covariance, and conditional expectation** - [Chapter 6, Expectation and Moments](../../06-Probability-Theory/04-Expectation-and-Moments/notes.md)
- **Joint distributions and conditional Gaussian formulas** - [Chapter 6, Joint Distributions](../../06-Probability-Theory/03-Joint-Distributions/notes.md)
- **Stochastic-process basics, stationarity, and spectral intuition** - [Chapter 6, Stochastic Processes](../../06-Probability-Theory/06-Stochastic-Processes/notes.md)
- **Markov chains and transition-based reasoning** - [Chapter 6, Markov Chains](../../06-Probability-Theory/07-Markov-Chains/notes.md)
- **Bayesian Gaussian updating and posterior uncertainty** - [Bayesian Inference](../04-Bayesian-Inference/notes.md)
- **Matrix algebra and positive definite matrices** - [Chapter 3, Advanced Linear Algebra](../../03-Advanced-Linear-Algebra/README.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Interactive simulations for ACF/PACF, ARIMA-style modeling, seasonality, spectral diagnostics, state-space models, and Kalman filtering |
| [exercises.ipynb](exercises.ipynb) | 8 graded exercises covering stationarity, ACF/PACF interpretation, AR stability, differencing, forecasting, spectral detection, and Kalman updates |

## Learning Objectives

After completing this section, you will:

- Define a time series as an observed realization of an underlying stochastic process and separate process assumptions from data analysis
- Use lag-operator and difference-operator notation to express temporal dependence compactly
- Compute and interpret mean functions, autocovariance functions, autocorrelation functions, and partial autocorrelations
- Distinguish white noise, random walks, trend-plus-noise models, and stationary linear processes
- Explain why stationarity matters for estimation, forecasting, and diagnostics
- Derive and analyze AR, MA, ARMA, and ARIMA models in terms of dependence structure, stability, and invertibility
- Use ACF, PACF, differencing, and information criteria in the Box-Jenkins identification workflow
- Construct one-step and multi-step forecasts and explain why forecast uncertainty grows with horizon
- Analyze seasonal structure with decomposition, seasonal differencing, and SARIMA-style modeling
- Rewrite linear dynamical systems in state-space form and carry out Kalman filter predict/update steps
- Interpret spectral density, periodograms, and frequency-domain views of temporal structure
- Connect classical time-series mathematics to Kalman-based tracking, sequence models, S4/Mamba-style state-space architectures, and modern forecasting systems

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 Why Time Order Matters](#11-why-time-order-matters)
  - [1.2 From Snapshots to Sequences](#12-from-snapshots-to-sequences)
  - [1.3 Historical Timeline](#13-historical-timeline)
  - [1.4 Why Time Series Matters for AI](#14-why-time-series-matters-for-ai)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Time Series as Observed Stochastic Processes](#21-time-series-as-observed-stochastic-processes)
  - [2.2 Lag Operator, Backshift Notation, and Difference Operators](#22-lag-operator-backshift-notation-and-difference-operators)
  - [2.3 Mean Function, Autocovariance, and Autocorrelation](#23-mean-function-autocovariance-and-autocorrelation)
  - [2.4 White Noise, Random Walks, and Trend-Plus-Noise Models](#24-white-noise-random-walks-and-trend-plus-noise-models)
  - [2.5 Stationarity: Strict vs Weak](#25-stationarity-strict-vs-weak)
- [3. Core Theory I: Dependence, Stationarity, and Diagnostics](#3-core-theory-i-dependence-stationarity-and-diagnostics)
  - [3.1 Why Stationarity Matters](#31-why-stationarity-matters)
  - [3.2 Autocorrelation Function (ACF)](#32-autocorrelation-function-acf)
  - [3.3 Partial Autocorrelation Function (PACF)](#33-partial-autocorrelation-function-pacf)
  - [3.4 Differencing, Unit Roots, and Random Walk Behavior](#34-differencing-unit-roots-and-random-walk-behavior)
  - [3.5 Wold Decomposition Intuition](#35-wold-decomposition-intuition)
- [4. Core Theory II: Linear Time-Series Models](#4-core-theory-ii-linear-time-series-models)
  - [4.1 Autoregressive Models AR(p)](#41-autoregressive-models-arp)
  - [4.2 Moving-Average Models MA(q)](#42-moving-average-models-maq)
  - [4.3 ARMA Models](#43-arma-models)
  - [4.4 ARIMA Models](#44-arima-models)
  - [4.5 Model Identification with ACF/PACF and Information Criteria](#45-model-identification-with-acfpacf-and-information-criteria)
- [5. Core Theory III: Forecasting, Seasonality, and State Space](#5-core-theory-iii-forecasting-seasonality-and-state-space)
  - [5.1 One-Step and Multi-Step Forecasting](#51-one-step-and-multi-step-forecasting)
  - [5.2 Seasonality and Seasonal Decomposition](#52-seasonality-and-seasonal-decomposition)
  - [5.3 SARIMA and Seasonal Structure](#53-sarima-and-seasonal-structure)
  - [5.4 State-Space Models](#54-state-space-models)
  - [5.5 Kalman Filter](#55-kalman-filter)
- [6. Advanced Topics](#6-advanced-topics)
  - [6.1 Spectral Analysis and the Frequency-Domain View](#61-spectral-analysis-and-the-frequency-domain-view)
  - [6.2 Periodogram and Fourier Decomposition](#62-periodogram-and-fourier-decomposition)
  - [6.3 Kalman Smoothing, Forecasting, and Missing Data](#63-kalman-smoothing-forecasting-and-missing-data)
  - [6.4 Multivariate Time Series Preview](#64-multivariate-time-series-preview)
  - [6.5 Change Points, Regime Shifts, and Structural Breaks](#65-change-points-regime-shifts-and-structural-breaks)
- [7. Applications in Machine Learning](#7-applications-in-machine-learning)
  - [7.1 Forecasting for Demand, Traffic, and Monitoring](#71-forecasting-for-demand-traffic-and-monitoring)
  - [7.2 Kalman Filtering for Tracking and Sensor Fusion](#72-kalman-filtering-for-tracking-and-sensor-fusion)
  - [7.3 Transformers, Fourier Features, and Temporal Encoding](#73-transformers-fourier-features-and-temporal-encoding)
  - [7.4 Structured State Space Models in Deep Learning](#74-structured-state-space-models-in-deep-learning)
  - [7.5 Foundation Models for Time Series](#75-foundation-models-for-time-series)
- [8. Common Mistakes](#8-common-mistakes)
- [9. Exercises](#9-exercises)
- [10. Why This Matters for AI (2026 Perspective)](#10-why-this-matters-for-ai-2026-perspective)
- [11. Conceptual Bridge](#11-conceptual-bridge)
- [References](#references)

---

## 1. Intuition

### 1.1 Why Time Order Matters

The first mistake people make with temporal data is to treat it like ordinary tabular data with a timestamp column attached.
That move destroys the central object of interest.
In time series,
order is not metadata.
Order is structure.
Yesterday changes what today means,
and both change what we should expect tomorrow to look like.

If a hospital latency dashboard records
`[81, 79, 80, 82, 300]` milliseconds in order,
the last point looks like an outage or anomaly.
If we shuffle the same numbers,
the interpretation becomes much less clear.
The sample mean is identical under shuffling.
The variance is identical under shuffling.
But the operational story is radically different because the temporal arrangement changes whether we read the last value as burst,
drift,
recovery,
or normal fluctuation.

Time order creates at least four kinds of structure that iid modeling discards.

1. **Dependence.**
   Nearby observations are often correlated.
   A traffic spike at 10:00 makes a spike at 10:01 more plausible.
2. **Persistence.**
   Effects can propagate.
   A warm CPU remains warm.
   A trending topic remains active.
3. **Seasonality.**
   Repetition can occur at known lags:
   hourly,
   daily,
   weekly,
   yearly,
   or tied to application cycles.
4. **State.**
   Hidden variables such as load,
   intent,
   weather,
   or user context can evolve in time and mediate what we observe.

In static statistics,
the central question is often
"what parameter best explains the dataset?"
In time-series analysis,
the central question becomes
"what dependence pattern generated this trajectory,
and what does that imply for the next few steps?"
That question is both predictive and structural.
We want good forecasts,
but we also want interpretable models of lagged influence,
propagation,
and noise.

This is why classical time-series analysis begins with visual and algebraic diagnostics rather than only with generic fitting.
We inspect trends,
seasonality,
serial dependence,
and innovation structure before choosing a model.
A sequence is not just a collection of values.
It is a history.

```text
TIME ORDER CREATES SIGNAL
=====================================================================

Shuffled observations:
  82, 300, 79, 81, 80
  Mean and variance are visible,
  but drift, burst, and persistence are not.

Ordered observations:
  79 -> 80 -> 81 -> 82 -> 300
  Now we can ask:
    - Is this a trend?
    - Is the last value an anomaly?
    - Does the next point stay high?
    - Did the system switch regimes?

Forecasting depends on order.
Diagnosis depends on order.
Control depends on order.

=====================================================================
```

Three examples make the point concrete.

**Example 1: energy demand.**
A utility company sees electricity load every 15 minutes.
Load at 6:00 a.m. says little by itself.
Load at 6:00 a.m. relative to 5:45 a.m.,
the same time yesterday,
and the same day last week reveals morning ramp,
daily seasonality,
and whether a storm or outage has shifted demand.

**Example 2: language tokens.**
The distribution of the next token in a sentence depends strongly on the prefix.
Token streams are discrete time series.
They are not usually modeled with ARIMA,
but they still obey the general lesson:
dependence across positions is the object of interest.

**Example 3: robot tracking.**
A drone's observed position is noisy.
The latent physical state evolves smoothly.
Consecutive observations must be interpreted relative to a dynamical model.
This is exactly the setting of state-space models and Kalman filtering.

Non-examples are equally helpful.

- A collection of independently drawn image-label pairs is usually not a time series,
  even if the file system stores them in order.
- A set of daily measurements can fail to be a useful time series if the process was reset independently every day.
- A log sequence can be time-indexed but still unsuitable for classical models if interventions destroy stationarity and dependence assumptions.

The operational takeaway is simple:
time order gives us leverage,
but only if we model it explicitly.

### 1.2 From Snapshots to Sequences

Probability theory already introduced stochastic processes as indexed families of random variables.
That is the process-level viewpoint.
Time-series analysis narrows the lens.
We usually do not observe an ensemble of many independent trajectories from the same process.
We observe one long realization:
$$
x_1, x_2, \ldots, x_T.
$$
The inferential challenge is to learn from that single path.

This matters because many formulas in classical statistics rely on repeated independent sampling.
Time-series data breaks that symmetry.
When $X_t$ and $X_{t+1}$ are correlated,
the effective amount of information in $T$ observations is smaller than in an iid sample of size $T$.
If dependence is strong,
naively plugging time-series data into iid confidence intervals or hypothesis tests leads to overconfidence.

The move from snapshots to sequences changes the object being summarized.

- For iid data,
  the empirical distribution is often the main summary.
- For time-series data,
  the empirical distribution is incomplete.
  We also need lag structure,
  persistence,
  and possibly hidden states or seasonal components.

Suppose two series share the same marginal histogram.
One is white noise.
The other alternates smoothly with a long cycle.
Their sample means,
sample variances,
and quantiles can be nearly identical.
Yet their forecasts,
anomaly thresholds,
and control strategies differ dramatically.
The difference lies in serial dependence.

Time-series analysis therefore adds new summaries to the descriptive-statistics toolkit.

- **Autocovariance** asks how observations separated by lag $h$ co-vary.
- **Autocorrelation** normalizes that dependence to a dimensionless scale.
- **Partial autocorrelation** isolates direct lag effects after controlling for shorter lags.
- **Spectral density** reorganizes the same dependence information in the frequency domain.

The subject also changes what counts as a model.

In regression,
we explain a response using covariates.
In time-series models,
the past of the series itself becomes a structured predictor.
An AR model regresses the present on past values.
An MA model regresses the present on past shocks.
A state-space model introduces hidden states that evolve over time and emit observations.

> **Backward reference:** The general theory of stationary stochastic processes,
> wide-sense stationarity,
> and the Wiener-Khinchin relationship was introduced in [Stochastic Processes](../../06-Probability-Theory/06-Stochastic-Processes/notes.md).
> Here we reinterpret those ideas statistically:
> as tools for fitting,
> diagnosing,
> and forecasting from one realized sequence.

This single-path perspective forces a modeling assumption that is so important it often goes unstated:
we need some form of regularity across time in order to learn from one path.
Stationarity,
local stationarity,
ergodicity,
or stable state-space dynamics are all attempts to make that learning possible.
Without some such regularity,
the future may be unrelated to the past in a way no amount of data can overcome.

### 1.3 Historical Timeline

Time-series analysis sits at the intersection of statistics,
probability,
signal processing,
control,
and econometrics.
The ideas were developed by different communities,
often under different names,
but they now form a shared mathematical toolkit.

| Year | Contributor | Contribution |
| --- | --- | --- |
| 1927 | G. Udny Yule | Introduces autoregressive ideas while studying sunspot cycles |
| 1931 | Harold Hotelling | Connects harmonic analysis and time-series decomposition |
| 1938 | Herman Wold | Wold decomposition: stationary series as filtered innovations |
| 1949 | Norbert Wiener | Strengthens prediction and filtering foundations in signal processing |
| 1960 | Rudolf E. Kalman | Introduces the Kalman filter for linear dynamical systems |
| 1970 | George Box and Gwilym Jenkins | Popularize the identification-estimation-diagnostic cycle for ARIMA modeling |
| 1980s | Brockwell, Davis, Hamilton | Consolidate modern statistical treatments of linear time-series analysis |
| 1990s | Durbin and Koopman | State-space methods and smoothing become standard across statistics and econometrics |
| 2014-2020 | Deep sequence modeling era | RNNs, temporal CNNs, and transformers dominate many sequence applications |
| 2022 | Gu et al. (S4) | Revive state-space modeling as a competitive deep sequence architecture |
| 2023-2024 | Mamba and selective SSMs | Show efficient state-space recurrence can rival attention on long contexts |
| 2023+ | Time-series foundation models | Large pretrained forecasting systems such as TimeGPT extend forecasting beyond hand-fit classical models |

The timeline matters because it explains why time-series notation can feel heterogeneous.
Econometrics emphasizes ARIMA and forecasting.
Signal processing emphasizes filters,
transfer functions,
and spectra.
Control theory emphasizes state-space models and recursive estimation.
Machine learning emphasizes representation learning,
sequence modeling,
and scale.
This section aims to translate between these dialects without losing the common core.

### 1.4 Why Time Series Matters for AI

Modern AI systems are saturated with temporal structure.
Sometimes that structure is explicit,
as in forecasting electricity demand.
Sometimes it is hidden inside the system,
as in optimizer trajectories or streaming inference.
Either way,
the same mathematical questions recur:
what persists,
what repeats,
what drifts,
what is signal,
and how uncertainty should evolve over time.

A few high-value AI settings make this concrete.

**Forecasting and operations.**
Inference traffic,
latency,
GPU utilization,
and data-center energy demand are time series.
Operational forecasting controls autoscaling,
queue management,
and alerting.
Classical baselines such as ARIMA,
exponential smoothing,
or Kalman-style models remain highly competitive when the data are structured and sample size is limited.

**Tracking and sensor fusion.**
Autonomous systems,
wearables,
and robotics continuously integrate noisy sequential measurements.
Kalman filtering remains the first mathematical model every ML practitioner should know for this setting because it converts a dynamical prior and noisy observations into a recursive uncertainty-aware estimate.

**Sequence modeling.**
Language,
audio,
video,
and multivariate telemetry are all sequences.
Transformers,
state-space models,
and temporal convolutional networks solve very different engineering problems,
but they are all learning some approximation to temporal dependence structure.

**Representation of scale.**
Long-range dependencies often appear as low-frequency structure.
Local bursts appear as high-frequency structure.
Spectral thinking explains why Fourier features,
positional encodings,
and temporal kernels repeatedly help modern models.

**Nonstationarity and drift.**
Deployed models do not live in frozen environments.
User behavior changes.
Hardware mix changes.
Data sources shift.
Time-series reasoning provides the language of drift detection,
change points,
and adaptive updating.

```text
TIME SERIES IN THE AI STACK
=====================================================================

Raw sequential signals
  -> latency, demand, clicks, sensor readings, tokens, hidden states

Questions we ask
  -> what repeats?
  -> what persists?
  -> what drifts?
  -> what happens next?
  -> how uncertain is the forecast?

Classical tools
  -> ACF/PACF, ARIMA, decomposition, Kalman, spectra

Modern AI tools
  -> transformers, temporal CNNs, S4, Mamba, forecasting foundation models

Common mathematics
  -> dependence, state, filtering, horizon growth, frequency structure

=====================================================================
```

The philosophical point is worth stating clearly.
Time-series analysis is not obsolete just because deep learning exists.
It is the minimal language in which we can judge whether a more complex sequential model has actually learned temporal structure or merely memorized local patterns.
The best modern practice often begins with the classical model,
learns what structure is present,
and only then asks whether a larger model is justified.

---

## 2. Formal Definitions

### 2.1 Time Series as Observed Stochastic Processes

A **time series** is a sequence of observations indexed by time,
usually written
$$
\{x_t\}_{t=1}^T
\quad \text{or} \quad
\{x_t\}_{t \in \mathbb{Z}}.
$$
Statistically,
we interpret these observations as one realization of an underlying stochastic process
$$
\{X_t\}_{t \in T}.
$$
The distinction between $X_t$ and $x_t$ is crucial.
$X_t$ is the random variable.
$x_t$ is the realized observed value.

This distinction prevents a common conceptual error.
When we compute a sample autocorrelation,
we are not claiming the realized data are random after we observed them.
We are using the realized path to infer properties of the random mechanism that generated it.

Formally,
if $T = \mathbb{Z}$ or $T = \{1,2,\ldots\}$,
then a stochastic process is a family of random variables on a common probability space.
Time-series analysis usually adds the following statistical assumptions.

1. The process has some stable structure across time,
   at least locally.
2. The observed path is informative about that structure.
3. Forecasts are made from the conditional distribution of future values given the observed past.

The forecast target is therefore not just a point:
$$
P(X_{T+h} \in A \mid X_1 = x_1, \ldots, X_T = x_T)
$$
for horizon $h$ and event set $A$.
Point forecasts,
interval forecasts,
and density forecasts are all summaries of this conditional law.

Examples help fix the idea.

**Example 1: univariate series.**
Daily demand,
temperature,
or latency measurements.
Each time step carries one scalar.

**Example 2: multivariate series.**
At time $t$ we observe
$$
\mathbf{x}_t \in \mathbb{R}^d,
$$
such as a vector of sensor readings,
financial variables,
or service metrics.
The present section is mainly univariate,
but the same intuition extends.

**Example 3: latent-state observation model.**
The observed sequence is only part of the story.
There may be hidden states $\mathbf{z}_t$ driving the evolution.
This leads to state-space models and Kalman filtering.

Two non-examples prevent overgeneralization.

- A batch of iid draws labeled by collection index is not automatically a time series.
- A process with arbitrary path dependence but no repeated structure may be mathematically a stochastic process,
  yet statistically too unconstrained for classical time-series inference.

> **Backward reference:** The full probability-theoretic definition of stochastic processes,
> filtrations,
> and sample paths belongs in [Stochastic Processes](../../06-Probability-Theory/06-Stochastic-Processes/notes.md).
> The canonical scope here is the statistical use of those objects for modeling observed temporal data.

The notation we will use throughout the section follows the repository convention:

- scalar observations: $x_t$
- vector observations: $\mathbf{x}_t$
- white-noise innovations: $\varepsilon_t$
- lag-$h$ autocovariance: $\gamma(h)$
- lag-$h$ autocorrelation: $\rho(h)$
- backshift operator: $L$

The emphasis on notation matters because time-series formulas become much cleaner once lag structure is expressed operatorially.

### 2.2 Lag Operator, Backshift Notation, and Difference Operators

The **lag operator** or **backshift operator** $L$ is defined by
$$
L X_t = X_{t-1}.
$$
Repeated application gives
$$
L^k X_t = X_{t-k}.
$$
This notation makes temporal recursions look like algebraic polynomials.

For example,
an AR(1) model
$$
X_t = \phi X_{t-1} + \varepsilon_t
$$
can be rewritten as
$$
(1 - \phi L) X_t = \varepsilon_t.
$$
An AR(2) model becomes
$$
(1 - \phi_1 L - \phi_2 L^2) X_t = \varepsilon_t.
$$
The polynomial
$$
\phi(L) = 1 - \phi_1 L - \cdots - \phi_p L^p
$$
is called the autoregressive polynomial.

Why is this useful?
Because questions of stability,
invertibility,
and differencing become questions about polynomial roots and algebraic manipulation.
That makes a dynamical model analyzable with tools from algebra and complex numbers rather than by raw recursion alone.

The **first-difference operator** is
$$
\nabla X_t = X_t - X_{t-1} = (1 - L) X_t.
$$
Higher-order differencing is
$$
\nabla^d X_t = (1 - L)^d X_t.
$$
Seasonal differencing at period $s$ is
$$
\nabla_s X_t = X_t - X_{t-s} = (1 - L^s) X_t.
$$

These operators serve distinct roles.

- Backshift notation expresses dependence compactly.
- Ordinary differencing removes low-order trends or unit-root behavior.
- Seasonal differencing removes repeating structure at lag $s$.

Examples are straightforward.

**Example 1: random walk.**
If
$$
X_t = X_{t-1} + \varepsilon_t,
$$
then
$$
(1 - L) X_t = \varepsilon_t.
$$
The random walk becomes white noise after one difference.

**Example 2: deterministic linear trend.**
If
$$
X_t = a + bt + \varepsilon_t,
$$
then
$$
\nabla X_t = b + \varepsilon_t - \varepsilon_{t-1}.
$$
Differencing removes the linear trend from the level but changes the noise structure.

**Example 3: seasonal level.**
If demand repeats every 24 hours,
then
$$
\nabla_{24} X_t = X_t - X_{t-24}
$$
removes a daily repeating baseline.

Non-examples are important here too.

- Differencing is not the same as demeaning.
  Demeaning removes a constant mean.
  Differencing removes persistence in levels.
- Applying too many differences can over-whiten the data,
  injecting unnecessary moving-average structure.

The operator view also clarifies notation for ARIMA models.
An ARIMA$(p,d,q)$ process satisfies
$$
\phi(L) (1-L)^d X_t = \theta(L) \varepsilon_t,
$$
where
$$
\theta(L) = 1 + \theta_1 L + \cdots + \theta_q L^q.
$$
This single equation encodes autoregression,
integration by differencing,
and moving-average dependence.

### 2.3 Mean Function, Autocovariance, and Autocorrelation

The first descriptive object for a time series is its **mean function**:
$$
\mu_t = \mathbb{E}[X_t].
$$
If $\mu_t$ is constant over time,
the process has a stable level.
If $\mu_t$ drifts,
the series carries trend or nonstationary mean behavior.

The second central object is the **autocovariance function**:
$$
\gamma(s,t) = \operatorname{Cov}(X_s, X_t).
$$
For weakly stationary processes this simplifies to a function of lag only:
$$
\gamma(h) = \operatorname{Cov}(X_t, X_{t+h}).
$$

The **autocorrelation function** normalizes by the variance:
$$
\rho(h) = \frac{\gamma(h)}{\gamma(0)}.
$$
This makes lag dependence comparable across scales because $\rho(h)$ is dimensionless and,
for variance-positive processes,
satisfies
$$
-1 \le \rho(h) \le 1.
$$

The conceptual meaning is simple.

- $\rho(1)$ measures one-step persistence.
- Large positive $\rho(h)$ means values $h$ steps apart tend to move together.
- Large negative $\rho(h)$ means oscillatory or alternating behavior.
- Near-zero $\rho(h)$ suggests little linear dependence at that lag.

The autocovariance and autocorrelation summarize only second-order linear structure.
That is both a strength and a limitation.
It is a strength because these summaries are tractable and deeply informative for linear models.
It is a limitation because nonlinear dependence can remain even when autocorrelation is small.

Examples:

**Example 1: white noise.**
If $\{\varepsilon_t\}$ is white noise with variance $\sigma^2$,
then
$$
\gamma(0)=\sigma^2,
\qquad
\gamma(h)=0 \text{ for } h\neq 0.
$$
Hence $\rho(h)=0$ for all nonzero lags.

**Example 2: AR(1).**
For
$$
X_t = \phi X_{t-1} + \varepsilon_t,
\qquad |\phi|<1,
$$
the autocorrelation decays geometrically:
$$
\rho(h)=\phi^{|h|}.
$$
Positive $\phi$ gives smooth persistence.
Negative $\phi$ gives alternating signs.

**Example 3: seasonal process.**
A strong daily cycle in hourly data often yields peaks in $\rho(h)$ around lags 24,
48,
72,
and so on.

Non-examples:

- Zero autocorrelation does not imply independence except in special families such as jointly Gaussian processes.
- A histogram with heavy tails does not by itself reveal temporal dependence;
  autocovariance concerns structure across time,
  not only distribution at one time.

In practice we estimate these objects from one realized sequence.
The sample mean is
$$
\bar{x} = \frac{1}{T} \sum_{t=1}^T x_t.
$$
The sample autocovariance at lag $h$ is often written
$$
\hat{\gamma}(h)
=
\frac{1}{T}
\sum_{t=1}^{T-h}
(x_t - \bar{x})(x_{t+h} - \bar{x}),
$$
or with $T-h$ in the denominator depending on convention.
The sample autocorrelation is
$$
\hat{\rho}(h) = \frac{\hat{\gamma}(h)}{\hat{\gamma}(0)}.
$$

Because time-series data are dependent,
sampling variability in $\hat{\rho}(h)$ is subtler than in iid settings.
Still,
the sample ACF remains one of the fastest ways to visually diagnose whether a series looks white-noise-like,
autoregressive,
seasonal,
or badly nonstationary.

### 2.4 White Noise, Random Walks, and Trend-Plus-Noise Models

The fastest way to build intuition for time-series modeling is to contrast three canonical objects:
white noise,
the random walk,
and a trend-plus-noise model.
They appear similar in short windows,
but they encode radically different dependence structures and forecasting logic.

**White noise** is the baseline innovation process.
We write
$$
\varepsilon_t \sim WN(0,\sigma^2)
$$
to mean
$$
\mathbb{E}[\varepsilon_t] = 0,
\qquad
\operatorname{Var}(\varepsilon_t)=\sigma^2,
\qquad
\operatorname{Cov}(\varepsilon_t,\varepsilon_s)=0 \text{ for } t\neq s.
$$
White noise is unpredictable from its own past in the linear second-order sense.
It is the ingredient we want residuals to resemble after a good model has explained the serial structure.

**Random walk** behavior is the opposite.
If
$$
X_t = X_{t-1} + \varepsilon_t,
$$
then shocks accumulate permanently into the level.
The variance grows with time,
the series wanders,
and nearby observations are strongly dependent.
Forecasts of a random walk are persistent because today's level is the best predictor of tomorrow's level,
but uncertainty spreads outward with horizon.

**Trend-plus-noise** models add deterministic structure:
$$
X_t = m_t + \varepsilon_t,
$$
where $m_t$ may be a constant,
linear trend,
piecewise trend,
or seasonal baseline.
The level changes over time,
but the nonstationarity comes from a deterministic mean function rather than from unit-root accumulation.

These models can look deceptively similar in finite data.
A trending series and a random walk can both move upward.
A random walk and a highly persistent AR(1) can both look smooth.
White noise with one large outlier can look like a regime shift.
Diagnostics exist to tell these cases apart.

The autocorrelation patterns differ sharply.

- White noise:
  $\rho(h)\approx 0$ for all nonzero lags.
- Random walk:
  autocorrelations remain high across many lags because levels share accumulated shocks.
- Trend-plus-noise:
  sample ACF often decays slowly,
  but much of that can vanish after detrending.

Forecast behavior also differs.

- White noise:
  the forecast is the mean,
  usually zero after centering.
- Random walk:
  the $h$-step forecast is the current level,
  with forecast variance growing roughly linearly in $h$.
- Trend-plus-noise:
  the forecast extrapolates the trend component,
  assuming that deterministic structure persists.

Examples:

**Example 1: sensor jitter.**
High-frequency noise around a stable temperature is often modeled as white-noise-like after removing the physical dynamics.

**Example 2: price or cumulative counts.**
A cumulative counter or stock-price-like level often resembles a random walk locally because shocks shift the level persistently.

**Example 3: website traffic with growth.**
A product rollout can create a linear or nonlinear trend on top of short-term noise,
which is more naturally trend-plus-noise than random walk.

Non-examples:

- White noise is not the same as Gaussian noise.
  Gaussianity is about the marginal distribution.
  White noise is about zero mean,
  constant variance,
  and zero autocovariance.
- A random walk is not stationary just because its increments are stationary.
  The levels themselves have time-varying variance.

These distinctions matter for modeling choices.
If a series is random-walk-like,
differencing may stabilize it.
If it is trend-plus-noise,
regression or decomposition may be more interpretable.
If it is already close to white noise,
there may be little forecastable structure left.

### 2.5 Stationarity: Strict vs Weak

Stationarity is the mathematical bargain that lets us learn from a single path.
It says,
in one form or another,
that the probabilistic structure does not change when we shift time.
Without some version of that assumption,
the future could obey different rules from the past,
and historical estimation becomes fragile or meaningless.

There are two standard notions.

**Strict stationarity** requires that for any integer shift $h$ and any collection of times
$t_1,\ldots,t_k$,
the joint distribution of
$$
(X_{t_1},\ldots,X_{t_k})
$$
is identical to that of
$$
(X_{t_1+h},\ldots,X_{t_k+h}).
$$
This is a full distributional invariance statement.

**Weak stationarity** or **covariance stationarity** requires only:

1. $\mathbb{E}[X_t] = \mu$ is constant,
2. $\operatorname{Var}(X_t)=\gamma(0)<\infty$ is constant,
3. $\operatorname{Cov}(X_t,X_{t+h}) = \gamma(h)$ depends only on lag $h$.

Weak stationarity is the workhorse notion for classical linear time-series analysis because AR,
MA,
ARMA,
and spectral methods are fundamentally second-order theories.

The relationship between the two notions is subtle.

- Strict stationarity does **not** automatically imply weak stationarity unless the second moment exists.
- Weak stationarity does **not** imply strict stationarity in general.
- For Gaussian processes,
  weak stationarity and strict stationarity coincide because Gaussian laws are determined by first and second moments.

Examples:

**Example 1: white noise.**
An iid Gaussian white-noise process is both strictly and weakly stationary.

**Example 2: stationary AR(1).**
If
$$
X_t = \phi X_{t-1} + \varepsilon_t,
\qquad |\phi|<1,
$$
with white-noise innovations,
then the process has constant mean and variance and is weakly stationary.
If the innovations are Gaussian,
the process is also strictly stationary.

**Example 3: Cauchy iid sequence.**
An iid Cauchy sequence is strictly stationary,
but weak stationarity is undefined because the variance does not exist.

Non-examples:

- A random walk is not weakly stationary because its variance depends on $t$.
- A deterministic trend plus noise is not stationary in levels because the mean changes over time.

Why do practitioners care so much about stationarity?

1. Sample ACF estimates are interpretable only when lag structure is stable.
2. ARMA theory assumes time-invariant dependence.
3. Forecast formulas and error propagation simplify dramatically.
4. Many asymptotic results rely on stable dependence rather than arbitrary drift.

> **Backward reference:** The abstract process-level definition of stationarity and the Wiener-Khinchin theorem were covered in [Stochastic Processes](../../06-Probability-Theory/06-Stochastic-Processes/notes.md).
> Here the focus is practical:
> how stationarity governs diagnostics,
> model fitting,
> and forecast validity.

In practice,
few real-world series are perfectly stationary in raw form.
Instead,
we aim for one of three strategies.

- Transform the series to approximate stationarity through differencing,
  detrending,
  or seasonal adjustment.
- Model nonstationary structure explicitly through ARIMA or state-space components.
- Accept local rather than global stationarity when the system evolves slowly.

The rest of the chapter largely asks:
how close can we get to a stationary representation that preserves the useful predictive structure?

---

## 3. Core Theory I: Dependence, Stationarity, and Diagnostics

### 3.1 Why Stationarity Matters

Stationarity is not a philosophical preference for tidy mathematics.
It is the condition that lets us estimate future-relevant structure from past data with finite samples.
If the dependence law changes arbitrarily over time,
then a correlation observed at lag 3 in January may say nothing about lag 3 in February.
A single realized path would be too heterogeneous to summarize with stable parameters.

To see why,
consider the sample autocorrelation estimator.
It pools products
$(x_t-\bar{x})(x_{t+h}-\bar{x})$
across time and treats them as estimating one common lag-$h$ quantity.
That only makes sense if lag-$h$ dependence is approximately invariant across $t$.
Weak stationarity provides exactly that invariance.

Forecasting makes the same bargain.
When we fit an AR(1) model and estimate a coefficient $\phi$,
we are assuming that the relationship between present and past is stable enough for tomorrow to obey roughly the same coefficient.
Without that assumption,
model parameters become historical artifacts rather than forecast tools.

Stationarity also clarifies what a "shock" means.
In a stationary process,
an innovation perturbs the system relative to a stable background law.
In a nonstationary process,
it can be hard to tell whether a large movement reflects noise,
trend,
seasonality,
policy intervention,
or a structural break.

Examples of why the assumption matters:

**Example 1: monitoring.**
If a service's latency distribution is stable apart from transient fluctuations,
then large deviations can be flagged against a stationary baseline.
If the service is migrating traffic and hardware every hour,
the baseline itself moves,
and anomaly logic must adapt.

**Example 2: macro forecasting.**
An ARIMA model for inflation or unemployment assumes some stable dynamics after differencing or detrending.
Regime changes,
policy shocks,
or crises violate that assumption and cause forecast breakdown.

**Example 3: continual-learning telemetry.**
Model drift metrics can exhibit changing mean and variance as the deployment population shifts.
Treating that as stationary noise leads to false assurance.

Weak stationarity is often enough for linear methods,
but strong forms are not always necessary.
The practical attitude is:
assume only enough stationarity to make the target summary meaningful.
For short-horizon forecasting,
local stability may suffice.
For long-run theoretical properties,
we often need more.

A useful diagnostic perspective is:
stationarity is not one yes/no property but a modeling approximation.
We ask whether the series is stationary enough for the tool we plan to use.
ARMA estimation,
seasonal adjustment,
and spectral analysis each tolerate different departures from ideal assumptions.

### 3.2 Autocorrelation Function (ACF)

The **autocorrelation function** is the first major diagnostic in classical time-series analysis.
It is defined for a weakly stationary process by
$$
\rho(h)=\frac{\gamma(h)}{\gamma(0)},
\qquad h \in \mathbb{Z}.
$$
Because $\gamma(0)$ is the variance,
the ACF measures serial dependence on a standardized scale.

The ACF is symmetric:
$$
\rho(-h)=\rho(h).
$$
Its value at lag zero is always
$$
\rho(0)=1.
$$
The remaining pattern tells us what type of persistence the series may contain.

The sample ACF is computed from observed data,
typically up to some maximum lag $H$.
A bar plot of $\hat{\rho}(1),\ldots,\hat{\rho}(H)$ is one of the fastest diagnostics in the subject.

What shapes should we look for?

1. **White noise.**
   Sample autocorrelations fluctuate near zero with no coherent decay.
2. **AR-type persistence.**
   The ACF decays gradually,
   often geometrically or as a damped oscillation.
3. **MA-type dependence.**
   The ACF cuts off after a small number of lags in the ideal population model.
4. **Seasonality.**
   Repeating peaks appear at seasonal lags.
5. **Nonstationarity.**
   The ACF stays near one for many lags,
   often indicating a trend or unit root in levels.

These signatures are heuristic,
not mechanical.
Finite samples blur the ideal patterns.
Still,
the ACF remains central because it compresses a large amount of structure into one plot.

Theoretical examples:

**White noise.**
$$
\rho(h)=0 \quad \text{for } h\neq 0.
$$

**AR(1).**
$$
\rho(h)=\phi^{|h|}.
$$
If $\phi=0.8$,
the decay is smooth and slow.
If $\phi=-0.8$,
the decay alternates sign.

**MA(1).**
If
$$
X_t = \varepsilon_t + \theta \varepsilon_{t-1},
$$
then
$$
\rho(1)=\frac{\theta}{1+\theta^2},
\qquad
\rho(h)=0 \text{ for } |h|>1.
$$

The sample ACF also supports rough significance bands under white-noise heuristics,
often drawn near
$$
\pm \frac{1.96}{\sqrt{T}}.
$$
These are not universal truth statements.
They are approximate guides for whether observed spikes are larger than one would expect from finite-sample randomness under no serial correlation.

Common mistakes with the ACF:

- Reading every small spike as meaningful.
- Forgetting that multiple lags are being inspected simultaneously.
- Using the ACF of a strongly trending series without first checking for nonstationarity.
- Treating the sample ACF as a proof of model order rather than one clue among several.

```text
ACF SHAPE HEURISTICS
=====================================================================

White noise:
  lag:  0 1 2 3 4 5
  acf:  1 . . . . .

AR-type persistence:
  lag:  0 1 2 3 4 5
  acf:  1 | | | . .
        gradual decay

MA(q):
  lag:  0 1 2 3 4 5
  acf:  1 | | . . .
        cutoff after q

Seasonality:
  lag:  0 . . s . . 2s
  acf:  1 . . | . . |

=====================================================================
```

The ACF is therefore a descriptive object,
an identification clue,
and a residual diagnostic all at once.

### 3.3 Partial Autocorrelation Function (PACF)

The **partial autocorrelation function** asks a more surgical question than the ACF.
At lag $h$,
instead of measuring the raw correlation between $X_t$ and $X_{t-h}$,
it measures the correlation that remains after removing the linear effects of the intervening lags
$X_{t-1},\ldots,X_{t-h+1}$.

Intuitively,
the PACF isolates **direct** lag influence.
If lag 3 appears correlated with the present only because lag 1 and lag 2 transmit the dependence,
the PACF at lag 3 may be small even when the ACF at lag 3 is not.

One way to define the PACF is as the last coefficient in the best linear predictor of $X_t$ from the previous $h$ lags:
$$
X_t = \phi_{h1} X_{t-1} + \cdots + \phi_{hh} X_{t-h} + u_t.
$$
Then the PACF at lag $h$ is
$$
\alpha(h)=\phi_{hh}.
$$

This makes the PACF especially useful for identifying autoregressive order.
For an ideal AR($p$) process,
the PACF cuts off after lag $p$.
That means:

- AR(1):
  strong PACF spike at lag 1,
  near zero afterward.
- AR(2):
  strong PACF spikes at lags 1 and 2,
  near zero afterward.

In contrast,
for an ideal MA($q$) process,
the PACF tends to decay gradually rather than cut off sharply.
This complementary behavior is one of the most famous heuristics in Box-Jenkins modeling.

Examples:

**Example 1: AR(1).**
The ACF decays geometrically,
but the PACF shows a single dominant lag-1 effect.

**Example 2: AR(2) with oscillation.**
The ACF may show a damped wave pattern,
but the PACF highlights that only the first two lags are directly needed in the linear recursion.

**Example 3: seasonal AR.**
A seasonal autoregressive component can create PACF spikes at seasonal lags,
such as 24 in hourly data.

Non-examples:

- A PACF cutoff in sample data is rarely exact.
  Noise,
  finite samples,
  and model misspecification blur the pattern.
- PACF does not detect all nonlinear dependence.
  It is a linear-diagnostic tool,
  not a universal dependence detector.

An intuitive geometric picture helps.
The ACF measures the angle between the present and a lagged copy.
The PACF measures the angle after projecting out the shorter-lag subspace.
That is why it reveals which lag contributes new predictive information.

```text
ACF VS PACF
=====================================================================

ACF at lag h:
  "How correlated are X_t and X_{t-h}?"

PACF at lag h:
  "After explaining X_t with lags 1,...,h-1,
   does lag h still add anything directly?"

Typical rule of thumb:
  AR(p)  -> PACF cuts off, ACF decays
  MA(q)  -> ACF cuts off, PACF decays

=====================================================================
```

The PACF is therefore less intuitive at first glance than the ACF,
but more informative about direct linear order.
Used together,
they provide the main visual grammar of classical model identification.

### 3.4 Differencing, Unit Roots, and Random Walk Behavior

One of the most common pathologies in raw time-series data is slow decay in the sample ACF.
That pattern often signals nonstationarity in levels,
especially trend or unit-root behavior.
Differencing is the standard first response,
but it should be applied with thought rather than ritual.

The simplest unit-root model is the random walk:
$$
X_t = X_{t-1} + \varepsilon_t.
$$
Using the lag operator,
$$
(1-L)X_t = \varepsilon_t.
$$
This shows immediately that one difference converts the random walk into white noise.

More generally,
an integrated process of order $d$ becomes stationary after $d$ ordinary differences.
This is the "I" in ARIMA.
If a seasonal pattern repeats every $s$ steps,
seasonal differencing with $(1-L^s)$ can remove that component.

The unit-root idea is important because it changes the effect of shocks.

- In a stationary AR(1),
  shocks decay over time.
- In a random walk,
  shocks persist forever in the level.

That distinction is operationally huge.
If a demand series receives a one-time shock and then reverts,
the system is stationary around a stable mean.
If the shock shifts the level permanently,
the system is better modeled as integrated or regime-shifting.

Examples:

**Example 1: cumulative counts.**
Cumulative user signups naturally behave like an integrated series.
First differences recover arrivals per period,
which are much closer to stationary.

**Example 2: price-like processes.**
Levels often look random-walk-like,
while returns
$$
\nabla X_t = X_t - X_{t-1}
$$
are much closer to stationary.

**Example 3: seasonal demand.**
Hourly load can become more stationary after seasonal differencing at lag 24,
because daily recurring patterns are removed.

Non-examples:

- Differencing is not a magic cure.
  It can remove meaningful long-run information.
- A trend-stationary process may be better handled by detrending than by repeated differencing.

This distinction matters.
A **difference-stationary** process becomes stationary only after differencing.
A **trend-stationary** process becomes stationary after removing a deterministic trend.
Both can show slow ACF decay in levels,
but their long-run interpretations differ.

Unit-root tests such as the Augmented Dickey-Fuller test and stationarity checks such as KPSS belong to the broader statistical toolbox,
but the core conceptual point is simpler:
before fitting a stationary model,
we must ask whether the observed persistence reflects genuine stable dependence or unresolved nonstationarity.

### 3.5 Wold Decomposition Intuition

The Wold decomposition is the theorem that makes linear stationary modeling feel inevitable rather than ad hoc.
In one common form,
it says that every purely nondeterministic covariance-stationary process can be written as an infinite moving average of orthogonal innovations:
$$
X_t = \mu + \sum_{j=0}^{\infty} \psi_j \varepsilon_{t-j},
\qquad
\psi_0 = 1,
$$
where $\{\varepsilon_t\}$ is white noise and the coefficients are square summable.

This is a deep structural statement.
It says that if a process is stationary and not hiding a perfectly predictable deterministic component,
then it can be represented as filtered shock accumulation.
Every observation is a weighted sum of present and past innovations.

Why does this matter?
Because it explains why MA,
AR,
ARMA,
and state-space models are not arbitrary modeling families.
They are tractable approximations to the general innovation-filtering structure guaranteed by Wold's theorem.

The theorem also clarifies the meaning of the innovation sequence.
An innovation is not simply noise in the casual sense.
It is the part of $X_t$ that cannot be linearly predicted from the infinite past.
If a fitted model is good,
its residuals should behave like innovations.

Examples:

**Example 1: AR(1).**
Even though the model is written recursively,
it has the infinite-MA representation
$$
X_t = \varepsilon_t + \phi \varepsilon_{t-1} + \phi^2 \varepsilon_{t-2} + \cdots
$$
when $|\phi|<1$.

**Example 2: ARMA models.**
Finite-order ARMA models are special cases of Wold-style linear innovation filters with rational transfer functions.

**Example 3: state-space models.**
A stable linear state-space system also implies an innovation representation once the hidden state is integrated out.

Non-examples:

- Wold decomposition is not a statement about arbitrary nonstationary processes.
- It does not imply every process is finite-order AR or MA;
  only that a stationary process has an infinite linear innovation representation.

> **Backward reference:** The fully general process-level statement belongs with stationary stochastic processes in [Stochastic Processes](../../06-Probability-Theory/06-Stochastic-Processes/notes.md).
> Here we use Wold as intuition for why classical linear time-series models are rich enough to matter and structured enough to estimate.

The practitioner-level lesson is elegant:
if a stationary series can be interpreted as filtered shocks,
then our job is to find a compact description of that filter.
AR,
MA,
ARMA,
ARIMA,
seasonal variants,
and state-space models are all answers to that one question.

---

## 4. Core Theory II: Linear Time-Series Models

### 4.1 Autoregressive Models AR(p)

An **autoregressive model of order $p$** writes the present as a linear function of the past $p$ observations plus an innovation:
$$
X_t = c + \phi_1 X_{t-1} + \cdots + \phi_p X_{t-p} + \varepsilon_t,
$$
where $\{\varepsilon_t\}$ is white noise.
Using the lag operator,
$$
\phi(L)X_t = c + \varepsilon_t,
$$
with
$$
\phi(L)=1-\phi_1L-\cdots-\phi_pL^p.
$$

The AR model is the simplest direct encoding of persistence.
If the recent past contains the relevant state information,
then a linear combination of lags can forecast the next value.

The AR(1) case already reveals the key idea:
$$
X_t = c + \phi X_{t-1} + \varepsilon_t.
$$
If $|\phi|<1$,
the process is stationary around mean
$$
\mu = \frac{c}{1-\phi},
$$
with variance
$$
\operatorname{Var}(X_t)=\frac{\sigma^2}{1-\phi^2}
$$
when the innovation variance is $\sigma^2$.
If $\phi$ is positive and close to one,
the series is highly persistent.
If $\phi$ is negative,
the series alternates.
If $\phi=1$,
the model becomes a random walk with drift,
which is no longer stationary.

The higher-order AR($p$) model generalizes this idea.
Its stability is determined by the roots of the characteristic polynomial
$$
1-\phi_1 z-\cdots-\phi_p z^p=0.
$$
The stationary solution exists when all roots lie outside the unit circle.
This root condition is the multivariate analogue of $|\phi|<1$.

Examples:

**Example 1: AR(1) demand persistence.**
Traffic at minute $t$ depends mostly on traffic at minute $t-1$ plus new shock.

**Example 2: AR(2) oscillation.**
An AR(2) can encode damped cyclical behavior that a one-lag model cannot.

**Example 3: mean reversion.**
An AR(1) with $0<\phi<1$ and constant mean captures the idea that unusually high values tend to drift back toward baseline.

Non-examples:

- An AR model is not appropriate when dependence comes mostly from past shocks rather than past levels.
- Strong unresolved seasonality or structural breaks can make a simple AR fit misleading.

The ACF/PACF behavior provides intuition.
For an AR($p$),
the ACF usually decays,
while the PACF cuts off after lag $p$ in the ideal population model.
This is why PACF spikes are classically associated with autoregressive order.

One practical interpretation of AR models is that they estimate a compressed state:
the past $p$ observations summarize everything relevant for the next prediction.
That is a crude but often effective approximation when the underlying system is smooth and low-dimensional.

### 4.2 Moving-Average Models MA(q)

A **moving-average model of order $q$** writes the present as a finite linear combination of the current and past $q$ innovations:
$$
X_t = \mu + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \cdots + \theta_q \varepsilon_{t-q}.
$$
In lag notation,
$$
X_t = \mu + \theta(L)\varepsilon_t,
$$
where
$$
\theta(L)=1+\theta_1L+\cdots+\theta_qL^q.
$$

The word "moving average" is historically standard but conceptually slightly misleading.
An MA model is not literally a rolling average of observations.
It is a finite filter of shocks.
Current observations inherit dependence because recent innovations still echo into the present.

This makes MA models natural when a shock affects the next few periods and then disappears.
Measurement error,
short-lived intervention effects,
and certain differenced deterministic structures often create this pattern.

The MA(1) case is
$$
X_t = \mu + \varepsilon_t + \theta \varepsilon_{t-1}.
$$
Its ACF has a simple signature:
nonzero at lag 1,
zero beyond lag 1 in the ideal population model.
This "cutoff" behavior is the mirror image of AR models,
whose PACF rather than ACF cuts off.

Invertibility plays the role for MA models that stability plays for AR models.
We usually require the roots of
$$
\theta(z)=1+\theta_1 z+\cdots+\theta_q z^q
$$
to lie outside the unit circle.
This ensures the model has a unique equivalent infinite-AR representation.

Examples:

**Example 1: one-step shock persistence.**
If a sensor glitch affects two adjacent measurements but not beyond,
an MA(1)-like structure may appear.

**Example 2: differenced signal.**
Differencing a trend-plus-noise series often creates short-range MA structure even if the original series did not look like an MA model directly.

**Example 3: smoothing filters.**
Some preprocessed measurement systems induce dependence that is more naturally interpreted as filtered innovations than as lagged state persistence.

Non-examples:

- An MA model is not ideal when dependence persists over many lags through the level itself;
  AR structure may be more compact.
- MA terminology should not be confused with moving averages used in descriptive dashboards.

The conceptual split between AR and MA models is worth remembering:
AR uses past **observations** as predictors;
MA uses past **shocks**.
That difference becomes powerful when we combine them.

### 4.3 ARMA Models

An **ARMA($p,q$)** model combines autoregressive and moving-average structure:
$$
\phi(L) X_t = c + \theta(L)\varepsilon_t.
$$
This is the classical workhorse for stationary linear time series because it allows both persistent state dynamics and short-run shock filtering.

Why combine them?
Because many real series display both behaviors at once.
A system may have inertia,
so current levels depend on recent levels,
while measurement or intervention effects also create temporary shock echoes.
ARMA models can represent this far more parsimoniously than a pure high-order AR or a pure high-order MA.

Stationarity still requires the AR polynomial roots to lie outside the unit circle.
Invertibility still requires the MA polynomial roots to lie outside the unit circle.
Under these conditions,
the model has both infinite-MA and infinite-AR representations.

The ACF and PACF of an ARMA model typically both decay rather than cut off sharply.
That is one reason order identification is harder for ARMA than for pure AR or pure MA models.
Visual heuristics remain useful,
but information criteria and residual diagnostics become more important.

Examples:

**Example 1: demand with inertia and shock echo.**
Demand today depends on demand yesterday,
but a system event also perturbs the next couple of observations.

**Example 2: smoothed control loop.**
A process with lagged physical dynamics plus filtered measurement noise naturally mixes AR and MA effects.

**Example 3: differenced macro data.**
After differencing to remove nonstationarity,
remaining persistence often fits naturally into an ARMA structure.

Non-examples:

- ARMA is not appropriate in raw form for clearly nonstationary level series.
- ARMA does not natively capture large seasonal cycles unless seasonal terms are added.

ARMA is the stationary core of the ARIMA family.
Conceptually,
it is the point where linear time-series analysis becomes flexible enough for many applications while remaining analyzable by ACF,
PACF,
roots,
innovations,
and likelihood-based estimation.

### 4.4 ARIMA Models

An **ARIMA($p,d,q$)** model extends ARMA to series that become approximately stationary after differencing:
$$
\phi(L)(1-L)^d X_t = c + \theta(L)\varepsilon_t.
$$
The parameter $d$ counts the number of ordinary differences required to stabilize the level behavior.

This is one of the most important conceptual moves in applied statistics.
Instead of forcing raw nonstationary data into a stationary model,
we transform the data into a scale where stationary dynamics make sense.

The three components have distinct meanings:

- $p$: autoregressive memory in the stationary transformed series
- $d$: integration order or number of differences
- $q$: moving-average shock structure in the stationary transformed series

ARIMA modeling is often taught procedurally,
but the structure is logical.

1. Diagnose whether the level series is plausibly stationary.
2. Difference if needed.
3. Model the transformed series with ARMA logic.
4. Translate forecasts back to the original level scale.

Examples:

**ARIMA(0,1,0).**
This is a random walk.

**ARIMA(1,1,0).**
The differenced series follows AR(1),
so changes persist for a while even after removing the unit-root component.

**ARIMA(0,1,1).**
The differenced series is MA(1),
which is closely related to exponential-smoothing behavior in some settings.

Non-examples:

- If deterministic trend explains the nonstationarity better than a stochastic unit root,
  blind differencing may hide interpretable structure.
- If multiple seasonalities dominate,
  a nonseasonal ARIMA can be structurally inadequate.

ARIMA succeeds because it respects two truths at once:
many real series are not stationary in raw levels,
and many become statistically manageable after a small number of transformations.
It is therefore less a single model than a disciplined workflow for turning raw persistence into stationary innovation dynamics.

### 4.5 Model Identification with ACF/PACF and Information Criteria

Box-Jenkins modeling is often summarized as
**identify -> estimate -> diagnose**.
Identification is the first stage:
choosing plausible model orders before fitting in detail.
The ACF and PACF provide the visual heuristics,
and information criteria provide the statistical tie-breakers.

The classical heuristic map is:

- AR($p$):
  PACF cuts off,
  ACF decays.
- MA($q$):
  ACF cuts off,
  PACF decays.
- ARMA($p,q$):
  both decay.
- Nonstationary levels:
  ACF decays very slowly,
  suggesting differencing.
- Seasonal structure:
  spikes repeat at seasonal lags.

These heuristics are not exact model-selection rules.
Finite samples,
measurement noise,
missing data,
and misspecification blur ideal patterns.
Still,
they remain valuable because they make model identification interpretable rather than purely brute-force.

Information criteria formalize the complexity tradeoff.
Two standard choices are:
$$
\operatorname{AIC} = -2 \log \hat{L} + 2k,
$$
$$
\operatorname{BIC} = -2 \log \hat{L} + k \log T,
$$
where $\hat{L}$ is the maximized likelihood,
$k$ is the number of estimated parameters,
and $T$ is sample size.

Lower values indicate a better tradeoff between fit and complexity.
AIC tends to prefer more flexible models.
BIC penalizes complexity more strongly and often favors more parsimonious orders.

Residual diagnostics complete the identification loop.
After fitting a candidate model,
we check whether residuals behave like white noise:

- no obvious autocorrelation,
- approximately stable variance,
- no unmodeled seasonal spikes,
- no glaring structural change.

If residual structure remains,
the model has not extracted the dependence pattern adequately.

This workflow remains useful even in the era of automated model search.
Automated systems can propose orders quickly,
but human understanding of the ACF,
PACF,
seasonality,
and residual failures remains the fastest way to detect when the automatic answer is structurally silly.

---

## 5. Core Theory III: Forecasting, Seasonality, and State Space

### 5.1 One-Step and Multi-Step Forecasting

The point of fitting a time-series model is rarely the fitted line itself.
It is the forecast distribution implied by that model.
At horizon $h$,
the target object is
$$
P(X_{T+h} \mid X_1=x_1,\ldots,X_T=x_T).
$$
Point forecasts,
interval forecasts,
and density forecasts are all summaries of this conditional law.

The **one-step-ahead forecast** uses information up to time $T$ to predict $X_{T+1}$.
In many linear models,
this forecast is relatively accurate because the most recent lag structure is directly observed.
The **multi-step forecast** predicts $X_{T+h}$ for $h>1$,
which requires recursively forecasting future intermediate states.
That compounding step is where uncertainty grows.

For an AR(1),
$$
X_t = c + \phi X_{t-1} + \varepsilon_t,
$$
the one-step conditional mean forecast is
$$
\hat{X}_{T+1\mid T} = c + \phi X_T.
$$
The two-step forecast substitutes the one-step forecast into itself:
$$
\hat{X}_{T+2\mid T} = c + \phi \hat{X}_{T+1\mid T}.
$$
Continuing yields
$$
\hat{X}_{T+h\mid T}
=
\mu + \phi^h (X_T-\mu),
$$
where $\mu = c/(1-\phi)$ for $|\phi|<1$.

This formula reveals the key behavior.
As $h$ grows,
the forecast mean reverts toward the long-run mean $\mu$,
and the forecast variance approaches the unconditional variance.
Short-horizon predictions remember the current state.
Long-horizon predictions forget it.

Different model classes imply different forecast logic.

- White noise:
  all future forecasts equal the mean.
- Random walk:
  all future point forecasts equal the current level,
  but variance grows with horizon.
- Stationary AR:
  forecasts decay toward the mean.
- Seasonal models:
  forecasts preserve periodic structure.
- State-space models:
  forecasts propagate a latent state and its covariance forward.

Forecast error growth is the other half of the story.
If the model is correctly specified,
the point forecast may be unbiased,
but the uncertainty interval should widen with horizon because more unobserved innovations contribute.
Operationally,
this means a forecast is not just a scalar.
It is a time-indexed uncertainty object.

Examples:

**Example 1: inventory planning.**
Tomorrow's demand forecast can be sharp enough for replenishment.
A four-week forecast needs much wider safety bands.

**Example 2: GPU capacity.**
Near-term load forecasts support autoscaling.
Long-horizon forecasts support planning,
but must acknowledge broad uncertainty.

**Example 3: robot tracking.**
A one-step physical forecast is often tight.
A long-horizon open-loop forecast without new measurements rapidly diffuses.

The practitioner lesson is simple:
good forecasting means modeling both the conditional mean path and the horizon-dependent uncertainty growth.

### 5.2 Seasonality and Seasonal Decomposition

Many series are shaped less by short-run persistence than by repeated periodic structure.
Energy load repeats daily and weekly.
Retail demand repeats weekly and yearly.
User activity repeats by hour-of-day and day-of-week.
This repeated structure is called **seasonality** even when the period is not literally a season.

A common decomposition writes
$$
X_t = T_t + S_t + R_t
$$
for additive trend $T_t$,
seasonal component $S_t$,
and remainder $R_t$,
or
$$
X_t = T_t S_t R_t
$$
for multiplicative effects when amplitude scales with level.

The decomposition is not a theorem about the world.
It is a useful descriptive model that separates long-run movement,
regular repetition,
and irregular leftovers.
This often makes both diagnostics and communication easier.

The simplest seasonal signature in the ACF is repeated spikes at lags
$s,2s,3s,\ldots$,
where $s$ is the seasonal period.
For hourly data,
$s=24$ captures daily repetition.
For daily retail data,
$s=7$ captures weekly cycles.

Seasonality can be handled in several ways.

1. Seasonal dummy variables or harmonic regressors
2. Seasonal differencing
3. Seasonal AR or MA terms
4. Explicit state-space seasonal components
5. Learned representations in modern deep models

Classical decomposition remains valuable because it answers a human question:
what part of this series is repeatable structure,
what part is trend,
and what part is surprise?

Examples:

**Example 1: call-center volume.**
Daily peaks and troughs dominate the raw series.
Removing seasonality reveals whether the underlying demand baseline is rising.

**Example 2: server traffic.**
Weekday/weekend effects can mask incidents unless the expected weekly pattern is separated from residual anomalies.

**Example 3: token usage.**
Enterprise API usage often exhibits strong business-hour cycles.
Ignoring seasonality makes normal noon spikes look anomalous.

Non-examples:

- Not every repeating bump is stable seasonality;
  some patterns come from temporary campaigns or interventions.
- A single seasonal period may be insufficient for data with multiple cycles,
  such as hourly-within-weekly structure.

Seasonal decomposition is therefore best read as a structured lens,
not a guarantee of truth.
It helps reveal what kind of repeated temporal geometry the model must capture.

### 5.3 SARIMA and Seasonal Structure

When seasonality is strong and recurring,
the ARIMA family extends naturally to include seasonal dependence.
A common notation is
$$
\operatorname{SARIMA}(p,d,q)\times(P,D,Q)_s,
$$
where $s$ is the seasonal period.

The model combines:

- nonseasonal AR order $p$
- ordinary differencing order $d$
- nonseasonal MA order $q$
- seasonal AR order $P$
- seasonal differencing order $D$
- seasonal MA order $Q$
- seasonal period $s$

In lag-operator form,
a SARIMA model is often written as
$$
\Phi(L^s)\phi(L)(1-L)^d (1-L^s)^D X_t
=
\Theta(L^s)\theta(L)\varepsilon_t.
$$

The beauty of this notation is that seasonality becomes algebraically parallel to ordinary lag structure.
A lag of 24 in hourly data is treated much like lag 1,
but at a different scale.

Seasonal differencing,
$$
(1-L^s)X_t = X_t - X_{t-s},
$$
removes repeating level effects.
Seasonal AR terms capture persistent dependence across full cycles.
Seasonal MA terms capture shock echo across cycles.

Examples:

**Example 1: hourly energy load.**
Daily seasonality with period $s=24$ often motivates seasonal differencing or seasonal AR terms.

**Example 2: weekly retail demand.**
Daily data with a weekly cycle suggest $s=7$.

**Example 3: monthly macro series.**
Annual seasonality in monthly data suggests $s=12$.

Non-examples:

- Multiple incommensurate seasonalities,
  such as hourly and weekly together,
  can strain a simple SARIMA.
- Learned calendar effects and external covariates may explain structure more transparently than pure seasonal lag terms.

Still,
SARIMA remains one of the most useful middle-ground models in practice:
richer than nonseasonal ARIMA,
lighter than full modern representation-learning pipelines,
and often surprisingly hard to beat on structured tabular forecasting tasks.

### 5.4 State-Space Models

ARIMA and related linear models describe dependence directly in the observed series.
**State-space models** instead describe an evolving hidden state that generates observations.
The standard linear-Gaussian form is
$$
\mathbf{z}_t = A \mathbf{z}_{t-1} + \mathbf{w}_t,
$$
$$
\mathbf{x}_t = C \mathbf{z}_t + \mathbf{v}_t,
$$
where $\mathbf{z}_t$ is the latent state,
$A$ is the transition matrix,
$C$ is the observation matrix,
and $\mathbf{w}_t,\mathbf{v}_t$ are process and observation noise terms.

This representation is powerful because it unifies many phenomena:

- local level models
- trend models
- seasonal models
- tracking systems
- linear control systems
- hidden-state views of time-series dynamics

The hidden state can encode position and velocity,
trend slope,
seasonal phase,
or other low-dimensional summaries of the past.
Instead of regressing directly on many observed lags,
we infer a compact latent state that evolves predictably.

Why is this such a useful abstraction?

1. It separates latent dynamics from noisy measurements.
2. It naturally handles missing observations.
3. It yields recursive estimation via the Kalman filter.
4. It generalizes to nonlinear and non-Gaussian settings.

Examples:

**Example 1: tracking position and velocity.**
Position is observed noisily,
but velocity is hidden.
The state evolves linearly under a simple motion model.

**Example 2: local level plus seasonal state.**
A demand series can be written as a latent level component plus latent seasonal factors.

**Example 3: sensor fusion.**
Different measurement streams can be linked to the same hidden physical state.

Non-examples:

- If there is no meaningful latent compression and the direct lag structure is already transparent,
  an ARIMA model may be simpler.
- State-space models are not automatically nonlinear deep sequence models;
  they are a classical probabilistic framework that modern architectures sometimes generalize.

One important conceptual bridge to the previous section is that linear-Gaussian state-space inference is repeated Bayesian updating.
At each time step,
we propagate a Gaussian prior through the dynamics and correct it with the new observation.
That recursive logic is the Kalman filter.

### 5.5 Kalman Filter

The **Kalman filter** is the optimal recursive estimator for linear dynamical systems with Gaussian noise.
If the hidden state and observation model are
$$
\mathbf{z}_t = A\mathbf{z}_{t-1} + \mathbf{w}_t,
\qquad
\mathbf{w}_t \sim \mathcal{N}(\mathbf{0},Q),
$$
$$
\mathbf{x}_t = C\mathbf{z}_t + \mathbf{v}_t,
\qquad
\mathbf{v}_t \sim \mathcal{N}(\mathbf{0},R),
$$
then the posterior distribution of the state remains Gaussian at every step.

The filter alternates between two stages.

**Predict step**
$$
\hat{\mathbf{z}}_{t \mid t-1} = A \hat{\mathbf{z}}_{t-1 \mid t-1},
$$
$$
P_{t \mid t-1} = A P_{t-1 \mid t-1} A^\top + Q.
$$

**Update step**
$$
K_t = P_{t \mid t-1} C^\top \left(C P_{t \mid t-1} C^\top + R\right)^{-1},
$$
$$
\hat{\mathbf{z}}_{t \mid t} =
\hat{\mathbf{z}}_{t \mid t-1}
+
K_t\left(\mathbf{x}_t - C\hat{\mathbf{z}}_{t \mid t-1}\right),
$$
$$
P_{t \mid t} = (I - K_t C) P_{t \mid t-1}.
$$

The term
$$
\mathbf{x}_t - C\hat{\mathbf{z}}_{t \mid t-1}
$$
is the **innovation** or prediction error.
The Kalman gain $K_t$ decides how much we trust the new measurement relative to the predicted state.

This is one of the cleanest pieces of mathematics in applied statistics.
It is just Gaussian Bayesian updating repeated through time.
If the observation noise covariance $R$ is small,
the filter trusts measurements more.
If the process noise covariance $Q$ is large,
the model admits that its dynamics are uncertain and becomes more responsive.

Examples:

**Example 1: one-dimensional tracking.**
A moving object's position is measured noisily at each step.
The filter smooths the observation stream while keeping up with true movement.

**Example 2: missing measurements.**
If an observation is absent,
the filter simply skips the update step and uses the predictive prior.

**Example 3: sensor fusion.**
Multiple noisy streams can update the same hidden state if the observation model is augmented appropriately.

Non-examples:

- The Kalman filter is not optimal for nonlinear or strongly non-Gaussian systems without modification.
- It is not a black-box denoiser;
  its performance depends on a dynamical model that should reflect the system.

```text
KALMAN FILTER AS REPEATED BAYES
=====================================================================

previous posterior over state
        |
        v
predict through dynamics  ->  Gaussian prior for current state
        |
        v
observe noisy measurement ->  innovation = observed - predicted
        |
        v
update with Kalman gain   ->  new Gaussian posterior

repeat for the next time step

=====================================================================
```

> **Backward reference:** The algebra of Gaussian posterior updating was developed in [Bayesian Inference](../04-Bayesian-Inference/notes.md).
> The Kalman filter is that same update repeated under linear dynamics and observation models.

The Kalman filter is one of the strongest examples in the entire curriculum of a classical statistical idea that still underlies modern AI systems:
recursion,
uncertainty propagation,
and hidden-state estimation in real time.

---

## 6. Advanced Topics

### 6.1 Spectral Analysis and the Frequency-Domain View

Time-series analysis has two complementary languages.
The time-domain language talks about lags,
recursions,
and autocovariances.
The frequency-domain language talks about cycles,
periodic components,
and how variance is distributed across temporal frequencies.

For a weakly stationary process,
the **spectral density** $f(\omega)$ is,
under standard conditions,
the Fourier transform of the autocovariance sequence:
$$
f(\omega) = \frac{1}{2\pi} \sum_{h=-\infty}^{\infty} \gamma(h)e^{-i\omega h},
\qquad \omega \in [-\pi,\pi].
$$
Conversely,
the autocovariance can be recovered from the spectrum.
This is the time-series version of the Wiener-Khinchin idea introduced earlier in the probability chapter.

The spectrum answers a question the ACF does not answer directly:
at what temporal scales does the process carry most of its variance?

- High power near zero frequency:
  slow variation,
  long-run drift,
  or persistent low-frequency behavior
- High power at seasonal frequencies:
  repeating cycles
- Broad high-frequency power:
  rapid local fluctuations

This view matters because some processes are easier to understand as filters than as lag recursions.
A moving-average model is literally a frequency filter.
Seasonal structure often becomes visually obvious in the spectrum.
Long-memory behavior shows itself through concentration at low frequencies.

Examples:

**Example 1: daily cycle.**
A strong daily periodic signal creates spectral peaks at the corresponding frequency and harmonics.

**Example 2: smooth trend-like series.**
Variance concentrates at low frequencies.

**Example 3: white noise.**
The spectral density is flat,
reflecting equal power across frequencies.

Non-examples:

- A spectrum is not only for deterministic sinusoids.
  It describes stochastic variance allocation across frequencies.
- Large low-frequency power does not by itself distinguish a trend from a unit root;
  interpretation still requires model context.

The time and frequency views are not rivals.
They are two coordinate systems for the same second-order structure.

### 6.2 Periodogram and Fourier Decomposition

The most common empirical entry point into spectral analysis is the **periodogram**,
which estimates how strongly different frequencies are represented in a finite sample.
For observations $x_0,\ldots,x_{T-1}$,
the discrete Fourier transform is
$$
d(\omega_j)=\sum_{t=0}^{T-1} x_t e^{-i\omega_j t},
$$
with Fourier frequencies
$$
\omega_j = \frac{2\pi j}{T}.
$$
The periodogram is then
$$
I(\omega_j)=\frac{1}{2\pi T}|d(\omega_j)|^2.
$$

Operationally,
the periodogram asks:
if we decompose the series into sinusoidal components,
which frequencies carry the most energy?

In finite samples,
the raw periodogram is noisy.
Smoothing or model-based spectral estimation is often used for more stable inference.
Still,
the raw periodogram is invaluable for detecting clear periodic structure and for building intuition about frequency content.

Examples:

**Example 1: hourly seasonality.**
A strong 24-hour cycle appears as a clear frequency spike.

**Example 2: synthetic sinusoid plus noise.**
The periodogram reveals the sinusoid even when the time-domain plot looks noisy.

**Example 3: transformer positional structure.**
Fourier-style encodings inject specific frequencies into the representation space,
which is one reason spectral language remains relevant in modern sequence modeling.

Non-examples:

- A periodogram peak is not automatically evidence of a stable causal mechanism.
  Finite samples,
  aliasing,
  and transient events can mislead.
- Fourier decomposition is not the same as an ARIMA model,
  though the two are often complementary.

The larger lesson is that temporal structure can be seen either as lag dependence or as frequency concentration.
A strong time-series practitioner learns to translate between both views.

### 6.3 Kalman Smoothing, Forecasting, and Missing Data

The Kalman filter uses information only up to the current time.
A **Kalman smoother** uses the full observation sequence
$x_1,\ldots,x_T$
to refine past state estimates.
This matters whenever we analyze a completed trajectory rather than make only online decisions.

Filtering asks:
what is the posterior state at time $t$ given data through time $t$?
Smoothing asks:
what is the posterior state at time $t$ given **all** observed data?

Because future measurements can clarify earlier uncertainty,
smoothing typically produces sharper latent-state estimates than filtering alone.
This is especially useful in tracking,
offline denoising,
and retrospective decomposition of trend and seasonality.

State-space models also handle missing data gracefully.
If an observation is absent at time $t$,
we propagate the predictive step and omit or modify the update.
This is far cleaner than ad hoc interpolation because the dynamics and uncertainty accounting remain explicit.

Forecasting from a state-space model is also natural.
Once the current posterior over the state is known,
we push that state forward through the transition matrix to obtain future predictive means and covariances.
This unifies tracking,
imputation,
and forecasting in one algebraic framework.

### 6.4 Multivariate Time Series Preview

Many real systems emit multiple linked sequences:
traffic,
latency,
error rate,
and GPU load;
or inflation,
unemployment,
and interest rates;
or position,
velocity,
and acceleration.
These require **multivariate time-series** models.

The simplest multivariate extension is the vector autoregression:
$$
\mathbf{X}_t = \mathbf{c} + A_1 \mathbf{X}_{t-1} + \cdots + A_p \mathbf{X}_{t-p} + \boldsymbol{\varepsilon}_t.
$$
Each series can influence itself and the others across time.
State-space models also extend naturally to multivariate observations and latent states.

This section does not develop the full multivariate theory.
That would overlap heavily with regression,
matrix methods,
and broader stochastic-process material.
The key preview is enough:
univariate lag logic generalizes to matrix-valued dependence,
and cross-series structure often matters as much as within-series persistence.

### 6.5 Change Points, Regime Shifts, and Structural Breaks

One of the biggest reasons time-series models fail in practice is that the data-generating process changes.
A pricing policy updates.
A product launches.
Users migrate regions.
A sensor is recalibrated.
A model deployment shifts request mix.
These are **change points** or **structural breaks**.

A structural break violates the stable-parameter story underlying classical stationary models.
The issue is not noise.
It is that the rules changed.

Typical symptoms include:

- abrupt shifts in mean
- changing variance
- different seasonal amplitude
- sudden forecast degradation
- residual autocorrelation after a previously adequate model

This matters because a model can look well specified over a historical window yet fail catastrophically after a regime change.
In many operational settings,
detecting the break is more valuable than squeezing a small gain from a marginally better stationary fit.

Examples:

**Example 1: new product launch.**
Traffic baseline jumps to a new level.

**Example 2: infrastructure migration.**
Latency variance drops because the serving stack changed.

**Example 3: distribution shift in a deployed model.**
Seasonal structure from a new market creates a previously absent weekly cycle.

Non-examples:

- A single outlier is not necessarily a regime shift.
- Strong seasonality is not itself a structural break if it has been stable all along.

The practical lesson is humble:
before concluding that a model family is bad,
check whether the system itself changed underneath it.

---

## 7. Applications in Machine Learning

### 7.1 Forecasting for Demand, Traffic, and Monitoring

Forecasting remains one of the highest-value uses of time-series models in ML systems.
Autoscaling,
budgeting,
inventory,
ad scheduling,
energy management,
and anomaly detection all begin with expectations about the next few observations.

Classical baselines are important here.
A strong ARIMA or seasonal state-space model often beats a complex deep model when:

- sample size is modest,
- seasonality is simple,
- explainability matters,
- covariates are limited,
- operational latency favors light models.

This is not nostalgia.
It is model-class alignment.
If the data are mostly linear,
seasonal,
and stable,
classical time-series tools can be the right inductive bias.

### 7.2 Kalman Filtering for Tracking and Sensor Fusion

Kalman filtering is used anywhere a hidden physical or logical state must be estimated sequentially from noisy measurements.
That includes robotics,
autonomous systems,
AR/VR tracking,
wearables,
industrial monitoring,
and even some online recommendation or control settings.

Its ML relevance is broader than the textbook tracking example suggests.
The main idea is recursive uncertainty-aware hidden-state estimation.
Whenever a system evolves,
measurements are noisy,
and decisions must be made online,
the Kalman viewpoint is a natural baseline.

### 7.3 Transformers, Fourier Features, and Temporal Encoding

Modern sequence models frequently reintroduce time-series ideas under different names.
Relative position encodings exploit translation structure across lags.
Fourier features inject known periodic bases.
Attention heads often specialize by temporal scale,
which is another way of saying they respond to different effective frequencies.

The classical language of seasonality,
spectra,
and lag structure helps interpret why these mechanisms work.
A transformer is not an ARIMA model,
but it still has to represent short-range persistence,
periodic structure,
and long-range low-frequency dependencies if it is to model sequential data well.

### 7.4 Structured State Space Models in Deep Learning

The recent revival of state-space models in deep learning,
including S4 and selective state-space models such as Mamba,
is a reminder that hidden-state recurrence never disappeared.
It was reengineered.

These models keep the classical intuition:
a latent state evolves,
new inputs modify it,
and outputs are read from that evolving state.
What changed is the scale,
parameterization,
and efficiency engineering required to compete with attention on long sequences.

The bridge to classical statistics is not superficial.
State-space structure is exactly the old idea that sequence data can be summarized by a dynamical hidden state rather than only by a raw lag window.

### 7.5 Foundation Models for Time Series

Large pretrained forecasting systems aim to transfer knowledge across many time-series tasks,
much as language models transfer across text tasks.
The promise is that broad pretraining can learn generic priors about seasonality,
trend,
event responses,
and cross-domain forecasting structure.

The caution is equally important:
foundation models do not remove the need for time-series reasoning.
Forecast horizon,
calibration,
seasonality,
regime shifts,
and evaluation leakage remain central.
A larger model changes the tool,
not the mathematics of what it means to forecast sequentially.

---

## 8. Common Mistakes

| # | Mistake | Why It Is Wrong | Fix |
| --- | --- | --- | --- |
| 1 | Treating a time series as iid tabular data with a timestamp column | This discards the lag structure that carries most predictive information and makes standard errors too optimistic | Start with plots, lag diagnostics, and an explicit temporal split before choosing a model |
| 2 | Using the sample ACF on a strongly trending series and interpreting the spikes literally | Trend and unit-root behavior induce slow ACF decay even when the stationary dependence structure is weak | Check stationarity first; detrend or difference before using ACF/PACF for order clues |
| 3 | Equating zero autocorrelation with independence | Nonlinear dependence can remain even when linear autocorrelation is near zero | Treat ACF as a linear diagnostic; inspect residuals, nonlinear features, and domain structure |
| 4 | Calling every upward-sloping series a random walk | Trend-stationary and difference-stationary behavior are different and lead to different long-run interpretations | Compare detrending and differencing; think about whether shocks should decay or persist |
| 5 | Over-differencing | Extra differencing can inject unnecessary MA structure and erase meaningful long-run level information | Use the smallest differencing order that stabilizes the series adequately |
| 6 | Treating ACF/PACF heuristics as exact model-selection rules | Finite samples blur ideal cutoff patterns and structural breaks can mimic higher order | Combine diagnostics with information criteria and residual checks |
| 7 | Ignoring seasonality because the short-lag ACF looks modest | Seasonal dependence can dominate forecasting while remaining invisible at nonseasonal lags | Inspect domain-specific periods and seasonal lags explicitly |
| 8 | Assuming Kalman filtering is only for aerospace tracking | The core idea is recursive Gaussian hidden-state estimation, which appears in many ML systems | Recognize state-space structure whenever latent dynamics and noisy observations coexist |
| 9 | Treating a good in-sample fit as proof of a good forecast model | Nonstationarity and structural breaks can make a fit historically accurate but operationally useless | Evaluate by forecast horizon, rolling-origin validation, and residual stability |
| 10 | Using random train/test shuffles for time-series evaluation | Leakage from the future into the past gives unrealistic performance estimates | Use chronological splits, rolling windows, or blocked validation |
| 11 | Interpreting a spectral peak as automatic evidence of a physical cycle | Finite-sample noise, aliasing, and transients can all create misleading peaks | Cross-check with domain knowledge, time-domain plots, and stability across windows |
| 12 | Assuming larger deep sequence models remove the need for temporal diagnostics | Model size does not solve leakage, drift, seasonality, or calibration by itself | Use classical diagnostics to understand what the deep model must learn and where it fails |

---

## 9. Exercises

### Exercise 1 `*` - White Noise vs Random Walk Diagnostics

You observe two synthetic series of equal length.
One is white noise and one is a random walk.
Explain how you would distinguish them using the time plot,
sample ACF,
and differencing.

### Exercise 2 `*` - Compute and Interpret ACF/PACF

Given a short observed series,
compute the sample autocovariance and sample autocorrelation at lags 1 and 2.
Explain what the sign and magnitude imply about short-range persistence.

### Exercise 3 `*` - AR(2) Stability from Characteristic Roots

Consider an AR(2) model with specified coefficients.
Form the characteristic polynomial,
solve for its roots,
and determine whether the process is stationary.

### Exercise 4 `**` - One-Step and Multi-Step AR Forecasting

For a fitted AR(1) process,
derive the one-step and multi-step forecast means.
Show how the forecast converges to the long-run mean and how uncertainty grows with horizon.

### Exercise 5 `**` - Seasonal Differencing and SARIMA Intuition

Given a weekly seasonal daily series,
explain what seasonal differencing does and how seasonal AR or MA terms would modify the model.

### Exercise 6 `**` - Periodogram-Based Frequency Detection

Construct a noisy sinusoidal signal and use the discrete Fourier transform or periodogram to identify the dominant frequency.
Explain why the time-domain pattern is less obvious than the spectral one.

### Exercise 7 `***` - Kalman Predict/Update for 1D Tracking

Write the predict and update equations for a one-dimensional tracking model.
Compute the posterior mean and variance after one noisy observation.

### Exercise 8 `***` - Time-Series Modeling in an ML Deployment Setting

Choose an AI system setting such as API traffic,
GPU load,
token throughput,
or sensor monitoring.
Describe which temporal structure matters,
which classical baseline you would start with,
and which failure modes would make you escalate to a richer model.

---

## 10. Why This Matters for AI (2026 Perspective)

| Concept | AI Impact |
| --- | --- |
| Autocorrelation and persistence | Explains why recent context carries predictive value in telemetry, click streams, and hidden-state trajectories |
| Stationarity and drift | Determines whether historical data can support future decisions; central for monitoring deployed systems |
| ACF/PACF diagnostics | Provides fast interpretable checks before expensive sequence-model experimentation |
| ARIMA and SARIMA | Strong operational baselines for demand, latency, and traffic forecasting where data are structured and limited |
| Forecast uncertainty growth | Essential for capacity planning, inventory, and safe decision thresholds over different horizons |
| State-space models | Unify hidden-state reasoning across control, tracking, denoising, and sequential latent-variable models |
| Kalman filtering | Gives a recursive uncertainty-aware baseline for tracking and sensor fusion that still underlies modern systems |
| Spectral analysis | Clarifies periodicity, low-frequency drift, and the role of Fourier-style features in temporal models |
| Change-point reasoning | Supports detection of regime shifts in production metrics, user behavior, and environment dynamics |
| S4/Mamba-style state-space architectures | Show that classical state evolution remains a live design principle in frontier long-sequence models |
| Foundation forecasting models | Extend transfer learning to time series, but still rely on core ideas of horizon, calibration, and temporal dependence |

---

## 11. Conceptual Bridge

Time series sits exactly between probability and action.
From probability theory,
it inherits stochastic-process language,
stationarity,
autocovariance,
and spectral duality.
From Bayesian inference,
it inherits recursive uncertainty updating,
especially in Gaussian state-space models.
From regression analysis,
it borrows linear prediction,
feature construction,
and model diagnostics.
From optimization and control,
it borrows the idea that sequential decisions depend on a latent dynamical state and forecast distribution rather than a one-shot estimate.

The backward links are therefore clear.

- [Stochastic Processes](../../06-Probability-Theory/06-Stochastic-Processes/notes.md) supplies the abstract process language,
  stationarity definitions,
  and spectral foundation.
- [Markov Chains](../../06-Probability-Theory/07-Markov-Chains/notes.md) supplies memoryless transition structure,
  which state-space and filtering ideas echo in more general forms.
- [Bayesian Inference](../04-Bayesian-Inference/notes.md) supplies the posterior-update viewpoint that reappears directly in Kalman filtering.

The forward links are equally important.

- [Regression Analysis](../06-Regression-Analysis/notes.md) will formalize linear predictive modeling with observed covariates,
  while time series already treats lagged values and seasonal indicators as structured predictors.
- Optimization will explain how sequential models are fit at scale.
- Information theory will sharpen the role of spectra,
  compression,
  and uncertainty in sequence modeling.
- Modern sequence-model chapters will revisit these same ideas in transformers,
  state-space deep models,
  and forecasting foundation systems.

```text
CURRICULUM POSITION
=====================================================================

Probability Theory
  -> stochastic processes, stationarity, spectra

Statistics
  -> estimation
  -> hypothesis testing
  -> Bayesian updating
  -> time series
  -> regression

Optimization / Control / Sequence Models
  -> forecasting decisions
  -> hidden-state learning
  -> long-context architectures

Time series is the bridge from "random evolution"
to "learned sequential prediction."

=====================================================================
```

The chapter-level takeaway is that time-series analysis teaches a habit of mind that transfers far beyond classical forecasting.
Whenever a system evolves,
whenever information arrives sequentially,
whenever uncertainty compounds with horizon,
and whenever hidden state matters,
the time-series viewpoint is the right starting language.

## References

1. George E. P. Box, Gwilym M. Jenkins, Gregory C. Reinsel, and Greta M. Ljung, _Time Series Analysis: Forecasting and Control_, 5th ed., Wiley, 2015.
2. Peter J. Brockwell and Richard A. Davis, _Introduction to Time Series and Forecasting_, 3rd ed., Springer, 2016.
3. James D. Hamilton, _Time Series Analysis_, Princeton University Press, 1994.
4. Andrew C. Harvey, _Forecasting, Structural Time Series Models and the Kalman Filter_, Cambridge University Press, 1989.
5. Simo Sarkka, _Bayesian Filtering and Smoothing_, Cambridge University Press, 2013.
6. Durbin, James and Siem Jan Koopman, _Time Series Analysis by State Space Methods_, 2nd ed., Oxford University Press, 2012.
7. Rob J Hyndman and George Athanasopoulos, _Forecasting: Principles and Practice_, 3rd ed., OTexts, 2021.
8. Rudolf E. Kalman, "A New Approach to Linear Filtering and Prediction Problems," _Journal of Basic Engineering_, 1960.
9. Berkeley Stat 153/248 course notes on characteristics and spectral analysis, 2026 editions.
10. MIT 14.384 lecture notes on state-space models and the Kalman filter, 2013 course materials.
11. Albert Gu, Karan Goel, and Christopher Re, "Efficiently Modeling Long Sequences with Structured State Spaces," ICLR 2022.
12. Tri Dao and Albert Gu, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," arXiv, 2023.
13. Azam Rahimian et al., "TimeGPT-1," 2023.
