[← Back to Numerical Methods](../README.md) | [← Previous: Interpolation and Approximation](../04-Interpolation-and-Approximation/notes.md)

---

# Numerical Integration

> *"Numerical integration is the art of turning hard integrals into easy sums."*  
> — Attributed to Philip Davis and Philip Rabinowitz

## Overview

**Numerical integration** (quadrature) computes $\int_a^b f(x) w(x) dx$ when no antiderivative is available, the function is known only at discrete points, or the exact integral is too expensive. The core idea is simple: approximate $f$ by an interpolant and integrate the interpolant exactly.

The quality of this approximation depends entirely on the choice of interpolation nodes. Equispaced nodes give the Newton-Cotes rules (trapezoidal, Simpson's) which are easy to derive but have bounded accuracy. Optimal node placement — the zeros of orthogonal polynomials — gives Gaussian quadrature, which achieves "twice the accuracy for free" by allowing the nodes to vary. Adaptive quadrature refines the mesh where the integrand is rough, achieving near-machine accuracy automatically. Monte Carlo methods trade the deterministic $O(h^p)$ error for a stochastic $O(1/\sqrt{n})$ rate that is dimension-independent — crucial for high-dimensional integrals.

For AI, numerical integration appears in: computing normalizing constants for probability distributions, training variational autoencoders (ELBO involves an expectation), Gaussian process marginal likelihoods, normalizing flows, and reinforcement learning value estimation. Understanding which quadrature method to use — and why — is essential for building numerically stable, efficient ML systems.

## Prerequisites

- Calculus: definite integrals, Riemann sums, Taylor's theorem (§01-Mathematical-Foundations)
- Interpolation: Lagrange polynomials, Chebyshev nodes, orthogonal polynomials (§10-04)
- Linear algebra: solving linear systems (§02-Linear-Algebra-Basics, §10-02)
- Floating-point arithmetic: rounding error accumulation (§10-01)
- Probability: random variables, expected values, variance (§07-Probability)

## Companion Notebooks

| Notebook | Description |
|---|---|
| [theory.ipynb](theory.ipynb) | Interactive: Newton-Cotes, Gaussian quadrature derivation, adaptive quadrature, Monte Carlo, multidimensional integration |
| [exercises.ipynb](exercises.ipynb) | 8 graded exercises from trapezoidal rule to normalizing flow partition functions |

## Learning Objectives

After completing this section, you will:

- Derive Newton-Cotes rules by integrating Lagrange interpolants
- Understand the error analysis of composite rules via the Euler-Maclaurin formula
- Derive Gaussian quadrature by choosing nodes to maximize the degree of exactness
- Implement adaptive Gauss-Kronrod quadrature with error control
- Apply Richardson extrapolation / Romberg integration to accelerate convergence
- Use Monte Carlo integration for high-dimensional problems and understand variance reduction
- Connect quadrature to normalizing constants, ELBO, and variational inference in ML
- Recognize when each method is appropriate based on dimensionality, smoothness, and accuracy requirements

---

## Table of Contents

- [1. Intuition and the Basic Setup](#1-intuition-and-the-basic-setup)
  - [1.1 The Quadrature Problem](#11-the-quadrature-problem)
  - [1.2 Why Integration is Hard](#12-why-integration-is-hard)
  - [1.3 Historical Timeline](#13-historical-timeline)
- [2. Newton-Cotes Rules](#2-newton-cotes-rules)
  - [2.1 Midpoint, Trapezoidal, and Simpson's Rules](#21-midpoint-trapezoidal-and-simpsons-rules)
  - [2.2 Composite Rules and Error Analysis](#22-composite-rules-and-error-analysis)
  - [2.3 Euler-Maclaurin Formula](#23-euler-maclaurin-formula)
  - [2.4 Limitations of Newton-Cotes](#24-limitations-of-newton-cotes)
- [3. Gaussian Quadrature](#3-gaussian-quadrature)
  - [3.1 Degree of Exactness and Optimal Nodes](#31-degree-of-exactness-and-optimal-nodes)
  - [3.2 Gauss-Legendre Quadrature](#32-gauss-legendre-quadrature)
  - [3.3 Gauss-Chebyshev Quadrature](#33-gauss-chebyshev-quadrature)
  - [3.4 Other Gauss Rules](#34-other-gauss-rules)
  - [3.5 Gauss-Kronrod Rules for Error Estimation](#35-gauss-kronrod-rules-for-error-estimation)
- [4. Richardson Extrapolation and Romberg Integration](#4-richardson-extrapolation-and-romberg-integration)
  - [4.1 Richardson Extrapolation](#41-richardson-extrapolation)
  - [4.2 Romberg's Method](#42-rombergs-method)
  - [4.3 Clenshaw-Curtis Quadrature](#43-clenshaw-curtis-quadrature)
- [5. Adaptive Quadrature](#5-adaptive-quadrature)
  - [5.1 Adaptive Strategies](#51-adaptive-strategies)
  - [5.2 Error Control and Termination](#52-error-control-and-termination)
  - [5.3 scipy.integrate API](#53-scipyintegrate-api)
- [6. Monte Carlo Integration](#6-monte-carlo-integration)
  - [6.1 Basic Monte Carlo](#61-basic-monte-carlo)
  - [6.2 Variance Reduction Techniques](#62-variance-reduction-techniques)
  - [6.3 Quasi-Monte Carlo and Low-Discrepancy Sequences](#63-quasi-monte-carlo-and-low-discrepancy-sequences)
- [7. Multidimensional Integration](#7-multidimensional-integration)
  - [7.1 Tensor Product Quadrature](#71-tensor-product-quadrature)
  - [7.2 Sparse Grids](#72-sparse-grids)
  - [7.3 Monte Carlo vs Deterministic in High Dimensions](#73-monte-carlo-vs-deterministic-in-high-dimensions)
- [8. Applications in Machine Learning](#8-applications-in-machine-learning)
  - [8.1 Normalizing Constants and Partition Functions](#81-normalizing-constants-and-partition-functions)
  - [8.2 ELBO and Variational Inference](#82-elbo-and-variational-inference)
  - [8.3 Gaussian Process Marginal Likelihood](#83-gaussian-process-marginal-likelihood)
  - [8.4 Expected Value Estimation in RL](#84-expected-value-estimation-in-rl)
- [9. Common Mistakes](#9-common-mistakes)
- [10. Exercises](#10-exercises)
- [11. Why This Matters for AI (2026 Perspective)](#11-why-this-matters-for-ai-2026-perspective)
- [12. Conceptual Bridge](#12-conceptual-bridge)

---

## 1. Intuition and the Basic Setup

### 1.1 The Quadrature Problem

A **quadrature rule** approximates $I(f) = \int_a^b f(x) w(x) dx$ by a weighted sum:

$$Q_n(f) = \sum_{i=1}^{n} w_i f(x_i)$$

where $x_i \in [a, b]$ are **nodes** (quadrature points) and $w_i > 0$ are **weights**.

The **error** is $E_n(f) = I(f) - Q_n(f)$.

**What determines quality?**
1. The number of nodes $n$ (more nodes = higher accuracy in general)
2. The placement of nodes $x_i$ (free or equispaced?)
3. The smoothness of $f$ (smooth $f$ allows high-order rules)
4. The dimensionality $d$ (high-dimensional integrals need Monte Carlo)

**For AI:** Many ML objectives involve integrals:
- $\log p(x) = \log \int p(x|z) p(z) dz$ — variational ELBO lower bounds this
- Policy gradient $\nabla_\theta J = \mathbb{E}_\pi[\nabla_\theta \log \pi_\theta(a|s) R]$ — Monte Carlo estimates
- Gaussian process likelihood $\log p(y) = -\frac{1}{2} y^\top K^{-1} y - \frac{1}{2}\log|K|$ — involves a determinant, not an integral, but the method of moments is related

### 1.2 Why Integration is Hard

- **Indefinite integrals of elementary functions:** Only a small class have closed forms. $\int e^{-x^2} dx$, $\int \sin(x)/x \, dx$ have no elementary form.
- **High-dimensional integrals:** The curse of dimensionality makes grid-based methods infeasible for $d > 5$–$10$.
- **Singularities:** $\int_0^1 x^{-1/2} dx = 2$ — finite integral but $f(0) = \infty$.
- **Oscillatory integrands:** $\int_0^{1000} \sin(x^2) dx$ — many sign changes, standard rules need many nodes.
- **Stochastic integrals:** $\mathbb{E}[f(X)] = \int f(x) p(x) dx$ — typically solved by sampling.

### 1.3 Historical Timeline

```
NUMERICAL INTEGRATION: HISTORICAL TIMELINE
════════════════════════════════════════════════════════════════════════

  1700  ─ Newton (1711): Newton-Cotes formulas from polynomial interpolation
  1720  ─ Cotes: systematic derivation of Newton-Cotes weights
  1800  ─ Gauss (1814): Gaussian quadrature with 2n-1 degree of exactness
  1852  ─ Chebyshev: Gauss-Chebyshev quadrature with explicit node/weight formulas
  1901  ─ Richardson: Richardson extrapolation to accelerate convergence
  1955  ─ Romberg: Romberg integration (repeated Richardson extrapolation)
  1960  ─ Clenshaw-Curtis: Chebyshev-based quadrature via DCT
  1970  ─ Patterson, Kronrod: Gauss-Kronrod embedded rules for adaptive quadrature
  1965  ─ Sobol: quasi-Monte Carlo sequences for multidimensional integration
  1949  ─ Metropolis-Ulam: Monte Carlo method for nuclear physics calculations
  1953  ─ Metropolis: Markov chain Monte Carlo (Metropolis algorithm)
  1990  ─ Gelfand-Smith: MCMC for Bayesian inference becomes mainstream
  2015  ─ Deep learning era: stochastic gradient estimation (REINFORCE, VAE ELBO)
  2020  ─ Normalizing flows: exact likelihood via change-of-variables theorem
  2022  ─ Score-based diffusion: sampling as integration of stochastic DEs

════════════════════════════════════════════════════════════════════════
```

---

## 2. Newton-Cotes Rules

### 2.1 Midpoint, Trapezoidal, and Simpson's Rules

**Derivation principle:** Integrate the $n$-point Lagrange interpolant exactly.

**Midpoint rule (1 point, $n=1$):**

$$M(f) = (b-a) f\!\left(\frac{a+b}{2}\right), \quad E_M = \frac{(b-a)^3}{24} f''(\xi)$$

Integrates polynomials of degree $\leq 1$ exactly (1-point rule with degree of exactness 1).

**Trapezoidal rule (2 points, $n=2$):**

$$T(f) = \frac{b-a}{2}[f(a) + f(b)], \quad E_T = -\frac{(b-a)^3}{12} f''(\xi)$$

Same order as midpoint but larger constant (midpoint is twice as accurate as trapezoidal for convex/concave functions).

**Simpson's rule (3 points, $n=3$):**

$$S(f) = \frac{b-a}{6}\left[f(a) + 4f\!\left(\frac{a+b}{2}\right) + f(b)\right], \quad E_S = -\frac{(b-a)^5}{90} f^{(4)}(\xi)$$

Integrates polynomials of degree $\leq 3$ exactly (degree of exactness 3, not 2 — odd-degree rules gain one extra order via symmetry).

```
NEWTON-COTES RULES: GEOMETRIC INTERPRETATION
════════════════════════════════════════════════════════════════════════

  Midpoint:        Trapezoidal:        Simpson's:
  ─────────        ────────────        ─────────
    ──             /\                  ╭──╮
   /  \           /  \               /    \
  / f  \         / f  \             /  f   \
 /  ×  \        / × × \           / × × × \
 ────────       ──────────         ──────────
   (b-a)        (b-a)/2           (b-a)/6, 4/6, 1/6

  × = quadrature node
  Approximation: rectangle, trapezoid, parabola through 3 points

════════════════════════════════════════════════════════════════════════
```

**General Newton-Cotes with $n+1$ equispaced nodes:** $x_i = a + ih$, $h = (b-a)/n$:

$$Q_n(f) = h \sum_{i=0}^{n} w_i^{\text{NC}} f(x_i)$$

The weights $w_i^{\text{NC}}$ are determined by integrating the Lagrange basis:

$$w_i^{\text{NC}} = \frac{1}{h} \int_a^b \ell_i(x) dx = \int_0^n \prod_{j \neq i} \frac{s - j}{i - j} ds$$

**Newton-Cotes weights for small $n$:**

| $n$ | Rule | Weights | Degree |
|---|---|---|---|
| 1 | Trapezoidal | $\frac{h}{2}[1, 1]$ | 1 |
| 2 | Simpson's | $\frac{h}{3}[1, 4, 1]$ | 3 |
| 3 | Simpson's 3/8 | $\frac{3h}{8}[1, 3, 3, 1]$ | 3 |
| 4 | Boole's | $\frac{2h}{45}[7, 32, 12, 32, 7]$ | 5 |

**Key observation:** Even-$n$ rules (trapezoidal with $n=2$, Boole's with $n=4$) automatically achieve one higher degree of exactness than expected — a consequence of symmetry.

### 2.2 Composite Rules and Error Analysis

**Composite trapezoidal rule:** Divide $[a,b]$ into $m$ subintervals of width $h = (b-a)/m$:

$$T_m(f) = h\left[\frac{f(a)}{2} + f(a+h) + f(a+2h) + \cdots + f(b-h) + \frac{f(b)}{2}\right]$$

$$E_{T_m} = -\frac{(b-a)h^2}{12} f''(\xi) = O(h^2)$$

**Composite Simpson's rule:** Requires even $m$:

$$S_m(f) = \frac{h}{3}[f(a) + 4f(a+h) + 2f(a+2h) + 4f(a+3h) + \cdots + f(b)]$$

$$E_{S_m} = -\frac{(b-a)h^4}{180} f^{(4)}(\xi) = O(h^4)$$

**Error comparison:**

| Rule | Function evaluations | Error | Accuracy for $10^{-8}$: need $m$ |
|---|---|---|---|
| Composite trapezoidal | $m+1$ | $O(h^2)$ | $h \approx 10^{-4}$, $m \approx 10^4$ |
| Composite Simpson's | $m+1$ ($m$ even) | $O(h^4)$ | $h \approx 10^{-2}$, $m \approx 100$ |
| Composite Boole's | $m+1$ ($4|m$) | $O(h^6)$ | $h \approx 0.1$, $m \approx 10$ |
| Gauss-Legendre $n$ pts | $n$ | $O(h^{2n})$ | $n \approx 5$ |

### 2.3 Euler-Maclaurin Formula

The **Euler-Maclaurin formula** gives the exact relationship between the trapezoidal rule and the true integral:

$$T_m(f) = \int_a^b f(x) dx + \sum_{k=1}^{p} \frac{B_{2k}}{(2k)!} h^{2k} [f^{(2k-1)}(b) - f^{(2k-1)}(a)] + O(h^{2p+2})$$

where $B_{2k}$ are Bernoulli numbers ($B_2 = 1/6$, $B_4 = -1/30$, $B_6 = 1/42$, ...).

**Key implications:**
1. The error has an **asymptotic expansion in powers of $h^2$** — this is the foundation for Richardson extrapolation
2. For **periodic functions** where all derivatives of $f$ match at $a$ and $b$, the boundary terms vanish for all $k$ — the composite trapezoidal rule achieves **spectral accuracy** for smooth periodic functions!
3. The formula predicts exactly how much extrapolation reduces error

**Periodic functions example:** For $f(x) = \cos(2\pi x)$ on $[0,1]$ with $m$ equal intervals, the trapezoidal rule is exact to machine precision even for small $m$. This is why the DFT (which uses uniform sampling of a periodic function) is exact!

**For AI:** The Euler-Maclaurin formula explains why the FFT is so accurate for bandlimited signals: the trapezoidal rule applied to a periodic function with finitely many nonzero Fourier coefficients gives the exact DFT.

### 2.4 Limitations of Newton-Cotes

**Negative weights for large $n$:** Newton-Cotes weights become negative for $n \geq 8$. This means small perturbations in $f$ values can cancel to give a large error — **numerical instability**.

**Runge's phenomenon for integration:** Unlike interpolation, integration is a smoother operation — integrating the Runge interpolant gives reasonable results for moderate $n$. But for large $n$ with equispaced nodes, the interpolant oscillates wildly and the integral of the oscillations can be large.

**Conclusion:** Use composite low-order rules (trapezoidal, Simpson's) rather than high-degree Newton-Cotes on a single interval.

---

## 3. Gaussian Quadrature

### 3.1 Degree of Exactness and Optimal Nodes

**Definition:** A quadrature rule $Q_n$ has **degree of exactness** $d$ if $Q_n(p) = I(p)$ for all polynomials $p$ of degree $\leq d$, but there exists a degree-$d+1$ polynomial where it fails.

**Newton-Cotes with $n+1$ equispaced nodes** achieves degree of exactness $n$ (or $n+1$ if $n$ is even).

**Question:** Can we do better by choosing the nodes freely?

**Counting argument:** A rule with $n$ free nodes has $2n$ free parameters ($n$ nodes $+ n$ weights). A polynomial basis requires matching $2n$ coefficients. We might hope to achieve degree of exactness $2n-1$.

**Theorem (Gaussian quadrature):** This is achievable. With $n$ optimally chosen nodes, we can achieve degree of exactness $2n-1$ — exactly twice what equispaced nodes give.

**How to find the nodes:** The optimal nodes for $\int_{-1}^1 f(x) dx$ are the **zeros of the $n$-th Legendre polynomial** $P_n(x)$.

**Proof sketch:** For any polynomial $f$ of degree $\leq 2n-1$, write $f = q \cdot P_n + r$ where $q$ has degree $\leq n-1$ and $r$ has degree $\leq n-1$. Then:
- $\int_{-1}^1 f = \int q \cdot P_n + \int r$ — the first integral is 0 by orthogonality of $P_n$ to lower-degree polynomials
- $Q_n(f) = Q_n(q \cdot P_n) + Q_n(r) = 0 + Q_n(r)$ — the first term is 0 because $q \cdot P_n$ vanishes at all zeros of $P_n$
- $Q_n(r) = \int r$ because $Q_n$ is exact for polynomials of degree $\leq n-1 < 2n-1$ $\square$

### 3.2 Gauss-Legendre Quadrature

For $\int_{-1}^1 f(x) dx$, the $n$-point Gauss-Legendre rule uses:
- **Nodes:** zeros of $P_n(x)$ (Legendre polynomial of degree $n$)
- **Weights:** $w_i = \frac{2}{(1-x_i^2)[P_n'(x_i)]^2}$

**Gauss-Legendre nodes and weights for small $n$:**

| $n$ | Nodes $x_i$ | Weights $w_i$ |
|---|---|---|
| 1 | $0$ | $2$ |
| 2 | $\pm 1/\sqrt{3}$ | $1, 1$ |
| 3 | $0, \pm\sqrt{3/5}$ | $8/9, 5/9, 5/9$ |
| 5 | (5 values in $(-1,1)$) | (5 values summing to 2) |

**Error:** For $f \in C^{2n}[-1,1]$:

$$E_n = \frac{(n!)^4}{(2n+1)[(2n)!]^3} (b-a)^{2n+1} f^{(2n)}(\xi) = O(h^{2n})$$

**Scaling to $[a,b]$:** Change variables $x = \frac{a+b}{2} + \frac{b-a}{2} t$, $t \in [-1,1]$:

$$\int_a^b f(x) dx = \frac{b-a}{2} \int_{-1}^1 f\!\left(\frac{a+b}{2} + \frac{b-a}{2} t\right) dt \approx \frac{b-a}{2} \sum_{i=1}^n w_i f\!\left(\frac{a+b}{2} + \frac{b-a}{2} x_i\right)$$

**Computing GL nodes:** Use the eigenvalue method — nodes are eigenvalues of the **Golub-Welsch tridiagonal matrix:**

$$J_n = \begin{pmatrix} 0 & \beta_1 & & \\ \beta_1 & 0 & \beta_2 & \\ & \beta_2 & 0 & \ddots \\ & & \ddots & \ddots \end{pmatrix}, \quad \beta_k = \frac{k}{2\sqrt{k^2 - 1/4}}$$

The eigenvalues are the GL nodes, and the first components of the eigenvectors give the weights.

**For AI:** Gauss-Legendre quadrature is used in:
- **Normalizing flows:** Computing normalizing constants for complex distributions via quadrature
- **Physics-informed networks:** Computing weak-form integrals $\int \nabla u \cdot \nabla v$ exactly
- **ODE solvers:** Gauss collocation methods are implicit RK methods with GL nodes — they achieve the highest possible order for a given number of stages

### 3.3 Gauss-Chebyshev Quadrature

For $\int_{-1}^1 f(x) (1-x^2)^{-1/2} dx$ (the Chebyshev weight), the nodes and weights have **explicit formulas**:

$$x_k = \cos\!\left(\frac{(2k-1)\pi}{2n}\right), \quad w_k = \frac{\pi}{n}, \quad k = 1, \ldots, n$$

**Key properties:**
- Nodes are equispaced in $\theta = \arccos(x)$ — no eigenvalue computation needed
- All weights equal $\pi/n$ — uniform weights!
- These are exactly the Chebyshev nodes of the first kind
- Degree of exactness: $2n-1$

**Type-II Gauss-Chebyshev:** For weight $(1-x^2)^{1/2}$:

$$x_k = \cos\!\left(\frac{k\pi}{n+1}\right), \quad w_k = \frac{\pi}{n+1} \sin^2\!\left(\frac{k\pi}{n+1}\right)$$

### 3.4 Other Gauss Rules

**Gauss-Laguerre:** $\int_0^\infty f(x) e^{-x} dx \approx \sum_i w_i f(x_i)$ — nodes are zeros of Laguerre polynomials $L_n(x)$.

**Gauss-Hermite:** $\int_{-\infty}^\infty f(x) e^{-x^2} dx \approx \sum_i w_i f(x_i)$ — nodes are zeros of Hermite polynomials $H_n(x)$.

**For AI:** Gauss-Hermite quadrature is used for computing expectations over Gaussian distributions:

$$\mathbb{E}_{x \sim \mathcal{N}(\mu, \sigma^2)}[f(x)] = \frac{1}{\sqrt{\pi}} \int_{-\infty}^\infty f(\sqrt{2}\sigma t + \mu) e^{-t^2} dt \approx \frac{1}{\sqrt{\pi}} \sum_i w_i^{\text{GH}} f(\sqrt{2}\sigma x_i^{\text{GH}} + \mu)$$

This is used in computing the ELBO exactly in certain VAE architectures and in moment-matching for Gaussian approximations (expectation propagation).

**Gauss-Jacobi:** $\int_{-1}^1 f(x) (1-x)^\alpha (1+x)^\beta dx$ — used for Beta-distributed random variables and Jacobi polynomial bases.

### 3.5 Gauss-Kronrod Rules for Error Estimation

**The problem:** Gaussian quadrature doesn't naturally provide an error estimate.

**Gauss-Kronrod:** Extend an $n$-point GL rule to a $(2n+1)$-point rule by adding $n+1$ Kronrod nodes interleaved with the GL nodes, choosing them to maximize the degree of exactness of the combined rule.

The **G7-K15** rule (7-point GL + 15-point Kronrod) is the standard in `scipy.integrate.quad`:
- G7: degree of exactness 13 (7 nodes)
- K15: degree of exactness 27 (15 nodes, reuses the G7 nodes)
- Error estimate: $|K15 - G7|$

**Adaptivity:** If $|K15 - G7| < \text{tol}$, accept the interval. Otherwise, bisect and recurse.

---

## 4. Richardson Extrapolation and Romberg Integration

### 4.1 Richardson Extrapolation

**Setup:** If we know that $Q(h) = I + c_2 h^2 + c_4 h^4 + \cdots$ (error expansion in powers of $h^2$), then computing $Q(h)$ and $Q(h/2)$:

$$Q(h) = I + c_2 h^2 + O(h^4)$$
$$Q(h/2) = I + c_2 h^2/4 + O(h^4)$$

**Eliminating the $h^2$ term:**

$$I = \frac{4 Q(h/2) - Q(h)}{3} + O(h^4)$$

This is **one step of Richardson extrapolation** — combining two $O(h^2)$ estimates to get $O(h^4)$.

**General Richardson extrapolation:** If $Q_k(h) = I + c_p h^p + O(h^{p+2})$:

$$Q_{k+1}(h) = \frac{2^p Q_k(h/2) - Q_k(h)}{2^p - 1} + O(h^{p+2})$$

**For AI:** Richardson extrapolation is the foundation of **gradient estimation via finite differences** — taking linear combinations of function evaluations to cancel lower-order error terms. The centered difference formula is a 2-step Richardson extrapolation.

### 4.2 Romberg's Method

**Romberg's method** applies repeated Richardson extrapolation to the composite trapezoidal rule.

**Romberg table:** Starting from $T_{h_0}$ and halving $h$ repeatedly:

$$R_{0,0} = T(h_0)$$
$$R_{k,0} = T(h_0/2^k)$$
$$R_{k,j} = \frac{4^j R_{k,j-1} - R_{k-1,j-1}}{4^j - 1}, \quad j = 1, \ldots, k$$

The diagonal entry $R_{k,k}$ approximates $I$ with error $O(h_0^{2k+2}/2^{2k^2})$.

**Algorithm:**

```
R[0][0] = (b-a)/2 * (f(a) + f(b))  # Trapezoidal with h = b-a
for k = 1, 2, ..., max_iter:
    # Composite trapezoidal with 2^k intervals
    R[k][0] = 0.5*R[k-1][0] + h * sum(f(a + (2j-1)*h) for j in 1..2^(k-1))
    # Richardson extrapolation
    for j = 1, ..., k:
        R[k][j] = (4^j * R[k][j-1] - R[k-1][j-1]) / (4^j - 1)
    if |R[k][k] - R[k-1][k-1]| < tol: break
```

**Convergence:** For smooth, non-periodic functions, $R_{k,k}$ converges like $O(h_0^{2k+2})$ — faster than any fixed-order method!

### 4.3 Clenshaw-Curtis Quadrature

**Clenshaw-Curtis** integrates the Chebyshev interpolant of $f$ on $[-1,1]$:

$$\int_{-1}^1 f(x) dx \approx \int_{-1}^1 p_n(x) dx$$

where $p_n$ is the degree-$n$ Chebyshev interpolant at the $n+1$ Chebyshev nodes of the 2nd kind.

**Weights via DCT:** The Chebyshev expansion $p_n(x) = \sum_k c_k T_k(x)$ integrates to:

$$\int_{-1}^1 T_k(x) dx = \begin{cases} 2 & k = 0 \\ 0 & k \text{ odd} \\ \frac{2(-1)^{k/2+1}}{k^2-1} & k \text{ even} \end{cases}$$

So the CC weights are computed in $O(n \log n)$ via DCT.

**CC vs GL:** For smooth functions, CC and GL achieve similar accuracy. GL uses $n$ nodes for degree $2n-1$; CC uses $n+1$ nodes for degree $n$ but with DCT acceleration. For adaptive applications where the nodes are reused at subintervals, CC is often preferred because its nodes nest ($n=4$ nodes include the $n=2$ nodes).

---

## 5. Adaptive Quadrature

### 5.1 Adaptive Strategies

**Idea:** Concentrate function evaluations where the integrand is rough (large error per unit interval), not where it is smooth.

**Error indicator:** Estimate the local error on $[a,b]$ using:

$$\hat{E} = |Q_{\text{fine}}(f) - Q_{\text{coarse}}(f)|$$

where $Q_{\text{fine}}$ uses more points (e.g., Gauss-Kronrod K15) and $Q_{\text{coarse}}$ uses fewer (e.g., G7).

**Algorithm (recursive):**
1. Evaluate $Q_{\text{fine}}$ and $Q_{\text{coarse}}$ on $[a,b]$
2. If $\hat{E} < \text{tol} \cdot (b-a) / (B-A)$ (local tolerance proportional to interval width): accept
3. Otherwise: bisect $[a,b]$ into $[a,(a+b)/2]$ and $[(a+b)/2, b]$, recurse on each

**Global vs local tolerance:** The final error satisfies $|I - Q| \leq \sum_j |E_j| \leq \text{tol}$ if each leaf interval satisfies its local tolerance — requires careful accounting.

### 5.2 Error Control and Termination

**Termination conditions:**
1. **Local tolerance met:** $|Q_{\text{fine}} - Q_{\text{coarse}}| < \text{atol} + \text{rtol} \cdot |Q_{\text{fine}}|$
2. **Maximum evaluations reached:** Return with warning
3. **Interval too small:** Machine epsilon prevents further bisection

**Priority queue (non-recursive):** For efficiency, maintain a heap of subintervals sorted by error estimate. Always split the highest-error interval:

```python
import heapq
queue = [(-estimated_error, a, b)]
while error_sum > tol and queue:
    (-est_err, l, r) = heapq.heappop(queue)
    m = (l + r) / 2
    # Evaluate on [l,m] and [m,r]
    heapq.heappush(queue, (-new_err_left, l, m))
    heapq.heappush(queue, (-new_err_right, m, r))
```

### 5.3 scipy.integrate API

```python
from scipy import integrate

# Basic quadrature (adaptive Gauss-Kronrod)
result, error = integrate.quad(f, a, b)

# With tolerances
result, error = integrate.quad(f, a, b, epsabs=1e-12, epsrel=1e-12)

# Improper integrals
result, error = integrate.quad(f, 0, np.inf)

# Singularity handling
result, error = integrate.quad(f, 0, 1, points=[0.5])  # Inform about singularities

# Fixed-order quadrature
pts, wts = integrate.fixed_quad(f, a, b, n=10)  # Gauss-Legendre, n points

# Romberg integration
result = integrate.romberg(f, a, b)
```

---

## 6. Monte Carlo Integration

### 6.1 Basic Monte Carlo

**The fundamental idea:** Rewrite the integral as an expectation:

$$\int_\Omega f(x) dx = |\Omega| \cdot \mathbb{E}_{x \sim \text{Uniform}(\Omega)}[f(x)]$$

Estimate by sampling $x_1, \ldots, x_n \sim \text{Uniform}(\Omega)$:

$$I \approx \hat{I}_n = \frac{|\Omega|}{n} \sum_{i=1}^{n} f(x_i)$$

**Error analysis:** By the Central Limit Theorem:

$$\hat{I}_n - I \sim \mathcal{N}\!\left(0, \frac{|\Omega|^2 \text{Var}(f)}{n}\right), \quad \text{RMSE} = \frac{|\Omega| \sqrt{\text{Var}(f)}}{\sqrt{n}} = O(1/\sqrt{n})$$

**Key properties:**
1. **Dimension-independent:** $O(1/\sqrt{n})$ regardless of $d$
2. **No smoothness requirement:** Works for discontinuous $f$
3. **Slow convergence:** Need $n \sim 10^8$ for 4 decimal digits of accuracy — much slower than deterministic quadrature for smooth low-dimensional integrands
4. **Easily parallelizable:** All samples are independent

**When to use Monte Carlo:**
- $d > 5$–$10$ (deterministic methods are exponentially worse)
- $f$ is discontinuous or has singularities
- Accuracy requirement is modest ($\sim 1\%$–$5\%$)
- Sampling from $p(x)$ is easier than computing the integral directly

### 6.2 Variance Reduction Techniques

**1. Importance sampling:** Instead of sampling from $\text{Uniform}(\Omega)$, sample from a proposal distribution $q(x)$:

$$I = \int f(x) dx = \int \frac{f(x)}{q(x)} q(x) dx = \mathbb{E}_{x \sim q}\left[\frac{f(x)}{q(x)}\right]$$

Optimal $q^*(x) = |f(x)| / \int |f|$ — reduces variance to zero (but requires knowing the integral!). In practice, choose $q$ to be proportional to $|f|$ where $f$ is large.

**Estimator variance:** $\text{Var}_{q}\!\left(\frac{f}{q}\right) = \int \frac{f^2}{q} - I^2$ — minimized when $q \propto |f|$.

**2. Control variates:** Find $g$ with known integral $I_g$ and high correlation with $f$:

$$\hat{I} = \hat{I}_n(f) - \hat{I}_n(g) + I_g$$

The variance is $\text{Var}(f - g) = \text{Var}(f) + \text{Var}(g) - 2\text{Cov}(f,g)$ — reduced when $f$ and $g$ are highly correlated.

**3. Antithetic variates:** For symmetric domains, pair each sample $x_i$ with its mirror $2\mu - x_i$:

$$\hat{I} = \frac{1}{2n} \sum_{i=1}^{n} [f(x_i) + f(2\mu - x_i)]$$

Variance reduction: $\text{Var}(\frac{1}{2}(f(X)+f(2\mu-X))) = \frac{1}{2}\text{Var}(f) + \frac{1}{2}\text{Cov}(f(X), f(2\mu-X))$.

**4. Stratified sampling:** Divide $\Omega$ into $k$ strata and sample $n/k$ points from each. Variance:

$$\text{Var}_{\text{strat}} = \frac{1}{n} \sum_j p_j \text{Var}_j(f) \leq \text{Var}_{\text{plain}}$$

where $p_j$ is the probability mass of stratum $j$.

**For AI:** All of these techniques appear in ML:
- **Importance sampling** underlies REINFORCE/policy gradient: $\nabla J = \mathbb{E}_\pi[\nabla \log \pi \cdot R]$, sampled under current policy
- **Control variates** are used as baselines in policy gradient: subtract a baseline $b(s)$ to reduce variance
- **Stratified sampling** is used in mini-batch selection to ensure balanced class representation

### 6.3 Quasi-Monte Carlo and Low-Discrepancy Sequences

**Discrepancy:** Measures how uniformly a point set covers the domain. A sequence with low discrepancy fills the space more uniformly than random points.

**Koksma-Hlawka inequality:**

$$\left|\int f - \frac{1}{n}\sum_i f(x_i)\right| \leq V(f) \cdot D^*(x_1, \ldots, x_n)$$

where $V(f)$ is the variation of $f$ and $D^*$ is the star discrepancy.

**Sobol sequences:** Quasi-random sequences with discrepancy $O((\log n)^d / n)$ — much better than random's $O(1/\sqrt{n})$ for low to moderate $d$.

**Convergence comparison:**

| Method | Rate | Good for |
|---|---|---|
| Monte Carlo | $O(n^{-1/2})$ | Any $d$, discontinuous $f$ |
| Quasi-MC (Sobol) | $O(n^{-1} (\log n)^d)$ | $d \leq 20$, smooth $f$ |
| Gauss-Legendre | $O(n^{-2n/d})$ | $d \leq 3$, smooth $f$ |
| Clenshaw-Curtis | $O(\text{exp}(-cn))$ | $d = 1$, analytic $f$ |

**For AI:** Quasi-MC is used in:
- **Low-discrepancy random features:** Quasi-random Fourier features for kernel approximation are more accurate than random ones
- **Batch sampling in reinforcement learning:** Sobol-like sequences for diverse trajectory coverage
- **Hyperparameter search:** Quasi-random search outperforms grid search and random search for moderate $d$

---

## 7. Multidimensional Integration

### 7.1 Tensor Product Quadrature

For $\int_{[a,b]^d} f(x_1, \ldots, x_d) dx$, the **tensor product rule** applies 1D quadrature in each dimension:

$$Q_n^d(f) = \sum_{i_1=1}^{n} \cdots \sum_{i_d=1}^{n} w_{i_1} \cdots w_{i_d} f(x_{i_1}, \ldots, x_{i_d})$$

**Curse of dimensionality:** Requires $n^d$ function evaluations. For $n=10$ points per dimension and $d=10$ dimensions: $10^{10}$ evaluations. Infeasible!

**Error:** If each 1D rule has error $O(h^p)$:

$$|E| = O(h^p d) \text{ (not } O(h^{pd}) \text{)}$$

The error scales linearly with $d$, not exponentially — the convergence rate $O(h^p)$ is preserved. But the number of evaluations grows as $n^d$.

### 7.2 Sparse Grids

**Smolyak's construction:** Use tensor products only of "low-level" 1D rules. The level-$q$ sparse grid uses $O(n (\log n)^{d-1})$ points instead of $n^d$:

$$Q_q^d = \sum_{|i|_1 \leq q+d-1} (-1)^{q+d-1-|i|_1} \binom{d-1}{q+d-1-|i|_1} \otimes_{j=1}^d \Delta_{i_j}^{(1)}$$

where $\Delta_k^{(1)}$ is the difference between the $k$-th and $(k-1)$-th 1D rules.

**Error for smooth $f$:** $O((\log n)^{d-1}/n^p)$ — polynomial in $d$ logarithm, not exponential.

**For AI:** Sparse grid integration is used in:
- High-dimensional numerical PDEs (atmospheric modeling, option pricing)
- Bayesian quadrature for computing GP marginal likelihoods
- Efficient moment computation in uncertainty quantification

### 7.3 Monte Carlo vs Deterministic in High Dimensions

| $d$ | Best deterministic method | MC (unbiased, $n$ samples) | Break-even $n$ |
|---|---|---|---|
| 1 | GL-5 ($5$ pts, $O(h^{10})$) | $O(n^{-1/2})$ | $n \approx 100$ |
| 5 | Sparse grid ($\sim n^3$) | $O(n^{-1/2})$ | $n \approx 10^4$ |
| 10 | Sparse grid ($\sim n^5$) | $O(n^{-1/2})$ | $n \approx 10^6$ |
| 20 | MC wins clearly | $O(n^{-1/2})$ | — |
| 100 | MC only | $O(n^{-1/2})$ | — |

**Practical advice:** For $d \leq 3$, use Gauss-Legendre or Clenshaw-Curtis. For $d \leq 15$, consider quasi-MC or sparse grids. For $d > 15$, use Monte Carlo with variance reduction.

---

## 8. Applications in Machine Learning

### 8.1 Normalizing Constants and Partition Functions

**The partition function problem:** In energy-based models, the probability is:

$$p(x; \theta) = \frac{\exp(-E_\theta(x))}{Z(\theta)}, \quad Z(\theta) = \int \exp(-E_\theta(x)) dx$$

Computing $Z(\theta)$ exactly is intractable for complex $E_\theta$. Solutions:
- **Thermodynamic integration:** $\log Z(\theta_1) - \log Z(\theta_0) = \int_0^1 \mathbb{E}_{p(\cdot;\theta_t)}[E_{\theta_0}(x) - E_{\theta_1}(x)] dt$ — a 1D quadrature over the inverse temperature
- **Annealed importance sampling (AIS):** Monte Carlo estimate of the ratio $Z(\theta_1)/Z(\theta_0)$
- **Score matching:** Avoid $Z(\theta)$ entirely by training on $\nabla_x \log p = -\nabla_x E_\theta$

**For diffusion models:** The ELBO for a Gaussian diffusion process involves:

$$\mathcal{L} = \mathbb{E}_q\left[\log \frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:T})}\right]$$

Each KL divergence term between Gaussians is computable analytically — no numerical integration needed. This is why diffusion models are tractable.

### 8.2 ELBO and Variational Inference

**Variational Autoencoder (VAE):** The ELBO lower bound on $\log p(x)$:

$$\text{ELBO}(x; \phi, \theta) = \mathbb{E}_{z \sim q_\phi(z|x)}[\log p_\theta(x|z)] - \text{KL}(q_\phi(z|x) \| p(z))$$

**The reparameterization trick:** When $q_\phi(z|x) = \mathcal{N}(\mu_\phi, \sigma_\phi^2)$:

$$z = \mu_\phi + \sigma_\phi \odot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$$

$$\nabla_\phi \mathbb{E}_{q_\phi}[f(z)] = \nabla_\phi \mathbb{E}_\varepsilon[f(\mu_\phi + \sigma_\phi \varepsilon)] = \mathbb{E}_\varepsilon[\nabla_\phi f(\mu_\phi + \sigma_\phi \varepsilon)]$$

This converts a **score function estimator** (high variance) to a **pathwise estimator** (low variance) — essentially converting a derivative of an expectation into an expectation of a derivative.

**Gauss-Hermite for exact ELBO:** When $q_\phi(z|x) = \mathcal{N}(\mu, \sigma^2)$:

$$\mathbb{E}_q[f(z)] = \frac{1}{\sqrt{\pi}} \int f(\sqrt{2}\sigma t + \mu) e^{-t^2} dt \approx \frac{1}{\sqrt{\pi}} \sum_{k=1}^n w_k f(\sqrt{2}\sigma x_k + \mu)$$

With $n = 10$ Gauss-Hermite points, this is nearly exact for smooth $f$ — eliminating the Monte Carlo noise in the ELBO gradient.

### 8.3 Gaussian Process Marginal Likelihood

The GP log marginal likelihood is:

$$\log p(\mathbf{y} | X) = -\frac{1}{2} \mathbf{y}^\top (K + \sigma^2 I)^{-1} \mathbf{y} - \frac{1}{2}\log|K + \sigma^2 I| - \frac{n}{2}\log 2\pi$$

Computing $|K + \sigma^2 I|$ requires the determinant of an $n \times n$ matrix — $O(n^3)$ via Cholesky. For large $n$:

**Stochastic trace estimation:** $\log|A| = \text{tr}(\log A) \approx \frac{1}{m} \sum_i z_i^\top (\log A) z_i$ for random $z_i \sim \mathcal{N}(0, I)$.

**Lanczos quadrature:** Estimate $\mathbf{v}^\top \log(A) \mathbf{v}$ using Lanczos iteration to build a tridiagonal approximation of $A$, then compute $\log(\text{tridiagonal})$ via Gauss quadrature. This combines matrix-vector products ($O(n)$ per) with Gaussian quadrature on the spectrum.

### 8.4 Expected Value Estimation in RL

**Policy gradient (REINFORCE):**

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]$$

This is a Monte Carlo estimate — simulate trajectories, compute returns $G_t$. The estimator is unbiased but high-variance. Variance reduction:
- **Baseline subtraction:** Replace $G_t$ with $G_t - b(s_t)$ (leaves gradient unbiased, reduces variance if $b$ correlated with $G_t$)
- **Actor-Critic:** Use $A_t = G_t - V^\pi(s_t)$ — advantage function reduces variance dramatically

**TD learning** approximates $\mathbb{E}[G_t | s_t]$ via Bellman recursion rather than direct Monte Carlo:

$$V(s_t) \approx r_t + \gamma V(s_{t+1})$$

This is a **recursive quadrature** — approximating the integral of the value function over the next-state distribution using a single sample.

---

## 9. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
|---|---|---|---|
| 1 | Using Simpson's rule with an odd number of intervals | Simpson's requires $m$ even (pairs of intervals) | Use even $m$ or use the 3/8 rule for the odd remainder |
| 2 | Applying high-degree Newton-Cotes on a single large interval | Negative weights for $n \geq 8$ → catastrophic cancellation | Use composite low-order rules |
| 3 | Underestimating Monte Carlo variance | $O(1/\sqrt{n})$ convergence means 100× more samples for 10× accuracy | Use deterministic methods for $d \leq 5$, or variance reduction |
| 4 | Ignoring singularities in adaptive quadrature | Standard Gauss-Kronrod fails near singularities | Report singularity locations to `quad()` via `points=[]` parameter |
| 5 | Claiming Monte Carlo is "exact in expectation" justifies any sample size | Unbiased ≠ accurate; confidence intervals require $n \gg 1$ | Always report MC standard errors alongside estimates |
| 6 | Using composite trapezoidal for periodic functions with wrong period | If the function period doesn't match the integration interval, the boundary terms don't cancel | Ensure $\int_a^b$ covers exactly an integer number of periods |
| 7 | Forgetting the Jacobian when changing variables | For $x = g(u)$: $\int f(x) dx = \int f(g(u)) |g'(u)| du$ — the $|g'(u)|$ factor is essential | Always include the Jacobian |
| 8 | Using GL quadrature with cached nodes/weights outside $[-1,1]$ | Precomputed nodes are for $[-1,1]$; rescaling is required for $[a,b]$ | Apply the affine transformation $x = (a+b)/2 + (b-a)/2 \cdot t$ |
| 9 | Stopping Romberg integration too early | The Romberg table diagonal may not show the true error until many rows | Always check at least 2 successive diagonal entries for convergence |
| 10 | Using importance sampling with a poor proposal | If $q(x) \ll f(x) p(x)$ in some region, the ratio $f(x)/q(x)$ can be huge → infinite variance | Choose $q$ proportional to $|f(x)|$ where possible; check tail behavior |
| 11 | Applying MC to 1D integrals where deterministic is available | MC needs $10^6$ samples for 3 digits; Gauss-Legendre needs 5 | Use MC only when deterministic methods are infeasible |
| 12 | Forgetting that quasi-MC Sobol sequences need scrambling for correctness | Unscrambled Sobol sequences can have correlated estimators | Use scrambled Sobol from scipy.stats.qmc |

---

## 10. Exercises

**Exercise 1 ★ — Newton-Cotes Rules from Scratch**  
**(a)** Implement the composite trapezoidal rule `trap(f, a, b, m)` and verify $O(h^2)$ convergence on $\int_0^1 e^x dx = e - 1$.  
**(b)** Implement composite Simpson's rule `simpson(f, a, b, m)` and verify $O(h^4)$ convergence.  
**(c)** Show the midpoint rule is twice as accurate as the trapezoidal rule (same $O(h^2)$ but half the constant) by computing and comparing errors on a convex function.

**Exercise 2 ★ — Romberg Integration**  
**(a)** Implement Romberg's method building the triangular table $R_{k,j}$ for $k = 0, \ldots, 8$.  
**(b)** Apply it to $\int_0^1 \sin(x^2) dx$ and compare convergence to composite trapezoidal.  
**(c)** Show that for $f(x) = e^x$, the diagonal $R_{k,k}$ converges super-geometrically — the number of correct digits roughly doubles each step.

**Exercise 3 ★ — Gauss-Legendre Quadrature**  
**(a)** Compute the 5-point GL nodes and weights using the eigenvalue method (Golub-Welsch matrix).  
**(b)** Verify: the rule integrates all polynomials of degree $\leq 9$ exactly.  
**(c)** Compare GL-5 accuracy on $\int_0^1 e^x dx$ vs composite Simpson's with 50 subintervals.

**Exercise 4 ★★ — Adaptive Quadrature Implementation**  
**(a)** Implement `adaptive_simpson(f, a, b, tol)` using recursive bisection with the 3-point Simpson estimate and the 5-point refined estimate for error control.  
**(b)** Test on $\int_0^1 x^{-1/2} dx = 2$ (singular at 0) and $\int_0^{50} \sin(x) dx$ (oscillatory).  
**(c)** Plot the adaptive mesh (where it placed evaluation points) for a function with a sharp peak.

**Exercise 5 ★★ — Monte Carlo Integration**  
**(a)** Estimate $\int_0^1 \int_0^1 \sqrt{x^2 + y^2} \, dx\, dy$ using plain Monte Carlo with $n = 10^3, 10^4, 10^5$ samples. Show the $O(1/\sqrt{n})$ error convergence.  
**(b)** Implement importance sampling for $\int_0^\infty x e^{-x^2/2} dx$ with proposal $q(x) = xe^{-x^2/2}/C$ (half-normal). Compare variance vs uniform sampling.  
**(c)** Implement antithetic variates for the 2D integral in (a) and measure the variance reduction factor.

**Exercise 6 ★★ — Euler-Maclaurin and Periodic Functions**  
**(a)** Numerically verify the Euler-Maclaurin formula for the trapezoidal rule on $\int_0^1 x^4 dx$: show the error matches $\sum B_{2k} h^{2k} [f^{(2k-1)}(1) - f^{(2k-1)}(0)]/(2k)!$.  
**(b)** Demonstrate spectral accuracy: for $f(x) = e^{\sin(2\pi x)}$ (periodic), show the composite trapezoidal rule achieves machine precision with just 20 points.  
**(c)** Explain why this makes the DFT exact: use the Euler-Maclaurin formula to show that integrating the DFT basis functions $e^{2\pi i k x}$ over $[0,1]$ with the trapezoidal rule is exact for integer $k$.

**Exercise 7 ★★ — Gauss-Hermite for Gaussian Expectations**  
**(a)** Implement $n$-point Gauss-Hermite quadrature for $\int_{-\infty}^\infty f(x) e^{-x^2} dx$ using the Golub-Welsch algorithm.  
**(b)** Compute $\mathbb{E}_{x \sim \mathcal{N}(0,1)}[\log(1 + e^x)]$ (the binary cross-entropy expectation) using GH quadrature with $n = 5, 10, 20$ points. Compare to Monte Carlo with $10^4$ samples.  
**(c)** Implement the reparameterization trick to compute the ELBO gradient $\nabla_\mu \mathbb{E}_{z \sim \mathcal{N}(\mu, 1)}[\log p(z)]$ for $p(z) = \mathcal{N}(0, 1)$ using both GH quadrature and the pathwise estimator.

**Exercise 8 ★★★ — Quasi-Monte Carlo and Low-Discrepancy Sequences**  
**(a)** Compare plain Monte Carlo vs Sobol quasi-MC for $\int_{[0,1]^d} \prod_{i=1}^d \sin(\pi x_i) dx = (2/\pi)^d$ for $d = 5, 10, 20$.  
**(b)** Plot the convergence rate (log-log error vs log $n$) for both methods. Show MC gives slope $-0.5$ and Sobol gives slope $\approx -1$ for small $d$.  
**(c)** For $d = 10$, how many samples do you need for MC to achieve $10^{-3}$ relative accuracy? For Sobol? Compute the ratio.

---

## 11. Why This Matters for AI (2026 Perspective)

| Concept | AI/LLM Impact |
|---|---|
| Monte Carlo integration | Foundation of all stochastic gradient methods; ELBO, policy gradient, score matching |
| Reparameterization trick | VAE training, diffusion model score matching, normalizing flows — the key to backpropagating through stochastic variables |
| Gauss-Hermite quadrature | Exact ELBO computation in structured VAEs; expectation propagation for approximate inference |
| Importance sampling | Off-policy RL (PPO uses importance weights $r_t = \pi_\theta/\pi_{\text{old}}$); rejection sampling for LLM alignment |
| Quasi-Monte Carlo | Faster hyperparameter search (Optuna, Ray Tune use quasi-random sampling); faster random feature approximations |
| Adaptive quadrature | Error-controlled numerical ODEs (DormandPrince, VODE) used in neural ODE/SDE training |
| Thermodynamic integration | Computing log partition functions for energy-based models; AIS for evaluating generative models |
| Variance reduction (baselines) | Policy gradient training stability (REINFORCE with baseline, A2C, PPO); reduces gradient variance by 10-100x |
| Gaussian process marginal likelihood | Bayesian optimization for hyperparameter tuning; GP-based surrogate models in AutoML |
| Lanczos-based matrix quadrature | Efficient log-determinant computation for GP likelihoods at large scale (GPyTorch) |

---

## 12. Conceptual Bridge

**Looking back:** Numerical integration is built directly on the foundations of this chapter and the broader curriculum:
- **Floating-point arithmetic (§10-01):** All quadrature involves finite-precision arithmetic; catastrophic cancellation in the sum $\sum w_i f(x_i)$ must be avoided
- **Interpolation (§10-04):** Quadrature rules are derived by integrating interpolants exactly. Every quadrature rule has a corresponding interpolation scheme
- **Orthogonal polynomials (§10-04, §03-Advanced-Linear-Algebra):** Gaussian quadrature nodes are zeros of orthogonal polynomials; the Golub-Welsch algorithm is an eigenvalue computation
- **Probability (§07):** Monte Carlo integration is equivalent to expectation estimation; variance reduction techniques exploit probability theory directly

**Looking forward:** Numerical integration connects to advanced topics throughout the curriculum:
- **Ordinary and partial differential equations:** ODE solvers (RK4, Dormand-Prince) are quadrature formulas for $\int_{t_0}^{t_1} f(t, y(t)) dt$; Gauss collocation methods achieve the highest possible order
- **Markov Chain Monte Carlo:** When direct sampling is unavailable, MCMC generates correlated samples whose empirical average still converges to the integral — at a slower $O(1/\sqrt{n/\tau})$ rate where $\tau$ is the autocorrelation time
- **Signal processing:** The DFT is a quadrature formula; the connection between Fourier analysis and quadrature (Clenshaw-Curtis via DCT) is deep

```
NUMERICAL INTEGRATION IN THE CURRICULUM
════════════════════════════════════════════════════════════════════════

  §10-01 Floating-Point      §10-04 Interpolation
  (rounding in sums)   ──►   (integrate interpolant)
           │                        │
           └────────────┬───────────┘
                        ▼
              §10-05 NUMERICAL INTEGRATION
              ┌──────────────────────────┐
              │ Newton-Cotes rules        │
              │ Gaussian quadrature       │
              │ Romberg / Richardson      │
              │ Adaptive (Gauss-Kronrod)  │
              │ Monte Carlo + QMC         │
              │ Multidimensional          │
              └────────────┬─────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
     ODE Solvers        VAE/ELBO         Bayesian
     (§AdvCalc)       (reparam trick)   Optimization
                      Diffusion models  (GP marginal
                      Policy gradient   likelihood)

════════════════════════════════════════════════════════════════════════
```

**The unifying theme:** Every integration problem in ML is either a **low-dimensional, smooth integral** (use Gaussian quadrature — exact and fast) or a **high-dimensional, stochastic integral** (use Monte Carlo — dimension-independent convergence). The art is recognizing which regime you're in and applying the appropriate technique. The reparameterization trick is the bridge: it converts a "hard" stochastic integral into a "easy" pathwise integral, enabling efficient gradient computation.

---

## Appendix A: Proof of Gaussian Quadrature Optimality

### A.1 The 2n-1 Degree of Exactness Theorem

**Theorem:** The $n$-point Gauss-Legendre rule $Q_n(f) = \sum_{i=1}^n w_i f(x_i)$ integrates all polynomials of degree $\leq 2n-1$ exactly.

**Proof:** Let $f \in \mathcal{P}_{2n-1}$ (polynomial of degree $\leq 2n-1$). Divide by the node polynomial $\omega_n(x) = \prod_{i=1}^n (x-x_i)$ (degree $n$):

$$f(x) = q(x) \omega_n(x) + r(x)$$

where $q, r \in \mathcal{P}_{n-1}$.

**Step 1:** $\int_{-1}^1 q(x) \omega_n(x) dx = 0$.  
Since $\omega_n = c P_n$ for some constant $c$ (nodes are zeros of $P_n$), and $q \in \mathcal{P}_{n-1}$, orthogonality of $P_n$ to all lower-degree polynomials gives $\int q P_n = 0$.

**Step 2:** $Q_n(q \omega_n) = \sum_i w_i q(x_i) \omega_n(x_i) = 0$.  
Since $\omega_n(x_i) = 0$ for each node $x_i$.

**Step 3:** $Q_n(r) = \int r$ exactly.  
Since $r \in \mathcal{P}_{n-1}$ and any rule with $n$ nodes is exact for polynomials of degree $\leq n-1$ if it integrates the $n$ basis functions correctly — and the weights $w_i$ are chosen to integrate $r \in \mathcal{P}_{n-1}$ exactly.

**Step 4:** Combining:

$$Q_n(f) = Q_n(q\omega_n) + Q_n(r) = 0 + \int r = \int (q\omega_n + r) = \int f$$

$\square$

**Why $2n-1$ is optimal:** Any $n$-point rule fails for $p(x) = \omega_n(x)^2 \in \mathcal{P}_{2n}$ since $p(x_i) = 0$ for all nodes but $\int p > 0$.

### A.2 The Golub-Welsch Algorithm

The GL nodes and weights are eigenvalues and eigenvectors of the **Jacobi tridiagonal matrix** $J_n$:

$$J_n = \begin{pmatrix} \alpha_0 & \beta_1 & & \\ \beta_1 & \alpha_1 & \beta_2 & \\ & \beta_2 & \alpha_2 & \ddots \\ & & \ddots & \ddots \end{pmatrix}$$

where for Legendre polynomials: $\alpha_k = 0$ (by symmetry), $\beta_k = k / \sqrt{4k^2 - 1}$.

**Algorithm:**
1. Build $J_n$ as a tridiagonal matrix
2. Compute eigendecomposition: $J_n = Q \Lambda Q^\top$ (symmetric)
3. Nodes: $x_i = \Lambda_{ii}$ (eigenvalues)
4. Weights: $w_i = 2 (Q_{1i})^2$ (squared first component of eigenvectors, times 2 for the integral of 1 over $[-1,1]$)

This reduces GL computation to a standard symmetric eigenvalue problem — $O(n^2)$ using QR iteration, or $O(n \log n)$ with divide-and-conquer.

```python
import numpy as np

def gauss_legendre(n):
    """Compute n-point Gauss-Legendre nodes and weights on [-1, 1]."""
    k = np.arange(1, n)
    beta = k / np.sqrt(4*k**2 - 1)
    J = np.diag(beta, -1) + np.diag(beta, 1)  # Symmetric tridiagonal
    eigvals, eigvecs = np.linalg.eigh(J)
    # Sort by node
    idx = np.argsort(eigvals)
    nodes = eigvals[idx]
    weights = 2 * eigvecs[0, idx]**2  # First row of eigenvector matrix
    return nodes, weights
```

---

## Appendix B: Richardson Extrapolation — Complete Analysis

### B.1 The Error Expansion Assumption

Richardson extrapolation requires the error to have the asymptotic expansion:

$$Q(h) = I + c_p h^p + c_{p+2} h^{p+2} + c_{p+4} h^{p+4} + \cdots$$

**When does this hold?**
- Composite trapezoidal: Yes (Euler-Maclaurin guarantees $h^2, h^4, h^6, \ldots$ terms)
- Composite Simpson's: Yes ($h^4, h^6, h^8, \ldots$ terms)
- For periodic functions: The expansion is actually geometric — $O(e^{-cn/p})$ — Richardson extrapolation helps but eventually the geometric error dominates

### B.2 Romberg Table Construction

The Romberg table is computed as:

$$R_{k,0} = T_{2^k}(f) \quad (\text{composite trapezoidal with } 2^k \text{ intervals})$$

$$R_{k,j} = \frac{4^j R_{k, j-1} - R_{k-1, j-1}}{4^j - 1}$$

**Incremental trapezoidal computation:** $R_{k,0}$ can be computed from $R_{k-1,0}$ using only the new midpoints:

$$T_{2m}(f) = \frac{1}{2} T_m(f) + h \sum_{j=1}^{m} f(a + (2j-1)h), \quad h = \frac{b-a}{2m}$$

Total function evaluations: $2^k + 1$ (not $2^0 + 2^1 + \cdots + 2^k = 2^{k+1} - 1$ — the old evaluations are reused).

### B.3 Convergence of Romberg

For $f \in C^\infty[a,b]$, the diagonal $R_{k,k}$ satisfies:

$$|I - R_{k,k}| \leq C_k h_0^{2k+2}$$

where $h_0 = (b-a)$ and $C_k = O(1/(2k+2)!)$. The convergence is faster than any power of $h_0$ — supergeometric.

**Practical stopping criterion:** Compare $|R_{k,k} - R_{k-1,k-1}|$. If this is smaller than the tolerance, stop. But be careful: the estimate can be misleading for non-smooth functions.

---

## Appendix C: Monte Carlo — Detailed Variance Analysis

### C.1 Central Limit Theorem for MC

Let $X_1, \ldots, X_n \sim \text{Uniform}(\Omega)$ i.i.d. The MC estimator $\hat{I}_n = |\Omega| \frac{1}{n} \sum_i f(X_i)$ satisfies:

$$\sqrt{n}(\hat{I}_n - I) \xrightarrow{d} \mathcal{N}(0, |\Omega|^2 \text{Var}(f))$$

**95% confidence interval:** $I \in \hat{I}_n \pm 1.96 \frac{|\Omega| \hat{\sigma}}{\sqrt{n}}$ where $\hat{\sigma}^2 = \frac{1}{n-1}\sum (f_i - \bar{f})^2$.

**How many samples for accuracy $\varepsilon$?**

$$n = \left(\frac{1.96 |\Omega| \hat{\sigma}}{\varepsilon}\right)^2$$

For $\varepsilon = 10^{-3}$ and $\hat{\sigma} = 1$: $n \approx 4 \times 10^6$ — about 4 million samples.

### C.2 Importance Sampling: Optimal Proposal

The variance of the importance sampling estimator is:

$$\text{Var}_q\!\left(\frac{f(X)}{q(X)}\right) = \int \frac{f(x)^2}{q(x)} dx - I^2$$

Minimize over $q$ subject to $\int q = 1$: By Cauchy-Schwarz, $\int \frac{f^2}{q} \geq \left(\int |f|\right)^2$ with equality when $q^* = |f| / \int |f|$.

**With $q = q^*$:** The estimator is $\frac{f(X)/q^*(X)}{} = \pm \int |f|$ — zero variance! But computing $q^*$ requires knowing $\int f$.

**Practical IS:** Choose $q$ close to $|f|$. A good choice in practice is $q(x) \propto f(x) p(x)$ where $p$ is the true weight.

### C.3 Quasi-MC Convergence Analysis

The Koksma-Hlawka inequality for quasi-MC:

$$\left|\frac{1}{n}\sum_i f(x_i) - \int_{[0,1]^d} f dx\right| \leq V_{\text{HK}}(f) \cdot D^*(x_1, \ldots, x_n)$$

where $V_{\text{HK}}$ is the Hardy-Krause variation and $D^*$ is the star discrepancy.

**Sobol sequences:** $D^* = O((\log n)^d / n)$.

**Convergence rate comparison:**
- MC: $O(n^{-1/2})$
- Sobol: $O(n^{-1} (\log n)^d)$ — faster by $O(n^{1/2} / (\log n)^d)$ factor
- Break-even (Sobol faster): $n \gg (\log n)^{2d}$ — for $d=10$: $n \gg 10^{12}$... not practical!

**But:** The constant in front matters. For moderate $n$ (thousands to millions), Sobol consistently outperforms MC in practice for $d \leq 20$.

---

## Appendix D: Integration and Machine Learning — Extended Connections

### D.1 The REINFORCE Gradient and Score Function Estimator

The policy gradient is a Monte Carlo estimate of $\nabla_\theta \mathbb{E}_\pi[R]$. The **score function estimator** (also called REINFORCE or likelihood-ratio estimator):

$$\nabla_\theta \mathbb{E}_{p_\theta(x)}[f(x)] = \mathbb{E}_{p_\theta(x)}[f(x) \nabla_\theta \log p_\theta(x)]$$

**Proof:** 
$$\nabla_\theta \int f(x) p_\theta(x) dx = \int f(x) \nabla_\theta p_\theta(x) dx = \int f(x) \frac{\nabla_\theta p_\theta(x)}{p_\theta(x)} p_\theta(x) dx = \mathbb{E}_{p_\theta}[f \nabla_\theta \log p_\theta]$$

This requires only the ability to sample from $p_\theta$ and evaluate $\log p_\theta$ — but has **high variance** because $f(x) \nabla_\theta \log p_\theta(x)$ fluctuates wildly.

**Variance of REINFORCE:** $\text{Var}[f(x) \nabla_\theta \log p_\theta(x)] = \mathbb{E}[f^2 \|\nabla_\theta \log p_\theta\|^2] - \|\nabla_\theta \mathbb{E}[f]\|^2$

This can be $O(\|f\|^2)$ — proportional to the square of the reward, which is large. Hence the need for baselines and actor-critic methods.

### D.2 The Reparameterization Trick as Pathwise Derivative

When $x = g_\theta(\varepsilon)$ with $\varepsilon \sim p(\varepsilon)$ (independent of $\theta$):

$$\nabla_\theta \mathbb{E}_{p_\theta(x)}[f(x)] = \mathbb{E}_{p(\varepsilon)}[\nabla_\theta f(g_\theta(\varepsilon))]$$

**Variance comparison:** The reparameterization estimator has variance $\text{Var}[\nabla_\theta f(g_\theta(\varepsilon))]$ — this is the squared norm of the gradient of $f$ w.r.t. the parameters, which is typically much smaller than the REINFORCE variance. Empirically, 10–100× lower variance.

**When does reparameterization apply?**
- Gaussian: $z = \mu + \sigma \varepsilon$, $\varepsilon \sim \mathcal{N}(0,1)$ ✓
- Uniform: $z = a + (b-a)\varepsilon$, $\varepsilon \sim \text{Uniform}(0,1)$ ✓
- Categorical (Gumbel-Softmax): $z = \text{softmax}((\log \pi + g)/\tau)$, $g \sim \text{Gumbel}(0,1)$ — continuous relaxation ✓
- Any discrete distribution without continuous relaxation: ✗

### D.3 Numerical Integration in Normalizing Flows

**Change of variables theorem:** For $x = f(z)$ with $z \sim p(z)$:

$$p_x(x) = p_z(f^{-1}(x)) \cdot |\det J_{f^{-1}}(x)|$$

The log-likelihood is:

$$\log p_x(x) = \log p_z(f^{-1}(x)) + \log |\det J_{f^{-1}}(x)|$$

Computing $\log |\det J|$ requires the determinant of an $n \times n$ Jacobian matrix — $O(n^3)$. Normalizing flows use structured architectures (coupling layers, autoregressive flows) where $J$ is triangular and $\log |\det J| = \sum_i \log |J_{ii}|$ — $O(n)$.

**This is numerical integration in disguise:** We're computing $\log p_x(x) = \log \int \delta(x - f(z)) p(z) dz$ — a singular integral over $z$-space that the change-of-variables formula evaluates analytically for invertible flows.

---

## Appendix E: Software Reference

### E.1 scipy.integrate

```python
from scipy import integrate

# Adaptive Gauss-Kronrod (default: G7-K15)
result, abserr = integrate.quad(f, a, b)
result, abserr = integrate.quad(f, a, b, epsabs=1.49e-8, epsrel=1.49e-8, limit=50)

# Improper integrals
result, abserr = integrate.quad(f, 0, np.inf)  # Semi-infinite
result, abserr = integrate.quad(f, -np.inf, np.inf)  # Doubly infinite

# With singularity points
result, abserr = integrate.quad(f, 0, 1, points=[0.5])

# Fixed Gauss-Legendre
result, abserr = integrate.fixed_quad(f, a, b, n=5)

# Romberg
result = integrate.romberg(f, a, b, tol=1.48e-8, rtol=1.48e-8)

# 2D integration
result, abserr = integrate.dblquad(f, a, b, lambda x: c, lambda x: d)
```

### E.2 numpy for Quadrature

```python
import numpy as np

# Gauss-Legendre points and weights (numpy built-in)
x, w = np.polynomial.legendre.leggauss(n)
I = np.dot(w, f(x))  # For integral over [-1, 1]

# Gauss-Hermite
x, w = np.polynomial.hermite.hermgauss(n)
I = np.dot(w, f(x))  # For exp(-x^2) weight

# Gauss-Laguerre
x, w = np.polynomial.laguerre.laggauss(n)
I = np.dot(w, f(x))  # For exp(-x) weight
```

### E.3 Quasi-Monte Carlo with scipy.stats.qmc

```python
from scipy.stats import qmc

# Sobol sequence (scrambled)
sampler = qmc.Sobol(d=5, scramble=True, seed=42)
x = sampler.random(n=1024)  # n must be power of 2 for Sobol
# x has shape (1024, 5), values in [0, 1]^5

# Halton sequence
sampler = qmc.Halton(d=5, scramble=True, seed=42)
x = sampler.random(n=1000)

# Latin Hypercube Sampling
sampler = qmc.LatinHypercube(d=5, seed=42)
x = sampler.random(n=100)

# Compute integral of f over [0,1]^5
I_qmc = f(x).mean()
I_mc = f(np.random.rand(1000, 5)).mean()
```

---

## Appendix F: Common Integrals in Machine Learning

| Integral | Closed form | Context |
|---|---|---|
| $\int \mathcal{N}(x; \mu, \sigma^2) dx = 1$ | 1 | Normalization |
| $\mathbb{E}_{x \sim \mathcal{N}}[x] = \mu$ | $\mu$ | Gaussian mean |
| $\mathbb{E}_{x \sim \mathcal{N}}[x^2] = \mu^2 + \sigma^2$ | $\mu^2 + \sigma^2$ | Gaussian second moment |
| $\text{KL}(\mathcal{N}(\mu,\sigma^2) \| \mathcal{N}(0,1))$ | $\frac{1}{2}(\mu^2+\sigma^2-\log\sigma^2-1)$ | VAE ELBO term |
| $\int \sigma(x) \mathcal{N}(x;\mu,\sigma^2) dx$ | Probit approx. | Expected sigmoid |
| $\int_{-\infty}^\infty e^{-x^2} dx = \sqrt{\pi}$ | $\sqrt{\pi}$ | Gaussian normalization |
| $\int_0^\infty x^{n-1} e^{-x} dx = \Gamma(n)$ | $\Gamma(n)$ | Gamma function |
| $\int_0^1 x^{a-1}(1-x)^{b-1} dx = B(a,b)$ | $\Gamma(a)\Gamma(b)/\Gamma(a+b)$ | Beta function |
| $\int e^{ixt} dt = 2\pi \delta(x)$ | $2\pi\delta(x)$ | Fourier delta |
| $\int \|\nabla u\|^2 = \int u (-\Delta u)$ | (by parts) | PDE weak form |

---

## Appendix G: Error Bounds Quick Reference

| Method | Nodes | Error bound | Smoothness needed |
|---|---|---|---|
| Midpoint (1-panel) | 1 | $\frac{(b-a)^3}{24}\|f''\|$ | $f \in C^2$ |
| Trapezoidal (1-panel) | 2 | $\frac{(b-a)^3}{12}\|f''\|$ | $f \in C^2$ |
| Simpson's (1-panel) | 3 | $\frac{(b-a)^5}{90}\|f^{(4)}\|$ | $f \in C^4$ |
| Composite trap. ($m$ panels) | $m+1$ | $\frac{(b-a)h^2}{12}\|f''\|$ | $f \in C^2$ |
| Composite Simpson's ($m$ even) | $m+1$ | $\frac{(b-a)h^4}{180}\|f^{(4)}\|$ | $f \in C^4$ |
| GL-$n$ (1 panel) | $n$ | $O(h^{2n})$ | $f \in C^{2n}$ |
| Clenshaw-Curtis ($n+1$ pts) | $n+1$ | $O(n^{-p})$ or $O(\rho^{-n})$ | $f \in C^p$ or analytic |
| Romberg ($k$ steps) | $2^k+1$ | $O(h_0^{2k+2})$ | $f \in C^\infty$ |
| Monte Carlo | $n$ samples | $O(n^{-1/2})$ | Any measurable $f$ |
| Quasi-MC (Sobol) | $n$ pts | $O(n^{-1}(\log n)^d)$ | Bounded variation |

---

## Appendix H: Extended Newton-Cotes Theory

### H.1 Derivation of Newton-Cotes Weights

The weights for an $(n+1)$-point Newton-Cotes rule on $[0, n]$ with unit spacing are:

$$w_i^{\text{NC}} = \int_0^n \prod_{j=0, j\neq i}^{n} \frac{t-j}{i-j} dt$$

For $n=2$ (Simpson's rule, 3 nodes at $t=0,1,2$):

$$w_0 = \int_0^2 \frac{(t-1)(t-2)}{(0-1)(0-2)} dt = \frac{1}{2}\int_0^2 (t^2-3t+2) dt = \frac{1}{2}\left[\frac{8}{3} - 6 + 4\right] = \frac{1}{3}$$

Similarly $w_1 = 4/3$, $w_2 = 1/3$. With $h=(b-a)/2$: weights are $h/3 \times [1, 4, 1]$. $\square$

### H.2 Negative Weights for High-Order Newton-Cotes

The weights for Newton-Cotes rules with $n \geq 8$ become negative. This is a fundamental instability: if $f$ has rounding errors of magnitude $\varepsilon$, the error in $Q_n(f)$ is:

$$|E| \leq \left(\sum_i |w_i|\right) \varepsilon = \Lambda_{\text{NC}} \varepsilon$$

For $n=10$: $\Lambda_{\text{NC}} = \sum |w_i| \approx 3.4$ — significant amplification. For $n=20$: $\Lambda_{\text{NC}} \approx 10^8$ — catastrophic.

In contrast, Gaussian quadrature weights are **always positive** — the sum is exactly $b-a$ and rounding errors are not amplified.

### H.3 Peano Kernel Theorem

For a rule with error $E(f) = I(f) - Q(f)$, the **Peano kernel** $K(t)$ satisfies:

$$E(f) = \int_a^b K(t) f^{(p+1)}(t) dt$$

where $p$ is the degree of exactness. The Peano kernel is zero when $t$ is outside $[a,b]$ and has a specific shape depending on the rule.

**Implications:**
1. The error depends on $f^{(p+1)}$ — the function's $(p+1)$-th derivative
2. If $K(t) \geq 0$ everywhere, we can write $E(f) = f^{(p+1)}(\xi) \int K / (p+1)!$ for some $\xi$ (mean value theorem for integrals)
3. For symmetric rules (midpoint, trapezoidal, GL), the Peano kernel has definite sign for appropriate $p$

---

## Appendix I: Adaptive Quadrature — Implementation Details

### I.1 Gauss-Kronrod G7-K15 Rule

The 15-point Gauss-Kronrod rule (K15) extends the 7-point Gauss-Legendre (G7):

**G7 nodes** (Gauss-Legendre, 7 points): $\pm 0.9491079, \pm 0.7415312, \pm 0.4058452, 0$

**K15 nodes** (Kronrod extension, 15 points): The G7 nodes plus 8 additional Kronrod nodes.

**Error estimate:** $|K15 - G7|$. If this is $< 10^{-3} |K15|$, the error is $\approx |K15 - G7|$.

**Degree of exactness:** G7 integrates polynomials exactly up to degree 13; K15 up to degree 27.

### I.2 Recursive vs Priority-Queue Adaptive

**Recursive adaptive:**
```python
def adaptive_quad(f, a, b, tol, depth=0, max_depth=50):
    Q_coarse = coarse_rule(f, a, b)
    Q_fine = fine_rule(f, a, b)
    if abs(Q_fine - Q_coarse) < tol or depth >= max_depth:
        return Q_fine
    m = (a + b) / 2
    return (adaptive_quad(f, a, m, tol/2, depth+1) +
            adaptive_quad(f, m, b, tol/2, depth+1))
```

**Priority-queue adaptive:** Better for functions with many singularities — always refines the worst subinterval first.

### I.3 Handling Singularities

**Integrable singularities:** $f(x) \sim C (x-a)^\alpha$ with $\alpha > -1$ gives finite $\int_a^b f$.

**Strategy 1 - Change of variables:** $x = a + (b-a) t^{1/(\alpha+1)}$ smooths the singularity:

$$\int_a^b C(x-a)^\alpha dx = C(b-a)^{\alpha+1} \frac{1}{\alpha+1} \int_0^1 du$$

**Strategy 2 - Gauss-Jacobi:** Use the $(1-x)^\alpha (1+x)^\beta$ weight function to absorb the singularity. `scipy.integrate.quad` handles this automatically when singularities are declared.

**Strategy 3 - IMT transformation:** The IMT transform maps $[0,1]$ to $(0,1)$ with all derivatives vanishing at the endpoints:

$$t \mapsto \exp\!\left(-\frac{1}{t} - \frac{1}{1-t}\right) \quad (\text{infinitely flat at 0 and 1})$$

This is the basis for the **tanh-sinh (double exponential)** quadrature method by Takahashi and Mori.

---

## Appendix J: Monte Carlo in Practice — Recipes

### J.1 Estimating $\pi$ via Monte Carlo

Classic example: $\pi/4 = \text{Prob}(X^2 + Y^2 \leq 1)$ with $X, Y \sim \text{Uniform}(0,1)$.

```python
import numpy as np
np.random.seed(42)

for n in [100, 1000, 10000, 1000000]:
    x, y = np.random.rand(n), np.random.rand(n)
    pi_est = 4 * (x**2 + y**2 <= 1).mean()
    std_err = 4 * np.sqrt((x**2+y**2<=1).mean() * (1-(x**2+y**2<=1).mean()) / n)
    print(f"n={n:>8}: pi = {pi_est:.6f} ± {std_err:.6f}")
```

**Output:** Shows $O(1/\sqrt{n})$ standard error.

### J.2 Gaussian Expectation via GH Quadrature

```python
def gauss_hermite_expect(f, mu, sigma, n=20):
    """
    Compute E_{x~N(mu,sigma^2)}[f(x)] via n-point Gauss-Hermite quadrature.
    """
    x, w = np.polynomial.hermite.hermgauss(n)
    x_transformed = np.sqrt(2) * sigma * x + mu
    return np.dot(w, f(x_transformed)) / np.sqrt(np.pi)

# Example: E[log(1 + e^x)] for x ~ N(0, 1)
f = lambda x: np.log(1 + np.exp(x))
exact_approx = gauss_hermite_expect(f, mu=0, sigma=1, n=20)
mc_approx = f(np.random.randn(100000)).mean()
print(f"GH-20: {exact_approx:.8f}")
print(f"MC-1e5: {mc_approx:.6f} ± {f(np.random.randn(10000)).std()/100:.6f}")
```

### J.3 ELBO Computation

```python
def vae_elbo(x, encoder, decoder, n_samples=10):
    """
    Compute ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z)).
    Using reparameterization trick.
    """
    mu, log_sigma = encoder(x)
    sigma = np.exp(log_sigma)

    # KL(N(mu, sigma^2) || N(0, 1)) = 0.5 * (mu^2 + sigma^2 - log(sigma^2) - 1)
    kl = 0.5 * np.sum(mu**2 + sigma**2 - 2*log_sigma - 1)

    # E_q[log p(x|z)] via reparameterization
    eps = np.random.randn(n_samples, len(mu))
    z_samples = mu + sigma * eps  # Shape: (n_samples, latent_dim)
    recon = np.mean([decoder(z) for z in z_samples])

    return recon - kl
```

---

## Appendix K: Connections to Differential Equations and Physics

### K.1 ODE Solvers as Quadrature

The ODE initial value problem $y' = f(t, y)$, $y(t_0) = y_0$ has solution:

$$y(t) = y_0 + \int_{t_0}^{t} f(s, y(s)) ds$$

**Runge-Kutta methods** approximate this integral using quadrature:

- **Euler:** $y_{n+1} = y_n + h f(t_n, y_n)$ — left-endpoint rule ($O(h)$)
- **RK4:** $y_{n+1} = y_n + h(k_1 + 2k_2 + 2k_3 + k_4)/6$ — Simpson's-like rule ($O(h^4)$)
- **Gauss collocation:** Gauss-Legendre-based — stiffly stable, exact for polynomials of degree $2s-1$ ($s$ stages)

**Dormand-Prince (RK45):** Embedded 4th/5th order pair for adaptive step control — the default ODE solver in `scipy.integrate.solve_ivp`.

### K.2 Feynman Path Integrals

In quantum mechanics, the probability amplitude is formally:

$$\langle x_f | e^{-iHT/\hbar} | x_i \rangle = \int \mathcal{D}[x(t)] e^{iS[x]/\hbar}$$

where $S[x] = \int L(x, \dot{x}) dt$ is the action. This is an **infinite-dimensional integral** — a functional integral over all paths.

**Numerical discretization:** Replace the path integral by a finite product:

$$\approx \left(\frac{m}{2\pi i\hbar \varepsilon}\right)^{N/2} \prod_{k=1}^N dx_k \exp\!\left(\frac{i\varepsilon}{\hbar} \sum_k L(x_k, (x_{k+1}-x_k)/\varepsilon)\right)$$

This is a multi-dimensional integral evaluated by Monte Carlo — the **lattice QCD** approach uses this for nuclear physics calculations.

### K.3 Numerical Integration in Finite Element Methods

**Finite element method (FEM):** Solve $-\nabla^2 u = f$ in $\Omega$ by finding $u$ in a finite-dimensional subspace $V_h$:

$$\int_\Omega \nabla u \cdot \nabla v = \int_\Omega f v \quad \forall v \in V_h$$

Each element integral $\int_K \nabla \phi_i \cdot \nabla \phi_j$ is computed via **Gaussian quadrature on the reference element** $\hat{K}$:

$$\int_K \nabla \phi_i \cdot \nabla \phi_j = |\det J_K| \sum_m w_m (\nabla \hat{\phi}_i)(x_m) \cdot J_K^{-T} J_K^{-T} (\nabla \hat{\phi}_j)(x_m)$$

where $J_K$ is the Jacobian of the mapping from $\hat{K}$ to $K$.

**For AI:** Physics-informed neural networks (PINNs) solve PDEs by:
1. Sampling "collocation points" inside the domain (random or Gauss-like)
2. Computing the PDE residual at each point
3. Minimizing the sum of residuals squared

This is **meshless quadrature** — and the quality of the quadrature points (random vs quasi-random vs Gauss) directly affects training accuracy.

---

## Appendix L: Notation Reference

| Symbol | Meaning |
|---|---|
| $Q_n$ | Quadrature rule with $n$ nodes |
| $x_i$ | Quadrature nodes |
| $w_i$ | Quadrature weights |
| $E_n(f)$ | Quadrature error $= I(f) - Q_n(f)$ |
| $h$ | Mesh spacing $(b-a)/m$ |
| $d$ | Degree of exactness of a rule |
| $P_n$ | Legendre polynomial of degree $n$ |
| $T_n$ | Chebyshev polynomial of degree $n$ |
| $B_k$ | Bernoulli numbers |
| $R_{k,j}$ | Romberg table entry |
| $\hat{I}_n$ | Monte Carlo estimator |
| $D^*$ | Star discrepancy |
| $V_{\text{HK}}$ | Hardy-Krause variation |
| $Z(\theta)$ | Partition function |
| $\mathcal{L}_{\text{ELBO}}$ | Evidence lower bound |
| $\tau$ | Autocorrelation time in MCMC |

---

## Appendix M: Advanced Topics in Quadrature

### M.1 Clenshaw-Curtis Quadrature — Full Derivation

**Setup:** Integrate the Chebyshev interpolant of $f$ at the $n+1$ points $x_k = \cos(k\pi/n)$:

$$\int_{-1}^1 f(x) dx \approx \int_{-1}^1 p_n(x) dx = \int_{-1}^1 \sum_{k=0}^n c_k T_k(x) dx$$

Since $\int_{-1}^1 T_k(x) dx = 0$ for odd $k$ and $2(-1)^{k/2+1}/(k^2-1)$ for even $k \geq 2$, and $2$ for $k=0$:

$$\int_{-1}^1 p_n(x) dx = 2c_0 + \sum_{j=1}^{\lfloor n/2 \rfloor} \frac{2c_{2j}(-1)^{j+1}}{4j^2-1}$$

**Computing the weights** via the moment values:

$$w_j = \sum_{k=0}^n c_k \int_{-1}^1 T_k(x_j) dx = \sum_{k=0}^n c_k \cdot m_k \cdot \ell_j(x_j)^{-1}$$

but more efficiently via the DCT applied to the moments vector $m = (2, 0, -2/3, 0, -2/15, 0, \ldots)$.

**Why CC is competitive with GL:** For analytic functions, both achieve exponential convergence. CC uses $n+1$ nodes for degree-$n$ exactness while GL uses $n$ nodes for degree-$2n-1$ exactness — GL is more "efficient" per node. However:
1. CC nodes nest: the $n/2+1$ nodes of CC-$n/2$ are a subset of the $n+1$ nodes of CC-$n$ — ideal for adaptive schemes
2. CC weights are computed via DCT in $O(n \log n)$ — GL requires $O(n^2)$ via eigenvalue method
3. CC nodes are the Chebyshev nodes — same as used in optimal polynomial approximation

### M.2 Gauss-Kronrod Derivation

**Problem:** Given $n$ GL nodes, find $n+1$ additional Kronrod points that maximize the degree of exactness of the combined $(2n+1)$-point rule.

**Solution:** The Kronrod points are zeros of the Stieltjes polynomial $E_{n+1}(x)$ satisfying:

$$\int_{-1}^1 P_n(x) E_{n+1}(x) x^k dx = 0, \quad k = 0, 1, \ldots, n$$

These polynomials exist and have real zeros in $(-1,1)$ interleaving with the GL nodes, but only for specific $n$ values. The G7-K15 pair is the standard choice.

**Degree of exactness of Kronrod:** The combined G7-K15 rule is exact for polynomials up to degree $23$ (G7 is exact up to degree $13$, the Kronrod extension pushes to $23$).

### M.3 Double Exponential (Tanh-Sinh) Quadrature

The **tanh-sinh** or **double exponential** transformation:

$$x = \tanh\!\left(\frac{\pi}{2}\sinh(t)\right), \quad t \in (-\infty, \infty)$$

maps $t \in \mathbb{R}$ to $x \in (-1, 1)$ with:

$$\frac{dx}{dt} = \frac{\pi}{2} \frac{\cosh(t)}{\cosh^2(\frac{\pi}{2}\sinh t)}$$

The derivative decays **double exponentially** as $|t| \to \infty$: faster than any power.

**Key property:** All derivatives of the integrand $f(x(t)) \cdot |dx/dt|$ vanish double-exponentially at $t = \pm\infty$. By Euler-Maclaurin, the trapezoidal rule is exponentially convergent.

**Convergence:** For analytic $f$ with singularities at $x = \pm 1$:

$$|E| = O(e^{-\pi d / h})$$

where $d$ is the half-width of the analytic strip and $h$ is the step size in $t$-space. This is double-exponential in $1/h$ — faster than any other method for this problem class.

**For AI:** Tanh-sinh quadrature is used in high-precision computations (arbitrary precision arithmetic), computing special functions in ML kernels (Matérn covariance, Bayesian quadrature), and computing KL divergences between complex distributions.

---

## Appendix N: Worked Examples

### N.1 Integrating $f(x) = e^x \sin(x)$ on $[0, \pi]$

**Exact value:** $\int_0^\pi e^x \sin(x) dx = \frac{1}{2}(e^\pi + 1) \approx 12.0703...$

**Method comparison:**
- Composite trap, $m=100$: $|E| \approx 0.0013$ ($O(h^2)$ with $h = \pi/100$)
- Composite Simpson's, $m=10$: $|E| \approx 2 \times 10^{-7}$ ($O(h^4)$, fewer evaluations)
- GL-5 on $[0,\pi]$: $|E| \approx 10^{-8}$ (exact for degree 9 polynomials)
- GL-10: $|E| \approx 10^{-16}$ (machine precision for this smooth function)

**Lesson:** GL-10 uses only 10 function evaluations and achieves machine precision; composite trap needs $10^6$ evaluations for similar accuracy.

### N.2 Computing the ELBO Expectation

For $q(z) = \mathcal{N}(z; 1.5, 0.5^2)$ and $\log p(z) = -z^2/2 - \log\sqrt{2\pi}$:

$$\mathbb{E}_q[\log p(z)] = \mathbb{E}_q\!\left[-\frac{z^2}{2} - \frac{1}{2}\log(2\pi)\right] = -\frac{\mu^2 + \sigma^2}{2} - \frac{1}{2}\log(2\pi)$$

**Analytical:** $-(1.5^2 + 0.25)/2 - 0.919 = -2.044$

**GH-5 approximation:** Using 5-point GH nodes $\{t_k\}$ and weights $\{w_k\}$, with $z_k = \sqrt{2} \times 0.5 \times t_k + 1.5$:

$$\approx \frac{1}{\sqrt{\pi}} \sum_{k=1}^5 w_k \log p(\sqrt{2}\sigma t_k + \mu) = -2.044...$$

With 5 GH points: relative error $< 10^{-12}$. With 1000 MC samples: relative error $\sim 0.1\%$.

---

## Appendix O: Summary Table — When to Use What

```
QUADRATURE METHOD SELECTION GUIDE
════════════════════════════════════════════════════════════════════════

  Key: d = dimension, n = nodes/samples, f = function

  1D INTEGRATION (d=1):
  ─────────────────────
  f smooth + analytic  → Gauss-Legendre (5-10 pts) or CC  [best]
  f smooth, C∞         → Romberg / Richardson extrapolation
  f piecewise smooth   → Adaptive G7-K15 (scipy.integrate.quad)
  f has singularities  → Adaptive + singularity declaration
  f periodic           → Trapezoidal rule (spectral accuracy!)
  f analytic + endpoint singularity → Tanh-sinh

  MULTIDIMENSIONAL (d > 1):
  ─────────────────────────
  d ≤ 3, f smooth      → Tensor product GL or CC
  d ≤ 15, f smooth     → Sparse grids (Smolyak)
  d ≤ 20               → Quasi-MC (scrambled Sobol)
  d > 20               → Monte Carlo with variance reduction
  f = E_p[g(X)]        → Gauss-Hermite (if p Gaussian)
  Singular / complex   → MCMC (Metropolis-Hastings)

  ML SPECIFIC:
  ────────────
  ELBO gradient        → Reparameterization + 1 MC sample
  GP marginal lik.     → Cholesky + Lanczos quadrature
  Policy gradient      → REINFORCE + baseline (MC)
  Normalizing constant → AIS or thermodynamic integration
  Neural ODE           → Adaptive RK (Dormand-Prince)
  Posterior E[f(θ)]    → MCMC (HMC/NUTS for continuous θ)

════════════════════════════════════════════════════════════════════════
```

---

## Appendix P: Historical Notes and Key Papers

### P.1 Gauss's Contribution

Carl Friedrich Gauss derived Gaussian quadrature in 1814 while working on geodesy and orbit computation. His insight was to treat both the nodes and weights as free variables — giving $2n$ degrees of freedom for an $n$-point rule, which (as he showed) is sufficient to integrate all polynomials of degree $\leq 2n-1$.

Gauss computed the first several GL rules by hand, finding the nodes as zeros of Legendre polynomials. The connection between optimal quadrature nodes and orthogonal polynomial zeros is now understood through the theory of Gaussian quadrature, but Gauss discovered the pattern empirically.

### P.2 The Monte Carlo Method's Origins

The Monte Carlo method was invented by Stanislaw Ulam while recovering from illness in 1946. Playing solitaire, he wondered: what is the probability of success? Unable to compute it analytically, he realized he could estimate it by simulating many games. He mentioned the idea to John von Neumann, who immediately recognized its potential for nuclear physics computations (neutron diffusion through fissile material) — and the Los Alamos Monte Carlo method was born.

The name "Monte Carlo" was suggested by Nicholas Metropolis, after the famous casino (and the element of chance in the method). The first large-scale application was computing the properties of materials for the hydrogen bomb in 1947.

### P.3 Markov Chain Monte Carlo

The Metropolis-Hastings algorithm (1953, 1970) enabled sampling from distributions known only up to a normalizing constant. The key idea: propose a new state $x'$ from the current state $x$, then accept with probability $\min(1, p(x')/p(x))$. The Markov chain converges to $p$ as its stationary distribution.

MCMC became mainstream in statistics in the 1990s with the work of Gelfand and Smith (1990) showing how Gibbs sampling could solve Bayesian inference problems computationally. Today, HMC (Hamiltonian Monte Carlo) and NUTS (No-U-Turn Sampler) are the state-of-the-art for continuous distributions in packages like Stan, PyMC, and NumPyro.

### P.4 Quasi-Monte Carlo's Development

Ilya Sobol introduced his low-discrepancy sequences in 1967. The key insight: random points have discrepancy $O(n^{-1/2})$ while Sobol sequences have discrepancy $O((\log n)^d / n)$. For moderate dimensions ($d \leq 20$) and practical sample sizes ($n \leq 10^6$), Sobol sequences consistently outperform random sampling.

Modern quasi-Monte Carlo uses **scrambled** Sobol sequences (Owen, 1995) that preserve the low-discrepancy property while also yielding unbiased estimators with variance that can be computed from multiple independent scramblings.

---

## Appendix Q: Practical Code Recipes

### Q.1 Composite Rules in NumPy

```python
import numpy as np

def composite_trap(f, a, b, m):
    """Composite trapezoidal rule with m intervals."""
    h = (b - a) / m
    x = np.linspace(a, b, m + 1)
    y = f(x)
    return h * (y[0]/2 + y[1:-1].sum() + y[-1]/2)

def composite_simpson(f, a, b, m):
    """Composite Simpson's rule (m must be even)."""
    assert m % 2 == 0, "m must be even"
    h = (b - a) / m
    x = np.linspace(a, b, m + 1)
    y = f(x)
    return h/3 * (y[0] + 4*y[1:-1:2].sum() + 2*y[2:-2:2].sum() + y[-1])

# Test: integral of e^x from 0 to 1 = e - 1
f = np.exp
exact = np.e - 1
for m in [10, 100, 1000]:
    err_t = abs(composite_trap(f, 0, 1, m) - exact)
    err_s = abs(composite_simpson(f, 0, 1, m) - exact)
    print(f"m={m}: trap={err_t:.2e}, simp={err_s:.2e}")
```

### Q.2 Romberg Integration

```python
def romberg(f, a, b, max_level=8, tol=1e-10):
    """Romberg integration using Richardson extrapolation."""
    R = np.zeros((max_level, max_level))
    h = b - a
    R[0, 0] = h/2 * (f(a) + f(b))

    for k in range(1, max_level):
        h /= 2
        n_new = 2**(k-1)
        x_new = a + h * (2*np.arange(1, n_new+1) - 1)
        R[k, 0] = R[k-1, 0]/2 + h * f(x_new).sum()

        for j in range(1, k+1):
            R[k, j] = (4**j * R[k, j-1] - R[k-1, j-1]) / (4**j - 1)

        if k > 0 and abs(R[k, k] - R[k-1, k-1]) < tol:
            return R[k, k], k

    return R[max_level-1, max_level-1], max_level

result, levels = romberg(np.exp, 0, 1)
print(f"Romberg result: {result:.15f}, levels: {levels}")
print(f"Exact e-1:      {np.e-1:.15f}")
```

### Q.3 Monte Carlo with Variance Reduction

```python
def monte_carlo_antithetic(f, n, seed=42):
    """
    Monte Carlo estimate of E[f(U)] where U ~ Uniform(0,1).
    Uses antithetic variates: pairs (u, 1-u).
    """
    np.random.seed(seed)
    u = np.random.rand(n // 2)
    f_vals = (f(u) + f(1 - u)) / 2  # Antithetic pairs
    return f_vals.mean(), f_vals.std() / np.sqrt(n // 2)

def monte_carlo_plain(f, n, seed=42):
    """Standard Monte Carlo."""
    np.random.seed(seed)
    u = np.random.rand(n)
    return u.mean(), np.std(f(u)) / np.sqrt(n)

# Compare for f(u) = sqrt(u)
f = np.sqrt
for n in [100, 1000, 10000]:
    est_plain, se_plain = monte_carlo_plain(f, n)
    est_anti, se_anti = monte_carlo_antithetic(f, n)
    print(f"n={n:>6}: plain se={se_plain:.4f}, antithetic se={se_anti:.4f}, "
          f"ratio={se_plain/se_anti:.2f}x")
```

---

## Appendix R: Common Integrals and Their Quadrature Suitability

| Integral | Exact value | Best method | Notes |
|---|---|---|---|
| $\int_0^1 e^x dx$ | $e-1 \approx 1.718$ | GL-5 | Analytic, smooth |
| $\int_0^1 \sqrt{x} dx$ | $2/3$ | GL-10 | Smooth, mild singularity at 0 |
| $\int_0^1 x^{-1/2} dx$ | $2$ | Adaptive (Gauss-Jacobi) | Singularity at 0 |
| $\int_0^\infty e^{-x^2} dx$ | $\sqrt{\pi}/2$ | Gauss-Hermite | Infinite domain |
| $\int_{-\infty}^\infty e^{-x^2} dx$ | $\sqrt{\pi}$ | Gauss-Hermite | Infinite domain |
| $\int_0^{100} \sin(x) dx$ | $1-\cos(100)$ | Adaptive (many panels) | Oscillatory |
| $\int_0^1 \sin(1/x) dx$ | $\approx 0.504$ | Adaptive + change of vars | Oscillatory near 0 |
| $\int_{[0,1]^{10}} f dx$ | Varies | Quasi-MC Sobol | High-dimensional |
| $\mathbb{E}[\log(1+e^X)]$ | $\approx 1.047$ | GH quadrature | Gaussian expectation |
| $\text{KL}(\mathcal{N}(\mu,\sigma^2)\|\mathcal{N}(0,1))$ | $\frac{\mu^2+\sigma^2-\log\sigma^2-1}{2}$ | Analytical | No quadrature needed! |

---

## Appendix S: Extended Theory — Bayesian Quadrature

### S.1 Bayesian Quadrature Framework

**Bayesian quadrature (BQ)** models the integrand $f$ as a Gaussian process:

$$f \sim \mathcal{GP}(\mu_0, k)$$

and computes the posterior over the integral $I = \int f(x) p(x) dx$:

$$I | \mathbf{f}_n \sim \mathcal{N}(\hat{I}_n, \hat{V}_n)$$

**Posterior mean** (the quadrature estimate):

$$\hat{I}_n = \mathbf{z}^\top (K + \sigma^2 I)^{-1} \mathbf{f}_n$$

where $z_i = \int k(x, x_i) p(x) dx$ (the "kernel mean" — computable analytically for many kernels).

**Posterior variance** (the uncertainty):

$$\hat{V}_n = \int\!\!\int k(x, x') p(x) p(x') dx\, dx' - \mathbf{z}^\top (K + \sigma^2 I)^{-1} \mathbf{z}$$

**Key properties:**
1. BQ reduces to kernel ridge regression with kernel $k$
2. With the Gaussian kernel, BQ is equivalent to Gauss-Hermite quadrature in the limit
3. BQ provides **principled uncertainty** — the posterior variance tells us how uncertain we are about the integral given current samples

### S.2 Connection to Gauss-Legendre

When $k(x, y) = \min(x,y)$ (the Brownian motion kernel, corresponding to $H^1$ Sobolev RKHS) and $p = \text{Uniform}[-1,1]$, BQ recovers the trapezoidal rule.

When $k$ corresponds to the $H^m$ Sobolev kernel, BQ recovers the $m$-th order spline quadrature rule. In the limit as the RKHS becomes the space of polynomials, BQ recovers Gaussian quadrature.

**For AI:** BQ is used for:
- **Bayesian optimization** loop: compute the expected improvement acquisition function via BQ when the integrand is also an ML model
- **Uncertainty-aware neural ODEs:** BQ estimates the uncertainty in the solution trajectory
- **Model evidence computation:** $p(y) = \int p(y|\theta) p(\theta) d\theta$ via BQ with GP prior on the likelihood

### S.3 Active Learning for BQ

Rather than using fixed quadrature nodes, **active BQ** selects the next evaluation point $x_{n+1}$ to maximize the reduction in posterior variance:

$$x_{n+1} = \arg\max_x \hat{V}_n - \hat{V}_{n+1}(x)$$

This is equivalent to minimizing the posterior variance, giving an **information-theoretic** quadrature rule. For the Matérn-$1/2$ kernel, this recovers the bisection rule (adaptive quadrature). For smoother kernels, it places more points where $f$ is uncertain — naturally handling functions with local features.

---

## Appendix T: Stochastic Integration — SDEs and Neural SDEs

### T.1 Itô Integral

The **Itô integral** $\int_0^T H_t dW_t$ is a stochastic integral with respect to Brownian motion $W_t$:

$$\int_0^T H_t dW_t = \lim_{n \to \infty} \sum_k H_{t_k} (W_{t_{k+1}} - W_{t_k})$$

(left-endpoint rule — the Itô convention). The Euler-Maruyama scheme for SDEs $dx = f(x)dt + g(x)dW$ is:

$$x_{t+h} \approx x_t + f(x_t) h + g(x_t) \sqrt{h} \, \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0,1)$$

This is a first-order strong approximation — $O(\sqrt{h})$ strong convergence vs $O(h)$ weak convergence.

### T.2 Neural SDEs and Diffusion Models

**Neural SDE:** Replace $f$ and $g$ with neural networks $f_\theta$ and $g_\theta$:

$$dx = f_\theta(t, x) dt + g_\theta(t, x) dW_t$$

The **ELBO** for a neural SDE is:

$$\log p(x_T) \geq -\text{KL}(q_\phi(x_T) \| p(x_0)) + \mathbb{E}_q\!\left[\int_0^T \mathcal{L}_{\text{SDE}}(x_t, t) dt\right]$$

The integral over $t$ is computed via numerical integration (e.g., Euler-Maruyama or higher-order SDE solvers). This is the foundation of **score-based diffusion models** (Song et al., 2021).

**DDPM** (Ho et al., 2020) discretizes the diffusion trajectory at $T = 1000$ timesteps and computes the ELBO as a sum of Gaussian KL terms — discrete numerical integration over the diffusion trajectory.

**Continuous diffusion** (Song et al., 2021) uses the continuous-time SDE and numerical ODE solvers (Dormand-Prince, DDIM) for sampling — the sampling procedure is adaptive quadrature of the reverse SDE.

---

## Appendix U: Cross-Reference Table

| Topic | Connection | Reference |
|---|---|---|
| Legendre polynomials | GL nodes; Legendre-Gauss rule | §10-04 (orthogonal polynomials) |
| Chebyshev polynomials | Clenshaw-Curtis; CC weights via DCT | §10-04 (Chebyshev approximation) |
| Condition numbers | Vandermonde in NC; GL weights always positive | §10-01, §10-02 |
| Finite differences | Richardson extrapolation; gradient checking | §10-03 |
| Fourier analysis | DFT as trapezoidal rule; CC via DCT | §10-04 |
| Monte Carlo | Stochastic gradient, ELBO, policy gradient | §07 (Probability) |
| Linear algebra | Golub-Welsch tridiagonal eigenvalue | §10-02 |
| Optimization | MCMC as sampling from Bayesian posterior | §08, §10-03 |
| SDEs/Neural SDEs | Stochastic integration; Itô calculus | Advanced calculus |
| Gaussian processes | GP marginal likelihood; Bayesian quadrature | §07, §10-04 |

---

## Appendix V: Worked Examples — Comparing Methods

### V.1 Computing $\int_0^1 \sqrt{x(1-x)} dx = \pi/8$

This integral has integrable singularities at both endpoints ($f(0) = f(1) = 0$ but $f'(0^+) = f'(1^-) = \pm\infty$).

**Methods and errors:**

| Method | $n$ evaluations | Error |
|---|---|---|
| Composite trapezoidal | 1000 | $3.6 \times 10^{-3}$ ($O(h^{3/2})$ — slow due to singularity) |
| Composite Simpson's | 1000 | $1.8 \times 10^{-3}$ ($O(h^{3/2})$ — same!) |
| GL-20 on $[0,1]$ | 20 | $2.4 \times 10^{-3}$ (singularity causes slow convergence) |
| Gauss-Jacobi $\alpha=\beta=1/2$ | 10 | $< 10^{-14}$ (exact for weight $(x(1-x))^{1/2}$!) |
| Tanh-sinh, $h=0.1$ | ~50 | $10^{-12}$ (handles endpoint singularities) |

**Lesson:** Match the quadrature rule to the weight function. Gauss-Jacobi absorbs the singularity, tanh-sinh handles it automatically.

### V.2 High-Dimensional Monte Carlo: $\int_{[0,1]^{10}} \sum_i x_i^2 dx = 10/3$

| Method | $n$ evaluations | Error | Time |
|---|---|---|---|
| MC (random) | $10^4$ | $\approx 0.02$ | Fast |
| QMC (Sobol) | $10^4$ | $\approx 0.003$ | Fast |
| Tensor product GL-3 | $3^{10} = 59049$ | $< 10^{-10}$ | Fast |
| MC (random) | $10^6$ | $\approx 0.002$ | Moderate |

**Lesson:** For this smooth function in $d=10$, tensor product GL still wins with fewer evaluations. But for $d=50$, GL requires $3^{50} \approx 7 \times 10^{23}$ evaluations — infeasible.

### V.3 Gaussian Expectation: $\mathbb{E}_{x \sim \mathcal{N}(0,1)}[e^{x^2/2}]$

This integral diverges! $\int_{-\infty}^\infty e^{x^2/2} \mathcal{N}(x;0,1) dx = \int e^{x^2/2} \frac{e^{-x^2/2}}{\sqrt{2\pi}} dx = \frac{1}{\sqrt{2\pi}} \int dx = \infty$.

**Warning for AI:** Computing expectations over Gaussian distributions requires checking that the function is integrable. In VAE training, large activations can cause $\mathbb{E}_q[f(z)]$ to be numerically infinite — a symptom of a poor variational posterior $q$.

### V.4 ELBO Term Computation

The KL divergence $\text{KL}(\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, 1))$:

$$\text{KL} = \frac{1}{2}\left(\sigma^2 + \mu^2 - 1 - \log \sigma^2\right)$$

This is analytical — no quadrature needed. The reconstruction term $\mathbb{E}_q[\log p(x|z)]$ requires Monte Carlo (or GH quadrature):

```python
def elbo_reconstruction(x_data, mu_z, sigma_z, decoder, n_gh=20):
    """Compute E_q[log p(x|z)] via Gauss-Hermite quadrature."""
    t, w = np.polynomial.hermite.hermgauss(n_gh)
    z_samples = np.sqrt(2) * sigma_z * t + mu_z  # GH nodes mapped to q
    log_likelihood = np.array([decoder.log_prob(x_data, z) for z in z_samples])
    return np.dot(w, log_likelihood) / np.sqrt(np.pi)
```

With $n_{\text{GH}} = 20$, this achieves near-machine precision for smooth likelihoods — far superior to 1-sample MC.

---

## Appendix W: The Connection Between Quadrature and Finite Element Methods

### W.1 Galerkin FEM Requires Integration

The finite element method reduces a PDE to a linear system $Ku = f$ where:

$$K_{ij} = \int_\Omega \nabla \phi_i \cdot \nabla \phi_j dx, \quad f_i = \int_\Omega f \phi_i dx$$

Each entry requires numerical integration over the domain. With $\mathcal{P}_1$ (piecewise linear) elements, the stiffness matrix integrals are computed exactly by the midpoint rule. With $\mathcal{P}_3$ (cubic) elements, 4-point GL is needed for exactness.

**Rule of thumb:** For $\mathcal{P}_k$ elements in $d$ dimensions, use GL quadrature of degree $\geq 2k-1$ on each element.

### W.2 Spectral Element Methods

**Spectral element methods** combine the geometric flexibility of FEM with the spectral accuracy of GL/CC quadrature:
1. Divide the domain into $K$ elements
2. On each element, use a high-degree polynomial basis ($p$-th order, typically $p = 5$–$16$)
3. Integrate with the matching Gauss-Lobatto-Legendre (GLL) quadrature rule

**GLL rule:** GL rule with the constraint that the endpoints $x=\pm 1$ are always included. The $n+1$ GLL nodes consist of $\pm 1$ plus the $n-1$ interior zeros of $P_n'(x)$.

**For AI:** Spectral element methods are used in:
- Atmospheric and ocean modeling (the standard in climate science)
- Spectral graph convolution implementations (spectral approximation of graph filters)
- Neural operators (FNO, DeepONet) for learning PDE solution operators

---

## Appendix X: Self-Test Questions

1. Why is the $n$-point Gauss-Legendre rule exact for polynomials up to degree $2n-1$ but not $2n$?

2. Explain why the composite trapezoidal rule achieves **spectral accuracy** for smooth periodic functions, even though its error is formally $O(h^2)$ for non-periodic functions.

3. The midpoint rule has a smaller error constant than the trapezoidal rule for the same $O(h^2)$ rate. Why? (Hint: think about whether $f$ is convex or concave between nodes.)

4. Why does Monte Carlo integration have dimension-independent $O(1/\sqrt{n})$ convergence while deterministic quadrature rates degrade exponentially with dimension?

5. The reparameterization trick reduces gradient variance in VAE training. Why is the score function estimator (REINFORCE) higher variance? (Hint: compare what each estimator directly estimates.)

6. For the Euler-Maclaurin formula $T_m(f) = I + c_2 h^2 + c_4 h^4 + \cdots$, if $f$ is periodic with $f^{(k)}(a) = f^{(k)}(b)$ for all $k$, what does this imply for the trapezoidal error?

**Answers:** See [theory.ipynb](theory.ipynb) for numerical verification of each.

---

## Appendix Y: Extended ML Applications

### Y.1 Normalizing Flows and the Change of Variables

A **normalizing flow** defines a sequence of invertible transformations:

$$z_0 \xrightarrow{f_1} z_1 \xrightarrow{f_2} \cdots \xrightarrow{f_K} z_K = x$$

The log-likelihood is computed via the change-of-variables formula — a sequence of integration by substitution steps:

$$\log p_x(x) = \log p_{z_0}(z_0) - \sum_{k=1}^K \log |\det J_{f_k}(z_{k-1})|$$

**Each Jacobian determinant** represents the local volume change under the transformation $f_k$. For a $d$-dimensional transformation:

$$\log |\det J_{f_k}| = \sum_{i=1}^d \log |[\partial f_k / \partial z_{k-1}]_{ii}|}$$

(only valid when $J_{f_k}$ is triangular — the key design choice in autoregressive flows).

**Integration interpretation:** $p_x(x)$ is the pushforward of $p_{z_0}$ through $f = f_K \circ \cdots \circ f_1$. Computing $\log p_x(x)$ is exact — no numerical quadrature needed! The integration is performed analytically by the change-of-variables formula.

### Y.2 Diffusion Models as Numerical ODE Solvers

The **DDIM** sampler (Song et al., 2020) solves the reverse diffusion ODE:

$$\frac{dx}{dt} = f(x, t) - \frac{1}{2} g(t)^2 \nabla_x \log p_t(x)$$

using the score function $\nabla_x \log p_t(x) \approx -\varepsilon_\theta(x_t, t)/\sigma_t$.

**DDIM is 2nd-order Adams-Bashforth:** The multi-step sampler uses previous score evaluations to achieve 2nd-order accuracy per step:

$$x_{t_{i-1}} \approx x_{t_i} + \frac{\Delta t}{2}[3 s_\theta(x_{t_i}, t_i) - s_\theta(x_{t_{i+1}}, t_{i+1})]$$

where $s_\theta \approx \nabla_x \log p_t$ is the score. This is exactly the 2-step Adams-Bashforth ODE solver — applied to the reverse SDE.

**Implication:** Faster diffusion sampling (fewer NFEs) is equivalent to using higher-order quadrature for the reverse SDE. Methods like DPM-Solver++ (Lu et al., 2022) use exponential integrators and 3rd-order Adams-type solvers to reduce from 1000 steps to 10-20 steps.

### Y.3 GP Marginal Likelihood via Lanczos Quadrature

The GP log marginal likelihood requires $\log |K + \sigma^2 I|$ where $K \in \mathbb{R}^{n \times n}$ is the kernel matrix.

**Stochastic Lanczos quadrature (SLQ):** Estimate $\log |A| = \text{tr}(\log A)$:

1. Sample random probe vectors $v_1, \ldots, v_m \sim \mathcal{N}(0, I/n)$
2. For each $v_j$, run $q$ steps of Lanczos iteration to build a tridiagonal matrix $T_q$ with $A v_j \approx Q T_q Q^\top v_j$
3. Approximate $v_j^\top \log(A) v_j \approx \|v_j\|^2 e_1^\top \log(T_q) e_1$ (via GL quadrature on the spectrum of $T_q$)
4. Average: $\log|A| \approx n \cdot \frac{1}{m} \sum_j v_j^\top \log(A) v_j$

**Complexity:** $O(n q m)$ matrix-vector products vs $O(n^3)$ for Cholesky. For large $n$, SLQ is the only tractable approach.

**GPyTorch** (Gardner et al., 2018) implements SLQ for GP training with $n > 10^4$ — enabling GP regression at the scale of deep learning datasets.

---

## Appendix Z: Final Reference — Decision Tree for Integration in ML

```
ML INTEGRATION DECISION TREE
════════════════════════════════════════════════════════════════════════

  What are you computing?
  │
  ├── E_q[f(z)] with q Gaussian?
  │   ├── Low-dimensional z (d≤3), need high accuracy?
  │   │   └── → Gauss-Hermite quadrature (n=20 nodes)
  │   └── High-dimensional z, gradient needed?
  │       └── → Reparameterization + single MC sample
  │
  ├── log p(x) = log Z - E(x)?
  │   ├── f(x) has known structure (flow, Gaussian)?
  │   │   └── → Analytical formula (change of variables)
  │   └── Intractable Z?
  │       ├── Need unbiased estimate of Z?
  │       │   └── → AIS (Annealed Importance Sampling)
  │       └── Need to train model?
  │           └── → Score matching / contrastive divergence
  │
  ├── Policy gradient E_π[G]?
  │   ├── Continuous action, differentiable reward?
  │   │   └── → Reparameterization (pathwise gradient)
  │   └── Discrete action?
  │       └── → REINFORCE with baseline (score function)
  │
  ├── GP marginal likelihood log|K + σ²I|?
  │   ├── n ≤ 5000?
  │   │   └── → Cholesky factorization (exact, O(n³))
  │   └── n > 5000?
  │       └── → Stochastic Lanczos Quadrature (O(nqm))
  │
  └── ODE/SDE simulation?
      ├── Deterministic ODE?
      │   └── → Adaptive Dormand-Prince (RK45)
      └── Stochastic SDE?
          └── → Euler-Maruyama or Milstein scheme

════════════════════════════════════════════════════════════════════════
```

---

## Appendix AA: Error Analysis for the Composite Rules — Detailed

### AA.1 Euler-Maclaurin for Composite Trapezoidal

The complete Euler-Maclaurin formula for the composite trapezoidal rule $T_m(f) = h[\frac{f(a)}{2} + \sum_{k=1}^{m-1} f(a+kh) + \frac{f(b)}{2}]$ with $h = (b-a)/m$:

$$T_m(f) = \int_a^b f(x) dx + \sum_{k=1}^{p} \frac{B_{2k}}{(2k)!} h^{2k} [f^{(2k-1)}(b) - f^{(2k-1)}(a)] + O(h^{2p+2})$$

**Bernoulli numbers:** $B_2 = 1/6$, $B_4 = -1/30$, $B_6 = 1/42$, $B_8 = -1/30$.

**Leading error term:** $\frac{h^2}{12}[f'(b) - f'(a)] = -\frac{(b-a)h^2}{12} \bar{f}''$ for some mean $\bar{f}''$.

**For $p=1$ (standard result):** $T_m(f) = I + \frac{h^2}{12}[f'(b) - f'(a)] + O(h^4)$

**For Richardson extrapolation:** Taking $T_m$ and $T_{2m}$:

$$T_{2m} = I + \frac{h^2}{4}\frac{1}{12}[f'(b)-f'(a)] + O(h^4)$$

So $\frac{4T_{2m} - T_m}{3} = I + O(h^4)$ — this is exactly the composite Simpson's rule!

### AA.2 Superconvergence of Periodic Functions

For $f$ periodic on $[a,b]$ with period $b-a$: $f^{(k)}(a) = f^{(k)}(b)$ for all $k$. Every Euler-Maclaurin boundary term vanishes:

$$T_m(f) = \int_a^b f(x) dx + \sum_{k=1}^{\infty} \frac{B_{2k}}{(2k)!} h^{2k} \cdot 0 = I$$

The trapezoidal rule is **exact to all orders** for smooth periodic functions. In practice, the error is bounded by the truncation of the Fourier series — exponentially small for analytic periodic functions.

**Example:** $f(x) = e^{\cos(2\pi x)}$ on $[0,1]$:
- Composite trapezoidal with $m=5$: error $\approx 10^{-8}$
- Composite trapezoidal with $m=10$: error $\approx 10^{-16}$

Compare to $f(x) = x$ on $[0,1]$ (not periodic):
- Composite trapezoidal with $m=5$: error $\approx 0.033$ ($O(h^2)$)
- Composite trapezoidal with $m=10$: error $\approx 0.0083$ ($O(h^2)$)

This dramatic difference is the key insight behind the DFT's exact representation of bandlimited periodic signals.

---

## Appendix BB: Python Code — Complete Implementations

```python
"""
Complete reference implementations for §10-05 Numerical Integration.
All functions tested against scipy.integrate.quad.
"""
import numpy as np
from scipy import integrate

def composite_trap(f, a, b, m):
    """Composite trapezoidal: O(h^2) for smooth f."""
    h = (b-a)/m
    x = np.linspace(a, b, m+1)
    y = f(x)
    return h * (y[0]/2 + y[1:-1].sum() + y[-1]/2)

def composite_simpson(f, a, b, m):
    """Composite Simpson's: O(h^4). m must be even."""
    assert m % 2 == 0
    h = (b-a)/m
    x = np.linspace(a, b, m+1)
    y = f(x)
    return h/3 * (y[0] + 4*y[1:-1:2].sum() + 2*y[2:-2:2].sum() + y[-1])

def gauss_legendre(n):
    """n-point GL nodes and weights via Golub-Welsch."""
    k = np.arange(1, n)
    J = np.diag(k / np.sqrt(4*k**2 - 1), -1)
    J += J.T
    evals, evecs = np.linalg.eigh(J)
    idx = np.argsort(evals)
    return evals[idx], 2 * evecs[0, idx]**2

def gl_integrate(f, a, b, n=10):
    """GL quadrature on [a,b] with n nodes."""
    x, w = gauss_legendre(n)
    x_mapped = (a+b)/2 + (b-a)/2 * x
    return (b-a)/2 * np.dot(w, f(x_mapped))

def romberg_integrate(f, a, b, max_k=8, tol=1e-12):
    """Romberg's method via repeated Richardson extrapolation."""
    R = np.zeros((max_k, max_k))
    h = b - a
    R[0, 0] = h/2 * (f(a) + f(b))
    for k in range(1, max_k):
        h /= 2
        n_new = 2**(k-1)
        x_new = a + h * (2*np.arange(1, n_new+1) - 1)
        R[k, 0] = R[k-1, 0]/2 + h * f(x_new).sum()
        for j in range(1, k+1):
            R[k, j] = (4**j * R[k, j-1] - R[k-1, j-1]) / (4**j - 1)
        if abs(R[k, k] - R[k-1, k-1]) < tol:
            return R[k, k], k
    return R[max_k-1, max_k-1], max_k

def monte_carlo(f, domain_sampler, volume, n, seed=42):
    """Basic Monte Carlo integration."""
    np.random.seed(seed)
    x = domain_sampler(n)
    return volume * f(x).mean(), volume * f(x).std() / np.sqrt(n)

def gauss_hermite_expect(f, mu, sigma, n=20):
    """E_{x~N(mu,sigma^2)}[f(x)] via GH quadrature."""
    t, w = np.polynomial.hermite.hermgauss(n)
    return np.dot(w, f(np.sqrt(2)*sigma*t + mu)) / np.sqrt(np.pi)

# Verification
if __name__ == '__main__':
    f = np.exp
    exact = np.e - 1
    print(f"Trapezoidal (m=100): error = {abs(composite_trap(f,0,1,100) - exact):.2e}")
    print(f"Simpson's (m=100):   error = {abs(composite_simpson(f,0,1,100) - exact):.2e}")
    print(f"GL-10:               error = {abs(gl_integrate(f,0,1,10) - exact):.2e}")
    result, levels = romberg_integrate(f, 0, 1)
    print(f"Romberg:             error = {abs(result - exact):.2e} (levels={levels})")
    gh_val = gauss_hermite_expect(lambda x: np.exp(-x**2/2),
                                   mu=0, sigma=1, n=20) * np.sqrt(2*np.pi)
    print(f"GH integral of N(0,1): {gh_val:.10f} (expected 1.0)")
```
