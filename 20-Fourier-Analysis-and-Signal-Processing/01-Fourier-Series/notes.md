[← Back to Fourier Analysis](../README.md) | [Next: Fourier Transform →](../02-Fourier-Transform/notes.md)

---

# Fourier Series

> _"It is not the duty of mathematics to explain nature, but to describe it with the greatest possible accuracy. Yet Fourier's discovery — that any periodic motion is a superposition of simple oscillations — turned out to be both a perfect description and the deepest explanation."_
> — paraphrasing Hermann Weyl

## Overview

A Fourier series decomposes a periodic function into an infinite sum of sinusoidal harmonics. What seems like a purely mathematical curiosity — writing a square wave as a sum of sines — turns out to be one of the most powerful analytical tools in science and engineering. Every sound you hear, every image you see, every signal your phone transmits, and every positional encoding in a large language model has Fourier series at its mathematical core.

This section develops the theory from scratch. We begin with the geometric idea: functions on $[-\pi, \pi]$ form an inner product space, and the trigonometric functions $\{1, \cos(nx), \sin(nx)\}$ form an orthonormal basis in that space. Computing a Fourier series is simply projecting a function onto each basis vector. This geometric picture immediately explains why the formulas look the way they do, and why the theory is so clean.

We then turn to the hard question: when does the series actually converge back to $f$? The answer — Dirichlet's theorem, $L^2$ convergence, and the Parseval identity — is where the real mathematics lies. We examine the Gibbs phenomenon (the persistent 9% overshoot at jump discontinuities that cannot be removed by adding more terms), and close with the applications that connect directly to modern AI: RoPE positional encodings in LLaMA-3, the spectral bias of neural networks, and random Fourier features for kernel approximation.

## Prerequisites

- **Complex exponentials** — $e^{i\theta} = \cos\theta + i\sin\theta$, Euler's formula ([§01 Mathematical Foundations](../../01-Mathematical-Foundations/README.md))
- **Integration** — definite integrals, integration by parts, improper integrals ([§04 Calculus Fundamentals](../../04-Calculus-Fundamentals/README.md))
- **Inner products and orthogonality** — abstract inner product spaces, orthonormal bases ([§12-02 Hilbert Spaces](../../12-Functional-Analysis/02-Hilbert-Spaces/notes.md))
- **Series convergence** — pointwise, uniform, and $L^2$ convergence ([§01 Mathematical Foundations](../../01-Mathematical-Foundations/README.md))

## Companion Notebooks

| Notebook | Description |
|----------|-------------|
| [theory.ipynb](theory.ipynb) | Interactive computation of Fourier series, convergence animations, Gibbs phenomenon, RoPE derivation |
| [exercises.ipynb](exercises.ipynb) | 8 graded problems from computing coefficients to proving Parseval's theorem to implementing RoPE |

## Learning Objectives

After completing this section, you will:

1. Compute real and complex Fourier coefficients for standard periodic functions (square wave, sawtooth, triangle wave)
2. State and apply the Dirichlet conditions for pointwise convergence of a Fourier series
3. Explain the difference between pointwise, uniform, and $L^2$ convergence in the context of Fourier series
4. Derive Parseval's identity and use it to evaluate infinite series (e.g., $\sum_{n=1}^\infty 1/n^2$)
5. Describe the Gibbs phenomenon and explain why it cannot be eliminated by adding more terms
6. Relate the smoothness of $f$ to the decay rate of its Fourier coefficients
7. Derive the RoPE positional encoding formula from the complex Fourier basis
8. Explain the spectral bias phenomenon and its implications for neural network training dynamics
9. Prove the orthonormality of $\{e^{inx}/\sqrt{2\pi}\}_{n \in \mathbb{Z}}$ in $L^2[-\pi,\pi]$
10. Apply Fejér's theorem to understand Cesàro summation and convergence acceleration

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 What Is a Fourier Series?](#11-what-is-a-fourier-series)
  - [1.2 Why It Matters for AI](#12-why-it-matters-for-ai)
  - [1.3 Historical Timeline](#13-historical-timeline)
  - [1.4 The Geometric Picture](#14-the-geometric-picture)
  - [1.5 Three Equivalent Representations](#15-three-equivalent-representations)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 The Space $L^2[-\pi,\pi]$](#21-the-space-l2-pipi)
  - [2.2 The Trigonometric System](#22-the-trigonometric-system)
  - [2.3 Fourier Coefficients](#23-fourier-coefficients)
  - [2.4 Partial Sums and the Dirichlet Kernel](#24-partial-sums-and-the-dirichlet-kernel)
- [3. Convergence Theory](#3-convergence-theory)
  - [3.1 Pointwise Convergence: Dirichlet's Theorem](#31-pointwise-convergence-dirichlets-theorem)
  - [3.2 $L^2$ Convergence and Completeness](#32-l2-convergence-and-completeness)
  - [3.3 Uniform Convergence](#33-uniform-convergence)
  - [3.4 The Gibbs Phenomenon](#34-the-gibbs-phenomenon)
  - [3.5 Fejér's Theorem and Cesàro Summability](#35-fejérs-theorem-and-cesàro-summability)
- [4. Parseval's Theorem and Energy](#4-parsevals-theorem-and-energy)
  - [4.1 Bessel's Inequality](#41-bessels-inequality)
  - [4.2 Parseval's Identity](#42-parsevals-identity)
  - [4.3 Applications: Evaluating Infinite Series](#43-applications-evaluating-infinite-series)
- [5. Properties of Fourier Coefficients](#5-properties-of-fourier-coefficients)
  - [5.1 Linearity, Shift, and Scaling](#51-linearity-shift-and-scaling)
  - [5.2 Differentiation and Integration](#52-differentiation-and-integration)
  - [5.3 Even and Odd Functions](#53-even-and-odd-functions)
  - [5.4 Smoothness and Spectral Decay](#54-smoothness-and-spectral-decay)
  - [5.5 Riemann-Lebesgue Lemma](#55-riemann-lebesgue-lemma)
- [6. Fourier Series on General Intervals](#6-fourier-series-on-general-intervals)
  - [6.1 Extension to $[-L, L]$](#61-extension-to--l-l)
  - [6.2 Half-Range Expansions](#62-half-range-expansions)
- [7. Applications in Machine Learning](#7-applications-in-machine-learning)
  - [7.1 Sinusoidal Positional Encodings in Transformers](#71-sinusoidal-positional-encodings-in-transformers)
  - [7.2 Rotary Positional Encoding (RoPE)](#72-rotary-positional-encoding-rope)
  - [7.3 Spectral Bias of Neural Networks](#73-spectral-bias-of-neural-networks)
  - [7.4 Random Fourier Features and Kernel Approximation](#74-random-fourier-features-and-kernel-approximation)
- [8. Common Mistakes](#8-common-mistakes)
- [9. Exercises](#9-exercises)
- [10. Why This Matters for AI (2026 Perspective)](#10-why-this-matters-for-ai-2026-perspective)
- [11. Conceptual Bridge](#11-conceptual-bridge)

---

## 1. Intuition

### 1.1 What Is a Fourier Series?

Strike a guitar string and you hear a pitch — the fundamental frequency. Listen more carefully and you hear overtones: frequencies at twice, three times, four times the fundamental. The rich timbre of a guitar versus a piano playing the same note comes entirely from the relative amplitudes of these overtones. A Fourier series is the precise mathematical statement of this physical observation: **every periodic function is a (possibly infinite) weighted sum of sinusoids at integer multiples of a fundamental frequency.**

More precisely, if $f$ is a $2\pi$-periodic function satisfying mild regularity conditions, then:

$$f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} \left[ a_n \cos(nx) + b_n \sin(nx) \right]$$

where the coefficients $a_n$ and $b_n$ are uniquely determined by $f$. The term with $n=1$ is the **fundamental harmonic**, and the terms with $n > 1$ are the **overtones** or **harmonics**.

The miracle is how general this is. The function $f$ does not have to be smooth. The square wave — which jumps discontinuously between $-1$ and $+1$ — has a Fourier series:

$$f(x) = \frac{4}{\pi}\left(\sin x + \frac{\sin 3x}{3} + \frac{\sin 5x}{5} + \cdots\right) = \frac{4}{\pi}\sum_{k=0}^\infty \frac{\sin((2k+1)x)}{2k+1}$$

Summing 100 terms gives a function that is nearly indistinguishable from the square wave — except near the jumps, where a small overshoot persists no matter how many terms you include. This is the **Gibbs phenomenon**, and we will study it carefully.

**Non-example:** Not every function has a convergent Fourier series in the pointwise sense. A function that is not in $L^2[-\pi,\pi]$ (e.g., $f(x) = 1/x$ near $x=0$) does not have a well-defined Fourier series. A continuous function can even have a Fourier series that diverges at a single point (Du Bois-Reymond, 1876). The correct framework is $L^2$ convergence, not pointwise convergence.

### 1.2 Why It Matters for AI

Fourier series are not an abstract curiosity — they are embedded in the architecture of modern LLMs, CNNs, and audio models:

**For AI:**
- **Positional encodings in Transformers** (Vaswani et al., 2017) use $\sin(pos/10000^{2i/d})$ and $\cos(pos/10000^{2i/d})$ — these are Fourier basis functions at geometrically spaced frequencies.
- **Rotary Position Embedding (RoPE)** (Su et al., 2021), used in LLaMA-3, GPT-NeoX, and Mistral, interprets token positions as rotations in the complex plane — directly using complex Fourier basis vectors $e^{in\theta}$.
- **Spectral bias** (Rahaman et al., 2019): neural networks learn low-frequency Fourier components of the target function first and high-frequency components last. This governs convergence speed, generalization, and the effectiveness of data augmentation.
- **Random Fourier features** (Rahimi & Recht, 2007): kernel machines (SVMs, GPs) can be approximated in $O(D)$ by projecting data onto $D$ random Fourier basis functions sampled from the kernel's spectral density.

### 1.3 Historical Timeline

```
FOURIER ANALYSIS — HISTORICAL MILESTONES
════════════════════════════════════════════════════════════════════════

  1748  Euler studies vibrating strings; writes trigonometric series
  1807  Fourier presents "Théorie de la chaleur" to Paris Academy;
        claims every function has a trigonometric series expansion
        (claim rejected — Laplace and Lagrange are skeptical)
  1822  Fourier publishes "Théorie analytique de la chaleur"
  1829  Dirichlet proves convergence for piecewise smooth functions;
        gives first rigorous proof of pointwise convergence
  1854  Riemann extends the theory; defines the Riemann integral partly
        for the purposes of studying Fourier series
  1873  Du Bois-Reymond exhibits a continuous function whose Fourier
        series diverges at a point — Fourier's original claim was wrong
  1900  Fejér proves Cesàro summability: every continuous function's
        Fourier series converges in the Cesàro sense
  1907  Parseval's identity proved rigorously by Fatou & Riesz
  1966  Carleson proves pointwise convergence a.e. for L² functions
        (Fields Medal, 2006)
  2017  Vaswani et al. use Fourier basis for transformer PE
  2021  Su et al. introduce RoPE; now standard in LLaMA-3, Mistral
  2022  FNet (Lee-Thorp) replaces attention with Fourier transforms
  2023  Spectral analysis of LLM weight matrices goes mainstream

════════════════════════════════════════════════════════════════════════
```

### 1.4 The Geometric Picture

The key insight that makes Fourier series transparent is to think of functions as **vectors in an infinite-dimensional inner product space**.

On the interval $[-\pi, \pi]$, define the inner product of two functions $f$ and $g$ by:

$$\langle f, g \rangle = \frac{1}{2\pi} \int_{-\pi}^{\pi} f(x)\overline{g(x)}\,dx$$

The functions $\{e^{inx}\}_{n \in \mathbb{Z}}$ form an **orthonormal set** under this inner product:

$$\langle e^{inx}, e^{imx} \rangle = \frac{1}{2\pi} \int_{-\pi}^{\pi} e^{i(n-m)x}\,dx = \delta_{nm}$$

where $\delta_{nm}$ is the Kronecker delta. This is easy to verify: if $n \neq m$, the integral of $e^{i(n-m)x}$ over a full period is zero.

Now the Fourier series $f(x) = \sum_n c_n e^{inx}$ is just the **expansion of $f$ in this orthonormal basis**. The coefficient $c_n$ is the projection of $f$ onto the basis vector $e^{inx}$:

$$c_n = \langle f, e^{inx} \rangle = \frac{1}{2\pi}\int_{-\pi}^{\pi} f(x) e^{-inx}\,dx$$

This is identical to the formula for coordinates in any orthonormal basis: $c_n = \langle f, \mathbf{e}_n \rangle$. The Fourier series is **not a formula to be memorized** — it is the inevitable consequence of projecting onto an orthonormal basis.

> **For AI:** This geometric view directly explains RoPE. In RoPE, each attention head dimension pair $(q_{2i}, q_{2i+1})$ is treated as a 2D vector and rotated by angle $m\theta_i$ for token position $m$. This rotation is multiplication by $e^{im\theta_i}$ — a Fourier basis vector. The inner product between rotated queries and keys then depends only on their relative position $m - n$, making the positional encoding **relative** rather than absolute.

### 1.5 Three Equivalent Representations

Every Fourier series can be written in three equivalent forms. Each has advantages in different contexts.

**Real (sine-cosine) form:**
$$f(x) = \frac{a_0}{2} + \sum_{n=1}^\infty \left[ a_n \cos(nx) + b_n \sin(nx) \right]$$
$$a_n = \frac{1}{\pi}\int_{-\pi}^{\pi} f(x)\cos(nx)\,dx, \quad b_n = \frac{1}{\pi}\int_{-\pi}^{\pi} f(x)\sin(nx)\,dx$$
*Best for: real-valued functions where you want to see real amplitudes explicitly.*

**Complex exponential form:**
$$f(x) = \sum_{n=-\infty}^{\infty} c_n e^{inx}, \quad c_n = \frac{1}{2\pi}\int_{-\pi}^{\pi} f(x)e^{-inx}\,dx$$
*Best for: computation, derivations, and AI applications (RoPE, FNet). More compact; the complex exponentials are eigenfunctions of differentiation.*

**Amplitude-phase form:**
$$f(x) = A_0 + \sum_{n=1}^\infty A_n \cos(nx + \phi_n)$$
where $A_n = \sqrt{a_n^2 + b_n^2}$, $\phi_n = \arctan(-b_n/a_n)$.
*Best for: signal processing applications where amplitude and phase are the physically meaningful quantities.*

**Conversion between forms:** $c_n = (a_n - ib_n)/2$ for $n > 0$, $c_0 = a_0/2$, and $c_{-n} = \overline{c_n}$ for real $f$. These identities follow immediately from Euler's formula $e^{inx} = \cos(nx) + i\sin(nx)$.


## 2. Formal Definitions

### 2.1 The Space $L^2[-\pi,\pi]$

The natural domain for Fourier series is the **square-integrable functions** on $[-\pi, \pi]$:

$$L^2[-\pi,\pi] = \left\{ f: [-\pi,\pi] \to \mathbb{C} \;\middle|\; \int_{-\pi}^{\pi} |f(x)|^2\,dx < \infty \right\}$$

This space carries the inner product:
$$\langle f, g \rangle = \frac{1}{2\pi}\int_{-\pi}^{\pi} f(x)\overline{g(x)}\,dx$$

and induced norm $\lVert f \rVert = \sqrt{\langle f, f \rangle}$. Two functions that agree except on a set of measure zero are identified. Under this identification, $L^2[-\pi,\pi]$ is a **complete inner product space** — a Hilbert space. The full abstract theory is in [§12-02 Hilbert Spaces](../../12-Functional-Analysis/02-Hilbert-Spaces/notes.md); here we use only the concrete definition.

**Examples in $L^2[-\pi,\pi]$:**
- Every bounded measurable function (e.g., square wave, triangle wave)
- $f(x) = |x|^{-1/3}$ (integrable singularity; $\lVert f \rVert < \infty$)
- $f(x) = \sin(nx)$ for any $n \in \mathbb{Z}$

**Non-examples:**
- $f(x) = 1/x$ near $x = 0$: $\int_{-\pi}^{\pi} 1/x^2\,dx = \infty$, so $f \notin L^2$
- $f(x) = e^{1/x^2}$ near $x = 0$: grows too fast

### 2.2 The Trigonometric System

**Definition (Trigonometric System):** The collection $\mathcal{T} = \{e^{inx}\}_{n \in \mathbb{Z}}$ is called the **complex trigonometric system** on $[-\pi, \pi]$.

**Theorem (Orthonormality):** The rescaled functions $\phi_n(x) = e^{inx}/\sqrt{2\pi}$ satisfy $\langle \phi_n, \phi_m \rangle_{L^2} = \delta_{nm}$, using the unweighted inner product $\langle f, g \rangle = \int_{-\pi}^{\pi} f\bar{g}\,dx$.

*Proof.* For $n \neq m$:
$$\int_{-\pi}^{\pi} e^{inx} e^{-imx}\,dx = \int_{-\pi}^{\pi} e^{i(n-m)x}\,dx = \frac{e^{i(n-m)x}}{i(n-m)}\Bigg|_{-\pi}^{\pi} = \frac{e^{i(n-m)\pi} - e^{-i(n-m)\pi}}{i(n-m)} = \frac{2\sin((n-m)\pi)}{n-m} = 0$$
since $n-m \in \mathbb{Z}\setminus\{0\}$ so $\sin((n-m)\pi) = 0$. For $n = m$: $\int_{-\pi}^{\pi} 1\,dx = 2\pi$. $\square$

The real trigonometric system $\{1, \cos x, \sin x, \cos 2x, \sin 2x, \ldots\}$ is similarly orthogonal, with norms $\lVert 1 \rVert = \sqrt{2\pi}$ and $\lVert \cos nx \rVert = \lVert \sin nx \rVert = \sqrt{\pi}$ for $n \geq 1$.

**Completeness (stated):** The trigonometric system is **complete** in $L^2[-\pi,\pi]$: for every $f \in L^2$ and every $\varepsilon > 0$, there exists a trigonometric polynomial $T(x) = \sum_{|n| \leq N} c_n e^{inx}$ such that $\lVert f - T \rVert < \varepsilon$. The proof uses the Weierstrass approximation theorem and is in [§12-02](../../12-Functional-Analysis/02-Hilbert-Spaces/notes.md).

### 2.3 Fourier Coefficients

**Definition (Fourier Coefficients):** For $f \in L^2[-\pi,\pi]$, the **complex Fourier coefficients** of $f$ are:
$$c_n = \frac{1}{2\pi}\int_{-\pi}^{\pi} f(x)\,e^{-inx}\,dx, \quad n \in \mathbb{Z}$$

The **real Fourier coefficients** are:
$$a_n = \frac{1}{\pi}\int_{-\pi}^{\pi} f(x)\cos(nx)\,dx \quad (n \geq 0), \quad b_n = \frac{1}{\pi}\int_{-\pi}^{\pi} f(x)\sin(nx)\,dx \quad (n \geq 1)$$

**Relations between forms:** For real $f$: $c_0 = a_0/2$, $c_n = (a_n - ib_n)/2$ for $n \geq 1$, and $c_{-n} = \overline{c_n}$ (Hermitian symmetry).

**Standard examples — complete computations:**

**(i) Square wave:** $f(x) = \operatorname{sgn}(x) = +1$ for $x > 0$, $-1$ for $x < 0$.

Since $f$ is odd, $a_n = 0$ for all $n$. For $b_n$:
$$b_n = \frac{2}{\pi}\int_0^\pi \sin(nx)\,dx = \frac{2}{\pi}\left[\frac{-\cos(nx)}{n}\right]_0^\pi = \frac{2}{\pi n}(1 - \cos(n\pi)) = \begin{cases} 4/(\pi n) & n \text{ odd} \\ 0 & n \text{ even} \end{cases}$$

So $f(x) = \frac{4}{\pi}\sum_{k=0}^\infty \frac{\sin((2k+1)x)}{2k+1}$.

**(ii) Sawtooth wave:** $f(x) = x$ for $x \in (-\pi, \pi)$, extended periodically.

$f$ is odd, so $a_n = 0$. For $b_n$, integrate by parts:
$$b_n = \frac{1}{\pi}\int_{-\pi}^{\pi} x\sin(nx)\,dx = \frac{2}{\pi}\int_0^\pi x\sin(nx)\,dx = \frac{2(-1)^{n+1}}{n}$$

So $f(x) = 2\sum_{n=1}^\infty \frac{(-1)^{n+1}}{n}\sin(nx) = 2\left(\sin x - \frac{\sin 2x}{2} + \frac{\sin 3x}{3} - \cdots\right)$.

**(iii) Triangle wave:** $f(x) = |x|$ for $x \in [-\pi, \pi]$.

$f$ is even, so $b_n = 0$. For $a_n$ ($n \geq 1$):
$$a_n = \frac{2}{\pi}\int_0^\pi x\cos(nx)\,dx = \frac{2}{\pi}\left[\frac{x\sin(nx)}{n} + \frac{\cos(nx)}{n^2}\right]_0^\pi = \frac{2(\cos(n\pi) - 1)}{\pi n^2} = \begin{cases} -4/(\pi n^2) & n \text{ odd} \\ 0 & n \text{ even}\end{cases}$$

And $a_0 = \pi$. So $f(x) = \frac{\pi}{2} - \frac{4}{\pi}\sum_{k=0}^\infty \frac{\cos((2k+1)x)}{(2k+1)^2}$.

**Non-examples:** Not every sequence $\{c_n\}$ is the Fourier coefficient sequence of an $L^2$ function. By Parseval's identity (§4.2), we need $\sum_n |c_n|^2 < \infty$. The sequence $c_n = 1$ for all $n$ is not a valid Fourier coefficient sequence.

### 2.4 Partial Sums and the Dirichlet Kernel

**Definition:** The $N$-th **partial sum** of the Fourier series of $f$ is:
$$S_N f(x) = \sum_{n=-N}^{N} c_n e^{inx}$$

A crucial observation: the partial sum can be written as a convolution:
$$S_N f(x) = (f * D_N)(x) = \frac{1}{2\pi}\int_{-\pi}^{\pi} f(t) D_N(x - t)\,dt$$

where the **Dirichlet kernel** $D_N$ is:
$$D_N(x) = \sum_{n=-N}^{N} e^{inx} = \frac{\sin((N + 1/2)x)}{\sin(x/2)}$$

The closed form follows by summing the geometric series $\sum_{n=-N}^N e^{inx}$ and simplifying using $e^{ix/2} - e^{-ix/2} = 2i\sin(x/2)$.

**Key properties of $D_N$:**
- $\frac{1}{2\pi}\int_{-\pi}^{\pi} D_N(x)\,dx = 1$ (normalized)
- $D_N$ oscillates with $N$ peaks near $x = 0$
- $D_N$ does NOT stay non-negative (unlike an approximate identity), causing convergence issues

The failure of $D_N$ to remain non-negative is precisely what allows the Gibbs phenomenon (§3.4) and prevents uniform convergence at discontinuities.


## 3. Convergence Theory

### 3.1 Pointwise Convergence: Dirichlet's Theorem

**Theorem (Dirichlet, 1829):** Let $f$ be a $2\pi$-periodic function that is **piecewise smooth** on $[-\pi,\pi]$ (i.e., $f$ and $f'$ are piecewise continuous). Then the Fourier series of $f$ converges for every $x$, and:
$$S_N f(x) \;\to\; \frac{f(x^+) + f(x^-)}{2} \quad \text{as } N \to \infty$$

At points of continuity, $f(x^+) = f(x^-)$, so the series converges to $f(x)$. At a jump discontinuity, the series converges to the **average of the left and right limits**.

*Proof sketch.* Write $S_N f(x) - \frac{f(x^+)+f(x^-)}{2}$ as an integral involving $D_N$ and the function $g_x(t) = f(x+t) - \frac{f(x^+)+f(x^-)}{2}$. The key step: $g_x(t)/\sin(t/2)$ is integrable if $f$ is piecewise smooth (the singularity at $t=0$ is removable). Then apply the **Riemann-Lebesgue Lemma** (§5.5): $\int h(t)\sin(Nt)\,dt \to 0$ as $N \to \infty$ for any integrable $h$. $\square$

**Dirichlet conditions (sufficient, not necessary):**
1. $f$ is bounded and has finitely many maxima, minima, and discontinuities on $[-\pi,\pi]$
2. $f$ has left and right derivatives everywhere

**What Dirichlet's theorem does NOT say:**
- It does not guarantee uniform convergence (fails at jumps due to Gibbs)
- It does not apply to all continuous functions (Du Bois-Reymond's example)
- It says nothing about $L^2$ convergence rate

### 3.2 $L^2$ Convergence and Completeness

The $L^2$ convergence theorem is stronger and cleaner than the pointwise result:

**Theorem ($L^2$ Convergence):** For every $f \in L^2[-\pi,\pi]$:
$$\lVert f - S_N f \rVert_{L^2} \to 0 \quad \text{as } N \to \infty$$

Equivalently, $f = \sum_{n=-\infty}^{\infty} c_n e^{inx}$ as an $L^2$ limit. This follows directly from the **completeness** of the trigonometric system in $L^2$ (stated in §2.2).

**Bessel's Inequality (preliminary to Parseval):** For any $f \in L^2$:
$$\sum_{n=-\infty}^{\infty} |c_n|^2 \leq \lVert f \rVert^2 = \frac{1}{2\pi}\int_{-\pi}^{\pi}|f(x)|^2\,dx$$

*Proof.* Expand $0 \leq \lVert f - S_N f \rVert^2 = \lVert f \rVert^2 - \sum_{|n| \leq N} |c_n|^2$. This shows the partial sums of $\sum |c_n|^2$ are bounded by $\lVert f \rVert^2$. Taking $N \to \infty$ gives Bessel. $\square$

Equality holds (Bessel becomes Parseval) exactly when the system is complete.

### 3.3 Uniform Convergence

**Theorem:** If $f \in C^1[-\pi,\pi]$ (continuously differentiable) and periodic, then $S_N f \to f$ **uniformly**.

*Proof sketch.* Integrate by parts: $c_n[f] = -c_n[f']/(in)$ for $n \neq 0$. So $|c_n| \leq \lVert f' \rVert_1 / (\pi |n|)$. The partial sums converge uniformly by the Weierstrass M-test since $\sum 1/n < \infty$ (conditionally). For $f \in C^k$, $|c_n| = O(n^{-k})$, giving faster uniform convergence. $\square$

**Key point:** Continuity alone is insufficient for uniform convergence. The counterexample (Du Bois-Reymond) is a continuous function whose Fourier series diverges at a single point.

### 3.4 The Gibbs Phenomenon

The Gibbs phenomenon is one of the most important practical facts about Fourier series.

**Observation:** Near a jump discontinuity at $x = x_0$, the partial sum $S_N f$ overshoots the function value by approximately $\frac{2}{\pi}\int_0^\pi \frac{\sin t}{t}\,dt - 1 \approx 0.0895$ — about **9% of the jump height** — regardless of how large $N$ is.

**Precise statement for the square wave:** For $f(x) = \operatorname{sgn}(x)$ with jump height $2$:
$$\lim_{N\to\infty} S_N f\!\left(\frac{\pi}{2N+1}\right) = \frac{2}{\pi}\int_0^\pi \frac{\sin t}{t}\,dt \approx 1.1790\ldots$$

So the overshoot is $\approx 0.179$, which is $9\%$ of the total jump of $2$.

**Why it persists:** The Dirichlet kernel $D_N$ has a tall central spike but also oscillating side lobes with total negative area $\approx -1/\pi$. These side lobes cannot be eliminated by taking $N$ larger — they just become narrower and move closer to the discontinuity.

**For AI:** The Gibbs phenomenon is why "ringing" artifacts appear when you sharply truncate a frequency spectrum (e.g., in audio compression, image filtering, or when a language model encounters out-of-distribution high-frequency tokens). Windowing functions (§20-03) are the engineering fix.

**Remedy — Fejér's theorem:** Instead of taking partial sums, take their **Cesàro means** $\sigma_N f = \frac{1}{N+1}\sum_{k=0}^N S_k f$. Fejér proved these converge everywhere and do NOT exhibit Gibbs overshoot.

### 3.5 Fejér's Theorem and Cesàro Summability

**Theorem (Fejér, 1900):** Let $f \in C[-\pi,\pi]$ be $2\pi$-periodic. Then $\sigma_N f \to f$ uniformly as $N \to \infty$.

The Cesàro means use the **Fejér kernel** $F_N = \frac{1}{N+1}\sum_{k=0}^N D_k$:
$$F_N(x) = \frac{1}{N+1}\left(\frac{\sin((N+1)x/2)}{\sin(x/2)}\right)^2 \geq 0$$

Unlike the Dirichlet kernel, **the Fejér kernel is non-negative**. This is why Cesàro summation fixes the Gibbs phenomenon — the averaging eliminates the negative side lobes.

**For AI:** Fejér's theorem is the prototype for **windowing** in signal processing. Multiplying a signal by a window function before taking the Fourier transform is equivalent to using a smoother summation kernel, which reduces spectral leakage and ringing (→ §20-03).

---

## 4. Parseval's Theorem and Energy

### 4.1 Bessel's Inequality

We proved Bessel's inequality in §3.2: $\sum_{n} |c_n|^2 \leq \lVert f \rVert^2$. This says the "energy" in the frequency representation is at most the energy in the time representation. Equality requires completeness.

### 4.2 Parseval's Identity

**Theorem (Parseval's Identity):** For $f \in L^2[-\pi,\pi]$ with Fourier coefficients $\{c_n\}$:
$$\frac{1}{2\pi}\int_{-\pi}^{\pi} |f(x)|^2\,dx = \sum_{n=-\infty}^{\infty} |c_n|^2$$

In real form: $\frac{a_0^2}{2} + \sum_{n=1}^\infty (a_n^2 + b_n^2) = \frac{1}{\pi}\int_{-\pi}^{\pi} |f(x)|^2\,dx$.

*Proof.* By completeness, $\lVert f - S_N f \rVert \to 0$. Then:
$$\lVert f \rVert^2 = \lim_{N\to\infty} \lVert S_N f \rVert^2 = \lim_{N\to\infty} \sum_{|n|\leq N} |c_n|^2 = \sum_{n=-\infty}^\infty |c_n|^2$$
using orthonormality of $\{e^{inx}\}$ for the second equality. $\square$

**Physical interpretation:** The total **energy** (power) of a signal equals the sum of energies in each frequency component. Fourier analysis is an **energy-preserving change of basis**.

**More general form:** For $f, g \in L^2$:
$$\frac{1}{2\pi}\int_{-\pi}^{\pi} f(x)\overline{g(x)}\,dx = \sum_{n=-\infty}^{\infty} c_n[f]\,\overline{c_n[g]}$$

This is the **polarization identity** version of Parseval.

### 4.3 Applications: Evaluating Infinite Series

Parseval's identity is a powerful tool for evaluating series that have no elementary closed form.

**Example 1 — Basel problem via triangle wave:** Apply Parseval to $f(x) = x$ on $(-\pi,\pi)$:
$$\frac{1}{\pi}\int_{-\pi}^{\pi} x^2\,dx = \frac{2\pi^2}{3}, \quad \text{and} \quad \sum_{n=1}^\infty b_n^2 = \sum_{n=1}^\infty \frac{4}{n^2}$$
Parseval gives $\frac{2\pi^2}{3} = 4\sum_{n=1}^\infty \frac{1}{n^2}$, so $\sum_{n=1}^\infty \frac{1}{n^2} = \frac{\pi^2}{6}$.

**Example 2 — $\sum 1/(2k+1)^2$:** Apply Parseval to the square wave. The non-zero Fourier coefficients are $b_{2k+1} = 4/(\pi(2k+1))$. Then:
$$\frac{1}{\pi}\int_{-\pi}^{\pi} 1\,dx = 2 = \sum_{k=0}^\infty \frac{16}{\pi^2(2k+1)^2}$$
giving $\sum_{k=0}^\infty \frac{1}{(2k+1)^2} = \frac{\pi^2}{8}$.


## 5. Properties of Fourier Coefficients

### 5.1 Linearity, Shift, and Scaling

Let $c_n[f]$ denote the $n$-th Fourier coefficient of $f$.

| Property | Statement | Proof idea |
|----------|-----------|------------|
| **Linearity** | $c_n[\alpha f + \beta g] = \alpha c_n[f] + \beta c_n[g]$ | Integral is linear |
| **Shift** | $c_n[f(\cdot - x_0)] = e^{-inx_0} c_n[f]$ | Change of variable $t = x - x_0$ |
| **Conjugation** | $c_n[\overline{f}] = \overline{c_{-n}[f]}$ | Conjugate the integral |
| **Reflection** | $c_n[f(-\cdot)] = c_{-n}[f]$ | Change of variable $x \to -x$ |

The **shift property** is the Fourier-series version of the time-shift property of the Fourier transform. It says: shifting a signal in time multiplies its Fourier coefficients by a complex exponential — i.e., introduces a **linear phase**. This is fundamental in signal alignment and is the mechanism behind **relative positional encodings** in transformers.

**For AI — RoPE connection:** In RoPE, the key and query vectors at position $m$ and $n$ are $\mathbf{q}_m = R_m \mathbf{q}$ and $\mathbf{k}_n = R_n \mathbf{k}$ where $R_m$ is a rotation matrix. The dot product $\mathbf{q}_m^\top \mathbf{k}_n = \mathbf{q}^\top R_{m-n} \mathbf{k}$ — it depends only on the **relative position** $m - n$. This is exactly the Fourier shift property: shifting both signals by the same amount leaves their inner product unchanged.

### 5.2 Differentiation and Integration

**Differentiation in frequency space:**
$$c_n[f'] = in \cdot c_n[f]$$

*Proof.* Integrate by parts: $c_n[f'] = \frac{1}{2\pi}\int_{-\pi}^{\pi} f'(x) e^{-inx}\,dx = \frac{1}{2\pi}\left[f(x)e^{-inx}\right]_{-\pi}^{\pi} + \frac{in}{2\pi}\int_{-\pi}^{\pi} f(x)e^{-inx}\,dx$. By periodicity, the boundary term vanishes, leaving $c_n[f'] = in\,c_n[f]$. $\square$

**Iterated differentiation:** $c_n[f^{(k)}] = (in)^k c_n[f]$.

**Implication for smoothness:** A function $f \in C^k$ (k-times continuously differentiable and periodic) has $|c_n| = O(|n|^{-k})$ as $|n| \to \infty$. The smoother $f$ is, the faster its Fourier coefficients decay. This is the quantitative content of §5.4.

**Integration:** $c_n\left[\int_0^x f(t)\,dt - \langle f \rangle x\right] = \frac{c_n[f]}{in}$ for $n \neq 0$, where $\langle f \rangle = c_0[f]$ is the mean.

### 5.3 Even and Odd Functions

**Even functions** ($f(-x) = f(x)$) have $b_n = 0$ for all $n$, so the Fourier series contains only cosines: $f(x) = \frac{a_0}{2} + \sum_{n=1}^\infty a_n \cos(nx)$ — a **cosine series**. The coefficients are $a_n = \frac{2}{\pi}\int_0^\pi f(x)\cos(nx)\,dx$.

**Odd functions** ($f(-x) = -f(x)$) have $a_n = 0$ for all $n$, so the Fourier series contains only sines: $f(x) = \sum_{n=1}^\infty b_n \sin(nx)$ — a **sine series**. The coefficients are $b_n = \frac{2}{\pi}\int_0^\pi f(x)\sin(nx)\,dx$.

**For AI — DCT connection:** The **Discrete Cosine Transform (DCT)** used in JPEG and MP3 compression is the even-extension Fourier series. The DCT coefficients $a_n$ are exactly the Fourier coefficients of the even extension of the signal to $[-\pi,\pi]$. This is why the DCT achieves better energy compaction than the DFT for real signals.

### 5.4 Smoothness and Spectral Decay

The relationship between regularity and spectral decay is fundamental to both signal processing and machine learning:

| Regularity of $f$ | Decay of $|c_n|$ | Practical implication |
|-------------------|------------------|-----------------------|
| $f \in L^2$ | $|c_n| \to 0$ (by Riemann-Lebesgue) | All coefficients vanish asymptotically |
| $f$ piecewise smooth, discontinuity | $|c_n| \sim C/|n|$ | Slow $1/n$ decay (Gibbs) |
| $f \in C^1$ (continuously diff.) | $|c_n| = O(1/n^2)$ | Faster decay, no Gibbs |
| $f \in C^k$ | $|c_n| = O(1/n^{k+1})$ | Super-algebraic if $k$ large |
| $f$ analytic | $|c_n| \leq Ce^{-r|n|}$ for some $r > 0$ | Exponential decay |

**For AI — Spectral bias (Rahaman et al., 2019):** Neural networks learn the target function's Fourier decomposition from lowest to highest frequency. Formally, if $f^* = \sum_n c_n e^{inx}$ is the target function, a network trained by gradient descent first approximates the $c_n$ for small $|n|$ (low frequencies) and only later approximates high-frequency components. This implies:
- **Benefit:** Implicit regularization toward smooth (low-frequency) solutions → good generalization
- **Cost:** Slow convergence on high-frequency targets; requires more data for texture-rich images
- **Fix:** Fourier feature embeddings (RFF, NeRF's positional encoding) inject high-frequency components explicitly

### 5.5 Riemann-Lebesgue Lemma

**Theorem (Riemann-Lebesgue):** If $f \in L^1[-\pi,\pi]$, then $c_n[f] \to 0$ as $|n| \to \infty$.

*Proof sketch.* For $f$ a step function: direct computation. For general $f$: approximate by step functions and use the $L^1$ bound. $\square$

**Significance:** This says every $L^1$ function has Fourier coefficients that vanish at high frequencies. The rate of decay depends on smoothness (§5.4), but the qualitative statement holds for all integrable functions. This is the mathematical foundation of the claim "smooth signals are compressible in the Fourier domain."

---

## 6. Fourier Series on General Intervals

### 6.1 Extension to $[-L, L]$

For a $2L$-periodic function $f$, the Fourier series on $[-L, L]$ is obtained by the change of variables $t = \pi x / L$:

$$f(x) = \frac{a_0}{2} + \sum_{n=1}^\infty \left[ a_n \cos\!\left(\frac{n\pi x}{L}\right) + b_n \sin\!\left(\frac{n\pi x}{L}\right) \right]$$

with coefficients:
$$a_n = \frac{1}{L}\int_{-L}^{L} f(x)\cos\!\left(\frac{n\pi x}{L}\right)dx, \quad b_n = \frac{1}{L}\int_{-L}^{L} f(x)\sin\!\left(\frac{n\pi x}{L}\right)dx$$

The **fundamental frequency** is $f_1 = 1/(2L)$, and the $n$-th harmonic has frequency $n/(2L)$.

**For AI — Transformer context length:** In a transformer with context length $T$, sinusoidal positional encodings use frequencies $1/10000^{2i/d}$ for $i = 0, \ldots, d/2 - 1$. This covers a geometric range of frequencies over the interval $[0, T]$ — analogous to sampling the Fourier series on $[0, T]$ at $d/2$ geometrically spaced frequencies. Longer context requires lower minimum frequencies, which is why RoPE's $\theta_{\min} = 10000^{-1}$ limits effective context length and why extended-context models (LLaMA-3-128K, Mistral) modify the base $\theta$.

### 6.2 Half-Range Expansions

If $f$ is defined on $[0, L]$ (not the full period), we can extend it to $[-L, L]$ in two ways:

**Even extension:** Extend by $f(-x) = f(x)$ → leads to a **cosine series** on $[0, L]$:
$$f(x) = \frac{a_0}{2} + \sum_{n=1}^\infty a_n \cos\!\left(\frac{n\pi x}{L}\right), \quad a_n = \frac{2}{L}\int_0^L f(x)\cos\!\left(\frac{n\pi x}{L}\right)dx$$

**Odd extension:** Extend by $f(-x) = -f(x)$ → leads to a **sine series** on $[0, L]$:
$$f(x) = \sum_{n=1}^\infty b_n \sin\!\left(\frac{n\pi x}{L}\right), \quad b_n = \frac{2}{L}\int_0^L f(x)\sin\!\left(\frac{n\pi x}{L}\right)dx$$

**Application:** Solving the heat equation on $[0, L]$ with Dirichlet boundary conditions ($f(0) = f(L) = 0$) uses the sine series expansion. With Neumann conditions ($f'(0) = f'(L) = 0$), use the cosine series.


## 7. Applications in Machine Learning

### 7.1 Sinusoidal Positional Encodings in Transformers

The original Transformer (Vaswani et al., 2017) uses positional encodings:
$$\text{PE}(pos, 2i) = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \quad \text{PE}(pos, 2i+1) = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

This assigns each position $pos \in \{0, 1, \ldots, T-1\}$ a $d_{\text{model}}$-dimensional vector whose components are sine and cosine values at geometrically spaced frequencies $\omega_i = 1/10000^{2i/d_{\text{model}}}$.

**Why this works:** The set of frequencies $\{\omega_i\}$ forms a geometric progression spanning from $1$ (very high frequency, distinguishes adjacent tokens) down to $10000^{-1}$ (very low frequency, distinguishes widely separated tokens). This is exactly the strategy of a Fourier series on a long interval: use many harmonics at different scales.

**Limitation:** These are **absolute** positional encodings — the embedding for position 5 is fixed regardless of context. This creates problems for generalization to longer sequences than seen in training.

### 7.2 Rotary Positional Encoding (RoPE)

RoPE (Su et al., 2021), now standard in LLaMA-2/3, Mistral, Falcon, and GPT-NeoX, encodes position as a **rotation** of the query and key vectors.

**Construction:** For a $d$-dimensional query vector $\mathbf{q}$, split into $d/2$ pairs $(q_{2i}, q_{2i+1})$. Rotate each pair by angle $m\theta_i$ where $m$ is the token position and $\theta_i = 10000^{-2i/d}$:

$$\begin{pmatrix} q'_{2i} \\ q'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}\begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}$$

This is multiplication by $e^{im\theta_i}$ in the complex representation $q_{2i} + iq_{2i+1}$.

**The key property:** The attention score between query at position $m$ and key at position $n$ is:
$$\text{score}(m, n) = \operatorname{Re}\!\left[\sum_{i=0}^{d/2-1} (q_{2i} + iq_{2i+1})e^{im\theta_i} \cdot \overline{(k_{2i} + ik_{2i+1})e^{in\theta_i}}\right]$$
$$= \operatorname{Re}\!\left[\sum_{i=0}^{d/2-1} (q_{2i} + iq_{2i+1})\overline{(k_{2i} + ik_{2i+1})} \cdot e^{i(m-n)\theta_i}\right]$$

The score depends only on $m - n$ — the **relative position**. This is the Fourier shift theorem: multiplying by $e^{im\theta_i}$ and taking a conjugate product gives sensitivity to relative displacement.

**Extended context:** The maximum effective context length is determined by the lowest frequency $\theta_{\min} = \theta_{d/2-1} = 10000^{-1}$. Models like LLaMA-3-128K extend context by scaling $\theta_i \to \theta_i \cdot (L_{\text{train}}/L_{\text{new}})$ (Position Interpolation, Chen et al., 2023) or by increasing the base from 10000 to 500000 (Rozière et al., 2023).

### 7.3 Spectral Bias of Neural Networks

**The phenomenon (Rahaman et al., 2019):** Neural networks trained with gradient descent learn a **biased** decomposition of the target function: low-frequency Fourier components are learned first, high-frequency components last.

**Mathematical statement:** Let $f^*$ be the target function with Fourier decomposition $f^*(x) = \sum_n c_n e^{inx}$. A network $f_\theta$ trained on a finite sample first converges on the low-$|n|$ components ($|c_n[f^* - f_\theta]|$ decreases for small $|n|$ first).

**Consequences for AI:**
1. **Good:** Networks converge to smooth solutions → implicit regularization → good out-of-distribution generalization for smooth targets
2. **Bad:** Learning high-frequency signals (sharp edges, fine texture) requires more data and training time
3. **NeRF / SIREN fix:** Sinusoidal activations (Sitzmann et al., 2020) or Fourier feature mappings $\gamma(\mathbf{x}) = [\cos(B\mathbf{x}), \sin(B\mathbf{x})]$ (Tancik et al., 2020) inject high-frequency components, overcoming spectral bias for 3D scene representation

**Mechanism:** The NTK (Neural Tangent Kernel) of a standard MLP has a spectrum that decays with frequency. High-frequency components of the target correspond to high-eigenvalue directions of the NTK, which converge slowly under gradient descent.

### 7.4 Random Fourier Features and Kernel Approximation

**The problem:** Kernel methods (SVMs, GPs) require storing and computing an $n \times n$ kernel matrix — $O(n^2)$ memory and $O(n^3)$ computation. For large $n$, this is infeasible.

**The solution (Rahimi & Recht, 2007):** Any shift-invariant kernel $k(\mathbf{x}, \mathbf{y}) = k(\mathbf{x} - \mathbf{y})$ can be written, by **Bochner's theorem**, as the Fourier transform of a non-negative measure:
$$k(\mathbf{x} - \mathbf{y}) = \int p(\boldsymbol{\omega}) e^{i\boldsymbol{\omega}^\top(\mathbf{x} - \mathbf{y})}\,d\boldsymbol{\omega}$$

Sampling $D$ frequencies $\boldsymbol{\omega}_1, \ldots, \boldsymbol{\omega}_D \sim p(\boldsymbol{\omega})$ and defining the feature map:
$$\phi(\mathbf{x}) = \frac{1}{\sqrt{D}}\left[\cos(\boldsymbol{\omega}_1^\top\mathbf{x}), \sin(\boldsymbol{\omega}_1^\top\mathbf{x}), \ldots, \cos(\boldsymbol{\omega}_D^\top\mathbf{x}), \sin(\boldsymbol{\omega}_D^\top\mathbf{x})\right]$$

gives $\mathbb{E}[\phi(\mathbf{x})^\top\phi(\mathbf{y})] = k(\mathbf{x} - \mathbf{y})$. This reduces kernel computation to $O(nD)$ — linear in the dataset size.

**Connection to Fourier series:** $\phi(\mathbf{x})$ is a finite Fourier expansion with randomly sampled frequencies. The approximation quality improves as $D$ increases, with concentration bounds showing $|k(\mathbf{x}-\mathbf{y}) - \phi(\mathbf{x})^\top\phi(\mathbf{y})| \leq \varepsilon$ with high probability for $D = O(d\log(1/\varepsilon)/\varepsilon^2)$.

---

## 8. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
|---|---------|----------------|-----|
| 1 | Using $c_n = \int_{-\pi}^{\pi} f e^{-inx} dx$ without the $\frac{1}{2\pi}$ factor | Missing normalization; coefficients will be off by $2\pi$ | Always check which convention your source uses; in this repo we use $c_n = \frac{1}{2\pi}\int$ |
| 2 | Assuming the Fourier series always converges pointwise | False: Du Bois-Reymond exhibited a continuous function whose series diverges at a point | State the convergence type explicitly; use $L^2$ convergence for most theory |
| 3 | Assuming the sum at a discontinuity equals $f(x)$ | The series converges to the **average** $\frac{f(x^+)+f(x^-)}{2}$ at a jump | Apply Dirichlet's theorem: check for discontinuities before claiming convergence to $f$ |
| 4 | Writing $c_{-n} = c_n$ for a real function | Correct relation is $c_{-n} = \overline{c_n}$ (conjugate, not equal) | For real $f$: $c_{-n} = \overline{c_n}$; only real if $c_n$ is real |
| 5 | Applying differentiation term-by-term without checking conditions | $c_n[f'] = inc_n[f]$ requires $f$ to be absolutely continuous and periodic | Check that $f \in AC[-\pi,\pi]$ (absolutely continuous) before differentiating term-by-term |
| 6 | Confusing Parseval's theorem with Bessel's inequality | Bessel gives $\leq$; Parseval gives $=$ — they differ | Parseval requires completeness of the trigonometric system; Bessel holds for any orthonormal set |
| 7 | Claiming $|c_n| = O(1/n)$ decay for a smooth function | Smooth functions have $O(1/n^k)$ for all $k$ — much faster than $1/n$ | $1/n$ decay indicates a jump discontinuity; smoothness gives faster algebraic or exponential decay |
| 8 | Forgetting to subtract the mean before computing sine/cosine coefficients | $c_0 = a_0/2$ is the DC component; it appears in the $a_0/2$ term, not in $a_n$ for $n \geq 1$ | Always compute $a_0$ separately; the series is $\frac{a_0}{2} + \sum_{n \geq 1}$ |
| 9 | Using the complex form with $e^{inx}$ but real form coefficients | Complex form requires $c_n = \frac{1}{2\pi}\int fe^{-inx}$; real form uses $a_n, b_n$ — these are different | Pick one form and use it consistently; convert via $c_n = (a_n - ib_n)/2$ |
| 10 | Claiming the Gibbs phenomenon goes away as $N \to \infty$ | The overshoot height stays at ~9% of the jump; only its width shrinks | Gibbs is a permanent feature of the partial sum near discontinuities; use windowing to mitigate |

---

## 9. Exercises

**Exercise 1 (★):** Compute the complex Fourier coefficients $c_n$ of the **triangle wave** $f(x) = |\,x\,|$ on $(-\pi, \pi)$ and write out the first five non-zero terms of the Fourier series. Verify that $f$ is even, so $c_n \in \mathbb{R}$.

**Exercise 2 (★):** Prove from first principles that the functions $\{1/\sqrt{2\pi},\, \cos(nx)/\sqrt{\pi},\, \sin(nx)/\sqrt{\pi}\}_{n \geq 1}$ form an orthonormal set in $L^2[-\pi,\pi]$ with the inner product $\langle f, g \rangle = \int_{-\pi}^{\pi} f\bar{g}\,dx$.

**Exercise 3 (★):** Using Parseval's identity applied to the square wave, show that $\sum_{k=0}^\infty \frac{1}{(2k+1)^2} = \frac{\pi^2}{8}$. Then use $\sum_{n=1}^\infty \frac{1}{n^2} = \sum_{k=1}^\infty \frac{1}{(2k)^2} + \sum_{k=0}^\infty \frac{1}{(2k+1)^2}$ to recover the Basel sum $\sum_{n=1}^\infty \frac{1}{n^2} = \frac{\pi^2}{6}$.

**Exercise 4 (★★):** Let $f(x) = e^{ax}$ for $x \in (-\pi, \pi)$ with $a \neq 0$ real. (a) Compute $c_n[f]$. (b) Apply Parseval's identity to obtain a formula for $\frac{1}{\sinh^2(\pi a)}$ in terms of a series involving $1/(a^2 + n^2)$. (c) Verify your answer for $a = 1$ numerically.

**Exercise 5 (★★):** Prove the **Riemann-Lebesgue Lemma**: if $f \in L^1[-\pi,\pi]$, then $c_n[f] \to 0$ as $|n| \to \infty$. (Hint: first prove it for step functions, then approximate general $f$.)

**Exercise 6 (★★):** Consider $f(x) = x^2$ on $(-\pi,\pi)$. (a) Find all Fourier coefficients $a_n, b_n$. (b) Show that $S_N f \to f$ uniformly. (c) Set $x = \pi$ in the resulting series to derive $\sum_{n=1}^\infty (-1)^{n+1}/n^2 = \pi^2/12$.

**Exercise 7 (★★★):** Implement RoPE from scratch in Python. (a) Given query and key vectors of dimension $d = 64$ and sequence length $T = 512$, construct the rotation matrices $R_m$ for each position $m$ using $\theta_i = 10000^{-2i/d}$. (b) Compute the rotated attention scores $\text{score}(m, n) = (R_m \mathbf{q})^\top (R_n \mathbf{k})$ for all pairs $(m, n)$. (c) Verify that $\text{score}(m, n)$ depends only on $m - n$ by checking $\text{score}(m, n) = \text{score}(m+k, n+k)$ for several values of $k$.

**Exercise 8 (★★★):** Implement **Random Fourier Features** for the RBF kernel. (a) For the Gaussian kernel $k(\mathbf{x}, \mathbf{y}) = e^{-\lVert \mathbf{x} - \mathbf{y} \rVert^2/(2\sigma^2)}$, show that the spectral density is $p(\boldsymbol{\omega}) = \mathcal{N}(\mathbf{0}, \sigma^{-2} I)$. (b) Implement the feature map $\phi(\mathbf{x}) \in \mathbb{R}^{2D}$ using $D = 100$ random frequency samples. (c) On a synthetic dataset in $\mathbb{R}^2$, compare the exact kernel matrix $K$ with the approximation $\Phi\Phi^\top$ and plot the approximation error as a function of $D$.

---

## 10. Why This Matters for AI (2026 Perspective)

| Concept | AI Impact | Concrete System |
|---------|-----------|-----------------|
| Fourier basis vectors $e^{inx}$ | Foundation of positional encodings in all transformer variants | RoPE in LLaMA-3, Gemma-2, Mistral-7B, Falcon |
| Complex Fourier form | Rotation = position; enables relative PE via shift theorem | RoPE (Su et al., 2021), xPos, YaRN |
| Parseval's identity | Energy preservation: Fourier is a unitary transform; spectral analysis of embeddings | WeightWatcher spectral analysis of LLM health |
| Spectral bias ($1/n^k$ decay) | Networks learn smooth functions first; governs training dynamics | SIREN (Sitzmann et al., 2020), NeRF frequency encoding |
| Gibbs phenomenon | Ringing in over-compressed audio/images; sudden context-length failure modes | JPEG compression artifacts, LLM boundary token issues |
| Fourier completeness | Any periodic signal can be exactly represented; digital audio encoding | MP3, AAC, Ogg Vorbis compression standards |
| Random Fourier features | $O(nD)$ kernel approximation replacing $O(n^2)$ kernel matrix | Large-scale SVM/GP inference (Rahimi & Recht, 2007) |

---

## 11. Conceptual Bridge

**Looking backward:** Fourier series is built on three pillars from earlier chapters: the **inner product** structure of $L^2$ (from [§12-02 Hilbert Spaces](../../12-Functional-Analysis/02-Hilbert-Spaces/notes.md)), **complex exponentials** $e^{inx}$ (from [§01 Mathematical Foundations](../../01-Mathematical-Foundations/README.md)), and **convergence of series** (from real analysis in [§04 Calculus](../../04-Calculus-Fundamentals/README.md)). If those foundations feel shaky, revisit them before proceeding.

**Looking forward:** Fourier series handles periodic functions on a bounded interval. The **Fourier Transform** ([§20-02](../02-Fourier-Transform/notes.md)) extends this to aperiodic signals on the entire real line by taking the period $T \to \infty$. The discrete, computationally feasible version is the **DFT and FFT** ([§20-03](../03-Discrete-Fourier-Transform-and-FFT/notes.md)). The **Convolution Theorem** ([§20-04](../04-Convolution-Theorem/notes.md)) shows why Fourier analysis is so powerful: convolution in time becomes multiplication in frequency. Finally, **Wavelets** ([§20-05](../05-Wavelets/notes.md)) overcome Fourier's fundamental limitation — the inability to localize in both time and frequency simultaneously.

```
POSITION IN THE FOURIER ANALYSIS CHAPTER
════════════════════════════════════════════════════════════════════════

  §20-01  Fourier Series          ◄── YOU ARE HERE
    ↓  (take T → ∞)
  §20-02  Fourier Transform
    ↓  (discretize: N samples)
  §20-03  DFT and FFT
    ↓  (mult. in freq ↔ conv. in time)
  §20-04  Convolution Theorem
    ↓  (localize in time AND freq)
  §20-05  Wavelets

  Prerequisites:              Forward pointers:
  §01 Mathematical Foundations → §20-02 (continuous FT)
  §04 Calculus Fundamentals    → §20-03 (discrete FT)
  §12-02 Hilbert Spaces        → §12-03 (kernel methods via Bochner)

════════════════════════════════════════════════════════════════════════
```

[← Back to Fourier Analysis](../README.md) | [Next: Fourier Transform →](../02-Fourier-Transform/notes.md)

---

## Appendix A: Extended Examples and Computations

### A.1 Full Derivation: Fourier Series of Common Waveforms

This appendix provides complete, step-by-step derivations for the Fourier coefficients of the most important periodic waveforms. These are not presented as exercises — they are reference derivations that you should work through once and understand deeply. Every step is motivated.

**A.1.1 Rectangular Pulse Train**

Define a pulse of width $2\tau$ centered at $x = 0$, repeated with period $2\pi$:
$$f(x) = \begin{cases} 1 & |x| \leq \tau \\ 0 & \tau < |x| \leq \pi \end{cases}$$

Complex coefficients: for $n \neq 0$:
$$c_n = \frac{1}{2\pi}\int_{-\pi}^{\pi} f(x)e^{-inx}\,dx = \frac{1}{2\pi}\int_{-\tau}^{\tau} e^{-inx}\,dx = \frac{1}{2\pi} \cdot \frac{e^{in\tau} - e^{-in\tau}}{in} = \frac{\sin(n\tau)}{n\pi} = \frac{\tau}{\pi}\operatorname{sinc}(n\tau/\pi)$$

where $\operatorname{sinc}(u) = \sin(\pi u)/(\pi u)$. For $n = 0$: $c_0 = \tau/\pi$ (the duty cycle).

**Key observation:** The envelope of $|c_n|$ versus $n$ follows a $\operatorname{sinc}$ function. The **first zero crossing** occurs at $n = \pi/\tau$. A narrower pulse ($\tau$ small) → wider spectrum (more high frequencies needed to represent the sharp edges). This is the **time-frequency trade-off** in action.

**Spectrum width:** The "bandwidth" (distance to first spectral null) is $B = 1/(2\tau)$ in normalized frequency. Narrow pulses are "broadband"; wide pulses (large $\tau/\pi \to 1$, approaching a constant) have most energy concentrated near $n = 0$.

**A.1.2 Full-Wave Rectified Sine**

$$f(x) = |\sin x|$$

$f$ is even and non-negative. Since $|\sin x| = |\sin(-x)|$, $b_n = 0$ for all $n$. For $a_n$:
$$a_0 = \frac{2}{\pi}\int_0^\pi \sin x\,dx = \frac{4}{\pi}$$
$$a_n = \frac{2}{\pi}\int_0^\pi |\sin x|\cos(nx)\,dx = \frac{2}{\pi}\int_0^\pi \sin x\cos(nx)\,dx$$

Using the product-to-sum formula $\sin x\cos(nx) = \frac{1}{2}[\sin((1+n)x) + \sin((1-n)x)]$:

For $n = 1$: $a_1 = \frac{1}{\pi}\int_0^\pi \sin(2x)\,dx = 0$.

For $n \neq 1$:
$$a_n = \frac{1}{\pi}\int_0^\pi [\sin((n+1)x) + \sin((1-n)x)]\,dx = \frac{1}{\pi}\left[\frac{-\cos((n+1)x)}{n+1} + \frac{\cos((1-n)x)}{1-n}\right]_0^\pi$$
$$= \frac{1}{\pi}\left(\frac{1-\cos((n+1)\pi)}{n+1} - \frac{1-\cos((n-1)\pi)}{n-1}\right) = \begin{cases} -\frac{4}{\pi(n^2-1)} & n \text{ even} \\ 0 & n \text{ odd, } n \neq 1 \end{cases}$$

So $|\sin x| = \frac{2}{\pi} - \frac{4}{\pi}\sum_{k=1}^\infty \frac{\cos(2kx)}{4k^2 - 1}$.

**Application:** The full-wave rectified sine is used in AC-to-DC conversion. Its spectrum has no odd harmonics — only even harmonics — which is why its Fourier representation converges faster (coefficients decay as $1/n^2$) than the square wave (which has $1/n$ decay).

**A.1.3 The Sawtooth Wave (Staircase Convergence)**

$f(x) = x$ on $(-\pi,\pi)$, $f(\pi) = f(-\pi) = 0$ (we assign the average at the jumps). We showed $f(x) = 2\sum_{n=1}^\infty \frac{(-1)^{n+1}}{n}\sin(nx)$.

Let us verify: at $x = \pi/2$, $f(\pi/2) = \pi/2$. The series gives:
$$2\sum_{n=1}^\infty \frac{(-1)^{n+1}}{n}\sin(n\pi/2) = 2\left(1 - \frac{1}{3} + \frac{1}{5} - \frac{1}{7} + \cdots\right) = 2 \cdot \frac{\pi}{4} = \frac{\pi}{2} \checkmark$$

This uses the Leibniz formula $1 - 1/3 + 1/5 - \cdots = \pi/4$.

**Convergence rate:** For the sawtooth, $|c_n| = 1/n$, so $\lVert f - S_N f \rVert^2 = 2\sum_{n>N} 1/n^2 \sim 2/N$ (from the integral test). Convergence is slow: to reduce the $L^2$ error below $\varepsilon$, we need $N \sim 2/\varepsilon^2$ terms.

### A.2 The Heat Equation: Fourier's Original Application

The problem that motivated Fourier's entire theory is the **heat equation** on a rod:
$$\frac{\partial u}{\partial t} = \alpha^2 \frac{\partial^2 u}{\partial x^2}, \quad x \in [0, L], \quad t > 0$$

with boundary conditions $u(0, t) = u(L, t) = 0$ (zero temperature at the ends) and initial condition $u(x, 0) = f(x)$.

**Solution by separation of variables:**

Assume $u(x, t) = X(x)T(t)$. Substituting: $X T' = \alpha^2 X'' T$, so $T'/T = \alpha^2 X''/X = -\lambda$ (both sides must equal the same constant). This gives:
$$X'' + \frac{\lambda}{\alpha^2} X = 0 \quad \text{with } X(0) = X(L) = 0$$
$$T' + \lambda T = 0$$

The boundary condition forces $X_n(x) = \sin(n\pi x/L)$ with eigenvalues $\lambda_n = (n\pi\alpha/L)^2$. The corresponding time solution is $T_n(t) = e^{-\lambda_n t}$.

**Superposition:** The general solution is:
$$u(x, t) = \sum_{n=1}^\infty b_n \sin\!\left(\frac{n\pi x}{L}\right) e^{-(n\pi\alpha/L)^2 t}$$

The coefficients $b_n$ are determined by the initial condition:
$$f(x) = u(x, 0) = \sum_{n=1}^\infty b_n \sin\!\left(\frac{n\pi x}{L}\right)$$

This is exactly the **sine series** (half-range expansion, §6.2). So $b_n = \frac{2}{L}\int_0^L f(x)\sin(n\pi x/L)\,dx$.

**Physical insight:** Each mode $\sin(n\pi x/L)$ decays at rate $e^{-(n\pi\alpha/L)^2 t}$. High-frequency modes (large $n$) decay much faster than low-frequency modes. At long times, the solution looks like just the $n = 1$ fundamental mode. This is Fourier's original discovery: heat diffusion is a low-pass filter in frequency space.

**For AI:** This is the origin of the spectral bias observation. In a sense, gradient descent on a neural network is like running the heat equation in weight space — it diffuses high-frequency components (noise) faster than low-frequency components (signal). The spectral bias of neural networks has exactly the same mathematical structure as heat diffusion.

### A.3 Dirichlet's Kernel: Detailed Analysis

Understanding why the Dirichlet kernel causes problems requires a careful look at its behavior.

**Closed form derivation:**
$$D_N(x) = \sum_{n=-N}^N e^{inx} = e^{-iNx} \cdot \frac{e^{i(2N+1)x} - 1}{e^{ix} - 1} = e^{-iNx} \cdot e^{iNx} \cdot \frac{e^{i(N+1/2)x} - e^{-i(N+1/2)x}}{e^{ix/2} - e^{-ix/2}}$$
$$= \frac{\sin((N+1/2)x)}{\sin(x/2)}$$

**Properties:**
1. $D_N(0) = 2N + 1$ (the value at the origin)
2. $\frac{1}{2\pi}\int_{-\pi}^{\pi} D_N(x)\,dx = 1$ (normalized)
3. $D_N$ has zeros at $x = 2\pi k/(2N+1)$ for $k = \pm 1, \pm 2, \ldots, \pm N$
4. $D_N$ alternates sign: it is NOT a non-negative approximate identity
5. $\lVert D_N \rVert_{L^1} \sim \frac{4}{\pi^2}\ln N$ (the $L^1$ norm grows logarithmically)

The logarithmic growth of $\lVert D_N \rVert_{L^1}$ is the root cause of convergence problems. A proper approximate identity would have $\lVert K_N \rVert_{L^1} = 1$ bounded — the Dirichlet kernel fails this, which is exactly why pointwise convergence can fail for continuous functions.

**Contrast with the Fejér kernel:**
$$F_N(x) = \frac{1}{N+1}\sum_{k=0}^N D_k(x) = \frac{1}{N+1}\left(\frac{\sin((N+1)x/2)}{\sin(x/2)}\right)^2$$

The Fejér kernel satisfies: $F_N \geq 0$, $\frac{1}{2\pi}\int F_N = 1$, and for any $\delta > 0$: $F_N(x) \to 0$ uniformly on $\{|x| \geq \delta\}$. These three properties make $F_N$ a genuine **approximate identity**, guaranteeing uniform convergence.

### A.4 Complex Analysis Connection: Laurent Series

The Fourier series of a function on $[-\pi,\pi]$ is closely related to the **Laurent series** in complex analysis. If $f$ has Fourier series $f(x) = \sum_n c_n e^{inx}$, define $z = e^{ix}$ (a point on the unit circle $|z| = 1$). Then:
$$f(x) = \sum_{n=-\infty}^{\infty} c_n z^n$$

This is a Laurent series in $z$ centered at the origin, evaluated on the unit circle. The Fourier coefficient $c_n$ is the $z^n$ coefficient of this Laurent series.

**Analyticity and series convergence:** If $f$ extends analytically to an annulus $r < |z| < 1/r$ (with $r < 1$), then the Laurent series converges absolutely and uniformly on the unit circle, and the Fourier coefficients decay exponentially: $|c_n| \leq C r^{|n|}$. This is the deep reason why analytic functions have exponentially decaying Fourier coefficients (§5.4).

**For AI:** The connection between Fourier analysis and analytic continuation underlies the theory of **analytic functions of neural networks** and the frequency-domain analysis of attention patterns. When LLM researchers analyze learned positional encodings in the complex plane, they are (often implicitly) using this Laurent series picture.

### A.5 Numerical Experiments: Convergence Visualization

The following experiments (implemented in [theory.ipynb](theory.ipynb)) illustrate the key convergence phenomena:

**Experiment 1 — Convergence speed:** Compute partial sums $S_N f$ for $N = 1, 5, 10, 50, 100$ for the square wave. Plot the $L^2$ error $\lVert f - S_N f \rVert^2 = \sum_{|n|>N} |c_n|^2$ as a function of $N$. Observe the $O(1/N)$ decay rate from the $|c_n| \sim 1/n$ coefficients.

**Experiment 2 — Gibbs phenomenon:** Plot $S_{100}f$ for the square wave. Zoom in near $x = 0$. Measure the overshoot height and verify it is $\approx 0.179 \approx 9\%$ of the jump height 2.

**Experiment 3 — Cesàro means fix Gibbs:** Plot $\sigma_{100} f$ (Cesàro means) alongside $S_{100} f$. Observe that the Gibbs overshoot is absent in the Cesàro means.

**Experiment 4 — Decay rate vs smoothness:** Compare the coefficient decay rate for: (a) square wave (discontinuous): $|c_n| \sim 1/n$; (b) triangle wave (continuous, piecewise linear): $|c_n| \sim 1/n^2$; (c) smooth bump function: $|c_n|$ decays super-polynomially. Plot $\log |c_n|$ vs $\log n$ to see the exponent.

**Experiment 5 — RoPE implementation:** Implement RoPE as in Exercise 7. Visualize the rotation matrices $R_m$ for positions $m = 0, 1, \ldots, 20$ and frequency index $i = 0$. Verify that consecutive positions differ by a fixed rotation angle $\theta_0$, confirming the Fourier basis interpretation.


---

## Appendix B: Advanced Topics

### B.1 Multidimensional Fourier Series

On a $d$-dimensional torus $\mathbb{T}^d = [-\pi,\pi]^d$, the Fourier series generalizes naturally. For $\mathbf{x} = (x_1,\ldots,x_d) \in \mathbb{T}^d$ and $\mathbf{n} = (n_1,\ldots,n_d) \in \mathbb{Z}^d$:

$$f(\mathbf{x}) = \sum_{\mathbf{n} \in \mathbb{Z}^d} c_{\mathbf{n}} e^{i\mathbf{n}\cdot\mathbf{x}}, \quad c_{\mathbf{n}} = \frac{1}{(2\pi)^d}\int_{\mathbb{T}^d} f(\mathbf{x}) e^{-i\mathbf{n}\cdot\mathbf{x}}\,d\mathbf{x}$$

The orthonormality relation is $\langle e^{i\mathbf{n}\cdot\mathbf{x}}, e^{i\mathbf{m}\cdot\mathbf{x}} \rangle = \delta_{\mathbf{n}\mathbf{m}}$, and Parseval generalizes to $\lVert f \rVert^2 = \sum_{\mathbf{n}} |c_{\mathbf{n}}|^2$.

**For AI — 2D images:** Images are functions on a 2D torus (with periodic boundary). The 2D Fourier series (or DFT) represents an image as a sum of 2D complex exponentials $e^{i(n_1 x_1 + n_2 x_2)}$. Low-frequency components $(n_1, n_2)$ near the origin represent slowly varying regions; high-frequency components represent edges and textures. JPEG compression discards high-frequency 2D DCT coefficients — this is the 2D cosine series applied to $8 \times 8$ blocks.

### B.2 Distributions and the Dirac Delta

For rigorous treatment of impulse-like functions in Fourier analysis, we need **distributions** (generalized functions). The **Dirac delta** $\delta_{x_0}$ is defined by:
$$\langle \delta_{x_0}, \phi \rangle = \phi(x_0) \quad \text{for all test functions } \phi \in C^\infty[-\pi,\pi]$$

Its Fourier coefficients are $c_n[\delta_0] = \frac{1}{2\pi}$, so its Fourier series is:
$$\delta_0(x) = \frac{1}{2\pi}\sum_{n=-\infty}^{\infty} e^{inx}$$

This is the **completeness relation** of the Fourier system: the identity operator decomposes as a sum over all frequency components. In functional analysis, this is written as $I = \sum_n |e^{inx}\rangle\langle e^{inx}|$ (in Dirac notation).

**Verification:** For any smooth $f$, $\langle \delta_0, f \rangle = f(0) = \langle \frac{1}{2\pi}\sum_n e^{inx}, f(x) \rangle = \frac{1}{2\pi}\sum_n \langle e^{inx}, f \rangle^* = \frac{1}{2\pi}\sum_n \overline{c_n[f]} \cdot 2\pi = \sum_n c_n[f] \cdot 1$... (Parseval confirms this is $f(0)$ by the inversion formula).

**Comb function:** The **Dirac comb** $\text{III}(x) = \sum_{k=-\infty}^{\infty} \delta(x - 2\pi k)$ has Fourier coefficients $c_n = \frac{1}{2\pi}$ for all $n$ — its Fourier series is the same as $\delta_0$. This might seem paradoxical, but it reflects the periodization in the series context.

### B.3 Pointwise Convergence: The Full Story (Carleson's Theorem)

The question of pointwise convergence of Fourier series remained open for 120 years after Dirichlet's 1829 result.

**Timeline of the convergence problem:**
- **Dirichlet (1829):** Piecewise smooth functions → pointwise convergence everywhere
- **Du Bois-Reymond (1876):** Exhibited a continuous function whose Fourier series diverges at one point
- **Kolmogorov (1923):** Exhibited an $L^1$ function whose Fourier series diverges everywhere!
- **Carleson (1966):** Proved that for every $f \in L^2$, the Fourier series converges **pointwise almost everywhere**
- **Hunt (1968):** Extended to $f \in L^p$ for any $p > 1$

Carleson's theorem (which won him the Abel Prize in 2006) resolved the central question: $L^2$ functions have Fourier series converging a.e. The proof is one of the most difficult in analysis, involving a refined version of the Calderón-Zygmund theory.

**Practical implication:** For all functions we encounter in ML (bounded measurable, piecewise smooth, $L^2$), Fourier series converge pointwise. The pathological examples require non-measurable functions or unbounded ones.

### B.4 Sobolev Spaces and Regularity via Fourier Series

**Definition (Sobolev spaces):** For $s \geq 0$, the Sobolev space $H^s[-\pi,\pi]$ consists of $L^2$ functions whose Fourier coefficients satisfy:
$$\lVert f \rVert_{H^s}^2 = \sum_{n=-\infty}^{\infty} (1 + n^2)^s |c_n|^2 < \infty$$

The parameter $s$ measures smoothness: $H^0 = L^2$, $H^1$ ≈ "once weakly differentiable in $L^2$", $H^s$ for $s > 1/2$ consists of continuous functions (Sobolev embedding theorem).

**Fourier characterization of regularity:** A function $f$ is in $H^s$ if and only if $|c_n| = O(n^{-s-1/2-\varepsilon})$ for some $\varepsilon > 0$. This makes Sobolev regularity entirely readable from the Fourier spectrum.

**For AI — Spectral bias quantified:** The spectral bias (§7.3) says networks approximate $f \in H^s$ more easily if $s$ is large (smooth target). Formally, a network trained by gradient descent with $n$ samples achieves $L^2$ risk of order $n^{-2s/(2s+1)}$ when targeting $f \in H^s$ — the standard minimax rate for nonparametric regression. Higher $s$ → faster convergence → need fewer samples.

**For AI — Kernel smoothness:** The RKHS of a kernel $k$ is a Sobolev space $H^s$. The RBF kernel corresponds to $H^\infty$ (infinitely smooth functions), while the Matérn kernel corresponds to $H^{s+d/2}$ for the regularity parameter $s$ and dimension $d$. This is why the choice of kernel implicitly assumes a smoothness level for the target function.

### B.5 Fejér-Riesz Theorem and Spectral Factorization

**Theorem (Fejér-Riesz):** Any trigonometric polynomial $p(x) = \sum_{|n| \leq N} c_n e^{inx}$ that is non-negative on $[-\pi,\pi]$ can be written as $p(x) = |q(x)|^2$ where $q(x) = \sum_{n=0}^N d_n e^{inx}$ is an analytic polynomial.

**Spectral factorization:** Given a non-negative spectral density $S(\omega) \geq 0$ (such as the power spectral density of a stationary process), we can find a causal filter $H(\omega)$ such that $|H(\omega)|^2 = S(\omega)$. This is the foundation of the **Wiener filter** (used in denoising) and **Cholesky-like factorizations** in probability theory.

**For AI — Connection to SSMs:** State Space Models (S4, Mamba) parameterize their convolution kernels through a spectral factorization. The kernel is written as $K = \text{IFFT}(H(\omega))$ where $H(\omega)$ is constrained to be the frequency response of a stable ARMA model. The Fejér-Riesz theorem guarantees that the spectral density $|H|^2$ can always be factored into a causal part.

### B.6 The Discrete Cosine Transform and Signal Compression

The **Discrete Cosine Transform (DCT)** of Type II is defined as:
$$X[k] = \sum_{n=0}^{N-1} x[n] \cos\!\left(\frac{\pi k (2n+1)}{2N}\right), \quad k = 0, 1, \ldots, N-1$$

This is the even-extension Fourier series (cosine series) applied to a finite sequence. The DCT has two critical properties for signal compression:

1. **Energy compaction:** For natural signals (images, audio), the DCT concentrates most of the signal energy in the first few low-frequency coefficients. The DCT of a typical $8 \times 8$ image block has 95%+ of its energy in the top-left $3 \times 3$ coefficients.

2. **Decorrelation:** The DCT approximately diagonalizes the covariance matrix of natural signals, making the coefficients nearly uncorrelated. This allows independent quantization of each coefficient.

**JPEG compression pipeline:**
1. Divide image into $8 \times 8$ blocks
2. Apply 2D DCT to each block
3. Quantize coefficients (coarser quantization for high frequencies)
4. Run-length encode the quantized sparse coefficients
5. Huffman encode

The quality parameter in JPEG controls the quantization step sizes — higher quality means finer quantization of the high-frequency DCT coefficients.

**DCT vs DFT:** The DCT avoids the "spectral leakage" (Gibbs phenomenon) of the DFT because the even extension of a finite sequence is continuous at the boundaries. This is why JPEG uses DCT rather than DFT.

### B.7 Uniform Convergence: Weierstrass M-Test Application

Let us prove uniform convergence of the triangle wave Fourier series rigorously.

**Claim:** The Fourier series $f(x) = \frac{\pi}{2} - \frac{4}{\pi}\sum_{k=0}^\infty \frac{\cos((2k+1)x)}{(2k+1)^2}$ converges uniformly.

**Proof via Weierstrass M-Test:** We need $\sum_k M_k < \infty$ where $M_k = \sup_x |a_k \cos((2k+1)x)| = |a_k| = \frac{4}{\pi(2k+1)^2}$.

Then $\sum_{k=0}^\infty M_k = \frac{4}{\pi}\sum_{k=0}^\infty \frac{1}{(2k+1)^2} = \frac{4}{\pi} \cdot \frac{\pi^2}{8} = \frac{\pi}{2} < \infty$.

Since the series of bounds converges, the Weierstrass M-test gives uniform convergence. $\square$

**Contrast with square wave:** For the square wave, $\sum_k |b_{2k+1}|/(2k+1) = \frac{4}{\pi}\sum_k \frac{1}{(2k+1)^2}$ would bound a DIFFERENT series than what appears. But the square wave's coefficients are $b_{2k+1} = 4/(\pi(2k+1))$, and $\sum 4/(\pi(2k+1))$ diverges (harmonic series). So the Weierstrass M-test fails — consistent with non-uniform convergence near the jumps.

### B.8 Connection to Functional Analysis: Completeness Proof Sketch

Here is a sketch of why the trigonometric system is complete in $L^2[-\pi,\pi]$ — a result we stated but did not prove in §2.2.

**Step 1:** Stone-Weierstrass theorem: Continuous periodic functions can be uniformly approximated by trigonometric polynomials (polynomials in $e^{inx}$).

**Step 2:** Density: Continuous functions are dense in $L^2[-\pi,\pi]$ (this is a standard result in measure theory).

**Step 3:** Combining: For any $f \in L^2$ and $\varepsilon > 0$, find continuous $g$ with $\lVert f - g \rVert_2 < \varepsilon/2$, then find trigonometric polynomial $T$ with $\lVert g - T \rVert_\infty < \varepsilon/(2\sqrt{2\pi})$, giving $\lVert g - T \rVert_2 \leq \sqrt{2\pi}\lVert g - T \rVert_\infty < \varepsilon/2$. Triangle inequality gives $\lVert f - T \rVert_2 < \varepsilon$. $\square$

**Full proof:** See [§12-02 Hilbert Spaces](../../12-Functional-Analysis/02-Hilbert-Spaces/notes.md) where the abstract completeness theorem is proved and the trigonometric system is given as the primary example.

---

## Appendix C: Problem-Solving Strategies

### C.1 Computing Fourier Coefficients: A Systematic Approach

Follow this checklist when computing Fourier coefficients:

1. **Check parity first.** Is $f$ even? → only $a_n$ terms. Odd? → only $b_n$ terms. Neither? → compute all.

2. **Check for known waveform.** Square wave, sawtooth, triangle wave, rectangular pulse — look up or recall the result.

3. **Split piecewise functions.** Write $\int_{-\pi}^{\pi} = \int_{-\pi}^{0} + \int_0^{\pi}$ if $f$ is defined piecewise.

4. **Use integration by parts** for polynomial × trig: $\int x^k \cos(nx)\,dx$. Set $u = x^k$, $dv = \cos(nx)dx$. Repeat until done.

5. **Check boundary terms.** For periodic $f$: boundary terms in integration by parts vanish. For non-periodic problems (PDEs), keep them.

6. **Use the differentiation theorem** as a shortcut: if you know $c_n[f']$, then $c_n[f] = c_n[f']/(in)$.

7. **Verify via Parseval.** Compute $\sum |c_n|^2$ and check it equals $\frac{1}{2\pi}\int |f|^2$ — this catches algebra errors.

### C.2 Determining Convergence Type

| Condition | Conclusion |
|-----------|------------|
| $f$ piecewise smooth | Pointwise convergence to $\frac{f(x^+)+f(x^-)}{2}$ (Dirichlet) |
| $f$ continuous + piecewise $C^1$ | $\lVert f - S_N f \rVert_\infty \to 0$ (uniform) |
| $f \in L^2$ | $\lVert f - S_N f \rVert_2 \to 0$ (mean-square) |
| $f \in C^k$, periodic | Uniform convergence; coefficients $O(1/n^k)$ |
| $f$ analytic | Exponential coefficient decay; exponentially fast uniform convergence |

### C.3 Recognizing When Fourier Series Appear in ML

Watch for these signatures of Fourier analysis in ML papers and code:

- **$\sin(\omega_k t)$ and $\cos(\omega_k t)$ in positional encodings** → Fourier basis
- **"Frequency domain" or "spectral" in the title** → likely uses DFT or FT
- **Complex multiplication or rotation in attention** → RoPE or ALiBi variant
- **"Low-pass filter" or "high-pass filter"** → filter = convolution = Fourier operation
- **$1/n^2$ or $1/n^4$ weight decay in regularization** → Sobolev norm regularization
- **"Spectral density" or "power spectral density"** → autocorrelation → Wiener-Khinchin → Fourier
- **State space model (SSM, S4, Mamba)** → the convolution kernel is parameterized in frequency domain


---

## Appendix D: Deep Dive — Proofs and Derivations

### D.1 Proof of Dirichlet's Theorem (Complete)

We prove the central convergence result: for $f$ piecewise smooth and $2\pi$-periodic, $S_N f(x) \to \frac{f(x^+) + f(x^-)}{2}$.

**Setup.** Write $S_N f(x) = \frac{1}{2\pi}\int_{-\pi}^{\pi} f(t) D_N(x-t)\,dt$. By periodicity, we may center the integral at $x$:
$$S_N f(x) = \frac{1}{2\pi}\int_{-\pi}^{\pi} f(x+t) D_N(t)\,dt$$

Since $\frac{1}{2\pi}\int_{-\pi}^{\pi} D_N(t)\,dt = 1$, we can write:
$$S_N f(x) - \frac{f(x^+)+f(x^-)}{2} = \frac{1}{2\pi}\int_{-\pi}^{\pi} \left[f(x+t) - \frac{f(x^+)+f(x^-)}{2}\right] D_N(t)\,dt$$

**Split the integral.** Since $D_N(t) = \frac{\sin((N+1/2)t)}{\sin(t/2)}$, and using $D_N(t) = D_N(-t)$ (even function):
$$= \frac{1}{2\pi}\int_0^{\pi} \left[f(x+t) - f(x^+) + f(x-t) - f(x^-)\right] \frac{\sin((N+1/2)t)}{\sin(t/2)}\,dt$$

**Key function.** Define:
$$g(t) = \frac{f(x+t) - f(x^+) + f(x-t) - f(x^-)}{\sin(t/2)} \cdot \frac{1}{2}$$

We need $g \in L^1[0,\pi]$. Near $t = 0$: the numerator behaves like $f'(x^+)t + (-f'(x^-))t$ (by piecewise smoothness), so numerator $= O(t)$, and $\sin(t/2) \sim t/2$, giving $g(t) = O(1)$ — bounded near $t = 0$. Away from $t = 0$: $g$ is bounded since $f$ is bounded and $\sin(t/2) \geq c > 0$. So $g \in L^1$.

**Apply Riemann-Lebesgue.** The expression becomes:
$$\frac{1}{2\pi}\int_0^{\pi} g(t) \sin\!\left(\left(N+\frac{1}{2}\right)t\right)\,dt = \frac{1}{2\pi}\hat{g}\!\left(N+\frac{1}{2}\right)$$

By the Riemann-Lebesgue lemma (proved below), this $\to 0$ as $N \to \infty$. $\square$

**Riemann-Lebesgue Lemma (proof):** For $h \in L^1[a,b]$:
$$\int_a^b h(t)\sin(\lambda t)\,dt \to 0 \quad \text{as } \lambda \to \infty$$

*Proof for step functions:* If $h = \mathbf{1}_{[c,d]}$, then $\int_c^d \sin(\lambda t)\,dt = \frac{\cos(\lambda c) - \cos(\lambda d)}{\lambda} \to 0$. By linearity, this holds for any step function. For general $L^1$ functions, approximate by step functions and use the $L^1$ bound. $\square$

### D.2 Parseval's Identity: Detailed Proof

**Theorem.** For $f \in L^2[-\pi,\pi]$ with complex Fourier coefficients $\{c_n\}_{n \in \mathbb{Z}}$:
$$\frac{1}{2\pi}\int_{-\pi}^{\pi} |f(x)|^2\,dx = \sum_{n=-\infty}^{\infty} |c_n|^2$$

**Proof (assuming completeness).** Consider the partial sum error:
$$\lVert f - S_N f \rVert^2 = \lVert f \rVert^2 - \lVert S_N f \rVert^2$$

The cross-term vanishes: $\langle f, S_N f \rangle = \sum_{|n| \leq N} |c_n|^2 = \lVert S_N f \rVert^2$ (by the projection property: $S_N f$ is the best approximation to $f$ from $\text{span}\{e^{inx} : |n| \leq N\}$).

So $\lVert f - S_N f \rVert^2 = \lVert f \rVert^2 - \sum_{|n| \leq N} |c_n|^2 \geq 0$ (Bessel).

By completeness, $\lVert f - S_N f \rVert \to 0$, so:
$$0 = \lim_{N\to\infty}\lVert f - S_N f \rVert^2 = \lVert f \rVert^2 - \lim_{N\to\infty}\sum_{|n| \leq N} |c_n|^2 = \lVert f \rVert^2 - \sum_{n=-\infty}^\infty |c_n|^2$$

This gives the result. $\square$

**Remark:** The key step is $\lVert f - S_N f \rVert^2 = \lVert f \rVert^2 - \lVert S_N f \rVert^2$. This is the **Pythagorean theorem** in $L^2$: $f$ decomposes orthogonally into $S_N f$ (the projection) and $f - S_N f$ (the residual). No analogy — it IS the Pythagorean theorem, in an infinite-dimensional space.

### D.3 Gibbs Phenomenon: Precise Quantification

Let $f(x) = \operatorname{sgn}(x)$ (square wave). We compute the exact overshoot at the jump $x = 0^+$.

The partial sum $S_N f(x) = \frac{4}{\pi}\sum_{k=0}^N \frac{\sin((2k+1)x)}{2k+1}$ has its first local maximum at $x_N = \frac{\pi}{2N+1}$ (the first zero of $(S_N f)'(x) = \frac{4}{\pi}\sum_{k=0}^N \cos((2k+1)x)$, which is zero when $x = \pi/(2N+1)$).

Computing:
$$S_N f\!\left(\frac{\pi}{2N+1}\right) = \frac{4}{\pi}\sum_{k=0}^N \frac{\sin\!\left(\frac{(2k+1)\pi}{2N+1}\right)}{2k+1}$$

This is a Riemann sum for $\frac{2}{\pi}\int_0^\pi \frac{\sin t}{t}\,dt$ with step $\Delta t = \pi/(2N+1)$. As $N \to \infty$:
$$\lim_{N\to\infty} S_N f\!\left(\frac{\pi}{2N+1}\right) = \frac{2}{\pi}\int_0^\pi \frac{\sin t}{t}\,dt = \frac{2}{\pi} \cdot \text{Si}(\pi) \approx \frac{2}{\pi} \times 1.8519 \approx 1.1790$$

The jump in $f$ is from $-1$ to $+1$ (height 2). The overshoot above the limiting value $+1$ is $1.1790 - 1.0 = 0.1790$, which is $8.95\% \approx 9\%$ of the jump height.

**Numerical value of $\text{Si}(\pi)$:**
$$\text{Si}(\pi) = \int_0^\pi \frac{\sin t}{t}\,dt = \sum_{k=0}^\infty \frac{(-1)^k \pi^{2k+1}}{(2k+1)\cdot(2k+1)!} \approx 1.8519$$

**Why $9\%$ is universal:** The integral $\frac{2}{\pi}\text{Si}(\pi) - 1 \approx 0.0895$ does not depend on the specific piecewise smooth function — only on the jump height. Every jump discontinuity in any piecewise smooth function gives a Gibbs overshoot of this universal fraction.

### D.4 The Weierstrass Approximation Theorem and Completeness

**Weierstrass Approximation Theorem (1885):** Every continuous function $f$ on $[a,b]$ can be uniformly approximated by algebraic polynomials.

**Trigonometric variant:** Every continuous $2\pi$-periodic function can be uniformly approximated by trigonometric polynomials $T(x) = \sum_{|n| \leq N} c_n e^{inx}$.

**Proof sketch (Fejér's approach):** The Cesàro means $\sigma_N f$ are trigonometric polynomials (each $S_k f$ is a trigonometric polynomial of degree $k$, and their average $\sigma_N f$ is also one). Fejér proved $\sigma_N f \to f$ uniformly for continuous $f$. $\square$

**Why completeness follows:** Given $f \in L^2$ and $\varepsilon > 0$:
1. Find continuous $g$ with $\lVert f - g \rVert_2 < \varepsilon/2$ (density of $C$ in $L^2$)
2. Find trigonometric polynomial $T$ with $\lVert g - T \rVert_\infty < \varepsilon/(2\sqrt{2\pi})$ (Weierstrass/Fejér)
3. Then $\lVert g - T \rVert_2 \leq \sqrt{2\pi}\lVert g - T \rVert_\infty < \varepsilon/2$
4. Triangle inequality: $\lVert f - T \rVert_2 < \varepsilon$ $\square$

### D.5 Fourier Series and Operator Theory

The Fourier series establishes a fundamental connection between function analysis and operator theory.

**The operator perspective:** Consider the differentiation operator $D = \frac{d}{dx}$ on periodic functions. The eigenfunctions of $D$ are exactly the complex exponentials:
$$D e^{inx} = in \cdot e^{inx}$$

with eigenvalue $in$. The Fourier series is the **eigenfunction expansion** of $D$. Every linear differential operator with constant coefficients:
$$L = a_k D^k + a_{k-1}D^{k-1} + \cdots + a_0$$

is diagonalized by the Fourier basis:
$$L e^{inx} = (a_k (in)^k + a_{k-1}(in)^{k-1} + \cdots + a_0) e^{inx} = p(in) e^{inx}$$

where $p(z) = a_k z^k + \cdots + a_0$ is the **symbol** of the operator.

**For AI:** Attention is (in the linear case) an operator on sequence positions. The linear attention matrix $A$ with entries $A_{mn} = K(m-n)$ (for a shift-invariant kernel) is diagonalized by the Fourier basis — its eigenvectors are the complex exponentials and its eigenvalues are the Fourier transform of $K$. This is why spectral analysis of attention patterns connects directly to Fourier theory, and why RoPE (which multiplies by $e^{im\theta}$) is so natural in this framework.

**Compact operators:** If the coefficients $p(in) \to 0$ as $|n| \to \infty$, then $L$ is a compact operator. The **inverse problem** (recovering $f$ from $Lf = g$) is then ill-posed (small changes in $g$ can cause large changes in $f$) — this is the mathematical underpinning of regularization in machine learning.

---

## Appendix E: Summary Tables

### E.1 Fourier Coefficient Reference

| Function $f(x)$ on $(-\pi,\pi)$ | $c_n$ (complex) | Notes |
|--------------------------------|-----------------|-------|
| $1$ (constant) | $c_0 = 1$, $c_n = 0$ ($n \neq 0$) | DC component only |
| $e^{iax}$ ($a \in \mathbb{Z}$) | $c_a = 1$, $c_n = 0$ ($n \neq a$) | Single frequency |
| $\cos(mx)$ | $c_m = c_{-m} = 1/2$, rest $0$ | |
| $\sin(mx)$ | $c_m = 1/(2i)$, $c_{-m} = -1/(2i)$ | |
| $\text{sgn}(x)$ (square wave) | $c_n = \frac{2}{in\pi}$ ($n$ odd), $0$ (even/zero) | $1/n$ decay |
| $x$ (sawtooth) | $c_n = \frac{(-1)^{n+1}}{in}$ ($n \neq 0$), $0$ ($n=0$) | $1/n$ decay |
| $|x|$ (triangle) | $c_n = -\frac{2}{\pi n^2}$ ($n$ odd), $c_0 = \pi/2$ | $1/n^2$ decay |
| $x^2$ | $c_0 = \pi^2/3$, $c_n = \frac{2(-1)^n}{n^2}$ ($n \neq 0$) | $1/n^2$ decay |
| $e^{ax}$ | $c_n = \frac{\sinh(\pi a)}{\pi(a-in)}$ | Exponential decay |
| $\delta(x)$ (Dirac) | $c_n = \frac{1}{2\pi}$ (all $n$) | No decay (distributional) |

### E.2 Properties Reference Table

| Property | Time domain | Frequency domain |
|----------|-------------|-----------------|
| Linearity | $\alpha f + \beta g$ | $\alpha c_n[f] + \beta c_n[g]$ |
| Shift | $f(x - x_0)$ | $e^{-inx_0} c_n[f]$ |
| Modulation | $e^{imx} f(x)$ | $c_{n-m}[f]$ |
| Differentiation | $f'(x)$ | $in \cdot c_n[f]$ |
| Integration | $\int_0^x f(t)\,dt$ | $c_n[f]/(in)$ ($n \neq 0$) |
| Reflection | $f(-x)$ | $c_{-n}[f]$ |
| Conjugation | $\overline{f(x)}$ | $\overline{c_{-n}[f]}$ |
| Convolution | $(f*g)(x) = \frac{1}{2\pi}\int f(t)g(x-t)\,dt$ | $c_n[f] \cdot c_n[g]$ |
| Multiplication | $f(x)g(x)$ | $\sum_k c_k[f] c_{n-k}[g]$ (convolution in freq) |

### E.3 Convergence Summary

| Type | Condition | Rate | Gibbs? |
|------|-----------|------|--------|
| $L^2$ | $f \in L^2$ (always) | $\lVert f - S_N f \rVert_2 \to 0$ | N/A |
| Pointwise | $f$ piecewise smooth | $O(1/N)$ error at smooth points | Yes at jumps |
| Uniform | $f \in C^1$, periodic | $O(1/N)$ in $\lVert \cdot \rVert_\infty$ | No |
| Superpolynomial | $f \in C^k$, periodic | $O(1/N^k)$ | No |
| Exponential | $f$ analytic | $O(e^{-cN})$ | No |
| Cesàro | $f$ continuous | Uniform (Fejér) | No |


---

## Appendix F: Complete Worked Problems

### F.1 Problem: Full Solution of Fourier Series for $f(x) = x(\pi - x)$ on $[0, \pi]$

This is a common PDE boundary value problem. We want the **sine series** of $f(x) = x(\pi - x)$ on $[0, \pi]$ (odd extension):

$$b_n = \frac{2}{\pi}\int_0^\pi x(\pi - x)\sin(nx)\,dx$$

**Step 1: Split the integral.**
$$b_n = \frac{2}{\pi}\left[\pi\int_0^\pi x\sin(nx)\,dx - \int_0^\pi x^2\sin(nx)\,dx\right]$$

**Step 2: Compute $\int_0^\pi x\sin(nx)\,dx$ by parts.**

Set $u = x$, $dv = \sin(nx)dx$:
$$\int_0^\pi x\sin(nx)\,dx = \left[\frac{-x\cos(nx)}{n}\right]_0^\pi + \frac{1}{n}\int_0^\pi \cos(nx)\,dx = \frac{-\pi\cos(n\pi)}{n} + \frac{\sin(nx)}{n^2}\bigg|_0^\pi = \frac{\pi(-1)^{n+1}}{n}$$

**Step 3: Compute $\int_0^\pi x^2\sin(nx)\,dx$ by parts twice.**

First integration by parts: $u = x^2$, $dv = \sin(nx)dx$:
$$\int_0^\pi x^2\sin(nx)\,dx = \left[\frac{-x^2\cos(nx)}{n}\right]_0^\pi + \frac{2}{n}\int_0^\pi x\cos(nx)\,dx = \frac{-\pi^2(-1)^n}{n} + \frac{2}{n}\int_0^\pi x\cos(nx)\,dx$$

Second integration by parts: $\int_0^\pi x\cos(nx)\,dx = \left[\frac{x\sin(nx)}{n}\right]_0^\pi - \frac{1}{n}\int_0^\pi \sin(nx)\,dx = 0 + \frac{1}{n^2}[\cos(nx)]_0^\pi = \frac{(-1)^n - 1}{n^2}$

So: $\int_0^\pi x^2\sin(nx)\,dx = \frac{-\pi^2(-1)^n}{n} + \frac{2((-1)^n - 1)}{n^3}$

**Step 4: Combine.**
$$b_n = \frac{2}{\pi}\left[\pi \cdot \frac{\pi(-1)^{n+1}}{n} - \left(\frac{-\pi^2(-1)^n}{n} + \frac{2((-1)^n-1)}{n^3}\right)\right]$$
$$= \frac{2}{\pi}\left[\frac{\pi^2(-1)^{n+1}}{n} + \frac{\pi^2(-1)^n}{n} - \frac{2((-1)^n-1)}{n^3}\right]$$
$$= \frac{2}{\pi}\left[0 - \frac{2((-1)^n - 1)}{n^3}\right] = \frac{-4((-1)^n - 1)}{\pi n^3} = \begin{cases} 8/(\pi n^3) & n \text{ odd} \\ 0 & n \text{ even} \end{cases}$$

**Result:** $f(x) = x(\pi - x) = \frac{8}{\pi}\sum_{k=0}^\infty \frac{\sin((2k+1)x)}{(2k+1)^3}$

**Verification via Parseval:** $\frac{2}{\pi}\int_0^\pi [x(\pi-x)]^2\,dx = \frac{2}{\pi}\int_0^\pi (x^2\pi^2 - 2x^3\pi + x^4)\,dx = \frac{2}{\pi}\left[\frac{\pi^2 \cdot \pi^3}{3} - \frac{\pi^5}{2} + \frac{\pi^5}{5}\right] = \frac{2\pi^4}{30} = \frac{\pi^4}{15}$

Parseval: $\sum_{k=0}^\infty b_{2k+1}^2 = \sum_{k=0}^\infty \frac{64}{\pi^2(2k+1)^6} = \frac{\pi^4}{15}$, giving $\sum_{k=0}^\infty \frac{1}{(2k+1)^6} = \frac{\pi^6}{960}$. (Verifiable!)

**Application to heat equation:** The solution of $u_t = \alpha^2 u_{xx}$ on $[0,\pi]$ with $u(0,t) = u(\pi,t) = 0$ and $u(x,0) = x(\pi-x)$ is:
$$u(x,t) = \frac{8}{\pi}\sum_{k=0}^\infty \frac{1}{(2k+1)^3}\sin((2k+1)x)\,e^{-\alpha^2(2k+1)^2 t}$$

At $t = 0$: this reduces to the Fourier series of $x(\pi-x)$. At large $t$: the dominant term is $\frac{8}{\pi}\sin(x)e^{-\alpha^2 t}$ — the solution relaxes toward zero via the fundamental mode.

### F.2 Problem: Fourier Series of $f(x) = e^{ix}$ — The Modulation Property

This trivial-seeming example illuminates the modulation property.

$f(x) = e^{ix}$ is already a single Fourier mode with $c_1 = 1$ and $c_n = 0$ for $n \neq 1$.

Now consider $g(x) = e^{iax}f(x) = e^{i(a+1)x}$ for integer $a$. By the modulation property:
$$c_n[g] = c_{n-a}[f] = \delta_{n-a, 1} = \delta_{n, a+1}$$

So $g$ has all energy concentrated at frequency $n = a+1$. This is the **frequency shift** or **modulation** property: multiplying by $e^{iax}$ shifts the spectrum by $a$.

**Application to RoPE in detail:** In RoPE, the query at position $m$ for dimension $i$ is:
$$q_{m,i}^{\text{rotated}} = (q_{2i} + iq_{2i+1}) \cdot e^{im\theta_i}$$

This is multiplication by the complex exponential $e^{im\theta_i}$ — i.e., modulation by frequency $m$. The inner product with key at position $n$:
$$\langle q_m^{\text{rot}}, k_n^{\text{rot}} \rangle = \operatorname{Re}\!\left[\sum_i (q_{2i}+iq_{2i+1})e^{im\theta_i} \cdot \overline{(k_{2i}+ik_{2i+1})e^{in\theta_i}}\right]$$
$$= \operatorname{Re}\!\left[\sum_i (q_{2i}+iq_{2i+1})\overline{(k_{2i}+ik_{2i+1})} \cdot e^{i(m-n)\theta_i}\right]$$

The score depends only on $(m-n)$ — **relative position** — by the modulation-then-conjugate-product structure. This is the Fourier basis shift theorem instantiated in transformer attention.

### F.3 Problem: Convergence Rate Comparison

Compare the convergence rates of the Fourier series for four functions:

**Function A:** Square wave (1 discontinuity per period)
- $|c_n| = O(1/n)$ → $\lVert f - S_N \rVert_2^2 = O(1/N)$
- Needs $N \sim 10^6$ terms for $10^{-3}$ relative error

**Function B:** Triangle wave (continuous, corners = 1 discontinuity in derivative)
- $|c_n| = O(1/n^2)$ → $\lVert f - S_N \rVert_2^2 = O(1/N^3)$
- Needs $N \sim 10^2$ terms for $10^{-3}$ relative error

**Function C:** Smooth periodic bump $f(x) = e^{\cos x}$
- $f$ is analytic → $|c_n|$ decays exponentially, $|c_n| \leq Ce^{-cn}$
- Needs $N \sim 20$ terms for $10^{-6}$ relative error

**Function D:** $f(x) = 1/2 - |x|/(2\pi)$ (slowly varying linear ramp)
- $|c_n| = 1/(2\pi^2 n^2)$ → similar to triangle wave
- Fast convergence

**Lesson:** The algebraic convergence rates for discontinuous or non-smooth functions make Fourier truncation expensive. This motivates wavelets (§20-05), which can represent localized features (edges) efficiently without paying the Gibbs penalty.

### F.4 Problem: Convolution in Fourier Space

**Setup:** Let $f(x) = \cos(x)$ and $g(x) = \sin(2x)$. Compute $(f * g)(x) = \frac{1}{2\pi}\int_{-\pi}^{\pi} f(t)g(x-t)\,dt$ using the convolution theorem.

**Using the convolution theorem:** $c_n[f*g] = c_n[f] \cdot c_n[g]$.

Fourier coefficients of $f = \cos x$: $c_1 = c_{-1} = 1/2$, rest zero.

Fourier coefficients of $g = \sin(2x)$: $c_2 = 1/(2i) = -i/2$, $c_{-2} = i/2$, rest zero.

Product $c_n[f] \cdot c_n[g]$: This is zero for all $n$ since the non-zero frequencies of $f$ ($n = \pm 1$) and $g$ ($n = \pm 2$) are disjoint!

So $(f * g)(x) = 0$ — the convolution of $\cos(x)$ and $\sin(2x)$ is identically zero.

**Direct verification:** $\frac{1}{2\pi}\int_{-\pi}^{\pi} \cos(t)\sin(2(x-t))\,dt = \frac{1}{2\pi}\int \cos t(\sin 2x\cos 2t - \cos 2x\sin 2t)\,dt$. Each integral is zero by orthogonality. $\checkmark$

**Lesson:** The convolution of two functions with non-overlapping spectra is zero. This is the **frequency-domain multiplication** interpretation of convolution. In signal processing, this means two signals at different frequencies don't interfere — the superposition principle holds in frequency space.

**For AI:** This is why attention heads can specialize to different frequency ranges. If head $h_1$ attends to low-frequency positional patterns and head $h_2$ attends to high-frequency syntactic patterns, their convolution-like operations in frequency space are orthogonal.

### F.5 Problem: Parseval and the Isoperimetric Inequality

The **isoperimetric inequality** — that among all closed curves of length $L$, the circle encloses the maximum area — has an elegant proof via Fourier series.

**Setup:** Parameterize a smooth closed curve of length $L$ by arclength: $(x(t), y(t))$ for $t \in [0, L]$, with period $L$. The constraint is $\dot{x}^2 + \dot{y}^2 = 1$.

**Area via Green's theorem:** $A = \frac{1}{2}\int_0^L (x\dot{y} - y\dot{x})\,dt$.

**Fourier expand:** $x(t) = \sum_n a_n e^{2\pi int/L}$, $y(t) = \sum_n b_n e^{2\pi int/L}$.

**Apply Parseval to the arc length constraint:** $1 = \dot{x}^2 + \dot{y}^2$ → after Parseval:
$$\sum_n \left(\frac{2\pi n}{L}\right)^2 (|a_n|^2 + |b_n|^2) = 1$$

**Bound the area:** Using the Fourier expansion and Cauchy-Schwarz:
$$|A| = \pi \left|\sum_n n(a_n\bar{b}_n - \bar{a}_n b_n)\right| \leq \pi \sum_n |n|(|a_n|^2 + |b_n|^2) \leq \frac{\pi L^2}{4\pi^2}\sum_n \frac{4\pi^2 n^2}{L^2}(|a_n|^2+|b_n|^2) = \frac{L^2}{4\pi}$$

Equality holds when only the $n = \pm 1$ terms are non-zero — i.e., when the curve is a circle.

**Conclusion:** $A \leq L^2/(4\pi)$, with equality for a circle. This is the isoperimetric inequality, proved entirely via Parseval's theorem.

**For AI:** Isoperimetric inequalities in function spaces underlie **capacity control** in learning theory. The VC dimension and Rademacher complexity measure the "surface area to volume ratio" of hypothesis classes in high-dimensional function spaces — and Fourier analysis provides the tools to compute these quantities.


---

## Appendix G: In-Depth AI Applications

### G.1 Transformer Positional Encodings: A Unified View

To understand the deep connection between Fourier series and transformer positional encodings, let us develop the theory carefully.

**The fundamental problem in attention.** The self-attention mechanism $\text{Attention}(Q,K,V) = \text{softmax}(QK^\top/\sqrt{d_k})V$ is **permutation-equivariant**: permuting the input tokens permutes the output, but the attention weights treat all positions symmetrically. Without positional encodings, "the cat sat on the mat" and "the mat sat on the cat" would produce the same attention weights.

**The sinusoidal solution (Vaswani et al., 2017).** Add a position-dependent vector $\text{PE}(m) \in \mathbb{R}^d$ to the $m$-th token embedding before attention. Use:
$$\text{PE}(m)_{2i} = \sin\!\left(\frac{m}{\theta_i}\right), \quad \text{PE}(m)_{2i+1} = \cos\!\left(\frac{m}{\theta_i}\right), \quad \theta_i = 10000^{2i/d}$$

**The key property.** For any fixed offset $k$, there exists a linear map $M_k \in \mathbb{R}^{d \times d}$ such that $\text{PE}(m + k) = M_k \text{PE}(m)$. This is because:
$$\begin{pmatrix} \sin((m+k)/\theta_i) \\ \cos((m+k)/\theta_i) \end{pmatrix} = \begin{pmatrix} \cos(k/\theta_i) & \sin(k/\theta_i) \\ -\sin(k/\theta_i) & \cos(k/\theta_i) \end{pmatrix} \begin{pmatrix} \sin(m/\theta_i) \\ \cos(m/\theta_i) \end{pmatrix}$$

Each dimension pair $(2i, 2i+1)$ transforms via a rotation by $k/\theta_i$ — a Fourier shift in 2D.

**Why different frequencies?** At frequency $1/\theta_i = 10000^{-2i/d}$:
- **$i = 0$:** $\theta_0 = 1$, period $= 2\pi \approx 6.28$ — distinguishes adjacent tokens
- **$i = d/4$:** $\theta_{d/4} = 10000^{1/2} \approx 100$ — distinguishes tokens $\sim 100$ apart
- **$i = d/2-1$:** $\theta_{d/2-1} = 10000$ — distinguishes tokens up to distance $10000$

This is exactly the geometric progression of harmonics in a Fourier series on the interval $[0, 10000]$.

**The Fourier basis interpretation.** Define $\mathbf{z}_m = \text{PE}(m)$. The inner product $\mathbf{z}_m \cdot \mathbf{z}_n = \sum_{i=0}^{d/2-1} \cos((m-n)/\theta_i)$ depends only on $m - n$. This is the **autocorrelation function** of the positional encoding — it measures how "similar" two positions are based on their relative offset.

**Limitation.** These encodings are absolute (the encoding of position 5 is the same regardless of whether the sequence has length 10 or 1000). For sequences longer than those seen in training, the high-frequency components ($\theta_0 = 1$) still work, but the model has never seen the low-frequency components ($\theta_{d/2-1} = 10000$) vary much. This causes length generalization failure.

### G.2 RoPE: Relative Positions via Complex Multiplication

**The RoPE construction.** Instead of adding $\text{PE}(m)$ to the embedding, RoPE **rotates** the query and key vectors:

$$q_m = R_m q, \quad k_n = R_n k$$

where $R_m$ is the block-diagonal rotation matrix:
$$R_m = \begin{pmatrix} R(\theta_0, m) & & \\ & R(\theta_1, m) & \\ & & \ddots \end{pmatrix}, \quad R(\theta, m) = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix}$$

**Computing the attention score.**
$$q_m^\top k_n = (R_m q)^\top (R_n k) = q^\top R_m^\top R_n k = q^\top R_{m-n} k$$

since $R_m^\top R_n = R_{n-m}$ (rotation by $m - n$). The score is a function of $m - n$ only — **position-independent in the absolute sense, position-sensitive in the relative sense**.

**The complex representation.** In complex notation, the $i$-th pair $(q_{2i}, q_{2i+1})$ maps to the complex number $q_{2i} + iq_{2i+1} \in \mathbb{C}$. The RoPE operation is:
$$q_{2i} + iq_{2i+1} \;\mapsto\; (q_{2i} + iq_{2i+1}) \cdot e^{im\theta_i}$$

This is multiplication by the unit complex number $e^{im\theta_i}$ — a phase rotation of the complex Fourier coefficient at frequency $\theta_i$.

**Long-context extrapolation.** The base frequency $\theta_i = 10000^{-2i/d}$ means the maximum period is $2\pi/\theta_{d/2-1} = 2\pi \times 10000 \approx 62832$. For positions beyond this, the encoding **wraps around** (becomes periodic) and position information is lost. This is why standard RoPE models struggle at contexts beyond $\sim 4096$ tokens.

**Fix — Extended RoPE:** Scale $\theta_i \to \theta_i / s$ where $s$ is the ratio of new context length to training context length (Position Interpolation, Chen et al., 2023). Alternatively, YaRN (Peng et al., 2023) uses a non-uniform frequency scaling based on Fourier analysis of the empirical attention patterns.

### G.3 Spectral Bias: Mathematical Theory

The spectral bias (or frequency principle) of neural networks has a rigorous formulation in terms of the NTK and Fourier analysis.

**The NTK framework.** For a wide neural network $f_\theta(\mathbf{x})$ trained with gradient descent, the NTK matrix $\Theta(\mathbf{x}, \mathbf{x}') = \nabla_\theta f_\theta(\mathbf{x}) \cdot \nabla_\theta f_\theta(\mathbf{x}')$ is approximately constant during training. The training dynamics are:
$$\frac{d f_\theta(\mathbf{x})}{dt} = -\int \Theta(\mathbf{x}, \mathbf{x}') \nabla_{f} \mathcal{L}(f(\mathbf{x}'), y')\,d\mathbf{x}'$$

This is a **kernel regression** problem in function space, where $\Theta$ acts as the kernel.

**Spectral decomposition.** Let $\lambda_k, \phi_k$ be eigenvalues/eigenfunctions of $\Theta$ in $L^2$. The projection of the target $f^*$ onto eigenvector $\phi_k$ has coefficient $\alpha_k = \langle f^*, \phi_k \rangle$. Under gradient descent:
$$\alpha_k(t) = \alpha_k(0) + (1 - e^{-\lambda_k t}) \alpha_k^\infty$$

where $\alpha_k^\infty$ is the target coefficient. The coefficient at eigenfrequency $k$ converges at rate $\lambda_k$ — fast if $\lambda_k$ large, slow if $\lambda_k$ small.

**The spectral bias.** For standard MLPs with Gaussian initialization and ReLU activations, the NTK has eigenvalues $\lambda_k$ that decrease with frequency (high-frequency eigenfunctions have small eigenvalues). Therefore:
- Low-frequency Fourier components have large eigenvalues → learned quickly
- High-frequency components have small eigenvalues → learned slowly

**Quantitative: the Barron approximation theorem.** Functions that can be represented with $O(C^2 \varepsilon^{-2})$ neurons (where $C = \int |\omega||\hat{f}(\omega)|\,d\omega$ is the Barron norm) can be approximated to $\varepsilon$ accuracy in $L^2$. The Barron norm penalizes high-frequency components: if $f^*$ has large high-frequency content ($|\hat{f^*}(\omega)|$ large for large $|\omega|$), then $C$ is large and the network needs more parameters.

**Implications for practice:**
1. **Overfitting to noise:** High-frequency noise (which has large $|\omega|$) is learned last → early stopping acts as a frequency low-pass filter
2. **Data augmentation:** Adding geometric augmentations (rotations, flips) destroys high-frequency features → forces the network to rely on low-frequency structure → better generalization
3. **NeRF:** Without positional encoding, networks can't represent the high-frequency variation in scene appearance → Fourier feature encodings inject high frequencies explicitly

### G.4 SIREN Networks: Replacing ReLU with Sine

**SIREN (Sitzmann et al., 2020).** Instead of using $\text{ReLU}$ activations, SIREN uses:
$$f_\theta(\mathbf{x}) = W_L \phi_{L-1} \circ \cdots \circ \phi_1(\mathbf{x}), \quad \phi_i(\mathbf{x}) = \sin(W_i \mathbf{x} + \mathbf{b}_i)$$

The derivative $\nabla_\mathbf{x} f$ is again a SIREN (with cosines → sines one layer deeper). This is the key property: **the derivative of a SIREN is also a SIREN** with the same architecture.

**Fourier interpretation.** The output of a SIREN is a sum of sinusoids at multiple frequencies and phases. The composition of sine activations creates a rich set of Fourier components. The Fourier spectrum of a SIREN output can represent high-frequency content that a standard ReLU network would struggle to learn.

**Mathematical property.** For a single-layer SIREN with $n$ neurons, the output is:
$$f(x) = \sum_{k=1}^n c_k \sin(\omega_k x + \phi_k)$$

a sum of $n$ Fourier basis functions with learned frequencies, amplitudes, and phases. Unlike a standard Fourier series with fixed frequencies $\omega_k = k$, SIREN can **choose** the frequencies freely — making it more expressive.

**Applications:** Implicit neural representations (NeRF, video), solving PDEs (physics-informed neural networks), generative modeling of images and 3D shapes.

### G.5 The Spectral Analysis of LLM Attention Patterns

Recent research (2023–2024) has revealed striking spectral structure in LLM attention patterns:

**Attention as a filter.** In a transformer layer, the attention mechanism computes:
$$\mathbf{y}_m = \sum_n A_{mn} V \mathbf{x}_n$$

where $A_{mn} = \text{softmax}_n(q_m^\top k_n / \sqrt{d_k})$. If $A_{mn}$ depends only on $m - n$ (which is approximately true for well-trained models with relative PE), then this is a **convolution** with kernel $a[k] = A_{mk}$. In frequency space:
$$\hat{Y}(\xi) = \hat{A}(\xi) \cdot \hat{X}(\xi)$$

The attention pattern is a frequency filter. Different heads learn different filters:
- **Local heads:** $A_{mn}$ large only for $|m - n|$ small → low-pass filter (smooth the sequence)
- **Global heads:** $A_{mn}$ relatively uniform → constant filter (compute the mean)
- **Periodic heads:** $A_{mn}$ large when $m - n \equiv k \pmod{T}$ → periodic filter (detect regular patterns)

**Empirical findings:**
- In BERT, some heads function as "copy heads" (attend to identical tokens) and "position heads" (attend to positions with specific relative offsets)
- In GPT-2/3, induction heads attend strongly to previous occurrences of the same token
- The frequency-domain view reveals why attention heads naturally factorize into frequency bands

**WeightWatcher (Martin & Mahoney, 2021).** The singular value spectra of weight matrices in LLMs follow power laws $\rho(\sigma) \sim \sigma^{-\alpha}$ with exponent $\alpha$ related to model quality. Healthy models have $\alpha \in [2, 4]$. Overfit or poorly trained models have $\alpha$ outside this range. This is a **Fourier analysis of the weight matrices** — the spectral density tells us about the effective rank and implicit regularization.

### G.6 Circular Convolution and LLMs

**The circular convolution theorem** (proved in §20-04) states that in the DFT domain, elementwise multiplication corresponds to circular convolution in the time domain. This has a direct connection to autoregressive language models.

**Autoregressive language models as circular filters.** A causal language model computes $p(x_t | x_1, \ldots, x_{t-1})$ by attending to all previous tokens. In the frequency domain:
$$\hat{P}(\xi) = \hat{A}(\xi) \cdot \hat{X}(\xi)$$

where $\hat{A}(\xi)$ is a one-sided (causal) filter. The **causal** constraint means $A_{mn} = 0$ for $m < n$ — only attending to past tokens. In frequency space, this corresponds to a filter whose phase response is $-\pi/2$ for all frequencies (a Hilbert transform).

**State space models.** Models like S4, H3, and Mamba parameterize the causal convolution kernel $h[n]$ directly in frequency space (via diagonal state space $A$ matrix). The output is:
$$y[t] = (x * h)[t] = \text{IFFT}(\hat{X} \cdot \hat{H})$$

where $\hat{H}$ is the learned frequency response. This is exactly the convolution theorem applied to sequence modeling, and it enables $O(N \log N)$ computation instead of $O(N^2)$ attention.


---

## Appendix H: Connections to Probability and Statistics

### H.1 Characteristic Functions as Fourier Transforms of Distributions

The **characteristic function** of a random variable $X$ is:
$$\varphi_X(t) = \mathbb{E}[e^{itX}] = \int_{-\infty}^{\infty} e^{itx}\,dF_X(x)$$

This is the Fourier transform of the probability measure $F_X$! The characteristic function exists for all random variables (unlike the moment generating function) and uniquely determines the distribution.

**Fourier inversion of densities.** If $X$ has density $p_X$:
$$\varphi_X(t) = \int e^{itx} p_X(x)\,dx = \hat{p}_X(-t)$$

The density is recovered by the inverse Fourier transform: $p_X(x) = \frac{1}{2\pi}\int e^{-itx} \varphi_X(t)\,dt$.

**Central limit theorem via characteristic functions.** Let $X_1, \ldots, X_n$ i.i.d. with mean $\mu$ and variance $\sigma^2$. The standardized sum $S_n = (X_1 + \cdots + X_n - n\mu)/(\sigma\sqrt{n})$ has characteristic function:
$$\varphi_{S_n}(t) = \left[\varphi_X\!\left(\frac{t}{\sigma\sqrt{n}}\right) e^{-it\mu/(\sigma\sqrt{n})}\right]^n$$

Expanding $\varphi_X(s) = 1 + \mu(is) + \frac{\mu^2 + \sigma^2}{2}(is)^2 + O(s^3)$ for small $s$ and substituting:
$$\varphi_{S_n}(t) = \left[1 - \frac{t^2}{2n} + O(n^{-3/2})\right]^n \to e^{-t^2/2} \quad \text{as } n \to \infty$$

The Gaussian $e^{-t^2/2}$ is the Fourier transform of the standard normal density. The CLT is a **convergence in Fourier space** to the Gaussian fixed point.

**For AI — Training stability.** The distribution of gradient updates in neural network training follows a CLT when the batch size is large: the stochastic gradient is a sum of $B$ per-sample gradients, so its distribution is approximately Gaussian for large $B$. This is why large-batch training behaves differently from small-batch training — the gradient noise becomes more Gaussian (less heavy-tailed) as $B$ grows.

### H.2 Autocorrelation and Power Spectral Density

For a stationary stochastic process $\{X_t\}$ with mean zero, the **autocorrelation function** is:
$$R_X(\tau) = \mathbb{E}[X_t X_{t+\tau}]$$

The **power spectral density (PSD)** is the Fourier transform of $R_X$:
$$S_X(f) = \int_{-\infty}^{\infty} R_X(\tau) e^{-2\pi if\tau}\,d\tau$$

This is the **Wiener-Khinchin theorem**: the PSD and autocorrelation are a Fourier pair.

**Interpretation of PSD:**
- $S_X(f) \geq 0$ for all $f$ (non-negative by Bochner's theorem)
- $\int S_X(f)\,df = R_X(0) = \mathbb{E}[X_t^2]$ (total power)
- $S_X(f)$ at frequency $f$ tells how much of the signal's power is concentrated near frequency $f$

**White noise:** $S_X(f) = N_0$ (constant) → $R_X(\tau) = N_0 \delta(\tau)$ (uncorrelated across all lags). White noise has equal power at all frequencies — it is the "maximally random" process.

**1/f noise (pink noise):** $S_X(f) \propto 1/f$ → found in LLM token frequency distributions (Zipf's law), financial returns, EEG signals. The 1/f spectrum is exactly at the boundary between stationary ($S_X$ integrable) and non-stationary processes.

**For AI:** The power spectral density of the hidden states in LLMs has been studied empirically. Models trained on natural language show 1/f-like spectra in their activations, consistent with the fractal/scale-free structure of language. This suggests that efficient LLM architectures should process information at multiple timescales simultaneously — exactly what multi-head attention (with different positional frequency ranges per head) achieves.

### H.3 Fourier Analysis and the Sampling Theorem

The **Shannon-Nyquist Sampling Theorem** connects continuous Fourier analysis to discrete signal representation:

**Theorem (Shannon, 1949; Nyquist, 1928):** A bandlimited signal $f$ with $\hat{f}(\xi) = 0$ for $|\xi| > W$ (bandwidth $W$) can be perfectly reconstructed from its samples $\{f(n/(2W))\}_{n \in \mathbb{Z}}$:
$$f(t) = \sum_{n=-\infty}^{\infty} f\!\left(\frac{n}{2W}\right) \operatorname{sinc}(2Wt - n)$$

**Proof sketch.** Define the sampled signal $f_s(t) = \sum_n f(n/(2W))\delta(t - n/(2W))$. Its Fourier transform is the **periodized spectrum**: $\hat{f}_s(\xi) = 2W \sum_k \hat{f}(\xi - 2kW)$. Since $f$ is bandlimited to $|\xi| \leq W$, the copies don't overlap (no aliasing). Applying an ideal low-pass filter with cutoff $W$ recovers $\hat{f}$, and inverse FT gives $f$. $\square$

**Aliasing:** If $f$ has frequency content above $W$ but is sampled at rate $2W$, the high-frequency copies in $\hat{f}_s$ overlap, corrupting the spectrum. This is **aliasing** — high-frequency components masquerade as low-frequency ones.

**For AI — Tokenization:** Tokenization is analogous to sampling. A tokenizer samples text at a rate of $\sim 0.7$ tokens/word (for byte-pair encoding). If the "semantic content" of text has structure at finer granularity than 1 word, the tokenizer introduces aliasing: some semantic distinctions are lost. This is why character-level models can capture morphological patterns that word-level models miss.

**Anti-aliasing in CNNs:** The standard strided convolution in CNNs (stride-2 for downsampling) can alias: it samples the feature map at rate $1/2$ without first applying a low-pass filter. Zhang (2019) showed that adding a blur filter before strided convolutions dramatically improves shift-equivariance. This is the CNN version of anti-aliasing.

---

## Appendix I: Implementation Details

### I.1 Numerical Computation of Fourier Coefficients

**Direct formula.** For a function given by samples $f_0, \ldots, f_{N-1}$ at equally spaced points $x_k = -\pi + 2\pi k/N$:
$$c_n \approx \frac{1}{N}\sum_{k=0}^{N-1} f_k e^{-inx_k} = \frac{1}{N}\text{DFT}(f)[n]$$

This approximates the integral $c_n = \frac{1}{2\pi}\int_{-\pi}^{\pi} f(x)e^{-inx}\,dx$ using the rectangle rule.

**Accuracy.** The approximation error is $O(1/N)$ for smooth functions and $O(1/N^k)$ for $C^k$ functions. For discontinuous functions, the error is $O(1/N)$ regardless of how fine the sampling.

**Using NumPy:**
```python
# N equispaced samples of f on [-pi, pi)
x = np.linspace(-np.pi, np.pi, N, endpoint=False)
f_samples = f(x)
# Fourier coefficients (approximate)
c = np.fft.fft(f_samples) / N
# c[n] corresponds to the n-th coefficient for n = 0, 1, ..., N-1
# Negative frequencies: c[N-n] corresponds to c_{-n}
```

**Aliasing in discrete computation.** The DFT computes only $N$ coefficients $c_0, \ldots, c_{N-1}$. For a bandlimited function with $c_n = 0$ for $|n| > N/2$, this is exact. Otherwise, the $n$-th computed coefficient is actually $c_n + c_{n+N} + c_{n-N} + \cdots$ (all aliased copies).

### I.2 Plotting Spectra Correctly

Common mistakes when plotting Fourier spectra:

1. **Frequency axis:** NumPy's FFT output has frequencies $[0, 1, \ldots, N/2-1, -N/2, \ldots, -1]/N$ in units of cycles/sample. Use `np.fft.fftfreq(N)` to get the correct axis.

2. **Centering:** Use `np.fft.fftshift` to rearrange the output so that zero frequency is in the center.

3. **One-sided vs two-sided:** For real signals, plot only $n \geq 0$ (the two-sided spectrum is symmetric). Use `np.fft.rfft` for real signals — it returns only the positive-frequency half.

4. **dB scale:** Power spectra are often plotted in dB: $10\log_{10}|c_n|^2$ or $20\log_{10}|c_n|$. This makes it easier to see components spanning many orders of magnitude.

5. **Parseval check:** Always verify $\sum |c_n|^2 \approx \frac{1}{N}\sum |f_k|^2$. If these don't match, there's a normalization error.

### I.3 RoPE Implementation Reference

```python
def rope_embed(x, positions, base=10000):
    """
    Apply RoPE to input tensor x.
    x: shape (seq_len, d_model) — query or key vectors
    positions: integer positions (seq_len,)
    Returns: rotated x, same shape
    """
    d = x.shape[-1]
    # Frequencies: theta_i = base^{-2i/d} for i = 0, ..., d/2 - 1
    i = np.arange(0, d, 2)
    freqs = base ** (-i / d)          # shape (d/2,)
    # Angles: m * theta_i
    angles = positions[:, None] * freqs[None, :]  # (seq_len, d/2)
    # Build rotation: cos(angle) and sin(angle)
    cos = np.cos(angles)   # (seq_len, d/2)
    sin = np.sin(angles)   # (seq_len, d/2)
    # Rotate pairs (x_{2i}, x_{2i+1})
    x_even = x[:, 0::2]   # shape (seq_len, d/2)
    x_odd  = x[:, 1::2]
    x_rot_even = x_even * cos - x_odd * sin
    x_rot_odd  = x_even * sin + x_odd * cos
    # Interleave back
    x_rot = np.empty_like(x)
    x_rot[:, 0::2] = x_rot_even
    x_rot[:, 1::2] = x_rot_odd
    return x_rot
```

**Verification:** For any queries $q_m$ and keys $k_n$ with relative position $r = m - n$:
```
score(m, n) = dot(rope_embed(q, [m]), rope_embed(k, [n]))
            == score(m + delta, n + delta)  # for any integer delta
```
This confirms that the attention score depends only on relative position.


---

## Appendix J: Fourier Series in Diverse Mathematical Contexts

### J.1 Fourier Series and Number Theory

The connection between Fourier analysis and number theory is profound and historically important.

**Dirichlet characters.** For a modulus $q$, a **Dirichlet character** $\chi: \mathbb{Z} \to \mathbb{C}$ is a completely multiplicative arithmetic function with period $q$. These are precisely the characters of the group $(\mathbb{Z}/q\mathbb{Z})^*$. Dirichlet's proof that there are infinitely many primes in any arithmetic progression $\{a, a+q, a+2q,\ldots\}$ with $\gcd(a,q) = 1$ uses Fourier analysis on the group $(\mathbb{Z}/q\mathbb{Z})^*$ — a finite version of Fourier series.

**The Poisson summation formula.** For a "nice" function $f$:
$$\sum_{n=-\infty}^{\infty} f(n) = \sum_{k=-\infty}^{\infty} \hat{f}(k)$$

This connects sums over integers to sums over their Fourier transforms. Applications include:
- **Jacobi theta functions:** $\theta(\tau) = \sum_{n=-\infty}^\infty e^{i\pi n^2 \tau}$ satisfies a functional equation from Poisson summation
- **Riemann zeta function:** The functional equation $\zeta(s) = 2^s\pi^{s-1}\sin(\pi s/2)\Gamma(1-s)\zeta(1-s)$ is proved via Poisson summation
- **Quadratic forms:** The number of ways to represent an integer as a sum of $k$ squares is expressed via theta functions

**For AI:** Poisson summation appears in the theory of neural network generalization. The **neural tangent kernel** at infinite width is related to the Fourier transform of the network architecture, and bounds on generalization involve the Poisson summation formula applied to the empirical Fourier spectrum of the training data.

### J.2 Fourier Series on Non-Euclidean Spaces

**Fourier analysis on groups.** The Fourier series on $[-\pi,\pi]$ is really harmonic analysis on the group $S^1 = \{e^{i\theta} : \theta \in [-\pi,\pi]\}$. The generalization to compact groups $G$ gives:
$$f(g) = \sum_{\pi \in \hat{G}} d_\pi \operatorname{tr}(\hat{f}(\pi) \pi(g))$$

where $\hat{G}$ is the set of irreducible unitary representations of $G$ and $d_\pi$ is the dimension of representation $\pi$. For $G = S^1$: all irreducible representations are 1-dimensional ($d_\pi = 1$), and we recover the ordinary Fourier series.

**For AI — Equivariant networks:** Group-equivariant neural networks (Cohen & Welling, 2016) build in symmetries (rotations, reflections) by working with group-Fourier analysis. A convolution network on $\mathbb{R}^2$ is equivariant to translations; lifting to $SE(2)$ (rigid motions) gives equivariance to rotation and translation simultaneously. The mathematical foundation is the group Fourier transform — Fourier series on a non-abelian group.

**Fourier analysis on graphs.** The graph Laplacian $L = D - A$ has eigenvectors $U = [u_1,\ldots,u_n]$ with eigenvalues $0 = \lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_n$. The **graph Fourier transform** is $\hat{f} = U^\top f$. Graph convolution (used in GCN, ChebNet — covered in [§11-04](../../11-Graph-Theory/04-Spectral-Graph-Theory/notes.md)) is multiplication in this graph Fourier domain:
$$f *_G h = U((U^\top f) \odot (U^\top h))$$

This is exactly the convolution theorem applied to graphs, replacing the standard Fourier basis $e^{inx}$ with the Laplacian eigenvectors $u_k$.

### J.3 Fourier Series and Quantum Mechanics

The mathematical structure of quantum mechanics is built on Fourier analysis.

**Position and momentum representations.** A quantum state $|\psi\rangle$ can be represented in the position basis: $\psi(x) = \langle x|\psi\rangle$ (the wavefunction), or in the momentum basis: $\tilde{\psi}(p) = \langle p|\psi\rangle$. The two representations are related by the Fourier transform:
$$\tilde{\psi}(p) = \frac{1}{\sqrt{2\pi\hbar}}\int \psi(x) e^{-ipx/\hbar}\,dx$$

The Heisenberg uncertainty principle $\Delta x \cdot \Delta p \geq \hbar/2$ is exactly the Fourier uncertainty principle $\sigma_t \cdot \sigma_\xi \geq 1/(4\pi)$ in disguise, with $p = \hbar\xi$ and $x = t$.

**For AI:** The mathematical structure of quantum mechanics (Hilbert space, operators, eigenvalues) and the mathematical structure of attention (inner products, softmax, value aggregation) are strikingly parallel. Several researchers have proposed "quantum attention" mechanisms, and the Fourier analysis connecting position and momentum representations underlies proposals for "quantum-inspired" sequence modeling.

### J.4 The Ergodic Theorem and Fourier Analysis

**Birkhoff's ergodic theorem.** For an ergodic measure-preserving transformation $T$ and $f \in L^1$:
$$\frac{1}{N}\sum_{k=0}^{N-1} f(T^k x) \to \int f\,d\mu \quad \text{a.e.}$$

The time average equals the space average.

**Connection to Fourier series.** For the rotation $T: x \mapsto x + \alpha \pmod{1}$ on the circle (irrational $\alpha$), Birkhoff's theorem gives:
$$\frac{1}{N}\sum_{k=0}^{N-1} f(x + k\alpha) \to \int_0^1 f\,dx$$

In Fourier space: $\frac{1}{N}\sum_{k=0}^{N-1} c_n[f] e^{2\pi ink\alpha} = c_n[f] \cdot \frac{1}{N}\sum_{k=0}^{N-1} e^{2\pi ink\alpha}$. For irrational $\alpha$ and $n \neq 0$: $\frac{1}{N}\sum e^{2\pi ink\alpha} \to 0$ (equidistribution) → only the $n = 0$ term survives → the time average equals $c_0[f] = \int f$.

**For AI:** The ergodic theorem underpins the mixing of training data. In language model training, a sequence of tokens is generated by a stationary stochastic process (natural language). Ergodicity says that sampling from a long enough sequence gives a representative sample of the distribution — justifying the use of random windowed crops for training, rather than requiring the full corpus at each step.

---

## Appendix K: Connections Between Sections

### K.1 Why This Section is the Foundation

All subsequent sections in this chapter build on the foundations laid here:

**→ §20-02 (Fourier Transform):** Take the period $T \to \infty$ in the Fourier series. The discrete sum $\sum_n c_n e^{inx}$ becomes the integral $\int \hat{f}(\xi)e^{2\pi i\xi x}\,d\xi$. Every property of the Fourier series has an analogue in the Fourier transform — the derivations are nearly identical. The key new feature is the continuous spectrum and the $L^2(\mathbb{R})$ setting.

**→ §20-03 (DFT and FFT):** Discretize the Fourier transform: sample $f$ at $N$ points and compute $N$ frequency coefficients. The DFT is the Fourier series applied to the sequence $(f_0, \ldots, f_{N-1})$ viewed as a periodic sequence of period $N$. The FFT is the efficient algorithm for computing it.

**→ §20-04 (Convolution Theorem):** The property $c_n[f*g] = c_n[f]\cdot c_n[g]$ (multiplication in frequency = convolution in time) is the single most important theorem in signal processing. It follows directly from the orthogonality of the Fourier basis and is proved here in the series setting; the full treatment with applications to CNNs is in §20-04.

**→ §20-05 (Wavelets):** Wavelets are a replacement for Fourier series that provides both time and frequency localization. The Haar wavelet is a piecewise constant basis — like the Fourier basis, but localized in time. The mathematical framework of multiresolution analysis is the generalization of the Fourier series framework (nested subspaces, orthonormal bases) to time-localized expansions.

### K.2 What to Read Next

After this section, you have two natural paths:

**Path A — Depth first:** Continue with §20-02 (Fourier Transform) to see how the theory extends to aperiodic signals and the full real line. This path eventually leads to the uncertainty principle and the deeper function space theory.

**Path B — Applications first:** Jump to §20-04 (Convolution Theorem) to see immediately how Fourier series powers CNNs and sequence models. The convolution theorem proof requires the Fourier transform (§20-02), so read that first.

**For the AI-focused reader:** The most direct path from this section to ML applications is:
1. §20-01 (this section) — Fourier series fundamentals
2. §20-03 (DFT/FFT) — computational implementation
3. §20-04 (Convolution Theorem) — CNNs and sequence models
4. §20-02 (Fourier Transform) — theory for spectral methods
5. §20-05 (Wavelets) — multi-scale models and compression


---

## Appendix L: The Fourier Series in Numerical Practice

### L.1 Accuracy Benchmarks: Theoretical vs. Observed

When implementing Fourier series approximations numerically, understanding the gap between theoretical and observed accuracy is essential.

**Experiment: Square wave approximation accuracy.**

For the square wave $f(x) = \text{sgn}(x)$ with $N$-term Fourier series $S_N f$:
- **Theoretical $L^2$ error:** $\lVert f - S_N f \rVert_2^2 = \sum_{k>N/2, k\text{ odd}} \frac{16}{\pi^2 k^2} \sim \frac{8}{\pi^2 N}$
- **Observed:** Plot $\lVert f - S_N f \rVert_2^2$ vs $N$ on a log-log scale. Slope should be $-1$, confirming $O(1/N)$ decay.
- **At jump:** $|S_N f(0.01) - 1| \approx 0.179$ regardless of $N$ (Gibbs)

**Experiment: Triangle wave approximation accuracy.**

For $f(x) = |x|$ with $|c_n| = 4/(\pi^2 n^2)$ (odd $n$):
- **Theoretical $L^2$ error:** $\lVert f - S_N f \rVert_2^2 = \sum_{k>N/2, k\text{ odd}} \frac{16}{\pi^4 k^4} \sim \frac{4}{3\pi^4 N^3}$
- **Observed:** slope $\approx -3$ on log-log plot, confirming $O(1/N^3)$ decay

**Experiment: Smooth function accuracy.**

For $f(x) = e^{\cos x}$ (analytic, $2\pi$-periodic):
- **Theoretical:** $|c_n| \leq C e^{-n}$ → $\lVert f - S_N f \rVert_2 \leq C' e^{-N}$
- **Observed:** semi-log plot of $\lVert f - S_N f \rVert_2$ vs $N$ is linear (exponential decay)
- Essentially machine-precision accuracy for $N \geq 30$

**Takeaway:** The rule of thumb is "smooth = exponentially good, nonsmooth = algebraically poor." For discontinuous signals, truncated Fourier series are often a poor choice — JPEG-style block transforms or wavelets are better.

### L.2 The Fast Fourier Transform Preview

The DFT (fully treated in §20-03) computes exactly the same Fourier coefficients as the Fourier series formula, but for a finite sequence and in $O(N \log N)$ time.

**The connection:** For samples $f_k = f(x_k)$ at $N$ equispaced points $x_k = -\pi + 2\pi k/N$:
$$\text{DFT}(f)[n] = \sum_{k=0}^{N-1} f_k e^{-2\pi ikn/N}$$

Comparing with $c_n = \frac{1}{2\pi}\int_{-\pi}^{\pi} f(x)e^{-inx}\,dx \approx \frac{1}{N}\sum_{k=0}^{N-1} f_k e^{-inx_k}$ (rectangle rule with $\Delta x = 2\pi/N$):

$$c_n \approx \frac{1}{N} \cdot \text{DFT}(f)[n]$$

So the DFT is just the Fourier series computation via a rectangle-rule quadrature, scaled by $N$.

**The FFT (Cooley-Tukey, 1965)** computes all $N$ DFT values in $O(N \log N)$ operations instead of the naïve $O(N^2)$. For $N = 10^6$: naïve DFT requires $10^{12}$ operations; FFT requires $\approx 2 \times 10^7$ operations — a factor of 50,000 speedup.

This speedup is what made digital signal processing, audio compression, and modern AI practically feasible. The STFT and mel-spectrogram preprocessing in Whisper, the frequency-domain convolutions in FNet, and the state-space model computations in S4/Mamba all rely on this speedup. Full treatment: [§20-03 Discrete Fourier Transform and FFT](../03-Discrete-Fourier-Transform-and-FFT/notes.md).

### L.3 Fourier Series vs. Polynomial Approximation

Both Fourier series and polynomial approximation (Taylor, Chebyshev, Legendre) approximate functions by basis expansions. Understanding when to use each is important.

| Aspect | Fourier Series | Polynomial (Chebyshev) |
|--------|---------------|----------------------|
| Best for | Periodic functions, global frequency analysis | Non-periodic functions on $[a,b]$ |
| Convergence for smooth $f$ | Exponential (analytic), algebraic (smooth) | Exponential (analytic), algebraic (smooth) |
| At discontinuities | Gibbs phenomenon (9% overshoot) | Runge phenomenon near endpoints |
| Complexity for $N$ terms | $O(N \log N)$ via FFT | $O(N^2)$ (or $O(N\log N)$ for Chebyshev via DCT) |
| AI applications | RoPE, spectrograms, convolutions | Polynomial activations, rational networks |
| Sparse representation | Works for bandlimited signals | Works for smooth functions |

**Key insight:** For **periodic** and **global** structure, use Fourier series. For **localized** and **non-periodic** structure, use wavelets (§20-05) or local polynomial bases. The choice of approximation basis should match the intrinsic structure of the function.

### L.4 When Fourier Series Fail: Pathological Examples

Understanding the failure modes of Fourier series deepens intuition.

**Example 1 — Du Bois-Reymond's construction (1876):** There exists a continuous $2\pi$-periodic function $f$ whose Fourier series diverges at $x = 0$. The construction uses the slow growth of $\lVert D_N \rVert_{L^1} \sim \frac{4}{\pi^2}\ln N$: build $f$ to make $|S_N f(0)|$ grow like $\ln N$.

**Example 2 — Kolmogorov's function (1923):** There exists an $L^1$ function whose Fourier series diverges everywhere. This shows that $L^1$ is too weak for pointwise Fourier convergence.

**Example 3 — Weierstrass nowhere-differentiable function:**
$$W(x) = \sum_{n=0}^\infty a^n \cos(b^n \pi x), \quad 0 < a < 1, \; ab > 1 + \frac{3\pi}{2}$$

This is a Fourier series (cosine series) that converges uniformly (geometric series bounds), but the sum is nowhere differentiable. The coefficients $a^n$ decay geometrically but the frequencies $b^n$ grow geometrically — the function has "fractal" structure with self-similar behavior at all scales.

**For AI:** Nowhere-differentiable functions are the worst case for neural network optimization. The loss landscape of a neural network restricted to certain parameter subspaces can exhibit Weierstrass-like behavior — highly oscillatory, with no gradient information about the global minimum. This motivates smooth parameterizations (Adam's momentum, gradient clipping) and understanding of the loss landscape geometry.

---

## Appendix M: Quick Reference

### M.1 Key Formulas at a Glance

$$\boxed{c_n = \frac{1}{2\pi}\int_{-\pi}^{\pi} f(x)e^{-inx}\,dx, \quad f(x) = \sum_{n=-\infty}^\infty c_n e^{inx}}$$

$$\boxed{a_n = \frac{1}{\pi}\int_{-\pi}^{\pi} f(x)\cos(nx)\,dx, \quad b_n = \frac{1}{\pi}\int_{-\pi}^{\pi} f(x)\sin(nx)\,dx}$$

$$\boxed{\text{Parseval: } \sum_{n=-\infty}^\infty |c_n|^2 = \frac{1}{2\pi}\int_{-\pi}^\pi |f(x)|^2\,dx}$$

$$\boxed{\text{Shift: } c_n[f(\cdot - x_0)] = e^{-inx_0}c_n[f], \quad \text{Deriv: } c_n[f'] = in\cdot c_n[f]}$$

$$\boxed{S_N f(x) = \frac{1}{2\pi}\int_{-\pi}^{\pi} f(t)D_N(x-t)\,dt, \quad D_N(x) = \frac{\sin((N+\frac{1}{2})x)}{\sin(x/2)}}$$

$$\boxed{\text{RoPE: }(q_{2i}+iq_{2i+1})e^{im\theta_i}, \;\theta_i = 10000^{-2i/d}}$$

$$\boxed{\text{Score}(m,n) = \operatorname{Re}\!\left[\sum_i (q_{2i}+iq_{2i+1})\overline{(k_{2i}+ik_{2i+1})} e^{i(m-n)\theta_i}\right]}$$

### M.2 Decision Tree: Which Tool to Use

```
Is your signal periodic?
├── YES → Fourier series (§20-01)
│   ├── Need to compute? → DFT/FFT (§20-03)
│   └── Need to convolve? → Convolution theorem (§20-04)
└── NO → Fourier transform (§20-02)
    ├── Bandlimited? → Sampling theorem
    ├── Need time localization? → Wavelets (§20-05)
    └── Aperiodic transient? → STFT (§20-03)

Is your signal discrete?
├── YES → DFT/FFT (§20-03)
└── NO → Fourier transform (§20-02)

Do you need time AND frequency localization?
└── YES → Wavelets (§20-05)
```


---

## Appendix N: Fourier Series and Modern Transformers — Extended Analysis

### N.1 YaRN and NTK-Aware Frequency Scaling

The original RoPE with base $10000$ runs out of positional expressiveness beyond $\sim 4096$ tokens. Several methods have been proposed to extend it:

**Linear Position Interpolation (PI) — Chen et al., 2023.**
Scale all positions by $s = L_{\text{orig}}/L_{\text{new}}$: position $m$ becomes $m/s$. This compresses the angles to keep $m\theta_i/s$ within the trained range $[0, 2\pi]$. Equivalently, reduce all frequencies: $\theta_i \to \theta_i/s$.

**Issue with linear interpolation:** The lowest-frequency components (large $\theta_i$, which change slowly) are barely affected, but the highest-frequency components (small $\theta_i$, which rotate rapidly) are compressed most aggressively. High-frequency positional information (distinguishing adjacent tokens) degrades.

**NTK-RoPE — Bloc97, 2023.** Scale the **base** rather than the positions: $10000 \to 10000^{s^{d/(d-2)}}$. This modifies low-frequency components more than high-frequency ones — a non-uniform scaling derived from the NTK analysis.

**YaRN — Peng et al., 2023.** Combines interpolation for middle frequencies with no interpolation for high frequencies (which extrapolate) and linear interpolation for low frequencies. The frequency ranges are: (i) $n < \alpha d/2$: unchanged; (ii) $\alpha d/2 \leq n < \beta d/2$: linear interpolation; (iii) $n \geq \beta d/2$: extrapolation. The parameters $\alpha, \beta$ are tuned per model. YaRN achieves strong performance at 128K context with only 0.1% of the parameters fine-tuned.

**The Fourier interpretation:** All these methods are attempting to find the right **frequency allocation** — which Fourier basis vectors should be responsible for which positional scales. The ideal is to have each frequency range $[1/(2\theta_i), 1/(2\theta_{i+1})]$ cover roughly equal logarithmic bandwidth, so no scale is under- or over-represented.

### N.2 Frequency Analysis of Layer Norms

**LayerNorm as a high-pass filter.** LayerNorm subtracts the mean and divides by standard deviation:
$$\text{LN}(\mathbf{x}) = \frac{\mathbf{x} - \bar{\mathbf{x}}}{\sigma_{\mathbf{x}}}$$

The mean subtraction is a **DC removal** in signal processing terms: it removes the $n=0$ (constant) Fourier component from the sequence. The division by $\sigma_\mathbf{x}$ is a gain normalization.

**Implication:** After LayerNorm, the token representation has zero mean. This concentrates the representational capacity in the non-DC Fourier components, reinforcing the tendency of transformers to represent relative (not absolute) positional information.

**RMSNorm** (used in LLaMA) only divides by the RMS (root mean square), without mean subtraction. It does not remove the DC component — preserving more absolute magnitude information. The trade-off: slightly less stable training vs. richer representation.

### N.3 Fourier Analysis of Embedding Matrices

**What does the Fourier transform of a token embedding matrix reveal?**

Let $E \in \mathbb{R}^{V \times d}$ be the embedding matrix for vocabulary size $V$ and dimension $d$. For a fixed dimension $j$, the vector $e_j = E_{:,j} \in \mathbb{R}^V$ gives the $j$-th coordinate of all token embeddings.

If we sort tokens by frequency rank (most common first) and look at $e_j$ as a function of token rank, we can ask: what is its Fourier transform?

**Empirical finding (2024):** The embedding vectors of tokens sorted by frequency show 1/f-like spectra in dimension $j$ vs. frequency rank — consistent with the Zipf distribution of token frequencies. This suggests that the embedding space has a natural "frequency ordering" that mirrors the frequency distribution of the training corpus.

**For interpretability:** The Fourier components of the embedding matrix at different "token frequencies" correspond to different semantic clusters. Low-frequency components (affecting the most common tokens) encode universal semantic primitives (noun/verb, positive/negative). High-frequency components encode rare, specific meanings.

### N.4 The Fourier Perspective on Attention Sinks

**The "attention sink" phenomenon** (Xiao et al., 2023): In all transformer models, the first token (often `<BOS>` or the period `.`) receives disproportionate attention from most heads, even when semantically irrelevant.

**Fourier explanation:** The attention sink is the **DC component** (zero frequency) of the attention pattern. In signal processing, a low-pass filter preserves the DC component. If attention heads are implicitly low-pass filtering the sequence, they must "send" the removed energy somewhere — and the first token becomes the "drain" for this energy.

**StreamingLLM (Xiao et al., 2023):** By always keeping the first 4 tokens in the KV cache (the "attention sinks"), models can be extended to arbitrarily long sequences without performance degradation. This is a practical application of the Fourier insight: preserve the DC reference point.

### N.5 Complete Worked Example: Sinusoidal PE Computation

Let $d = 8$ (small for illustration), positions $\{0, 1, 2, 3, 4\}$.

**Frequencies:** $\omega_i = 10000^{-2i/8}$ for $i = 0, 1, 2, 3$:
- $\omega_0 = 1.0000$
- $\omega_1 = 10000^{-1/4} \approx 0.1000$
- $\omega_2 = 10000^{-1/2} \approx 0.0100$
- $\omega_3 = 10000^{-3/4} \approx 0.0010$

**Positional encoding matrix** $\text{PE} \in \mathbb{R}^{5 \times 8}$:

| pos | sin(pos·1.0) | cos(pos·1.0) | sin(pos·0.1) | cos(pos·0.1) | sin(pos·0.01) | cos(pos·0.01) | sin(pos·0.001) | cos(pos·0.001) |
|-----|-------------|-------------|--------------|-------------|---------------|--------------|----------------|---------------|
| 0   | 0.000       | 1.000       | 0.000        | 1.000       | 0.000         | 1.000        | 0.000          | 1.000         |
| 1   | 0.841       | 0.540       | 0.100        | 0.995       | 0.010         | 1.000        | 0.001          | 1.000         |
| 2   | 0.909       | -0.416      | 0.199        | 0.980       | 0.020         | 1.000        | 0.002          | 1.000         |
| 3   | 0.141       | -0.990      | 0.296        | 0.955       | 0.030         | 1.000        | 0.003          | 1.000         |
| 4   | -0.757      | -0.654      | 0.389        | 0.921       | 0.040         | 1.000        | 0.004          | 1.000         |

**Observations:**
- **Dimension 0-1** ($\omega_0 = 1$): Oscillates rapidly — completes a full cycle in $2\pi \approx 6.28$ positions. Distinguishes adjacent tokens.
- **Dimension 2-3** ($\omega_1 = 0.1$): Oscillates at 1/10th the rate. Distinguishes tokens up to ~63 apart.
- **Dimension 4-5** ($\omega_2 = 0.01$): Very slow oscillation. Distinguishes tokens up to ~628 apart.
- **Dimension 6-7** ($\omega_3 = 0.001$): Extremely slow. Distinguishes tokens up to ~6283 apart.

This is the Fourier series multi-resolution strategy: use harmonics at multiple scales to encode position information at all scales simultaneously. Each dimension pair $(2i, 2i+1)$ is a Fourier basis function at a different frequency — the entire positional encoding is a **multi-frequency Fourier expansion** of the position index.


---

## Appendix O: Further Reading and References

### O.1 Primary References

**Textbooks (Classical):**
- **Körner, T.W.** (1988). *Fourier Analysis*. Cambridge University Press. — The most readable rigorous treatment; includes Gibbs phenomenon, convergence theory, and applications to number theory.
- **Katznelson, Y.** (2004). *An Introduction to Harmonic Analysis*, 3rd ed. Cambridge. — Definitive graduate reference.
- **Stein, E. & Weiss, G.** (1971). *Introduction to Fourier Analysis on Euclidean Spaces*. Princeton. — The multidimensional extension.
- **Tolstov, G.P.** (1962). *Fourier Series* (Silverman translation). Dover. — Inexpensive, detailed, complete derivations of all standard results.

**Textbooks (Applied):**
- **Oppenheim, A. & Willsky, A.** (1997). *Signals and Systems*, 2nd ed. Prentice Hall. — Standard engineering reference for Fourier methods in signal processing.
- **Bracewell, R.N.** (2000). *The Fourier Transform and Its Applications*, 3rd ed. McGraw-Hill. — Intuitive engineering treatment with excellent visualizations.

### O.2 Key ML Papers Using Fourier Methods

| Paper | Year | Contribution | Fourier connection |
|-------|------|-------------|-------------------|
| Vaswani et al., "Attention Is All You Need" | 2017 | Transformer; sinusoidal PE | Fourier basis for position |
| Rahimi & Recht, "Random Features for Large-Scale Kernel Machines" | 2007 | Random Fourier features | Bochner's theorem |
| Rahaman et al., "On the Spectral Bias of Neural Networks" | 2019 | Spectral bias phenomenon | NTK Fourier decomposition |
| Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" | 2021 | RoPE | Complex Fourier rotation |
| Lee-Thorp et al., "FNet: Mixing Tokens with Fourier Transforms" | 2021 | FNet | DFT replaces attention |
| Sitzmann et al., "Implicit Neural Representations with Periodic Activations" | 2020 | SIREN | Fourier basis activations |
| Tancik et al., "Fourier Features Let Networks Learn High Frequency Functions" | 2020 | Fourier feature encoding | NeRF; overcomes spectral bias |
| Gu et al., "Efficiently Modeling Long Sequences with Structured State Spaces" | 2021 | S4 model | Frequency domain parameterization |
| Peng et al., "YaRN: Efficient Context Window Extension of LLMs" | 2023 | Extended context | RoPE frequency scaling |
| Xiao et al., "Efficient Streaming Language Models with Attention Sinks" | 2023 | StreamingLLM | DC component preservation |

### O.3 Connections to Other Sections in This Curriculum

| Topic | Where to find it | Connection to Fourier Series |
|-------|-----------------|------------------------------|
| $L^2$ Hilbert spaces, completeness | [§12-02 Hilbert Spaces](../../12-Functional-Analysis/02-Hilbert-Spaces/notes.md) | The abstract framework for Fourier series |
| Inner products on function spaces | [§12-01 Normed Spaces](../../12-Functional-Analysis/01-Normed-Spaces/notes.md) | The $\langle f,g \rangle$ that defines Fourier coefficients |
| Kernel methods, RKHS | [§12-03 Kernel Methods](../../12-Functional-Analysis/03-Kernel-Methods/notes.md) | Bochner's theorem; RFFs |
| Complex numbers, series | [§01 Mathematical Foundations](../../01-Mathematical-Foundations/README.md) | $e^{inx}$, Euler's formula |
| Gradient descent, loss landscape | [§08 Optimization](../../08-Optimization/README.md) | NTK, spectral bias |
| Spectral graph convolution | [§11-04 Spectral Graph Theory](../../11-Graph-Theory/04-Spectral-Graph-Theory/notes.md) | Graph Fourier transform |
| Probability characteristic functions | [§06 Probability Theory](../../06-Probability-Theory/README.md) | Bochner, CLT via characteristic functions |


---

## Appendix P: Common Error Patterns in Code

When implementing Fourier series and Fourier-based methods in Python, these are the most frequent bugs:

### P.1 Normalization Factor Confusion

**Bug:** Using `np.fft.fft(x)` without dividing by `N`.

```python
# WRONG: FFT includes the N normalization in a different convention
c_wrong = np.fft.fft(x)  # These are N * c_n, not c_n!

# CORRECT: divide by N to get Fourier coefficients
c_correct = np.fft.fft(x) / N
```

**Why:** NumPy's FFT convention is $X[k] = \sum_{n=0}^{N-1} x[n] e^{-2\pi ink/N}$ (no $1/N$ factor). The mathematical Fourier series coefficient is $c_n = \frac{1}{N}\sum_{n=0}^{N-1} x[n] e^{-2\pi ink/N}$.

### P.2 Frequency Axis Misidentification

```python
# WRONG: assuming output index k corresponds to frequency k
freqs = np.arange(N)  # Wrong!

# CORRECT: use fftfreq
freqs = np.fft.fftfreq(N, d=1.0/sample_rate)  # In Hz
# or for [-pi, pi] interval:
freqs = np.fft.fftfreq(N) * N  # integer wavenumbers -N/2 to N/2
```

**Why:** NumPy FFT output has frequencies $[0, 1, \ldots, N/2-1, -N/2, \ldots, -1]$ cycles/sample, not $[0, 1, \ldots, N-1]$.

### P.3 Forgetting to Handle Real FFT Symmetry

For real inputs, $c_{-n} = \overline{c_n}$ (Hermitian symmetry). The positive-frequency half is sufficient:

```python
# For real input, use rfft (returns only positive frequencies):
c = np.fft.rfft(x) / N     # Length N//2 + 1
freqs = np.fft.rfftfreq(N) # Corresponding frequencies

# To recover the full spectrum, remember to double all |c_n|^2 for n > 0:
power_onesided = np.abs(c)**2
power_onesided[1:-1] *= 2   # Double all except DC and Nyquist
```

### P.4 Aliasing from Insufficient Sampling

```python
# DANGER: if signal has frequency content above Nyquist
t = np.linspace(0, 1, 100)     # 100 samples per second
x = np.sin(2 * np.pi * 80 * t)  # 80 Hz signal

# Nyquist frequency = 50 Hz; 80 Hz is ABOVE Nyquist
# The DFT will show this as a false 20 Hz component (80 - 100/2 = 30 ≠ 20... wait)
# Actually: 80 Hz aliases to 100 - 80 = 20 Hz.
# ALWAYS check: max frequency in signal < sample_rate / 2
```

### P.5 Misinterpreting Parseval in Discrete vs. Continuous Setting

**Continuous Parseval:** $\frac{1}{2\pi}\int_{-\pi}^\pi |f|^2 = \sum_n |c_n|^2$

**Discrete Parseval (DFT):** $\frac{1}{N}\sum_{k=0}^{N-1} |x[k]|^2 = \frac{1}{N^2}\sum_{n=0}^{N-1} |X[n]|^2$, where $X = \text{DFT}(x)$.

```python
# Check Parseval numerically:
x = np.random.randn(N)
X = np.fft.fft(x)
lhs = np.mean(np.abs(x)**2)           # (1/N) * sum |x[k]|^2
rhs = np.mean(np.abs(X/N)**2)         # (1/N) * sum |c_n|^2
assert np.isclose(lhs, rhs), f"Parseval failed: {lhs} != {rhs}"
```

---

*This completes the §20-01 Fourier Series reference notes. Continue with [§20-02 Fourier Transform →](../02-Fourier-Transform/notes.md) to see how the period-$T$ Fourier series becomes the continuous Fourier transform as $T \to \infty$.*


---

## Appendix Q: Historical Vignettes

### Q.1 Fourier's Audacity and the Reception of His Theory

Joseph Fourier (1768–1830) was not primarily a mathematician. He was a physicist and administrator who, as Napoleon's prefect of Isère, oversaw the draining of malarial swamps and the construction of roads. His mathematical masterpiece grew from a practical problem: how does heat conduct through a solid body?

When Fourier presented his theory to the Paris Academy in 1807, the reaction was hostile. The paper was rejected — or rather, shelved — on the grounds that the claim "every function can be represented as a trigonometric series" was both wrong (they thought) and insufficiently rigorous (it was). The judges included Laplace, Lagrange, and Monge — the greatest mathematicians of the age.

Lagrange's objection was mathematical: a sum of smooth functions (sines and cosines) cannot converge to a discontinuous function. This was a perfectly reasonable objection in 1807, before the modern theory of pointwise vs. $L^2$ convergence had been developed. Fourier's answer — essentially, "but it does!" — was more right than Lagrange, but not for the reasons Fourier gave.

Fourier persisted. In 1822, he published his masterpiece *Théorie analytique de la chaleur*, which remains one of the great works of mathematical physics. His key equations — the heat equation, the Fourier series, and the Fourier integral — were all present, though not rigorously proved.

The rigorous proof came from Dirichlet in 1829. The controversy about pointwise convergence of the Fourier series of a continuous function was not finally resolved until Carleson in 1966 — 159 years after Fourier's original claim. Mathematics sometimes takes time to catch up with correct intuitions.

### Q.2 Dirichlet's Proof: The Birth of Rigorous Analysis

Peter Gustav Lejeune Dirichlet (1805–1859) was a student of Fourier's. His 1829 paper giving the first rigorous proof of pointwise convergence is often cited as one of the founding documents of modern analysis — not just because of the result, but because of the method.

Dirichlet's proof introduced two innovations that transformed mathematics:

1. **The modern definition of function.** Before Dirichlet, a "function" meant something expressible by a formula. Dirichlet explicitly worked with functions that could behave differently on different parts of their domain, introducing what we now recognize as the modern concept. His paper is one of the first places where the contemporary idea of function (as any rule assigning outputs to inputs) appears clearly.

2. **Careful convergence arguments.** Dirichlet's proof carefully tracked when limits could be exchanged and when they could not — a concern that Cauchy had introduced but that Dirichlet applied systematically to Fourier series. This careful accounting of convergence, rather than treating series as formal objects, is the hallmark of 19th-century rigor.

In this sense, Fourier series didn't just give us a useful computational tool. They provoked the development of the rigorous analysis that underlies all of modern mathematics and machine learning theory.

### Q.3 Carleson's Theorem: The 159-Year-Old Problem

For 120 years after Dirichlet's theorem (1829), the following question remained open:

**Does the Fourier series of every continuous function converge pointwise everywhere?**

- 1873: Du Bois-Reymond showed NO for continuous functions at one point
- 1923: Kolmogorov showed NO for $L^1$ functions everywhere (!)
- But for the natural space $L^2$...?

In 1966, Lennart Carleson proved: **YES, the Fourier series of every $L^2$ function converges pointwise almost everywhere.** This was one of the most celebrated theorems in 20th-century analysis. Carleson received the Abel Prize in 2006 — the only prize given specifically for this theorem.

The proof uses a remarkable decomposition of the Dirichlet kernel into "tiles" in the time-frequency plane, combined with a sophisticated stopping-time argument. It is considered one of the most difficult proofs in all of harmonic analysis — spanning 30+ pages of dense mathematics.

For our purposes, the practical upshot is: every $L^2$ function (which includes every function we encounter in ML) has its Fourier series converging pointwise almost everywhere. The theoretical pathologies of convergence failure are measure-zero exceptions.


---

## Appendix R: Self-Assessment Checklist

Use this checklist to verify understanding before moving to §20-02.

### R.1 Computational Mastery

- [ ] Can you compute the Fourier coefficients of the square wave, sawtooth, and triangle wave from scratch without looking them up?
- [ ] Can you use the differentiation theorem ($c_n[f'] = in\,c_n[f]$) to derive the triangle wave coefficients from the square wave coefficients?
- [ ] Can you apply Parseval's identity to evaluate $\sum_{n=1}^\infty 1/n^4$ (via the Fourier series of $x^2$)?
- [ ] Can you identify the decay rate of Fourier coefficients ($O(1/n)$, $O(1/n^2)$, exponential) from a description of the function?
- [ ] Can you write the partial sum $S_N f$ as a convolution with the Dirichlet kernel?

### R.2 Theoretical Understanding

- [ ] Can you state Dirichlet's theorem and identify what happens at a jump discontinuity?
- [ ] Can you explain the Gibbs phenomenon and why it persists at any truncation order?
- [ ] Can you state Parseval's identity and explain its physical meaning (energy conservation)?
- [ ] Can you explain why $L^2$ convergence is the "correct" convergence for Fourier series?
- [ ] Can you explain why the Fejér kernel (Cesàro means) fixes the Gibbs phenomenon while the Dirichlet kernel doesn't?
- [ ] Can you explain the Riemann-Lebesgue lemma and its consequence: smooth signals have decaying Fourier coefficients?

### R.3 AI Connections

- [ ] Can you derive the sinusoidal PE formula from the Fourier basis perspective?
- [ ] Can you explain RoPE in terms of complex Fourier multiplication and the shift theorem?
- [ ] Can you explain the spectral bias phenomenon and its consequences for neural network training?
- [ ] Can you explain why Random Fourier Features approximate kernel methods and how Bochner's theorem is involved?
- [ ] Can you explain the attention sink phenomenon using Fourier analysis (DC component)?

### R.4 Connections to Other Sections

- [ ] Do you understand that the Fourier Transform (§20-02) is the $T \to \infty$ limit of the Fourier series?
- [ ] Do you understand that the DFT (§20-03) is the Fourier series for a finite discrete sequence?
- [ ] Do you understand that the convolution theorem (§20-04) is a consequence of Fourier series orthogonality ($c_n[f*g] = c_n[f]\cdot c_n[g]$)?
- [ ] Do you understand that wavelets (§20-05) overcome the time-localization failure of Fourier series?

If you cannot answer YES to all items in R.1 and R.2, and at least 4 items in R.3 and R.4, review the corresponding sections before proceeding.

---

*End of §20-01 Fourier Series*

[← Back to Fourier Analysis](../README.md) | [Next: Fourier Transform →](../02-Fourier-Transform/notes.md)

---

## Appendix S: Summary of Key Results

### S.1 The Core Theorems

**Theorem 1 (Dirichlet, 1829).** If $f$ is $2\pi$-periodic and piecewise smooth, then for every $x$:
$$\sum_{n=-N}^N c_n e^{inx} \;\to\; \frac{f(x^+) + f(x^-)}{2} \quad \text{as } N \to \infty.$$

**Theorem 2 ($L^2$ convergence).** For every $f \in L^2[-\pi,\pi]$:
$$\left\lVert f - \sum_{|n|\leq N} c_n e^{inx} \right\rVert_{L^2} \to 0 \quad \text{as } N \to \infty.$$

**Theorem 3 (Parseval–Bessel).** For $f \in L^2[-\pi,\pi]$:
$$\sum_{n=-\infty}^\infty |c_n|^2 = \frac{1}{2\pi}\int_{-\pi}^\pi |f(x)|^2 \, dx.$$

**Theorem 4 (Fejér, 1900).** For every $f \in C[-\pi,\pi]$ (continuous and $2\pi$-periodic):
$$\frac{1}{N+1}\sum_{k=0}^N S_k f \to f \quad \text{uniformly as } N \to \infty.$$

**Theorem 5 (Riemann–Lebesgue).** For $f \in L^1[-\pi,\pi]$:
$$c_n[f] \to 0 \quad \text{as } |n| \to \infty.$$

**Theorem 6 (Carleson, 1966).** For every $f \in L^2[-\pi,\pi]$:
$$\sum_{|n|\leq N} c_n e^{inx} \to f(x) \quad \text{for almost every } x \in [-\pi,\pi].$$

### S.2 The Most Important Formula

Among all the formulas in this section, one stands out as the most consequential for both mathematics and AI:

$$\boxed{c_n[f] = \frac{1}{2\pi}\int_{-\pi}^{\pi} f(x)\, e^{-inx}\, dx \qquad \longleftrightarrow \qquad f(x) = \sum_{n=-\infty}^{\infty} c_n\, e^{inx}}$$

Every other result in this section — convergence theorems, Parseval, the shift property, RoPE, spectral bias — is a consequence or application of this pair.

