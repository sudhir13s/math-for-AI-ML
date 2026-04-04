[← Back to Numerical Methods](../README.md) | [Next: Numerical Integration →](../05-Numerical-Integration/notes.md)

---

# Interpolation and Approximation

> *"The art of interpolation is the art of making the most of what you know to say something useful about what you don't."*  
> — Attributed to Carl Friedrich Gauss

## Overview

Given a set of data points, **interpolation** constructs a function that passes exactly through every point, while **approximation** finds the "best" function from some class that fits the data in a least-squares or minimax sense. These are among the oldest problems in numerical mathematics, yet they remain central to modern AI: positional encodings in transformers use sinusoidal interpolation, kernel methods are approximation schemes, and neural networks themselves are universal approximators whose expressive power is quantified by approximation theory.

The critical insight is that the *choice of basis* determines everything. Monomials ($1, x, x^2, \ldots$) are algebraically simple but numerically catastrophic (Vandermonde matrices are notoriously ill-conditioned). Chebyshev polynomials, derived from trigonometry, minimize the maximal interpolation error among all polynomial interpolants — they are the basis that tames Runge's phenomenon. Splines trade global accuracy for local control, achieving smooth, stable interpolation at the cost of piecewise complexity. Each basis encodes a different prior about the function being approximated.

This section covers polynomial interpolation, Chebyshev theory, splines, least-squares approximation, Fourier approximation, and radial basis functions, with explicit connections to ML applications including kernel methods, positional encodings, and neural function approximation.

## Prerequisites

- Real analysis: continuity, differentiability, Taylor's theorem (§01-Mathematical-Foundations)
- Linear algebra: matrix factorization, least squares, condition numbers (§02-Linear-Algebra-Basics, §10-02)
- Floating-point arithmetic: rounding error, numerical stability (§10-01)
- Norms and inner products (§02-Linear-Algebra-Basics)

## Companion Notebooks

| Notebook | Description |
|---|---|
| [theory.ipynb](theory.ipynb) | Interactive derivations: Lagrange interpolation, Chebyshev nodes, spline construction, least-squares fitting, Fourier series |
| [exercises.ipynb](exercises.ipynb) | 8 graded exercises from Vandermonde conditioning to neural tangent kernel approximation |

## Learning Objectives

After completing this section, you will:

- Construct polynomial interpolants using Lagrange and Newton divided-difference forms
- Explain Runge's phenomenon and why equispaced nodes are dangerous for high-degree interpolation
- Derive Chebyshev nodes as the optimal interpolation points minimizing the node polynomial
- Implement natural and clamped cubic splines by solving a tridiagonal linear system
- Apply least-squares polynomial fitting via QR decomposition and the normal equations
- Compute discrete Fourier series and use FFT for fast polynomial evaluation
- Understand radial basis function (RBF) interpolation and its connection to kernel methods
- Connect approximation theory to neural network universality and positional encodings in transformers

---

## Table of Contents

- [1. Intuition and Motivation](#1-intuition-and-motivation)
  - [1.1 The Interpolation Problem](#11-the-interpolation-problem)
  - [1.2 Why Approximation Matters for AI](#12-why-approximation-matters-for-ai)
  - [1.3 Historical Timeline](#13-historical-timeline)
- [2. Polynomial Interpolation](#2-polynomial-interpolation)
  - [2.1 Lagrange Form](#21-lagrange-form)
  - [2.2 Newton Divided Differences](#22-newton-divided-differences)
  - [2.3 The Interpolation Error Theorem](#23-the-interpolation-error-theorem)
  - [2.4 Runge's Phenomenon](#24-runges-phenomenon)
- [3. Chebyshev Polynomials and Optimal Nodes](#3-chebyshev-polynomials-and-optimal-nodes)
  - [3.1 Chebyshev Polynomials](#31-chebyshev-polynomials)
  - [3.2 Chebyshev Nodes](#32-chebyshev-nodes)
  - [3.3 Minimax Approximation](#33-minimax-approximation)
  - [3.4 Clenshaw-Curtis Quadrature Connection](#34-clenshaw-curtis-quadrature-connection)
- [4. Spline Interpolation](#4-spline-interpolation)
  - [4.1 Piecewise Polynomial Interpolation](#41-piecewise-polynomial-interpolation)
  - [4.2 Cubic Splines: Derivation and Construction](#42-cubic-splines-derivation-and-construction)
  - [4.3 B-Splines](#43-b-splines)
  - [4.4 Tension Splines and Shape Preservation](#44-tension-splines-and-shape-preservation)
- [5. Least-Squares Approximation](#5-least-squares-approximation)
  - [5.1 Polynomial Least Squares](#51-polynomial-least-squares)
  - [5.2 Orthogonal Polynomials and Gram-Schmidt](#52-orthogonal-polynomials-and-gram-schmidt)
  - [5.3 Minimax vs Least-Squares](#53-minimax-vs-least-squares)
- [6. Fourier Approximation](#6-fourier-approximation)
  - [6.1 Trigonometric Interpolation](#61-trigonometric-interpolation)
  - [6.2 Discrete Fourier Transform](#62-discrete-fourier-transform)
  - [6.3 Fast Fourier Transform](#63-fast-fourier-transform)
  - [6.4 Aliasing and the Nyquist Theorem](#64-aliasing-and-the-nyquist-theorem)
- [7. Radial Basis Functions and Kernel Methods](#7-radial-basis-functions-and-kernel-methods)
  - [7.1 RBF Interpolation](#71-rbf-interpolation)
  - [7.2 Connection to Reproducing Kernel Hilbert Spaces](#72-connection-to-reproducing-kernel-hilbert-spaces)
  - [7.3 Gaussian Processes as Bayesian Approximation](#73-gaussian-processes-as-bayesian-approximation)
- [8. Applications in Machine Learning](#8-applications-in-machine-learning)
  - [8.1 Positional Encodings: Sinusoidal Interpolation](#81-positional-encodings-sinusoidal-interpolation)
  - [8.2 Neural Networks as Universal Approximators](#82-neural-networks-as-universal-approximators)
  - [8.3 Feature Maps and Kernel Approximation](#83-feature-maps-and-kernel-approximation)
- [9. Common Mistakes](#9-common-mistakes)
- [10. Exercises](#10-exercises)
- [11. Why This Matters for AI (2026 Perspective)](#11-why-this-matters-for-ai-2026-perspective)
- [12. Conceptual Bridge](#12-conceptual-bridge)

---

## 1. Intuition and Motivation

### 1.1 The Interpolation Problem

**Interpolation:** Given $n+1$ data pairs $(x_0, y_0), (x_1, y_1), \ldots, (x_n, y_n)$ with distinct nodes $x_i$, find a function $p$ such that $p(x_i) = y_i$ for all $i$.

**Approximation:** Find a function $p$ from some class $\mathcal{F}$ that minimizes some error $\|f - p\|$ over $\mathcal{F}$, possibly without passing through every data point.

The fundamental tension:
- **Interpolation** (zero training error) risks overfitting and instability (Runge's phenomenon)
- **Approximation** (nonzero training error) trades exactness for stability and generalization

This is exactly the bias-variance tradeoff in machine learning, seen through the lens of function approximation.

**For AI:** Every neural network layer computes a learned nonlinear function. Understanding which function classes can be represented and how efficiently is approximation theory. The expressive power of ReLU networks, the smoothness of GELU activations, and the positional encoding design in transformers all draw from this theory.

```
FUNCTION APPROXIMATION LANDSCAPE
════════════════════════════════════════════════════════════════════════

  Data:  (x_0,y_0), ..., (x_n,y_n)
                      │
              What class of f?
               ┌──────┴──────┐
          Global              Local
        ┌──────┐           ┌──────┐
     Polynomial          Piecewise
     (Lagrange,          (Splines,
      Chebyshev)          B-splines)
                               │
                         ┌─────┴──────┐
                      Smooth         Nonsmooth
                     (cubic)          (linear,
                                      B-spline)
  
  Special structures:
  ─ Trigonometric: periodic functions, FFT
  ─ RBF/kernel:    scattered data, RKHS, GP
  ─ Neural:        composition, ReLU networks

════════════════════════════════════════════════════════════════════════
```

### 1.2 Why Approximation Matters for AI

**1. Positional encodings:** The original transformer (Vaswani et al., 2017) uses sinusoidal positional encodings $\text{PE}(pos, 2i) = \sin(pos / 10000^{2i/d})$. This is a trigonometric interpolation scheme allowing the model to interpolate positions not seen during training.

**2. Kernel methods and SVMs:** The kernel trick $k(x, x') = \phi(x)^\top \phi(x')$ computes inner products in a feature space implicitly. The representer theorem shows that the optimal solution lies in the span of the training data — exactly the RBF interpolant form.

**3. Neural network expressiveness:** The universal approximation theorem states that a single hidden layer with enough neurons can approximate any continuous function on a compact set. The rate of approximation depends on the function's smoothness — exactly what approximation theory quantifies.

**4. Learned embeddings:** Word embeddings, graph node embeddings, and molecule representations are all continuous approximations of discrete objects — the embedding space provides an interpolation structure.

**5. Physics-informed neural networks (PINNs):** These solve PDEs by fitting a neural network as the solution function, treating the problem as function approximation subject to differential equation constraints.

### 1.3 Historical Timeline

```
INTERPOLATION AND APPROXIMATION: HISTORICAL TIMELINE
════════════════════════════════════════════════════════════════════════

  1700  ─ Newton's divided differences (1711): systematic polynomial interp.
  1795  ─ Gauss: least-squares method for orbit of Ceres
  1800  ─ Legendre: method of least squares (1805); orthogonal polynomials
  1853  ─ Chebyshev: polynomials with equioscillation, minimax approximation
  1900  ─ Runge: demonstrates instability of high-degree equispaced interp. (1901)
  1906  ─ Lebesgue: Lebesgue constant quantifies interpolation error
  1946  ─ Schoenberg: B-splines introduced for smoothing problems
  1965  ─ Cooley-Tukey: Fast Fourier Transform algorithm
  1970  ─ de Boor: stable B-spline evaluation algorithms
  1985  ─ Chui, Mallat: wavelet theory as multi-resolution approximation
  1989  ─ Cybenko, Hornik: universal approximation theorems for neural nets
  2017  ─ Vaswani et al.: sinusoidal positional encodings in transformers
  2020  ─ Tancik et al.: Fourier features for neural radiance fields (NeRF)
  2022  ─ RoPE (Su et al.): rotary positional encodings for LLMs (LLaMA)
  2024  ─ NTK theory: neural tangent kernel as linearized approximation

════════════════════════════════════════════════════════════════════════
```

---

## 2. Polynomial Interpolation

### 2.1 Lagrange Form

**Theorem (Existence and Uniqueness):** For any $n+1$ distinct nodes $x_0, x_1, \ldots, x_n$ and values $y_0, y_1, \ldots, y_n$, there exists a unique polynomial $p_n \in \mathcal{P}_n$ (degree $\leq n$) satisfying $p_n(x_i) = y_i$ for all $i$.

**Proof sketch:** The interpolation conditions $p_n(x_i) = y_i$ give a linear system $Va = y$ where $V$ is the Vandermonde matrix with $V_{ij} = x_i^j$. Since the $x_i$ are distinct, $V$ is invertible (its determinant is $\prod_{i > j}(x_i - x_j) \neq 0$). Uniqueness follows. $\square$

**Lagrange basis polynomials:** Define

$$\ell_j(x) = \prod_{\substack{k=0 \\ k \neq j}}^{n} \frac{x - x_k}{x_j - x_k}$$

Each $\ell_j$ is a degree-$n$ polynomial satisfying $\ell_j(x_i) = \delta_{ij}$ (Kronecker delta). The **Lagrange interpolating polynomial** is:

$$p_n(x) = \sum_{j=0}^{n} y_j \ell_j(x)$$

**Barycentric Lagrange Form:** The naive Lagrange form requires $O(n^2)$ operations to evaluate. The barycentric form is numerically stable and requires only $O(n)$ after $O(n^2)$ preprocessing:

$$p_n(x) = \frac{\sum_{j=0}^{n} \frac{w_j}{x - x_j} y_j}{\sum_{j=0}^{n} \frac{w_j}{x - x_j}}, \quad w_j = \frac{1}{\prod_{k \neq j}(x_j - x_k)}$$

The division by the denominator (which equals 1 by the Lagrange interpolation of the constant function $f \equiv 1$) provides automatic normalization and excellent numerical stability — it is one of the most stable algorithms in numerical analysis.

**For AI:** The attention mechanism in transformers can be viewed as a weighted sum $\sum_j \alpha_j v_j$ where the attention weights $\alpha_j = \text{softmax}(q^\top k_j / \sqrt{d})$ play a role analogous to the Lagrange basis weights $\ell_j(x)$. Both select how much each "stored value" contributes to the output.

### 2.2 Newton Divided Differences

The Newton form expresses $p_n$ in a basis adapted for sequential addition of data points:

$$p_n(x) = [y_0] + [y_0, y_1](x - x_0) + [y_0, y_1, y_2](x - x_0)(x - x_1) + \cdots$$

**Divided differences** are defined recursively:

$$[y_i] = y_i$$
$$[y_i, y_{i+1}, \ldots, y_{i+k}] = \frac{[y_{i+1}, \ldots, y_{i+k}] - [y_i, \ldots, y_{i+k-1}]}{x_{i+k} - x_i}$$

**Key properties:**
1. **Symmetric:** $[y_0, y_1, \ldots, y_n]$ is symmetric in all its arguments
2. **Derivative connection:** $[y_0, y_1, \ldots, y_n] = f^{(n)}(\xi) / n!$ for some $\xi$ in the convex hull of $\{x_0, \ldots, x_n\}$
3. **Incremental:** Adding a new data point $(x_{n+1}, y_{n+1})$ extends the Newton form by one term — $O(n)$ update vs $O(n^2)$ for recomputing Lagrange

**Newton divided difference table:**

```
x_0  │  y_0
     │         [y_0, y_1]
x_1  │  y_1               [y_0, y_1, y_2]
     │         [y_1, y_2]               [y_0, y_1, y_2, y_3]
x_2  │  y_2               [y_1, y_2, y_3]
     │         [y_2, y_3]
x_3  │  y_3
```

**Horner's rule** evaluates the Newton form in $O(n)$ operations:

```python
p = coeffs[n]
for i in range(n-1, -1, -1):
    p = p * (x - nodes[i]) + coeffs[i]
```

### 2.3 The Interpolation Error Theorem

**Theorem:** Let $f \in C^{n+1}[a, b]$ and $p_n$ be its interpolant at distinct nodes $x_0, \ldots, x_n \in [a,b]$. Then for any $x \in [a,b]$, there exists $\xi_x \in (a,b)$ such that:

$$f(x) - p_n(x) = \frac{f^{(n+1)}(\xi_x)}{(n+1)!} \omega_{n+1}(x)$$

where $\omega_{n+1}(x) = \prod_{i=0}^{n}(x - x_i)$ is the **node polynomial**.

**Proof sketch:** Define $g(t) = f(t) - p_n(t) - \lambda \omega_{n+1}(t)$ where $\lambda$ is chosen so that $g(x) = 0$. Then $g$ has $n+2$ zeros: $x_0, \ldots, x_n, x$. By Rolle's theorem, $g'$ has $n+1$ zeros, $g''$ has $n$ zeros, ..., $g^{(n+1)}$ has 1 zero $\xi_x$. Computing $g^{(n+1)}(\xi_x) = 0$ gives $\lambda = f^{(n+1)}(\xi_x)/(n+1)!$. $\square$

**Implications:**
1. The error is controlled by $\|\omega_{n+1}\|_\infty = \max_{x \in [a,b]} |\omega_{n+1}(x)|$ — choosing nodes to minimize this is Chebyshev's contribution
2. Smooth functions ($f^{(n+1)}$ small) are approximated well
3. High-degree polynomials require $f$ to have many bounded derivatives

**Lebesgue constant:** The interpolation error can also be bounded by:

$$\|f - p_n\|_\infty \leq (1 + \Lambda_n) \|f - p_n^*\|_\infty$$

where $p_n^*$ is the best polynomial approximant of degree $n$ and $\Lambda_n = \max_{x \in [a,b]} \sum_j |\ell_j(x)|$ is the **Lebesgue constant**. The Lebesgue constant quantifies how much the interpolant amplifies errors in $y_i$.

- Equispaced nodes: $\Lambda_n \sim 2^n / (e n \ln n)$ — **exponential growth** (Runge's phenomenon)
- Chebyshev nodes: $\Lambda_n \sim \frac{2}{\pi} \ln(n+1) + 0.52$ — **logarithmic growth** (nearly optimal)

### 2.4 Runge's Phenomenon

**Runge's phenomenon (1901):** The interpolant of $f(x) = 1/(1+25x^2)$ at $n+1$ equispaced nodes on $[-1,1]$ diverges as $n \to \infty$, even though $f$ is infinitely differentiable!

The reason is purely numerical: the node polynomial $\omega_{n+1}(x)$ for equispaced nodes is extremely large near the boundary of $[-1, 1]$. Even though $f^{(n+1)}$ is bounded, the product grows without bound.

```
RUNGE'S PHENOMENON: EQUISPACED vs CHEBYSHEV NODES
════════════════════════════════════════════════════════════════════════

  f(x) = 1/(1+25x²),  n=15 interpolation points

  Equispaced nodes (-1, -12/15, ..., 12/15, 1):
  ─ Huge oscillations near ±1
  ─ Max error ≈ 10² (function max is 1!)
  ─ Lebesgue constant Λ₁₅ ≈ 5000

  Chebyshev nodes (cos((2k+1)π/(2n+2))):
  ─ Near-optimal approximation
  ─ Max error ≈ 10⁻³
  ─ Lebesgue constant Λ₁₅ ≈ 3.5

  Moral: Node placement is as important as degree!

════════════════════════════════════════════════════════════════════════
```

**For AI:** Equispaced sampling is the default in practice (uniform time steps, uniform grid). Understanding why this can fail for high-degree polynomial interpolation is essential context for:
- Time-series models that use polynomial basis functions
- Finite element methods in physics-informed neural networks
- Feature engineering with polynomial features (always use regularization or orthogonal polynomials)

---

## 3. Chebyshev Polynomials and Optimal Nodes

### 3.1 Chebyshev Polynomials

The **Chebyshev polynomials of the first kind** are defined via the remarkable trigonometric identity:

$$T_n(\cos\theta) = \cos(n\theta), \quad \theta \in [0, \pi]$$

Equivalently, on $[-1, 1]$:

$$T_0(x) = 1, \quad T_1(x) = x, \quad T_n(x) = 2x T_{n-1}(x) - T_{n-2}(x)$$

**Key properties:**

| Property | Formula |
|---|---|
| Boundedness | $\|T_n\|_\infty = 1$ on $[-1,1]$ |
| Roots (Chebyshev points of the 1st kind) | $x_k = \cos\!\left(\frac{(2k+1)\pi}{2n}\right)$, $k = 0, \ldots, n-1$ |
| Extrema (Chebyshev points of the 2nd kind) | $x_k = \cos\!\left(\frac{k\pi}{n}\right)$, $k = 0, \ldots, n$ |
| Orthogonality | $\int_{-1}^{1} T_m(x) T_n(x) \frac{dx}{\sqrt{1-x^2}} = \frac{\pi}{2} \delta_{mn}$ (for $m,n \geq 1$) |
| Leading coefficient | $T_n(x) = 2^{n-1} x^n + \text{lower order}$ |
| Minimax property | Among all monic polynomials of degree $n$, $T_n(x)/2^{n-1}$ has the smallest $\infty$-norm on $[-1,1]$ |

**Why the trigonometric definition works:** $T_n(\cos\theta) = \cos(n\theta)$ is periodic in $\theta$ and stays in $[-1,1]$. The change of variables $x = \cos\theta$ maps $[0,\pi]$ to $[-1,1]$, compressing the uniform distribution on $\theta$ to a non-uniform (arcsine) distribution on $x$ that weights the endpoints more heavily.

**Chebyshev series:** Any continuous function $f: [-1,1] \to \mathbb{R}$ can be expanded as:

$$f(x) = \sum_{k=0}^{\infty} c_k T_k(x), \quad c_k = \frac{2}{\pi} \int_{-1}^{1} f(x) T_k(x) \frac{dx}{\sqrt{1-x^2}}$$

(with $c_0$ divided by 2). The Chebyshev series converges faster than any fixed power of $1/n$ for analytic functions — **exponential convergence for analytic $f$**.

**For AI:** Chebyshev polynomials appear in:
- **Spectral graph neural networks:** ChebNet approximates the graph spectral filter $g(\Lambda)$ by a Chebyshev polynomial in $\Lambda$, avoiding eigendecomposition
- **Polynomial activation functions:** Some neural architectures use Chebyshev polynomials as activation functions for efficient function approximation
- **Numerical PDE solvers:** Spectral methods using Chebyshev expansions achieve exponential accuracy for smooth solutions

### 3.2 Chebyshev Nodes

**The Optimal Node Placement Problem:** Which $n+1$ nodes $x_0, \ldots, x_n \in [-1,1]$ minimize $\max_{x \in [-1,1]} |\omega_{n+1}(x)|$ where $\omega_{n+1}(x) = \prod_i (x - x_i)$?

**Answer:** The Chebyshev points of the first kind (roots of $T_{n+1}$):

$$x_k = \cos\left(\frac{(2k+1)\pi}{2(n+1)}\right), \quad k = 0, 1, \ldots, n$$

With these nodes, $\omega_{n+1}(x) = T_{n+1}(x) / 2^n$ and:

$$\max_{x \in [-1,1]} |\omega_{n+1}(x)| = \frac{1}{2^n}$$

This is the **smallest possible** value (by the minimax property of $T_{n+1}$).

**Error bound with Chebyshev nodes:**

$$\|f - p_n\|_\infty \leq \frac{\|f^{(n+1)}\|_\infty}{2^n (n+1)!}$$

Compare to equispaced nodes where $\|\omega_{n+1}\|_\infty \sim n! h^{n+1}$ (much larger for large $n$).

**Rescaling to $[a,b]$:** The Chebyshev nodes on $[a,b]$ are:

$$x_k = \frac{a+b}{2} + \frac{b-a}{2} \cos\left(\frac{(2k+1)\pi}{2(n+1)}\right)$$

### 3.3 Minimax Approximation

The **minimax problem** asks: among all polynomials $p \in \mathcal{P}_n$, find $p^*$ minimizing $\|f - p\|_\infty$.

**Chebyshev's theorem (Equioscillation theorem):** $p^*$ is the best approximation if and only if there exist at least $n+2$ points $x_0 < x_1 < \cdots < x_{n+1}$ in $[a,b]$ where the error equioscillates:

$$f(x_k) - p^*(x_k) = (-1)^k E, \quad E = \pm \|f - p^*\|_\infty$$

**Remez algorithm:** An iterative algorithm that finds the minimax polynomial by exchanging equioscillation points. Converges quadratically.

**For smooth functions:** The minimax polynomial approximation converges exponentially for analytic functions — meaning $\|f - p_n^*\|_\infty \leq C \rho^{-n}$ for some $\rho > 1$ depending on the domain of analyticity.

### 3.4 Clenshaw-Curtis Quadrature Connection

The Chebyshev points of the second kind (extrema of $T_n$) are central to **Clenshaw-Curtis quadrature** (see §10-05). The key insight: integrating the Chebyshev interpolant exactly gives highly accurate quadrature weights. This is treated in detail in the Numerical Integration section.

---

## 4. Spline Interpolation

### 4.1 Piecewise Polynomial Interpolation

**Motivation:** High-degree global polynomial interpolation is unstable (Runge). Instead, use a low-degree polynomial on each subinterval — **piecewise polynomials** or **splines**.

**Definition:** Given nodes $a = x_0 < x_1 < \cdots < x_n = b$ (breakpoints), a **spline of degree $k$** is a function $S: [a,b] \to \mathbb{R}$ such that:
1. On each interval $[x_i, x_{i+1}]$, $S$ is a polynomial of degree $\leq k$
2. $S \in C^{k-1}[a,b]$ — has $k-1$ continuous derivatives globally

**Piecewise linear interpolation ($k=1$):** Connect adjacent data points with straight lines. Simple, $O(n)$ construction, but only $C^0$ — derivatives are discontinuous. Error: $O(h^2)$ where $h = \max_i (x_{i+1} - x_i)$.

**Piecewise cubic ($k=3$):** The sweet spot — 4 degrees of freedom per interval, 2 boundary conditions at each interior node ($C^2$ continuity), leaving 2 free conditions (boundary conditions). Error: $O(h^4)$.

### 4.2 Cubic Splines: Derivation and Construction

**Natural cubic spline:** Given $n+1$ data points, find $S \in C^2[a,b]$ piecewise cubic satisfying:
1. $S(x_i) = y_i$ for $i = 0, \ldots, n$ (interpolation)
2. $S''(x_0) = S''(x_n) = 0$ (natural boundary conditions — zero curvature at endpoints)

**Derivation:** On $[x_i, x_{i+1}]$ with $h_i = x_{i+1} - x_i$, let $M_i = S''(x_i)$ be the unknown second derivatives. Since $S$ is cubic on each interval and $S''$ is linear:

$$S''(x) = M_i \frac{x_{i+1} - x}{h_i} + M_{i+1} \frac{x - x_i}{h_i}$$

Integrating twice and applying $S(x_i) = y_i$, $S(x_{i+1}) = y_{i+1}$:

$$S(x) = M_i \frac{(x_{i+1}-x)^3}{6h_i} + M_{i+1} \frac{(x-x_i)^3}{6h_i} + \left(y_i - \frac{M_i h_i^2}{6}\right)\frac{x_{i+1}-x}{h_i} + \left(y_{i+1} - \frac{M_{i+1} h_i^2}{6}\right)\frac{x-x_i}{h_i}$$

**Imposing $C^1$ continuity** (matching first derivatives at interior nodes):

$$h_{i-1} M_{i-1} + 2(h_{i-1} + h_i) M_i + h_i M_{i+1} = 6 \left(\frac{y_{i+1}-y_i}{h_i} - \frac{y_i - y_{i-1}}{h_{i-1}}\right), \quad i = 1, \ldots, n-1$$

This gives a **tridiagonal linear system** for $M_1, \ldots, M_{n-1}$ (with $M_0 = M_n = 0$ for natural BC). Tridiagonal systems are solved in $O(n)$ with the Thomas algorithm — spectacularly efficient.

```
CUBIC SPLINE SYSTEM (tridiagonal)
════════════════════════════════════════════════════════════════════════

  ┌                                  ┐ ┌   ┐   ┌   ┐
  │ 2(h₀+h₁)  h₁                    │ │M₁ │   │r₁ │
  │ h₁      2(h₁+h₂)  h₂            │ │M₂ │   │r₂ │
  │           h₂    2(h₂+h₃)  h₃   │ │M₃ │ = │r₃ │
  │                  ⋱      ⋱   ⋱   │ │ ⋮ │   │ ⋮ │
  │                    h_{n-2} 2(h_{n-2}+h_{n-1})│ │M_{n-1}│   │r_{n-1}│
  └                                  ┘ └   ┘   └   ┘

  rᵢ = 6[(yᵢ₊₁-yᵢ)/hᵢ - (yᵢ-yᵢ₋₁)/hᵢ₋₁]  (second finite differences of y)

  Solved by Thomas algorithm in O(n) operations.

════════════════════════════════════════════════════════════════════════
```

**Error analysis:** For the natural cubic spline:

$$\|f - S\|_\infty \leq \frac{5}{384} h^4 \|f^{(4)}\|_\infty$$

where $h = \max h_i$. The $O(h^4)$ convergence makes cubic splines highly accurate for modest grid spacing.

**Boundary condition options:**
- **Natural:** $S''(a) = S''(b) = 0$ (minimizes $\int |S''|^2$)
- **Clamped:** $S'(a) = f'(a)$, $S'(b) = f'(b)$ (uses derivative information)
- **Not-a-knot:** $S'''$ continuous at $x_1$ and $x_{n-1}$ (used when derivatives unknown)
- **Periodic:** $S'(a) = S'(b)$, $S''(a) = S''(b)$ (for periodic data)

**Minimum curvature property:** The natural cubic spline minimizes $\int_a^b |S''(x)|^2 dx$ among all twice-differentiable interpolants. This is the **thin plate spline** principle: the spline is the smoothest interpolant.

**For AI:** 
- **Neural ODEs** use cubic spline interpolation to densify trajectory predictions
- **Temporal point processes** use spline bases to model intensity functions
- The minimum curvature property connects to L2 regularization: the RKHS norm for the Matérn-3/2 kernel is equivalent to $\int |S''|^2$

### 4.3 B-Splines

B-splines (**basis splines**) provide a numerically stable, local basis for the spline space. Rather than working with the second-derivative representation, B-splines use recursively defined local basis functions.

**Definition (Cox-de Boor recursion):** Given a **knot vector** $t_0 \leq t_1 \leq \cdots \leq t_{n+k}$, the B-spline basis functions of degree $p$ are:

$$B_{i,0}(x) = \begin{cases} 1 & t_i \leq x < t_{i+1} \\ 0 & \text{otherwise} \end{cases}$$

$$B_{i,p}(x) = \frac{x - t_i}{t_{i+p} - t_i} B_{i,p-1}(x) + \frac{t_{i+p+1} - x}{t_{i+p+1} - t_{i+1}} B_{i+1,p-1}(x)$$

**Key properties:**
- **Local support:** $B_{i,p}(x) = 0$ outside $[t_i, t_{i+p+1}]$ — changing one control point affects only a local region
- **Partition of unity:** $\sum_i B_{i,p}(x) = 1$ for all $x$ — control points are convex combinations
- **Non-negativity:** $B_{i,p}(x) \geq 0$
- **Convex hull property:** The spline curve lies within the convex hull of its control points

**B-spline curve:** $S(x) = \sum_{i=0}^{n} c_i B_{i,p}(x)$ where $c_i$ are control point coefficients.

**For AI:** B-splines appear in:
- **Computer graphics:** NURBS (Non-Uniform Rational B-Splines) for representing curved surfaces in 3D rendering used in diffusion model outputs
- **Kolmogorov-Arnold Networks (KAN):** Replace MLP neurons with learned B-spline functions — a direct application of spline approximation theory in neural networks
- **Time-series modeling:** Spline bases for smooth covariate functions in survival analysis and temporal models

### 4.4 Tension Splines and Shape Preservation

Standard cubic splines can overshoot — producing local extrema not present in the data. **Monotone splines** (Fritsch-Carlson algorithm) adjust slopes to preserve monotonicity:

1. Compute slopes from divided differences
2. If $d_i$ has opposite sign to adjacent differences, set $d_i = 0$
3. Rescale slopes using $\alpha_i^2 + \beta_i^2 \leq 9$ condition

**For AI:** Monotone spline constraints appear in calibration curves, cumulative distribution function modeling, and isotonic regression.

---

## 5. Least-Squares Approximation

### 5.1 Polynomial Least Squares

Given $m$ data points $(x_1, y_1), \ldots, (x_m, y_m)$ with $m > n+1$, find the degree-$n$ polynomial minimizing:

$$\min_{p \in \mathcal{P}_n} \sum_{i=1}^{m} (y_i - p(x_i))^2 = \min_{c \in \mathbb{R}^{n+1}} \|Vc - y\|_2^2$$

where $V \in \mathbb{R}^{m \times (n+1)}$ is the Vandermonde matrix $V_{ij} = x_i^j$.

**The Vandermonde conditioning problem:** Even for moderate $n$, $\kappa(V) = \|V\| \|V^{-1}\|$ grows like $O(\rho^n)$ for some $\rho > 1$. Solving the normal equations $V^\top V c = V^\top y$ squares the condition number: $\kappa(V^\top V) = \kappa(V)^2$.

**Solution via QR decomposition:** Compute $V = QR$ (thin QR), then $c = R^{-1} Q^\top y$. This is stable and avoids squaring the condition number:

$$\kappa_{\text{QR solve}} = \kappa(V) \quad \text{vs} \quad \kappa_{\text{Normal eq.}} = \kappa(V)^2$$

**For AI:** This is the same condition-number-squaring problem as in §10-02. Linear regression with polynomial features should always use QR or SVD, never the normal equations with the Vandermonde matrix directly.

### 5.2 Orthogonal Polynomials and Gram-Schmidt

The Vandermonde ill-conditioning arises because the monomial basis $\{1, x, x^2, \ldots\}$ is nearly linearly dependent for large $n$ and typical node distributions. The fix: use an **orthogonal polynomial basis** for the weight function.

**Orthogonal polynomial construction (three-term recurrence):** For a weight function $w(x) > 0$ on $[a,b]$:

$$\phi_0(x) = 1, \quad \phi_1(x) = x - \alpha_0$$
$$\phi_{k+1}(x) = (x - \alpha_k) \phi_k(x) - \beta_k \phi_{k-1}(x)$$

where the coefficients are:

$$\alpha_k = \frac{\langle x \phi_k, \phi_k \rangle}{\langle \phi_k, \phi_k \rangle}, \quad \beta_k = \frac{\langle \phi_k, \phi_k \rangle}{\langle \phi_{k-1}, \phi_{k-1} \rangle}$$

and $\langle f, g \rangle = \int_a^b f(x) g(x) w(x) dx$.

**Standard families:**

| Weight $w(x)$ | Interval | Polynomials |
|---|---|---|
| $1$ | $[-1, 1]$ | Legendre $P_n$ |
| $(1-x^2)^{-1/2}$ | $[-1, 1]$ | Chebyshev $T_n$ |
| $e^{-x}$ | $[0, \infty)$ | Laguerre $L_n$ |
| $e^{-x^2}$ | $(-\infty, \infty)$ | Hermite $H_n$ |

**Discrete orthogonal polynomials:** For a finite set of nodes with weights $w_i$, the discrete inner product $\langle f, g \rangle = \sum_i w_i f(x_i) g(x_i)$ gives discrete orthogonal polynomials. Using these as the basis makes the least-squares problem diagonal: $\hat{c}_k = \langle f, \phi_k \rangle / \langle \phi_k, \phi_k \rangle$.

**For AI:** Hermite polynomials appear in quantum mechanics (wavefunction basis) and in the theory of Gaussian integrals. The connection to attention mechanism: the softmax function can be written as a ratio of Hermite polynomial evaluations at zero.

### 5.3 Minimax vs Least-Squares

**Minimax (Chebyshev) approximation:** Minimizes $\|f - p\|_\infty = \max_x |f(x) - p(x)|$.
- Equioscillation at $n+2$ points (Chebyshev's theorem)
- Found by the Remez algorithm
- Best for: applications where the worst case matters (safety-critical, numerical tables)

**Least-squares ($L^2$) approximation:** Minimizes $\|f - p\|_2^2 = \int |f - p|^2 w dx$.
- Found by orthogonal projection: $c_k = \langle f, \phi_k \rangle / \|\phi_k\|^2$
- Best for: statistical fitting, machine learning (minimizes MSE)

**Comparison:**

| Criterion | Minimax | Least-Squares |
|---|---|---|
| Error norm | $L^\infty$ | $L^2$ |
| Algorithm | Remez (iterative) | QR / orthogonal projection |
| Sensitivity to outliers | High | Lower (outliers at max) |
| Computation | $O(n^2)$ per Remez step | $O(mn)$ one-shot |
| ML analogy | Max-margin (SVM) | MSE (regression) |

---

## 6. Fourier Approximation

### 6.1 Trigonometric Interpolation

For **periodic functions** on $[0, 2\pi]$, the natural interpolating functions are trigonometric polynomials. Given $n$ equispaced nodes $x_j = 2\pi j / n$ for $j = 0, \ldots, n-1$, the **trigonometric interpolant** is:

$$p(x) = \sum_{k=-n/2}^{n/2-1} c_k e^{ikx}, \quad c_k = \frac{1}{n} \sum_{j=0}^{n-1} y_j e^{-ikx_j}$$

The coefficients $c_k$ are exactly the **Discrete Fourier Transform (DFT)** of the data $y_j$.

**Error for periodic functions:** For a function with Fourier series $f(x) = \sum_k \hat{f}_k e^{ikx}$:

$$\|f - p_n\|_\infty \leq 2\sum_{|k| > n/2} |\hat{f}_k|$$

For smooth periodic functions, the Fourier coefficients decay rapidly (exponentially for analytic functions), so this error is very small — **spectral accuracy**.

### 6.2 Discrete Fourier Transform

**Definition:** The DFT of a vector $y \in \mathbb{C}^n$ is:

$$Y_k = \sum_{j=0}^{n-1} y_j \omega_n^{-jk}, \quad k = 0, 1, \ldots, n-1$$

where $\omega_n = e^{2\pi i/n}$ is the primitive $n$th root of unity.

**Matrix form:** $Y = F_n y$ where $(F_n)_{jk} = \omega_n^{-jk}$.

The matrix $F_n$ is **unitary** (up to scaling): $F_n^* F_n = n I$, so the inverse DFT (IDFT) is:

$$y_j = \frac{1}{n} \sum_{k=0}^{n-1} Y_k \omega_n^{jk}$$

**Properties:**
- **Parseval's theorem:** $\|Y\|^2 = n \|y\|^2$ (energy conservation)
- **Convolution theorem:** DFT of convolution = pointwise product of DFTs
- **Real signals:** If $y_j \in \mathbb{R}$, then $Y_{n-k} = \overline{Y_k}$ (conjugate symmetry)

**For AI:** The convolution theorem is the mathematical foundation of CNNs: a convolution in spatial domain equals pointwise multiplication in frequency domain. FFT-based convolution has complexity $O(n \log n)$ vs $O(n^2)$ for direct convolution.

### 6.3 Fast Fourier Transform

**Naive DFT:** $O(n^2)$ — computing each of $n$ output values requires summing $n$ terms.

**Cooley-Tukey FFT (1965):** Exploits the factorization of $n$ (assuming $n = 2^m$) to reduce complexity to $O(n \log n)$.

**Radix-2 Decimation-in-Time:** Split $y$ into even/odd indices:

$$Y_k = \underbrace{\sum_{j=0}^{n/2-1} y_{2j} \omega_n^{-2jk}}_{E_k} + \omega_n^{-k} \underbrace{\sum_{j=0}^{n/2-1} y_{2j+1} \omega_n^{-2jk}}_{O_k}$$

Since $\omega_n^{-2j} = \omega_{n/2}^{-j}$, both $E$ and $O$ are DFTs of size $n/2$. The butterfly pattern:

```
DFT BUTTERFLY STRUCTURE (n=8)
════════════════════════════════════════════════════════════════════════

  Input (bit-reversed)          Output (in-order)
  y₀  ────┐                       Y₀
  y₄  ────┤─ butterfly ─────────  Y₁
  y₂  ────┤─ butterfly ─────────  Y₂
  y₆  ────┤─ butterfly ─────────  Y₃
  y₁  ────┤─ butterfly ─────────  Y₄
  y₅  ────┤─ butterfly ─────────  Y₅
  y₃  ────┤─ butterfly ─────────  Y₆
  y₇  ────┘─ butterfly ─────────  Y₇

  3 stages × 4 butterflies = 12 operations vs 64 for naive DFT.
  Speedup: O(n²) → O(n log n).

════════════════════════════════════════════════════════════════════════
```

**Complexity:** $T(n) = 2T(n/2) + O(n)$ solves to $T(n) = O(n \log n)$. For $n = 10^6$, FFT requires $\sim 2 \times 10^7$ operations vs $10^{12}$ for naive DFT — a factor of $50{,}000\times$ speedup.

**For AI:** FFT is used in:
- Spectral convolution layers (FNet, S4, Hyena)
- Long-range attention in Performer (via random Fourier features)
- Frequency-domain training of neural networks

### 6.4 Aliasing and the Nyquist Theorem

**Aliasing:** When sampling at rate $f_s$, frequencies above $f_s/2$ (the Nyquist frequency) are indistinguishable from lower frequencies. The frequency $f$ aliases to $f - f_s$ if $f > f_s/2$.

**Nyquist-Shannon sampling theorem:** A signal with bandwidth $B$ (no frequencies above $B$) can be perfectly reconstructed from samples taken at rate $f_s \geq 2B$.

**For AI:** Aliasing matters in:
- Audio processing (e.g., sample rate 44.1 kHz supports up to 22 kHz — just above human hearing)
- Image super-resolution: downsampling without anti-aliasing introduces artifacts
- Periodic positional encodings: the frequency range $[10^{-4}, 1]$ in transformer PEs avoids aliasing for typical sequence lengths

---

## 7. Radial Basis Functions and Kernel Methods

### 7.1 RBF Interpolation

For **scattered data** (non-uniform nodes in $\mathbb{R}^d$), polynomial interpolation becomes problematic. **Radial basis function (RBF) interpolation** uses:

$$s(x) = \sum_{j=1}^{n} c_j \phi(\|x - x_j\|), \quad x \in \mathbb{R}^d$$

where $\phi: [0, \infty) \to \mathbb{R}$ is a radial function. Common choices:

| Name | $\phi(r)$ | Smoothness |
|---|---|---|
| Gaussian | $e^{-\varepsilon^2 r^2}$ | $C^\infty$ |
| Multiquadric | $\sqrt{1 + \varepsilon^2 r^2}$ | $C^\infty$ |
| Inverse multiquadric | $(1 + \varepsilon^2 r^2)^{-1/2}$ | $C^\infty$ |
| Thin plate spline | $r^2 \log r$ | $C^1$ |
| Matérn-3/2 | $(1 + \sqrt{3}r/\ell)\exp(-\sqrt{3}r/\ell)$ | $C^2$ |
| Wendland | piecewise polynomial, compact support | $C^{2k}$ |

**Interpolation system:** The coefficients $c$ solve the linear system:

$$\Phi c = y, \quad \Phi_{ij} = \phi(\|x_i - x_j\|)$$

For **positive definite** RBFs (Gaussian, Matérn), $\Phi$ is positive definite — guaranteed unique solution.

**For AI:** The Gaussian RBF kernel is the most common kernel in kernel machines (SVM, Gaussian processes). The shape parameter $\varepsilon$ (or lengthscale $\ell$) controls the smoothness/locality tradeoff — analogous to the learning rate in neural networks.

### 7.2 Connection to Reproducing Kernel Hilbert Spaces

**Reproducing Kernel Hilbert Space (RKHS):** For a symmetric positive definite kernel $k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$, the RKHS $\mathcal{H}_k$ is the completion of $\text{span}\{k(\cdot, x) : x \in \mathcal{X}\}$ with the inner product satisfying the **reproducing property:**

$$\langle f, k(\cdot, x) \rangle_{\mathcal{H}} = f(x)$$

**Representer theorem:** The minimizer of $\min_{f \in \mathcal{H}} \sum_i (y_i - f(x_i))^2 + \lambda \|f\|_{\mathcal{H}}^2$ has the form:

$$f^* = \sum_{j=1}^{n} c_j k(\cdot, x_j)$$

This is exactly the RBF interpolant! The RKHS norm $\|f\|_{\mathcal{H}}$ acts as a regularizer controlling smoothness.

**For AI:** The representer theorem is foundational for:
- **Kernel SVMs:** The decision boundary is a linear combination of support vectors in RKHS
- **Gaussian process regression:** The posterior mean is the RKHS interpolant; posterior variance is the RKHS error
- **Neural tangent kernel (NTK):** Infinite-width neural networks converge to kernel regression with the NTK kernel

### 7.3 Gaussian Processes as Bayesian Approximation

A **Gaussian process** $f \sim \mathcal{GP}(\mu, k)$ is a distribution over functions where any finite collection $(f(x_1), \ldots, f(x_n))$ is multivariate Gaussian with mean $(\mu(x_1), \ldots, \mu(x_n))$ and covariance matrix $K_{ij} = k(x_i, x_j)$.

**GP posterior:** Given observations $y_i = f(x_i) + \varepsilon_i$ with $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$:

$$f(x^*) | \mathbf{y} \sim \mathcal{N}(\mu^*, \sigma^{*2})$$
$$\mu^* = k^{*\top}(K + \sigma^2 I)^{-1} \mathbf{y}$$
$$\sigma^{*2} = k(x^*, x^*) - k^{*\top}(K + \sigma^2 I)^{-1} k^*$$

where $k^*_i = k(x^*, x_i)$.

The posterior mean $\mu^*$ is precisely the kernel ridge regression solution — Bayesian inference gives both a point estimate and uncertainty quantification "for free".

**For AI:** Gaussian processes are used for:
- **Bayesian optimization:** Surrogate model for expensive black-box functions (hyperparameter tuning)
- **Neural architecture search (NAS):** GP surrogate for network performance prediction
- **Uncertainty quantification:** GP posterior variance estimates prediction confidence

---

## 8. Applications in Machine Learning

### 8.1 Positional Encodings: Sinusoidal Interpolation

The original transformer uses sinusoidal positional encodings (Vaswani et al., 2017):

$$\text{PE}(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad \text{PE}(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

**Approximation theory perspective:** These are trigonometric interpolation functions. The frequencies span $[10000^{-1}, 1]$ — about 4 decades — covering both local (high frequency) and global (low frequency) position information.

**Why sinusoidal?** For any offset $k$, $\text{PE}(pos+k)$ is a linear function of $\text{PE}(pos)$ — the relative position $k$ can be expressed via a rotation matrix. This allows the model to attend to relative positions.

**RoPE (Rotary Position Embedding, Su et al. 2022):** Used in LLaMA, Mistral, GPT-NeoX. Applies a position-dependent rotation to query and key vectors:

$$q_m = R_m q, \quad k_n = R_n k, \quad R_m = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix}$$

The inner product $q_m^\top k_n = q^\top R_m^\top R_n k = q^\top R_{m-n} k$ depends only on relative position $m-n$ — exact relative position encoding via complex multiplication.

**For AI:** The interpolation challenge arises when extending context length. A model trained with RoPE on 2048 tokens needs to generalize to 8192 — trigonometric **extrapolation**, not interpolation. Methods like YaRN (Peng et al., 2023) modify the frequency range to enable longer context via interpolation of the position indices.

### 8.2 Neural Networks as Universal Approximators

**Cybenko's theorem (1989):** Any continuous function $f: [0,1]^d \to \mathbb{R}$ can be approximated uniformly by a single hidden layer network:

$$f(x) \approx \sum_{j=1}^{N} c_j \sigma(a_j^\top x + b_j)$$

for any continuous sigmoidal $\sigma$. As $N \to \infty$, the approximation error goes to zero.

**Modern variants:**
- **Barron's theorem (1993):** Functions with finite Barron complexity $C_f = \int |\hat{f}(\omega)| \|\omega\| d\omega < \infty$ can be approximated with $O(1/N)$ squared $L^2$ error using $N$ neurons — independent of dimension $d$. This avoids the curse of dimensionality for this function class.
- **ReLU depth separation:** Deep ReLU networks of depth $L$ can represent functions that would require $O(2^n)$ neurons in a shallow network — depth exponentially increases expressiveness
- **Approximation vs optimization:** Universal approximation guarantees *existence* of weights; finding them via gradient descent is a separate (hard) problem

**KAN (Kolmogorov-Arnold Networks, 2024):** Replace the node-wise nonlinearity with edge-wise learned univariate functions (represented as B-splines):

$$\text{KAN}(x) = \sum_{j} \phi_{L,j}(\ldots \sum_i \phi_{1,ij}(x_i) \ldots)$$

Based on the Kolmogorov-Arnold representation theorem: every continuous function of $n$ variables can be written as a composition of univariate functions and addition. KANs directly implement this with learnable splines.

### 8.3 Feature Maps and Kernel Approximation

**Random Fourier Features (Rahimi-Recht, 2007):** The shift-invariant kernel $k(x, y) = k(x-y)$ can be written (Bochner's theorem) as:

$$k(x - y) = \mathbb{E}_{\omega \sim p(\omega)}[e^{i\omega^\top(x-y)}]$$

where $p(\omega)$ is the spectral density. Approximating with $D$ random frequencies:

$$k(x, y) \approx z(x)^\top z(y), \quad z(x) = \frac{1}{\sqrt{D}} \begin{pmatrix} \cos(\omega_1^\top x + b_1) \\ \vdots \\ \cos(\omega_D^\top x + b_D) \end{pmatrix}$$

This turns a kernel method into a linear model in feature space — $O(nD)$ instead of $O(n^2)$ for kernel matrix.

**For AI:** Random Fourier features are the foundation of:
- **Performer (Choromanski et al., 2020):** Approximates softmax attention with $O(n)$ complexity using random feature maps
- **Neural Tangent Kernel implementations:** Computing NTK efficiently for large networks

---

## 9. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
|---|---|---|---|
| 1 | Using equispaced nodes for high-degree polynomial interpolation | Runge's phenomenon: error grows exponentially near endpoints | Use Chebyshev nodes or splines |
| 2 | Solving least-squares via the normal equations with Vandermonde matrix | Condition number squared: $\kappa(V^\top V) = \kappa(V)^2$ grows like $\rho^{2n}$ | Use QR decomposition or orthogonal polynomial basis |
| 3 | Extrapolating polynomial interpolants | All polynomial interpolants diverge outside $[x_0, x_n]$ unless the function is actually polynomial | Use local models (splines) or explicit extrapolation assumptions |
| 4 | Setting spline boundary conditions to natural ($S''=0$) when data is not natural | Introduces artificial inflection points if the true function has nonzero curvature at endpoints | Use clamped or not-a-knot conditions when derivative information is available |
| 5 | Confusing interpolation degree with approximation quality | High degree interpolation is not always more accurate (Runge) | Match the basis to the data's smoothness and node count |
| 6 | Forgetting that DFT assumes periodic extension | Discontinuity at boundaries causes Gibbs phenomenon — oscillations near jumps | Apply windowing (Hann, Hamming) or use non-uniform FFT |
| 7 | Using Gaussian RBF with too large an $\varepsilon$ | Leads to nearly singular $\Phi$ matrix ($\kappa \sim e^{\varepsilon^2 r^2}$) | Tune $\varepsilon$ via cross-validation or use stable algorithms (contour Padé) |
| 8 | Treating the Lebesgue constant as just a theoretical concept | The Lebesgue constant directly bounds interpolation error amplification — if $\Lambda_n$ is large, small errors in $y_i$ cause large errors in $p_n$ | Always check the Lebesgue constant for your node choice |
| 9 | Assuming neural network universality implies easy learning | UAT says approximating functions *exists*, not that gradient descent finds it | Understand the approximation-optimization gap |
| 10 | Misinterpreting sinusoidal PE frequencies as sampling frequencies | The $10000^{2i/d}$ factor controls the wavelength (period), not a sampling rate | Think in terms of wavelengths: smallest = $2\pi$, largest = $2\pi \times 10000$ |
| 11 | Using fixed-$\varepsilon$ RBF for high-dimensional data | RBF interpolation is ill-conditioned in high dimensions without careful selection | Use compactly supported RBFs (Wendland) or regularized regression |
| 12 | Evaluating Lagrange basis polynomials naively | Numerical cancellation in $\prod_{k \neq j}(x - x_k)$ for large $n$ | Use barycentric Lagrange form: numerically stable and $O(n)$ per evaluation |

---

## 10. Exercises

**Exercise 1 ★ — Newton Divided Differences**  
Implement the Newton divided difference algorithm and evaluate the interpolating polynomial using Horner's rule.  
(a) Compute the divided difference table for $(0,1), (1,3), (2,7), (3,13)$.  
(b) Implement `newton_interp(nodes, vals, x)` using the divided difference table and Horner's method.  
(c) Verify the result matches the Lagrange interpolant.

**Exercise 2 ★ — Runge's Phenomenon**  
Demonstrate Runge's phenomenon numerically and show that Chebyshev nodes tame it.  
(a) For $f(x) = 1/(1+25x^2)$, compute the interpolation error at $n = 5, 10, 15, 20$ with equispaced nodes.  
(b) Repeat with Chebyshev nodes. Plot both error curves on a log scale.  
(c) Compute the Lebesgue constant for both node sets at $n = 15$.

**Exercise 3 ★ — Natural Cubic Spline**  
Build a cubic spline solver from scratch using the tridiagonal system.  
(a) Derive the tridiagonal system for the second derivatives $M_i$.  
(b) Implement the Thomas algorithm for tridiagonal systems.  
(c) Test on $f(x) = \sin(\pi x)$ with 10 equispaced nodes. Plot the spline and its error.

**Exercise 4 ★★ — Vandermonde Conditioning and QR**  
Show that QR decomposition is numerically superior to the normal equations for polynomial fitting.  
(a) Construct the Vandermonde matrix for $n = 15$ equispaced nodes. Compute $\kappa(V)$ and $\kappa(V^\top V)$.  
(b) Fit a degree-15 polynomial to noisy data by: (i) normal equations, (ii) QR factorization.  
(c) Compare the residuals and coefficient sensitivity to noise.

**Exercise 5 ★★ — Chebyshev Approximation**  
Compute the Chebyshev series and compare it to the minimax polynomial.  
(a) Compute the first 20 Chebyshev coefficients of $f(x) = |x|$ using the DCT.  
(b) Show that the truncated Chebyshev series achieves near-minimax approximation.  
(c) For $n = 10$, plot: exact $f$, Chebyshev approximation, and best polynomial approximation. Compute the difference in $\infty$-norm.

**Exercise 6 ★★ — FFT and the Convolution Theorem**  
Use the FFT to compute a convolution efficiently.  
(a) Implement a 1D discrete convolution via FFT: `fft_convolve(x, h)`.  
(b) Verify correctness against `np.convolve(x, h)` for a Gaussian filter.  
(c) Measure and compare the time complexity of FFT vs direct convolution for $n = 1000, 10000$.

**Exercise 7 ★★ — RBF Interpolation and Kernel Ridge Regression**  
Connect RBF interpolation to kernel methods.  
(a) Implement Gaussian RBF interpolation for 1D scattered data.  
(b) Show that as the regularization $\lambda \to 0$, kernel ridge regression approaches RBF interpolation.  
(c) Demonstrate the shape parameter effect: for $\varepsilon = 0.1, 1, 10$, plot the RBF interpolant and the condition number of $\Phi$.

**Exercise 8 ★★★ — Positional Encoding Interpolation**  
Analyze sinusoidal positional encodings from an approximation theory perspective.  
(a) Implement transformer sinusoidal PEs for dimension $d = 64$.  
(b) Show that $\text{PE}(pos + k)$ is a linear function of $\text{PE}(pos)$ (derive the rotation matrix).  
(c) Simulate context length extrapolation: train a simple model on positions $\{0,\ldots,511\}$ and evaluate on $\{512,\ldots,2047\}$. Compare original PEs, linear interpolated PEs, and NTK-scaled PEs.

---

## 11. Why This Matters for AI (2026 Perspective)

| Concept | AI/LLM Impact |
|---|---|
| Chebyshev nodes and minimax | ChebNet spectral GNNs avoid explicit eigendecomposition; used in molecular property prediction |
| Cubic splines | Neural ODEs use spline interpolation for continuous-time dynamics; used in physics-informed networks |
| B-splines | KAN networks (2024) parameterize learnable functions as B-splines; emerging alternative to MLPs |
| FFT and convolution theorem | FNet, S4, Hyena use FFT-based long-range mixing as efficient attention alternative ($O(n\log n)$) |
| Random Fourier features | Performer achieves $O(n)$ attention via kernel approximation; NTK analysis of neural nets |
| Sinusoidal interpolation | RoPE positional encodings in LLaMA, Mistral, GPT-4 class models |
| Context length extrapolation | YaRN, LongRoPE interpolate position indices to extend 4K-trained models to 128K+ tokens |
| Gaussian processes | Bayesian optimization for hyperparameter search in AutoML; uncertainty quantification in autonomous vehicles |
| RKHS and representer theorem | Foundation for kernel SVMs, support vector regression, and theoretical analysis of neural networks |
| Universal approximation | Theoretical basis for depth/width scaling laws; understanding which architectures can represent what functions |

---

## 12. Conceptual Bridge

**Looking back:** This section builds on floating-point arithmetic (§10-01) — all numerical interpolation is subject to rounding, and the condition numbers of Vandermonde and RBF matrices determine how much rounding error is amplified. From numerical linear algebra (§10-02), we use QR decomposition for stable least-squares fitting and tridiagonal solvers for spline construction. The optimization perspective (§10-03) underlies minimax approximation (Remez algorithm) and least-squares fitting. Earlier linear algebra (§02) provided the theory of least squares, orthogonality, and projections.

**Looking forward:** The next section (§10-05: Numerical Integration) uses interpolation directly — quadrature rules are derived by integrating interpolating polynomials. Gaussian quadrature nodes are the zeros of orthogonal polynomials. Clenshaw-Curtis quadrature uses Chebyshev nodes and the DCT for efficient weight computation. Everything built here — Chebyshev theory, orthogonal polynomials, FFT — feeds directly into numerical integration.

The connection to the broader curriculum:
- **Graph theory** (§11): Spectral graph convolutions approximate the spectral filter via Chebyshev polynomials in the graph Laplacian eigenvalues
- **Probability** (§07): Gaussian processes are probabilistic interpolants; Fourier analysis of stochastic processes uses spectral density
- **Optimization** (§08): The minimax approximation problem is solved by Chebyshev equioscillation, connecting to constrained optimization and duality

```
INTERPOLATION AND APPROXIMATION IN THE CURRICULUM
════════════════════════════════════════════════════════════════════════

  §10-01 Floating-Point      §02 Linear Algebra
  (rounding errors)    ──►  (least squares, QR)
           │                        │
           └────────────┬───────────┘
                        ▼
              §10-04 INTERPOLATION
              AND APPROXIMATION
              ┌──────────────────┐
              │ Lagrange / Newton│
              │ Chebyshev / Runge│
              │ Cubic splines    │
              │ Least squares    │
              │ FFT / DFT        │
              │ RBF / RKHS       │
              └────────┬─────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
    §10-05          §11 GNNs      AI Apps
  Numerical        (ChebNet,    (RoPE, KAN,
  Integration      SpectralGCN)  Performer)

════════════════════════════════════════════════════════════════════════
```

**Key insight:** The choice of basis is the choice of inductive bias. Monomials assume no structure; Chebyshev assumes bounded derivatives; splines assume local smoothness; trigonometric functions assume periodicity; RBFs assume isotropy. Neural networks *learn* their basis from data — but the theoretical guarantees come from understanding which function classes can be represented and approximated efficiently.

---

## Appendix A: Barycentric Lagrange Interpolation — Algorithm and Analysis

The **barycentric form** is the preferred implementation of polynomial interpolation.

### A.1 Derivation

Define the **node polynomial** $\ell(x) = \prod_{j=0}^{n}(x - x_j)$ and **barycentric weights** $w_j = 1/\prod_{k \neq j}(x_j - x_k)$.

Then $\ell_j(x) = \ell(x) \cdot w_j / (x - x_j)$ and:

$$p(x) = \ell(x) \sum_{j=0}^{n} \frac{w_j}{x - x_j} y_j$$

Since $\sum_j \ell_j(x) = 1$ implies $\ell(x) \sum_j w_j/(x-x_j) = 1$, dividing:

$$p(x) = \frac{\sum_{j=0}^{n} \frac{w_j}{x - x_j} y_j}{\sum_{j=0}^{n} \frac{w_j}{x - x_j}}$$

### A.2 Numerical Stability

**Theorem (Higham 2004):** The barycentric formula is backward stable: it computes the exact interpolant of slightly perturbed data $\tilde{y}_j = y_j (1 + \delta_j)$ with $|\delta_j| \leq O(\varepsilon_{\text{mach}} n)$.

**Practical implications:**
1. Pre-compute weights $w_j$ once: $O(n^2)$ preprocessing
2. Evaluate at any $x$: $O(n)$ per evaluation
3. Adding a new node $(x_{n+1}, y_{n+1})$: $O(n)$ update to weights
4. No division by $\ell(x)$ — the denominator is the normalizing sum, never computed separately

### A.3 Barycentric Weights for Special Nodes

**Equispaced nodes** on $[-1,1]$: $w_j = (-1)^j \binom{n}{j}$ — grows like $2^n$.

**Chebyshev nodes of the 2nd kind** $x_j = \cos(j\pi/n)$: $w_j = (-1)^j \delta_j$ where $\delta_0 = \delta_n = 1/2$, $\delta_j = 1$ otherwise.

**Chebyshev nodes of the 1st kind** $x_j = \cos((2j+1)\pi/(2n+2))$: $w_j = (-1)^j \sin((2j+1)\pi/(2n+2))$.

The Chebyshev weights are computed in $O(n)$ and are moderate in size — explaining their superior numerical properties.

```python
def barycentric_weights_cheb2(n):
    """Barycentric weights for Chebyshev points of the 2nd kind."""
    w = np.ones(n + 1)
    w[0] = 0.5; w[n] = 0.5
    w[1::2] = -1
    return w

def barycentric_eval(x, nodes, values, weights):
    """Evaluate barycentric interpolant at x."""
    diffs = x - nodes
    # Handle case x == node exactly
    exact = np.where(diffs == 0)[0]
    if len(exact) > 0:
        return values[exact[0]]
    w_over_d = weights / diffs
    return (w_over_d @ values) / w_over_d.sum()
```

---

## Appendix B: Divided Difference Properties and Connections

### B.1 The Divided Difference as a Derivative

**Theorem:** If $f \in C^n[a,b]$ and $x_0, \ldots, x_n \in [a,b]$ are distinct, then:

$$[y_0, y_1, \ldots, y_n] = \frac{f^{(n)}(\xi)}{n!}$$

for some $\xi$ in the convex hull of $\{x_0, \ldots, x_n\}$.

**Proof:** Apply Rolle's theorem $n$ times, as in the interpolation error theorem. The $n$-th divided difference telescopes to $f^{(n)}(\xi)/n!$. $\square$

**Corollary:** For equally spaced nodes $x_j = x_0 + jh$:

$$[y_0, y_1, \ldots, y_n] = \frac{\Delta^n y_0}{n! h^n}$$

where $\Delta^n$ is the $n$-th **forward difference operator**: $\Delta^0 y_j = y_j$, $\Delta y_j = y_{j+1} - y_j$, $\Delta^n y_j = \Delta(\Delta^{n-1} y_j)$.

### B.2 Newton's Forward Difference Formula

For equispaced nodes $x_j = x_0 + jh$, with $x = x_0 + sh$:

$$p_n(x) = \sum_{k=0}^{n} \binom{s}{k} \Delta^k y_0$$

where $\binom{s}{k} = s(s-1)\cdots(s-k+1)/k!$ is the generalized binomial coefficient. This is Newton's **forward difference interpolation formula** — the precursor to finite difference schemes in PDE solvers.

### B.3 Connection to Finite Difference Schemes

The divided difference is exactly the finite difference approximation to the derivative:

$$[y_0, y_1] = \frac{y_1 - y_0}{x_1 - x_0} \approx f'(x_0), \quad [y_0, y_1, y_2] = \frac{[y_1,y_2] - [y_0,y_1]}{x_2 - x_0} \approx \frac{f''(\xi)}{2}$$

For equispaced nodes: $[y_0, y_1] = \frac{\Delta y_0}{h} \approx f'(x_0) + O(h)$ (first-order forward difference).

**For AI:** Numerical differentiation (§10-03) uses finite differences — they are divided differences with equispaced nodes. The second divided difference gives the second derivative approximation needed for Newton's method and Hessian-vector products.

---

## Appendix C: Chebyshev Polynomial Deep Dive

### C.1 Chebyshev Expansion Coefficients via DCT

The Chebyshev coefficients of $f$ on $[-1,1]$:

$$c_k = \frac{2}{\pi} \int_0^\pi f(\cos\theta) \cos(k\theta) d\theta$$

can be computed via the **Discrete Cosine Transform (DCT)** at the Chebyshev nodes:

$$c_k \approx \frac{2}{n} \sum_{j=0}^{n-1} f(\cos\theta_j) \cos(k\theta_j), \quad \theta_j = \frac{\pi(2j+1)}{2n}$$

This is a DCT-II computation, achievable in $O(n \log n)$ using FFT.

**Algorithm:**
1. Evaluate $f$ at $n$ Chebyshev nodes
2. Apply DCT-II (via FFT: reflect and take FFT, or use `scipy.fft.dct`)
3. Normalize: $c_0 \leftarrow c_0/n$, $c_k \leftarrow c_k/n$ for $k > 0$

### C.2 Clenshaw Recurrence for Evaluation

To evaluate $S(x) = \sum_{k=0}^n c_k T_k(x)$ without computing each $T_k(x)$:

```
b[n+2] = 0, b[n+1] = 0
for k = n, n-1, ..., 1:
    b[k] = 2x * b[k+1] - b[k+2] + c[k]
result = x * b[1] - b[2] + c[0]
```

This is the **Clenshaw algorithm** — stable $O(n)$ evaluation of Chebyshev series.

### C.3 Chebyshev Approximation Convergence Rates

For functions with various smoothness:

| Smoothness | Convergence of $c_k$ | Convergence of best approx $E_n$ |
|---|---|---|
| Analytic in ellipse $E_\rho$ | $|c_k| \leq C\rho^{-k}$ | $E_n \leq C\rho^{-n}$ (exponential) |
| $f \in C^p[-1,1]$ | $|c_k| \leq C k^{-p}$ | $E_n \leq C n^{-p}$ (algebraic) |
| Lipschitz continuous | $|c_k| \leq C k^{-1}$ | $E_n \leq C n^{-1} \log n$ |
| Discontinuous | $|c_k| \leq C k^{-1}$ (Gibbs) | $E_n$ does not converge uniformly |

**For AI:** The exponential convergence for analytic functions is why spectral methods achieve machine precision with far fewer points than finite differences. This matters for PINN solvers — use spectral collocation (Chebyshev) not finite differences when the solution is smooth.

---

## Appendix D: B-Spline Implementation Details

### D.1 De Boor's Algorithm

Evaluating a B-spline $S(x) = \sum_i c_i B_{i,p}(x)$ at a point $x$ in knot interval $[t_j, t_{j+1})$:

```
d[i] = c[i]  for i = j-p, j-p+1, ..., j

for r = 1, 2, ..., p:
    for i = j, j-1, ..., j-p+r:
        alpha = (x - t[i]) / (t[i+p-r+1] - t[i])
        d[i] = (1-alpha) * d[i-1] + alpha * d[i]

result = d[j]
```

This is the **de Boor algorithm** — $O(p^2)$ per evaluation, numerically stable.

### D.2 Knot Vectors

The knot vector $T = (t_0 \leq t_1 \leq \cdots \leq t_{n+p+1})$ controls the spline:
- **Uniform:** $t_i = i/n$ — equal spacing, best for smooth periodic data
- **Clamped/open:** $t_0 = \cdots = t_p = 0$, $t_{n+1} = \cdots = t_{n+p+1} = 1$ — spline passes through first and last control points
- **Multiple knots:** Repeating $t_i = t_{i+1}$ reduces continuity (degree $p$ knot = discontinuous $(p-1)$-th derivative)

### D.3 B-Splines in KAN Architecture

KolmogorovArnold Networks (KAN, Liu et al. 2024) represent each network edge $\phi_{l,i,j}$ as a learnable B-spline:

$$\phi(x) = w_b \cdot b(x) + w_s \cdot \text{spline}(x)$$

where $b(x) = x/(1+e^{-x})$ (SiLU) is a base function and $\text{spline}(x) = \sum_i c_i B_i(x)$ is a B-spline with learnable coefficients $c_i$ and optionally adaptive grid.

**Training:** The grid is fixed initially; coefficients $c_i$ are learned by gradient descent. Periodically, the grid is updated to better cover the range of activations — **grid extension**.

**Advantages over MLP:** KANs can represent functions with simple structure (e.g., $f(x,y) = \sin(x+y)$) with far fewer parameters than MLPs. The spline representation is interpretable — you can read off the learned function shape.

---

## Appendix E: Spline Interpolation — Error Analysis and Conditioning

### E.1 Cubic Spline Error Bounds

**Theorem (Hall-Meyer, 1976):** Let $f \in C^4[a,b]$ and $S$ be the not-a-knot cubic spline. Then:

$$\|f^{(k)} - S^{(k)}\|_\infty \leq C_k h^{4-k} \|f^{(4)}\|_\infty, \quad k = 0, 1, 2, 3$$

with constants $C_0 = 5/384$, $C_1 = 1/24$, $C_2 = 3/8$, $C_3 = O(1/h)$ (for the derivative).

**Practical significance:**
- Halving the mesh spacing $h$ reduces position error by $2^4 = 16$
- The $O(h^4)$ convergence is "super-convergent" — better than the $O(h^2)$ one might expect from a piecewise cubic

### E.2 Condition Number of the Spline System

The tridiagonal matrix for cubic splines has entries in $[h_{\min}, h_{\max}]$ range. For equispaced nodes with $h_i = h$:

$$A = h \begin{pmatrix} 4 & 1 & & \\ 1 & 4 & 1 & \\ & \ddots & \ddots & \ddots \\ & & 1 & 4 \end{pmatrix} / 3$$

This is a diagonally dominant tridiagonal matrix with $\kappa(A) = O(1)$ — the condition number is bounded independent of $n$! This is why cubic spline computation is numerically stable.

### E.3 Why Natural Splines Minimize Curvature

**Proof of minimum curvature property:** Let $g$ be any twice-differentiable interpolant. Write $g = S + h$ where $S$ is the natural cubic spline. Then $h(x_i) = 0$ for all nodes, and $S''(a) = S''(b) = 0$.

$$\int_a^b (g'')^2 dx = \int_a^b (S'')^2 dx + 2\int_a^b S'' h'' dx + \int_a^b (h'')^2 dx$$

The cross term: integrate by parts twice, using $S'''' = 0$ on each interval (cubic), $h(x_i) = 0$ at nodes, and $S''(a) = S''(b) = 0$. The cross term vanishes, and $\int (h'')^2 dx \geq 0$, so $\int (g'')^2 dx \geq \int (S'')^2 dx$. $\square$

---

## Appendix F: The Fast Fourier Transform — Detailed Analysis

### F.1 Cooley-Tukey Algorithm Complexity

The radix-2 FFT recursion $T(n) = 2T(n/2) + cn$ with $T(1) = 0$:
- Unrolling: $T(n) = cn \log_2 n$ multiplications and additions
- For $n = 2^{20} \approx 10^6$: $\sim 20 \times 10^6$ operations vs $\sim 10^{12}$ for naive DFT

**General $n$:** When $n = p_1^{a_1} \cdots p_k^{a_k}$, FFT achieves $O(n(a_1 + \cdots + a_k)) = O(n \log n)$. For prime $n$, Rader's algorithm reduces to a convolution of size $n-1$. The "FFT is fast" claim requires $n$ with small prime factors — choose $n = 2^k$ in practice.

### F.2 Numerical Issues in FFT

**Rounding error:** The FFT error is $O(\log_2 n \cdot \varepsilon_{\text{mach}})$ — logarithmic growth, much better than direct DFT's $O(n \varepsilon_{\text{mach}})$.

**Planck's convention vs engineering convention:** The sign in $e^{\pm 2\pi i jk/n}$ differs between conventions. NumPy uses $e^{-2\pi i jk/n}$ in `np.fft.fft` (physics/engineering convention).

**Bit-reversal permutation:** The in-place Cooley-Tukey FFT requires reordering inputs by bit-reversal. For output-order $k$, the input index is the bit-reversal of $k$ in $\log_2 n$ bits.

### F.3 FFT Applications in Deep Learning

**Spectral convolution (S4, Hyena):** The state-space model layer computes:

$$y = u * h$$

where $h$ is a learned convolutional kernel. Using FFT:

$$y = \text{IFFT}(\text{FFT}(u) \odot \text{FFT}(h))$$

Complexity: $O(L \log L)$ vs $O(L^2)$ for direct attention. For sequence length $L = 8192$, this is a $\sim 10\times$ speedup.

**FNet (Lee-Thorp et al., 2021):** Replace self-attention with 2D FFT:

```python
x = torch.fft.fft2(x, norm='ortho').real
```

Unparameterized — no learned weights — yet achieves 92-97% of BERT accuracy on GLUE. Demonstrates that long-range mixing doesn't require learned attention patterns.

---

## Appendix G: Radial Basis Functions — Stability and Algorithms

### G.1 The Shape Parameter Dilemma

For Gaussian RBFs $\phi(r) = e^{-\varepsilon^2 r^2}$, the interpolation matrix $\Phi_{ij} = e^{-\varepsilon^2 \|x_i - x_j\|^2}$:
- **Large $\varepsilon$ (narrow):** $\Phi \approx I$ (well-conditioned), but poor approximation quality
- **Small $\varepsilon$ (flat):** Near-flat Gaussians, $\kappa(\Phi) \sim e^{n\varepsilon^{-2}}$ (catastrophically ill-conditioned)

The **uncertainty principle for RBFs:** You cannot simultaneously have a well-conditioned system and high approximation accuracy. This is Schaback's uncertainty principle.

**Solutions:**
1. **Regularized RBF:** Minimize $\sum_i (s(x_i) - y_i)^2 + \lambda \|s\|_{\mathcal{H}}^2$ — ridge regression with RBF kernel
2. **Compactly supported RBFs (Wendland):** Sparse $\Phi$, cheap to solve, but less smooth
3. **Contour-Padé algorithm:** Reformulate using contour integration, avoiding catastrophic cancellation

### G.2 Matérn Kernels and Sobolev Spaces

The **Matérn kernel** of order $\nu$ corresponds to the RKHS being the Sobolev space $H^{\nu + d/2}(\mathbb{R}^d)$:

$$k_\nu(r) = \frac{2^{1-\nu}}{\Gamma(\nu)} \left(\frac{\sqrt{2\nu} r}{\ell}\right)^\nu K_\nu\left(\frac{\sqrt{2\nu} r}{\ell}\right)$$

where $K_\nu$ is the modified Bessel function. Special cases:
- $\nu = 1/2$: $k(r) = e^{-r/\ell}$ (exponential, $C^0$ functions)
- $\nu = 3/2$: $k(r) = (1 + \sqrt{3}r/\ell)e^{-\sqrt{3}r/\ell}$ ($C^2$ functions)
- $\nu = 5/2$: $k(r) = (1 + \sqrt{5}r/\ell + 5r^2/(3\ell^2))e^{-\sqrt{5}r/\ell}$ ($C^4$ functions)
- $\nu \to \infty$: Gaussian ($C^\infty$ functions)

**For ML:** The Matérn-5/2 kernel is the default in most GP libraries (scikit-learn, GPyTorch) because it provides enough smoothness without the extreme sensitivity of the Gaussian kernel.

### G.3 Sparse Gaussian Processes

**Inducing point methods:** For $n$ data points and $m \ll n$ inducing points $\mathbf{z}$:

$$p(f | \mathbf{y}) \approx q(f) = \int p(f | \mathbf{u}) q(\mathbf{u}) d\mathbf{u}$$

where $\mathbf{u} = f(\mathbf{z})$ are function values at inducing points. Complexity: $O(nm^2)$ instead of $O(n^3)$.

**Variational inducing methods (SVGP, Hensman et al.):** Optimize inducing point locations jointly with model parameters via variational lower bound. Enables stochastic mini-batch training for GPs — analogous to stochastic gradient descent for neural networks.

---

## Appendix H: Fourier Features and Neural Tangent Kernel

### H.1 Bochner's Theorem and Random Fourier Features

**Bochner's theorem:** A continuous, shift-invariant kernel $k(x,y) = k(x-y)$ is positive definite if and only if it is the Fourier transform of a non-negative measure $p(\omega)$:

$$k(\Delta) = \int e^{i\omega^\top \Delta} p(\omega) d\omega$$

**Random Fourier features** approximate this as a Monte Carlo integral:

$$k(x, y) \approx z(x)^\top z(y), \quad z(x) = \sqrt{2/D}[\cos(\omega_1^\top x + b_1), \ldots, \cos(\omega_D^\top x + b_D)]$$

where $\omega_j \sim p(\omega)$ and $b_j \sim \text{Uniform}[0, 2\pi]$.

**Convergence:** With $D$ random features, the approximation error satisfies:

$$\sup_{x,y} |k(x,y) - z(x)^\top z(y)| = O(1/\sqrt{D})$$

with high probability.

### H.2 The Neural Tangent Kernel

An infinite-width neural network trained by gradient descent with squared loss implements **kernel regression** with the **neural tangent kernel (NTK)**:

$$k_{\text{NTK}}(x, x') = \mathbb{E}_{\theta \sim \text{init}} \left[\nabla_\theta f(x;\theta)^\top \nabla_\theta f(x';\theta)\right]$$

**Key results:**
1. At initialization, $k_{\text{NTK}}$ is fixed (independent of $\theta$)
2. For infinite width, $k_{\text{NTK}}$ stays constant throughout training
3. The trained model converges to the kernel regression solution: $f(x) = k_{\text{NTK}}(x, X)(k_{\text{NTK}}(X,X))^{-1} y$

**Implications:**
- Infinite-width networks are *not* superior to kernel SVMs (they implement kernel regression)
- The expressiveness advantage of neural networks comes from finite width and feature learning
- NTK theory breaks down for finite-width networks with large learning rates (the "feature learning regime")

**For AI:** The NTK framework explains:
- Why overparameterized networks can fit any training data (kernel interpolation)
- Why neural scaling laws exist (larger networks have richer NTK kernels)
- The transition from "kernel regime" to "feature learning regime" in modern LLM training

---

## Appendix I: Practical Implementation Guide

### I.1 Python Ecosystem for Interpolation

```python
from scipy import interpolate

# Lagrange/Barycentric interpolation
interp = interpolate.BarycentricInterpolator(nodes, values)
y_new = interp(x_new)

# Cubic spline
cs = interpolate.CubicSpline(x, y, bc_type='natural')
y_pred = cs(x_new)
y_deriv = cs(x_new, 1)  # First derivative

# B-spline fitting
from scipy.interpolate import make_interp_spline
bspline = make_interp_spline(x, y, k=3)  # cubic
y_pred = bspline(x_new)

# Smoothing spline (least-squares B-spline)
tck = interpolate.splrep(x, y, s=0.1)  # s = smoothing factor
y_pred = interpolate.splev(x_new, tck)

# RBF interpolation
rbf = interpolate.RBFInterpolator(X_2d, y, kernel='gaussian', epsilon=1.0)
y_new = rbf(X_test)
```

### I.2 NumPy/SciPy FFT

```python
import numpy as np
from scipy.fft import fft, ifft, fftfreq, dct, idct

# FFT of a real signal
n = 1024
t = np.linspace(0, 1, n, endpoint=False)
y = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*13*t)

Y = fft(y)
freqs = fftfreq(n, d=1/n)  # Frequencies in cycles/unit

# Power spectral density
psd = np.abs(Y)**2 / n

# Inverse FFT
y_recovered = np.real(ifft(Y))

# DCT for Chebyshev coefficients
from scipy.fft import dct
coeffs = dct(f_values, type=2, norm='ortho') / n  # Normalized
```

### I.3 Chebyshev Interpolation with Chebfun-style Code

```python
def cheb_nodes(n, a=-1, b=1):
    """Chebyshev nodes of the 2nd kind on [a,b]."""
    k = np.arange(n+1)
    x = np.cos(np.pi * k / n)  # on [-1, 1]
    return (a + b)/2 + (b - a)/2 * x

def cheb_coeffs(f_values):
    """Chebyshev expansion coefficients via DCT."""
    n = len(f_values) - 1
    from scipy.fft import dct
    c = dct(f_values[::-1], type=1) / n  # DCT-I
    c[0] /= 2; c[-1] /= 2
    return c

def cheb_eval(c, x):
    """Clenshaw algorithm for Chebyshev series evaluation."""
    n = len(c) - 1
    b2, b1 = 0.0, 0.0
    for k in range(n, 0, -1):
        b2, b1 = b1, 2*x*b1 - b2 + c[k]
    return x*b1 - b2 + c[0]
```

---

## Appendix J: Worked Examples — Interpolation in Practice

### J.1 Fitting a Binding Energy Curve

In computational chemistry, the binding energy $E(r)$ as a function of inter-atomic distance $r$ is often known at a few points from DFT calculations. We want a smooth interpolant for geometry optimization.

```
Problem:  r (Å):  1.8   2.0   2.2   2.5   3.0   4.0
          E (eV): -2.5  -3.1  -3.2  -3.0  -2.7  -2.3

Method: Cubic spline with not-a-knot BC
Result: Minimum at r* ≈ 2.24 Å, E* ≈ -3.21 eV
Derivatives: E'(r*) ≈ 0 (equilibrium), E''(r*) > 0 (stable)
```

This is used in molecular dynamics force fields and materials design ML models.

### J.2 Positional Encoding Extrapolation

**Problem:** A model trained with sinusoidal PEs on positions $\{0, \ldots, 511\}$ needs to handle position 1024.

**Analysis:** At dimension $i$, the PE period is $T_i = 2\pi \cdot 10000^{2i/d}$:
- Small $i$ (low freq): $T_0 = 2\pi \approx 6.3$ — position 1024 has been seen many times
- Large $i$ (high freq): $T_{d-1} = 2\pi \times 10000 \approx 62832$ — period >> 512, position 1024 lies in same "first cycle"

**YaRN solution:** Scale position indices by $s = L'/L$ (training length / target length):

$$\text{PE}_{\text{YaRN}}(pos, i) = \text{PE}(pos \cdot s, i)$$

This maps position 1024 in the target context to position $1024 \times (512/1024) = 512$ in the training range — guaranteed to be seen. At the cost of reduced resolution at low frequencies.

### J.3 Kernel Method vs Neural Network

For a 1D regression problem with $n = 100$ points from $f(x) = \sin(3x) + 0.1\varepsilon$:

| Method | Parameters | Training error | Test error | Time |
|---|---|---|---|---|
| Degree-5 polynomial (QR) | 6 | 0.09 | 0.11 | $<$1ms |
| Cubic spline (natural) | 100 | 0.0 (exact) | 0.10 | $<$1ms |
| Gaussian RBF ($\varepsilon=1$) | 100 | 0.0 (exact) | 0.09 | 10ms |
| Gaussian RBF + ridge ($\lambda=10^{-3}$) | 100 | 0.008 | **0.09** | 10ms |
| 2-layer MLP (100 units) | 10200 | 0.002 | 0.10 | 1s |

For smooth 1D functions with $n < 1000$, splines and kernel methods outperform neural networks in accuracy and efficiency. Neural networks win for high-dimensional data, compositional structure, and when priors about the function class are unknown.

---

## Appendix K: Connections to Other Chapters

| Topic | Connection | Reference |
|---|---|---|
| Eigenvalue decomposition | Spectral methods diagonalize differentiation/integration operators | §03-Advanced-Linear-Algebra |
| Fourier series | Spectral analysis of signals, convolutional layers | §07-Probability (spectral density) |
| Taylor series | Polynomial approximation of smooth functions | §01-Mathematical-Foundations |
| Optimization | Remez algorithm, RKHS optimization (representer theorem) | §08-Optimization, §10-03 |
| Graph theory (spectral GNNs) | Chebyshev polynomial filters on graph Laplacian | §11-Graph-Theory |
| Numerical integration | Gaussian quadrature uses zeros of orthogonal polynomials | §10-05 |
| Floating-point | Vandermonde ill-conditioning; Chebyshev stability | §10-01 |
| Linear algebra | QR for least squares; tridiagonal solver for splines | §10-02 |

---

## Appendix L: Lebesgue Constants and Interpolation Stability

### L.1 Definition and Computation

The **Lebesgue constant** for a set of nodes $\{x_0, \ldots, x_n\}$ is:

$$\Lambda_n = \max_{x \in [a,b]} \sum_{j=0}^{n} |\ell_j(x)|$$

It measures the worst-case amplification of errors:

$$\|p_n\|_\infty \leq \Lambda_n \|y\|_\infty, \quad \|f - p_n\|_\infty \leq (1 + \Lambda_n) E_n^*$$

where $E_n^* = \inf_{p \in \mathcal{P}_n} \|f - p\|_\infty$ is the best approximation error.

**Growth rates:**

| Node set | $\Lambda_n$ growth |
|---|---|
| Equispaced on $[-1,1]$ | $\sim 2^n / (e n \ln n)$ (exponential) |
| Chebyshev points (1st kind) | $\leq \frac{2}{\pi} \ln(n+1) + 1$ (logarithmic) |
| Chebyshev points (2nd kind) | $\leq \frac{2}{\pi} \ln(n) + 0.974$ (logarithmic) |
| Optimal nodes | $\geq \frac{2}{\pi} \ln(n+1) - 1$ (optimal up to constant) |
| Gauss-Legendre nodes | $O(\log n)$ |

**Practical consequence:** At $n = 30$, equispaced $\Lambda_{30} \approx 10^7$ while Chebyshev $\Lambda_{30} \approx 4.3$. An error of $10^{-6}$ in the data values would cause a $\sim 10$ error in the interpolant with equispaced nodes but only $4 \times 10^{-6}$ with Chebyshev.

### L.2 Lebesgue Function Visualization

```
LEBESGUE FUNCTION |sum_j |ℓ_j(x)||  (n=10)
════════════════════════════════════════════════════════════════════════

  20 ─                    Equispaced nodes
     │                   ╱╲          ╱╲
  15 ─                  ╱  ╲        ╱  ╲
     │                 ╱    ╲      ╱    ╲
  10 ─       ╱╲       ╱      ╲╱╲ ╱
     │      ╱  ╲     ╱        ╲ ╱
   5 ─  ╱╲╱    ╲╱╲ ╱
     │╱╱            ╲╱╲
   1 ─────────────────────────────────── Chebyshev (≈2.0 everywhere)
     -1                                 1

  Equispaced Lebesgue function peaks near ±1 (endpoint Runge zones).
  Chebyshev function is nearly constant across [-1,1].

════════════════════════════════════════════════════════════════════════
```

---

## Appendix M: Multidimensional Interpolation

### M.1 Tensor Product Methods

For functions on $[a,b]^d$, a **tensor product** approach uses one-dimensional bases in each dimension:

$$p(x_1, \ldots, x_d) = \sum_{i_1=0}^{n} \cdots \sum_{i_d=0}^{n} c_{i_1,\ldots,i_d} \phi_{i_1}(x_1) \cdots \phi_{i_d}(x_d)$$

**Curse of dimensionality:** Requires $(n+1)^d$ basis functions — exponential in $d$. For $n=10$, $d=10$: $11^{10} \approx 2.6 \times 10^{10}$ coefficients.

**Sparse grids (Smolyak):** Use only tensor products of 1D bases where $i_1 + \cdots + i_d \leq n + d - 1$. Complexity: $O(n (\log n)^{d-1})$ — much better. Used in high-dimensional integration and interpolation (quantum chemistry, finance).

### M.2 Scattered Data in High Dimensions

For scattered data in $\mathbb{R}^d$ with $d > 3$, polynomial interpolation and tensor product methods fail. Alternatives:

**RBF interpolation:** $s(x) = \sum_j c_j \phi(\|x - x_j\|)$ — mesh-free, works for any $d$. But condition number grows exponentially with $n$ for Gaussian RBF.

**Nearest-neighbor interpolation:** $s(x) = y_{j^*}$ where $j^* = \arg\min_j \|x - x_j\|$. Discontinuous, $O(1)$ per query with KD-tree preprocessing.

**Inverse distance weighting (IDW):** $s(x) = \sum_j w_j(x) y_j / \sum_j w_j(x)$ where $w_j(x) = 1/\|x - x_j\|^p$. Continuous but not smooth.

**For AI:** High-dimensional function approximation is precisely the task of neural networks. The kernel trick (kernel regression) extends RBF to high dimensions through the kernel matrix $K_{ij} = k(x_i, x_j)$ — but computing and inverting this $n \times n$ matrix is $O(n^3)$.

### M.3 Triangulation-Based Methods

For scattered 2D data, **Delaunay triangulation** partitions the domain into non-overlapping triangles. Piecewise linear or cubic interpolation on each triangle:

- **Linear (barycentric):** $C^0$, $O(1)$ per evaluation after triangulation
- **Clough-Tocher cubic:** $C^1$, splits each triangle into 3 subtriangles

`scipy.interpolate.LinearNDInterpolator` and `CloughTocher2DInterpolator` implement these.

---

## Appendix N: Approximation Theory Connections to Deep Learning

### N.1 Width vs Depth in Approximation

**Shallow networks (width $N$, depth 1):**
$$\|f - f_N\|_{L^2} = O(N^{-2s/d})$$
for $f$ in the Sobolev space $W^{s,2}(\mathbb{R}^d)$ — polynomial rate, cursed by dimension.

**Deep ReLU networks:** For functions with compositional structure $f = f_1 \circ f_2 \circ \cdots \circ f_L$ where each $f_\ell: \mathbb{R}^{d_\ell} \to \mathbb{R}^{d_{\ell+1}}$ is $d_\ell$-dimensional:

$$\|f - f_N\|_{L^\infty} = O(N^{-2s/d_{\max}})$$

where $d_{\max} = \max_\ell d_\ell$ — the approximation rate depends on the **intrinsic dimension** of the composition, not the ambient dimension $d$.

**Implication:** Deep networks exploit compositional structure to avoid the curse of dimensionality. A function like $f(x) = \sin(x_1 + x_2) \cdot e^{x_3 x_4}$ has intrinsic dimension 2 even though $d = 4$ — deep networks can approximate it with $O(N^{-s})$ error.

### N.2 The Kolmogorov-Arnold Representation Theorem

**Theorem (Kolmogorov, 1957):** Every continuous function $f: [0,1]^n \to \mathbb{R}$ can be written as:

$$f(x_1, \ldots, x_n) = \sum_{q=0}^{2n} \Phi_q\left(\sum_{p=1}^n \phi_{q,p}(x_p)\right)$$

for continuous functions $\phi_{q,p}: [0,1] \to \mathbb{R}$ and $\Phi_q: \mathbb{R} \to \mathbb{R}$.

**Significance:** Every multivariate continuous function can be written using only univariate functions and addition. No curse of dimensionality in this representation.

**Practical issue:** The inner functions $\phi_{q,p}$ may be nowhere differentiable — impossible to learn efficiently with gradient methods. The theorem is an existence result, not a constructive algorithm.

**KAN circumvents this:** By restricting to learned B-splines (differentiable, finitely parameterized), KAN trades the full generality of Kolmogorov for a practical learnable version.

### N.3 Function Classes and Approximation Rates

```
APPROXIMATION RATE BY FUNCTION CLASS AND METHOD
════════════════════════════════════════════════════════════════════════

  Function class          Best method          Rate
  ─────────────────────   ──────────────────   ──────────────────
  Polynomial of deg ≤ k   Exact polynomial     0 (exact)
  Analytic in disk Dρ     Chebyshev truncation O(ρ⁻ⁿ) (exponential)
  Cᵖ on [a,b]             Poly degree n        O(n⁻ᵖ)
  Sobolev Wˢ'²(Ω)         Spline               O(hˢ)
  Barron class            Shallow network      O(N⁻¹/²) (dim-free!)
  Compositional           Deep network         O(N⁻²ˢ/d_max)
  Arbitrary continuous    Universal approx.    ε-approx, no rate

  N = number of neurons, n = polynomial degree, h = mesh size

════════════════════════════════════════════════════════════════════════
```

---

## Appendix O: Summary Reference Tables

### O.1 Interpolation Methods Comparison

| Method | Nodes | Error order | Stability | AI use |
|---|---|---|---|---|
| Lagrange (equispaced) | Any | $O(h^{n+1})$ — but large! | Poor (Runge) | Avoid |
| Lagrange (Chebyshev) | Chebyshev | $O(2^{-n})$ for analytic | Good ($\Lambda_n = O(\log n)$) | ChebNet |
| Newton divided diff. | Any | Same as Lagrange | Moderate | Incremental |
| Cubic spline (natural) | Equispaced | $O(h^4)$ | Excellent | Neural ODEs |
| B-spline | Arbitrary knots | $O(h^{p+1})$ | Excellent | KAN |
| Trigonometric | Equispaced | Exponential (smooth periodic) | Good | FFT, PE |
| RBF (Gaussian) | Scattered | Exponential (analytic) | Sensitive to $\varepsilon$ | Kernels, GP |
| Polynomial LS (QR) | Any | Best $L^2$ approx | Good | Feature eng. |

### O.2 Key Theorems Quick Reference

| Theorem | Statement | Location |
|---|---|---|
| Existence/uniqueness | Unique degree-$n$ interpolant for $n+1$ distinct nodes | §2.1 |
| Interpolation error | $f - p_n = f^{(n+1)}(\xi)\omega_{n+1}/(n+1)!$ | §2.3 |
| Chebyshev minimax | $T_n/2^{n-1}$ has smallest $\infty$-norm among monic degree-$n$ polys | §3.1 |
| Equioscillation | Best approx iff error equioscillates at $\geq n+2$ points | §3.3 |
| Spline min. curvature | Natural cubic spline minimizes $\int |S''|^2$ | §4.2 |
| Representer theorem | Regularized RKHS problem solved by finite expansion | §7.2 |
| Universal approximation | Shallow networks dense in $C([0,1]^d)$ | §8.2 |
| Nyquist theorem | Reconstruct $B$-band-limited signal from rate $\geq 2B$ samples | §6.4 |
| Bochner's theorem | PD shift-invariant kernels = FT of non-negative measures | §H.1 |
| Kolmogorov-Arnold | All $f \in C([0,1]^n)$ = composition of univariates | §N.2 |

---

## Appendix P: Extended Exercises with Solutions

### P.1 Barycentric Interpolation — Full Implementation

```python
import numpy as np

def cheb2_nodes(n, a=-1.0, b=1.0):
    """Chebyshev nodes of the 2nd kind (n+1 nodes) on [a,b]."""
    k = np.arange(n + 1)
    x = np.cos(np.pi * k / n)  # on [-1, 1]
    return 0.5*(a+b) + 0.5*(b-a)*x

def barycentric_weights(nodes):
    """Compute barycentric weights for given nodes."""
    n = len(nodes) - 1
    w = np.ones(n + 1)
    for j in range(n + 1):
        for k in range(n + 1):
            if k != j:
                w[j] *= (nodes[j] - nodes[k])
    return 1.0 / w

def cheb2_weights(n):
    """Barycentric weights for Chebyshev 2nd kind nodes (O(n) formula)."""
    w = np.ones(n + 1)
    w[0] = 0.5
    w[n] = 0.5
    w[1::2] = -1.0
    return w

def bary_eval(x, nodes, values, weights):
    """Evaluate barycentric interpolant at scalar x."""
    diffs = x - nodes
    mask = diffs == 0.0
    if np.any(mask):
        return values[mask][0]
    w_d = weights / diffs
    return np.dot(w_d, values) / w_d.sum()

# Example: interpolate f(x) = 1/(1+25x^2) with Chebyshev nodes
n = 20
nodes = cheb2_nodes(n)
f = lambda x: 1.0 / (1 + 25*x**2)
values = f(nodes)
weights = cheb2_weights(n)

x_test = np.linspace(-1, 1, 500)
y_interp = np.array([bary_eval(xi, nodes, values, weights) for xi in x_test])
y_exact = f(x_test)
print(f"Max error (Chebyshev n=20): {np.abs(y_interp - y_exact).max():.2e}")
```

### P.2 Cubic Spline — Full Implementation

```python
def natural_cubic_spline(x, y):
    """
    Compute natural cubic spline coefficients.
    Returns: (a, b, c, d) polynomial coefficients for each interval.
    """
    n = len(x) - 1
    h = np.diff(x)

    # Set up tridiagonal system for second derivatives M
    rhs = 6 * (np.diff(y[1:] / h[1:] - y[:-1] / h[:-1]))  # Simplified
    # More careful: second differences of function values
    dd = np.zeros(n - 1)
    for i in range(1, n):
        dd[i-1] = 6 * ((y[i+1] - y[i])/h[i] - (y[i] - y[i-1])/h[i-1])

    # Tridiagonal matrix: diagonal, sub/super diagonal
    diag = 2*(h[:-1] + h[1:])
    off = h[1:-1]

    # Thomas algorithm (tridiagonal solver)
    n_sys = n - 1
    c_diag = diag.copy()
    c_rhs = dd.copy()

    # Forward sweep
    for i in range(1, n_sys):
        m = off[i-1] / c_diag[i-1]
        c_diag[i] -= m * off[i-1]
        c_rhs[i] -= m * c_rhs[i-1]

    # Back substitution
    M = np.zeros(n + 1)  # Second derivatives at nodes
    M[n_sys] = c_rhs[n_sys-1] / c_diag[n_sys-1]
    for i in range(n_sys-2, -1, -1):
        M[i+1] = (c_rhs[i] - off[i] * M[i+2]) / c_diag[i]
    # M[0] = M[n] = 0 (natural BC)

    return M, h

def eval_spline(x_eval, x_nodes, y_nodes, M, h):
    """Evaluate natural cubic spline at x_eval."""
    i = np.searchsorted(x_nodes, x_eval, side='right') - 1
    i = np.clip(i, 0, len(x_nodes) - 2)
    t = x_eval - x_nodes[i]
    hi = h[i]
    S = (M[i] * (x_nodes[i+1] - x_eval)**3 / (6*hi) +
         M[i+1] * (x_eval - x_nodes[i])**3 / (6*hi) +
         (y_nodes[i] - M[i]*hi**2/6) * (x_nodes[i+1]-x_eval)/hi +
         (y_nodes[i+1] - M[i+1]*hi**2/6) * (x_eval-x_nodes[i])/hi)
    return S
```

### P.3 Chebyshev Spectral Differentiation Matrix

A key tool in spectral methods is the **differentiation matrix** $D$ such that $(Df)_j \approx f'(x_j)$ at Chebyshev nodes:

```python
def cheb_diff_matrix(n):
    """
    Chebyshev spectral differentiation matrix for n+1 nodes.
    D @ f_values approximates f' at Chebyshev nodes.
    """
    if n == 0:
        return np.array([[0.]])
    x = np.cos(np.pi * np.arange(n+1) / n)
    c = np.ones(n+1)
    c[0] = 2; c[n] = 2
    c *= (-1)**np.arange(n+1)

    D = np.zeros((n+1, n+1))
    for i in range(n+1):
        for j in range(n+1):
            if i != j:
                D[i, j] = c[i] / (c[j] * (x[i] - x[j]))

    # Diagonal: sum condition
    D -= np.diag(D.sum(axis=1))
    return D

# Example: differentiate f(x) = sin(pi*x) on [-1,1]
n = 32
D = cheb_diff_matrix(n)
x = np.cos(np.pi * np.arange(n+1) / n)
f_vals = np.sin(np.pi * x)
f_prime_approx = D @ f_vals
f_prime_exact = np.pi * np.cos(np.pi * x)
print(f"Spectral diff error: {np.abs(f_prime_approx - f_prime_exact).max():.2e}")
```

---

## Appendix Q: Advanced Topics in Function Approximation

### Q.1 Padé Approximants

**Definition:** A Padé approximant $[m/n](x)$ is a rational function $P_m(x)/Q_n(x)$ that agrees with the Taylor series of $f$ up to order $m+n$:

$$f(x) - \frac{P_m(x)}{Q_n(x)} = O(x^{m+n+1})$$

**Advantages over polynomials:**
- Can represent poles and asymptotic behavior
- Often much more accurate than same-degree Taylor polynomials
- Diagonal Padé $[n/n]$ often converges to analytic functions in larger domains

**Example:** The exponential function:
$$e^x \approx [2/2](x) = \frac{1 + x/2 + x^2/12}{1 - x/2 + x^2/12}$$

This is the foundation of Padé-based matrix exponential algorithms (used in ODE solvers and graph neural networks for diffusion operators).

### Q.2 Wavelet Approximation

**Wavelets** provide multi-resolution approximation: approximate a function at coarse scale first, then add fine-scale details.

The **Daubechies wavelet** basis $\{\psi_{j,k}(x) = 2^{j/2}\psi(2^j x - k)\}$ provides:
- **Orthogonal** decomposition
- **Local support:** $\psi_{j,k}$ has support $\sim 2^{-j}$ around $2^{-j}k$
- **Vanishing moments:** $\int x^p \psi(x) dx = 0$ for $p = 0, \ldots, p_0$
- **Adaptive sparsity:** Functions with local irregularities (edges, singularities) represented sparsely

**For AI:** Wavelets appear in:
- Image compression (JPEG 2000 uses wavelet decomposition)
- Multi-scale neural architectures (UNet uses hierarchical scales)
- Graph wavelets for multi-resolution graph signals

### Q.3 The NUFFT and Non-Uniform Sampling

The standard FFT assumes uniform sampling. For non-uniform samples $\{x_j\}$, the **Non-Uniform FFT (NUFFT)** computes:

$$F_k = \sum_{j=1}^{n} f_j e^{-2\pi i k x_j}, \quad k = -n/2, \ldots, n/2-1$$

in $O(n \log n)$ by:
1. Convolving point masses at $x_j$ with a Gaussian kernel onto a uniform grid
2. Applying FFT on the uniform grid
3. Deconvolving the Gaussian in frequency domain

**For AI:** NUFFT is used in:
- **MRI reconstruction:** k-space data is sampled along non-uniform trajectories
- **Molecular dynamics:** Computing electrostatic interactions via particle-mesh Ewald (PME)
- **Astronomy:** Radio telescope arrays measure non-uniform Fourier samples

---

## Appendix R: Reproducible Experiments

All code in this section is reproducible with `numpy`, `scipy`, and `matplotlib`. The key experiments:

```python
# Experiment 1: Runge's phenomenon comparison
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BarycentricInterpolator

f = lambda x: 1/(1+25*x**2)
x_plot = np.linspace(-1, 1, 500)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for n in [5, 10, 15]:
    # Equispaced
    nodes_eq = np.linspace(-1, 1, n+1)
    p_eq = BarycentricInterpolator(nodes_eq, f(nodes_eq))

    # Chebyshev
    nodes_ch = np.cos(np.pi * np.arange(n+1) / n)
    p_ch = BarycentricInterpolator(nodes_ch, f(nodes_ch))

    err_eq = np.abs(f(x_plot) - p_eq(x_plot)).max()
    err_ch = np.abs(f(x_plot) - p_ch(x_plot)).max()
    print(f"n={n}: equispaced error={err_eq:.2e}, Chebyshev error={err_ch:.2e}")

# Experiment 2: Lebesgue constants
for n in range(1, 20):
    nodes = np.cos(np.pi * np.arange(n+1) / n)  # Chebyshev 2nd kind
    x_test = np.linspace(-1, 1, 1000)
    # Compute Lebesgue function: sum of |ell_j(x)|
    weights = np.ones(n+1)
    weights[0] = 0.5; weights[n] = 0.5; weights[1::2] = -1.0
    lebesgue = []
    for x in x_test:
        diffs = x - nodes
        if np.any(diffs == 0):
            lebesgue.append(1.0)
        else:
            w_d = weights / diffs
            # Lagrange basis values: ell_j = w_j/diff_j / sum(w_k/diff_k)
            total = w_d.sum()
            lebesgue.append(np.abs(w_d / total).sum())
    print(f"n={n}: Lambda_n = {max(lebesgue):.4f}")
```

---

## Appendix S: Historical Notes on Interpolation

### S.1 Newton's Contribution

Isaac Newton introduced the method of divided differences in "Methodus Differentialis" (1711). His motivation was astronomical: computing planetary positions at arbitrary times from a table of observed positions at discrete times. The divided difference table allowed him to update the interpolant incrementally as new observations came in — an $O(n)$ update, remarkable for 1711.

### S.2 Chebyshev's Insight

Pafnuty Chebyshev (1853) asked: which monic polynomial of degree $n$ has the smallest maximum absolute value on $[-1,1]$? His answer — $T_n(x)/2^{n-1}$ — was the first systematic use of the minimax criterion in approximation theory. Chebyshev was motivated by mechanism design: he wanted to minimize the deviation from ideal motion in steam engine linkages.

The connection between the trigonometric definition $T_n(\cos\theta) = \cos(n\theta)$ and the polynomial form was noticed by Chebyshev himself — using the addition formula for cosines iteratively.

### S.3 Runge's Warning

Carl Runge published his famous counterexample in 1901. The function $1/(1+25x^2)$ is analytic everywhere in the complex plane except at $\pm i/5$ — which lie inside the unit circle. This means the function's Taylor series has radius of convergence $1/5 < 1$, and polynomial interpolation at equispaced nodes tries to capture behavior that is "barely analytic" in the relevant region.

Runge's result was important as a warning against increasing polynomial degree. The modern response (Chebyshev nodes) came gradually through the 20th century.

### S.4 Cooley-Tukey and the FFT

The Cooley-Tukey FFT algorithm (1965) is often cited as one of the most important algorithms of the 20th century. Its impact was immediate: it made digital signal processing practical, enabled the digital revolution in audio/communications, and later became the backbone of scientific computing.

Remarkably, the core idea (exploiting DFT factorizations) was known to Gauss in 1805 — unpublished in his collected works. The algorithm was independently rediscovered multiple times before Cooley and Tukey's landmark paper.

### S.5 The Representer Theorem and RKHS

The representer theorem (Kimeldorf-Wahba, 1971) established that regularized function estimation over an RKHS has a finite-dimensional solution — transforming an infinite-dimensional optimization into a finite linear algebra problem. This unified spline smoothing, kernel regression, and support vector machines under one mathematical framework. The same theorem, applied to the neural tangent kernel, explains why overparameterized neural networks trained by gradient descent converge to global minima that generalize well.

---

## Appendix T: Rational Approximation and Padé Theory

### T.1 Padé Table

For a function with Taylor series $f(x) = \sum_{k=0}^\infty c_k x^k$, the Padé approximant $[m/n]$ satisfies:

$$f(x) \cdot Q_n(x) - P_m(x) = O(x^{m+n+1})$$

The Padé table arranges these approximants in a grid by $(m, n)$:

```
PADÉ TABLE for e^x
════════════════════════════════════════════════════════════════════════

  n\m │  0          1           2
  ────┼──────────────────────────────────────────────────────
   0  │  1          1+x         1+x+x²/2
   1  │  1/(1-x)   (2+x)/(2-x)   (6+2x)/(6-4x+x²)
   2  │  1/(1-x+x²/2)  (6+4x+x²)/(6-2x+x²/6)  ...

  Diagonal [n/n] entries converge fastest for analytic functions.

════════════════════════════════════════════════════════════════════════
```

**Convergence:** For functions analytic in a disk $|x| < R$, the diagonal Padé $[n/n]$ converges to $f$ in the entire disk — in contrast to Taylor series which may diverge for $|x| > r$ where $r$ is the radius of convergence.

### T.2 Matrix Padé and the Matrix Exponential

The matrix exponential $e^A$ is computed using Padé approximants in most numerical libraries. The $(m,m)$ diagonal Padé:

$$e^A \approx [D_m(A)]^{-1} N_m(A)$$

where $N_m$ and $D_m$ are matrix polynomials of degree $m$. Combined with **scaling and squaring** (using $e^A = (e^{A/2^s})^{2^s}$):

1. Scale: $A' = A/2^s$ with $\|A'\| \leq 1$
2. Compute Padé approximant $e^{A'} \approx [m/m](A')$
3. Square: $e^A \approx (e^{A'})^{2^s}$

This is used in `scipy.linalg.expm` and is essential for:
- Continuous-time Markov chain transition matrices
- Neural ODE solvers (computing $e^{tA}$ exactly)
- Graph diffusion for GNNs (heat kernel $e^{-tL}$)

---

## Appendix U: Connection to Modern LLM Architectures

### U.1 RoPE as Trigonometric Interpolation

**Rotary Position Embedding (RoPE)** represents positions as rotations in 2D subspaces of the embedding:

$$q_m = R_\Theta^m q, \quad k_n = R_\Theta^n k$$

where $R_\Theta^m$ is a block-diagonal rotation matrix with blocks $\begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix}$.

The attention score:

$$q_m^\top k_n = q^\top R_\Theta^{m-n} k$$

depends only on relative position $m-n$. The frequencies $\theta_i = 10000^{-2(i-1)/d}$ match the sinusoidal PE frequencies — this is the same trigonometric interpolation scheme.

**LongRoPE (2024):** For context length extension, scales frequencies non-uniformly:
- High-frequency dimensions (small $i$): scale by $\lambda_{\text{long}} > 1$ (extend range)
- Low-frequency dimensions (large $i$): scale by $\lambda_{\text{short}} \approx 1$ (minimal change)

This is essentially frequency-domain interpolation of the positional encoding.

### U.2 Flash Attention and Numerical Integration

The attention matrix $A_{ij} = \text{softmax}(q_i^\top k_j / \sqrt{d})$ can be viewed as a matrix of **weights for a kernel regression**:

$$\text{Attention}(Q, K, V) = A V = \sum_j A_{ij} v_j$$

This is a discrete approximation to the integral:

$$y(x) = \int k(x, z) v(z) \mu(dz)$$

where $k(x, z) = e^{q(x)^\top k(z)} / \int e^{q(x)^\top k(z')} \mu(dz')$ is a normalized kernel. Understanding attention as kernel regression connects to RBF interpolation theory — the "query point" $q_i$ selects a weighted combination of "function values" $v_j$ based on "proximity" $q_i^\top k_j$.

### U.3 NeRF and Neural Radiance Fields

Neural Radiance Fields (NeRF, Mildenhall et al. 2020) represent a 3D scene as:

$$f_\theta: (x, y, z, \theta, \phi) \to (\text{RGB}, \sigma)$$

The key challenge: standard MLPs cannot represent high-frequency spatial details. **Fourier feature encoding** solves this:

$$\gamma(x) = [\sin(2^0\pi x), \cos(2^0\pi x), \ldots, \sin(2^L\pi x), \cos(2^L\pi x)]$$

This is the **frequency embedding** — equivalent to the first $L$ Fourier basis functions. The Fourier features lift the input to a space where the target function is smooth, enabling efficient approximation by the MLP.

**Connection to RFF:** Tancik et al. (2020) showed that Fourier features are equivalent to sampling the random Fourier feature map at specific fixed frequencies rather than random ones — targeted at the frequencies of the scene being rendered.

---

## Appendix V: Error Analysis Summary and Complexity Reference

### V.1 Computational Complexity

| Operation | Complexity | Notes |
|---|---|---|
| Lagrange interpolation setup | $O(n^2)$ | Computing $n$ basis polynomials |
| Barycentric weights setup | $O(n^2)$ | General nodes; $O(n)$ for Chebyshev |
| Barycentric evaluation | $O(n)$ per point | Most efficient form |
| Newton divided differences | $O(n^2)$ | Table construction |
| Newton evaluation (Horner) | $O(n)$ per point | |
| Cubic spline setup | $O(n)$ | Thomas algorithm on tridiagonal system |
| Cubic spline evaluation | $O(\log n)$ | Binary search + $O(1)$ eval |
| FFT of length $n$ | $O(n \log n)$ | Cooley-Tukey, $n = 2^k$ |
| Least-squares QR | $O(mn^2)$ | $m$ data points, degree $n$ |
| RBF interpolation setup | $O(n^3)$ | Dense system solve |
| GP posterior | $O(n^3)$ | Cholesky of $n \times n$ kernel matrix |
| Sparse GP (SVGP) | $O(nm^2)$ | $m$ inducing points |

### V.2 Error Summary

| Method | Error bound | When tight |
|---|---|---|
| Lagrange (equispaced, $n$) | Can grow exponentially | Runge function, large $n$ |
| Lagrange (Chebyshev, $n$) | $\|f^{(n+1)}\|_\infty / (2^n (n+1)!)$ | Analytic $f$ |
| Cubic spline | $\frac{5}{384} h^4 \|f^{(4)}\|_\infty$ | Smooth $f$, equispaced $h$ |
| Least-squares (degree $n$) | $(1 + \Lambda_n) E_n^*(f)$ | Depends on node Lebesgue constant |
| Trigonometric ($n$ terms) | $2\sum_{|k|>n/2} |\hat{f}_k|$ | Exponential for analytic periodic $f$ |
| RBF (Gaussian) | Exponential in $n$ for analytic $f$ | Sensitive to shape parameter $\varepsilon$ |

### V.3 Choosing the Right Method

```
DECISION TREE: WHICH INTERPOLATION METHOD?
════════════════════════════════════════════════════════════════════════

  Data type?
  ├── Periodic function on [0,2π]?
  │   └── USE: Trigonometric interpolation (FFT)
  │       Best accuracy for smooth periodic functions
  │
  ├── Scattered data in R^d (d≥2)?
  │   ├── Need uncertainty estimate?
  │   │   └── USE: Gaussian Process regression
  │   └── No uncertainty needed?
  │       └── USE: RBF interpolation (Matérn kernel)
  │
  └── 1D data on [a,b]?
      ├── Few points (n ≤ 30), need exact interpolation?
      │   └── USE: Barycentric Lagrange (Chebyshev nodes)
      │
      ├── Many points (n > 30) or need local control?
      │   └── USE: Cubic splines (natural or not-a-knot BC)
      │
      └── More data than parameters (m > n)?
          ├── Need derivative information?
          │   └── USE: Smoothing spline
          └── Polynomial features?
              └── USE: QR least-squares (NOT normal equations)

════════════════════════════════════════════════════════════════════════
```

---

## Appendix W: Notation Reference

Throughout this section, the following notation is used consistently:

| Symbol | Meaning |
|---|---|
| $x_j$ or $t_j$ | Interpolation nodes / knots |
| $y_j$ | Data values $f(x_j)$ |
| $n$ | Degree of polynomial interpolant (using $n+1$ nodes) |
| $\ell_j(x)$ | $j$-th Lagrange basis polynomial |
| $w_j$ | Barycentric weight for node $x_j$ |
| $\omega_{n+1}(x)$ | Node polynomial $\prod_{j=0}^n (x - x_j)$ |
| $\Lambda_n$ | Lebesgue constant |
| $T_n(x)$ | Chebyshev polynomial of degree $n$ |
| $[f_0, f_1, \ldots, f_k]$ | $k$-th divided difference |
| $S(x)$ | Spline function |
| $M_i$ | Second derivative of spline at node $i$: $S''(x_i)$ |
| $h_i$ | Mesh spacing $x_{i+1} - x_i$ |
| $B_{i,p}(x)$ | $i$-th B-spline basis function of degree $p$ |
| $E_n(f)$ | Best polynomial approximation error $\inf_{p \in \mathcal{P}_n} \|f-p\|$ |
| $\hat{f}_k$ | $k$-th Fourier coefficient |
| $c_k$ | $k$-th Chebyshev coefficient |
| $k(x,y)$ | Kernel function |
| $\mathcal{H}_k$ | Reproducing Kernel Hilbert Space with kernel $k$ |
| $\phi(r)$ | Radial basis function evaluated at $r = \|x - x_j\|$ |
| $\varepsilon$ | RBF shape parameter |
| $\ell$ | RBF lengthscale |
