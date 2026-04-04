[← Previous Chapter: Information Theory](../09-Information-Theory/README.md) | [Next Chapter: Graph Theory →](../11-Graph-Theory/README.md)

---

# Chapter 10 — Numerical Methods

> _"The purpose of computing is insight, not numbers."_ — Richard Hamming

## Overview

Numerical methods are the computational backbone of modern machine learning. Every training run involves floating-point arithmetic, every optimizer is a numerical algorithm, and every kernel or attention mechanism is a numerical approximation. This chapter develops the mathematical and algorithmic foundations that determine whether a computation is accurate, stable, and efficient.

The progression moves from the fundamental question of how computers represent real numbers (§01), through the linear algebra workhorses that power transformer inference and training (§02), to the optimization algorithms that train models (§03), the interpolation and approximation theory behind positional encodings and function approximators (§04), and finally the integration methods underlying probabilistic ML and variational inference (§05).

**The conceptual arc:** understand error sources (§01) → solve linear systems accurately (§02) → optimize objectives efficiently (§03) → approximate functions stably (§04) → integrate distributions analytically or numerically (§05).

---

## Subsection Map

| #   | Subsection                                                                                     | What It Covers                                                                                    | Canonical Topics                                                                                                                                          |
| --- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 01  | [Floating-Point Arithmetic](01-Floating-Point-Arithmetic/notes.md)                             | IEEE 754 representation, rounding error, catastrophic cancellation, mixed-precision training      | IEEE 754, machine epsilon, ULP, guard digits, Kahan summation, FP16/BF16/FP8, loss scaling, condition number of FP operations                            |
| 02  | [Numerical Linear Algebra](02-Numerical-Linear-Algebra/notes.md)                               | Stable algorithms for linear systems, least squares, eigenvalues, and matrix decompositions       | LU with pivoting, QR factorization (Householder/Gram-Schmidt), SVD algorithms (Golub-Reinsch), iterative methods (CG, GMRES), conditioning and stability |
| 03  | [Numerical Optimization](03-Numerical-Optimization/notes.md)                                   | Implementation-level optimization algorithms: line search, quasi-Newton, trust region, curvature  | Armijo/Wolfe conditions, BFGS two-loop recursion, trust region Cauchy point, finite-difference gradient check, Adam as diagonal quasi-Newton, HVP        |
| 04  | [Interpolation and Approximation](04-Interpolation-and-Approximation/notes.md)                 | Polynomial interpolation, splines, B-splines, FFT, RBF, and their role in encodings/architectures | Lagrange/Newton/barycentric interpolation, Runge's phenomenon, Chebyshev nodes, cubic splines, B-splines (KAN), least-squares QR, FFT, RKHS, RoPE        |
| 05  | [Numerical Integration](05-Numerical-Integration/notes.md)                                     | Quadrature rules, Monte Carlo, and their role in probabilistic ML and variational inference       | Newton-Cotes, Gaussian quadrature, Romberg/Richardson, adaptive quadrature, Monte Carlo, quasi-MC (Sobol/Halton), Gauss-Hermite, reparameterization trick |

---

## Reading Order and Dependencies

```
01-Floating-Point-Arithmetic       (error model: machine epsilon, ULP, condition numbers)
        ↓
02-Numerical-Linear-Algebra        (stable solvers: LU, QR, SVD, iterative methods)
        ↓
03-Numerical-Optimization          (implementation: line search, quasi-Newton, trust region)
        ↓
04-Interpolation-and-Approximation (function approx: splines, Chebyshev, FFT, RBF, RoPE)
        ↓
05-Numerical-Integration           (quadrature: Gaussian, Monte Carlo, variational ELBO)
        ↓
Chapter 11 — Graph Theory
```

---

## What Belongs Where — Canonical Homes

| Topic                                                    | Canonical Home | Preview Only In                        |
| -------------------------------------------------------- | -------------- | -------------------------------------- |
| IEEE 754, FP16/BF16/FP8, loss scaling                   | §01            | §02 (stability context)                |
| Condition numbers of matrix problems                     | §01, §02       | —                                      |
| LU factorization (numerical, with pivoting)              | §02            | §03 (Newton step uses LU)              |
| QR factorization (Householder, Gram-Schmidt)             | §02            | §04 (least-squares via QR)             |
| SVD algorithms (Golub-Reinsch)                           | §02            | —                                      |
| Iterative solvers (CG, GMRES, preconditioners)           | §02            | §03 (CG for quadratics)                |
| Armijo/Wolfe line search (implementation)                | §03            | —                                      |
| BFGS/L-BFGS two-loop recursion                           | §03            | —                                      |
| Trust region methods (Cauchy point, dog-leg)             | §03            | —                                      |
| Finite-difference gradient and Hessian checks            | §03            | —                                      |
| Polynomial interpolation (Lagrange, Newton, barycentric) | §04            | —                                      |
| Chebyshev nodes, Chebyshev polynomials                   | §04            | §05 (Clenshaw-Curtis quadrature)       |
| Cubic splines (natural, clamped)                         | §04            | —                                      |
| B-splines (de Boor algorithm, KAN connection)            | §04            | —                                      |
| Least-squares polynomial fitting (QR vs normal eqs)     | §04            | §02 (QR theory)                        |
| FFT (Cooley-Tukey, DFT exactness)                        | §04            | §05 (trapezoidal / spectral accuracy)  |
| RBF interpolation, RKHS, representer theorem             | §04            | §05 (Bayesian quadrature preview)      |
| Sinusoidal positional encoding, RoPE                     | §04            | —                                      |
| Newton-Cotes rules (midpoint, trapezoid, Simpson)        | §05            | —                                      |
| Gaussian quadrature (GL, GH, GC, Golub-Welsch)          | §05            | —                                      |
| Romberg integration, Richardson extrapolation            | §05            | §01 (Richardson as error cancellation) |
| Adaptive quadrature (Gaussian-Kronrod)                   | §05            | —                                      |
| Monte Carlo integration, variance reduction              | §05            | —                                      |
| Quasi-Monte Carlo (Sobol, Halton, discrepancy)           | §05            | —                                      |
| Gauss-Hermite quadrature for Gaussian expectations       | §05            | —                                      |
| Reparameterization trick, ELBO gradient                  | §05            | —                                      |

---

## Overlap Danger Zones

### 1. Numerical Linear Algebra ↔ Advanced Linear Algebra (Chapter 3)

- **Ch3** covers the mathematical theory of eigenvalues, SVD, matrix decompositions, and matrix functions.
- **§02** covers the numerical algorithms for computing these (Golub-Reinsch SVD, Lanczos, shift-and-invert, stability analysis).
- §02 may reference Ch3 for mathematical properties; §02 is the canonical home for algorithmic implementation and numerical stability.

### 2. Numerical Optimization ↔ Optimization Chapter (Chapter 8)

- **Ch8** covers the mathematical theory of convergence, convexity, and algorithm design for optimization.
- **§03** covers implementation: line search termination, quasi-Newton two-loop recursion, trust-region geometry, gradient checking.
- §03 may reference Ch8 for convergence theory; §03 is the canonical home for numerically stable implementation details.

### 3. Interpolation ↔ Function Approximation (Ch 12/21)

- **§04** covers classical interpolation and approximation with polynomial/spline/RBF methods.
- **Ch12 (Functional Analysis)** covers infinite-dimensional approximation theory, Sobolev spaces, and RKHS in depth.
- §04 previews RKHS and representer theorem; Ch12 is the canonical home for functional analysis theory.

### 4. Numerical Integration ↔ Probability (Chapter 6)

- **Ch6** covers the probability theory of Monte Carlo (LLN, CLT, importance sampling as probability concepts).
- **§05** covers the numerical algorithms for Monte Carlo, quasi-MC, and quadrature rules.
- §05 may reference Ch6 for probabilistic foundations; §05 is the canonical home for integration algorithms.

---

## Key Cross-Chapter Dependencies

**From Chapter 3 (Advanced Linear Algebra):**
- Eigenvalue theory → §02 (power iteration, Lanczos), §03 (curvature analysis via Hessian spectrum)
- SVD theory → §02 (Golub-Reinsch algorithm), §04 (low-rank approximation)
- Positive definite matrices → §02 (Cholesky, CG), §05 (Gaussian quadrature Jacobi matrix)

**From Chapter 4–5 (Calculus):**
- Taylor series and error terms → §01 (FP error), §04 (interpolation error), §05 (quadrature error)
- Integration theory → §05 (Riemann sums, measure theory preview)
- Differential equations → §03 (continuous-time optimization dynamics)

**From Chapter 6–7 (Probability and Statistics):**
- Monte Carlo theory (LLN, CLT) → §05 (MC integration, confidence intervals)
- Gaussian distributions → §05 (Gauss-Hermite for Gaussian expectations, ELBO)
- Kernel methods → §04 (RKHS, RBF interpolation)

**From Chapter 8 (Optimization):**
- Algorithm theory (GD, BFGS, trust region) → §03 (numerical implementation)
- Adam optimizer → §03 (diagonal quasi-Newton interpretation)

**Into Chapter 11 (Graph Theory):**
- Iterative linear solvers → §11 (graph Laplacian eigenvalue algorithms)
- Sparse matrix methods → §11 (adjacency/Laplacian matrix computation)

---

## ML Concept Map

| ML Concept                                       | Numerical Foundation                                                         | Section  |
| ------------------------------------------------ | ---------------------------------------------------------------------------- | -------- |
| BF16/FP16 mixed-precision training               | IEEE 754 formats, loss scaling, catastrophic cancellation                    | §01      |
| Flash Attention (numerically stable softmax)     | Log-sum-exp trick, catastrophic cancellation avoidance                       | §01      |
| Transformer weight initialization                | Condition number → stable eigenspectrum at init                              | §01, §02 |
| Solving normal equations for linear regression   | LU/Cholesky with pivoting, QR for conditioning                               | §02      |
| Conjugate gradient for quadratic objectives      | CG as direct solver; preconditioned CG for ill-conditioned Hessians          | §02      |
| Lanczos for Hessian spectrum in interpretability | Iterative eigenvalue method for large sparse Hessians                        | §02, §03 |
| Gradient checking in debugging                   | Finite-difference approximation, step size selection                         | §03      |
| L-BFGS for fine-tuning and second-stage training | Two-loop recursion, limited-memory quasi-Newton                              | §03      |
| Sinusoidal positional encoding (transformers)    | Polynomial/Fourier interpolation, rotation property                          | §04      |
| RoPE (rotary positional encoding)                | Barycentric interpolation, rotation matrix, frequency extrapolation          | §04      |
| KAN (Kolmogorov-Arnold Networks)                 | B-spline basis functions, de Boor algorithm                                  | §04      |
| Random Fourier features (kernel approximation)   | Bochner's theorem, RBF Fourier transform, Monte Carlo feature sampling       | §04      |
| Variational inference (ELBO optimization)        | Gaussian quadrature + reparameterization trick for stochastic gradient       | §05      |
| VAE training (reparameterization)                | Pathwise gradient via change of variables; variance vs score function        | §05      |
| Normalizing flows (change of variables)          | Numerical Jacobian computation, determinant stability                        | §02, §05 |
| Bayesian deep learning (Laplace approximation)   | Gaussian quadrature for posterior predictive; Hessian-based covariance       | §05      |
| Diffusion model noise schedules                  | Numerical integration of SDEs, Euler-Maruyama discretization                 | §05      |
| Hyperparameter search (Bayesian optimization)    | Gaussian process expected improvement via Gauss-Hermite quadrature           | §05      |

---

## Prerequisites

Before starting this chapter, ensure you are comfortable with:

- **Floating-point basics** — binary representation, overflow/underflow — any introductory CS or numerical methods text
- **Matrix operations and decompositions** — $LU$, $QR$, eigendecomposition — [Chapter 3](../03-Advanced-Linear-Algebra/README.md)
- **Gradients and Taylor series** — $\nabla f$, Taylor remainder, big-O notation — [Chapter 4–5](../04-Calculus-Fundamentals/README.md)
- **Probability and integration** — expected values, Gaussian distributions — [Chapter 6](../06-Probability-Theory/README.md)
- **Optimization concepts** — GD, Newton's method, convergence — [Chapter 8](../08-Optimization/README.md)

---

## 2026 Study Plan

If your goal is to build rigorous numerical intuition for modern AI/LLM systems:

1. **Master the error model** — Read: [§01 Floating-Point Notes](01-Floating-Point-Arithmetic/notes.md) — Run: [§01 Theory](01-Floating-Point-Arithmetic/theory.ipynb) — Practice: [§01 Exercises](01-Floating-Point-Arithmetic/exercises.ipynb) — Outcome: You can reason about when FP16/BF16 fail and how to stabilize numerics.

2. **Stable linear algebra** — Read: [§02 Numerical Linear Algebra Notes](02-Numerical-Linear-Algebra/notes.md) — Run: [§02 Theory](02-Numerical-Linear-Algebra/theory.ipynb) — Practice: [§02 Exercises](02-Numerical-Linear-Algebra/exercises.ipynb) — Outcome: You can implement and diagnose LU, QR, CG, and Lanczos in ML contexts.

3. **Implement optimizers from scratch** — Read: [§03 Numerical Optimization Notes](03-Numerical-Optimization/notes.md) — Run: [§03 Theory](03-Numerical-Optimization/theory.ipynb) — Practice: [§03 Exercises](03-Numerical-Optimization/exercises.ipynb) — Outcome: You can implement Armijo line search, L-BFGS two-loop recursion, and gradient checking.

4. **Understand encodings and approximation** — Read: [§04 Interpolation Notes](04-Interpolation-and-Approximation/notes.md) — Run: [§04 Theory](04-Interpolation-and-Approximation/theory.ipynb) — Practice: [§04 Exercises](04-Interpolation-and-Approximation/exercises.ipynb) — Outcome: You can derive RoPE, explain Runge's phenomenon, and understand KAN basis functions.

5. **Probabilistic integration** — Read: [§05 Numerical Integration Notes](05-Numerical-Integration/notes.md) — Run: [§05 Theory](05-Numerical-Integration/theory.ipynb) — Practice: [§05 Exercises](05-Numerical-Integration/exercises.ipynb) — Outcome: You understand Gaussian quadrature, Monte Carlo variance reduction, and the reparameterization trick.

Recommended pacing: 2-3 sessions per subsection. Prioritize §01 and §04 if your work touches mixed-precision training and positional encoding research.

---

## 2026 Numerical Best Practices for AI/LLMs

1. Default to BF16 over FP16 for training: wider dynamic range eliminates loss-scaling overhead.
2. Use online (Kahan/Welford) summation for stable mean/variance over long sequences.
3. Prefer QR to normal equations for any least-squares problem with κ > 10⁴.
4. Apply the log-sum-exp trick everywhere softmax or log-probabilities appear.
5. Prefer iterative solvers (CG, MINRES) over dense factorizations for problems with sparse structure.
6. Verify gradients with finite differences at the start of every custom backward pass.
7. Check FP8 quantization against BF16 reference using relative error metrics, not just accuracy proxies.
8. For Gaussian expectations in probabilistic models, try Gauss-Hermite (n=20) before resorting to Monte Carlo.
9. When MC variance is high, use quasi-MC (Sobol) or antithetic variates before increasing sample count.
10. Profile numerically expensive operations — attention, normalization, loss computation — with condition number estimates.

---

## Curated Online Resources

### Core Theory

1. Trefethen and Bau, _Numerical Linear Algebra_ — Gold standard for §02 topics: https://people.maths.ox.ac.uk/trefethen/text.html
2. Trefethen, _Approximation Theory and Approximation Practice_ — ATAP, gold standard for §04: https://people.maths.ox.ac.uk/trefethen/ATAP/
3. Numerical Recipes (Press et al.) — Practical implementation reference for all sections: https://numerical.recipes/

### ML-Focused Numerical References

1. Higham, _Accuracy and Stability of Numerical Algorithms_ — Deep reference for §01–§02
2. Robert and Casella, _Monte Carlo Statistical Methods_ — Rigorous MC theory for §05
3. Rasmussen and Williams, _Gaussian Processes for Machine Learning_ — §04 RKHS, §05 Bayesian quadrature: http://gaussianprocess.org/gpml/

### Canonical Papers You Should Know

1. Golub and Welsch (1969) — Calculation of Gauss quadrature rules: the eigenvalue algorithm for §05
2. Vaswani et al. (2017), _Attention Is All You Need_ — Sinusoidal PE (§04 application)
3. Su et al. (2024), _RoFormer: Enhanced Transformer with Rotary Position Embedding_ — RoPE (§04)
4. Liu et al. (2024), _KAN: Kolmogorov-Arnold Networks_ — B-spline activations (§04)
5. Micikevicius et al. (2018), _Mixed Precision Training_ — FP16 loss scaling (§01)

---

[← Previous Chapter: Information Theory](../09-Information-Theory/README.md) | [Next Chapter: Graph Theory →](../11-Graph-Theory/README.md)
