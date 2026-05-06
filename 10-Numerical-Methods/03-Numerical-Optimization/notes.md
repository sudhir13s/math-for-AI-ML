[<- Back to Section02 Numerical Linear Algebra](../02-Numerical-Linear-Algebra/notes.md) | [Next: Interpolation and Approximation ->](../04-Interpolation-and-Approximation/notes.md)

---

# Numerical Optimization

> _"In theory there is no difference between theory and practice. In practice there is."_ - Yogi Berra

## Overview

Numerical optimization is the computational science of finding minima (or maxima) of functions. While Section08-Optimization covered the mathematical theory of gradient-based methods, convergence analysis, and adaptive learning rates, this section focuses on the **numerical implementation** aspects: how floating-point arithmetic affects algorithm behavior, how to implement robust line search, how to exploit problem structure, and how the algorithms used in deep learning trace their lineage to classical numerical optimization.

The central insight: every optimization algorithm is implicitly solving a sequence of linear systems. Understanding this connection - and how the numerical properties of those systems determine convergence - is the key to understanding why Adam works, why Newton's method is quadratically convergent, and why gradient descent can stall.

**Scope note:** Gradient descent theory, convergence rates, and adaptive learning rates are fully covered in [Section08-Optimization](../../08-Optimization/README.md). This section covers the *numerical implementation* aspects that Section08 does not address in depth: line search implementations, trust region methods, quasi-Newton methods (L-BFGS), second-order methods, and the floating-point behavior of optimization algorithms.

## Prerequisites

- [Section01 Floating-Point Arithmetic](../01-Floating-Point-Arithmetic/notes.md) - machine epsilon, condition numbers, stability
- [Section02 Numerical Linear Algebra](../02-Numerical-Linear-Algebra/notes.md) - CG, condition numbers, matrix decompositions
- [Section08-01 Gradient Descent](../../08-Optimization/01-Gradient-Descent/notes.md) - gradient descent theory
- [Section08-05 Stochastic Optimization](../../08-Optimization/05-Stochastic-Optimization/notes.md) - SGD, Adam, momentum

## Companion Notebooks

| Notebook | Description |
|----------|-------------|
| [theory.ipynb](theory.ipynb) | Line search, Newton's method, L-BFGS, trust region, convergence analysis |
| [exercises.ipynb](exercises.ipynb) | 10 graded exercises on optimization algorithms and their numerical properties |

## Learning Objectives

After completing this section, you will:

- Implement Wolfe condition line search and explain why it's needed for convergence
- Explain why Newton's method is quadratically convergent and when it fails numerically
- Implement L-BFGS and explain the two-loop recursion
- Understand trust region methods as an alternative to line search
- Analyze the effect of condition number on gradient descent convergence
- Connect Adam's update rule to diagonal quasi-Newton methods
- Implement numerical differentiation and analyze its accuracy
- Explain why finite differences have an optimal step size

---

## Table of Contents

- [1. Intuition](#1-intuition)
- [2. Line Search Methods](#2-line-search-methods)
  - [2.1 Exact Line Search](#21-exact-line-search)
  - [2.2 Wolfe Conditions](#22-wolfe-conditions)
  - [2.3 Backtracking Armijo](#23-backtracking-armijo)
  - [2.4 Strong Wolfe and Zoom](#24-strong-wolfe-and-zoom)
- [3. Newton's Method](#3-newtons-method)
  - [3.1 Quadratic Convergence](#31-quadratic-convergence)
  - [3.2 Modified Newton for Non-Convex](#32-modified-newton-for-non-convex)
  - [3.3 Inexact Newton Methods](#33-inexact-newton-methods)
- [4. Quasi-Newton Methods](#4-quasi-newton-methods)
  - [4.1 BFGS Update](#41-bfgs-update)
  - [4.2 L-BFGS: Limited Memory BFGS](#42-l-bfgs-limited-memory-bfgs)
  - [4.3 SR1 Update](#43-sr1-update)
- [5. Trust Region Methods](#5-trust-region-methods)
  - [5.1 Trust Region Framework](#51-trust-region-framework)
  - [5.2 Cauchy Point and Dogleg](#52-cauchy-point-and-dogleg)
  - [5.3 Trust Region vs Line Search](#53-trust-region-vs-line-search)
- [6. Numerical Differentiation](#6-numerical-differentiation)
  - [6.1 Finite Differences](#61-finite-differences)
  - [6.2 Optimal Step Size](#62-optimal-step-size)
  - [6.3 Complex-Step Derivative](#63-complex-step-derivative)
- [7. Constrained Optimization Numerics](#7-constrained-optimization-numerics)
  - [7.1 Projected Gradient](#71-projected-gradient)
  - [7.2 Augmented Lagrangian](#72-augmented-lagrangian)
- [8. Applications in Machine Learning](#8-applications-in-machine-learning)
  - [8.1 Adam as Diagonal Quasi-Newton](#81-adam-as-diagonal-quasi-newton)
  - [8.2 L-BFGS for Full-Batch Training](#82-l-bfgs-for-full-batch-training)
  - [8.3 Hessian-Free Optimization](#83-hessian-free-optimization)
  - [8.4 Natural Gradient and Fisher Information](#84-natural-gradient-and-fisher-information)
- [9. Common Mistakes](#9-common-mistakes)
- [10. Exercises](#10-exercises)
- [11. Why This Matters for AI (2026 Perspective)](#11-why-this-matters-for-ai-2026-perspective)
- [12. Conceptual Bridge](#12-conceptual-bridge)

---

## 1. Intuition

Numerical optimization differs from mathematical optimization in one fundamental way: we compute with floating-point numbers, not exact reals. This has profound consequences:

**The gradient is never exact:** Even with automatic differentiation, the computed gradient $\hat{g}$ satisfies $\hat{g} = g + O(\varepsilon_{\text{mach}} \|g\|)$. For well-conditioned objectives, this is fine. For ill-conditioned objectives near a minimum, gradients can be numerically zero before we actually reach the minimum.

**The Hessian is expensive and often unavailable:** Computing $H \in \mathbb{R}^{d \times d}$ requires $O(d^2)$ memory and $O(d^2)$ Hessian-vector products. For $d = 10^8$ (typical LLM size), storing $H$ requires $10^{16}$ bytes - impossible. This is why first-order and quasi-Newton methods dominate.

**Ill-conditioning slows convergence:** Gradient descent on a quadratic $f(x) = \frac{1}{2} x^\top H x$ converges at rate $\left(\frac{\kappa-1}{\kappa+1}\right)^k$ - exponentially slow for large $\kappa(H)$. Preconditioning (approximating $H^{-1}$) is the key to faster convergence.

**The minimum is almost never reached:** Gradient descent converges to a point where $\|g\| < \varepsilon$ for some tolerance $\varepsilon$. In deep learning, we stop early (for generalization, not just optimization), so the numerical optimization problem is always underdetermined by design.

---

## 2. Line Search Methods

A **line search** determines how far to move in a given direction $p_k$:
$$x_{k+1} = x_k + \alpha_k p_k$$

The step size $\alpha_k$ must be chosen carefully: too small -> slow convergence; too large -> divergence or oscillation.

### 2.1 Exact Line Search

Exact line search minimizes $\phi(\alpha) = f(x_k + \alpha p_k)$ exactly:
$$\alpha_k = \arg\min_{\alpha > 0} \phi(\alpha)$$

**For quadratic objectives:** $f(x) = \frac{1}{2} x^\top H x - b^\top x$, the exact step is:
$$\alpha_k = \frac{p_k^\top r_k}{p_k^\top H p_k}, \quad r_k = b - Hx_k$$

This is exactly the CG step size - CG with steepest descent direction is "exact line search gradient descent."

**Limitation:** For general nonlinear $f$, finding the exact minimum of $\phi(\alpha)$ requires many function evaluations. Inexact line search conditions (Wolfe, Armijo) are far more practical.

### 2.2 Wolfe Conditions

The **Wolfe conditions** define a set of acceptable step sizes that ensure:
1. Sufficient decrease (Armijo condition):
   $$f(x_k + \alpha_k p_k) \leq f(x_k) + c_1 \alpha_k \nabla f_k^\top p_k$$
2. Curvature condition:
   $$\nabla f(x_k + \alpha_k p_k)^\top p_k \geq c_2 \nabla f_k^\top p_k$$

with $0 < c_1 < c_2 < 1$. Typical values: $c_1 = 10^{-4}$, $c_2 = 0.9$ (line search for CG/BFGS) or $c_2 = 0.1$ (Wolfe for steepest descent).

**Strong Wolfe conditions** replace the curvature condition with:
$$|\nabla f(x_k + \alpha_k p_k)^\top p_k| \leq c_2 |\nabla f_k^\top p_k|$$

This prevents steps at points where the derivative is positive but not too large (i.e., on the "other side" of a local minimum in the line direction).

**Theorem (Zoutendijk):** If the search direction satisfies $p_k^\top g_k < 0$ (descent direction) and the Wolfe conditions are satisfied, then:
$$\sum_{k=0}^\infty \frac{(\nabla f_k^\top p_k)^2}{\|p_k\|^2} < \infty$$

This guarantees $\liminf_{k \to \infty} \|\nabla f_k\| = 0$ - the gradients eventually become small.

### 2.3 Backtracking Armijo

The simplest practical line search:

```python
def backtracking_armijo(f, g, x, p, alpha0=1.0, rho=0.5, c1=1e-4):
    """Backtracking line search satisfying Armijo condition."""
    alpha = alpha0
    f0 = f(x)
    slope = g(x) @ p  # Directional derivative (should be negative)

    while f(x + alpha * p) > f0 + c1 * alpha * slope:
        alpha *= rho

    return alpha
```

**Analysis:** The Armijo condition ensures "sufficient decrease" - each step reduces $f$ by at least a fraction of what a linear model predicts. Backtracking automatically adjusts the step when the initial trial step is too large.

**For AI:** PyTorch's `torch.optim.LBFGS` uses a strong Wolfe line search. Standard SGD/Adam do not use line search - the learning rate is a fixed hyperparameter.

### 2.4 Strong Wolfe and Zoom

The **zoom algorithm** finds a step satisfying the strong Wolfe conditions:

1. **Bracket phase:** Find an interval $[\alpha_{\text{lo}}, \alpha_{\text{hi}}]$ that must contain a Wolfe-satisfying step. This is done by starting with $\alpha = 0$, then $\alpha = \alpha_{\text{init}}$, and doubling until either the Armijo condition fails or the slope at the right endpoint becomes positive.

2. **Zoom phase:** Bisect (or use cubic/quadratic interpolation) within the bracket until a strong Wolfe point is found.

**Implementation:** `scipy.optimize.line_search_wolfe2` implements this algorithm. It is the standard choice for L-BFGS implementations.

---

## 3. Newton's Method

Newton's method for minimizing $f: \mathbb{R}^d \to \mathbb{R}$ applies Newton's method to the gradient equation $\nabla f(x) = 0$:

$$x_{k+1} = x_k - H_k^{-1} \nabla f_k$$

where $H_k = \nabla^2 f(x_k)$ is the Hessian.

### 3.1 Quadratic Convergence

**Theorem:** If $f$ is twice continuously differentiable, $H^* = \nabla^2 f(x^*)$ is positive definite at the minimum $x^*$, and $x_0$ is sufficiently close to $x^*$, then Newton's method converges quadratically:
$$\|x_{k+1} - x^*\| \leq c \|x_k - x^*\|^2$$

**Intuition:** Each step uses a quadratic model $m_k(p) = f_k + g_k^\top p + \frac{1}{2} p^\top H_k p$, which is exact to second order. Quadratic convergence means errors shrink quadratically: if $\|e_k\| = 10^{-3}$, then $\|e_{k+1}\| \approx 10^{-6}$, $\|e_{k+2}\| \approx 10^{-12}$, etc.

**Cost per iteration:** Solving $H_k d_k = -g_k$ requires $O(d^3)$ flops (direct Cholesky/LU factorization) or $O(d^2)$ (if $H_k$ is known to be SPD and well-conditioned). Plus $O(d^2)$ to form the Hessian.

**For AI:** Newton's method is impractical for modern LLMs ($d \sim 10^{10}$). However, it motivates all quasi-Newton methods, which approximate $H_k^{-1}$ cheaply.

### 3.2 Modified Newton for Non-Convex

At non-convex points (saddle points or near maxima), $H_k$ may be indefinite. Applying Newton's step $d = -H^{-1}g$ can move toward a maximum, not a minimum.

**Strategy 1: Hessian modification.** Replace $H_k$ with $H_k + \tau_k I$ where $\tau_k$ is chosen to make the modified Hessian positive definite. This is the **Levenberg-Marquardt damping** idea.

The modified Newton step solves:
$$(H_k + \tau_k I) d_k = -g_k$$

**Choosing $\tau_k$:**
- If $H_k$ is already positive definite with $\lambda_{\min}(H_k) > \delta$: set $\tau_k = 0$.
- Otherwise: set $\tau_k = |\lambda_{\min}(H_k)| + \delta$ for some $\delta > 0$.

**Strategy 2: Modified Cholesky.** Factor $H_k + E_k = L_k D_k L_k^\top$ where $E_k$ is chosen minimally to make $H_k + E_k$ positive definite. The **Gill-Murray-Wright** modified Cholesky algorithm does this robustly.

### 3.3 Inexact Newton Methods

For large-scale problems, solving $H_k d_k = -g_k$ exactly is too expensive. **Inexact Newton** methods find $d_k$ such that:
$$\|H_k d_k + g_k\| \leq \eta_k \|g_k\|$$

where $\eta_k \in (0, 1)$ is the **forcing sequence**. Using CG to solve the linear system approximately (Newton-CG or truncated Newton):

```
for outer iteration k:
    solve H_k d = -g_k using CG, stopping when
    ||H_k d + g_k|| <= eta_k ||g_k||
    x_{k+1} = x_k + alpha_k d_k  (line search for alpha_k)
```

**Convergence:** If $\eta_k \to 0$ fast enough (e.g., $\eta_k = \min(0.5, \sqrt{\|\nabla f_k\|})$), inexact Newton converges super-linearly. If $\eta_k$ is kept constant, linear convergence with rate $\eta$.

**For AI:** Inexact Newton methods (Hessian-free optimization) were used for training neural networks in the pre-deep-learning era (Martens, 2010). The Hessian-vector product $Hv$ can be computed in $O(d)$ time via automatic differentiation (Pearlmutter's trick), making each CG iteration cost $O(d)$ without forming $H$.

---

## 4. Quasi-Newton Methods

Quasi-Newton methods build up an approximation $B_k \approx H_k$ (or $H_k^{-1}$) using gradient information at each step, without forming the Hessian explicitly.

### 4.1 BFGS Update

The **Broyden-Fletcher-Goldfarb-Shanno (BFGS)** update is the most successful quasi-Newton method.

**Setup:** After step $k$, define:
$$s_k = x_{k+1} - x_k \quad \text{(step taken)}$$
$$y_k = \nabla f_{k+1} - \nabla f_k \quad \text{(gradient change)}$$

**Secant equation:** The updated Hessian approximation $B_{k+1}$ should satisfy:
$$B_{k+1} s_k = y_k \quad \text{(secant condition)}$$

This is analogous to finite difference approximation of the Hessian.

**BFGS update (inverse Hessian form):**
$$H_{k+1} = \left(I - \rho_k s_k y_k^\top\right) H_k \left(I - \rho_k y_k s_k^\top\right) + \rho_k s_k s_k^\top$$

where $\rho_k = 1 / (y_k^\top s_k)$.

**Properties:**
- $H_{k+1}$ is symmetric positive definite if $H_k$ is SPD and $y_k^\top s_k > 0$
- $H_{k+1}$ satisfies the secant condition: $H_{k+1} y_k = s_k$
- BFGS converges super-linearly (between linear and quadratic) for smooth strongly convex functions

**Curvature condition:** $y_k^\top s_k > 0$ is guaranteed when using the Wolfe line search (hence why Wolfe conditions are essential for BFGS). Without it, the update can make $H_{k+1}$ indefinite.

### 4.2 L-BFGS: Limited Memory BFGS

Full BFGS requires $O(d^2)$ memory to store $H_k$. For $d = 10^6$, this is $\sim 8$ TB - impractical.

**L-BFGS** stores only the last $m$ pairs $(s_k, y_k)$ and implicitly computes $H_k g$ via the **two-loop recursion**:

```python
def lbfgs_direction(g, s_list, y_list):
    """Compute H^{-1} g using the two-loop L-BFGS recursion."""
    m = len(s_list)
    rho = [1.0 / (y @ s) for s, y in zip(s_list, y_list)]

    q = g.copy()
    alpha = np.zeros(m)

    # First loop (backward)
    for i in range(m-1, -1, -1):
        alpha[i] = rho[i] * s_list[i] @ q
        q -= alpha[i] * y_list[i]

    # Scale by initial Hessian approximation
    # H_0^{-1} = (s_m^T y_m) / (y_m^T y_m) * I (Barzilai-Borwein scaling)
    if m > 0:
        gamma = (s_list[-1] @ y_list[-1]) / (y_list[-1] @ y_list[-1])
    else:
        gamma = 1.0
    r = gamma * q

    # Second loop (forward)
    for i in range(m):
        beta = rho[i] * y_list[i] @ r
        r += s_list[i] * (alpha[i] - beta)

    return -r  # Search direction
```

**Memory requirement:** $O(md)$ - just $2m$ vectors of dimension $d$. Typically $m = 5$ to $20$.

**Convergence:** L-BFGS with $m$ vectors has memory of $m$ gradient changes, giving super-linear convergence locally (linear globally for general non-convex $f$).

**For AI:** `torch.optim.LBFGS` implements L-BFGS. It is used for full-batch training and fine-tuning tasks where the full gradient is available and noise-free. For stochastic settings (mini-batch), L-BFGS is not directly applicable because the secant condition $y_k^\top s_k > 0$ fails when gradients are noisy.

### 4.3 SR1 Update

The **Symmetric Rank-1 (SR1)** update is simpler:
$$B_{k+1} = B_k + \frac{(y_k - B_k s_k)(y_k - B_k s_k)^\top}{(y_k - B_k s_k)^\top s_k}$$

**Advantage:** SR1 can capture indefinite Hessians (saddle points), unlike BFGS (which always produces SPD updates).

**Disadvantage:** Not guaranteed to remain positive definite; can fail numerically if $(y_k - B_k s_k)^\top s_k \approx 0$.

---

## 5. Trust Region Methods

Trust region methods take a different approach: instead of choosing a direction and then a step size (line search), they choose a step size first and then find the best direction within that radius.

### 5.1 Trust Region Framework

At each iteration, define a **trust region radius** $\Delta_k$ and solve:
$$\min_{p} m_k(p) = f_k + g_k^\top p + \frac{1}{2} p^\top B_k p \quad \text{subject to} \quad \|p\| \leq \Delta_k$$

where $B_k$ is an approximation to the Hessian.

**Ratio test:** After computing the step $p_k$, compare the actual reduction to the predicted reduction:
$$\rho_k = \frac{f_k - f(x_k + p_k)}{m_k(0) - m_k(p_k)} = \frac{\text{actual reduction}}{\text{predicted reduction}}$$

**Update rules:**
- $\rho_k < 0.25$: shrink trust region ($\Delta_{k+1} = 0.25 \Delta_k$); reject step
- $0.25 \leq \rho_k < 0.75$: keep trust region; accept step
- $\rho_k \geq 0.75$: expand trust region ($\Delta_{k+1} = 2\Delta_k$); accept step

### 5.2 Cauchy Point and Dogleg

The **exact trust region subproblem** is an eigenvalue problem that costs $O(d^3)$. Cheaper approximations:

**Cauchy point:** The point that minimizes $m_k$ along the steepest descent direction subject to the trust region constraint:
$$p^C = -\alpha^C g_k, \quad \alpha^C = \begin{cases} \Delta_k / \|g_k\| & \text{if } g_k^\top B_k g_k \leq 0 \\ \min\left(\frac{\|g_k\|^2}{g_k^\top B_k g_k}, \frac{\Delta_k}{\|g_k\|}\right) & \text{otherwise} \end{cases}$$

The Cauchy point gives linear convergence (same as gradient descent with exact line search).

**Dogleg method:** Interpolate between the Cauchy point and the unconstrained Newton step $p^N = -B_k^{-1} g_k$:
- If $\|p^N\| \leq \Delta_k$: take the Newton step
- Else: move along the path $p^C \to p^N$ until hitting the trust region boundary

This gives super-linear convergence near the optimum.

### 5.3 Trust Region vs Line Search

| Property | Trust Region | Line Search |
|----------|-------------|------------|
| Convergence guarantee | Global (any Hessian approx) | Requires descent direction |
| Indefinite Hessian | Handled naturally | Needs modification |
| Cost per iteration | Higher (subproblem solve) | Lower (backtracking) |
| Parameter tuning | Trust region radius $\Delta$ | Learning rate $\alpha$ |
| Non-smooth objectives | Can be extended | Harder |

**For AI:** Trust region methods are used in reinforcement learning (TRPO, PPO) where the "trust region" constrains policy updates. The KL divergence constraint in TRPO plays the role of the trust region: $D_{\text{KL}}(\pi_{\theta_k} \| \pi_{\theta}) \leq \delta$.

---

## 6. Numerical Differentiation

When analytic gradients are unavailable (or to verify automatic differentiation), we use finite difference approximations.

### 6.1 Finite Differences

**Forward difference:**
$$\frac{\partial f}{\partial x_i} \approx \frac{f(x + h e_i) - f(x)}{h}$$

Error: $O(h) + O(\varepsilon_{\text{mach}} / h)$.

**Centered difference:**
$$\frac{\partial f}{\partial x_i} \approx \frac{f(x + h e_i) - f(x - h e_i)}{2h}$$

Error: $O(h^2) + O(\varepsilon_{\text{mach}} / h)$ - more accurate but requires two function evaluations.

**Richardson extrapolation:** Eliminate the leading-order error term:
$$f'(x) \approx \frac{4 \cdot \text{CD}(h/2) - \text{CD}(h)}{3}$$

where $\text{CD}(h)$ denotes centered difference with step $h$. Error: $O(h^4)$.

### 6.2 Optimal Step Size

The total error in centered difference is:
$$\text{Error}(h) = \underbrace{C_1 h^2}_{\text{truncation}} + \underbrace{C_2 \varepsilon_{\text{mach}} / h}_{\text{rounding}}$$

Minimizing over $h$: set $\frac{d}{dh}\text{Error} = 2C_1 h - C_2 \varepsilon_{\text{mach}} / h^2 = 0$, giving:
$$h^* = \left(\frac{C_2 \varepsilon_{\text{mach}}}{2C_1}\right)^{1/3} \approx \varepsilon_{\text{mach}}^{1/3}$$

For fp64: $h^* \approx (2.2 \times 10^{-16})^{1/3} \approx 6 \times 10^{-6}$.

Minimum achievable error: $\text{Error}(h^*) \approx 3(C_1 C_2^2 \varepsilon_{\text{mach}}^2)^{1/3} \approx \varepsilon_{\text{mach}}^{2/3}$.

For forward difference: $h^* \approx \varepsilon_{\text{mach}}^{1/2} \approx 1.5 \times 10^{-8}$, minimum error $\approx \varepsilon_{\text{mach}}^{1/2}$.

### 6.3 Complex-Step Derivative

The **complex-step method** avoids cancellation entirely:
$$f'(x) \approx \frac{\text{Im}[f(x + ih)]}{h}$$

where $i = \sqrt{-1}$. This requires evaluating $f$ at a complex argument.

**Error:** $O(h^2)$ - same as centered difference, but **no rounding error** from cancellation (imaginary part is computed by addition, not subtraction). With $h = 10^{-20}$, the error is $\sim 10^{-40}$ - essentially machine precision.

**Limitation:** Requires the function $f$ to be analytically extendable to complex arguments (which almost all smooth functions are, but branch cuts can cause issues).

---

## 7. Constrained Optimization Numerics

### 7.1 Projected Gradient

For optimization over a convex set $\mathcal{C}$ (e.g., the simplex, a box, or the unit sphere):
$$x_{k+1} = \mathcal{P}_\mathcal{C}(x_k - \alpha_k \nabla f(x_k))$$

where $\mathcal{P}_\mathcal{C}(y) = \arg\min_{x \in \mathcal{C}} \|x - y\|_2$ is the Euclidean projection.

**Common projections:**
- **Box** $\{x : \ell \leq x \leq u\}$: $\mathcal{P}(y)_i = \min(\max(y_i, \ell_i), u_i)$ - O(n)
- **Simplex** $\{x : x \geq 0, \sum x_i = 1\}$: sort and find threshold (O(n log n))
- **Ball** $\{x : \|x\| \leq r\}$: $\mathcal{P}(y) = r y / \|y\|$ if $\|y\| > r$, else $y$ - O(n)

**For AI:** Gradient clipping is a projected gradient step onto the ball $\{g : \|g\| \leq c\}$. Softmax output can be seen as optimization over the simplex. Spectral normalization maintains weights on the unit spectral norm ball via projected gradient.

### 7.2 Augmented Lagrangian

For equality constrained problems $\min_x f(x) \text{ s.t. } c(x) = 0$, the **augmented Lagrangian** is:
$$\mathcal{L}_\rho(x, \lambda) = f(x) + \lambda^\top c(x) + \frac{\rho}{2} \|c(x)\|^2$$

The alternating direction method of multipliers (ADMM) minimizes this iteratively:
1. $x_{k+1} = \arg\min_x \mathcal{L}_{\rho}(x, \lambda_k)$ (minimize over primal variable)
2. $\lambda_{k+1} = \lambda_k + \rho c(x_{k+1})$ (dual update)

**ADMM in AI:** Used in distributed optimization (splitting model across devices), constrained optimization problems in RL, and structured sparsity (e.g., group lasso).

---

## 8. Applications in Machine Learning

### 8.1 Adam as Diagonal Quasi-Newton

Adam's update rule:
$$x_{t+1} = x_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \varepsilon} \odot \hat{m}_t$$

can be rewritten as:
$$x_{t+1} = x_t - \alpha H_t^{-1} g_t$$

where $H_t = \text{diag}(\sqrt{\hat{v}_t} + \varepsilon)$ is a diagonal approximate Hessian.

**This is diagonal quasi-Newton.** The second moment $\hat{v}_t$ estimates $\mathbb{E}[g_t^2]$, which is proportional to the diagonal of the Fisher information matrix. Dividing by $\sqrt{\hat{v}_t}$ pre-conditions the gradient by the square root of the diagonal Fisher.

**Why RMS, not raw variance?** The square root $\sqrt{v_t}$ instead of $v_t$ is the key. For a quadratic $f$ with diagonal Hessian $H = \text{diag}(h_1, \ldots, h_d)$, the stationary distribution of stochastic gradients has variance $\propto h_i$ in coordinate $i$. So $\sqrt{v_t} \approx \sqrt{h_i}$, and $g / \sqrt{v_t} \approx g \cdot h_i^{-1/2}$. This is a half-step towards the Newton direction $H^{-1}g = g / h_i$.

### 8.2 L-BFGS for Full-Batch Training

L-BFGS is the standard algorithm for full-batch optimization of small-to-medium neural networks:

```python
optimizer = torch.optim.LBFGS(
    model.parameters(),
    lr=1.0,           # Inner learning rate (line search scales this)
    max_iter=20,      # CG iterations per step
    max_eval=25,      # Max function evaluations per step
    history_size=100, # L-BFGS memory
    line_search_fn='strong_wolfe'
)

def closure():
    optimizer.zero_grad()
    loss = criterion(model(X), y)
    loss.backward()
    return loss

for epoch in range(100):
    optimizer.step(closure)
```

**When to use L-BFGS:**
- Batch size = full dataset (no noise)
- Medium-scale problems ($d \lesssim 10^7$)
- Need high accuracy (e.g., physics simulation, function approximation)
- Fine-tuning pre-trained models with small LoRA adapters

**When NOT to use L-BFGS:**
- Large-scale stochastic training (gradient noise breaks secant condition)
- Online/streaming data
- Very large models where memory is constrained

### 8.3 Hessian-Free Optimization

**Hessian-free** (HF) optimization (Martens, 2010) computes Newton steps using CG applied to $H d = -g$ without forming $H$, using the Gauss-Newton approximation:
$$H \approx J^\top J \quad \text{(Gauss-Newton, always SPD)}$$

Hessian-vector products $Hv$ are computed via automatic differentiation:
```python
def hvp(loss, params, v):
    """Hessian-vector product Hv via autograd."""
    grad = torch.autograd.grad(loss, params, create_graph=True)
    grad_v = sum(g.view(-1) @ v_part.view(-1)
                 for g, v_part in zip(grad, v))
    return torch.autograd.grad(grad_v, params)
```

This enables CG iterations that cost $O(d)$ per step (same as a forward+backward pass).

**Historical note:** HF optimization achieved state-of-the-art results on deep network training in 2010-2012, before SGD with momentum/Adam became dominant. For recurrent networks with long sequences (high curvature), HF still has advantages.

### 8.4 Natural Gradient and Fisher Information

The **natural gradient** is:
$$\tilde{g} = F^{-1} g$$

where $F = \mathbb{E}[\nabla \log p_\theta \nabla \log p_\theta^\top]$ is the Fisher information matrix. This is Newton's method with the Hessian replaced by the Fisher matrix.

**Why the Fisher?** The KL divergence $D_{\text{KL}}(p_\theta \| p_{\theta + \delta\theta})$ is approximately $\frac{1}{2} \delta\theta^\top F \delta\theta$. The natural gradient moves in the direction of steepest descent in the space of probability distributions (Riemannian manifold with metric $F$), not in parameter space.

**K-FAC (Kronecker-Factored Approximate Curvature):** Approximates $F \approx A \otimes G$ where $A$ is the covariance of activations and $G$ is the covariance of gradient signals. This gives $F^{-1} \approx A^{-1} \otimes G^{-1}$, which can be computed efficiently.

---

## 9. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
|---|---------|---------------|-----|
| 1 | Using too large a step in finite differences | Rounding error dominates: $h \gg \varepsilon^{1/3}$ | Use $h = \varepsilon^{1/3}$ for centered difference |
| 2 | Using too small a step in finite differences | Catastrophic cancellation: $f(x+h) - f(x) \approx 0$ in fp | Use $h = \varepsilon^{1/3}$, never $h < \varepsilon^{1/2}$ |
| 3 | Applying L-BFGS to stochastic objectives | Gradient noise violates secant condition $y_k^\top s_k > 0$ | Use Adam/SGD for mini-batch; L-BFGS only for full batch |
| 4 | Ignoring the Wolfe condition in BFGS | BFGS update may produce indefinite $H$ | Always use Wolfe line search with BFGS |
| 5 | Using forward finite differences for gradient checking | Only $O(\sqrt{\varepsilon})$ accuracy | Use centered differences or complex-step |
| 6 | Forgetting the Hessian modification in Newton's method | Newton step points toward maximum at saddle points | Add $\tau I$ or use modified Cholesky |
| 7 | Setting learning rate too large and expecting convergence | Divergence is guaranteed for $\eta > 2/\lambda_{\max}(H)$ | Use line search or tune $\eta$ carefully |
| 8 | Using Adam for convex problems when L-BFGS is available | Adam has higher per-iteration cost and worse final accuracy | L-BFGS for full-batch convex problems |
| 9 | Checking gradient only at initial point | Gradients can be correct initially but wrong near optimal due to numerical issues | Gradient check at multiple points including near-optimal |
| 10 | Treating trust region radius as a learning rate | Trust region is in parameter space; effective step size depends on Hessian | Tune trust region ratio test parameters separately |
| 11 | Forgetting that Adam's epsilon matters | $\varepsilon = 10^{-8}$ (default) can dominate for very small gradients | Increase $\varepsilon$ for small models; use $10^{-7}$ for transformers |
| 12 | Not resetting L-BFGS history when task changes | Old curvature information misleads new optimization | Reset with `optimizer.state_dict()` or create new optimizer |

---

## 10. Exercises

### Exercise 1 * - Backtracking Armijo Line Search

Implement the backtracking Armijo line search.

(a) Write `armijo_search(f, grad_f, x, p, alpha0=1.0, rho=0.5, c1=1e-4)`.

(b) Test on $f(x, y) = (x-2)^2 + 10(y-x^2)^2$ (Rosenbrock variant). Start at $(0, 0)$ with descent direction $p = -\nabla f(0, 0)$.

(c) Plot $f(x + \alpha p)$ and the Armijo line, marking the accepted step.

### Exercise 2 * - Newton's Method

Implement Newton's method with line search.

(a) Write `newton_method(f, grad_f, hess_f, x0, tol)` using backtracking Armijo line search.

(b) Test on the Rosenbrock function $f(x, y) = 100(y - x^2)^2 + (1-x)^2$.

(c) Compare convergence (iterations to $10^{-8}$ accuracy) between gradient descent, Newton's method, and gradient descent with exact line search.

### Exercise 3 ** - L-BFGS Two-Loop Recursion

Implement the L-BFGS two-loop recursion.

(a) Write `lbfgs_direction(g, s_list, y_list, m)` that returns the L-BFGS search direction.

(b) Implement a full L-BFGS optimizer using the strong Wolfe line search.

(c) Test on quadratic minimization with varying condition numbers. Compare convergence to gradient descent.

### Exercise 4 ** - Finite Difference Gradient Check

Implement gradient checking via finite differences.

(a) Write `grad_check(f, grad_f, x, h_vals)` that computes the centered difference gradient for each $h$ in `h_vals` and compares to `grad_f(x)`.

(b) Find the optimal $h$ empirically for several test functions.

(c) Plot error vs $h$ on log-log scale. Verify the $O(h^2) + O(\varepsilon/h)$ form.

### Exercise 5 ** - Trust Region vs Line Search

Compare trust region and line search methods.

(a) Implement a simple trust region method with Cauchy point for a 2D test function.

(b) Test on a non-convex function with saddle points. Show that trust region methods don't require modification of the Hessian.

(c) Plot the trajectory of iterates for both methods.

### Exercise 6 ** - Adam as Diagonal Quasi-Newton

Demonstrate Adam's connection to diagonal preconditioning.

(a) Implement gradient descent, Adam, and diagonal-preconditioned gradient descent.

(b) Test on a quadratic $f(x) = \frac{1}{2} x^\top \text{diag}(1, 100, 1000, 10000) x$.

(c) Show that Adam converges at a rate close to the optimal diagonal-preconditioned rate, and much faster than vanilla gradient descent.

### Exercise 7 ** - Numerical Differentiation Accuracy

Implement finite differences and compare to automatic differentiation.

(a) Implement forward, centered, and Richardson extrapolation finite differences.

(b) Find the optimal $h$ for each method on $f(x) = \sin(x^2) e^{-x}$.

(c) Implement the complex-step derivative and verify it achieves near machine precision.

### Exercise 8 *** - Hessian-Vector Product via Autograd

Compute Hessian-vector products efficiently using automatic differentiation.

(a) Write `hvp(model, x, v)` that computes $Hv$ without forming $H$, using PyTorch's `autograd.grad` with `create_graph=True`.

(b) Implement Newton-CG (truncated Newton) using CG applied to the Hessian-vector product.

(c) Test on a small neural network (2 layers, 50 hidden units). Compare convergence to Adam.

---

## 11. Why This Matters for AI (2026 Perspective)

| Aspect | Impact |
|--------|--------|
| **Adam as quasi-Newton** | Understanding Adam through the lens of diagonal preconditioning explains its robustness and motivates improvements like AdaHessian, Shampoo |
| **L-BFGS for fine-tuning** | Fine-tuning small LoRA adapters with full batch uses L-BFGS; achieving faster convergence than Adam in low-noise regime |
| **Hessian-vector products** | Used in mechanistic interpretability (computing Hessian eigenvalue spectra), continual learning (Fisher-based regularization), and meta-learning |
| **Natural gradient** | K-FAC and Shampoo implement approximate natural gradient; used in LLM pre-training at Google (TPU-optimized K-FAC) |
| **Trust region in RL** | TRPO/PPO use trust region methods to constrain policy updates; direct precursor to RLHF stability |
| **Conjugate gradient in optimization** | Newton-CG underlies Hessian-free optimization; CG is used in solving the trust region subproblem |
| **Line search in LBFGS** | Wolfe conditions ensure the secant condition holds; critical for L-BFGS stability |
| **Gradient checking** | Essential for debugging custom autodiff implementations; every major framework implements it |
| **Fisher information** | Quantifies "how much information" each training example provides; central to active learning, Bayesian deep learning, and continual learning |

---

## 12. Conceptual Bridge

**What you learned:** This section bridges the gap between theoretical optimization (Section08) and practical computation. The key themes:

1. **Line search ensures convergence:** Without proper step size selection (Armijo, Wolfe), gradient-based methods may diverge or stagnate. Adam uses a fixed learning rate with momentum - it lacks a formal convergence guarantee for general non-convex problems but works well in practice.

2. **Quasi-Newton approximates the second-order structure:** BFGS/L-BFGS use past gradient differences to build up a Hessian approximation. Adam does this diagonally, using running second-moment estimates. Shampoo/K-FAC do it in a structured block-diagonal form.

3. **Condition number determines convergence rate:** For gradient descent on a quadratic, the convergence rate is exactly $\left(\frac{\kappa-1}{\kappa+1}\right)^k$. Preconditioning (approximating the Hessian) reduces the effective condition number and accelerates convergence.

4. **Floating-point limits gradient accuracy:** For smooth functions, automatic differentiation gives near-exact gradients. Finite differences are limited by $\varepsilon_{\text{mach}}^{1/3}$ accuracy. This matters for gradient checking and numerical Hessian computation.

```
NUMERICAL OPTIMIZATION IN CONTEXT
========================================================================

  Section08 Optimization (theory)     ->   Section10-03 Numerical Optimization
  +-- Gradient descent           --> Line search, convergence guarantees
  +-- Adam/momentum              --> Diagonal quasi-Newton interpretation
  +-- Convergence theory         --> Condition number analysis
  +-- Constrained opt.          --> Projected gradient, ADMM

  Section10-02 Numerical Linear Algebra ->  Section10-03 Numerical Optimization
  +-- CG method                  --> Newton-CG, Hessian-free opt
  +-- Condition numbers          --> GD convergence rate
  +-- Iterative refinement       --> Inexact Newton methods
  +-- Sparse systems             --> Large-scale quasi-Newton

========================================================================
```

*[<- Back to Section02 Numerical Linear Algebra](../02-Numerical-Linear-Algebra/notes.md) | [Next: Section04 Interpolation and Approximation ->](../04-Interpolation-and-Approximation/notes.md)*

---

## Appendix A: Wolfe Conditions - Detailed Analysis

### A.1 Why the Armijo Condition Alone Is Insufficient

The Armijo condition $f(x_k + \alpha p_k) \leq f_k + c_1 \alpha g_k^\top p_k$ prevents steps that are too long (the function must decrease by at least $c_1$ times the predicted linear decrease). But it doesn't prevent steps that are too short!

**Example:** Consider $f(x) = x^2$, $p = -1$ (descent direction). The Armijo condition with $c_1 = 10^{-4}$ is satisfied for any $\alpha \in (0, 2 - 10^{-4})$. But if we choose $\alpha_k = 10^{-10}$, the step is effectively zero and we make no progress.

The **curvature condition** prevents this: it requires the slope $\nabla f_{k+1}^\top p_k$ at the new point to be at least as large (in magnitude) as a fraction $c_2$ of the initial slope. This ensures the step is not too short.

### A.2 Wolfe Conditions Ensure BFGS Remains Well-Defined

The BFGS update requires $y_k^\top s_k > 0$ to maintain positive definiteness of the Hessian approximation. The curvature condition guarantees this:

$$y_k^\top s_k = (\nabla f_{k+1} - \nabla f_k)^\top s_k = (\nabla f_{k+1}^\top p_k - \nabla f_k^\top p_k) \alpha_k$$

The curvature condition says $\nabla f_{k+1}^\top p_k \geq c_2 \nabla f_k^\top p_k$. Since $\nabla f_k^\top p_k < 0$ (descent direction) and $c_2 \in (0, 1)$:
$$y_k^\top s_k = \alpha_k (\underbrace{\nabla f_{k+1}^\top p_k - \nabla f_k^\top p_k}_{\geq (c_2 - 1)\nabla f_k^\top p_k = (1-c_2)|\nabla f_k^\top p_k| > 0})$$

So $y_k^\top s_k > 0$ - the curvature condition ensures the BFGS update remains positive definite.

---

## Appendix B: L-BFGS Two-Loop Recursion - Derivation

The L-BFGS direction $d = -H_k g_k$ where $H_k$ approximates the inverse Hessian. $H_k$ is implicitly defined by:
$$H_k = V_{k-1}^\top \cdots V_{k-m}^\top H_0 V_{k-m} \cdots V_{k-1} + \sum_{i=k-m}^{k-1} \rho_i V_{k-1}^\top \cdots V_{i+1}^\top s_i s_i^\top V_{i+1} \cdots V_{k-1}$$

where $V_i = I - \rho_i y_i s_i^\top$ and $\rho_i = 1 / (y_i^\top s_i)$.

Directly computing $H_k g$ via this formula requires $O(m^2 d)$ operations - expensive if we expand the products. The two-loop recursion computes $H_k g$ in $O(md)$ operations by expanding the matrix-vector product recursively without ever forming $H_k$.

**Two-loop recursion (Nocedal, 1980):**

**First loop** (backward, $i = k-1, k-2, \ldots, k-m$):
$$\alpha_i = \rho_i s_i^\top q_{i+1}, \quad q_i = q_{i+1} - \alpha_i y_i$$

starting with $q_k = g_k$.

**Middle:** Scale by initial Hessian: $r = H_0 q_{k-m}$.

**Second loop** (forward, $i = k-m, k-m+1, \ldots, k-1$):
$$\beta_i = \rho_i y_i^\top r_i, \quad r_{i+1} = r_i + s_i (\alpha_i - \beta_i)$$

The result $d = -r_k = -H_k g_k$ is the L-BFGS search direction.

---

## Appendix C: Trust Region Subproblem - Eigenvalue Approach

The exact trust region subproblem:
$$\min_p \frac{1}{2} p^\top B p + g^\top p \quad \text{s.t.} \|p\| \leq \Delta$$

is solved by the **More-Sorensen** algorithm (1983):

**KKT conditions:** The optimal $p^*$ satisfies:
$$(B + \lambda^* I) p^* = -g, \quad \lambda^* \geq 0, \quad \lambda^* (\Delta - \|p^*\|) = 0$$

where $\lambda^*$ is the Lagrange multiplier.

**Algorithm:**
1. If $\|p^N\| \leq \Delta$ (Newton step within trust region): $p^* = p^N$, $\lambda^* = 0$
2. Otherwise, solve for $\lambda^* > 0$ such that $\|(B + \lambda^* I)^{-1} g\| = \Delta$

Step 2 is a one-dimensional root-finding problem. Since $\|(B + \lambda I)^{-1} g\|$ is monotonically decreasing in $\lambda$ for $\lambda > -\lambda_{\min}(B)$, this can be solved by Newton's method in 1D.

**Cost:** One eigendecomposition of $B$ ($O(d^3)$) plus a few 1D Newton iterations.

---

## Appendix D: Gradient Checking Best Practices

### D.1 What to Check

A gradient check compares $\frac{\partial f}{\partial x_i}$ from `autograd` with the finite difference approximation. The relative error should be:
- $< 10^{-7}$: essentially correct (autograd is accurate to ~10 digits)
- $10^{-7}$ to $10^{-5}$: probably correct (some numerical noise)
- $> 10^{-5}$: potential bug in gradient implementation
- $> 10^{-2}$: definite bug

### D.2 Pitfalls

1. **Non-differentiable operations:** ReLU at zero, `abs(0)`, `max(a, b)` at the boundary - finite differences may give a subgradient, not the same subgradient as autograd.

2. **Numerically sensitive operations:** Softmax with very large logits, log(tiny number) - both methods may have errors here, but they may differ.

3. **Wrong random seed:** Always check with a fixed random seed; bugs can be masked by lucky initialization.

4. **Checking only the gradient, not the Jacobian:** For vector-valued functions, check the full Jacobian, not just a scalar projection.

### D.3 Implementation

```python
def gradient_check(f, x, grad_f, eps=1e-5, rtol=1e-5, atol=1e-8):
    """
    Compare analytical gradient to centered finite differences.
    Returns max relative error across all coordinates.
    """
    n = len(x.ravel())
    g_analytic = grad_f(x).ravel()
    g_numerical = np.zeros(n)

    for i in range(n):
        x_plus = x.copy().ravel()
        x_minus = x.copy().ravel()
        x_plus[i] += eps
        x_minus[i] -= eps
        g_numerical[i] = (f(x_plus.reshape(x.shape)) -
                          f(x_minus.reshape(x.shape))) / (2 * eps)

    # Relative error (with absolute tolerance floor)
    denom = np.maximum(np.abs(g_analytic), atol)
    rel_err = np.abs(g_analytic - g_numerical) / denom
    return rel_err.max(), rel_err
```

---

## Appendix E: Adam - Complete Update Rule and Variants

### E.1 Standard Adam

```python
def adam_step(g, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """One step of Adam optimizer."""
    m = beta1 * m + (1 - beta1) * g          # First moment
    v = beta2 * v + (1 - beta2) * g**2       # Second moment
    m_hat = m / (1 - beta1**t)               # Bias correction
    v_hat = v / (1 - beta2**t)               # Bias correction
    delta = lr * m_hat / (np.sqrt(v_hat) + eps)  # Update
    return m, v, delta
```

### E.2 AdamW - Weight Decay Decoupling

Standard Adam with L2 regularization: add $\lambda \|x\|^2$ to the loss. This makes the update:
$$\delta = \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon} + \lambda x_t$$

The weight decay term is scaled by $\hat{v}_t^{-1/2}$ - large for coordinates with small gradient variance. **AdamW** decouples weight decay from the adaptive scaling:
$$\delta = \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon} + \lambda x_t \cdot \text{(no adaptive scaling)}$$

This is a better regularizer: all weights decay at the same rate, not faster for rarely-updated weights.

### E.3 AdaFactor - Memory-Efficient Adam

**AdaFactor** (Shazeer & Stern, 2018) reduces Adam's $O(d)$ memory for second moments by approximating the second moment matrix as a rank-1 outer product:
$$v_t \approx r_t c_t^\top \quad \text{where } r_t \in \mathbb{R}^m, c_t \in \mathbb{R}^n$$

For a weight matrix $W \in \mathbb{R}^{m \times n}$, this reduces memory from $O(mn)$ to $O(m + n)$.

**Trade-off:** Less accurate second moment estimate -> slightly worse convergence in practice, but viable for very large models where Adam's memory cost is prohibitive.

### E.4 Shampoo - Block Preconditioning

**Shampoo** (Gupta et al., 2018) applies a full-matrix preconditioner to each layer:

For layer weight $W \in \mathbb{R}^{m \times n}$:
$$L_t = L_{t-1} + \frac{dL}{dW} \left(\frac{dL}{dW}\right)^\top \in \mathbb{R}^{m \times m}$$
$$R_t = R_{t-1} + \left(\frac{dL}{dW}\right)^\top \frac{dL}{dW} \in \mathbb{R}^{n \times n}$$

Update: $W \leftarrow W - \alpha L_t^{-1/4} \frac{dL}{dW} R_t^{-1/4}$

This approximates the natural gradient (full Fisher) with only $O(m^2 + n^2)$ memory per layer, vs $O((mn)^2)$ for full Fisher. Shampoo is used in Google's LLM pre-training (via distributed Shampoo implementation).

---

## Appendix F: Convergence Analysis Review

### F.1 Gradient Descent Convergence

**Smooth convex case ($f$ has $L$-Lipschitz gradients, $f$ convex):**
$$f(x_k) - f(x^*) \leq \frac{L \|x_0 - x^*\|^2}{2k}$$
with step $\eta = 1/L$.

**Strongly convex case ($f$ is $\mu$-strongly convex, $L$-smooth):**
$$f(x_k) - f(x^*) \leq \left(1 - \frac{\mu}{L}\right)^k (f(x_0) - f(x^*))$$

Convergence rate: $\kappa = L/\mu$ iterations to reduce error by $e^{-1}$.

**Non-convex case:** $\min_{k \leq T} \|\nabla f(x_k)\|^2 \leq \frac{2(f(x_0) - f^*)}{T \eta (2 - L\eta)}$

### F.2 Newton's Method Convergence

Near a non-degenerate minimum $x^*$:
$$\|x_{k+1} - x^*\| \leq \frac{M}{2\lambda_{\min}(H^*)} \|x_k - x^*\|^2$$

where $M$ is the Lipschitz constant of $\nabla^2 f$ and $\lambda_{\min}(H^*)$ is the smallest eigenvalue of the Hessian at $x^*$.

**Basin of convergence:** Newton's method converges from any $x_0$ satisfying $\|x_0 - x^*\| < \frac{2\lambda_{\min}(H^*)^2}{M \kappa(H^*)}$.

### F.3 BFGS Convergence

For strongly convex $f$ with Lipschitz continuous Hessian, BFGS converges super-linearly:
$$\lim_{k \to \infty} \frac{\|x_{k+1} - x^*\|}{\|x_k - x^*\|} = 0$$

and $r$-linearly convergent globally:
$$\|x_k - x^*\| \leq c \rho^k$$

for some $c > 0$ and $\rho \in (0, 1)$.

---

## Appendix G: Software Ecosystem

### G.1 SciPy Optimize

```python
from scipy.optimize import minimize, check_grad

# L-BFGS-B
result = minimize(f, x0, jac=grad_f, method='L-BFGS-B',
                  options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10})

# Newton-CG
result = minimize(f, x0, jac=grad_f, hess=hess_f, method='Newton-CG')

# Trust region methods
result = minimize(f, x0, jac=grad_f, hess=hess_f, method='trust-ncg')
result = minimize(f, x0, jac=grad_f, hess=hess_f, method='trust-exact')

# Gradient checking
err = check_grad(f, grad_f, x0)  # Should be < 1e-5 for correct gradient
```

### G.2 PyTorch Optimizers

```python
import torch.optim as optim

# Standard first-order
sgd    = optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=1e-4)
adam   = optim.Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
adamw  = optim.AdamW(params, lr=1e-3, weight_decay=0.01)

# Second-order
lbfgs  = optim.LBFGS(params, lr=1, max_iter=20, history_size=100,
                      line_search_fn='strong_wolfe')

# Usage with closure (required for LBFGS and some line search methods)
def closure():
    optimizer.zero_grad()
    loss = criterion(model(X), y)
    loss.backward()
    return loss

optimizer.step(closure)
```

### G.3 JAX Optimizers (Optax)

```python
import optax

# Standard optimizers
optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(params)

# One step
grads = jax.grad(loss_fn)(params)
updates, opt_state = optimizer.update(grads, opt_state, params)
params = optax.apply_updates(params, updates)

# Composing transformations
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),  # Gradient clipping
    optax.scale_by_adam(),           # Adam scaling
    optax.scale(-lr),                # Learning rate
)
```

---

## Appendix H: Summary and Quick Reference

```
NUMERICAL OPTIMIZATION - ALGORITHM SELECTION GUIDE
========================================================================

  STOCHASTIC (mini-batch, noisy gradients)
  -----------------------------------------
  Adam (default for LLMs/transformers)
  AdamW (with weight decay - preferred for fine-tuning)
  SGD + momentum (slower to converge, better generalization sometimes)
  Adafactor (when memory is very constrained)
  Shampoo/K-FAC (when second-order info is affordable)

  FULL-BATCH (all data, low noise)
  ---------------------------------
  L-BFGS (the gold standard for full-batch)
  Nonlinear CG (memory efficient, similar to L-BFGS)
  Newton-CG (when Hessian-vector products are cheap)
  Trust-region (when Hessian is indefinite)

  CONVERGENCE RATE COMPARISON (quadratic objective,  = 100)
  ---------------------------------------------------------
  GD (optimal step):    ~100 iters to 1e-6 (rate = (-1)/(+1) = 0.98)
  CG:                   ~10 iters to 1e-6 (rate \\approx (\\sqrt-1)/(\\sqrt+1) = 0.82)
  L-BFGS (m=10):        ~10 iters to 1e-6 (super-linear)
  Newton:               ~3 iters to 1e-6 (quadratic convergence!)

  GRADIENT CHECKING STEP SIZE
  ----------------------------
  Forward differences:  h = sqrt(eps_mach) \\approx 1.5e-8
  Centered differences: h = cbrt(eps_mach) \\approx 6e-6
  Complex step:         h = 1e-20 (essentially exact)

========================================================================
```

*[<- Back to Section02 Numerical Linear Algebra](../02-Numerical-Linear-Algebra/notes.md) | [Next: Section04 Interpolation and Approximation ->](../04-Interpolation-and-Approximation/notes.md)*

---

## Appendix I: Extended Case Studies

### I.1 Why Adam Doesn't Use Line Search

Deterministic optimization (with exact gradients) benefits greatly from line search: at each step, we can evaluate $f$ as many times as needed to find a good step size. But stochastic optimization with mini-batches cannot use line search:

**Problem 1: Inconsistent function values.** The function value $f(x; B_k)$ computed on batch $B_k$ is different from $f(x; B_{k+1})$ computed on the next batch. We cannot meaningfully compare $f(x_k; B_k)$ to $f(x_k + \alpha p; B_k)$ if the batch changes between evaluation.

**Problem 2: Gradient noise breaks secant condition.** The L-BFGS secant condition $y_k^\top s_k > 0$ requires the gradient change $y_k = g_{k+1} - g_k$ to be in a specific relationship to the step $s_k$. With stochastic gradients, $y_k$ is corrupted by noise, and the condition may fail.

**Adam's solution:** Instead of adapting the step size using function evaluations (line search), Adam adapts using the history of gradient squares ($v_t$). This is "implicit line search" - steps are small where gradients have been large historically, and large where gradients have been small.

### I.2 When L-BFGS Beats Adam

**Task:** Minimize the loss of a 2-layer network on a small dataset (N=1000, d=100).

| Method | Iterations | Final loss | Time |
|--------|-----------|------------|------|
| Adam ($\eta=10^{-3}$) | 10,000 | $3.2 \times 10^{-5}$ | 45s |
| Adam ($\eta=10^{-4}$) | 10,000 | $8.1 \times 10^{-4}$ | 45s |
| L-BFGS ($m=20$) | 150 | $1.1 \times 10^{-8}$ | 12s |

L-BFGS reaches much lower loss faster when:
- Gradient is exact (full batch)
- The loss function is smooth (no noise)
- High accuracy is needed (physics simulation, calibration)

**But for LLM pretraining (N=$10^{12}$, d=$10^{10}$):** L-BFGS is impractical. Adam is the right choice.

### I.3 The Role of Condition Number in Transformer Training

During LLM pretraining, the condition number of the loss Hessian at different points in training:

- **Early training ($t = 0$):** Random initialization, $\kappa \sim O(1)$ for linear layers. Gradients flow well.
- **Mid training:** Parameters have adapted, $\kappa$ can reach $10^3$ to $10^6$ for some layers. This is where gradient clipping matters.
- **Late training:** Near convergence, $\kappa$ stabilizes. The learning rate schedule (cosine decay, linear decay) effectively adapts to the local curvature.

**Weight decay effect on condition number:** Adding L2 regularization $\lambda \|W\|_F^2$ shifts all Hessian eigenvalues by $+2\lambda$, changing condition number from $\kappa = \lambda_{\max}/\lambda_{\min}$ to $(\lambda_{\max} + 2\lambda) / (\lambda_{\min} + 2\lambda)$, which is smaller (better conditioned). This is one reason AdamW (with explicit weight decay) often trains more stably than Adam with L2 in the loss.

---

## Appendix J: Numerical Analysis of Backpropagation

### J.1 Floating-Point Error in Backprop

The backpropagation algorithm computes $\partial L / \partial W$ by the chain rule:
$$\delta_l = \frac{\partial L}{\partial z_l} = f'(z_l) \odot (W_{l+1}^\top \delta_{l+1})$$

In floating-point, each multiplication and activation application introduces $O(\varepsilon_{\text{mach}})$ relative error. For an $L$-layer network, the accumulated gradient error is:
$$\|\delta \hat{g}\| / \|g\| = O(L \varepsilon_{\text{mach}} \cdot \kappa_{\text{jacobian}})$$

where $\kappa_{\text{jacobian}}$ is the condition number of the Jacobian of the network (which can be large for deep networks with poor initialization).

### J.2 Mixed Precision and Gradient Accuracy

In bf16 training, gradients are computed in bf16 with $\varepsilon_{\text{mach}} \approx 8 \times 10^{-3}$. This means:
- Parameter updates have $\sim 3$ significant digits of accuracy
- Accumulated over many steps, this can introduce drift

**Why it works despite low precision:** The optimizer (Adam) effectively computes a smoothed gradient over many steps via the exponential moving average $m_t$. Noise in individual gradient estimates is averaged out. The effective gradient accuracy is $O(\varepsilon_{\text{bf16}} / \sqrt{T})$ after $T$ steps, which is much better than single-step precision.

### J.3 Gradient Accumulation and Precision

When accumulating gradients over $K$ mini-batches before an update:
$$G = \frac{1}{K} \sum_{k=1}^K g_k$$

In fp32 with Kahan summation: error $= O(\varepsilon_{\text{fp32}})$ independent of $K$.
In bf16 with naive summation: error $= O(K \varepsilon_{\text{bf16}})$ - grows with accumulation steps.

**Fix:** Accumulate gradients in fp32, even if the forward/backward pass uses bf16. This is the default behavior in PyTorch AMP when `scaler.scale(loss).backward()` is used.

---

## Appendix K: Connection to Statistical Learning Theory

Numerical optimization connects to statistical learning theory through the bias-variance tradeoff:

**Optimization error:** $f(x_k) - f(x^*)$ - how far we are from the training loss minimum.

**Approximation error:** $f(x^*) - f^*_{\text{Bayes}}$ - how far the best model in our function class is from the Bayes optimal.

**Estimation error:** $f^*_{\text{Bayes}} - $ (error on new data) - generalization gap.

**The over-optimization paradox:** Running optimization to convergence (small optimization error) often leads to overfitting (large estimation error). This is why:
- We use early stopping (stop before convergence)
- We use weight decay (adds regularization that changes $x^*$)
- We use dropout and other regularizers (change the optimization landscape)

**Implicit regularization in SGD:** Stochastic gradient descent, by virtue of its noise, converges to "flat minima" - regions where the Hessian has small eigenvalues. Such minima generalize better (Hochreiter & Schmidhuber, 1997; Keskar et al., 2016). This is NOT a numerical issue - it's a feature of the stochastic optimization process.

---

## Appendix L: Historical Timeline of Optimization Methods

```
OPTIMIZATION METHODS: TIMELINE
========================================================================

  1847  Cauchy - steepest descent method (gradient descent)
  1944  Levenberg - damped least squares (LM algorithm)
  1952  Kaczmarz - projection-based iterative method
  1952  Hestenes & Stiefel - conjugate gradient method
  1959  Broyden, Fletcher, Goldfarb, Shanno (BFGS) - quasi-Newton
  1963  Marquardt - generalized LM algorithm
  1970  Sargent & Sebastian - first L-BFGS-like method
  1980  Nocedal - limited-memory BFGS
  1988  Armijo - sufficient decrease condition
  1969  Wolfe - curvature condition for line search
  1994  Byrd, Lu, Nocedal, Zhu - L-BFGS-B (bounds constrained)
  1997  Hochreiter & Schmidhuber - flat minima and generalization
  2010  Martens - Hessian-free optimization for deep nets
  2012  Duchi et al. - Adagrad
  2014  Kingma & Ba - Adam
  2015  Bottou - stochastic optimization in ML survey
  2016  Keskar et al. - sharp vs flat minima for generalization
  2017  Loshchilov & Hutter - AdamW (decoupled weight decay)
  2018  Gupta et al. - Shampoo (block-diagonal preconditioning)
  2018  Anil et al. - Scalable second-order optimization
  2021+ Vyas et al., Dettmers - training efficiency, mixed precision

========================================================================
```

---

## Appendix M: Chapter Position and Summary

```
Section10 NUMERICAL METHODS - SECTION 03 SUMMARY
========================================================================

  PREREQUISITE CHAIN:
  Section01 Floating Point -> Section02 Numerical LA -> Section03 Numerical Optimization

  CONNECTIONS FROM THIS SECTION:
    +--------------------------------------------------------------+
    |  Line search (Wolfe)      -> Used in L-BFGS, BFGS            |
    |  Newton's method          -> Quasi-Newton (BFGS, L-BFGS)     |
    |  L-BFGS                   -> torch.optim.LBFGS               |
    |  Trust region             -> TRPO, PPO (RL policy updates)    |
    |  Diagonal quasi-Newton    -> Adam, AdaGrad, RMSProp          |
    |  Natural gradient         -> K-FAC, Shampoo                  |
    |  Condition numbers        -> Section02 Numerical LA                 |
    |  Gradient checking        -> Debugging autodiff implementations|
    +--------------------------------------------------------------+

  KEY FORMULAS:
    GD convergence rate:   (-1)/(+1) per iteration
    CG convergence rate:   (\\sqrt-1)/(\\sqrt+1) per iteration
    Newton convergence:    quadratic near minimum
    L-BFGS convergence:    super-linear (between GD and Newton)

    Optimal finite diff h: cbrt(eps_mach) \\approx 6e-6  (centered)
    Finite diff error:     O(h^2) + O(eps/h)
    Complex step error:    O(h^2) - no rounding error

========================================================================
```

*Section statistics: 12 main sections, 22 subsections, 12 common mistakes, 10 exercises, 13 appendices.*

*[<- Back to Section02 Numerical Linear Algebra](../02-Numerical-Linear-Algebra/notes.md) | [Next: Section04 Interpolation and Approximation ->](../04-Interpolation-and-Approximation/notes.md)*

---

## Appendix N: Additional Optimization Algorithms

### N.1 Nonlinear Conjugate Gradient

The **nonlinear conjugate gradient (NCG)** method extends CG to nonlinear objectives by using the same direction-update structure but with gradient changes instead of exact Hessian-vector products.

The search directions are:
$$d_{k+1} = -g_{k+1} + \beta_k d_k$$

where the scalar $\beta_k$ determines which variant of NCG:

| Formula | Name | Best for |
|---------|------|----------|
| $\beta_k = \frac{g_{k+1}^\top g_{k+1}}{g_k^\top g_k}$ | Fletcher-Reeves | Smooth convex |
| $\beta_k = \frac{g_{k+1}^\top (g_{k+1} - g_k)}{g_k^\top g_k}$ | Polak-Ribiere | General nonlinear |
| $\beta_k = \max\left(\frac{g_{k+1}^\top (g_{k+1} - g_k)}{d_k^\top (g_{k+1} - g_k)}, 0\right)$ | Hestenes-Stiefel | Non-convex |

**NCG vs L-BFGS:**
- NCG: $O(d)$ memory (just the previous direction)
- L-BFGS ($m$ vectors): $O(md)$ memory, faster convergence
- NCG: simpler to implement, same asymptotic convergence order

### N.2 Stochastic Gradient Descent with Momentum

**SGD + Momentum:**
$$v_{t+1} = \mu v_t - \alpha g_t, \quad x_{t+1} = x_t + v_{t+1}$$

Nesterov accelerated gradient (NAG):
$$v_{t+1} = \mu v_t - \alpha \nabla f(x_t + \mu v_t), \quad x_{t+1} = x_t + v_{t+1}$$

**Connection to CG:** Heavy ball (momentum SGD) is the Riemann form of CG for quadratic objectives. For non-quadratic objectives, momentum helps navigate ravines (valleys where the gradient is large perpendicular to the valley but small along it).

**Optimal momentum for quadratics:** $\mu^* = \left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^2$. With this momentum, gradient descent with momentum has the same convergence rate as CG: $O(\sqrt{\kappa})$ vs $O(\kappa)$ without momentum.

### N.3 Stochastic Variance Reduction

For finite-sum objectives $f(x) = \frac{1}{n} \sum_i f_i(x)$, **variance-reduced** methods achieve better convergence than SGD by occasionally computing the full gradient:

**SVRG (Johnson & Zhang, 2013):**
```
for outer loop s = 1, 2, ..., S:
    compute full gradient \\mu = f(x_s)
    for inner loop t = 1, ..., m:
        sample i uniformly
        g_t = f_i(x_t) - f_i(x_s) + \\mu  # Variance-reduced gradient
        x_{t+1} = x_t - \\alpha g_t
    x_{s+1} = x_{t_m} (some iterate from inner loop)
```

The key: $\mathbb{E}[g_t] = \nabla f(x_t)$ (unbiased) and $\text{Var}(g_t) \to 0$ as $x_t \to x^*$ (variance vanishes at optimum). SVRG achieves linear convergence (like GD) with per-iteration cost similar to SGD.

**For AI:** SVRG is rarely used for deep learning because the gradient computation involves complex non-linearities (making the variance reduction calculation non-trivial) and because large datasets make computing the full gradient expensive. Adam with decaying learning rate achieves comparable practical performance without the exact full gradient.

---

## Appendix O: Extended Exercises

### O.1 Implementing the Wolfe Zoom Algorithm (***)

The Wolfe zoom algorithm is the industry-standard line search for L-BFGS. Implement it from scratch:

```python
def wolfe_zoom(f, g, x, p, alpha_lo, alpha_hi, phi_lo, phi0, dphi0,
               c1=1e-4, c2=0.9, max_iter=20):
    """
    Find alpha satisfying strong Wolfe conditions within [alpha_lo, alpha_hi].
    phi_lo = f(x + alpha_lo * p), phi0 = f(x), dphi0 = g(x) @ p
    """
    for i in range(max_iter):
        # Cubic interpolation to find trial step
        alpha_j = 0.5 * (alpha_lo + alpha_hi)  # Simplified: bisection
        phi_j = f(x + alpha_j * p)

        if phi_j > phi0 + c1 * alpha_j * dphi0 or phi_j >= phi_lo:
            alpha_hi = alpha_j
        else:
            dphi_j = g(x + alpha_j * p) @ p
            if abs(dphi_j) <= -c2 * dphi0:
                return alpha_j  # Strong Wolfe satisfied
            if dphi_j * (alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo
            alpha_lo = alpha_j
            phi_lo = phi_j

    return alpha_j  # Return best found
```

### O.2 Numerical Gradient Check via PyTorch (*)

```python
import torch
from torch.autograd import gradcheck

# Define a custom function
class MyFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x ** 3 + torch.sin(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output * (3 * x**2 + torch.cos(x))

# Gradient check with double precision
x = torch.randn(5, dtype=torch.float64, requires_grad=True)
result = gradcheck(MyFunc.apply, x, eps=1e-6, atol=1e-4, rtol=1e-3)
print(f"Gradient check passed: {result}")
```

### O.3 Computing Hessian Spectrum via Lanczos (***)

The Hessian spectrum of a trained neural network reveals the curvature structure of the loss landscape:

```python
def hessian_spectrum_lanczos(loss, params, n_iters=100):
    """
    Compute the extremal eigenvalues of the Hessian via Lanczos iteration.
    Each iteration requires one Hessian-vector product (one backward pass).
    """
    d = sum(p.numel() for p in params)

    # Initial vector
    v = [torch.randn_like(p) for p in params]
    v_norm = sum(torch.sum(vi**2) for vi in v).sqrt()
    v = [vi / v_norm for vi in v]

    alphas, betas = [], []

    for i in range(n_iters):
        # Hessian-vector product
        Hv = hvp(loss, params, v)

        alpha = sum(torch.sum(vi * Hvi) for vi, Hvi in zip(v, Hv)).item()
        alphas.append(alpha)

        # Orthogonalize
        w = [Hvi - alpha * vi for Hvi, vi in zip(Hv, v)]
        if i > 0:
            w = [wi - beta * vi_old for wi, vi_old in zip(w, v_old)]

        beta = sum(torch.sum(wi**2) for wi in w).sqrt().item()
        betas.append(beta)

        v_old = v
        v = [wi / beta for wi in w]

    # The Lanczos tridiagonal matrix T has diag=alphas, off-diag=betas
    T = np.diag(alphas) + np.diag(betas[:-1], 1) + np.diag(betas[:-1], -1)
    eigvals = np.linalg.eigvalsh(T)
    return eigvals
```

---

## Appendix P: Key Theorems Reference

### P.1 Convergence Theorems

**Theorem 1 (Gradient Descent):** For $f$ $L$-smooth with $\eta = 1/L$:
- Convex: $f(x_k) - f^* = O(1/k)$
- $\mu$-strongly convex: $\|x_k - x^*\| = O((1 - \mu/L)^k)$

**Theorem 2 (CG):** For SPD $A$ with condition number $\kappa = \lambda_{\max}/\lambda_{\min}$:
$$\frac{\|e_k\|_A}{\|e_0\|_A} \leq 2\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^k$$

**Theorem 3 (Newton):** Near a non-degenerate minimum, Newton's method converges quadratically:
$$\|x_{k+1} - x^*\| \leq C \|x_k - x^*\|^2$$

**Theorem 4 (BFGS):** For strongly convex $f$ with Lipschitz Hessian, BFGS converges super-linearly.

**Theorem 5 (Zoutendijk):** For descent direction $p_k$ with Wolfe line search:
$$\sum_k \frac{|\nabla f_k^\top p_k|^2}{\|p_k\|^2} < \infty \implies \liminf \|\nabla f_k\| = 0$$

### P.2 Optimality Conditions

**First-order necessary:** $\nabla f(x^*) = 0$

**Second-order sufficient:** $\nabla f(x^*) = 0$ and $\nabla^2 f(x^*) \succ 0$ (positive definite Hessian)

**KKT conditions** (constrained): See [Section08-04 Constrained Optimization](../../08-Optimization/04-Constrained-Optimization/notes.md)

---

*End of Section03 Numerical Optimization*

*[<- Back to Section02 Numerical Linear Algebra](../02-Numerical-Linear-Algebra/notes.md) | [Next: Section04 Interpolation and Approximation ->](../04-Interpolation-and-Approximation/notes.md)*

---

## Appendix Q: Deep Dive - The Rosenbrock Function

The **Rosenbrock function** $f(x, y) = 100(y - x^2)^2 + (1-x)^2$ is the canonical test for optimization algorithms:

**Properties:**
- Minimum at $(1, 1)$ with $f(1,1) = 0$
- The valley $y = x^2$ is nearly flat along the $x$ direction but steep perpendicular to it
- Condition number of the Hessian at the minimum: $\kappa(H^*) = 2508$ (for the standard form)
- Not convex globally, but has a unique global minimum

**Hessian at the minimum $(1,1)$:**
$$H = \begin{pmatrix} 802 & -400 \\ -400 & 200 \end{pmatrix}$$

Eigenvalues: $\lambda_1 \approx 1001$, $\lambda_2 \approx 1$.

**Performance comparison on Rosenbrock:**
| Method | Iterations to $10^{-8}$ |
|--------|------------------------|
| Gradient descent (optimal $\eta$) | $\approx 10^5$ |
| CG (Fletcher-Reeves) | $\approx 1000$ |
| L-BFGS ($m=5$) | $\approx 30$ |
| Newton (with line search) | $\approx 15$ |
| BFGS | $\approx 40$ |

**Why it's hard for gradient descent:** The "banana-shaped" valley means gradient descent oscillates across the valley and makes tiny progress along the valley. Line search helps but doesn't fix the fundamental curvature mismatch.

**Why L-BFGS works well:** After a few steps, L-BFGS builds up good curvature information along the valley direction, allowing it to take large steps along the valley.

### Q.1 Numerical Analysis of Rosenbrock

```python
import numpy as np

def rosenbrock(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def rosenbrock_grad(x):
    g = np.zeros(2)
    g[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    g[1] = 200 * (x[1] - x[0]**2)
    return g

def rosenbrock_hess(x):
    H = np.zeros((2, 2))
    H[0, 0] = -400 * (x[1] - x[0]**2) + 800 * x[0]**2 + 2
    H[0, 1] = -400 * x[0]
    H[1, 0] = -400 * x[0]
    H[1, 1] = 200
    return H

# Condition number at the minimum
H_star = rosenbrock_hess(np.array([1.0, 1.0]))
eigvals = np.linalg.eigvalsh(H_star)
kappa = eigvals.max() / eigvals.min()
print(f"kappa(H*) = {kappa:.2f}")
# Expected: ~2508
```

---

## Appendix R: Optimization Landscape Analysis for LLMs

Recent research (2020-2024) has revealed interesting properties of the optimization landscape for large language models:

### R.1 The Edge of Stability

**Edge of Stability (EoS) phenomenon** (Cohen et al., 2021): During neural network training with a constant learning rate $\eta$, the largest Hessian eigenvalue $\lambda_{\max}(H)$ tends to increase until it reaches $\lambda_{\max} \approx 2/\eta$ and then stabilizes there.

**Implication:** The loss oscillates but decreases on average. The learning rate implicitly controls the curvature of the loss landscape that training "settles into."

**Numerical analysis view:** $2/\eta$ is the stability threshold for gradient descent ($\eta < 2/\lambda_{\max}$ for convergence). The EoS phenomenon shows that neural networks are often training at the boundary of instability - aggressive but not diverging.

### R.2 Catastrophic Forgetting and Flat Minima

**Observation (Hochreiter & Schmidhuber, 1997; Keskar et al., 2016):** Models trained with small batches (high gradient noise) tend to find "flatter" minima where $\lambda_{\max}(H^*)$ is small. Models trained with large batches find "sharper" minima with large $\lambda_{\max}$.

**Flat minima generalize better** because small perturbations to the weights (due to test distribution shift, quantization, etc.) cause smaller changes in the loss.

**Numerical measure of sharpness:**
$$\text{Sharpness} = \max_{\|\delta\| \leq \epsilon} \frac{f(x^* + \delta) - f(x^*)}{\epsilon^2} \approx \lambda_{\max}(\nabla^2 f(x^*))$$

**SAM (Sharpness-Aware Minimization, Foret et al., 2021)** explicitly minimizes sharpness by finding the worst-case perturbation and minimizing there:
$$\min_x \max_{\|\epsilon\| \leq \rho} f(x + \epsilon) \approx \min_x f\left(x + \rho \frac{\nabla f}{\|\nabla f\|}\right)$$

This requires two gradient evaluations per step but consistently improves generalization.

### R.3 Loss Landscape at Scale

For billion-parameter models (GPT-3, LLaMA, etc.), direct computation of the Hessian is impossible. Researchers use:

1. **Implicit Hessian-vector products** (via Pearlmutter's trick) to estimate Hessian properties
2. **Lanczos iteration** to find the leading eigenvalues of the Hessian
3. **Local quadratic approximation** using finite differences around a trained model

**Key findings (Ghorbani et al., 2019; Yao et al., 2020):**
- The Hessian has a "bulk + outlier" structure: most eigenvalues are near zero (flat directions), with a few large eigenvalues (curved directions)
- The number of large eigenvalues grows with model capacity
- The condition number at convergence is surprisingly manageable ($\kappa \sim 10^3$ to $10^4$) for well-trained models

---

## Appendix S: Practical Implementation Notes

### S.1 Warm Starting L-BFGS

When fine-tuning a pre-trained model with L-BFGS, warm-start the optimizer by providing the final state from a previous run:

```python
# Load pre-trained weights
model.load_state_dict(torch.load('pretrained.pt'))

# Create L-BFGS optimizer
optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, history_size=20)

# Optionally load optimizer state for warm start
try:
    optimizer.load_state_dict(torch.load('optimizer_state.pt'))
    print("Warm-starting from previous optimizer state")
except FileNotFoundError:
    print("Starting fresh L-BFGS")
```

### S.2 Debugging Failed Optimization

**Symptom: Loss not decreasing after several steps**
- Check gradient norm: `g_norm = sum(p.grad.norm()**2 for p in params) ** 0.5`
- If $\|g\| < \varepsilon$: already at a local minimum or learning rate too small
- If $\|g\|$ is large: check for NaN gradients, verify loss computation

**Symptom: Loss oscillating or diverging**
- Reduce learning rate by 10x
- Check condition number: `kappa = max_sv / min_sv` for key weight matrices
- Add gradient clipping: `torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)`

**Symptom: L-BFGS says "function evaluation limit reached"**
- Increase `max_eval` parameter
- Reduce model complexity or regularize more strongly
- Check if the closure is deterministic (L-BFGS requires consistent function values)

### S.3 Hyperparameter Sensitivity

For Adam in practice:
- **Learning rate $\alpha$**: most sensitive; typical values $10^{-4}$ to $10^{-3}$ for LLMs
- **$\beta_1 = 0.9$, $\beta_2 = 0.999$**: largely insensitive; $\beta_2 = 0.95$ for very noisy gradients
- **$\varepsilon = 10^{-8}$**: increase to $10^{-6}$ or $10^{-7}$ for numerical stability in fp16/bf16
- **Weight decay**: treat as regularization, not a numerical parameter; $10^{-2}$ for large models

For L-BFGS:
- **history_size $m$**: 10-100; larger $m$ = better curvature approximation but more memory
- **max_iter**: 5-20 CG iterations per step
- **line_search_fn**: always use `'strong_wolfe'` for robustness

---

*End of Section03 Numerical Optimization Notes (1600+ lines)*

*[<- Section02 Numerical Linear Algebra](../02-Numerical-Linear-Algebra/notes.md) | [-> Section04 Interpolation and Approximation](../04-Interpolation-and-Approximation/notes.md)*

---

## Appendix T: Convergence Rate Summary Tables

### T.1 First-Order Methods

| Method | Cost/Iter | Conv. Rate (smooth convex) | Conv. Rate (strongly convex, ) |
|--------|-----------|---------------------------|----------------------------------|
| Gradient descent | $O(d)$ | $O(1/k)$ | $O((1 - 1/\kappa)^k)$ |
| GD (optimal step) | $O(d)$ | $O(L/k)$ | $O(((-1)/(+1))^k)$ |
| GD + momentum | $O(d)$ | $O(1/k^2)$ (Nesterov) | $O(((\sqrt{}-1)/(\sqrt{}+1))^{2k})$ |
| CG | $O(d)$ | $O(1/k^2)$ | $O(((\sqrt{}-1)/(\sqrt{}+1))^k)$ |
| SGD | $O(d)$ | $O(1/\sqrt{k})$ | $O(1/k)$ |
| Adam | $O(d)$ | $O(1/\sqrt{k})$ | Empirically fast |

### T.2 Second-Order and Quasi-Newton Methods

| Method | Memory | Cost/Iter | Convergence |
|--------|--------|-----------|-------------|
| Newton | $O(d^2)$ | $O(d^3)$ | Quadratic near $x^*$ |
| Newton-CG | $O(d)$ | $O(d)$ per CG step | Super-linear |
| BFGS | $O(d^2)$ | $O(d^2)$ | Super-linear |
| L-BFGS | $O(md)$ | $O(md)$ | Super-linear locally |
| Diagonal precond. | $O(d)$ | $O(d)$ | Linear, rate $((\sqrt{_D}-1)/...)^k$ |

where $\kappa_D$ is the condition number after diagonal preconditioning.

### T.3 Trust Region vs Line Search

| Aspect | Line Search | Trust Region |
|--------|-------------|--------------|
| Convergence guarantee | Yes (Wolfe) | Yes (ratio test) |
| Non-convex Hessian | Needs modification | Natural handling |
| Implementation complexity | Moderate | Higher |
| Typical algorithms | BFGS, L-BFGS | Dog-leg, GLTR |

---

## Appendix U: Worked Numerical Example - L-BFGS Step

**Setup:** Minimize $f(x) = x_1^2 + 10x_2^2$ starting from $x_0 = (2, 1)$.

**Step 0:** $x_0 = (2, 1)$, $g_0 = (4, 20)$. Since $m=0$, initial direction $d_0 = -g_0 = (-4, -20)$.

Line search: $f(x_0 + \alpha d_0) = (2 - 4\alpha)^2 + 10(1 - 20\alpha)^2$. Minimize: $\alpha^* = 51/4040 \approx 0.01263$.

**After step 0:** $x_1 = (2 - 4 \cdot 0.01263, 1 - 20 \cdot 0.01263) = (1.9495, 0.7475)$.

$g_1 = (3.899, 14.95)$.

**Update L-BFGS history:**
$s_0 = x_1 - x_0 = (-0.0505, -0.2525)$
$y_0 = g_1 - g_0 = (-0.101, -5.05)$
$\rho_0 = 1/(y_0^\top s_0) = 1/(0.101 \cdot 0.0505 + 5.05 \cdot 0.2525) = 1/0.2798 \approx 3.574$

**Step 1 direction (L-BFGS, $m=1$):**

First loop: $\alpha_0 = \rho_0 (s_0^\top g_1) = 3.574 \cdot ((-0.0505)(3.899) + (-0.2525)(14.95)) = 3.574 \cdot (-4.003) \approx -14.31$

$q = g_1 - \alpha_0 y_0 = (3.899 - (-14.31)(-0.101), 14.95 - (-14.31)(-5.05))$
$= (3.899 - 1.445, 14.95 - 72.27) = (2.454, -57.32)$

Scale: $\gamma = s_0^\top y_0 / (y_0^\top y_0) = 0.2798 / (0.101^2 + 5.05^2) \approx 0.01097$

$r = \gamma q = (0.02692, -0.629)$

Second loop: $\beta_0 = \rho_0 (y_0^\top r) = 3.574 \cdot ((-0.101)(0.02692) + (-5.05)(-0.629)) = 3.574 \cdot 3.171 \approx 11.33$

$r \leftarrow r + s_0 (\alpha_0 - \beta_0) = (0.02692, -0.629) + (-0.0505)(-14.31 - 11.33) \cdot (1, 0) + (-0.2525)(-14.31 - 11.33) \cdot (0, 1)$
$= (0.02692 + 1.294, -0.629 + 6.463) = (1.321, 5.834)$

Direction: $d_1 = -r = (-1.321, -5.834)$.

This is much better aligned with the steepest descent direction scaled by approximate Hessian inverse: true Newton direction is $H^{-1}g_1 = (3.899/2, 14.95/20) = (1.950, 0.748)$, so $d_1 \approx -H^{-1}g_1$ within a scaling factor.

---

*[<- Section02 Numerical Linear Algebra](../02-Numerical-Linear-Algebra/notes.md) | [-> Section04 Interpolation and Approximation](../04-Interpolation-and-Approximation/notes.md)*

---

## Appendix V: Extended Exercises with Solutions

### V.1 Verifying Adam's Diagonal Preconditioning (**)

**Problem:** On the quadratic $f(x) = \frac{1}{2} x^\top H x$ with $H = \text{diag}(h_1, \ldots, h_d)$:

(a) After many steps of Adam (small mini-batches with noise $\sigma_i^2 = h_i$ per coordinate), the second moment $v_\infty \approx h_i$.

(b) Show that Adam's effective step in coordinate $i$ is approximately:
$$\Delta x_i \approx -\frac{\alpha}{\sqrt{h_i} + \varepsilon} g_i \approx -\frac{\alpha}{\sqrt{h_i}} g_i$$

This is gradient descent with step $\alpha / \sqrt{h_i}$ in coordinate $i$.

(c) For the quadratic $f$, this corresponds to the Newton step $H^{-1} g$ with step size $\alpha / \sqrt{h_i} \cdot g_i / (h_i g_i) = \alpha / \sqrt{h_i}$ ... wait, the Newton step in coordinate $i$ is $g_i / h_i$. Adam gives $g_i / \sqrt{h_i}$ - this is a half-Newton step (square root of the Hessian diagonal, not the Hessian diagonal itself).

**This is why Adam is not full Newton:** Adam uses $\sqrt{H_{\text{diag}}}$ as the preconditioner, not $H_{\text{diag}}$. For a quadratic, the optimal preconditioner is $H$ (Newton), not $\sqrt{H}$.

**Consequence for convergence:** With Adam on a quadratic $H = \text{diag}(h_1, \ldots, h_d)$:
- Gradient descent rate: $1 - 2\alpha \mu$ where $\mu = \min h_i / \kappa = h_{\min}$
- Adam rate: $1 - 2\alpha / \sqrt{h_{\max}} \cdot h_{\min}^{1/2}$ - better but not optimal
- Optimal diagonal Newton rate: $1 - 2\alpha$ (constant, independent of $\kappa$)

### V.2 Finite Difference Gradient Check (*)

Implement a gradient check for a custom loss function:

```python
def custom_loss(W, X, y):
    """Binary cross-entropy with sigmoid activation."""
    z = X @ W
    p = 1 / (1 + np.exp(-z))
    return -np.mean(y * np.log(p + 1e-12) + (1-y) * np.log(1 - p + 1e-12))

def custom_grad(W, X, y):
    """Analytical gradient."""
    z = X @ W
    p = 1 / (1 + np.exp(-z))
    return X.T @ (p - y) / len(y)

# Test
n, d = 100, 5
X = np.random.randn(n, d)
y = (np.random.randn(n) > 0).astype(float)
W = np.random.randn(d)

# Numerical gradient
h = 1e-5
g_num = np.zeros(d)
for i in range(d):
    W_plus  = W.copy(); W_plus[i]  += h
    W_minus = W.copy(); W_minus[i] -= h
    g_num[i] = (custom_loss(W_plus, X, y) - custom_loss(W_minus, X, y)) / (2*h)

g_ana = custom_grad(W, X, y)
rel_err = np.abs(g_num - g_ana).max() / (np.abs(g_ana).max() + 1e-12)
print(f"Max relative error: {rel_err:.2e}")  # Should be < 1e-5
```

### V.3 Trust Region Radius Adaptation (**)

Implement the trust region ratio test and radius update:

```python
def trust_region_update(f, x, p, delta, f_k, m_decrease):
    """Update trust region radius based on ratio test."""
    f_new = f(x + p)
    actual = f_k - f_new
    rho = actual / m_decrease  # Ratio: actual/predicted decrease

    # Accept or reject step
    x_new = x + p if rho > 0.1 else x

    # Update radius
    if rho < 0.25:
        delta_new = 0.25 * delta
    elif rho > 0.75 and abs(np.linalg.norm(p) - delta) < 1e-8:
        delta_new = min(2 * delta, 1e6)  # Cap at maximum
    else:
        delta_new = delta

    return x_new, delta_new, rho
```

---

## Appendix W: Self-Assessment Checklist

After completing Section03, you should be able to:

**Core algorithms:**
- [ ] Implement backtracking Armijo line search
- [ ] Explain why the Wolfe curvature condition is needed for BFGS
- [ ] Implement the L-BFGS two-loop recursion
- [ ] Implement the trust region ratio test and radius update
- [ ] Compute numerical gradients and find the optimal step size

**Conceptual understanding:**
- [ ] Explain the connection between Adam and diagonal quasi-Newton methods
- [ ] Explain why L-BFGS doesn't work for stochastic optimization
- [ ] Explain why Newton's method is quadratically convergent
- [ ] State the convergence rate of CG vs gradient descent
- [ ] Explain when to use line search vs trust region

**AI connections:**
- [ ] Explain Adam's adaptive learning rate as diagonal preconditioning
- [ ] Explain why TRPO/PPO use trust region constraints
- [ ] Explain the role of the Wolfe line search in PyTorch's L-BFGS
- [ ] Describe Shampoo/K-FAC as block quasi-Newton methods
- [ ] Connect gradient clipping to projected gradient descent

*Total content: Section1-Section12 (main), Appendices A-W (extended). Full coverage of numerical optimization for ML practitioners.*

---

## Appendix X: GPU-Accelerated Optimization

### X.1 Batch Size and Optimization Geometry

The **batch size** has a profound effect on optimization:

**Large batch:** The gradient $g_B = \frac{1}{|B|} \sum_{i \in B} \nabla f_i(x)$ approximates the full gradient with low variance. This enables line search, quasi-Newton methods, and faster convergence per epoch.

**Small batch:** High gradient noise. Adam's second moment estimate $v_t$ is noisy. The training trajectory is stochastic, which (counterintuitively) improves generalization.

**Linear scaling rule** (Goyal et al., 2017): If you increase batch size by $k$, increase learning rate by $\sqrt{k}$ (for SGD) or by $k$ (for the warmup phase). This keeps the signal-to-noise ratio of gradient estimates roughly constant.

### X.2 Gradient Accumulation as Larger Batch

For very large models that don't fit in GPU memory, **gradient accumulation** simulates a larger batch by summing gradients over multiple forward-backward passes before applying the optimizer update:

```python
accumulation_steps = 4
optimizer.zero_grad()

for step, batch in enumerate(dataloader):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        loss = model(batch) / accumulation_steps  # Scale loss
    scaler.scale(loss).backward()

    if (step + 1) % accumulation_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

**Numerical consideration:** Accumulating gradients in fp32 (not bf16) is critical. If the gradient accumulator itself is in bf16, the accumulated error grows linearly with accumulation steps.

### X.3 ZeRO Optimization and Distributed Adam

**ZeRO** (Zero Redundancy Optimizer, Rajbhandari et al., 2020) shards the optimizer state across GPUs:

| ZeRO Stage | What's Sharded | Memory Savings |
|-----------|---------------|----------------|
| Stage 1 | Optimizer states ($m$, $v$ in Adam) | 4x |
| Stage 2 | + Gradients | 8x |
| Stage 3 | + Parameters | $\propto N_{\text{gpus}}$ |

For a 7B parameter model with Adam:
- fp32 params: 28 GB
- fp32 Adam states ($m$, $v$): 56 GB
- **Total:** 84 GB per GPU -> requires 8x A100 (80 GB) without ZeRO
- With ZeRO-3 across 64 GPUs: ~1.3 GB per GPU

**Numerical impact:** ZeRO doesn't change the mathematical update - it just distributes the storage. The optimizer state is identical to non-distributed Adam.

---

## Appendix Y: Connections to Bayesian Optimization

**Bayesian optimization** is a black-box optimization method that uses a probabilistic model (usually a Gaussian Process) to model the objective function and selects the next evaluation point by maximizing an **acquisition function**.

**Connection to numerical optimization:**
- The acquisition function maximization is itself a classical optimization problem (e.g., L-BFGS on the acquisition function)
- The surrogate model (GP) must be numerically stable: Cholesky factorization of the kernel matrix
- Ill-conditioning of the kernel matrix -> add noise $\sigma^2 I$ (numerical regularization)

**For AI hyperparameter tuning:** Bayesian optimization is used to tune:
- Learning rate, batch size, weight decay
- Architecture hyperparameters (number of layers, hidden dimensions)
- Training duration (early stopping threshold)

Tools: Optuna (Python), GPyOpt, BoTorch (PyTorch-native).

---

## Appendix Z: Final Notes

**This section covered:**

1. Line search methods (Armijo, Wolfe, backtracking) - essential for convergence guarantees
2. Newton's method - quadratic convergence, modified Newton for non-convex settings
3. Quasi-Newton methods (BFGS, L-BFGS) - practical second-order optimization
4. Trust region methods - alternative to line search for non-convex landscapes
5. Numerical differentiation - finite differences, optimal step size, complex-step
6. Constrained optimization - projected gradient, ADMM
7. AI applications - Adam as diagonal quasi-Newton, L-BFGS for full-batch training, HF optimization

**Key takeaways:**
- Adam is not an arbitrary heuristic: it approximates diagonal preconditioning of the gradient
- L-BFGS is the gold standard for full-batch optimization but fails for stochastic settings
- The condition number of the loss Hessian fundamentally limits gradient descent convergence
- Trust region methods (TRPO, PPO) provide a sound theoretical framework for policy gradient RL
- Gradient checking is essential for debugging custom autodiff implementations

*Notes length: ~2000 lines. Companion: [theory.ipynb](theory.ipynb) (50+ cells) | [exercises.ipynb](exercises.ipynb) (10 exercises)*

---

## Appendix AA: Worked Examples

### AA.1 Computing the Optimal Step Size for Gradient Descent

**Problem:** Minimize $f(x) = x_1^2 + 50x_2^2$ with gradient descent. Find the optimal fixed step size.

**Analysis:**
- Hessian: $H = \text{diag}(2, 100)$
- $\lambda_{\max} = 100$, $\lambda_{\min} = 2$, $\kappa = 50$
- Optimal step: $\eta^* = 2/(\lambda_{\max} + \lambda_{\min}) = 2/102 \approx 0.0196$
- Convergence rate: $(\kappa - 1)/(\kappa + 1) = 49/51 \approx 0.961$ per iteration
- Iterations to $10^{-6}$ reduction: $\log(10^{-6}) / \log(0.961) \approx 334$ iterations

**Python verification:**
```python
H_diag = np.array([2., 100.])
kappa = 50
eta = 2 / (100 + 2)
rate = (kappa - 1) / (kappa + 1)

x = np.array([1., 1.])
errors = [np.linalg.norm(x)]
for _ in range(500):
    g = H_diag * x
    x -= eta * g
    errors.append(np.linalg.norm(x))

# Check: errors should decrease geometrically at rate ~0.961
actual_rate = errors[10] / errors[9]
print(f"Theoretical rate: {rate:.4f}")
print(f"Actual rate:      {actual_rate:.4f}")  # Should match
```

### AA.2 L-BFGS on the Rosenbrock Function

```python
from scipy.optimize import minimize

def rosenbrock(x): return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2
def rosenbrock_grad(x):
    return np.array([-400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0]),
                     200*(x[1]-x[0]**2)])

x0 = np.array([-1.0, 1.0])
result = minimize(rosenbrock, x0, jac=rosenbrock_grad, method='L-BFGS-B',
                  options={'maxiter': 1000, 'gtol': 1e-12})

print(f"Solution: {result.x}")          # Should be [1., 1.]
print(f"Function value: {result.fun}")  # Should be ~0
print(f"Iterations: {result.nit}")      # Typically ~30-50
```

---

## Appendix BB: Glossary

| Term | Definition |
|------|-----------|
| **Armijo condition** | Sufficient decrease condition: $f(x + \alpha p) \leq f(x) + c_1 \alpha g^\top p$ |
| **Backtracking line search** | Multiply step by $\rho < 1$ until Armijo condition satisfied |
| **BFGS** | Quasi-Newton method that builds inverse Hessian approximation from $s, y$ pairs |
| **Conjugate direction** | $d_1, d_2$ are $A$-conjugate if $d_1^\top A d_2 = 0$ |
| **Curvature condition** | Ensures step is not too small; required for BFGS stability |
| **Descent direction** | $p$ such that $\nabla f^\top p < 0$ |
| **Fletcher-Reeves** | NCG formula: $\beta = \|g_{k+1}\|^2 / \|g_k\|^2$ |
| **L-BFGS** | BFGS with limited memory ($m$ vector pairs); $O(md)$ cost |
| **Secant condition** | $B_{k+1} s_k = y_k$; ensures Hessian approx matches observed curvature |
| **Super-linear convergence** | $\lim \|e_{k+1}\| / \|e_k\| = 0$; faster than linear, slower than quadratic |
| **Trust region** | Sphere of radius $\Delta$ within which the quadratic model is trusted |
| **Two-loop recursion** | Algorithm for efficiently computing $H_k^{-1} g$ in L-BFGS |
| **Wolfe conditions** | Armijo + curvature conditions; standard for line search in quasi-Newton methods |

---

*[<- Section02 Numerical Linear Algebra](../02-Numerical-Linear-Algebra/notes.md) | [-> Section04 Interpolation and Approximation](../04-Interpolation-and-Approximation/notes.md)*

*End of Section03 Numerical Optimization*

---

## Appendix CC: Further Reading

### Books

- **Nocedal & Wright, "Numerical Optimization"** (2nd ed., 2006) - The definitive reference. Chapters 2-7 cover all algorithms in this section in depth.
- **Boyd & Vandenberghe, "Convex Optimization"** - Available free online. Rigorous treatment of convex optimization theory.
- **Bertsekas, "Nonlinear Programming"** - Comprehensive treatment including constrained methods.

### Papers

- Kingma & Ba, "Adam: A Method for Stochastic Optimization" (2014) - Original Adam paper
- Nocedal, "Updating quasi-Newton matrices with limited storage" (1980) - Original L-BFGS paper
- Martens, "Deep learning via Hessian-free optimization" (2010) - Hessian-free for neural networks
- Gupta, Koren, Singer, "Shampoo: Preconditioned stochastic tensor optimization" (2018) - Shampoo
- Foret et al., "Sharpness-Aware Minimization" (2021) - SAM optimizer
- Cohen et al., "Gradient descent on neural networks typically occurs at the edge of stability" (2021) - EoS phenomenon

### Software

- `scipy.optimize` - Reference implementations of L-BFGS-B, Newton-CG, trust region methods
- `torch.optim` - Adam, AdamW, LBFGS, SGD with momentum, Adagrad, RMSProp
- `optax` (JAX) - Composable optimizer building blocks
- `bfgs` (Julia) - Clean reference implementation of BFGS
- `Optuna` - Bayesian hyperparameter optimization

---

## Appendix DD: Summary Statistics

```
Section03 NUMERICAL OPTIMIZATION - SUMMARY
========================================================================

  Main sections:        12 (Intuition through Conceptual Bridge)
  Subsections:          22
  Common mistakes:      12
  Exercises:            8 (* to ***)
  Appendices:           30 (A through DD)

  Algorithms covered:   15 (GD, CG, NCG, BFGS, L-BFGS, SR1, Newton,
                           inexact Newton, trust region, Cauchy point,
                           dogleg, Armijo, Wolfe zoom, ADMM, Adam)

  AI connections:       10+ (Adam, Shampoo, K-FAC, L-BFGS, TRPO/PPO,
                            gradient clipping, SAM, natural gradient,
                            HF optimization, edge of stability)

  Approximate notes length: 2000 lines
  Theory cells: 50+ (planned)
  Exercises: 10 graded problems (24+ cells)

========================================================================
```

---

## Appendix EE: Numerical Experiments Catalog

### EE.1 Comparing Optimizers on Standard Test Functions

The following test functions are standard benchmarks for optimization algorithms:

| Function | Formula | Dimension | Challenge |
|----------|---------|-----------|-----------|
| Rosenbrock | $\sum_{i=1}^{n-1} [100(x_{i+1} - x_i^2)^2 + (1-x_i)^2]$ | $n$ | Ill-conditioned valley |
| Rastrigin | $10n + \sum_i [x_i^2 - 10\cos(2\pi x_i)]$ | $n$ | Many local minima |
| Booth | $(x + 2y - 7)^2 + (2x + y - 5)^2$ | 2 | Simple quadratic |
| Beale | $(1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2$ | 2 | Nonlinear |
| Himmelblau | $(x^2 + y - 11)^2 + (x + y^2 - 7)^2$ | 2 | Multiple global minima |

### EE.2 Expected Results

Running L-BFGS, gradient descent, and conjugate gradient on the Rosenbrock function ($n=2$):

```
Starting from x0 = (-1, 0):
  Method         Iterations  Final Loss    Final ||g||
  -----------------------------------------------------
  GD (optimal)   ~10,000    ~1e-5         ~1e-3
  CG (FR)        ~500       ~1e-8         ~1e-4
  BFGS           ~50        ~1e-10        ~1e-5
  L-BFGS (m=10)  ~40        ~1e-10        ~1e-5
  Newton         ~20        ~1e-12        ~1e-6
```

These numbers are approximate and depend on the line search implementation and tolerance settings.

### EE.3 Optimizer Sensitivity Analysis

For Adam on a quadratic loss with varying condition numbers:

| (H) | Steps to \\varepsilon=10^-4 | Steps to \\varepsilon=10^-8 |
|-------|------------------|------------------|
| 1 | ~100 | ~200 |
| 10 | ~200 | ~400 |
| 100 | ~600 | ~1200 |
| 1000 | ~2000 | ~4000 |
| 10000 | ~6000 | ~12000 |

Adam improves on vanilla gradient descent but still scales poorly with condition number. The rate scales as $O(\kappa^{1/2})$ (from the diagonal preconditioning) vs $O(\kappa)$ for gradient descent.

---

*[<- Section02 Numerical Linear Algebra](../02-Numerical-Linear-Algebra/notes.md) | [-> Section04 Interpolation and Approximation](../04-Interpolation-and-Approximation/notes.md)*

---

*End of Section03 Numerical Optimization. See [theory.ipynb](theory.ipynb) for implementations and [exercises.ipynb](exercises.ipynb) for practice problems.*

*Total content: 30+ appendices, 12 main sections, 10 exercises, 15 algorithms covered.*

## Appendix FF: Practice Problems

1. Implement backtracking Armijo search and test on $f(x) = e^x - x$.
2. Derive the BFGS update from the constraint of minimal change: $\min \|B - B_k\|$ subject to $Bs_k = y_k$, $B = B^\top$.
3. Show that for the quadratic $f(x) = \frac{1}{2}x^\top Hx$, the optimal fixed step for gradient descent is $\alpha^* = 2/(\lambda_{\max} + \lambda_{\min})$.
4. Implement the complex-step derivative for $f(x) = \sin(\ln(x^2 + 1))$ and verify against `numpy.grad`.
5. Implement L-BFGS with strong Wolfe line search for the Rosenbrock function and count iterations to $\|\nabla f\| < 10^{-8}$.
6. Prove that if $y_k^\top s_k > 0$ and $H_k \succ 0$, then the BFGS update $H_{k+1} \succ 0$.
7. Implement the Cauchy point computation for the trust region subproblem on a 2D quadratic.
8. Show that Adam converges to the critical point $g = 0$ for convex $f$ with decaying learning rate $\alpha_t = \alpha/\sqrt{t}$.

9. Implement gradient accumulation with fp32 accumulators and bf16 forward/backward, and verify the accumulated gradient matches the full-batch gradient.
10. Implement ZeRO stage 1 (shard optimizer states across 2 processes using Python multiprocessing) and verify the same parameters result.

---

*Section complete.*

---

*[<- Section02 Numerical Linear Algebra](../02-Numerical-Linear-Algebra/notes.md) | [-> Section04 Interpolation and Approximation](../04-Interpolation-and-Approximation/notes.md)*

*End of file.*
