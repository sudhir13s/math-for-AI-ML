[Previous Chapter: Statistics](../07-Statistics/README.md) | [Next Chapter: Information Theory](../09-Information-Theory/01-Entropy/notes.md)

---

# Chapter 8 - Optimization

> _"Training a model is optimization under uncertainty, geometry, and compute constraints."_

## Overview

Optimization is the mathematical engine of machine learning. Every trained model is the result
of an algorithm moving through parameter space to reduce an objective while dealing with
curvature, noise, constraints, regularization, schedules, and finite compute.

This rewrite treats the chapter as a single arc: convex guarantees first, deterministic
first-order methods second, curvature third, constraints fourth, stochastic scale fifth,
nonconvex landscapes sixth, adaptive optimizer state seventh, regularization eighth,
hyperparameter search ninth, and schedule design tenth.

## Subsection Map

| # | Subsection | What It Covers | Canonical Topics |
| --- | --- | --- | --- |
| 01 | [Convex Optimization](01-Convex-Optimization/notes.md) | the canonical home for convex sets, convex functions, smoothness, strong convexity, condition numbers, convex problem classes, and the first full view of Lagrangian duality. | convex sets, convex combinations, convex functions, Jensen inequality, first-order characterization, second-order characterization, smoothness, strong convexity |
| 02 | [Gradient Descent](02-Gradient-Descent/notes.md) | the canonical home for first-order deterministic optimization, descent lemmas, step-size regimes, convergence rates, momentum, Nesterov acceleration, and line-search theory. | gradient direction, descent lemma, constant step size, exact line search, backtracking line search, Armijo condition, Wolfe conditions, convex convergence |
| 03 | [Second-Order Methods](03-Second-Order-Methods/notes.md) | the canonical home for Hessian-aware methods: Newton, damped Newton, Gauss-Newton, quasi-Newton, natural gradient, Hessian-vector products, and curvature tradeoffs. | Hessian matrix, quadratic model, Newton step, Newton decrement, damped Newton, modified Cholesky, trust-region preview, Gauss-Newton |
| 04 | [Constrained Optimization](04-Constrained-Optimization/notes.md) | the canonical home for equality and inequality constraints, KKT conditions, full duality machinery, projected methods, penalties, barriers, ADMM, and ML constraint design. | feasible set, active constraint, equality constraints, inequality constraints, Lagrangian, stationarity, primal feasibility, dual feasibility |
| 05 | [Stochastic Optimization](05-Stochastic-Optimization/notes.md) | the canonical home for stochastic gradient oracles, minibatch estimators, variance, SGD convergence, variance reduction, batch-size scaling, distributed SGD, and gradient-noise diagnostics. | stochastic objective, empirical risk, population risk, unbiased gradient oracle, gradient variance, minibatch estimator, batch-size scaling, critical batch size |
| 06 | [Optimization Landscape](06-Optimization-Landscape/notes.md) | the canonical home for critical points, saddles, Hessian spectra, sharpness, flatness, mode connectivity, edge of stability, and nonconvex training-path geometry. | critical point, local minimum, saddle point, strict saddle, plateau, Hessian spectrum, negative curvature, degeneracy |
| 07 | [Adaptive Learning Rate](07-Adaptive-Learning-Rate/notes.md) | the canonical home for per-parameter and layerwise adaptation: AdaGrad, RMSProp, Adam, AMSGrad, AdamW, Adafactor, LARS, LAMB, and modern preconditioning previews. | effective learning rate, diagonal preconditioner, AdaGrad accumulator, RMSProp exponential averaging, Adam first moment, Adam second moment, bias correction, epsilon stabilizer |
| 08 | [Regularization Methods](08-Regularization-Methods/notes.md) | the canonical home for regularization as optimization geometry: penalties, constraints, weight decay, dropout, early stopping, spectral controls, SAM, and implicit bias. | explicit penalty, constraint equivalence, L2 penalty, weight decay, AdamW decay, L1 penalty, soft thresholding, elastic net |
| 09 | [Hyperparameter Optimization](09-Hyperparameter-Optimization/notes.md) | the canonical home for search spaces, grid and random search, Bayesian optimization, acquisition functions, multi-fidelity methods, Hyperband, ASHA, PBT, and leakage-aware tuning. | configuration space, conditional parameter, log-uniform sampling, grid search, random search, Sobol initialization, surrogate model, Gaussian process |
| 10 | [Learning Rate Schedules](10-Learning-Rate-Schedules/notes.md) | the canonical home for time-varying learning rates: constant, step, exponential, polynomial, warmup, cosine, cyclic, one-cycle, WSD, cooldown, and batch-size coupling. | schedule function, constant learning rate, step decay, exponential decay, polynomial decay, linear warmup, warmup ratio, cosine annealing |

## Reading Order and Dependencies

```text
01-Convex-Optimization               (Convex Optimization)
        ↓
02-Gradient-Descent                  (Gradient Descent)
        ↓
03-Second-Order-Methods              (Second-Order Methods)
        ↓
04-Constrained-Optimization          (Constrained Optimization)
        ↓
05-Stochastic-Optimization           (Stochastic Optimization)
        ↓
06-Optimization-Landscape            (Optimization Landscape)
        ↓
07-Adaptive-Learning-Rate            (Adaptive Learning Rate)
        ↓
08-Regularization-Methods            (Regularization Methods)
        ↓
09-Hyperparameter-Optimization       (Hyperparameter Optimization)
        ↓
10-Learning-Rate-Schedules           (Learning Rate Schedules)
        ↓
Chapter 9 - Information Theory
```

## What Belongs Where - Canonical Homes

| Topic Family | Canonical Home | Preview Only In |
| --- | --- | --- |
| convex sets, convex combinations, convex functions, Jensen inequality, first-order characterization | [Convex Optimization](01-Convex-Optimization/notes.md) | Neighboring sections use brief references only |
| gradient direction, descent lemma, constant step size, exact line search, backtracking line search | [Gradient Descent](02-Gradient-Descent/notes.md) | Neighboring sections use brief references only |
| Hessian matrix, quadratic model, Newton step, Newton decrement, damped Newton | [Second-Order Methods](03-Second-Order-Methods/notes.md) | Neighboring sections use brief references only |
| feasible set, active constraint, equality constraints, inequality constraints, Lagrangian | [Constrained Optimization](04-Constrained-Optimization/notes.md) | Neighboring sections use brief references only |
| stochastic objective, empirical risk, population risk, unbiased gradient oracle, gradient variance | [Stochastic Optimization](05-Stochastic-Optimization/notes.md) | Neighboring sections use brief references only |
| critical point, local minimum, saddle point, strict saddle, plateau | [Optimization Landscape](06-Optimization-Landscape/notes.md) | Neighboring sections use brief references only |
| effective learning rate, diagonal preconditioner, AdaGrad accumulator, RMSProp exponential averaging, Adam first moment | [Adaptive Learning Rate](07-Adaptive-Learning-Rate/notes.md) | Neighboring sections use brief references only |
| explicit penalty, constraint equivalence, L2 penalty, weight decay, AdamW decay | [Regularization Methods](08-Regularization-Methods/notes.md) | Neighboring sections use brief references only |
| configuration space, conditional parameter, log-uniform sampling, grid search, random search | [Hyperparameter Optimization](09-Hyperparameter-Optimization/notes.md) | Neighboring sections use brief references only |
| schedule function, constant learning rate, step decay, exponential decay, polynomial decay | [Learning Rate Schedules](10-Learning-Rate-Schedules/notes.md) | Neighboring sections use brief references only |

## Overlap Danger Zones

1. Convexity versus constraints: convex duality is introduced in 01; full KKT machinery belongs to 04.
2. Deterministic GD versus SGD: 02 owns exact gradients; 05 owns stochastic oracles and gradient noise.
3. Second-order methods versus adaptive methods: 03 owns curvature matrices; 07 owns diagonal and layerwise adaptive optimizer state.
4. Regularization versus statistics: 08 owns optimization penalties and constraints; Chapter 7 owns statistical bias-variance and regression interpretation.
5. Learning-rate choice versus schedule shape: 02 analyzes fixed step sizes; 10 owns time-varying schedules.
6. Hyperparameter optimization versus schedule formulas: 09 owns search procedures; 10 owns the mathematical shapes being searched.

## Key Cross-Chapter Dependencies

- From Chapter 3: eigenvalues, PSD matrices, SVD, matrix norms, and condition number.
- From Chapters 4-5: gradients, Hessians, Jacobians, Taylor expansions, and backpropagation.
- From Chapters 6-7: probability, expectation, empirical risk, MLE, MAP, and validation logic.
- Into Chapter 9: cross-entropy, KL divergence, Fisher information, and information geometry.
- Into Chapter 10: numerical stability, finite-difference checks, implementation-level line search, and floating-point effects.

## 2026 Optimization Best Practices For AI/LLMs

1. Start from AdamW or a well-tested first-order baseline, then justify any more exotic optimizer with ablations.
2. Separate optimizer choice, regularization, schedule, batch size, and precision; each changes the effective update.
3. Always log gradient norm, update norm, parameter norm, learning rate, loss, validation metric, and optimizer-state summaries.
4. Use warmup when early curvature, normalization statistics, or optimizer state makes full-rate updates unstable.
5. Treat batch size and learning rate as coupled variables, especially under gradient accumulation and data parallelism.
6. Do not tune on the test set; use validation and nested evaluation for model-selection claims.
7. Check notebook-scale experiments before claiming a large-run behavior is understood.
8. Use higher-precision small-batch checks when mixed precision produces suspicious loss spikes.
9. Prefer explicit scope boundaries over duplicate theory across sections.
10. Run every notebook top-to-bottom after edits; valid JSON is not the same as a valid learning artifact.

## Curated Online Resources

- Boyd and Vandenberghe, Convex Optimization: https://web.stanford.edu/~boyd/cvxbook/
- Stanford EE364A Convex Optimization: https://web.stanford.edu/class/ee364a/
- Nocedal and Wright, Numerical Optimization.
- Bottou, Curtis, and Nocedal, Optimization Methods for Large-Scale Machine Learning.
- PyTorch optimizer and scheduler docs: https://docs.pytorch.org/docs/stable/optim.html
- Optax optimizer transformations: https://optax.readthedocs.io/
