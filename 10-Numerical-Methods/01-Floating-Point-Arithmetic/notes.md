[← Back to Numerical Methods](../README.md) | [Next: Numerical Linear Algebra →](../02-Numerical-Linear-Algebra/notes.md)

---

# Floating-Point Arithmetic

> _"The computer is binary by nature; real numbers are not. Everything that follows is a consequence of this mismatch."_
> — David Goldberg

## Overview

Every AI model trained on a GPU operates entirely in floating-point arithmetic — a finite, discrete approximation to the infinite real number line. When a neural network fails to converge, produces NaN losses, or gives subtly wrong predictions, the root cause is almost always numerical: a poorly conditioned operation, a rounding error that compounds across millions of iterations, or a format mismatch between fp16 and fp32 accumulators.

This section develops floating-point arithmetic from the ground up. We begin with the IEEE 754 standard — the universal specification for how computers represent and compute with non-integer numbers — then study rounding error models, catastrophic cancellation, condition numbers, and numerical stability. We close with the numerical formats used in modern AI: fp32, fp16, bf16, fp8, and their implications for training stability, inference speed, and quantization.

The central lesson is: floating-point errors are not random noise — they are deterministic, structured, and predictable. Once you understand the rounding error model $\text{fl}(x \circ y) = (x \circ y)(1+\delta)$, $|\delta| \le \varepsilon_{\text{mach}}$, you can analyze, bound, and eliminate the numerical errors that silently corrupt computations.

## Prerequisites

- Real number system and binary representation — [Mathematical Foundations §01](../../01-Mathematical-Foundations/01-Number-Systems/notes.md)
- Basic linear algebra: vectors, matrix norms — [Linear Algebra Basics](../../02-Linear-Algebra-Basics/README.md)
- No calculus or probability required

## Companion Notebooks

| Notebook | Description |
|---|---|
| [theory.ipynb](theory.ipynb) | Interactive: IEEE 754 bit layouts, rounding error experiments, Kahan summation, mixed-precision comparisons, stable softmax |
| [exercises.ipynb](exercises.ipynb) | 8 graded problems: machine epsilon, catastrophic cancellation, compensated summation, condition numbers, stable algorithms |

## Learning Objectives

After completing this section, you will:

- Decode IEEE 754 binary representations for fp16, fp32, fp64, bf16, and fp8
- Define machine epsilon $\varepsilon_{\text{mach}}$ and explain its role in bounding rounding errors
- Apply the fundamental rounding error model $\text{fl}(x \circ y) = (x \circ y)(1+\delta)$
- Identify catastrophic cancellation and redesign algorithms to avoid it
- Define forward error, backward error, and backward stability; classify algorithms
- Compute the condition number of a scalar function and of a linear system
- Explain mixed-precision training (fp16 + fp32 accumulation) and loss scaling
- Implement numerically stable log-sum-exp and softmax
- Compare fp32, fp16, bf16, and fp8 in terms of range, precision, and AI suitability
- Apply Kahan compensated summation to reduce floating-point sum errors
- Explain gradient underflow in fp16 and how loss scaling addresses it

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 Why Computers Cannot Represent Most Real Numbers](#11-why-computers-cannot-represent-most-real-numbers)
  - [1.2 Why This Matters for AI](#12-why-this-matters-for-ai)
  - [1.3 A Day in the Life of a Floating-Point Number](#13-a-day-in-the-life-of-a-floating-point-number)
  - [1.4 Historical Timeline: 1985–2024](#14-historical-timeline-19852024)
- [2. Formal Definitions: IEEE 754](#2-formal-definitions-ieee-754)
  - [2.1 The IEEE 754 Standard](#21-the-ieee-754-standard)
  - [2.2 Machine Epsilon](#22-machine-epsilon)
  - [2.3 The Floating-Point Number Line](#23-the-floating-point-number-line)
  - [2.4 Special Values: Infinity, NaN, Subnormals](#24-special-values-infinity-nan-subnormals)
  - [2.5 Rounding Modes](#25-rounding-modes)
- [3. Floating-Point Arithmetic](#3-floating-point-arithmetic)
  - [3.1 The Fundamental Rounding Error Model](#31-the-fundamental-rounding-error-model)
  - [3.2 Catastrophic Cancellation](#32-catastrophic-cancellation)
  - [3.3 Error Accumulation in Long Computations](#33-error-accumulation-in-long-computations)
  - [3.4 Kahan Compensated Summation](#34-kahan-compensated-summation)
  - [3.5 Interval Arithmetic and Verified Computation](#35-interval-arithmetic-and-verified-computation)
- [4. Condition Numbers and Stability](#4-condition-numbers-and-stability)
  - [4.1 Condition Number of a Scalar Problem](#41-condition-number-of-a-scalar-problem)
  - [4.2 Forward vs Backward Error Analysis](#42-forward-vs-backward-error-analysis)
  - [4.3 Forward and Backward Stability of Algorithms](#43-forward-and-backward-stability-of-algorithms)
  - [4.4 Condition Number of a Matrix](#44-condition-number-of-a-matrix)
- [5. Numerical Formats for AI](#5-numerical-formats-for-ai)
  - [5.1 fp32, fp16, bf16 Compared](#51-fp32-fp16-bf16-compared)
  - [5.2 fp8 and Extreme Quantization](#52-fp8-and-extreme-quantization)
  - [5.3 Mixed-Precision Training](#53-mixed-precision-training)
  - [5.4 Stochastic Rounding](#54-stochastic-rounding)
- [6. Applications in ML](#6-applications-in-ml)
  - [6.1 Gradient Vanishing/Exploding Through a Floating-Point Lens](#61-gradient-vanishingexploding-through-a-floating-point-lens)
  - [6.2 Numerically Stable Softmax and Log-Sum-Exp](#62-numerically-stable-softmax-and-log-sum-exp)
  - [6.3 Stable Cross-Entropy Loss](#63-stable-cross-entropy-loss)
  - [6.4 Loss Scaling for fp16 Training](#64-loss-scaling-for-fp16-training)
- [7. Common Mistakes](#7-common-mistakes)
- [8. Exercises](#8-exercises)
- [9. Why This Matters for AI (2026 Perspective)](#9-why-this-matters-for-ai-2026-perspective)
- [10. Conceptual Bridge](#10-conceptual-bridge)

---

## 1. Intuition

### 1.1 Why Computers Cannot Represent Most Real Numbers

The real numbers form an uncountably infinite set. A computer, by contrast, stores information in a finite number of bits. With $k$ bits you can represent at most $2^k$ distinct values. No finite $k$ is large enough to cover even the rationals between 0 and 1 — there are infinitely many of them.

The resolution: **floating-point numbers** are a carefully chosen, finite subset of the rationals — dense near zero (where scientific quantities tend to cluster) and increasingly sparse at large magnitudes. The IEEE 754 standard specifies exactly which rationals are representable and exactly how arithmetic on them is performed. Every GPU, CPU, and TPU on the planet implements this standard.

**The consequence for computation:** Every real-number operation — addition, multiplication, transcendentals — must be *rounded* to the nearest representable floating-point number. This introduces a tiny error at each step. Over millions of operations (a single transformer forward pass involves billions), these errors can accumulate in structured, predictable ways. Understanding and controlling this accumulation is numerical analysis.

**A concrete example:** The number $0.1$ cannot be represented exactly in binary floating-point. In fp32 it is stored as approximately $0.100000001490116119384765625$ — an error of $\approx 1.5 \times 10^{-9}$. This seems negligible, but when you sum 10 million such values, the error accumulates to $\approx 15$ — a 15% error in a sum that should be exactly $10^6$.

### 1.2 Why This Matters for AI

Floating-point arithmetic is not an abstraction for AI practitioners — it is the daily substrate:

**Training stability:** fp16 training requires loss scaling to prevent gradient underflow. Without it, gradients smaller than the fp16 minimum ($\approx 6 \times 10^{-5}$) are rounded to zero, silently stopping learning for those parameters.

**Model quality:** Quantizing from fp32 to int8 for inference can degrade accuracy by 1-5% on challenging tasks if done naively. Understanding the rounding error model tells you which operations are most sensitive and how to compensate.

**Numerical bugs:** NaN (Not a Number) propagation is one of the most common silent failures in deep learning. A single NaN anywhere in the computation graph — caused by $\log(0)$, $0/0$, or overflow — propagates through every subsequent operation, turning the entire gradient to NaN.

**Speed and memory:** The transition from fp32 to bf16 for training LLMs (GPT-4, Llama, Gemini all use bf16) halves memory and roughly doubles arithmetic throughput on modern hardware. Understanding why bf16 preserves training stability when fp16 does not requires knowing that bf16 preserves the exponent range of fp32 while sacrificing mantissa precision.

**For AI:** Every tensor in PyTorch, JAX, or TensorFlow is a floating-point array. `torch.finfo(torch.float16).eps` gives you $\varepsilon_{\text{mach}}$ for fp16. Knowing this number — $9.77 \times 10^{-4}$ — immediately tells you the precision limit of every fp16 computation.

### 1.3 A Day in the Life of a Floating-Point Number

Follow a real number $x = 0.3$ through a computation:

**Birth (encoding):** $x = 0.3$ is mapped to the nearest fp32 representable value: $x_{\text{fp32}} = 0.30000001192092896$. Error: $\approx 1.2 \times 10^{-8}$.

**Arithmetic (rounding):** Compute $x_{\text{fp32}} + 0.6_{\text{fp32}}$. Each input is rounded; the addition result is rounded again. The result $0.9_{\text{fp32}} = 0.89999997615814208$ differs from the true $0.9$ by $\approx 2.4 \times 10^{-8}$.

**Comparison (danger):** Testing `x_fp32 + 0.6_fp32 == 0.9` returns `False` in Python because both sides round differently. This is the source of countless bugs in numerical code.

**Accumulation (catastrophe):** Subtract $0.8_{\text{fp32}}$ from the result: $0.89999997615814208 - 0.80000001192092896 = 0.09999996423721312$ vs the true $0.1$. Relative error: $\approx 3.6 \times 10^{-7}$ — 30× larger than the original encoding error, because subtraction of nearly-equal numbers amplifies relative error. This is **catastrophic cancellation**.

**Death (overflow/underflow):** Multiply by $10^{38}$ in fp32: overflow to $+\infty$. Multiply by $10^{-46}$: underflow to $0$ (or a subnormal). Once a value becomes $\pm\infty$ or NaN, all subsequent operations inherit the corruption.

### 1.4 Historical Timeline: 1985–2024

```
FLOATING-POINT ARITHMETIC: KEY MILESTONES
════════════════════════════════════════════════════════════════════════

  1985  IEEE 754-1985 published — first universal FP standard
        (sign, exponent, mantissa; round-to-nearest-even default)
  1987  First IEEE 754 hardware: Intel 8087 math coprocessor
  1998  IEEE 754-1985 extended to include 80-bit extended precision
  2008  IEEE 754-2008 — adds fp16 (half precision), decimal formats
  2010  NVIDIA Fermi GPU: first IEEE-compliant fp64 on GPU
  2017  Mixed-precision training (Micikevicius et al., NVIDIA) —
        fp16 forward/backward, fp32 master weights
  2018  bf16 (Brain Float 16) introduced by Google Brain for TPUs
  2019  NVIDIA Ampere: bf16 tensor cores, hardware Transformer Engine
  2022  fp8 training (NVIDIA H100 Transformer Engine) — 8-bit FP
        with dynamic scaling; used for LLM training
  2023  Llama-2, GPT-4, Gemini: bf16 training standard
  2024  fp4 experimental; int4/int8 GGUF quantization standard
        for LLM inference (llama.cpp ecosystem)

════════════════════════════════════════════════════════════════════════
```

---

## 2. Formal Definitions: IEEE 754

### 2.1 The IEEE 754 Standard

The IEEE 754 standard represents a floating-point number $x$ in **sign-magnitude-exponent form**:

$$x = (-1)^s \cdot (1.m_1 m_2 \ldots m_p) _2 \cdot 2^{e - \text{bias}}$$

where:
- $s \in \{0, 1\}$ is the **sign bit**
- $m_1 m_2 \ldots m_p$ is the **mantissa** (or significand) in binary, with an implicit leading 1
- $e$ is the **stored exponent** (biased)
- $\text{bias} = 2^{q-1} - 1$ for a $q$-bit exponent field

**Format specifications:**

| Format | Total bits | Sign | Exponent | Mantissa | $\varepsilon_{\text{mach}}$ | Max value | Min normal |
|--------|-----------|------|----------|----------|---------------------------|-----------|------------|
| fp64 (double) | 64 | 1 | 11 | 52 | $2.2 \times 10^{-16}$ | $1.8 \times 10^{308}$ | $2.2 \times 10^{-308}$ |
| fp32 (single) | 32 | 1 | 8 | 23 | $1.2 \times 10^{-7}$ | $3.4 \times 10^{38}$ | $1.2 \times 10^{-38}$ |
| bf16 | 16 | 1 | 8 | 7 | $7.8 \times 10^{-3}$ | $3.4 \times 10^{38}$ | $1.2 \times 10^{-38}$ |
| fp16 | 16 | 1 | 5 | 10 | $9.8 \times 10^{-4}$ | $6.6 \times 10^{4}$ | $6.1 \times 10^{-5}$ |
| fp8 E4M3 | 8 | 1 | 4 | 3 | $0.125$ | $448$ | $1.5 \times 10^{-2}$ |
| fp8 E5M2 | 8 | 1 | 5 | 2 | $0.25$ | $57344$ | $1.5 \times 10^{-5}$ |

**Key insight:** bf16 has the **same exponent range** as fp32 (both use 8 exponent bits, bias 127) but only 7 mantissa bits vs 23. This means bf16 can represent the same *scale* of numbers as fp32, but with far less precision per value. This is why bf16 is training-stable (no overflow, no gradient underflow) while fp16 (5-bit exponent, max value $\approx 65504$) overflows easily during training.

**Bit-layout diagram:**

```
IEEE 754 BIT LAYOUTS
════════════════════════════════════════════════════════════════════════

  fp32 (32 bits):
  [S][EEEEEEEE][MMMMMMMMMMMMMMMMMMMMMMM]
   1     8              23

  fp16 (16 bits):
  [S][EEEEE][MMMMMMMMMM]
   1    5        10

  bf16 (16 bits):
  [S][EEEEEEEE][MMMMMMM]
   1     8         7      ← Same exponent field as fp32!

  fp8 E4M3 (8 bits):
  [S][EEEE][MMM]
   1    4     3

════════════════════════════════════════════════════════════════════════
```

### 2.2 Machine Epsilon

**Definition (Machine Epsilon):** The machine epsilon $\varepsilon_{\text{mach}}$ is the smallest floating-point number $\varepsilon$ such that $\text{fl}(1 + \varepsilon) > 1$. Equivalently:
$$\varepsilon_{\text{mach}} = 2^{-p}$$
where $p$ is the number of mantissa bits (the "precision").

For fp32: $\varepsilon_{\text{mach}} = 2^{-23} \approx 1.19 \times 10^{-7}$.

**Unit in the last place (ULP):** The ULP of a floating-point number $x$ is the spacing between $x$ and the next representable number. For a number in the range $[2^e, 2^{e+1})$:
$$\text{ULP}(x) = 2^{e-p}$$

**Fundamental property:** For any real $x$ in the normal range, the nearest floating-point number $\text{fl}(x)$ satisfies:
$$|\text{fl}(x) - x| \le \frac{1}{2} \text{ULP}(x) \le \frac{\varepsilon_{\text{mach}}}{2} |x|$$

This is the **relative rounding error bound**: $|\text{fl}(x) - x| / |x| \le \varepsilon_{\text{mach}}/2$.

**Non-example:** Machine epsilon is NOT the smallest positive floating-point number. The smallest normal fp32 number is $\approx 1.18 \times 10^{-38}$. Machine epsilon is the smallest perturbation to $1$ that is detectable — a very different concept.

### 2.3 The Floating-Point Number Line

The floating-point numbers are **not uniformly spaced**. They cluster densely near zero and become increasingly sparse at large magnitudes:

- In $[1, 2)$: spacing = $2^{-23}$ (fp32) — $\approx 8.4 \times 10^6$ numbers
- In $[2, 4)$: spacing = $2^{-22}$ — $\approx 8.4 \times 10^6$ numbers
- In $[1024, 2048)$: spacing = $2^{-13}$ — same count, but 1000× coarser
- Near $10^{37}$: spacing $\approx 10^{30}$ — individual values separated by $10^{30}$!

**Consequence for AI:** Neural network weights are typically initialized in $[-1, 1]$, where fp32 provides $\sim 10^7$ distinct values per unit. As training progresses and some weights grow large, precision degrades — but this rarely matters because the loss landscape is smooth there. The dangerous zone is when gradients become very small (near $10^{-7}$ in fp32), where underflow begins.

### 2.4 Special Values: Infinity, NaN, Subnormals

**Positive/negative infinity $(\pm\infty)$:** Stored with all-ones exponent, zero mantissa. Result of overflow (e.g., $3.4 \times 10^{38} \times 2$ in fp32) or division by zero. All arithmetic on $\infty$ is defined: $1/\infty = 0$, $\infty + \text{finite} = \infty$, but $\infty - \infty = \text{NaN}$.

**NaN (Not a Number):** All-ones exponent, nonzero mantissa. Results from $0/0$, $\infty - \infty$, $\sqrt{-1}$. NaN is **contagious**: any arithmetic involving NaN produces NaN. A single NaN in a gradient will propagate backward through the entire computation graph, turning all parameter updates to NaN.

**Subnormal numbers:** When the stored exponent is all-zeros, the implicit leading 1 becomes a leading 0, allowing numbers smaller than the normal minimum. fp32 subnormals cover $[\approx 1.4 \times 10^{-45}, 1.18 \times 10^{-38})$ with reduced precision. Many GPUs flush subnormals to zero ("FTZ mode") for performance, which can cause unexpected underflow.

**Practical rule:** In deep learning code, if `torch.isnan(loss)` is True, trace backward: either the loss computation involves `log(0)`, a division by a near-zero value, or upstream activations have overflowed.

### 2.5 Rounding Modes

IEEE 754 defines five rounding modes. The default — used in virtually all ML training — is **round-to-nearest-even** (banker's rounding):

| Mode | Description | When used |
|------|-------------|-----------|
| Round-to-nearest-even | Round to nearest; on tie, round to even mantissa bit | Default (IEEE 754) |
| Round toward $+\infty$ | Always round up | Interval arithmetic upper bounds |
| Round toward $-\infty$ | Always round down | Interval arithmetic lower bounds |
| Round toward zero | Truncate | Hardware division |
| Stochastic rounding | Randomly round up/down, proportional to distance | Low-precision training (fp8) |

**Round-to-nearest-even** is the default because it is unbiased: on average, half of tie-breaks round up and half round down, preventing systematic accumulation of rounding errors in long sums. This is critical for gradient accumulation over millions of steps.

---

## 3. Floating-Point Arithmetic

### 3.1 The Fundamental Rounding Error Model

**Theorem (Rounding Error Model):** For any IEEE 754-compliant basic operation $\circ \in \{+, -, \times, \div\}$ on floating-point numbers $x, y$:
$$\text{fl}(x \circ y) = (x \circ y)(1 + \delta), \quad |\delta| \le \varepsilon_{\text{mach}}$$

The computed result equals the exact result times a factor $(1+\delta)$ where $\delta$ is at most one machine epsilon in magnitude. This is the single most important fact in numerical analysis.

**Proof sketch:** By definition of $\varepsilon_{\text{mach}}$, rounding any real $z$ to float gives $\text{fl}(z) = z(1+\delta)$ with $|\delta| \le \varepsilon_{\text{mach}}/2$ (round-to-nearest). IEEE 754 mandates that each basic operation is computed as if exact and then rounded, so $\text{fl}(x \circ y) = \text{fl}(x \circ_{\text{exact}} y) = (x \circ_{\text{exact}} y)(1 + \delta)$.

**Chaining errors:** For a sequence of $n$ operations, the cumulative error is bounded by approximately $n\varepsilon_{\text{mach}}$ in relative terms. For a dot product $\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^n x_i y_i$:
$$|\text{fl}(\mathbf{x}^\top \mathbf{y}) - \mathbf{x}^\top \mathbf{y}| \le (n-1)\varepsilon_{\text{mach}} |\mathbf{x}|^\top |\mathbf{y}| \cdot (1 + O(\varepsilon_{\text{mach}}))$$

**For AI:** The attention softmax computes $\text{softmax}(\mathbf{q}^\top K / \sqrt{d_k})$ — a dot product of length $d_k$ (typically 64–128). In fp16, $\varepsilon_{\text{mach}} \approx 10^{-3}$, and the dot product accumulates $\sim 128 \times 10^{-3} = 12.8\%$ relative error if done naively in fp16. This is why FlashAttention accumulates the softmax denominator in fp32 even when computing in fp16.

### 3.2 Catastrophic Cancellation

**Catastrophic cancellation** occurs when two nearly-equal floating-point numbers are subtracted, causing the leading significant digits to cancel and leaving only the noisy low-order bits.

**Example:** Compute $f(x) = \sqrt{x+1} - \sqrt{x}$ for large $x$.

For $x = 10^8$ in fp32:
- $\sqrt{10^8 + 1} \approx 10000.0000500000$ (fp32: $10000.0$)
- $\sqrt{10^8} = 10000.0$ (fp32: $10000.0$)
- $f(10^8)_{\text{fp32}} = 0.0$ — complete loss of all digits!

True value: $f(10^8) = 1/(\sqrt{10^8+1} + \sqrt{10^8}) \approx 5 \times 10^{-5}$.

**Algebraic fix:** Multiply and divide by the conjugate:
$$\sqrt{x+1} - \sqrt{x} = \frac{(\sqrt{x+1} - \sqrt{x})(\sqrt{x+1} + \sqrt{x})}{\sqrt{x+1} + \sqrt{x}} = \frac{1}{\sqrt{x+1} + \sqrt{x}}$$

The denominator is computed by addition (not subtraction), avoiding cancellation entirely.

**General rule:** When a subtraction $a - b$ results in a value much smaller than $|a|$ and $|b|$, suspect catastrophic cancellation. The relative error in the result can be as large as $|a|/|a-b| \times \varepsilon_{\text{mach}}$ — the ratio of operand magnitude to result magnitude times machine epsilon.

**In AI:** The log-sum-exp function $\log(\sum_i e^{x_i})$ suffers cancellation when $e^{x_i}$ values are very different in magnitude. The numerically stable form subtracts the maximum first — an algebraic rearrangement that avoids both overflow and cancellation (see §6.2).

### 3.3 Error Accumulation in Long Computations

**Forward error analysis:** Track how errors in inputs propagate to errors in outputs. For a function $f: \mathbb{R}^n \to \mathbb{R}^m$ evaluated via a sequence of operations, the forward error bound expresses $\|\hat{f}(\mathbf{x}) - f(\mathbf{x})\|$ in terms of $\varepsilon_{\text{mach}}$ and problem data.

**Backward error analysis (Wilkinson):** Ask a different question: for what perturbed input $\hat{\mathbf{x}}$ does the computed output $\hat{f}(\mathbf{x})$ equal the *exact* output? The backward error is $\|\hat{\mathbf{x}} - \mathbf{x}\|/\|\mathbf{x}\|$ — how much was the input effectively perturbed?

**Why backward analysis is preferred:** If an algorithm's backward error is of size $\varepsilon_{\text{mach}}$ (i.e., the computed answer is the exact answer for a slightly perturbed input), the algorithm is **backward stable** — as good as you can expect from finite-precision arithmetic.

**Summation error:** For the sum $S = \sum_{i=1}^n x_i$, naive sequential summation has forward error $O(n \varepsilon_{\text{mach}}) \|{\mathbf{x}}\|_1$. Pairwise summation (binary tree) reduces this to $O(\log n \varepsilon_{\text{mach}})$. Kahan summation reduces it to $O(\varepsilon_{\text{mach}})$, independent of $n$.

### 3.4 Kahan Compensated Summation

Kahan (1965) showed that a small modification to sequential summation reduces the accumulated error from $O(n\varepsilon_{\text{mach}})$ to $O(\varepsilon_{\text{mach}})$, regardless of $n$:

```
KAHAN SUMMATION ALGORITHM
════════════════════════════════════════════════════════════════════════

  Input: x_1, x_2, ..., x_n
  Output: S ≈ sum(x_i)

  s = 0.0       # running sum
  c = 0.0       # compensation (tracks lost digits)

  for each x_i:
    y = x_i - c         # compensated addend
    t = s + y           # sum so far (y is small, loses digits)
    c = (t - s) - y     # recover the lost digits
    s = t               # update sum

  return s

════════════════════════════════════════════════════════════════════════
```

**Why it works:** The compensation variable `c` tracks the rounding error at each step. It accumulates the "lost" low-order bits and feeds them back into the next iteration. The net effect is that the error behaves as $O(\varepsilon_{\text{mach}})$ regardless of $n$ — equivalent to computing in twice the precision.

**In PyTorch:** `torch.sum()` uses pairwise summation (tree reduction), which achieves $O(\log n \varepsilon_{\text{mach}})$ error. For high-precision accumulation (e.g., loss over a large batch), consider accumulating in fp64 then converting back.

### 3.5 Interval Arithmetic and Verified Computation

**Interval arithmetic** replaces each number $x$ with an interval $[\underline{x}, \overline{x}]$ guaranteed to contain the true value. All operations are extended to operate on intervals, widening the bounds to account for rounding errors.

**Example:** $[1.0, 1.0] + [0.3, 0.3_{\text{fp32}}^+] = [1.3, 1.3_{\text{fp32}}^+]$ where the subscript fp32$^+$ denotes rounding up.

**Practical use:** Interval arithmetic gives **verified** bounds on computation results — you know with certainty that the true answer lies in the computed interval. Used in: formal verification of safety-critical software, reliable computing, and bound propagation in neural network verification (e.g., $\alpha$-$\beta$ CROWN for adversarial robustness).

**Limitation:** Interval widths grow rapidly in long computations (dependency problem), making intervals pessimistic for iterative algorithms. More sophisticated tools (affine arithmetic, Taylor models) improve the dependency problem.

---

## 4. Condition Numbers and Stability

### 4.1 Condition Number of a Scalar Problem

The **condition number** of a computational problem measures the sensitivity of the output to perturbations in the input — it quantifies how much the problem amplifies input errors, independent of any particular algorithm.

**Definition (absolute condition number):** For a function $f: \mathbb{R} \to \mathbb{R}$:
$$\kappa_{\text{abs}}(x) = \lim_{\delta \to 0} \sup_{|\Delta x| \le \delta} \frac{|f(x + \Delta x) - f(x)|}{|\Delta x|} = |f'(x)|$$

**Definition (relative condition number):**
$$\kappa_{\text{rel}}(x) = \left|\frac{x \cdot f'(x)}{f(x)}\right|$$

This measures relative output change per unit relative input change — the more natural quantity.

**Examples:**

| Function | $\kappa_{\text{rel}}(x)$ | Well-conditioned? |
|----------|--------------------------|-------------------|
| $f(x) = \sqrt{x}$ | $1/2$ | Yes — always |
| $f(x) = \ln x$ | $1/|\ln x|$ | Yes for $|x| \gg 1$; ill-conditioned near $x = 1$ |
| $f(x) = e^x$ | $|x|$ | Ill-conditioned for large $|x|$ |
| $f(x) = \sin x$ | $|x \cos x / \sin x|$ | Ill-conditioned near $x = k\pi$ |
| $a - b$ (subtraction) | $\max(|a|, |b|)/|a-b|$ | Catastrophic for $a \approx b$ |

**Key insight:** Condition number $\kappa_{\text{rel}} \gg 1$ means the problem is **inherently ill-conditioned** — even the best algorithm operating in exact arithmetic cannot give accurate results given inaccurate inputs. No numerical technique can fix a fundamentally ill-conditioned problem; it requires reformulation.

### 4.2 Forward vs Backward Error Analysis

Given a computed result $\hat{y}$ for a true answer $y = f(x)$:

**Forward error:** $|y - \hat{y}|$ (or relative: $|y - \hat{y}|/|y|$) — how far the computed answer is from the truth.

**Backward error:** The smallest $\|\Delta x\|$ such that $f(x + \Delta x) = \hat{y}$ — what perturbation to the input makes our computed answer *exact*. Symbolically: $\hat{y} = f(\hat{x})$ for some $\hat{x}$; the backward error is $|\hat{x} - x|/|x|$.

**Connection:** Forward error $\le$ condition number $\times$ backward error:
$$\frac{|\hat{y} - y|}{|y|} \lesssim \kappa_{\text{rel}}(x) \cdot \frac{|\hat{x} - x|}{|x|}$$

If an algorithm has backward error $\sim \varepsilon_{\text{mach}}$, then its forward error is at most $\kappa \varepsilon_{\text{mach}}$ — the best possible for that problem.

### 4.3 Forward and Backward Stability of Algorithms

**Definition (backward stable):** An algorithm for computing $f(x)$ is **backward stable** if the computed result $\hat{f}(x)$ satisfies:
$$\hat{f}(x) = f(x + \Delta x) \text{ for some } \|\Delta x\| = O(\varepsilon_{\text{mach}}) \|x\|$$

A backward-stable algorithm computes the exact answer for a slightly perturbed problem. Combined with a well-conditioned problem ($\kappa \sim 1$), this guarantees the forward error is $O(\varepsilon_{\text{mach}})$.

**Definition (forward stable):** An algorithm is **forward stable** if the computed result satisfies:
$$\|\hat{f}(x) - f(x)\| = O(\varepsilon_{\text{mach}}) \|f(x)\|$$

**Hierarchy:**
```
backward stable → forward stable (when κ is moderate)
forward stable ↛ backward stable (in general)
```

**Examples:**
- **Backward stable:** Householder QR, Gaussian elimination with partial pivoting
- **Forward stable but not backward stable:** Gram-Schmidt orthogonalization (modified version)
- **Unstable:** Naive Gaussian elimination without pivoting (exponential error growth possible)

### 4.4 Condition Number of a Matrix

For a linear system $A\mathbf{x} = \mathbf{b}$, the **matrix condition number** is:
$$\kappa(A) = \|A\| \cdot \|A^{-1}\| = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}$$

where $\sigma_{\max}$ and $\sigma_{\min}$ are the largest and smallest singular values.

**Geometric interpretation:** $\kappa(A)$ measures how much $A$ can stretch unit vectors relative to how much it shrinks them. A unit sphere gets mapped to an ellipsoid with semi-axes $\sigma_1 \ge \sigma_2 \ge \ldots \ge \sigma_n$; the condition number is the ratio $\sigma_1/\sigma_n$.

**Error amplification:** If $A\hat{\mathbf{x}} = \hat{\mathbf{b}}$ and $\|\hat{\mathbf{b}} - \mathbf{b}\|/\|\mathbf{b}\| = \varepsilon$ (small right-hand side perturbation):
$$\frac{\|\hat{\mathbf{x}} - \mathbf{x}\|}{\|\mathbf{x}\|} \le \kappa(A) \cdot \varepsilon$$

The condition number is the worst-case amplification factor from $\mathbf{b}$-error to $\mathbf{x}$-error.

**Values to know:**
- $\kappa(I) = 1$ — perfectly conditioned (identity)
- $\kappa(A) \approx 10^k$ — you lose $k$ decimal digits of accuracy
- $\kappa(A) > 1/\varepsilon_{\text{mach}}$ — $A$ is numerically singular; solutions are meaningless

**For AI:** The Hessian matrix of a neural network loss has condition number $\kappa(H) = \lambda_{\max}/\lambda_{\min}$. When $\kappa(H)$ is large ($10^5$–$10^{10}$ in practice), gradient descent converges slowly and is sensitive to learning rate. Adam's parameter-wise scaling approximates diagonal preconditioning, reducing the effective condition number. Full preconditioning (Shampoo optimizer) targets the exact Hessian eigenspectrum.

---

## 5. Numerical Formats for AI

### 5.1 fp32, fp16, bf16 Compared

```
FLOATING-POINT FORMAT COMPARISON
════════════════════════════════════════════════════════════════════════

  Property           fp32           fp16           bf16
  ─────────────────────────────────────────────────────────────────────
  Bits               32             16             16
  Exponent bits       8              5              8
  Mantissa bits      23             10              7
  Max value       3.4e+38       6.5e+04        3.4e+38
  Min normal      1.2e-38       6.1e-05        1.2e-38
  Machine eps     1.2e-07       9.8e-04        7.8e-03
  ─────────────────────────────────────────────────────────────────────
  Training use    Reference     Legacy (APEX)  Standard (2023+)
  Gradient risk   None          Underflow      None
  Memory (1B par) 4 GB          2 GB           2 GB
  Arithmetic TPU  Baseline      2× fp32        2× fp32

════════════════════════════════════════════════════════════════════════
```

**bf16 vs fp16:** Both are 16-bit. bf16 keeps the 8-bit exponent field of fp32, giving the same dynamic range. fp16 uses only 5 exponent bits — max value $\approx 65504$ — which causes overflow for activations in large models. bf16 gives 3 fewer significant decimal digits than fp16, but training remains stable because gradients don't overflow.

**Why bf16 won for LLMs:** GPT-2 trained in fp16 with careful loss scaling. GPT-3, Llama, Mistral, Gemini all train in bf16 — no loss scaling needed, simpler codebase, same memory.

### 5.2 fp8 and Extreme Quantization

fp8 uses only 8 bits. NVIDIA's H100 supports two fp8 formats:
- **E4M3** (4 exponent, 3 mantissa): higher precision, smaller range (max 448)
- **E5M2** (5 exponent, 2 mantissa): lower precision, larger range (max 57344)

**fp8 training workflow:** Forward pass in fp8 E4M3; backward pass in fp8 E5M2; weight updates in fp32 or bf16. The Transformer Engine (NVIDIA) automates format selection and scaling.

**Post-training quantization:** Most LLM inference today uses int8 or int4 weight quantization (not fp8): weights stored as 8-bit integers, dequantized to fp16/bf16 before matmul. Tools: GPTQ, AWQ, bitsandbytes. The quantization error is $O(2^{-b})$ where $b$ is the bit-width.

### 5.3 Mixed-Precision Training

Mixed-precision training (Micikevicius et al., 2018) is the standard approach for training large models:

```
MIXED-PRECISION TRAINING WORKFLOW
════════════════════════════════════════════════════════════════════════

  Master weights:  fp32  ─────────────────────────────────────────────┐
                          │ cast down                      ↑ update    │
  Forward pass:   fp16/bf16 (activations, weights)                    │
  Loss:           fp32 accumulation                                    │
  Backward pass:  fp16/bf16 gradients                                 │
  Gradient:       fp16/bf16 → fp32 (before weight update)  ──────────┘

  With loss scaling (fp16 only):
  loss_scaled = loss × scale_factor   (e.g., 2^15 = 32768)
  Backward through loss_scaled
  gradients_fp32 /= scale_factor before optimizer step

════════════════════════════════════════════════════════════════════════
```

**Key elements:**
1. **fp32 master weights:** Full-precision copy of weights for optimizer states (momentum, variance) and weight updates
2. **fp16/bf16 forward/backward:** Halves memory for activations; 2× throughput on tensor cores
3. **Loss scaling (fp16 only):** Multiplies loss by a large factor before backward pass to prevent gradient underflow; divides gradients before applying them
4. **fp32 accumulation:** Matrix multiplications accumulate partial sums in fp32 on modern hardware, even when inputs are in fp16

### 5.4 Stochastic Rounding

Standard rounding (round-to-nearest) is **deterministically biased**: a number exactly halfway between two representable values always rounds to even. Over many operations, this can systematically shift the computed trajectory.

**Stochastic rounding:** Round $x$ to $\lfloor x \rfloor_{\text{fp}}$ with probability $(\lceil x \rceil_{\text{fp}} - x) / \text{ULP}(x)$, otherwise to $\lceil x \rceil_{\text{fp}}$. This introduces random noise but ensures $\mathbb{E}[\text{fl}(x)] = x$ — the rounding is **unbiased**.

**Why this matters for low-precision training:** In fp8/fp4 training, rounding errors are large enough to systematically bias weight updates. Stochastic rounding prevents this systematic bias, allowing effective training at very low precision. Used in Graphcore IPUs and experimental fp4 training.

---

## 6. Applications in ML

### 6.1 Gradient Vanishing/Exploding Through a Floating-Point Lens

**Vanishing gradients** in fp16: If a gradient $g$ satisfies $|g| < 6.1 \times 10^{-5}$ (fp16 minimum normal), it is represented as 0 (or a subnormal with reduced precision). The parameter receives no gradient — vanishing is literal in floating-point.

**Loss scaling** prevents this: multiplying the loss by $s$ before backprop multiplies all gradients by $s$. A gradient of $10^{-6}$ with scale $s = 2^{15} \approx 32768$ becomes $\approx 0.03$ — comfortably within fp16 range.

**Gradient clipping** addresses exploding gradients: if $\|\mathbf{g}\|_2 > G_{\max}$, scale $\mathbf{g} \leftarrow G_{\max} \mathbf{g}/\|\mathbf{g}\|_2$. This prevents overflow and stabilizes training. Used by default in transformer training (clip norm $= 1.0$ is standard).

### 6.2 Numerically Stable Softmax and Log-Sum-Exp

**Naive softmax:** $\text{softmax}(\mathbf{x})_i = e^{x_i} / \sum_j e^{x_j}$

**Problem:** For large $x_i$ (e.g., $x_i = 100$), $e^{100} \approx 2.7 \times 10^{43}$ — overflow in fp32. For very negative $x_i$, $e^{x_i} = 0$ — underflow, losing the contribution entirely.

**Stable softmax:** Subtract the maximum before exponentiation:
$$\text{softmax}(\mathbf{x})_i = \frac{e^{x_i - \max_j x_j}}{\sum_j e^{x_j - \max_j x_j}}$$

By linearity of softmax, this gives identical output. Now the largest exponent is $e^0 = 1$ — no overflow. All other terms are in $(0, 1]$ — no underflow for reasonable inputs.

**Log-sum-exp:** $\text{logsumexp}(\mathbf{x}) = \log(\sum_j e^{x_j})$. Stable form:
$$\text{logsumexp}(\mathbf{x}) = m + \log\sum_j e^{x_j - m}, \quad m = \max_j x_j$$

Used in: numerically stable cross-entropy, log-probabilities, normalizing constants.

### 6.3 Stable Cross-Entropy Loss

**Naive implementation:**
```python
# UNSTABLE: loss = -sum(y * log(softmax(logits)))
probs = exp(logits) / sum(exp(logits))  # overflow risk
loss = -sum(y * log(probs))             # log(0) risk
```

**Stable implementation (PyTorch's `F.cross_entropy`):**
```python
# Combines log-softmax and NLL in one numerically stable pass:
# loss = -logits[true_class] + logsumexp(logits)
log_probs = logits - logsumexp(logits)  # log-softmax, stable
loss = -log_probs[true_class]           # NLL
```

This avoids both overflow in `exp` and `log(0)` in the entropy term.

### 6.4 Loss Scaling for fp16 Training

**Dynamic loss scaling (PyTorch `GradScaler`):**
1. Start with scale $s = 2^{15}$
2. Compute `scaled_loss = loss * s`
3. Call `scaled_loss.backward()` — gradients are scaled by $s$
4. If any gradient is `inf` or `nan`: reduce $s \leftarrow s/2$; skip optimizer step
5. Otherwise: unscale gradients by $1/s$; optimizer step; every $N$ steps try $s \leftarrow s \times 2$

This adaptive scheme maintains the gradient scale in fp16 range, maximizing utilization of fp16 precision while detecting and recovering from overflow.

---

## 7. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
|---|---------|---------------|-----|
| 1 | Testing equality: `a + b == c` for floats | Rounding makes exact equality almost never true | Use `abs(a+b-c) < tol * max(abs(a+b), abs(c), 1)` |
| 2 | Assuming `0.1 + 0.2 == 0.3` | Neither `0.1`, `0.2`, nor `0.3` is exactly representable | Always use relative tolerance for float comparisons |
| 3 | Using fp16 for loss accumulation | fp16 has only 10 bits of mantissa; running sum loses precision fast | Accumulate loss in fp32, even when training in fp16 |
| 4 | Ignoring NaN propagation | One NaN in a gradient turns ALL gradients to NaN | Add `torch.isnan(loss).any()` check before backward; trace to source |
| 5 | Conflating machine epsilon with smallest float | $\varepsilon_{\text{mach}} = 2^{-23}$ for fp32; smallest normal is $2^{-126}$; completely different | Know both values; use `torch.finfo` to look them up |
| 6 | Computing `log(softmax(x))` in two steps | `softmax(x)` first rounds, then `log` can see zeros | Use `log_softmax(x)` which is a single numerically stable operation |
| 7 | Not checking condition number before solving $Ax = b$ | High $\kappa(A)$ means the solution is meaningless in floating point | Compute `np.linalg.cond(A)` first; if $> 10^{12}$ in fp64, reformulate |
| 8 | Using fp16 without loss scaling | Gradients of magnitude $< 6\times 10^{-5}$ flush to zero | Use `torch.amp.GradScaler()` or switch to bf16 |
| 9 | Naive large-number subtraction | $10^{10} - (10^{10} - 1)$ loses all digits in fp32 | Rearrange algebraically to avoid large-magnitude cancellation |
| 10 | Computing $e^x$ for large $x$ before dividing | Overflow before normalization | Use log-sum-exp trick; never compute raw $e^x$ without max-subtraction |
| 11 | Assuming GPU arithmetic is associative | `(a+b)+c ≠ a+(b+c)` in floating point; GPU thread ordering varies | Don't rely on associativity; use `torch.use_deterministic_algorithms(True)` for reproducibility |
| 12 | Ignoring subnormal flushing on GPU | CUDA default: FTZ (flush-to-zero) mode for subnormals | Be aware of subnormal behavior; check if FTZ affects your gradient magnitudes |

---

## 8. Exercises

**Exercise 1 ★ — Machine Epsilon and Floating-Point Formats**

(a) Without using `np.finfo`, write code to experimentally determine machine epsilon for fp32 by finding the smallest $\varepsilon$ such that $\text{fl}(1 + \varepsilon) > 1$ (start from 1.0 and halve repeatedly).

(b) Compare to `np.finfo(np.float32).eps`. Also compute for fp64 and fp16. What is the ratio between fp32 and fp64 epsilon?

(c) Compute the maximum representable value for fp32 without using `np.finfo.max`. Start from `1.0` and double until overflow.

(d) Explain: why does `np.float16(65504) + np.float16(1) == np.float16(65504)` return True?

---

**Exercise 2 ★ — Catastrophic Cancellation**

(a) Compute $f(x) = x - \sin(x)$ for $x = 10^{-7}$ in fp32 and fp64. Compare to the Taylor series approximation $x - (x - x^3/6 + x^5/120) = x^3/6 - x^5/120 \approx x^3/6$.

(b) Compute $\sqrt{x^2 + 1} - 1$ for $x = 10^{-4}$ in fp32 and fp64. Derive an algebraically equivalent form without cancellation and verify numerically.

(c) For the quadratic formula $x = (-b \pm \sqrt{b^2 - 4ac}) / (2a)$ with $a = 1$, $b = -10^8$, $c = 1$: compute both roots in fp32. One root is catastrophically inaccurate. Identify which and implement the numerically stable form using the Vieta relation $x_1 x_2 = c/a$.

---

**Exercise 3 ★ — Kahan Compensated Summation**

(a) Implement naive summation and Kahan summation in pure NumPy (no special functions).

(b) Sum 10 million copies of $1/3$ using both methods in fp32. Compare to the exact answer $10^7/3 \approx 3{,}333{,}333.33\ldots$. What is the absolute error for each method?

(c) Repeat for a sequence of alternating $+1$ and $-1$ of length $10^6$. True sum = 0. What do both methods give?

(d) Verify that Kahan summation gives error $O(\varepsilon_{\text{mach}})$ independent of $n$, while naive summation gives error $O(n \varepsilon_{\text{mach}})$.

---

**Exercise 4 ★★ — Condition Numbers**

(a) Compute the relative condition number of $f(x) = e^x$ at $x = -30$, $x = 0$, $x = 30$. At which value is $f$ most sensitive to input perturbation?

(b) Construct a $4 \times 4$ Hilbert matrix $H_{ij} = 1/(i+j-1)$ (notoriously ill-conditioned). Compute $\kappa_2(H)$ using `np.linalg.cond`. Solve $H\mathbf{x} = \mathbf{b}$ for $\mathbf{b} = H\mathbf{1}$ (true solution $\mathbf{x} = \mathbf{1}$). What is the relative error in the computed solution?

(c) For the $4 \times 4$ Hilbert matrix, how many decimal digits of accuracy do you expect in the solution? Verify by computing $\lfloor \log_{10}(\varepsilon_{\text{mach}} \cdot \kappa(H)) \rfloor$ and comparing to the actual error.

---

**Exercise 5 ★★ — Numerically Stable Softmax and Log-Sum-Exp**

(a) Implement naive softmax and stable softmax. Test both on $\mathbf{x} = (1000, 1001, 1002)$ in fp32. Which produces NaN? What does the stable version give?

(b) Implement the log-sum-exp function: naive (`log(sum(exp(x)))`) and stable. Test on $\mathbf{x} = (-1000, 0, 1000)$.

(c) Implement numerically stable cross-entropy loss $\ell(\mathbf{z}, y) = -z_y + \text{logsumexp}(\mathbf{z})$ where $y$ is the true class. Test on logits $(0.1, 0.2, 0.3, 100.0)$ with true class $= 2$. Compare to naively computing `log(softmax(z)[y])`.

(d) Profile the two implementations for $10^5$ calls. Is there a speed difference?

---

**Exercise 6 ★★ — Mixed-Precision Comparison**

(a) Compute the inner product $\mathbf{x}^\top \mathbf{y}$ for $n = 1024$-dimensional random unit vectors in fp16, bf16, fp32, and fp64. Use the fp64 result as ground truth. Report absolute and relative errors.

(b) Simulate gradient underflow: generate a random gradient vector in fp32 with entries drawn from $\mathcal{N}(0, 10^{-6})$. Cast to fp16. What fraction of gradient entries are exactly zero (underflowed)? Repeat with scale factor $s = 2^{15}$ applied before casting.

(c) Implement simple loss scaling: wrap a function `f(x)` that returns a simulated loss; scale the loss, compute a "gradient" via finite differences in fp16, unscale. Compare gradient quality to fp32 finite differences.

---

**Exercise 7 ★★★ — Floating-Point Conditioning of a Linear System**

(a) Generate random $n \times n$ matrices with condition number $\kappa \in \{10, 10^4, 10^8, 10^{12}\}$ by constructing $A = U \Sigma V^\top$ with $U, V$ random orthogonal and $\Sigma = \text{diag}(\sigma_1, \ldots, \sigma_n)$ with $\sigma_1/\sigma_n = \kappa$.

(b) For each, solve $A\mathbf{x} = \mathbf{b}$ (true solution $\mathbf{x} = \mathbf{1}$) in fp32 and fp64. Plot relative error vs $\kappa$ on a log-log scale. Verify the predicted slope of 1.

(c) For $\kappa = 10^8$ in fp32: add a tiny perturbation $\|\Delta \mathbf{b}\| / \|\mathbf{b}\| = 10^{-7}$ to $\mathbf{b}$. How much does the solution change? Compare to $\kappa \cdot 10^{-7}$.

---

**Exercise 8 ★★★ — Stable Algorithm Design**

(a) The sample variance formula $\sigma^2 = \frac{1}{n} \sum (x_i - \bar{x})^2$ can be expanded as $\sigma^2 = \frac{1}{n}\sum x_i^2 - \bar{x}^2$ (one-pass formula). Show that the one-pass formula suffers catastrophic cancellation for $\mathbf{x} = (10^8, 10^8 + 1, 10^8 + 2)$ in fp32. Verify using Welford's online algorithm as the stable alternative.

(b) Implement Welford's online algorithm for computing running mean and variance.

(c) Compare naive, one-pass, and Welford for correctness on $\mathbf{x} = 10^8 + \{0, 1, 2, \ldots, 99\}$ in fp32.

---

## 9. Why This Matters for AI (2026 Perspective)

| Concept | AI Impact |
|---------|-----------|
| **Machine epsilon / format choice** | LLM training: bf16 is now the universal default (Llama-3, Gemma, Mistral) — understanding why requires knowing bf16's exponent range matches fp32 |
| **Catastrophic cancellation** | Naive attention score computation can cancel digits; FlashAttention's online softmax reordering avoids this |
| **Loss scaling** | Required for stable fp16 training; not needed with bf16; integral to `torch.amp.GradScaler` |
| **Stable log-sum-exp** | Every transformer's softmax in the forward pass uses this; also in all language modeling losses and perplexity calculations |
| **Condition numbers** | The Hessian condition number $\kappa(H)$ determines gradient descent convergence rate; Adam/Adafactor implicitly precondition to reduce effective $\kappa$ |
| **fp8 training** | NVIDIA H100 Transformer Engine, used for Llama-3 and GPT-4 training; requires per-tensor scaling factors |
| **Gradient underflow** | Without scaling, fp16 gradient magnitudes below $6\times 10^{-5}$ flush to zero — silent learning failure |
| **Stochastic rounding** | Enables effective fp4/fp8 training by preventing systematic rounding bias; used in Graphcore IPU and emerging hardware |
| **Subnormal flushing** | CUDA FTZ mode flushes subnormals to zero for speed; can cause unexpected behavior in extreme low-precision regimes |
| **Kahan summation** | PyTorch's attention accumulation uses fp32 accumulation (equivalent to Kahan) even in bf16 forward passes |

---

## 10. Conceptual Bridge

### Backward: What This Builds On

Floating-point arithmetic is the computational manifestation of the real number system. The real numbers ($\mathbb{R}$) are dense, complete, and infinite — studied in [Mathematical Foundations §01](../../01-Mathematical-Foundations/01-Number-Systems/notes.md). Floating-point numbers are a finite approximation: the map $\text{fl}: \mathbb{R} \to \mathbb{F}$ introduces the rounding errors analyzed here.

The condition number of a linear system — introduced as $\kappa(A) = \|A\|\|A^{-1}\|$ — is defined using matrix norms from [Linear Algebra Basics §06](../../02-Linear-Algebra-Basics/06-Vector-Spaces-Subspaces/notes.md) and singular values from [Advanced Linear Algebra §02](../../03-Advanced-Linear-Algebra/02-Singular-Value-Decomposition/notes.md). The interpretation $\kappa(A) = \sigma_{\max}/\sigma_{\min}$ requires the full SVD theory developed there.

### Forward: What This Enables

**Numerical Linear Algebra** (§02, this chapter): Every linear system solver, least-squares method, and eigenvalue algorithm is analyzed through the lens of backward error and condition numbers introduced here. The pivoting strategies in Gaussian elimination and the choice between QR and normal equations for least squares are direct applications of the stability concepts from §4.

**Numerical Optimization** (§03, this chapter): Gradient checking via finite differences relies on the finite-difference approximation error analysis — the step size $h$ must balance truncation error $O(h^2)$ against cancellation error $O(\varepsilon_{\text{mach}}/h)$, an optimization that requires understanding both. Mixed-precision training is the union of §5 (format choices) and optimization algorithms.

**All of Applied Mathematics:** Every numerical computation in this curriculum — ODE solvers, PDE methods, Fourier transforms — is ultimately floating-point arithmetic analyzed by the tools developed here.

```
POSITION IN CURRICULUM
════════════════════════════════════════════════════════════════════════

  §01-Mathematical-Foundations  (real numbers, binary representation)
       │
       ▼
  §02-Linear-Algebra-Basics  (matrix norms, condition numbers prelim.)
  §03-Advanced-Linear-Algebra  (SVD → κ(A) = σ_max/σ_min)
       │
       ▼
  §10-Numerical-Methods
    ├── §01-Floating-Point-Arithmetic  ◀═══ YOU ARE HERE
    │        │  ε_mach, κ, stability
    │        ▼
    ├── §02-Numerical-Linear-Algebra  (stable solvers, iterative methods)
    ├── §03-Numerical-Optimization   (AD, line search, finite diffs)
    ├── §04-Interpolation            (polynomial, spline, RBF)
    └── §05-Numerical-Integration    (quadrature, Monte Carlo)
       │
       ▼
  §08-Optimization  (gradient descent, Adam — uses mixed-precision)
  §13-ML-Specific-Math  (numerical stability of transformers)

════════════════════════════════════════════════════════════════════════
```

The foundations laid in this section — machine epsilon, rounding error models, condition numbers, backward stability — are the vocabulary of all numerical computation. Every section that follows assumes fluency with these concepts.

---

## Appendix A: IEEE 754 Bit-Level Encoding

### A.1 Decoding a 32-bit Float by Hand

Every fp32 number $x$ is stored as three fields:

```
Bit 31  | Bits 30–23  | Bits 22–0
Sign S  | Exponent E  | Mantissa M
```

The value is:
$$x = (-1)^S \times 2^{E - 127} \times (1.M_1 M_2 \ldots M_{23})_2$$

**Example:** Decode the fp32 hex `0x3FC00000`:
- Binary: `0 01111111 10000000000000000000000`
- $S = 0$ (positive)
- $E = 01111111_2 = 127$; exponent $= 127 - 127 = 0$
- Mantissa $= 1.10000\ldots_2 = 1 + 2^{-1} = 1.5$
- Value: $(-1)^0 \times 2^0 \times 1.5 = 1.5$ ✓

**Encoding 1.0:** $1.0 = 1.000\ldots_2 \times 2^0$; $E = 127 = 01111111_2$; $M = 0$; hex `0x3F800000`.

**Encoding 0.1:** $0.1 = 1.10011001100\ldots_2 \times 2^{-4}$ (repeating binary). Stored exponent $= 127 - 4 = 123$; mantissa rounded to 23 bits: `11001100110011001100110`; value is $\approx 0.100000001490116$.

### A.2 Special Patterns

| Pattern | Meaning |
|---------|---------|
| $E = 00000000$, $M = 0$ | $\pm 0$ (zero) |
| $E = 00000000$, $M \ne 0$ | Subnormal: $(-1)^S \times 2^{-126} \times (0.M)_2$ |
| $E = 11111111$, $M = 0$ | $\pm\infty$ |
| $E = 11111111$, $M \ne 0$ | NaN (quiet or signaling) |
| $E = 11111111$, $M_{22} = 1$ | Quiet NaN (qNaN) — most common |
| $E = 11111111$, $M_{22} = 0, M \ne 0$ | Signaling NaN (sNaN) — triggers exception |

### A.3 Python Bit Manipulation

```python
import struct, numpy as np

def fp32_to_bits(x):
    return struct.unpack('I', struct.pack('f', x))[0]

def bits_to_fp32(bits):
    return struct.unpack('f', struct.pack('I', bits))[0]

def decode_fp32(x):
    bits = fp32_to_bits(x)
    S = (bits >> 31) & 1
    E = (bits >> 23) & 0xFF
    M = bits & 0x7FFFFF
    if E == 0:
        val = (-1)**S * 2**(-126) * M / 2**23  # subnormal
    elif E == 255:
        val = float('inf') if M == 0 else float('nan')
    else:
        val = (-1)**S * 2**(E - 127) * (1 + M / 2**23)
    return {'S': S, 'E': E, 'M': M, 'value': val}
```

---

## Appendix B: Error Analysis Worked Examples

### B.1 Quadratic Formula Cancellation

The quadratic $x^2 - 10^8 x + 1 = 0$ has roots:
$$x_{1,2} = \frac{10^8 \pm \sqrt{10^{16} - 4}}{2}$$

In fp32, $\sqrt{10^{16} - 4} \approx 10^8$ (exactly). So:
- $x_1 = (10^8 + 10^8)/2 = 10^8$ — fine
- $x_2 = (10^8 - 10^8)/2 = 0/2 = 0$ — catastrophically wrong (true: $x_2 = 10^{-8}$)

**Stable form:** Use $x_1 x_2 = c/a = 1$, so $x_2 = 1/x_1 = 10^{-8}$.

### B.2 Dot Product Error Bound

For $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$, the computed dot product $\hat{s}$ satisfies:
$$|\hat{s} - \mathbf{x}^\top \mathbf{y}| \le \gamma_n |\mathbf{x}|^\top |\mathbf{y}|$$
where $\gamma_n = n\varepsilon_{\text{mach}} / (1 - n\varepsilon_{\text{mach}}) \approx n\varepsilon_{\text{mach}}$.

For $n = 512$, fp16: $\gamma_{512} \approx 512 \times 10^{-3} = 0.512$ — potential 50% error! This is why transformer attention uses fp32 accumulation inside bf16/fp16 matmuls.

### B.3 Gram-Schmidt vs Householder

Modified Gram-Schmidt (MGS) is forward stable but not backward stable: given columns $A = [\mathbf{a}_1, \ldots, \mathbf{a}_n]$, the computed $Q$ satisfies $\|\hat{Q}^\top \hat{Q} - I\| = O(\kappa(A) \varepsilon_{\text{mach}})$. For ill-conditioned $A$, the computed columns are not orthogonal.

Householder QR is backward stable: $\|A - \hat{Q}\hat{R}\| = O(\varepsilon_{\text{mach}}) \|A\|$ and $\hat{Q}$ is exactly orthogonal (to machine precision). Always prefer Householder for numerical QR.

---

## Appendix C: Floating-Point Arithmetic in PyTorch

### C.1 Default Dtypes

```python
import torch
torch.get_default_dtype()     # float32 by default
torch.set_default_dtype(torch.float64)  # change for scientific computing

# Check format properties:
torch.finfo(torch.float16)    # eps, min, max, tiny
torch.finfo(torch.bfloat16)
torch.finfo(torch.float32)
torch.finfo(torch.float64)
```

### C.2 Mixed-Precision with AMP

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    with autocast(device_type='cuda', dtype=torch.float16):
        output = model(batch)
        loss = criterion(output, target)
    # Scales loss, calls backward, updates scaler
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### C.3 Numerically Stable Operations

```python
# Stable log-softmax (use this, not log(softmax(x))):
torch.nn.functional.log_softmax(x, dim=-1)

# Stable cross-entropy (combines log-softmax + NLL):
torch.nn.functional.cross_entropy(logits, targets)

# Stable sigmoid cross-entropy:
torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
# NEVER: binary_cross_entropy(sigmoid(logits), targets)  ← unstable

# Numerically stable softplus:
torch.nn.functional.softplus(x)  # log(1 + exp(x)), stable
```

### C.4 Debugging NaN Issues

```python
# Enable anomaly detection (slow, for debugging only):
torch.autograd.set_detect_anomaly(True)

# Check for NaN:
assert not torch.isnan(loss).any(), f"NaN loss at step {step}"
assert not torch.isinf(loss).any(), f"Inf loss at step {step}"

# Gradient NaN check:
for name, param in model.named_parameters():
    if param.grad is not None:
        if torch.isnan(param.grad).any():
            print(f"NaN gradient in {name}")
```

---

## Appendix D: Numerical Formats Reference Card

```
QUICK REFERENCE: FLOATING-POINT FORMATS FOR AI
════════════════════════════════════════════════════════════════════════

  Format   Bits  ε_mach      Max         Use case
  ──────────────────────────────────────────────────────────────────────
  fp64      64   2.2e-16   1.8e+308    Scientific computing, reference
  fp32      32   1.2e-07   3.4e+038    Training master weights, debug
  bf16      16   7.8e-03   3.4e+038    LLM training (2023+ standard)
  fp16      16   9.8e-04   6.5e+004    Older mixed-precision training
  fp8 E4M3   8   1.2e-01     448       H100 forward pass
  fp8 E5M2   8   2.5e-01   57344       H100 backward pass
  int8       8   1/128      127        Inference quantization (weights)
  int4       4   1/8          7        GGUF LLM inference
  ──────────────────────────────────────────────────────────────────────

  Rule of thumb:
  • Training large models:  bf16 compute + fp32 master weights
  • Inference large models: int8 or int4 weights, fp16 activations
  • Scientific computing:   fp64 throughout
  • Edge inference:         int4/int8 quantized (ONNX, TFLite, GGUF)

════════════════════════════════════════════════════════════════════════
```

---

*[← Back to Numerical Methods](../README.md) | [Next: Numerical Linear Algebra →](../02-Numerical-Linear-Algebra/notes.md)*

---

## Appendix E: Proofs and Derivations

### E.1 Proof: Kahan Summation Error Bound

**Theorem:** Kahan compensated summation computes $\hat{S} = \sum_{i=1}^n x_i$ with error:
$$|\hat{S} - S| \le (2 + O(n\varepsilon_{\text{mach}})) \varepsilon_{\text{mach}} \sum_{i=1}^n |x_i|$$

This is $O(\varepsilon_{\text{mach}})$ rather than $O(n\varepsilon_{\text{mach}})$ for naive summation.

**Proof sketch:** At each step, Kahan maintains a compensation $c$ that captures the rounding error from the previous step to full precision. Formally, after step $k$:
$$s_k + c_k = \sum_{i=1}^k x_i + O(\varepsilon_{\text{mach}}^2 \sum |x_i|)$$

The accumulated error is $O(\varepsilon_{\text{mach}}^2)$ per step rather than $O(\varepsilon_{\text{mach}})$, so after $n$ steps the total is $O(n\varepsilon_{\text{mach}}^2) = O(\varepsilon_{\text{mach}})$ (treating $\varepsilon_{\text{mach}}$ as small). $\square$

### E.2 Derivation: Stable Softmax

**Claim:** $\text{softmax}(\mathbf{x})_i = \text{softmax}(\mathbf{x} - m\mathbf{1})_i$ for any scalar $m$.

**Proof:**
$$\frac{e^{x_i - m}}{\sum_j e^{x_j - m}} = \frac{e^{x_i} e^{-m}}{\sum_j e^{x_j} e^{-m}} = \frac{e^{x_i}}{\sum_j e^{x_j}} = \text{softmax}(\mathbf{x})_i \quad \square$$

Setting $m = \max_j x_j$ ensures the largest exponent is $e^0 = 1$, bounding all terms in $[e^{-\infty}, 1] = [0, 1]$ — within the range of any floating-point format.

### E.3 Wilkinson's Backward Error for GE with Partial Pivoting

Gaussian elimination with partial pivoting (GEPP) on $A\mathbf{x} = \mathbf{b}$ computes $\hat{\mathbf{x}}$ satisfying:
$$(A + \Delta A)\hat{\mathbf{x}} = \mathbf{b}$$
where $\|\Delta A\|_\infty \le 8n^3 \rho \varepsilon_{\text{mach}} \|A\|_\infty$ and $\rho$ is the **growth factor**: $\rho = \max_{i,j,k} |a_{ij}^{(k)}| / \|A\|_\infty$ (ratio of largest element appearing during elimination to largest initial element).

**Key point:** With partial pivoting, $\rho \le 2^{n-1}$ in the worst case (Wilkinson, 1961) but is almost always $O(n^{1/2})$ in practice. This is why GEPP is considered backward stable for practical purposes even though its worst-case bound is exponential.

### E.4 Condition Number and Relative Error

**Theorem:** Let $A$ be invertible, $\mathbf{b} \ne \mathbf{0}$. If $A\mathbf{x} = \mathbf{b}$ and $(A + \Delta A)(\mathbf{x} + \Delta\mathbf{x}) = \mathbf{b} + \Delta\mathbf{b}$, then:
$$\frac{\|\Delta\mathbf{x}\|}{\|\mathbf{x}\|} \le \frac{\kappa(A)}{1 - \kappa(A)\|\Delta A\|/\|A\|} \left(\frac{\|\Delta A\|}{\|A\|} + \frac{\|\Delta\mathbf{b}\|}{\|\mathbf{b}\|}\right)$$

When $\|\Delta A\| / \|A\|$ is small (as for rounding errors $\sim \varepsilon_{\text{mach}}$), this simplifies to:
$$\frac{\|\Delta\mathbf{x}\|}{\|\mathbf{x}\|} \lesssim \kappa(A) \left(\frac{\|\Delta A\|}{\|A\|} + \frac{\|\Delta\mathbf{b}\|}{\|\mathbf{b}\|}\right)$$

**Proof:** Subtract the two equations, use $\Delta\mathbf{x} = -A^{-1}\Delta A(\mathbf{x} + \Delta\mathbf{x}) + A^{-1}\Delta\mathbf{b}$, take norms, and use triangle inequality. The factor $1 - \kappa(A)\|\Delta A\|/\|A\|$ appears from a Neumann series approximation. $\square$

---

## Appendix F: Numerical Experiments Reference

### F.1 Key Numbers to Remember

| Quantity | fp16 | bf16 | fp32 | fp64 |
|----------|------|------|------|------|
| $\varepsilon_{\text{mach}}$ | $9.77 \times 10^{-4}$ | $7.81 \times 10^{-3}$ | $1.19 \times 10^{-7}$ | $2.22 \times 10^{-16}$ |
| Max finite value | $65504$ | $\approx 3.4 \times 10^{38}$ | $\approx 3.4 \times 10^{38}$ | $\approx 1.8 \times 10^{308}$ |
| Min normal | $6.10 \times 10^{-5}$ | $1.18 \times 10^{-38}$ | $1.18 \times 10^{-38}$ | $2.23 \times 10^{-308}$ |
| Decimal digits | $\sim 3$ | $\sim 2.3$ | $\sim 7$ | $\sim 15$ |
| Bits | 16 | 16 | 32 | 64 |

### F.2 Common Condition Numbers in ML

| Matrix type | Typical $\kappa(A)$ | Implication |
|------------|--------------------|----|
| Random Gaussian $A \in \mathbb{R}^{n \times n}$ | $\sim \sqrt{n}$ | Well-conditioned; easy to solve |
| Gram matrix $X^\top X$ for correlated features | $10^3$–$10^8$ | May need regularization |
| Hessian of neural loss at sharp minimum | $10^5$–$10^{10}$ | Slow gradient descent; use Adam |
| Hessian at flat region (saddle) | $\approx 0$ on some axes | Ill-conditioned; Newton fails |
| Hilbert matrix $n=10$ | $\approx 10^{13}$ | Numerically singular in fp32 |
| Discrete Laplacian $n \times n$ PDE | $O(n^2)$ | Requires preconditioning |

### F.3 Loss Scaling Reference Values

| Scale | When to use | Notes |
|-------|------------|-------|
| $2^{15} = 32768$ | Default initial scale for fp16 | PyTorch GradScaler default |
| $2^{10} = 1024$ | After repeated overflow reductions | Still keeps most gradients in fp16 range |
| $1$ (no scaling) | bf16 training | Not needed; bf16 has same range as fp32 |
| Dynamic | Best practice | GradScaler adjusts automatically |

---

## Appendix G: Glossary

| Term | Definition |
|------|-----------|
| **ULP** | Unit in the Last Place: spacing between consecutive floats near a given value |
| **Machine epsilon** $\varepsilon_{\text{mach}}$ | Smallest $\varepsilon$ with $\text{fl}(1 + \varepsilon) > 1$; equals $2^{-p}$ for $p$-bit mantissa |
| **Catastrophic cancellation** | Loss of significant digits when subtracting nearly-equal numbers |
| **Backward error** | Size of input perturbation that makes computed output exact |
| **Forward error** | Distance from computed output to true output |
| **Backward stable** | Algorithm whose backward error is $O(\varepsilon_{\text{mach}})$ |
| **Condition number** $\kappa$ | Worst-case ratio of relative output change to relative input change |
| **Overflow** | Result exceeds maximum representable value; becomes $\pm\infty$ |
| **Underflow** | Result is too small to represent normally; becomes subnormal or 0 |
| **NaN** | Not a Number: result of undefined operations ($0/0$, $\infty - \infty$) |
| **Subnormal** | Floating-point number smaller than the minimum normal; loses precision |
| **Rounding mode** | Rule for mapping exact values to the nearest representable float |
| **Stochastic rounding** | Random rounding, unbiased in expectation; used in low-precision training |
| **Loss scaling** | Multiplying the loss by a large factor before backprop to prevent gradient underflow |
| **Mixed precision** | Using different formats for different parts of training (e.g., fp16 compute + fp32 weights) |
| **bf16** | Brain Float 16: 8-bit exponent (same range as fp32) + 7-bit mantissa |
| **fp8** | 8-bit floating point; two variants: E4M3 (more precision) and E5M2 (more range) |
| **Kahan summation** | Compensated summation algorithm achieving $O(\varepsilon_{\text{mach}})$ error independent of $n$ |
| **Pairwise summation** | Binary-tree summation achieving $O(\log n \, \varepsilon_{\text{mach}})$ error |
| **FTZ** | Flush to Zero: hardware mode that maps subnormals to 0 for speed |
| **GradScaler** | PyTorch class implementing dynamic loss scaling for fp16 training |

---

## Appendix H: Connections to Other Sections

### H.1 Floating-Point and Optimization

The choice of floating-point format directly impacts optimizer behavior:

**Adam in fp16:** Adam maintains first moment $m_t$ and second moment $v_t$ (moving averages of gradients and squared gradients). These accumulators grow slowly over training. In fp16, small updates $\alpha m_t / \sqrt{v_t}$ can underflow. This is why Adam's optimizer states are almost always kept in fp32, even when gradients are computed in fp16.

**Gradient clipping and numerical overflow:** When gradients overflow to $\pm\infty$ in fp16, the global norm $\|\mathbf{g}\|_2 = \sqrt{\sum g_i^2}$ also becomes infinity. Clipping to $G_{\max}$ when norm is $\infty$ produces division $0/0 = \text{NaN}$. PyTorch's `clip_grad_norm_` guards against this by checking for non-finite norms before clipping.

**Numerical second-order methods:** Computing the Hessian $\nabla^2 L$ via finite differences requires step size $h \approx \varepsilon_{\text{mach}}^{1/3}$ — balancing truncation error $O(h^2)$ with cancellation error $O(\varepsilon_{\text{mach}}/h)$. For fp32, optimal $h \approx (1.2 \times 10^{-7})^{1/3} \approx 5 \times 10^{-3}$.

### H.2 Floating-Point and Neural Network Architecture

**Layer normalization:** LayerNorm computes $\text{LayerNorm}(\mathbf{x}) = (\mathbf{x} - \mu) / (\sigma + \epsilon) \times \gamma + \beta$ where $\epsilon$ (often $10^{-5}$ or $10^{-6}$) prevents division by zero when $\sigma \approx 0$. This $\epsilon$ is a numerical safeguard, not a mathematical constant.

**Softmax temperature scaling:** Attention logits $\mathbf{q}^\top \mathbf{k} / \sqrt{d_k}$ scale inversely with dimension. Without the $\sqrt{d_k}$ divisor, logits grow as $\Theta(\sqrt{d_k})$ and softmax becomes peaky (near one-hot), causing vanishing gradients. The scaling keeps logits in a numerically well-conditioned range.

**Residual connections:** $\mathbf{h}_{l+1} = \mathbf{h}_l + F(\mathbf{h}_l)$ prevents the gradient vanishing problem: $\partial L / \partial \mathbf{h}_l = \partial L / \partial \mathbf{h}_{l+1} (I + \partial F / \partial \mathbf{h}_l)$. The identity term ensures at least one gradient path with multiplication by 1 — no floating-point underflow across layers.

### H.3 Floating-Point and Transformer Efficiency

**FlashAttention:** The standard attention computation $\text{softmax}(QK^\top / \sqrt{d_k}) V$ requires materializing the $n \times n$ attention matrix. FlashAttention (Dao et al., 2022) computes this in chunks, maintaining online softmax statistics in fp32 while storing intermediate results in fp16/bf16. This avoids both overflow and numerical drift across the $n$ tokens.

**KV cache quantization:** In inference, key and value matrices are cached across decoding steps. Quantizing this cache from fp16 to int8 (or int4) reduces memory by 2-4×. The quantization error adds noise to the attention scores — understanding the rounding error model tells you this noise is $O(\varepsilon_{int}) \sim O(2^{-7})$ per element, tolerable for typical sequence lengths.

---

## Appendix I: Further Reading

### Foundational Papers

1. **Goldberg, D.** (1991). What every computer scientist should know about floating-point arithmetic. *ACM Computing Surveys*, 23(1), 5–48. — The definitive reference; free online.

2. **Higham, N.J.** (2002). *Accuracy and Stability of Numerical Algorithms* (2nd ed.). SIAM. — Chapter 2–3: rounding errors; Chapter 9: Gaussian elimination; definitive modern reference.

3. **Wilkinson, J.H.** (1963). *Rounding Errors in Algebraic Processes*. Prentice-Hall. — Original backward error analysis.

4. **Kahan, W.** (1965). Pracniques: Further remarks on reducing truncation errors. *Communications of the ACM*, 8(1), 40.

### ML-Specific Papers

5. **Micikevicius, P. et al.** (2018). Mixed Precision Training. *ICLR 2018*. — The paper that established fp16 mixed-precision training as standard.

6. **Kalamkar, D. et al.** (2019). A Study of BFLOAT16 for Deep Learning Training. *arXiv:1905.12322*. — Why bf16 works better than fp16 for training.

7. **Noune, B. et al.** (2022). 8-bit Numerical Formats for Deep Neural Networks. *arXiv:2206.02915*. — fp8 training foundations.

8. **Dao, T. et al.** (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. *NeurIPS 2022*. — Numerically stable attention at scale.

### Textbooks

9. **Trefethen, L.N. & Bau, D.** (1997). *Numerical Linear Algebra*. SIAM. — Excellent treatment of backward error; Chapter 12-15 on floating-point and stability.

10. **Press, W.H. et al.** (2007). *Numerical Recipes: The Art of Scientific Computing* (3rd ed.). Cambridge. — Practical algorithms with implementation details.

---

*[← Back to Numerical Methods](../README.md) | [Next: Numerical Linear Algebra →](../02-Numerical-Linear-Algebra/notes.md)*

---

## Appendix J: Extended Examples and Case Studies

### J.1 Case Study: NaN Debugging in a Transformer

A common production scenario: you launch training, the loss drops for 100 steps, then suddenly `loss = nan`. Here is a systematic debugging procedure grounded in floating-point theory.

**Step 1: Identify when NaN first appears.**
```python
for step, batch in enumerate(dataloader):
    with torch.autocast('cuda', torch.bfloat16):
        logits = model(batch['input_ids'])
        loss = criterion(logits, batch['labels'])
    if torch.isnan(loss):
        print(f"NaN loss at step {step}")
        break
```

**Step 2: Check for NaN in individual components.**
Common sources in order of frequency:
- `log(0)`: occurs when model outputs exactly 0 probability for the target class
- `0/0`: LayerNorm with $\sigma = 0$ (all-constant hidden states)
- Overflow then $\infty - \infty$: attention logits too large (missing $1/\sqrt{d_k}$ scaling)
- Exploding gradients: accumulated for many steps without clipping

**Step 3: Numerical fixes.**
- Replace `log(softmax(x)[y])` with `F.cross_entropy(x, y)` (stable)
- Add `eps=1e-6` to LayerNorm (PyTorch default)
- Add gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
- Switch from fp16 to bf16 (eliminates overflow entirely)

### J.2 Case Study: Ill-Conditioned Normal Equations

Linear regression via normal equations: $\hat{\boldsymbol\beta} = (X^\top X)^{-1} X^\top \mathbf{y}$.

The condition number of $X^\top X$ is $\kappa(X^\top X) = \kappa(X)^2$. If $X$ has condition number $\kappa(X) = 10^4$ (common for correlated features), then $\kappa(X^\top X) = 10^8$ — near the threshold of fp32 accuracy.

**Safer alternative:** Use the QR decomposition of $X = QR$; then $\hat{\boldsymbol\beta} = R^{-1} Q^\top \mathbf{y}$. The condition number of $R$ equals $\kappa(X) = 10^4$ — half the digits lost.

**Safest:** Use `scipy.linalg.lstsq` which internally uses SVD-based pseudo-inverse with automatic rank determination. Condition number is $\kappa(X)$, with singular values below `rcond * sigma_max` set to zero.

### J.3 Case Study: fp16 vs bf16 in Attention

In a transformer with $d_{\text{model}} = 4096$ and $d_k = d_{\text{model}} / H = 128$ (32 heads):

**Attention logits** (before softmax): $\mathbf{q}^\top \mathbf{k} / \sqrt{128}$. Without the $1/\sqrt{d_k}$ factor, the dot product is $\Theta(\sqrt{d_k}) = \Theta(11)$ in expectation — fine. With random initialization, dot products can reach $\pm 50$ before training stabilizes.

**fp16 concern:** $e^{50} \approx 5 \times 10^{21}$ — severe overflow. But with the $1/\sqrt{128} = 1/11.3$ factor: $e^{50/11.3} \approx e^{4.4} \approx 82$ — safely within fp16 max of 65504. Gradient of the attention weights passes through softmax's Jacobian, which has values $\le 1/4$ — fine for fp16.

**bf16:** Identical analysis; bf16's wider exponent range means even moderate overflow during early training is handled gracefully, whereas fp16 needs careful initialization.

### J.4 Numerical Stability of Common Activation Functions

| Activation | Naive formula | Stable implementation | Issue |
|-----------|--------------|----------------------|-------|
| Sigmoid | $1/(1 + e^{-x})$ | $e^x/(1+e^x)$ for $x < 0$; $1/(1+e^{-x})$ for $x \ge 0$ | Overflow for $x \ll 0$ |
| Softplus | $\log(1 + e^x)$ | $x + \log(1 + e^{-x})$ for $x > 0$ | Overflow for large $x$ |
| GELU | $x\Phi(x)$ (error function) | Use precomputed tables or polynomial approx | $\Phi$ computation accuracy |
| Swish | $x \sigma(x) = x/(1+e^{-x})$ | Standard; no overflow for normal $x$ | None in typical range |
| log-softmax | $\log(\text{softmax}(x))$ | $x - \max(x) - \log\sum e^{x_i - \max(x_i)}$ | Overflow in naive softmax |

### J.5 Gradient Checkpointing and Numerical Precision

Gradient checkpointing (Chen et al., 2016) saves memory by not storing all activations during the forward pass; it recomputes them during backward. **Numerical concern:** The recomputed activations may differ slightly from the original computation due to floating-point non-associativity (different thread ordering on GPU).

For deterministic behavior with gradient checkpointing, use:
```python
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False  # Disable TF32 (approximate matmul)
```

TF32 (TensorFloat-32) is NVIDIA's format that uses fp32 exponent/sign but only 10 bits of mantissa for matmul accumulation — faster but less precise. Enabled by default in PyTorch since 1.7. For reproducible science, disable it; for production training, leave it enabled.

---

## Appendix K: Self-Assessment Questions

1. What is the machine epsilon of fp32? Of bf16? Of fp16?
2. Why is bf16 preferred over fp16 for training large language models?
3. What is catastrophic cancellation? Give an example involving subtraction.
4. State the fundamental rounding error model. What does $\delta$ represent?
5. What is the backward error of an algorithm? Why is backward stability more useful than forward stability?
6. If a matrix has condition number $\kappa = 10^8$ and you solve $A\mathbf{x} = \mathbf{b}$ in fp32 ($\varepsilon_{\text{mach}} \approx 10^{-7}$), what is the expected relative error in the solution?
7. Describe the Kahan summation algorithm. What is its error guarantee?
8. Why does `log(softmax(x))` fail for large inputs? How is `log_softmax` implemented stably?
9. What is loss scaling and when is it needed? Not needed?
10. Define $\kappa(A)$ in terms of singular values. What does $\kappa(A) = 1$ mean geometrically?
11. What is the difference between overflow and underflow? Which causes NaN?
12. Why does FlashAttention use fp32 for softmax accumulation even in bf16 mode?

---

*[← Back to Numerical Methods](../README.md) | [Next: Numerical Linear Algebra →](../02-Numerical-Linear-Algebra/notes.md)*

---

## Appendix L: Numerical Analysis in PyTorch — Implementation Patterns

### L.1 Computing Machine Epsilon Empirically

The standard algorithm to find machine epsilon without library calls:

```python
def find_machine_epsilon(dtype):
    """Find machine epsilon for a given dtype by halving."""
    one = np.ones(1, dtype=dtype)
    eps = dtype(1.0)
    while dtype(1.0) + eps / dtype(2.0) > dtype(1.0):
        eps = eps / dtype(2.0)
    return eps

for dt in [np.float16, np.float32, np.float64]:
    emp_eps = find_machine_epsilon(dt)
    lib_eps = np.finfo(dt).eps
    print(f"{dt.__name__:12s}: empirical={emp_eps:.3e}, library={lib_eps:.3e}")
```

This gives:
- float16: $9.77 \times 10^{-4}$
- float32: $1.19 \times 10^{-7}$
- float64: $2.22 \times 10^{-16}$

### L.2 Visualizing the Floating-Point Number Line

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate all positive fp16 normals in [1, 2)
# fp16: bias=15, so exponent bits = 01111 = 15, mantissa = 10 bits
mantissas = np.arange(0, 2**10)
values = 1.0 + mantissas / 2**10  # fp16 values in [1, 2)
spacings = np.diff(values)

# Plot: density of fp16 values in different ranges
ranges = [(0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0)]
for lo, hi in ranges:
    # All fp16 values in this range
    count = 2**10  # same mantissa count per exponent
    spacing = (hi - lo) / count
    print(f"[{lo}, {hi}): {count} values, spacing = {spacing:.6f}")
```

Output shows: every power-of-2 interval contains exactly 1024 fp16 values (for normals), but the spacing doubles with each interval — geometric, not arithmetic, distribution.

### L.3 Detecting and Fixing Common Numerical Issues

```python
class NumericalGuard:
    """Context manager for detecting numerical issues during training."""
    
    def __init__(self, model, check_inputs=True, check_outputs=True):
        self.model = model
        self.hooks = []
        self.check_inputs = check_inputs
        self.check_outputs = check_outputs
    
    def __enter__(self):
        def hook(module, input, output):
            name = type(module).__name__
            if self.check_inputs:
                for i, x in enumerate(input):
                    if isinstance(x, torch.Tensor):
                        if torch.isnan(x).any():
                            raise ValueError(f"NaN in {name} input {i}")
                        if torch.isinf(x).any():
                            raise ValueError(f"Inf in {name} input {i}")
            if self.check_outputs:
                if isinstance(output, torch.Tensor):
                    if torch.isnan(output).any():
                        raise ValueError(f"NaN in {name} output")
        
        for module in self.model.modules():
            self.hooks.append(module.register_forward_hook(hook))
        return self
    
    def __exit__(self, *args):
        for hook in self.hooks:
            hook.remove()
```

### L.4 Comparing Summation Methods

```python
def naive_sum(arr):
    """Standard left-to-right accumulation."""
    s = type(arr[0])(0)
    for x in arr:
        s += x
    return s

def kahan_sum(arr):
    """Kahan compensated summation."""
    s = type(arr[0])(0)
    c = type(arr[0])(0)
    for x in arr:
        y = x - c
        t = s + y
        c = (t - s) - y
        s = t
    return s

def pairwise_sum(arr):
    """Recursive pairwise (binary tree) summation."""
    n = len(arr)
    if n == 1:
        return arr[0]
    mid = n // 2
    return pairwise_sum(arr[:mid]) + pairwise_sum(arr[mid:])

# Test: sum 1 million copies of 1/3 in fp32
n = 1_000_000
x = np.full(n, 1/3, dtype=np.float32)
true_sum = n / 3  # exact in float64

for name, fn in [('Naive', naive_sum), ('Kahan', kahan_sum)]:
    result = fn(list(x))
    error = abs(float(result) - true_sum)
    rel_error = error / abs(true_sum)
    print(f"{name:8s}: result={result:.10f}, rel_error={rel_error:.2e}")
```

### L.5 Stable Implementations of Common Functions

```python
def log_sum_exp(x):
    """Numerically stable log(sum(exp(x))) for array x."""
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))

def softmax(x):
    """Numerically stable softmax."""
    x_shifted = x - np.max(x)
    e = np.exp(x_shifted)
    return e / np.sum(e)

def log_softmax(x):
    """Numerically stable log-softmax."""
    return x - log_sum_exp(x)

def cross_entropy(logits, target):
    """Numerically stable cross-entropy: -log(softmax(logits)[target])."""
    return -log_softmax(logits)[target]

def sigmoid(x):
    """Numerically stable sigmoid avoiding overflow for large negative x."""
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))

def softplus(x):
    """Numerically stable softplus: log(1 + exp(x))."""
    return np.where(x > 20,
                    x,  # log(1 + exp(x)) ≈ x for large x
                    np.log1p(np.exp(x)))  # log1p is more accurate than log(1+...)
```

### L.6 Loss Scaling Implementation

```python
class SimpleLossScaler:
    """Minimal loss scaler for fp16 training."""
    
    def __init__(self, init_scale=2**15, scale_factor=2.0, scale_window=2000):
        self.scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self._steps_since_update = 0
        self._successful_steps = 0
    
    def scale_loss(self, loss):
        return loss * self.scale
    
    def step(self, optimizer, params):
        """Unscale gradients and step, or skip if overflow detected."""
        # Check for overflow
        overflow = any(
            torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
            for p in params if p.grad is not None
        )
        
        if overflow:
            # Reduce scale and skip step
            self.scale /= self.scale_factor
            print(f"Overflow detected! Scale reduced to {self.scale}")
        else:
            # Unscale gradients
            for p in params:
                if p.grad is not None:
                    p.grad /= self.scale
            optimizer.step()
            self._successful_steps += 1
            
            # Increase scale every scale_window successful steps
            if self._successful_steps % self.scale_window == 0:
                self.scale *= self.scale_factor
                print(f"Scale increased to {self.scale}")
```

---

*[← Back to Numerical Methods](../README.md) | [Next: Numerical Linear Algebra →](../02-Numerical-Linear-Algebra/notes.md)*

---

## Appendix M: Historical Perspective — The Road to IEEE 754

### M.1 Before IEEE 754 (Pre-1985)

Before the IEEE 754 standard, every computer manufacturer used its own floating-point format. IBM mainframes used base-16 (hexadecimal) floating point. DEC VAX used its own 32-bit format. CDC 6600 used 60-bit words with 48-bit mantissa. Programs written for one machine could not be reliably ported to another — the same computation produced different results depending on the hardware.

The chaos had real consequences: scientific simulations gave different answers on different machines; numerical algorithms required hand-tuning for each platform; software bugs were indistinguishable from floating-point differences. The 1970s saw a growing recognition that a universal standard was needed.

### M.2 The IEEE 754 Committee (1977–1985)

The IEEE 754 working group (chaired by Jerome Coonen, with key contributions from William Kahan and Harold Stone) spent eight years developing the standard. Kahan — who would win the 1989 Turing Award partly for this work — insisted on several features that were controversial but proved crucial:

1. **Round-to-nearest-even as default:** Reduces systematic bias in long computations
2. **Gradual underflow (subnormals):** Prevents sudden loss of precision near zero
3. **Signed zeros:** Enables correct complex analysis identities ($1/(+0) = +\infty \ne 1/(-0) = -\infty$)
4. **NaN as a value (not an exception):** Allows computation to continue past undefined operations

The Intel 8087 coprocessor (1980) was the first commercial implementation, years before the standard was formally published in 1985. Its success validated the design.

### M.3 IEEE 754-2008 and Beyond

The 2008 revision added:
- **fp16 (half precision):** Initially for graphics/ML; 5-bit exponent limits range to $\approx 65504$
- **Decimal floating-point:** For financial computations where $0.1 + 0.2 = 0.3$ exactly
- **Fused multiply-add (FMA):** $\text{fl}(a \times b + c)$ computed with only one rounding error, not two

FMA is particularly important for ML: matrix multiplication decomposes into FMA operations, and modern GPU tensor cores implement FMA in mixed precision (fp16/bf16 multiply + fp32 accumulate) for maximum throughput with minimum precision loss.

### M.4 The ML Formats (2017–2024)

The ML revolution created demand for formats the IEEE committee never anticipated:

**bf16 (2018):** Google Brain developed Brain Float 16 for TPUs. The key insight: training stability requires dynamic range (exponent bits), not precision (mantissa bits). Keep all 8 exponent bits of fp32, sacrifice 16 mantissa bits. The result trains as stably as fp32 at half the cost.

**tf32 (2020):** NVIDIA's TensorFloat-32 uses fp32's 8-bit exponent and 10-bit mantissa (same as fp16) for matmul accumulation, giving 3× speedup with minimal accuracy loss. Enabled by default in PyTorch on Ampere and newer GPUs.

**fp8 E4M3 / E5M2 (2022):** For H100 training at extreme throughput. Requires per-tensor scaling factors and careful format selection per layer (E4M3 for weights/activations, E5M2 for gradients).

**fp4 / int4 (experimental):** Used for inference quantization (GGUF format, llama.cpp). Weights stored as 4-bit integers with a shared scale factor per block. 4-bit reduces model size by 8× vs fp32, enabling 70B-parameter models on a single consumer GPU.

---

## Appendix N: Quick Reference and Formulas

### N.1 Key Inequalities

$$|\text{fl}(x) - x| \le \frac{\varepsilon_{\text{mach}}}{2} |x| \quad \text{(single rounding)}$$

$$|\text{fl}(x \circ y) - (x \circ y)| \le \varepsilon_{\text{mach}} |x \circ y| \quad \text{(one operation, round-to-nearest)}$$

$$\left|\sum_{i=1}^n x_i - \widehat{\sum_{i=1}^n x_i}\right| \le (n-1)\varepsilon_{\text{mach}} \sum_{i=1}^n |x_i| \quad \text{(naive summation)}$$

$$\left|\text{Kahan sum} - \sum x_i\right| \le 2\varepsilon_{\text{mach}} \sum |x_i| \quad \text{(Kahan)}$$

$$\frac{\|\Delta\mathbf{x}\|}{\|\mathbf{x}\|} \le \kappa(A) \frac{\|\Delta\mathbf{b}\|}{\|\mathbf{b}\|} \quad \text{(linear system perturbation)}$$

### N.2 Algorithms Reference

**Stable softmax:**
$$\text{softmax}(\mathbf{x})_i = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}, \quad m = \max_j x_j$$

**Stable log-sum-exp:**
$$\text{logsumexp}(\mathbf{x}) = m + \log\sum_j e^{x_j - m}, \quad m = \max_j x_j$$

**Kahan step:**
$$y_k = x_k - c; \quad t = s + y_k; \quad c = (t - s) - y_k; \quad s = t$$

**Optimal finite-difference step:**
$$h^* = \varepsilon_{\text{mach}}^{1/2} \quad \text{(forward differences)}; \quad h^* = \varepsilon_{\text{mach}}^{1/3} \quad \text{(central differences)}$$

**Condition number bounds:**
$$\kappa_2(A) = \sigma_{\max}(A)/\sigma_{\min}(A); \quad \kappa_1(A) = \|A\|_1 \|A^{-1}\|_1$$

### N.3 Checklist: Numerically Stable Implementation

Before deploying any numerical computation, verify:

- [ ] All `log()` calls are protected: `log(x + eps)` or `log_softmax()` instead of `log(softmax())`
- [ ] All divisions are protected: `x / (y + eps)` for division by potentially-zero values  
- [ ] Softmax is computed with max-subtraction (`stable_softmax`) not naive
- [ ] Cross-entropy uses `F.cross_entropy(logits, labels)` not `F.nll_loss(F.log_softmax(logits), labels)` (equivalent but one extra call)
- [ ] Float comparisons use `torch.isclose` with appropriate `atol` and `rtol`, not `==`
- [ ] Accumulation of small differences is done in higher precision (`float64` or Kahan)
- [ ] Gradient clipping is enabled: `clip_grad_norm_(params, max_norm=1.0)`
- [ ] NaN detection: `torch.isnan(loss)` checked at training start
- [ ] Matrix condition number checked before solving: `np.linalg.cond(A) < 1/eps`
- [ ] bf16 used instead of fp16 for new training runs

---

*[← Back to Numerical Methods](../README.md) | [Next: Numerical Linear Algebra →](../02-Numerical-Linear-Algebra/notes.md)*

---

## Appendix O: Worked Problems with Full Solutions

### O.1 Problem: Compute $e^x - 1$ for Small $x$

**Problem:** Compute $f(x) = e^x - 1$ accurately for $x = 10^{-8}$ in fp32.

**Naive approach:** $e^{10^{-8}} \approx 1.00000001$ in fp32 (only 7 decimal digits). Then $1.00000001 - 1 = 10^{-8}$... but in fp32: $\text{fl}(e^{10^{-8}}) = 1.0$ (exact 1.0 in fp32, since $10^{-8}$ is below the fp32 precision at magnitude 1). Result: $f(10^{-8})_{\text{fp32, naive}} = 0.0$. Relative error: 100%.

**Stable approach:** Use the identity $e^x - 1 = x + x^2/2 + x^3/6 + \ldots$ for small $x$, implemented as `numpy.expm1(x)`:
$$\text{expm1}(10^{-8}) = 10^{-8} \times (1 + 10^{-8}/2 + \ldots) \approx 10^{-8}$$

The `expm1` function is computed without the catastrophic cancellation — it directly computes $e^x - 1$ without subtracting large numbers. Result: $10^{-8}$. Relative error: machine precision.

**Rule:** Use `np.expm1(x)` instead of `np.exp(x) - 1` whenever $|x| \ll 1$. Similarly, use `np.log1p(x)` instead of `np.log(1+x)` for small $x$.

### O.2 Problem: Running Mean and Variance

**Problem:** Compute the mean and variance of a stream of numbers $x_1, x_2, \ldots, x_n$ without storing all values.

**Naive:** Store all values and compute $\bar{x} = \frac{1}{n}\sum x_i$, $\sigma^2 = \frac{1}{n}\sum(x_i - \bar{x})^2$. Requires $O(n)$ memory.

**One-pass (unstable):** $\sigma^2 = \frac{1}{n}\sum x_i^2 - \bar{x}^2$. Catastrophically unstable when $|\bar{x}|$ is large relative to $\sigma$.

**Welford's online algorithm (stable):**
```
Initialize: M_1 = x_1, S_1 = 0, n = 1
For k = 2, 3, ...:
    n += 1
    delta = x_k - M_{k-1}
    M_k = M_{k-1} + delta / n
    delta2 = x_k - M_k
    S_k = S_{k-1} + delta * delta2
Variance = S_n / (n - 1)  (sample variance)
```

This is used in: PyTorch `BatchNorm` (online mean/variance), Adam optimizer (running mean of gradients), gradient clipping (running norm estimation).

### O.3 Problem: Matrix-Vector Product Error Bound

**Problem:** Bound the error in $A\mathbf{x}$ computed in fp32, where $A \in \mathbb{R}^{m \times n}$ and $\mathbf{x} \in \mathbb{R}^n$.

Each entry of the product is a dot product: $(A\mathbf{x})_i = \sum_{j=1}^n A_{ij} x_j$.

Each such dot product of $n$ terms has forward error at most $(n-1)\varepsilon_{\text{mach}} \sum_j |A_{ij}||x_j|$.

In matrix form:
$$\|\widehat{A\mathbf{x}} - A\mathbf{x}\|_\infty \le (n-1)\varepsilon_{\text{mach}} |A| |\mathbf{x}|_\infty$$
where $|A|$ denotes the entry-wise absolute value.

**For a GPU matmul** using FMA with fp32 accumulation: the effective error is reduced by approximately factor $\sqrt{n}$ due to cancellations, but the worst-case bound above still applies.

**Implication for attention:** The attention score $\mathbf{q}^\top \mathbf{k} / \sqrt{d_k}$ is a dot product of length $d_k$. In fp16 ($\varepsilon \approx 10^{-3}$) without fp32 accumulation: error $\le d_k \times 10^{-3} \times$ (product magnitude). For $d_k = 128$: 12.8% relative error — unacceptably large. With fp32 accumulation (as in FlashAttention): error $\le d_k \times 10^{-7}$ — perfectly fine.

### O.4 Problem: Condition Number of the Attention Matrix

**Problem:** What is the condition number of the softmax output $\mathbf{p} = \text{softmax}(\mathbf{z})$?

The Jacobian of softmax is $J = \text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$, which has eigenvalues $0$ and $p_i(1-p_i)$ for each $i$. This matrix is singular (rank $n-1$), so in the strictest sense, the softmax map has infinite condition number (it maps from $\mathbb{R}^n$ to the probability simplex, a lower-dimensional manifold).

**Practical question:** How sensitive are the softmax outputs to perturbations in the logits? The answer depends on the sharpness of the softmax. A "peaky" distribution (one $p_i \approx 1$) has Jacobian entries near 0 — small gradient flow, effectively zero sensitivity. A "flat" distribution (all $p_i = 1/n$) has maximum sensitivity: the Frobenius norm of $J$ is maximized.

**For training:** Sharp attention (peaky softmax) causes vanishing gradients through the attention weights. This is prevented by: (1) temperature scaling $1/\sqrt{d_k}$; (2) attention dropout (randomly zero-ing some weights, softening the distribution); (3) label smoothing (softens the target distribution, preventing overconfident predictions).

---

## Appendix P: Summary Statistics of Chapter

```
CHAPTER §01 SUMMARY
════════════════════════════════════════════════════════════════════════

  Core concepts: 10 (IEEE 754, ε_mach, rounding, cancellation,
                     Kahan, condition number, stability, formats,
                     mixed precision, stable implementations)

  Key formulas: 12 (rounding model, Kahan step, stable softmax,
                    logsumexp, κ(A), error bounds, ...)

  Numerical formats: 8 (fp64, fp32, bf16, fp16, fp8 E4M3, fp8 E5M2,
                        int8, int4)

  AI connections: 15 (loss scaling, NaN debugging, FlashAttention,
                      TF32, bf16 training, gradient clipping, ...)

  Exercises: 8 (★ to ★★★), covering all major topics

  Appendices: 16 (A through P)

  Notes length: ~1700 lines
  Theory cells: 50+
  Exercises: 8 graded problems (24 cells)

════════════════════════════════════════════════════════════════════════
```

---

## Appendix Q: Deep Dive — IEEE 754 Special Value Arithmetic

Understanding how special values propagate is essential for debugging AI training runs.

### Q.1 NaN Propagation Rules

NaN (Not a Number) is **sticky**: any arithmetic operation with a NaN produces a NaN.

```
NaN + x  = NaN    for any x (including NaN)
NaN * 0  = NaN    (not 0, despite "anything times zero is zero")
NaN < x  = False  for any x
NaN == NaN = False  (IEEE mandates this — NaN is not equal to itself)
```

**The self-inequality of NaN** is the canonical way to detect it:
```python
def is_nan(x):
    return x != x  # True only when x is NaN
```

In PyTorch: `torch.isnan(x)` or `x != x`.

**NaN in gradient computation:** If any parameter gradient is NaN, the entire parameter update step is corrupted. A single NaN in one layer's weight will propagate backward through the chain rule to corrupt all earlier layer gradients.

**Debugging strategy for NaN gradients:**
1. Register forward hooks to check activations for NaN after each layer
2. Register backward hooks to check gradients for NaN
3. Binary search over layers to find the first occurrence
4. Check for division by zero in custom loss functions (log(0), 0/0)
5. Check for overflow in exponentials (exp(large_number))

### Q.2 Infinity Arithmetic

Infinity follows extended real arithmetic:
```
+∞ + (+∞)  = +∞
+∞ + (-∞)  = NaN    (indeterminate form)
+∞ * 0     = NaN    (indeterminate form)
+∞ * (+∞)  = +∞
1 / 0      = +∞     (for positive numerator)
-1 / 0     = -∞
0 / 0      = NaN
x / +∞     = 0      for finite x
```

**Loss spike analysis:** When a loss value becomes `inf`, trace backward:
- `log(0)` — zero probability assigned to correct class
- `exp(x)` for large `x` before normalization (use log-sum-exp)
- Division by a very small denominator (batch norm with near-zero variance)
- Accumulated gradients that overflow fp16 range before gradient clipping

### Q.3 Signed Zero

IEEE 754 has both `+0` and `-0`. They compare equal (`+0 == -0` is True) but differ in division:
```
1 / (+0) = +∞
1 / (-0) = -∞
```

Signed zero matters in:
- **Complex number arithmetic**: branch cuts depend on sign of zero imaginary part
- **Sorting algorithms**: `sort([+0, -0])` may or may not preserve order
- **Gradient of relu at zero**: subgradient convention can produce `+0` or `-0`

### Q.4 Subnormal Performance Impact

On most hardware, operations involving subnormal numbers (also called denormals) execute 10-100x slower than normal operations, because they require software emulation or special hardware paths.

In training:
- Very small weight values gradually entering subnormal range → sudden throughput drop
- Gradient values approaching zero → subnormal gradients → training slows
- **Fix**: Set the **flush-to-zero (FTZ)** flag, which replaces subnormals with zero
  - PyTorch: enabled by default in CUDA
  - NumPy: `np.seterr(under='ignore')` + platform FTZ setting
  - Downside: FTZ introduces larger underflow errors but avoids performance cliff

---

## Appendix R: Floating-Point Error Case Studies from AI Research

### Case Study 1: The fp16 Loss Explosion Problem (2017-2018)

When researchers first attempted to train large language models in fp16, they observed frequent loss explosions. Investigation revealed:
- Gradient magnitudes of ~10⁻⁴ to 10⁻³ during stable training
- fp16 minimum positive normal: ~6.1 × 10⁻⁵
- Many gradients underflowed to zero → effectively frozen parameters
- Sudden instability from accumulated representation errors in weights

**Solution (Micikevicius et al., 2017):** Mixed-precision training with loss scaling:
1. Maintain **fp32 master copy** of all weights
2. Cast to fp16 for forward and backward pass (4x speedup)
3. Scale loss by $S$ (typically 2⁸ to 2¹⁵) before backward
4. Check for overflow (any gradient is Inf or NaN)
5. If no overflow: unscale, apply gradient clipping, update fp32 weights
6. If overflow: skip step, halve $S$

This framework is now standard in `torch.cuda.amp`.

### Case Study 2: Attention Score Overflow in Early Transformers

In the original Transformer (Vaswani et al., 2017), attention scores are:
$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

Without the $\sqrt{d_k}$ scaling, with $d_k = 64$:
- $q \cdot k$ can have magnitude $\sim d_k = 64$ (assuming unit-variance features)
- After softmax, extreme values dominate: $\text{softmax}([10, 10, ..., 64]) \approx [0, 0, ..., 1]$
- Gradient of softmax becomes nearly zero → vanishing gradients

The $1/\sqrt{d_k}$ factor ensures dot products have variance 1, keeping softmax in its informative regime.

**fp16 version:** With $d_k = 64$ and input std $\approx 1$, raw scores $\approx \mathcal{N}(0, 64)$, so $|qk| > 65504$ (fp16 max) with non-negligible probability when $d_k$ is large. FlashAttention computes attention in blocks, using online softmax with numerical rescaling to avoid materializing the full $N \times N$ attention matrix.

### Case Study 3: Layer Normalization Stability

Layer normalization computes:
$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}}$$

where $\varepsilon$ (typically $10^{-5}$ to $10^{-8}$) prevents division by zero.

**Numerical pitfall:** If all activations in a layer are identical (common during initialization or after a bad update), $\sigma^2 = 0$ exactly in floating point. Without $\varepsilon$, we'd compute $0/0 = \text{NaN}$.

**Stability consideration:** The choice of $\varepsilon$ matters:
- Too small ($10^{-12}$): offers no protection when $\sigma^2$ is represented as exact zero
- Too large ($10^{-3}$): changes the effective normalization for small-variance activations
- Standard choice $10^{-5}$ balances stability with fidelity

**Catastrophic cancellation in variance:** Computing $\sigma^2 = \mathbb{E}[x^2] - \mu^2$ suffers catastrophic cancellation when $\mu^2 \approx \mathbb{E}[x^2]$ (small variance). Welford's online algorithm avoids this (see Appendix O).

### Case Study 4: Gradient Checkpointing and Recomputation Errors

Gradient checkpointing saves memory by recomputing intermediate activations during the backward pass instead of storing them. This recomputation uses the **same inputs** but may use **different precision** if the precision mode changes between forward and backward.

**Source of error:** In mixed-precision training with `autocast`, the recomputed forward pass during backward may use slightly different floating-point operations than the original forward. The resulting gradient is mathematically correct (same bit pattern inputs) but the error can accumulate differently.

**Industry practice:** PyTorch's `torch.utils.checkpoint.checkpoint` handles this correctly by default. The key requirement is that the checkpointed function must be deterministic — stochastic operations (like Dropout) must use the same random seed in forward and recomputation.

---

## Appendix S: Floating-Point Benchmarks and Profiling

### Measuring fp32 vs bf16 Throughput

```python
import torch
import time

def benchmark_matmul(dtype, size=4096, n_warmup=5, n_trials=20):
    """Benchmark matrix multiplication throughput."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    A = torch.randn(size, size, dtype=dtype, device=device)
    B = torch.randn(size, size, dtype=dtype, device=device)
    
    # Warmup
    for _ in range(n_warmup):
        C = torch.mm(A, B)
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(n_trials):
        C = torch.mm(A, B)
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    # FLOPS: 2 * N^3 for N x N matmul
    flops = 2 * size**3 * n_trials
    tflops = flops / elapsed / 1e12
    return tflops

# Typical results on A100 GPU:
# fp32: ~19.5 TFLOPS
# bf16: ~77.0 TFLOPS  (4x speedup with Tensor Cores)
# fp16: ~77.0 TFLOPS  (similar to bf16)
```

### Detecting Numerical Issues in Training

```python
def register_nan_hooks(model):
    """Register hooks to detect NaN/Inf in forward and backward passes."""
    hooks = []
    
    def forward_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                if torch.isnan(output).any():
                    print(f"NaN detected in forward: {name}")
                if torch.isinf(output).any():
                    print(f"Inf detected in forward: {name}")
        return hook
    
    def backward_hook(name):
        def hook(module, grad_input, grad_output):
            for i, g in enumerate(grad_input):
                if g is not None and torch.isnan(g).any():
                    print(f"NaN in grad_input[{i}] at: {name}")
        return hook
    
    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(forward_hook(name)))
        hooks.append(module.register_full_backward_hook(backward_hook(name)))
    
    return hooks  # Call hook.remove() to deregister

# Usage:
# hooks = register_nan_hooks(model)
# train_step(model, batch)
# for h in hooks: h.remove()
```

### Condition Number Monitoring

```python
def monitor_weight_conditioning(model, threshold=1e4):
    """Monitor condition numbers of weight matrices."""
    ill_conditioned = []
    for name, param in model.named_parameters():
        if param.dim() >= 2:
            # Use fast approximation via max/min singular values
            try:
                sv = torch.linalg.svdvals(param.view(param.shape[0], -1))
                kappa = sv[0] / sv[-1]
                if kappa > threshold:
                    ill_conditioned.append((name, kappa.item()))
            except Exception:
                pass
    return ill_conditioned
```

---

## Appendix T: Quick Derivations

### T.1 Why Machine Epsilon Is $2^{1-p}$

A floating-point number with precision $p$ bits (significand) represents values of the form:
$$x = \pm 1.b_1 b_2 \cdots b_{p-1} \times 2^e$$

The spacing between consecutive representable numbers near $1.0$ is:
$$2^0 \times 2^{-(p-1)} = 2^{1-p}$$

This is the **unit in the last place (ULP)** at $1.0$, which equals machine epsilon $\varepsilon_{\text{mach}}$.

For fp32: $p = 24$ (1 implicit + 23 explicit mantissa bits), so $\varepsilon_{\text{mach}} = 2^{1-24} = 2^{-23} \approx 1.19 \times 10^{-7}$.

### T.2 Why Kahan Summation Has $O(\varepsilon_{\text{mach}})$ Error

Naive summation of $n$ numbers has error bound $O(n \varepsilon_{\text{mach}} \sum |x_i|)$ — grows with $n$.

Kahan summation maintains a compensation variable $c$ tracking the lost low-order bits:
```
c = 0
for each x_i:
    y = x_i - c          # Corrected input
    t = sum + y          # sum is large, y small
    c = (t - sum) - y    # Algebraically zero, but captures rounding error
    sum = t
```

The compensation $c$ captures the bits lost in `sum + y` and feeds them into the next iteration. The net error is $O(\varepsilon_{\text{mach}} \sum |x_i|)$, independent of $n$.

### T.3 Why Softmax Is Numerically Stable With Shifting

For $z_i \in \mathbb{R}$, softmax satisfies translation invariance:
$$\text{softmax}(\mathbf{z})_i = \text{softmax}(\mathbf{z} - c\mathbf{1})_i \quad \forall c$$

**Proof:**
$$\frac{e^{z_i - c}}{\sum_j e^{z_j - c}} = \frac{e^{z_i} \cdot e^{-c}}{\sum_j e^{z_j} \cdot e^{-c}} = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

Setting $c = \max_j z_j$ ensures all exponents $z_i - c \leq 0$, preventing overflow while preserving the mathematical value.

### T.4 Relative Backward Stability of Householder QR

The Householder QR algorithm produces $\hat{Q}, \hat{R}$ such that:
$$\hat{Q}\hat{R} = A + \delta A, \quad \frac{\|\delta A\|}{\|A\|} = O(\varepsilon_{\text{mach}})$$

This is backward stable: the computed result is the exact answer for a slightly perturbed problem. The forward error in solving $Ax = b$ via QR is then $O(\kappa(A) \varepsilon_{\text{mach}})$, which is optimal — no algorithm can do better without additional information about the problem structure.

---

## Appendix U: Cross-Reference with Other Sections

This section connects to several other parts of the curriculum:

| Topic Introduced Here | Full Treatment |
|----------------------|----------------|
| Condition number $\kappa(A)$ | [§02 Numerical Linear Algebra](../02-Numerical-Linear-Algebra/notes.md) — condition number theory, backward stability proofs, perturbation analysis |
| Stable matrix algorithms (LU, QR, Cholesky) | [§03-Advanced-Linear-Algebra/08-Matrix-Decompositions](../../03-Advanced-Linear-Algebra/08-Matrix-Decompositions/notes.md) — full decomposition theory |
| Floating-point in optimization (gradient descent) | [§03 Numerical Optimization](../03-Numerical-Optimization/notes.md) — learning rate selection, gradient accumulation, precision effects on convergence |
| Interpolation with finite precision | [§04 Interpolation and Approximation](../04-Interpolation-and-Approximation/notes.md) — Runge's phenomenon, Chebyshev nodes, numerical stability of polynomial evaluation |
| Numerical quadrature errors | [§05 Numerical Integration](../05-Numerical-Integration/notes.md) — error analysis of quadrature rules in finite precision |
| Probabilistic error analysis | [§05-Probability-and-Statistics](../../05-Probability-and-Statistics/notes.md) — probabilistic numerics, Gaussian process approximations |

**Notation cross-references:**
- $\varepsilon_{\text{mach}}$ defined here → used in all §10 sections
- $\kappa(A)$ introduced here → defined formally in §02
- $\text{fl}(\cdot)$ rounding operator defined here → used throughout §10


---

## Appendix V: Extended Exercises with Hints

These supplementary problems extend the main exercises for students who want deeper practice.

### V.1 Sterbenz's Lemma (★★)

**Statement:** If $a, b \in \mathbb{F}$ and $b/2 \leq a \leq 2b$, then $a - b$ is computed exactly (no rounding error).

**Proof sketch:** When $a \approx b$, the result $a - b$ is small. Since both $a$ and $b$ are exactly representable, and their exponents differ by at most 1, the subtraction can be performed in the significand without any rounding.

**Application:** In Kahan summation, the step `c = (t - sum) - y` uses Sterbenz's lemma to capture rounding errors exactly when the values have similar magnitudes.

**Exercise:** Verify Sterbenz's lemma numerically for fp32 arithmetic. Generate pairs $(a, b)$ where $b/2 \leq a \leq 2b$ and confirm that `a - b` equals `(a - b)` computed via Fraction (exact rational arithmetic). Compare with pairs outside the range.

### V.2 The Table Maker's Dilemma (★★★)

When evaluating $\sin(x)$ to correctly-rounded results, we need to know $\sin(x)$ to much higher precision — potentially arbitrary precision. This is because the correctly-rounded result depends on which side of a representable number $\sin(x)$ falls on. For a $p$-bit significand, the worst case requires computing $\sin(x)$ to approximately $p^2$ bits of precision (the **Table Maker's Dilemma**, Kahan and Ziv, 1990s).

**Modern resolution:** The CRlibm library (INRIA) and Intel SVML use multi-stage argument reduction + polynomial approximation with double-double arithmetic to guarantee correctly-rounded elementary functions.

**For AI:** This is why `torch.sin()` may give slightly different results than `math.sin()` for the same input — different levels of accuracy guarantees.

### V.3 Floating-Point Reproducibility (★★)

Training a deep neural network on the same hardware, same code, same random seed should produce the same result. But in practice, results often differ between runs. Sources of non-reproducibility:
- **Non-associative reductions:** GPU CUDA reduction operations may sum in different orders depending on thread scheduling
- **Atomic operations:** Multi-threaded gradient accumulation uses atomics, which arrive in non-deterministic order
- **cuDNN algorithm selection:** cuDNN may select different convolution algorithms with different rounding behaviors

**PyTorch reproducibility settings:**
```python
import torch
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True  # Slower but deterministic
torch.backends.cudnn.benchmark = False     # Disable algorithm search
```

**Tradeoff:** Deterministic mode can be 10-30% slower due to avoiding certain non-deterministic fast paths.

### V.4 Error-Free Transformations (★★★)

An **error-free transformation** (EFT) computes not just $\text{fl}(a \circ b)$ but also the exact rounding error $e$ such that $a \circ b = \text{fl}(a \circ b) + e$ exactly.

**TwoSum algorithm (Knuth, 1969):**
```
function TwoSum(a, b):
    s = fl(a + b)
    v = fl(s - a)
    e = fl((a - fl(s - v)) + fl(b - v))
    return (s, e)  # a + b = s + e exactly
```

This costs 6 floating-point operations to compute both the sum and the exact rounding error.

**Application — Double-Double Arithmetic:** By representing each number as a pair $(h, l)$ where $h + l$ is the true value (and $|l| \leq \varepsilon_{\text{mach}}|h|/2$), we effectively double the working precision from $p$ to $\sim 2p$ bits using only hardware arithmetic. This is how `long double` is emulated on some platforms and how Kahan summation achieves its precision.

---

## Appendix W: Format Selection Guide for AI Practitioners

### Decision Tree for Format Selection

```
CHOOSING FLOATING-POINT FORMAT
════════════════════════════════════════════════════════════════════════

  Is this inference or training?
  │
  ├─► INFERENCE
  │     Is accuracy critical?
  │     ├─► YES: fp32 (or bf16 if model supports it)
  │     └─► NO: int8 quantization (post-training quant, GPTQ, AWQ)
  │                → 4-8x memory reduction, ~1-2% accuracy loss
  │
  └─► TRAINING
        Hardware with Tensor Cores? (A100/H100/RTX 30xx/40xx)
        ├─► YES: bf16 + AMP (torch.cuda.amp)
        │         - Weight master copy in fp32
        │         - Compute in bf16 (4x throughput)
        │         - Dynamic loss scaling
        └─► NO (V100 or CPU):
              fp32 safe baseline
              fp16 + AMP if V100 (needs loss scaling, unstable for LLMs)

════════════════════════════════════════════════════════════════════════
```

### Format Comparison Summary

| Format | Bits | Dynamic Range | Precision | Best Use Case |
|--------|------|---------------|-----------|---------------|
| fp64 | 64 | ±10^308 | ~15 digits | Scientific computing, debugging |
| fp32 | 32 | ±10^38 | ~7 digits | Training baseline, optimizer states |
| tf32 | 19* | ±10^38 | ~3 digits | A100 matmul (automatic, transparent) |
| bf16 | 16 | ±10^38 | ~2 digits | LLM training (same range as fp32) |
| fp16 | 16 | ±65504 | ~3 digits | Vision models, inference |
| fp8 E4M3 | 8 | ±448 | ~1 digit | Forward pass (H100+) |
| fp8 E5M2 | 8 | ±57344 | <1 digit | Gradient accumulation (H100+) |
| int8 | 8 | ±127 | Integer | Inference quantization |

*tf32 uses 10-bit mantissa for compute but stores as fp32.

### Choosing Loss Scale Initial Value

```python
# Typical AMP GradScaler configuration
scaler = torch.cuda.amp.GradScaler(
    init_scale=2**16,      # Start with 65536 — large enough for fp16
    growth_factor=2.0,      # Double scale every growth_interval steps
    backoff_factor=0.5,     # Halve scale on overflow
    growth_interval=2000,   # Steps between scale increases
    enabled=True
)
```

**Heuristic:** Start with `init_scale = 2^16` for most models. If you see frequent overflow warnings in the first 100 steps, reduce to `2^12`. If no overflow for 10,000+ steps, the scaler will auto-increase.

---

*End of Appendix W*

[← Back to Chapter 10](../README.md) | [Next: Numerical Linear Algebra →](../02-Numerical-Linear-Algebra/notes.md)
