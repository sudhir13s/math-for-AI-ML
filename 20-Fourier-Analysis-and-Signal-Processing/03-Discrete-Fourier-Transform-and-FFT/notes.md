[<- Back to Fourier Analysis](../README.md) | [Previous: Fourier Transform <-](../02-Fourier-Transform/notes.md) | [Next: Convolution Theorem ->](../04-Convolution-Theorem/notes.md)

---

# Discrete Fourier Transform and FFT

> _"The FFT is the most important numerical algorithm of our lifetime."_
> - Gilbert Strang, MIT Mathematics

## Overview

The Discrete Fourier Transform (DFT) is the numerically computable version of the Fourier Transform: given $N$ equally-spaced samples of a signal, it produces $N$ complex frequency coefficients that describe the signal's spectral content. The DFT is not merely an approximation to the continuous Fourier Transform - it is an exact, invertible change of basis on $\mathbb{C}^N$, with its own complete theory of properties, convergence, and error.

What makes the DFT indispensable is the **Fast Fourier Transform (FFT)**: the Cooley-Tukey algorithm (1965) reduces the $O(N^2)$ matrix-vector product defining the DFT to $O(N \log N)$ arithmetic operations through a recursive divide-and-conquer strategy. For $N = 2^{20} \approx 10^6$, this is the difference between $10^{12}$ and $2 \times 10^7$ operations - five orders of magnitude. The FFT is the engine behind digital audio, MRI reconstruction, radar, telecommunications, JPEG compression, and - increasingly - large language models, neural operators, and efficient sequence modeling architectures.

This section develops the DFT rigorously from first principles, derives the Cooley-Tukey algorithm and its $O(N \log N)$ complexity, treats spectral leakage and windowing in depth, introduces the Short-Time Fourier Transform (STFT) for time-varying signals, and traces the FFT through modern AI systems: Whisper's mel spectrogram pipeline, the Fourier Neural Operator, Monarch matrices, and FlashFFTConv.

## Prerequisites

- **Fourier Transform** - continuous FT definition ($\xi$-convention), Plancherel's theorem, uncertainty principle, Poisson summation formula ([Section 20-02](../02-Fourier-Transform/notes.md))
- **Fourier Series** - complex Fourier coefficients, orthonormality of complex exponentials ([Section 20-01](../01-Fourier-Series/notes.md))
- **Complex numbers** - $e^{i\theta} = \cos\theta + i\sin\theta$, roots of unity, complex modulus and argument ([Section 01](../../01-Mathematical-Foundations/README.md))
- **Linear algebra** - matrix-vector products, unitary matrices, change of basis, inner products ([Section 02](../../02-Linear-Algebra-Basics/README.md))
- **Sampling theory** - Nyquist-Shannon theorem (introduced here; connection to Section 02-7.5 Poisson summation)

## Companion Notebooks

| Notebook | Description |
|----------|-------------|
| [theory.ipynb](theory.ipynb) | DFT matrix, FFT from scratch, butterfly diagrams, window functions, STFT spectrograms, Whisper mel pipeline, FNO spectral layer |
| [exercises.ipynb](exercises.ipynb) | 10 graded problems: DFT by hand through Monarch butterfly factorization |

## Learning Objectives

After completing this section, you will:

1. Define the $N$-point DFT and compute it by hand for small $N$ using roots of unity
2. Express the DFT as a unitary matrix transformation and verify orthonormality of the Fourier basis vectors
3. State and prove the Cooley-Tukey radix-2 FFT recurrence and derive the $O(N\log_2 N)$ complexity
4. Draw and interpret a butterfly diagram for an 8-point DFT
5. Apply all standard DFT properties: linearity, circular shift, modulation, conjugate symmetry, Parseval
6. Explain spectral leakage and quantify the tradeoff between main-lobe width and side-lobe level for Hann, Hamming, Blackman, and Kaiser windows
7. Set up a correct FFT-based spectral analysis: choose $N$, build the frequency axis, apply fftshift, handle negative frequencies
8. Define the STFT and describe the uncertainty tradeoff between time and frequency resolution
9. Implement the mel-spectrogram pipeline used by OpenAI Whisper (Radford et al., 2022)
10. Describe the Fourier Neural Operator (Li et al., 2021) and implement its truncated spectral convolution layer
11. Explain Monarch matrices (Dao et al., 2022) as a product-of-sparse-butterfly-matrices parameterization of FFT-like transforms
12. Identify and correct the eight most common DFT errors: index conventions, frequency axis misuse, leakage misdiagnosis, normalization ambiguity

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 From Continuous to Discrete: Sampling a Spectrum](#11-from-continuous-to-discrete-sampling-a-spectrum)
  - [1.2 What the DFT Does Geometrically](#12-what-the-dft-does-geometrically)
  - [1.3 Why O(N log N) Changed the World](#13-why-on-log-n-changed-the-world)
  - [1.4 Historical Timeline](#14-historical-timeline)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 The N-Point DFT](#21-the-n-point-dft)
  - [2.2 Inverse DFT and Normalization Conventions](#22-inverse-dft-and-normalization-conventions)
  - [2.3 The DFT Matrix](#23-the-dft-matrix)
  - [2.4 DFT as a Change of Basis](#24-dft-as-a-change-of-basis)
  - [2.5 Relation to the Continuous Fourier Transform](#25-relation-to-the-continuous-fourier-transform)
- [3. Properties of the DFT](#3-properties-of-the-dft)
  - [3.1 Linearity](#31-linearity)
  - [3.2 Circular Shift](#32-circular-shift)
  - [3.3 Modulation (Frequency Shift)](#33-modulation-frequency-shift)
  - [3.4 Conjugate Symmetry for Real Inputs](#34-conjugate-symmetry-for-real-inputs)
  - [3.5 Parseval's Theorem for the DFT](#35-parsevals-theorem-for-the-dft)
  - [3.6 Circular Convolution - Preview](#36-circular-convolution--preview)
  - [3.7 DFT Properties Master Table](#37-dft-properties-master-table)
- [4. The Fast Fourier Transform Algorithm](#4-the-fast-fourier-transform-algorithm)
  - [4.1 Complexity of Naive DFT](#41-complexity-of-naive-dft)
  - [4.2 The Cooley-Tukey Insight: Divide and Conquer](#42-the-cooley-tukey-insight-divide-and-conquer)
  - [4.3 Butterfly Diagram](#43-butterfly-diagram)
  - [4.4 Decimation in Time vs Decimation in Frequency](#44-decimation-in-time-vs-decimation-in-frequency)
  - [4.5 Bit-Reversal Permutation](#45-bit-reversal-permutation)
  - [4.6 Complexity Analysis: O(N log N)](#46-complexity-analysis-on-log-n)
  - [4.7 Variants and Extensions](#47-variants-and-extensions)
- [5. Spectral Leakage and Windowing](#5-spectral-leakage-and-windowing)
  - [5.1 The Leakage Problem](#51-the-leakage-problem)
  - [5.2 Window Functions](#52-window-functions)
  - [5.3 The Main-Lobe/Side-Lobe Tradeoff](#53-the-main-lobeside-lobe-tradeoff)
  - [5.4 Scalloping Loss and the Picket Fence Effect](#54-scalloping-loss-and-the-picket-fence-effect)
  - [5.5 Overlap-Add and Overlap-Save](#55-overlap-add-and-overlap-save)
- [6. Frequency Resolution and Zero-Padding](#6-frequency-resolution-and-zero-padding)
  - [6.1 Frequency Resolution](#61-frequency-resolution)
  - [6.2 Zero-Padding](#62-zero-padding)
  - [6.3 The Nyquist-Shannon Sampling Theorem](#63-the-nyquist-shannon-sampling-theorem)
  - [6.4 Practical FFT Setup Checklist](#64-practical-fft-setup-checklist)
- [7. Short-Time Fourier Transform](#7-short-time-fourier-transform)
  - [7.1 Motivation: Time-Varying Spectra](#71-motivation-time-varying-spectra)
  - [7.2 STFT Definition](#72-stft-definition)
  - [7.3 The Spectrogram](#73-the-spectrogram)
  - [7.4 Uncertainty Principle for STFT](#74-uncertainty-principle-for-stft)
  - [7.5 Synthesis and Perfect Reconstruction](#75-synthesis-and-perfect-reconstruction)
- [8. Applications in Machine Learning](#8-applications-in-machine-learning)
  - [8.1 Whisper: FFT Pipeline for Speech Recognition](#81-whisper-fft-pipeline-for-speech-recognition)
  - [8.2 Fourier Neural Operator (FNO)](#82-fourier-neural-operator-fno)
  - [8.3 Monarch Matrices and Structured FFT](#83-monarch-matrices-and-structured-fft)
  - [8.4 FlashFFTConv and Long-Range Convolutions](#84-flashfftconv-and-long-range-convolutions)
  - [8.5 Spectral Methods in Graph Neural Networks](#85-spectral-methods-in-graph-neural-networks)
- [9. Common Mistakes](#9-common-mistakes)
- [10. Exercises](#10-exercises)
- [11. Why This Matters for AI (2026 Perspective)](#11-why-this-matters-for-ai-2026-perspective)
- [12. Conceptual Bridge](#12-conceptual-bridge)

---

## 1. Intuition

### 1.1 From Continuous to Discrete: Sampling a Spectrum

Every practical computation with signals must at some point confront a fundamental constraint: computers cannot store or process continuous functions. What they can store is a finite sequence of numbers - samples of a signal taken at discrete time instants. The passage from the continuous world of the Fourier Transform to the discrete world of the DFT is therefore not a mathematical nicety but a computational necessity.

Suppose we record $N$ samples of a signal $x(t)$ at a sampling rate of $f_s$ samples per second:

$$x[n] = x(n \cdot T_s), \quad T_s = \frac{1}{f_s}, \quad n = 0, 1, \ldots, N-1$$

The total observation time is $T = N \cdot T_s$. What frequencies can we see in this finite record? Two constraints emerge immediately:

1. **Highest observable frequency** (Nyquist): any frequency above $f_s/2$ will be indistinguishable from a lower frequency due to aliasing - two distinct sinusoids will produce identical sample sequences.
2. **Frequency resolution**: the smallest frequency difference we can distinguish is $\Delta f = f_s / N = 1/T$ - the reciprocal of the observation duration.

The DFT honours both constraints. It examines the signal at exactly $N$ equally-spaced frequency bins:

$$\xi_k = \frac{k \cdot f_s}{N}, \quad k = 0, 1, \ldots, N-1$$

covering one full period $[0, f_s)$ of the periodic spectrum. The first $N/2$ bins correspond to positive frequencies $[0, f_s/2)$; the remaining $N/2$ bins correspond to negative frequencies $(-f_s/2, 0)$ wrapped around to $[f_s/2, f_s)$. This wrapping is the discrete manifestation of the Nyquist limit.

**For AI:** Every neural network that processes time-series data - speech, ECG, seismic, financial - ultimately works with sampled signals. Understanding the frequency axis setup of the DFT is essential for interpreting what spectral features (mel filterbanks in Whisper, power spectral density in EEG classifiers) actually represent. Misinterpreting negative-frequency bins is a common source of subtle bugs in spectrogram preprocessing.

### 1.2 What the DFT Does Geometrically

The space of $N$-point complex sequences $\mathbb{C}^N$ has dimension $N$. The DFT provides a complete orthogonal basis for this space - the **Fourier basis** - consisting of $N$ complex sinusoids at the frequencies $\xi_k = k/N$:

$$\mathbf{f}_k = \begin{bmatrix} 1 \\ \omega_N^k \\ \omega_N^{2k} \\ \vdots \\ \omega_N^{(N-1)k} \end{bmatrix}, \quad \omega_N = e^{2\pi i / N}, \quad k = 0, 1, \ldots, N-1$$

Each basis vector $\mathbf{f}_k$ is a sampled complex sinusoid at frequency $k/N$ cycles per sample. The DFT coefficient $X[k]$ is the inner product of the signal $\mathbf{x}$ with the $k$-th basis vector:

$$X[k] = \langle \mathbf{x}, \mathbf{f}_k \rangle = \sum_{n=0}^{N-1} x[n]\, \overline{\omega_N^{nk}} = \sum_{n=0}^{N-1} x[n]\, \omega_N^{-nk}$$

The DFT is therefore a **change of basis** from the standard basis (time domain) to the Fourier basis (frequency domain). The magnitude $|X[k]|$ measures how strongly the signal oscillates at frequency $k/N$ cycles per sample; the phase $\angle X[k]$ measures the phase of that oscillation.

The inverse DFT reconstructs the signal by superimposing all $N$ sinusoids with their computed amplitudes and phases:

$$x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k]\, \omega_N^{nk}$$

The factor $1/N$ comes from the normalization of the basis - the $\mathbf{f}_k$ have squared norm $\lVert \mathbf{f}_k \rVert^2 = N$, not 1. Using the orthonormal basis $\mathbf{f}_k / \sqrt{N}$, both forward and inverse transforms carry a $1/\sqrt{N}$ factor. Different software packages use different conventions; always check.

**Geometric picture:** if you think of the signal as a vector in $\mathbb{C}^N$, the DFT rotates this vector into a new coordinate system. The new coordinate axes are complex sinusoids. The DFT matrix $F_N$ is the rotation matrix (more precisely, a scaled unitary matrix). The operation $\mathbf{X} = F_N \mathbf{x}$ is a matrix-vector product; naive evaluation takes $O(N^2)$ operations. The FFT computes this same product in $O(N \log N)$.

### 1.3 Why O(N log N) Changed the World

To appreciate the scale of the FFT's impact, consider the arithmetic:

| $N$ | Naive DFT ($N^2$) | FFT ($N \log_2 N$) | Speedup |
|-----|------------------|---------------------|---------|
| 64 | 4,096 | 384 | 11 |
| 1,024 | 1,048,576 | 10,240 | 102 |
| 65,536 | $4.3 \times 10^9$ | $1.0 \times 10^6$ | 4,369 |
| $10^6$ | $10^{12}$ | $2.0 \times 10^7$ | 50,000 |

Before the FFT, computing the spectrum of a 1-second audio clip sampled at 44.1 kHz was infeasible in real time. After the FFT, it takes microseconds. The entire modern infrastructure of digital audio, wireless communications (OFDM), radar, MRI, seismology, and antenna design rests on this algorithmic breakthrough.

The secret is that the DFT matrix $F_N$ has enormous **structure**: it is a Vandermonde matrix with entries that are powers of $N$-th roots of unity, and these entries satisfy recursive relationships that allow massive cancellation of work through a divide-and-conquer decomposition.

**For AI (2026):** Long-context language models face a quadratic bottleneck in self-attention: processing a 100K-token context with full attention requires $\sim 10^{10}$ operations. The FFT offers an alternative: if a sequence operation can be formulated as a convolution, it can be computed in $O(N \log N)$ using the FFT. This is the foundation of FNet (Lee-Thorp et al., 2022), FlashFFTConv (Dao et al., 2023), S4 (Gu et al., 2022), and the Fourier Neural Operator (Li et al., 2021). The $O(N \log N)$ vs $O(N^2)$ complexity gap, already dramatic at $N = 10^3$, becomes existential at $N = 10^5$.

### 1.4 Historical Timeline

The story of the FFT is a case study in mathematical rediscovery and the transformative effect of the right algorithm at the right moment.

**1805 - Carl Friedrich Gauss** derived what we now recognize as the FFT algorithm while computing asteroid orbits. His manuscript lay unpublished and largely unknown.

**1903 - Carl Runge** independently derived a related fast algorithm for trigonometric interpolation.

**1942 - Danielson and Lanczos** published a recursive factorization of the DFT based on even-odd splitting, establishing the core mathematical structure.

**1965 - James Cooley and John Tukey** published "An Algorithm for the Machine Calculation of Complex Fourier Series," the paper that launched the digital signal processing revolution. Their algorithm was motivated by Tukey's desire to detect Soviet nuclear tests through seismic monitoring. The paper, just two pages, is one of the most cited in the history of applied mathematics.

**1966 - Gentleman and Sande** introduced the "decimation in frequency" variant; Bergland gave a comprehensive tutorial; the algorithm became standard in every engineering discipline within a decade.

**1984 - FFTW** - Frigo and Johnson developed FFTW ("Fastest Fourier Transform in the West"), an adaptive library that benchmarks and selects the optimal algorithm for any $N$ and hardware. FFTW ships in NumPy, SciPy, and is the backbone of essentially all scientific computing FFT pipelines.

**2021 - Fourier Neural Operator** (Li et al.) applies the DFT inside a neural network layer to learn solution operators for PDEs in $O(N \log N)$.

**2022 - Monarch matrices** (Dao et al.) parameterize FFT-like structured transforms as products of sparse butterfly matrices, enabling learnable near-FFT computations in transformer architectures.

---

## 2. Formal Definitions

### 2.1 The N-Point DFT

**Definition (DFT).** Let $\mathbf{x} = (x[0], x[1], \ldots, x[N-1]) \in \mathbb{C}^N$ be a finite sequence. The **$N$-point Discrete Fourier Transform** of $\mathbf{x}$ is the sequence $\mathbf{X} = (X[0], X[1], \ldots, X[N-1]) \in \mathbb{C}^N$ defined by:

$$X[k] = \sum_{n=0}^{N-1} x[n]\, \omega_N^{-nk}, \quad k = 0, 1, \ldots, N-1$$

where $\omega_N = e^{2\pi i / N}$ is the **primitive $N$-th root of unity**.

The complex number $\omega_N^{-nk} = e^{-2\pi i nk/N}$ is a sampled complex sinusoid at frequency $k/N$ cycles per sample, evaluated at time $n$. The DFT coefficient $X[k]$ measures the **complex amplitude** (magnitude and phase) of the frequency-$k$ component of $\mathbf{x}$.

**Sign convention.** The minus sign in $\omega_N^{-nk}$ is the physics/engineering convention (analysis kernel $e^{-2\pi i}$). Some mathematical texts use $\omega_N^{+nk}$ (positive sign) for the forward transform; this merely relabels forward and inverse. We follow NumPy/SciPy: forward DFT has the $-$ sign.

**Roots of unity.** The $N$ values $\{\omega_N^0, \omega_N^1, \ldots, \omega_N^{N-1}\}$ are the $N$-th roots of unity - equally spaced points on the unit circle in $\mathbb{C}$, starting at 1 and rotating counterclockwise by $2\pi/N$ at each step. They satisfy:

$$\sum_{n=0}^{N-1} \omega_N^{nk} = \begin{cases} N & \text{if } k \equiv 0 \pmod{N} \\ 0 & \text{otherwise} \end{cases}$$

This **orthogonality of roots of unity** is the key identity underlying all DFT properties and the inversion formula.

**Standard examples for small $N$:**

*$N=2$:* $\omega_2 = e^{i\pi} = -1$. The 2-point DFT is:
$$X[0] = x[0] + x[1], \quad X[1] = x[0] - x[1]$$
This is the **Hadamard/Walsh transform** in 2D - just a sum and difference.

*$N=4$:* $\omega_4 = e^{i\pi/2} = i$. The 4-point DFT:
$$X[0] = x[0]+x[1]+x[2]+x[3], \quad X[1] = x[0] - ix[1] - x[2] + ix[3]$$
$$X[2] = x[0]-x[1]+x[2]-x[3], \quad X[3] = x[0]+ix[1]-x[2]-ix[3]$$

*Non-example:* An infinite sequence $x[n]$ for $n \in \mathbb{Z}$ does NOT have a DFT - you must first window or truncate it to $N$ points. Applying the DFT directly to an infinite sequence is a category error; what you would compute is the $z$-transform or DTFT (Discrete-Time Fourier Transform), which is distinct from the DFT.

### 2.2 Inverse DFT and Normalization Conventions

**Definition (IDFT).** The **Inverse Discrete Fourier Transform** is:

$$x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k]\, \omega_N^{nk}, \quad n = 0, 1, \ldots, N-1$$

**Verification.** Substitute the DFT into the IDFT:

$$\frac{1}{N}\sum_k X[k]\omega_N^{nk} = \frac{1}{N}\sum_k \left(\sum_m x[m]\omega_N^{-mk}\right)\omega_N^{nk} = \sum_m x[m] \cdot \frac{1}{N}\sum_k \omega_N^{(n-m)k} = \sum_m x[m]\, \delta[n-m] = x[n]$$

using the orthogonality of roots of unity in the penultimate step.

**Three normalization conventions** (all valid, software-dependent):

| Convention | Forward DFT | Inverse DFT | Used by |
|---|---|---|---|
| Engineering (default) | $X[k] = \sum_n x[n]\omega_N^{-nk}$ | $x[n] = \frac{1}{N}\sum_k X[k]\omega_N^{nk}$ | NumPy, SciPy, MATLAB |
| Symmetric (unitary) | $X[k] = \frac{1}{\sqrt{N}}\sum_n x[n]\omega_N^{-nk}$ | $x[n] = \frac{1}{\sqrt{N}}\sum_k X[k]\omega_N^{nk}$ | Some textbooks |
| Inverse-normalized | $X[k] = \frac{1}{N}\sum_n x[n]\omega_N^{-nk}$ | $x[n] = \sum_k X[k]\omega_N^{nk}$ | Rare; avoid |

**For AI:** Always check which convention a library uses. NumPy uses engineering convention: `np.fft.fft` has no $1/N$ factor, `np.fft.ifft` has $1/N$. PyTorch's `torch.fft.fft` matches NumPy. When mixing libraries or implementing from scratch, convention mismatch is a common silent bug.

### 2.3 The DFT Matrix

The DFT is a linear map $\mathbf{X} = F_N \mathbf{x}$, where the **DFT matrix** $F_N \in \mathbb{C}^{N \times N}$ has entries:

$$(F_N)_{kn} = \omega_N^{-kn}, \quad k, n = 0, 1, \ldots, N-1$$

The matrix is fully specified by its single generator $\omega_N$. Written out explicitly for $N=4$:

$$F_4 = \begin{bmatrix} 1 & 1 & 1 & 1 \\ 1 & -i & -1 & i \\ 1 & -1 & 1 & -1 \\ 1 & i & -1 & -i \end{bmatrix}$$

**Unitarity.** The DFT matrix satisfies $F_N F_N^* = N I_N$, where $F_N^*$ is the conjugate transpose. Equivalently, the **normalized DFT matrix** $\frac{1}{\sqrt{N}}F_N$ is unitary:

$$\frac{1}{\sqrt{N}}F_N \cdot \left(\frac{1}{\sqrt{N}}F_N\right)^* = I_N$$

**Proof.** The $(k, l)$ entry of $F_N F_N^*$ is:

$$(F_N F_N^*)_{kl} = \sum_{n=0}^{N-1} \omega_N^{-kn} \overline{\omega_N^{-ln}} = \sum_{n=0}^{N-1} \omega_N^{(l-k)n} = \begin{cases} N & \text{if } k=l \\ 0 & \text{if } k \neq l \end{cases}$$

by the orthogonality of roots of unity. Therefore $F_N F_N^* = NI$, so $F_N^{-1} = \frac{1}{N}F_N^*$, confirming the IDFT formula: $\mathbf{x} = \frac{1}{N}F_N^* \mathbf{X}$.

**Key structural properties of $F_N$:**

- **Vandermonde structure**: $F_N$ is a Vandermonde matrix with nodes $\omega_N^0, \omega_N^{-1}, \ldots, \omega_N^{-(N-1)}$
- **Symmetric**: $(F_N)_{kn} = (F_N)_{nk}$ - the matrix is symmetric (not Hermitian)
- **Periodic indexing**: all indices are taken modulo $N$, making the transform naturally circular

### 2.4 DFT as a Change of Basis

The columns of $F_N^*$ (equivalently, the rows of $F_N$ conjugated) are the Fourier basis vectors:

$$\mathbf{f}_k = \frac{1}{\sqrt{N}}\begin{bmatrix} 1 \\ \omega_N^k \\ \omega_N^{2k} \\ \vdots \\ \omega_N^{(N-1)k} \end{bmatrix} \in \mathbb{C}^N, \quad k = 0, 1, \ldots, N-1$$

These form an **orthonormal basis** for $\mathbb{C}^N$: $\langle \mathbf{f}_k, \mathbf{f}_l \rangle = \delta_{kl}$.

The DFT coefficient $X[k]/\sqrt{N}$ is the coordinate of $\mathbf{x}$ in the direction $\mathbf{f}_k$:

$$\frac{X[k]}{\sqrt{N}} = \left\langle \mathbf{x}, \mathbf{f}_k \right\rangle = \frac{1}{\sqrt{N}}\sum_{n=0}^{N-1} x[n]\,\omega_N^{-nk}$$

**Fourier basis as "frequency detector" vectors:** $\mathbf{f}_k$ oscillates at exactly $k$ cycles across the $N$ samples. If $\mathbf{x}$ is itself a pure sinusoid at frequency $k_0$, then $|X[k]|$ is zero for all $k \neq k_0$ and equals $N$ at $k = k_0$. The DFT "detects" which frequencies are present by computing inner products with all $N$ basis sinusoids simultaneously.

**Contrast with DTFT (Discrete-Time Fourier Transform):** The DTFT is defined for infinite sequences as $X(e^{j\omega}) = \sum_{n=-\infty}^\infty x[n]e^{-j\omega n}$ and produces a continuous spectrum over $\omega \in [0, 2\pi)$. The DFT is a sampled version of the DTFT, evaluating it at $N$ equally-spaced frequencies. The DFT is the only version that is both finite and exactly invertible.

### 2.5 Relation to the Continuous Fourier Transform

The DFT approximates the continuous Fourier Transform of a sampled signal. Specifically, if $x(t)$ is bandlimited to $[-f_s/2, f_s/2)$ and sampled at rate $f_s$:

$$X[k] \approx \hat{x}\!\left(\frac{k}{N T_s}\right) \cdot \frac{1}{T_s} \cdot \frac{1}{N} = \frac{1}{T_s}\hat{x}(k\,\Delta f)$$

where $\Delta f = f_s/N$ is the frequency bin spacing. The DFT thus provides a discretized snapshot of the continuous spectrum at $N$ frequency points.

Three key approximation errors arise:

1. **Aliasing**: frequencies above $f_s/2$ fold back into $[0, f_s/2)$ - see Section 6.3
2. **Leakage**: the signal is observed for only $N$ samples, equivalent to multiplying by a rectangular window, causing sidelobe contamination in the spectrum - see Section 5
3. **Picket fence**: the DFT only evaluates the spectrum at $N$ discrete frequencies, potentially missing peaks between bins - see Section 6.2

---

## 3. Properties of the DFT

All DFT properties follow from linearity of the sum and the orthogonality of roots of unity. We state each property, prove it, and note its discrete-specific character where it differs from the continuous FT.

### 3.1 Linearity

$$\mathcal{F}\{ax[n] + by[n]\}[k] = a\,X[k] + b\,Y[k]$$

**Proof.** Immediate from linearity of summation: $\sum_n (ax[n]+by[n])\omega_N^{-nk} = a\sum_n x[n]\omega_N^{-nk} + b\sum_n y[n]\omega_N^{-nk}$.

Linearity makes the DFT a linear operator on $\mathbb{C}^N$, represented by the matrix $F_N$.

### 3.2 Circular Shift

If $y[n] = x[n - m \bmod N]$ (shift $\mathbf{x}$ right by $m$ positions with wrap-around), then:

$$Y[k] = \omega_N^{-mk}\, X[k]$$

**Proof.**
$$Y[k] = \sum_{n=0}^{N-1} x[n-m]\,\omega_N^{-nk} \overset{l=n-m}{=} \sum_{l=0}^{N-1} x[l]\,\omega_N^{-(l+m)k} = \omega_N^{-mk} \sum_{l=0}^{N-1} x[l]\,\omega_N^{-lk} = \omega_N^{-mk} X[k]$$

**Important distinction from continuous case.** In the continuous FT, a time shift by $\tau$ introduces the factor $e^{-2\pi i\xi\tau}$. In the DFT, the shift is **circular** (modular): shifting right pushes the last sample to the first position. If you apply a linear (non-circular) shift to a finite-length signal, the result is NOT simply the DFT multiplied by a phase factor - truncation effects occur. This is a frequent source of confusion when applying the shift property carelessly.

**Application.** In signal processing, circular shift models delay in a circular buffer. In machine learning, circular convolution (Section 3.6) exploits this property to compute convolutions via FFT.

### 3.3 Modulation (Frequency Shift)

If $y[n] = \omega_N^{mn} x[n]$ (multiply by a complex sinusoid), then:

$$Y[k] = X[k - m \bmod N]$$

This is the **dual** of the circular shift property: multiplication in time = shift in frequency.

**Application.** Frequency-shift keying (FSK) in communications. In transformers, rotary position encoding (RoPE) multiplies embeddings by complex exponentials - this is modulation in the DFT sense, shifting the frequency-domain representation of each token by its position.

### 3.4 Conjugate Symmetry for Real Inputs

If $x[n] \in \mathbb{R}$ for all $n$, then:

$$X[N-k] = X[k]^*, \quad k = 0, 1, \ldots, N-1$$

**Proof.** $X[N-k] = \sum_n x[n]\omega_N^{-n(N-k)} = \sum_n x[n]\omega_N^{nk} = \sum_n x[n]\overline{\omega_N^{-nk}} = \overline{X[k]}$, using $x[n] \in \mathbb{R}$ so $x[n] = \overline{x[n]}$ and $\omega_N^{nN} = 1$.

**Consequence.** Only $\lfloor N/2 \rfloor + 1$ DFT coefficients $X[0], X[1], \ldots, X[N/2]$ are independent; the rest are determined by conjugate symmetry. NumPy's `rfft` exploits this to return only the non-redundant half, saving memory and computation. For AI applications working with real-valued signals (audio, time-series), always use `rfft`/`irfft`.

**DC and Nyquist bins:** $X[0] = \sum_n x[n] \in \mathbb{R}$ (always real, proportional to the mean). For even $N$: $X[N/2] = \sum_n x[n](-1)^n \in \mathbb{R}$ (always real, proportional to the alternating sum).

### 3.5 Parseval's Theorem for the DFT

$$\sum_{n=0}^{N-1} |x[n]|^2 = \frac{1}{N} \sum_{k=0}^{N-1} |X[k]|^2$$

**Proof.** Since $\frac{1}{\sqrt{N}}F_N$ is unitary, it preserves the $\ell^2$ norm:

$$\lVert \mathbf{x} \rVert^2 = \left\lVert \frac{1}{\sqrt{N}}F_N \mathbf{x} \right\rVert^2 = \frac{1}{N}\lVert \mathbf{X} \rVert^2$$

**Interpretation.** Energy is perfectly preserved by the DFT - the total energy in the time domain equals the total energy in the frequency domain (up to the $1/N$ normalization factor). This is the discrete analog of Plancherel's theorem from Section 02.

**Application.** The **power spectral density** (PSD) is $|X[k]|^2 / N$, which by Parseval sums to the total signal power. In Whisper's mel spectrogram, the log power $\log(|X[k]|^2)$ of each FFT bin is computed and then summed over mel filterbanks.

### 3.6 Circular Convolution - Preview

The most important DFT property is the **circular convolution theorem**: pointwise multiplication in frequency corresponds to circular convolution in time.

**Definition.** The **circular (cyclic) convolution** of $\mathbf{x}$ and $\mathbf{y}$ in $\mathbb{C}^N$ is:

$$(x \circledast y)[n] = \sum_{m=0}^{N-1} x[m]\, y[n-m \bmod N]$$

**Circular Convolution Theorem.** $\mathcal{F}\{x \circledast y\}[k] = X[k] \cdot Y[k]$.

> **Preview:** This theorem is the entire foundation of FFT-based convolution. The full treatment - including the distinction between circular and linear convolution, overlap-add/save for long signals, filter design, and the connection to CNNs - is the subject of [Section 20-04 Convolution Theorem](../04-Convolution-Theorem/notes.md). Here we note the fact; the applications follow in Section 04.

**Why circular?** The DFT implicitly assumes the signal is **periodic** with period $N$. The product $X[k]Y[k]$ is the DFT of the **circular** (not linear) convolution of $\mathbf{x}$ and $\mathbf{y}$. To compute linear convolution via FFT, zero-pad both sequences to length $N + M - 1$ before taking the DFT (Section 6.2).

### 3.7 DFT Properties Master Table

| Property | Time domain $x[n]$ | Frequency domain $X[k]$ |
|---|---|---|
| Linearity | $ax[n] + by[n]$ | $aX[k] + bY[k]$ |
| Circular shift | $x[n-m \bmod N]$ | $\omega_N^{-mk} X[k]$ |
| Modulation | $\omega_N^{mn} x[n]$ | $X[k-m \bmod N]$ |
| Conjugate symmetry | $x[n] \in \mathbb{R}$ | $X[N-k] = X[k]^*$ |
| Time reversal | $x[-n \bmod N]$ | $X[-k \bmod N] = X[N-k]$ |
| Conjugation | $x[n]^*$ | $X[-k \bmod N]^* = X[N-k]^*$ |
| Circular convolution | $(x \circledast y)[n]$ | $X[k]\cdot Y[k]$ |
| Pointwise product | $x[n] \cdot y[n]$ | $\frac{1}{N}(X \circledast Y)[k]$ |
| Parseval | $\sum_n \lvert x[n] \rvert^2$ | $\frac{1}{N}\sum_k \lvert X[k] \rvert^2$ |
| Duality | $X[n]$ | $N\, x[-k \bmod N]$ |


---

## 4. The Fast Fourier Transform Algorithm

### 4.1 Complexity of Naive DFT

Direct evaluation of $X[k] = \sum_{n=0}^{N-1} x[n]\omega_N^{-nk}$ for all $k = 0, \ldots, N-1$ requires:

- $N$ complex multiplications per output bin (multiplying each $x[n]$ by $\omega_N^{-nk}$)
- $N-1$ complex additions per bin
- Total: $N^2$ complex multiplications, $N(N-1)$ complex additions
- Each complex multiplication = 4 real multiplications + 2 additions

For $N = 1024$: about $10^6$ complex multiplications. For $N = 65536$: about $4 \times 10^9$. At 2010 hardware speeds (~$10^9$ operations/second), a 65536-point DFT takes ~4 seconds. Audio applications need hundreds per second. Naive DFT is simply not usable.

### 4.2 The Cooley-Tukey Insight: Divide and Conquer

Assume $N = 2^m$ (a power of 2). Split the sum into even-indexed and odd-indexed samples:

$$X[k] = \sum_{n=0}^{N-1} x[n]\omega_N^{-nk} = \underbrace{\sum_{r=0}^{N/2-1} x[2r]\,\omega_N^{-2rk}}_{E[k]} + \underbrace{\sum_{r=0}^{N/2-1} x[2r+1]\,\omega_N^{-(2r+1)k}}_{O[k]}$$

Observe that $\omega_N^{-2rk} = e^{-2\pi i(2r)k/N} = e^{-2\pi irk/(N/2)} = \omega_{N/2}^{-rk}$. Therefore:

$$E[k] = \sum_{r=0}^{N/2-1} x[2r]\,\omega_{N/2}^{-rk} = \text{DFT}_{N/2}\!\left(\{x[0], x[2], x[4], \ldots\}\right)[k]$$

$$O[k] = \omega_N^{-k}\sum_{r=0}^{N/2-1} x[2r+1]\,\omega_{N/2}^{-rk} = \omega_N^{-k} \cdot \text{DFT}_{N/2}\!\left(\{x[1], x[3], x[5], \ldots\}\right)[k]$$

Both $E[k]$ and $O[k]$ are $(N/2)$-point DFTs, and both are periodic in $k$ with period $N/2$. For $k = 0, \ldots, N/2-1$:

$$\boxed{X[k] = E[k] + \omega_N^{-k} O[k]}$$
$$\boxed{X[k + N/2] = E[k] - \omega_N^{-k} O[k]}$$

This is the **Cooley-Tukey butterfly**: two $N/2$-point DFTs plus $N/2$ complex multiplications (by the **twiddle factors** $\omega_N^{-k}$) yield the full $N$-point DFT.

**Recursion:** Apply the same split to each $N/2$-point DFT, then to each $N/4$-point DFT, and so on, until we reach 2-point DFTs (which are trivial: $X[0] = x[0]+x[1]$, $X[1] = x[0]-x[1]$).

### 4.3 Butterfly Diagram

The Cooley-Tukey recursion is beautifully represented as a **signal flow graph** called the butterfly diagram. For an 8-point DFT ($N=8$, $\log_2 8 = 3$ stages):

```
DFT-8 BUTTERFLY DIAGRAM (Decimation in Time)
========================================================================

  Input           Stage 1          Stage 2          Stage 3     Output
  (bit-rev)       N/8 DFTs         N/4 DFTs         N/2 DFTs

  x[0] ----+--------------+---------------+-------- X[0]
            |              |               |
  x[4] ----+W0   ...      |               |
            |              |               |
  x[2] ----+              |W0   ...       |
            |              |               |
  x[6] ----+              |               |
            |              |               |
  x[1] ----+              |               |W0
            |              |               |
  x[5] ----+              |               |W1
            |              |               |
  x[3] ----+              |               |W2
            |              |               |
  x[7] ------------------------------------------ X[7]

  Each node: X[k] = E[k] + W^k * O[k]
             X[k+N/2] = E[k] - W^k * O[k]

  W^k = _N^{-k} = e^{-2ik/N}  (twiddle factor)

========================================================================
```

Each **butterfly** is a 2-input, 2-output operation:

```
  a --+------ a + Wb
      |    
      |     
      |        (butterfly crossing)
      |     
      |    
  b --+-- a - Wb
```

For $N = 2^m$, the diagram has $m = \log_2 N$ stages, each containing $N/2$ butterflies. Total butterflies: $\frac{N}{2}\log_2 N$.

### 4.4 Decimation in Time vs Decimation in Frequency

**Decimation in Time (DIT):** Split the input by even/odd indices. The input must be in **bit-reversed order** (see Section 4.5), while the output emerges in natural order. This is the form described above, and the most common in practice.

**Decimation in Frequency (DIF):** Split the output by even/odd frequencies. The input is in natural order, while the output emerges in bit-reversed order. The butterfly computation slightly differs but the complexity is identical.

Both variants require exactly the same number of operations. The choice is often determined by memory access patterns or hardware constraints.

**In-place computation:** The butterfly computation can be performed in-place - the two outputs overwrite the two inputs without needing extra memory. This makes the FFT extremely memory-efficient for large $N$.

### 4.5 Bit-Reversal Permutation

The DIT-FFT requires the input in bit-reversed order. For $N = 8$ (3-bit indices):

| Natural index | Binary | Bit-reversed | Bit-reversed decimal |
|---|---|---|---|
| 0 | 000 | 000 | 0 |
| 1 | 001 | 100 | 4 |
| 2 | 010 | 010 | 2 |
| 3 | 011 | 110 | 6 |
| 4 | 100 | 001 | 1 |
| 5 | 101 | 101 | 5 |
| 6 | 110 | 011 | 3 |
| 7 | 111 | 111 | 7 |

**Why bit-reversal arises:** Each stage of the DIT recursion separates samples by their last bit (even vs odd) - what was the last bit of the original index becomes the first bit at the deepest recursion level. After $\log_2 N$ stages of even-odd splitting, the indices are fully bit-reversed.

**Efficient computation:** Bit-reversal can be computed in $O(N)$ time using integer bit-reversal operations or a simple loop with `bin()` in Python. Hardware FFT chips often include dedicated bit-reversal circuits.

### 4.6 Complexity Analysis: O(N log N)

Let $T(N)$ be the number of complex multiplications required by the FFT. The recursion is:

$$T(N) = 2\,T(N/2) + \frac{N}{2}$$

where $N/2$ twiddle-factor multiplications are needed to combine two $N/2$-point DFTs. With $T(1) = 0$:

**Solving the recurrence:**
$$T(N) = 2T(N/2) + N/2$$
$$= 2\bigl[2T(N/4) + N/4\bigr] + N/2 = 4T(N/4) + N/2 + N/2$$
$$= 4\bigl[2T(N/8) + N/8\bigr] + N = 8T(N/8) + 3N/2$$
$$\vdots$$
$$= N \cdot T(1) + \frac{N}{2}\log_2 N = \frac{N}{2}\log_2 N$$

**Total operations:** $\frac{N}{2}\log_2 N$ complex multiplications and $N\log_2 N$ complex additions. This is $O(N \log N)$.

**Comparison with naive DFT ($N^2$ multiplications):**

| $N$ | FFT / DFT ratio |
|---|---|
| 64 | $3 / 64 \approx 4.7\%$ |
| 1,024 | $5 / 1024 \approx 0.49\%$ |
| $10^6$ | $\approx 0.002\%$ |

For large $N$, the FFT uses less than 0.01% of the operations required by the naive DFT.

**Empirical validation:** The $O(N \log N)$ curve can be confirmed by timing `numpy.fft.fft` for $N = 2^k$, $k = 6, \ldots, 20$ - the log-log plot should have slope $\approx 1$ (since $N \log N \approx N^1$ for moderate $N$, growing only slowly faster than linear).

### 4.7 Variants and Extensions

**Mixed-radix FFT.** The pure radix-2 algorithm requires $N$ to be a power of 2. Real-world signals rarely have power-of-2 lengths. Mixed-radix algorithms factor $N = N_1 \times N_2 \times \cdots \times N_r$ into small prime factors and apply a generalized Cooley-Tukey recursion. With $N = N_1 N_2$, the DFT can be decomposed into $N_1$ DFTs of size $N_2$ and $N_2$ DFTs of size $N_1$. Common radices: 2, 3, 4, 5, 7, 8.

**Split-radix FFT.** Combines radix-2 and radix-4 butterflies to achieve the minimum theoretical operation count: approximately $\frac{1}{3}N\log_2 N$ complex multiplications. The split-radix algorithm is the most operation-efficient known for power-of-2 lengths.

**Prime-length algorithms:**
- **Rader's algorithm** (1968): Converts a prime-length DFT into a circular convolution of composite length, solvable by FFT.
- **Bluestein's algorithm** (1970): Converts any DFT into a convolution via the identity $nk = \binom{n+k}{2} - \binom{n}{2} - \binom{k}{2}$, enabling FFT-based computation for any $N$.

**Real-valued FFT (rfft).** For real inputs, conjugate symmetry (Section 3.4) means only $N/2+1$ output values are independent. `np.fft.rfft` computes only these, roughly halving computation and memory.

**FFTW (Fastest Fourier Transform in the West).** FFTW (Frigo & Johnson, 2005) automatically selects the best algorithm for any $N$ and hardware through a "plan" computed at initialization time. It achieves near-optimal performance across a wide range of $N$ and machine architectures. All major scientific computing libraries (NumPy, SciPy, MATLAB, Julia) use FFTW or FFTW-compatible libraries under the hood.

**For AI:** PyTorch's `torch.fft.fft` uses CUDA-accelerated FFT (cuFFT) on GPU. For sequence lengths that are not powers of 2, cuFFT uses mixed-radix algorithms. When tuning sequence lengths for FFT-based models (FNet, FNO, FlashFFTConv), choosing $N = 2^k$ or $N = 2^a \cdot 3^b \cdot 5^c$ avoids prime-length slowdowns.

---

## 5. Spectral Leakage and Windowing

### 5.1 The Leakage Problem

The DFT assumes that the $N$ samples represent one complete period of a periodic signal. In reality, we observe a finite segment of a continuous signal - we multiply by a **rectangular window** that is 1 inside $[0, N-1]$ and 0 outside. In frequency, multiplication by a rectangular window corresponds to **convolution** with the DFT of the rectangle - the **Dirichlet kernel**:

$$W_{\text{rect}}[k] = \sum_{n=0}^{N-1} \omega_N^{-nk} = \begin{cases} N & k=0 \\ \frac{1-\omega_N^{-Nk}}{1-\omega_N^{-k}} = \frac{\sin(\pi k)}{\sin(\pi k/N)}e^{-i\pi k(N-1)/N} & k \neq 0 \end{cases}$$

For non-integer frequencies (frequencies that do not land exactly on a DFT bin), the Dirichlet kernel's large sidelobes spread energy from the true frequency into adjacent bins - this is **spectral leakage**.

**Example.** Suppose the signal is $x(t) = \cos(2\pi f_0 t)$ with $f_0 = 3.5 f_s/N$ (a non-integer number of cycles in the window). The true spectrum has just two peaks at $\pm f_0$. The DFT shows energy spread across all $N$ bins, with the main peak at the nearest bin ($k=3$ or $k=4$) and significant sidelobe contamination at all other bins.

**Consequence for AI:** In Whisper's mel spectrogram, a sharp onset (e.g., a consonant burst) at a non-bin frequency would appear smeared across multiple mel filterbanks. Window choice directly affects how well transient events are separated in the spectrogram.

### 5.2 Window Functions

A **window function** $w[n]$ tapers the signal to zero at its endpoints, reducing the discontinuity that causes leakage. The windowed DFT is:

$$X_w[k] = \sum_{n=0}^{N-1} w[n]\, x[n]\, \omega_N^{-nk}$$

The spectrum $X_w$ is the DFT of $w[n]x[n]$ - by the product property (Section 3.7), this equals the circular convolution of $X$ and $W$ (normalized by $1/N$). Windows with smaller sidelobes produce less leakage at the cost of a wider main lobe (reduced frequency resolution).

**Standard window functions** (all defined for $n = 0, 1, \ldots, N-1$):

**Rectangular (Boxcar):**
$$w[n] = 1$$
- Main lobe width: $4\pi/N$ (narrowest)
- Peak sidelobe: $-13\,\mathrm{dB}$ (highest leakage)
- Use case: when the signal exactly fits $N$ samples; spectral analysis of periodic signals at bin frequencies

**Hann (Hanning):**
$$w[n] = \frac{1}{2}\left(1 - \cos\!\frac{2\pi n}{N-1}\right)$$
- Main lobe width: $8\pi/N$
- Peak sidelobe: $-31.5\,\mathrm{dB}$
- Use case: default choice for general spectral analysis; used in Whisper STFT

**Hamming:**
$$w[n] = 0.54 - 0.46\cos\!\frac{2\pi n}{N-1}$$
- Main lobe width: $8\pi/N$
- Peak sidelobe: $-42.7\,\mathrm{dB}$
- Use case: narrowband applications requiring low sidelobes; slightly wider main lobe than Hann

**Blackman:**
$$w[n] = 0.42 - 0.5\cos\!\frac{2\pi n}{N-1} + 0.08\cos\!\frac{4\pi n}{N-1}$$
- Main lobe width: $12\pi/N$
- Peak sidelobe: $-58\,\mathrm{dB}$
- Use case: high-dynamic-range spectral analysis

**Kaiser:**
$$w[n] = \frac{I_0\!\left(\beta\sqrt{1-(2n/(N-1)-1)^2}\right)}{I_0(\beta)}$$
where $I_0$ is the zeroth-order modified Bessel function and $\beta$ is a shape parameter.
- $\beta = 0$: rectangular; $\beta \approx 5$: similar to Hamming; $\beta \approx 8.6$: similar to Blackman
- Chebyshev-optimal: for a given main-lobe width, the Kaiser window minimizes peak sidelobe level
- Use case: when precise sidelobe control is required; used in scientific spectral analysis and filter design

### 5.3 The Main-Lobe/Side-Lobe Tradeoff

Every window function must satisfy a fundamental constraint: the product of main-lobe width (in frequency bins) and side-lobe level (in dB) cannot be simultaneously minimized. This is a consequence of the uncertainty principle applied to finite sequences.

| Window | Main-lobe width (bins) | Peak sidelobe (dB) | Sidelobe rolloff (dB/oct) |
|---|---|---|---|
| Rectangular | 2 | $-13$ | $-6$ |
| Hann | 4 | $-31.5$ | $-18$ |
| Hamming | 4 | $-42.7$ | $-6$ |
| Blackman | 6 | $-58$ | $-18$ |
| Blackman-Harris | 8 | $-92$ | $-6$ |
| Kaiser ($\beta=8.6$) | 8 | $-80$ | $-6$ |

**Coherent gain:** the window amplitude gain, $\sum_n w[n]/N$. Windows with lower sidelobes also have lower coherent gain (the main peak is smaller), which must be corrected by dividing by the coherent gain when measuring amplitudes.

**Equivalent noise bandwidth (ENBW):** the width of a rectangular window that would pass the same noise power. Hann has ENBW $= 1.5$ bins - it is effectively 1.5 times wider than the rectangular window for noise measurements.

### 5.4 Scalloping Loss and the Picket Fence Effect

The DFT evaluates the spectrum at exactly $N$ uniformly-spaced bins. If a sinusoid's true frequency $f_0$ falls exactly halfway between two bins (at $k + 0.5$ in bin units), then both neighboring bins are equally attenuated. The worst-case attenuation - the **scalloping loss** - depends on the window:

- Rectangular: $-3.9\,\mathrm{dB}$ (worst case for nearest-bin amplitude estimate)
- Hann: $-1.4\,\mathrm{dB}$
- Blackman: $-0.8\,\mathrm{dB}$

This is the **picket fence effect**: the DFT spectrum looks like a picket fence - it shows signal energy at the posts (bins) but may miss peaks in the gaps between posts.

**Mitigation strategies:**

1. **Zero-padding** (Section 6.2): append zeros to increase $N$ without changing the observation window, interpolating the spectrum between bins. This does NOT improve frequency resolution (the bandwidth $1/T$ is unchanged) but allows finer frequency estimation.

2. **Parabolic interpolation**: fit a parabola to the peak bin and its two neighbors; the parabola's maximum is a better estimate of the true frequency.

3. **Quinn's estimator** and **Jacobsen's estimator**: closed-form formulas giving sub-bin frequency estimates with low computational cost.

### 5.5 Overlap-Add and Overlap-Save

For long signals (audio tracks, continuous data streams), we cannot apply a single DFT to the entire signal. We must process it in blocks. Two standard methods handle the boundary effects at block edges:

**Overlap-Add (OLA):**
1. Segment the input into blocks of length $M$
2. Window each block with $w[n]$ (length $M$)
3. Zero-pad each block to length $N \geq M + L - 1$ (where $L$ is the filter length)
4. Compute DFT-based processing on each block
5. Overlap and add the outputs (adjacent outputs share $L-1$ samples)

**Overlap-Save (OLS):**
1. Segment the input into blocks of length $N$ with overlap of $L-1$ samples
2. DFT-based processing on each block (no zero-padding needed)
3. Discard the first $L-1$ samples of each output block (the "contaminated" samples)
4. Concatenate the remaining $N - L + 1$ samples

Both methods allow exact (non-approximative) processing of long signals with FFT-based operations. The COLA (Constant Overlap-Add) condition for perfect reconstruction is:

$$\sum_{m=-\infty}^{\infty} w[n - mH] = C \quad \forall n$$

where $H$ is the hop size. Hann windows with 50% overlap ($H = N/2$) satisfy COLA, making them the standard for STFT-based audio processing.

---

## 6. Frequency Resolution and Zero-Padding

### 6.1 Frequency Resolution

The **frequency resolution** of the DFT is:

$$\Delta f = \frac{f_s}{N} = \frac{1}{T}$$

where $T = N/f_s$ is the total observation duration. This fundamental limit says: to distinguish two sinusoids at frequencies $f_1$ and $f_2$, you need to observe the signal for at least $T = 1/|f_1 - f_2|$ seconds. No amount of zero-padding or upsampling can circumvent this - it is a direct consequence of the uncertainty principle from Section 02.

**The time-bandwidth product is fixed:** $T \cdot \Delta f = 1$. You cannot simultaneously have:
- Short observation window (good for tracking rapid changes)
- Fine frequency resolution (good for distinguishing close frequencies)

**For AI:** Whisper uses a 25 ms analysis window (400 samples at 16 kHz) with 10 ms hop size, giving $\Delta f = 1/0.025 = 40$ Hz frequency resolution. This is a deliberate engineering choice: fine enough to separate speech formants (typically 100-300 Hz apart) while short enough to track fast phoneme transitions.

### 6.2 Zero-Padding

**Zero-padding**: appending $M - N$ zeros to a signal of length $N$ before computing an $M$-point DFT ($M > N$).

**What zero-padding does:** it computes $M$ equally-spaced samples of the **continuous** DTFT of the original $N$-point signal (i.e., the DTFT sampled at $M$ bins instead of $N$). This is **frequency-domain interpolation** - it fills in points between the original $N$ bins.

**What zero-padding does NOT do:** improve frequency resolution. The DTFT itself has resolution limited by $1/N$ (the original number of samples). Zero-padding only reveals the DTFT more finely; it cannot resolve frequencies closer than $1/N$ apart.

**Applications of zero-padding:**

1. **Sub-bin frequency estimation** (combine with interpolation): zero-padding by factor of 4-8 allows visual identification of spectral peaks with sub-bin accuracy.

2. **Linear convolution via FFT:** to convolve signals of lengths $L_1$ and $L_2$ linearly (not circularly), zero-pad both to length $\geq L_1 + L_2 - 1$ before DFT multiplication (Section 3.6). The circular convolution of the zero-padded sequences equals the linear convolution of the originals.

3. **Next power-of-2**: zero-pad to the nearest power of 2 to maximize FFT efficiency, e.g., 1000 -> 1024.

**For AI:** The Fourier Neural Operator (Section 8.2) explicitly truncates the spectrum to the lowest $K$ frequencies before learning, implicitly treating the signal as if zero-padded beyond the training domain. FlashFFTConv similarly uses zero-padding to convert long circular convolutions to linear ones.

### 6.3 The Nyquist-Shannon Sampling Theorem

**Theorem (Shannon, 1949; Nyquist, 1928; Whittaker, 1915).** If a continuous signal $x(t)$ is bandlimited - meaning $\hat{x}(\xi) = 0$ for $|\xi| > B$ - then $x(t)$ can be perfectly reconstructed from samples at rate $f_s \geq 2B$.

**Why the DFT cares:** if $f_s < 2B$, frequencies above $f_s/2$ **alias** onto lower frequencies. Specifically, a sinusoid at frequency $f_0 > f_s/2$ produces the same sample sequence as a sinusoid at frequency $f_s - f_0 < f_s/2$. In the DFT, bin $k$ and bin $N-k$ represent the same physical frequency if $f_s$ is not high enough.

**The aliasing formula:** a signal at frequency $f_0$ aliases to $f_{\text{alias}} = |f_0 - \text{round}(f_0/f_s) \cdot f_s|$, i.e., the nearest copy within $[0, f_s/2)$.

**Connection to Poisson summation** (from Section 02-7.5): the spectrum of the sampled signal is the sum of periodically-shifted copies of the continuous spectrum:

$$X_s(\xi) = f_s \sum_{k=-\infty}^{\infty} \hat{x}(\xi - kf_s)$$

If adjacent copies overlap (i.e., $f_s < 2B$), they contaminate each other - this is aliasing. If $f_s \geq 2B$, the copies are non-overlapping and the original spectrum can be recovered by applying a lowpass filter with cutoff at $f_s/2$.

**Practical rule:** In practice, use $f_s \geq 2.5B$ with an anti-aliasing lowpass filter at $f_s/2$ before the ADC. Audio CDs use $f_s = 44.1$ kHz because human hearing extends to $\sim 20$ kHz (requires $f_s > 40$ kHz), with the extra 2.1 kHz providing a transition band for the anti-aliasing filter.

### 6.4 Practical FFT Setup Checklist

When computing an FFT-based spectral analysis, follow this sequence:

```
PRACTICAL FFT CHECKLIST
========================================================================

  1. CHOOSE N
     - Prefer N = 2^k (fastest FFT)
     - If not possible: N = 2^a * 3^b * 5^c (still fast)
     - N controls frequency resolution: f = fs / N

  2. APPLY WINDOW
     - Default: Hann window (good sidelobes, modest resolution loss)
     - High dynamic range: Blackman or Kaiser ( approx 8)
     - If signal is periodic with exactly N samples: rectangular

  3. COMPUTE FFT
     - Real input: use rfft (returns N/2+1 values)
     - Complex input: use fft (returns N values)

  4. BUILD FREQUENCY AXIS
     - For rfft: freqs = np.fft.rfftfreq(N, d=1/fs)   [0, fs/2]
     - For fft:  freqs = np.fft.fftfreq(N, d=1/fs)    [0, fs/2, -fs/2, 0]
     - After fftshift: freqs = [-fs/2, ..., 0, ..., fs/2]

  5. INTERPRET MAGNITUDE
     - |X[k]| / N  ->  amplitude of the sinusoid
     - |X[k]|^2 / N  ->  power spectral density
     - Divide by window coherent gain for absolute amplitudes

  6. HANDLE NEGATIVE FREQUENCIES
     - Use fftshift(X) to center zero-frequency
     - For real inputs: mirror the positive half (rfft returns one-sided)

========================================================================
```

---

## 7. Short-Time Fourier Transform

### 7.1 Motivation: Time-Varying Spectra

The DFT and continuous FT assume the signal's spectral content is **stationary** - constant over the entire observation window. This assumption fails catastrophically for real-world signals:

- **Speech**: phonemes last 20-100 ms, with frequency content changing at each phoneme boundary
- **Music**: pitch and timbre evolve continuously; a melody is precisely a time-varying spectrum
- **EEG**: neural oscillations appear and disappear; seizure onset is detected by spectral changes
- **Radar**: target velocity changes the Doppler shift over time

The **Short-Time Fourier Transform (STFT)** addresses this by computing a sequence of DFTs on overlapping short segments of the signal. The result is a two-dimensional representation: time on one axis, frequency on the other.

### 7.2 STFT Definition

**Definition (STFT).** The Short-Time Fourier Transform of signal $x[n]$ with analysis window $w[m]$ of length $M$ and hop size $H$ is:

$$\text{STFT}\{x\}[l, k] = \sum_{m=0}^{M-1} x[lH + m]\, w[m]\, \omega_N^{-mk}, \quad l = 0, 1, \ldots, L-1,\; k = 0, 1, \ldots, N-1$$

where:
- $l$ is the frame index (time axis), $lH$ is the starting sample of frame $l$
- $H$ is the hop size (number of samples between consecutive frames, $H \leq M$)
- $N \geq M$ is the FFT size (zero-padding if $N > M$)
- $L = \lfloor (T_{\text{signal}} - M) / H \rfloor + 1$ is the number of frames

**Parameters and their effects:**

| Parameter | Effect |
|---|---|
| Window length $M$ | Longer $M$ -> better frequency resolution, worse time resolution |
| Hop size $H$ | Smaller $H$ -> more frames (denser time axis), more computation |
| FFT size $N$ | $N > M$: zero-padding for denser frequency grid (not more resolution) |
| Window type | Hann (default), Hamming, Blackman - sidelobe/resolution tradeoff (Section 5.2) |

**Whisper's STFT parameters:** $M = 400$ samples (25 ms), $H = 160$ samples (10 ms), $N = 512$, window = Hann. At 16 kHz, this gives 257 frequency bins and one frame every 10 ms.

### 7.3 The Spectrogram

**Definition (Spectrogram).** The **spectrogram** is the squared magnitude of the STFT:

$$\text{Spectrogram}[l, k] = |\text{STFT}\{x\}[l, k]|^2$$

It represents the **power** (not amplitude) of each frequency at each time frame. Displayed as a heatmap (time on x-axis, frequency on y-axis, color = power), it visualizes how the frequency content of the signal evolves over time.

**Log-power spectrogram:** $10\log_{10}(|\text{STFT}|^2)$ in decibels. The logarithmic scale compresses the dynamic range and matches the perceptual sensitivity of human hearing (roughly logarithmic in both time and frequency).

**Mel spectrogram:** sum the spectrogram over mel-frequency filterbanks (triangular filters spaced on the mel scale, which is a perceptual frequency scale logarithmic above 1 kHz). Used in Whisper, most modern ASR systems, and many audio AI models. Full details in Section 8.1.

**Chirp spectrogram** (example): a signal sweeping linearly from $f_1$ to $f_2$ shows a diagonal stripe in the spectrogram - the instantaneous frequency increases linearly with time. This illustrates how the STFT tracks time-varying frequency content.

### 7.4 Uncertainty Principle for STFT

The STFT cannot escape the Heisenberg uncertainty principle from Section 02. For the windowed transform with window $w$ of duration $T_w$ and bandwidth $B_w$:

$$T_w \cdot B_w \geq \frac{1}{4\pi}$$

For a Hann window of length $M$ samples (duration $T_w = M/f_s$), the effective bandwidth is $B_w \approx 2f_s/M$ (the -3 dB bandwidth of the Hann spectral window). The time-frequency resolution product is fixed:

$$\Delta t \cdot \Delta f \geq \frac{1}{4\pi} \approx 0.08$$

**Practical consequence:** you cannot simultaneously achieve:
- Fine time resolution (short window $\to$ small $M$)
- Fine frequency resolution (long window $\to$ large $M$)

For speech recognition (Whisper): $M = 400$ samples -> $\Delta t = 25\,\text{ms}$, $\Delta f \approx 40\,\text{Hz}$. For musical pitch tracking (need $\Delta f < 1\,\text{Hz}$ to distinguish semitones near 100 Hz): requires $M > 16000$ samples ($> 1$ second window). The STFT is inadequate for simultaneously good time and frequency resolution across scales - **this is exactly why wavelets exist** (Section 05 Wavelets).

### 7.5 Synthesis and Perfect Reconstruction

**Overlap-Add synthesis:** Given STFT frames $S[l, k]$ (possibly modified), reconstruct $x[n]$ by:

$$\hat{x}[n] = \frac{\sum_l w_s[n - lH] \cdot \text{IDFT}\{S[l, \cdot]\}[n-lH]}{\sum_l w_a[n-lH]\, w_s[n-lH]}$$

where $w_a$ is the analysis window and $w_s$ is the synthesis window.

**COLA (Constant Overlap-Add) condition:** for perfect reconstruction of unmodified STFT:

$$\sum_{l=-\infty}^{\infty} w[n - lH]^2 = C \quad \forall n \quad \text{(using synthesis window } w_s = w_a = w \text{)}$$

Hann window with $H = M/2$ (50% overlap): $w[n]^2 + w[n-M/2]^2 = 1$ for all $n$ - COLA satisfied.

**Modified DFT (MDCT):** audio codecs (AAC, Vorbis, Opus) use the MDCT - a cosine-based transform with 50% overlap and critically sampled (no redundancy). The MDCT satisfies exact reconstruction (called "Time Domain Aliasing Cancellation") and is why modern audio codecs can achieve high quality at low bit rates. Full filter bank theory is developed in Section 05 Wavelets.

> **Forward reference:** The STFT and MDCT are the simplest multi-resolution analyses. Wavelets generalize this by allowing different window lengths at different frequencies - short windows at high frequencies (good time resolution) and long windows at low frequencies (good frequency resolution). See [Section 20-05 Wavelets](../05-Wavelets/notes.md) for the full treatment.

---

## 8. Applications in Machine Learning

### 8.1 Whisper: FFT Pipeline for Speech Recognition

OpenAI Whisper (Radford et al., 2022) is a 680M-parameter transformer trained on 680,000 hours of multilingual speech. Its input is not raw audio but an **80-channel log-mel spectrogram** computed by a fixed FFT pipeline.

**The Whisper preprocessing pipeline:**

```
WHISPER AUDIO PREPROCESSING PIPELINE
========================================================================

  Raw audio (PCM, 16 kHz)
        |
        
  1. Resample to 16 kHz (if needed)
        |
        
  2. Pad/trim to 30-second segments (480,000 samples)
        |
        
  3. STFT: window=Hann(400), hop=160, N_FFT=512
     -> Complex spectrogram: shape (257, T_frames)
        |
        
  4. Power spectrogram: |STFT|2 -> shape (257, T_frames)
        |
        
  5. Apply 80 mel filterbanks (triangular, mel-spaced, 0-8000 Hz)
     -> Mel power spectrogram: shape (80, T_frames)
        |
        
  6. Log compression: log10(max(mel_spec, 1e-10))
        |
        
  7. Normalize: (log_mel - mean) / 4  (empirically determined)
        |
        
  Transformer input: shape (80, 3000) for 30-second clip
  (3000 frames  10ms/frame = 30 seconds)

========================================================================
```

**Mel filterbanks:** The mel scale maps frequency $f$ (Hz) to perceived pitch $m$ (mel):

$$m = 2595 \log_{10}\!\left(1 + \frac{f}{700}\right)$$

Filterbanks are triangular filters equally spaced on the mel scale, covering the range $[f_{\min}, f_{\max}] = [0, 8000]\,\text{Hz}$. Human speech information is concentrated below 8 kHz; the mel scale matches the roughly logarithmic frequency resolution of the cochlea.

**Why log-mel spectrograms?** (1) Log compression matches the roughly logarithmic perception of loudness (Stevens' power law). (2) Log mel spectrograms are approximately translation-equivariant in the log-frequency dimension - a pitch shift is a translation in log-frequency. (3) Empirically, they produce much better ASR results than raw spectrograms or MFCCs (Mel-Frequency Cepstral Coefficients) for large-scale transformer models.

**For AI:** Every audio transformer (Whisper, AudioPaLM, MusicGen, AudioLM) uses a similar FFT -> mel spectrogram pipeline as its front-end. Understanding the pipeline is essential for debugging audio preprocessing bugs, which are among the most common sources of silent failure in audio AI systems.

### 8.2 Fourier Neural Operator (FNO)

The Fourier Neural Operator (Li et al., 2021, NeurIPS) is a neural architecture for learning solution operators of PDEs. Instead of learning to map one function to another pointwise, the FNO learns in the frequency domain.

**Problem setting:** given input function $a : D \to \mathbb{R}$ (e.g., initial condition of a PDE), learn the operator $\mathcal{G} : a \mapsto u$ where $u$ is the solution (e.g., PDE solution at time $T$).

**FNO layer:** each FNO layer $v_t \mapsto v_{t+1}$ applies:

$$v_{t+1}(x) = \sigma\!\left(W v_t(x) + \mathcal{F}^{-1}\!\left[R_\phi \cdot \mathcal{F}[v_t]\right](x)\right)$$

where:
- $\mathcal{F}$ is the DFT along the spatial dimension
- $R_\phi \in \mathbb{C}^{K \times d_v \times d_v}$ is a learned complex weight matrix (parametrized by $\phi$)
- $K$ is the number of retained frequency modes (truncation threshold)
- $W$ is a local linear transform (pointwise)
- $\sigma$ is a nonlinearity

**Key innovation:** truncate to the lowest $K$ Fourier modes before applying $R_\phi$:

$$(\mathcal{F}[v_t])_k = 0 \quad \text{for } k > K$$

This implements a **global convolution** with a low-frequency kernel - learning only the large-scale structure of the operator, discarding high-frequency noise. For 2D problems (e.g., Navier-Stokes), truncate to $(K_1, K_2)$ modes in each dimension.

**Computational cost:** $O(N \log N)$ for the DFT plus $O(K d_v^2)$ for the spectral convolution, vs. $O(N^2)$ for full global convolution. For $N = 10^5$ and $K = 20$: the FNO spectral layer is $\sim 250$ cheaper than full attention.

**For AI:** FNO was the first neural architecture to demonstrate that learning in frequency space outperforms learning in physical space for many PDE tasks. It has been extended to weather prediction (Pathak et al., 2022 "FourCastNet"), fluid dynamics, and climate modeling. The core idea - truncate, learn in frequency, invert - generalizes the classic Galerkin method from numerical PDE.

### 8.3 Monarch Matrices and Structured FFT

Dao et al. (2022, "Monarch: Expressive Structured Matrices for Efficient and Accurate Training") observed that the FFT matrix $F_N$ can be factored as a product of sparse **butterfly matrices**:

$$F_N = B_{\log_2 N} \cdots B_2 B_1 P$$

where $P$ is the bit-reversal permutation and each $B_j$ is a block-diagonal matrix with $N/2$ butterfly factors. Each $B_j$ has exactly $N$ non-zero entries (2 per butterfly), giving $O(N)$ parameters per stage and $O(N \log N)$ total parameters for the full factorization.

**Monarch parameterization:** instead of fixing the butterfly structure to be the FFT, learn the butterfly weights:

$$M = P L P^\top R$$

where $L$ and $R$ are block-diagonal "butterfly-like" matrices. This gives a learnable structured matrix with $O(N\sqrt{N})$ parameters (between $O(N)$ for diagonal and $O(N^2)$ for dense).

**Applications:**
- **Efficient attention**: Monarch matrices can approximate attention in $O(N\sqrt{N})$ vs $O(N^2)$
- **Hyperspherical projection**: replacing standard linear layers with Monarch matrices in transformers for 2 speedup with minimal accuracy loss
- **FFT-based initialization**: Monarch matrices initialized as FFT butterfly factors learn sparse frequency representations naturally

**For AI:** The butterfly matrix decomposition connects signal processing (FFT structure) to deep learning (efficient matrix approximation). Models like FLASHATTENTION-2 and several emerging efficient transformer variants build on butterfly-structured operators as efficient replacements for dense weight matrices.

### 8.4 FlashFFTConv and Long-Range Convolutions

Dao et al. (2023, "FlashFFTConv: Efficient Convolutions for Long Sequences with Tensor Cores") addresses a key bottleneck in long-context sequence modeling: computing long convolutions efficiently on modern GPU hardware.

**The problem:** State Space Models (SSMs) like S4 (Gu et al., 2022) and Mamba (Gu & Dao, 2023) compute long convolutions in $O(N \log N)$ using the FFT-based convolution theorem. However, standard FFT implementations are memory-bandwidth-bound and do not efficiently utilize the Tensor Cores (matrix-multiply units) of modern GPUs (A100, H100).

**FlashFFTConv solution:** decompose the long convolution kernel $h$ of length $N$ using the Monarch factorization into a product of shorter matrix operations, enabling Tensor Core utilization while preserving $O(N \log N)$ asymptotic complexity.

**Key results:**
- 2-7.3 speedup over standard FFT convolution for sequences up to 4M tokens
- Linear memory in sequence length (vs. quadratic for full attention)
- Enables training SSMs on sequences orders of magnitude longer than attention allows

**Connection to Mamba:** Mamba (Gu & Dao, 2023) uses selective state spaces - a time-varying SSM where the transition matrices depend on the input. The resulting recurrence can be parallelized via a parallel scan (prefix sum), NOT an FFT. FlashFFTConv applies specifically to time-invariant SSMs like S4.

### 8.5 Spectral Methods in Graph Neural Networks

Graph Neural Networks (GCNs) performing spectral graph convolution use the **graph Laplacian eigenvectors** as a "Fourier basis" for the graph. The connection is:

- Graph Laplacian $L = D - A$ (where $D$ is degree matrix, $A$ is adjacency)
- Spectral decomposition $L = U \Lambda U^\top$ provides "graph Fourier modes" $U$
- Graph convolution: $y = U \cdot g_\theta(\Lambda) \cdot U^\top x$ where $g_\theta$ is a learnable spectral filter

This is a graph-domain analog of the DFT, with the regular frequency grid replaced by the graph eigenvalue spectrum.

> **Forward reference to Section 11-04:** The full treatment of spectral graph convolution - including ChebNet (Defferrard et al., 2016), GCN (Kipf & Welling, 2017), and the connection to the graph Laplacian - is the canonical content of [Section 11-04 Spectral Graph Theory](../../11-Graph-Theory/04-Spectral-Methods/notes.md). Here we note only the structural analogy: DFT is to regular grids as graph Fourier transform is to irregular graphs.


---

## 9. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
|---|---------|---------------|-----|
| 1 | Using `np.fft.fft` output directly as frequency-domain amplitude without normalization | `fft` returns $X[k] = \sum_n x[n]\omega_N^{-nk}$; the amplitude of a unit sinusoid is $N/2$, not 1 | Divide by $N$ (or $N/2$ for one-sided real FFT) to recover true sinusoidal amplitudes |
| 2 | Building frequency axis as `np.arange(N) * fs / N` and treating all $N$ bins as positive frequencies | Bins $k > N/2$ represent **negative** frequencies wrapped around; calling $X[N-1]$ "frequency $(N-1)f_s/N$" is incorrect | Use `np.fft.fftfreq(N, d=1/fs)` which returns the correct signed frequencies; apply `fftshift` to center zero |
| 3 | Forgetting to apply `fftshift` before plotting a spectrum | Without `fftshift`, the negative-frequency bins appear at the right end of the plot, not the left, making the spectrum appear discontinuous | Always call `fftshift(X)` and `fftshift(freqs)` before plotting a two-sided spectrum |
| 4 | Concluding that zero-padding improves frequency resolution | Zero-padding increases the density of frequency bins but does NOT reduce $\Delta f = 1/T$; two sinusoids closer than $1/T$ remain unresolvable regardless of zero-padding | Say "zero-padding interpolates the spectrum" not "improves resolution"; use longer observation windows for genuine resolution improvement |
| 5 | Applying DFT-property formulas (e.g., shift property) to linearly-shifted signals | The DFT shift property applies only to **circular** shifts. A linear shift (with zeros entering from one end) introduces boundary effects that invalidate $Y[k] = \omega_N^{-mk}X[k]$ | Use circular indexing explicitly, or use the linear convolution approach with appropriate zero-padding |
| 6 | Confusing the DFT normalization convention of different libraries | NumPy puts $1/N$ on `ifft` only; some textbooks use $1/\sqrt{N}$ on both; PyTorch matches NumPy by default but has `norm` parameter options | Always check the documentation of the specific function; print a known signal (e.g., pure DC) and verify the expected coefficient |
| 7 | Misinterpreting spectral leakage as separate frequency components | When a sinusoid does not land on a DFT bin, its energy spreads to neighboring bins, appearing as multiple "peaks" in the spectrum | Apply a window function before the DFT and look for main lobes (not individual bins) as frequency indicators |
| 8 | Treating the Nyquist frequency bin $X[N/2]$ as a complex value | For real input, $X[N/2]$ is always real (by conjugate symmetry: $X[N/2] = X[N/2]^*$). Its imaginary part should be zero (up to numerical noise) | Assert `np.allclose(X[N//2].imag, 0)` for real input; take only the real part when reporting the Nyquist coefficient |
| 9 | Using an FFT size smaller than the signal length | If $N < \text{len}(x)$, NumPy silently **truncates** the input; this discards signal information and produces incorrect results | Set $N \geq \text{len}(x)$; for linear convolution, use $N \geq L_1 + L_2 - 1$ |
| 10 | Applying windowing after the FFT | Windows must be applied to the time-domain signal **before** the FFT. Multiplying the frequency-domain output by a window function is a circular convolution in frequency, not spectral smoothing | Multiply `x * w` in the time domain, then compute `fft(x * w)` |
| 11 | Overlooking numerical precision loss in the FFT | FFT algorithms propagate floating-point errors; for $N = 2^{20}$, accumulated rounding errors can be $\sim 10^{-9}$; this matters for precision-sensitive applications | Use `np.float64` (double precision) inputs; for critically sensitive applications, consider the inverse error bound $\epsilon_{\text{FFT}} \approx \epsilon_{\text{machine}} \log_2 N$ |
| 12 | Applying DFT to complex-valued signals and expecting `rfft` to work | `rfft` assumes real input and returns $N/2+1$ values; calling it on complex input produces incorrect results (it treats the input as real, discarding imaginary parts) | Use `fft` (not `rfft`) for complex inputs; use `rfft` only when you are certain the signal is real-valued |

---

## 10. Exercises

**Exercise 1** * - DFT by Hand

**(a)** Compute the 4-point DFT of $\mathbf{x} = (1, 0, -1, 0)$ by hand using $\omega_4 = i$.
Show all four coefficients $X[0], X[1], X[2], X[3]$.

**(b)** Verify that $X[0]$ equals the sum of all samples (DC component) and that $|X[k]|^2$ satisfies Parseval's theorem.

**(c)** What is the IDFT of $\mathbf{X} = (4, 0, 0, 0)$? Compute by hand and verify.

**(d)** For $\mathbf{x} = (1, 1, 1, 1)$, what is the DFT? Explain geometrically why $X[k] = 0$ for $k \neq 0$.

---

**Exercise 2** * - DFT Matrix Properties

**(a)** Construct the 44 DFT matrix $F_4$ explicitly and verify numerically that $F_4 F_4^* = 4I$.

**(b)** Show that $F_4^2 = N P$ where $P$ is the time-reversal permutation matrix (i.e., applying the DFT twice returns the time-reversed signal scaled by $N$).

**(c)** Compute the eigenvalues of $F_8$ and show they belong to $\{1, -1, i, -i\}$. (Hint: $F_N^4 = N^2 I$.)

**(d)** For a random complex vector $\mathbf{x} \in \mathbb{C}^8$, numerically verify the unitary property: $\lVert F_8\mathbf{x}/\sqrt{8} \rVert_2 = \lVert \mathbf{x} \rVert_2$.

---

**Exercise 3** * - FFT Implementation from Scratch

**(a)** Implement the radix-2 Cooley-Tukey FFT recursively in Python (without using `numpy.fft`). Your function should accept any power-of-2 length.

**(b)** Test on inputs of length $N \in \{8, 64, 512, 4096\}$. Verify against `numpy.fft.fft` with relative error $< 10^{-10}$.

**(c)** Time your implementation vs `numpy.fft.fft` for $N = 2^{14}$. By what factor is NumPy faster?

**(d)** Implement the bit-reversal permutation as a standalone function and verify it matches the permutation implicitly used by your FFT.

---

**Exercise 4** ** - Spectral Leakage and Windowing

**(a)** Generate a signal $x[n] = \cos(2\pi \cdot 3.7 \cdot n/N)$ for $N = 128$ samples (a sinusoid at a non-integer frequency). Compute and plot the DFT magnitude for rectangular, Hann, and Blackman windows.

**(b)** Quantify the leakage: for the rectangular window, what fraction of the total spectral energy lies outside the main lobe (bins $\{3, 4\}$)?

**(c)** Two sinusoids at frequencies $f_1 = 5$ and $f_2 = 5.5$ cycles over $N = 64$ samples. Show that they cannot be resolved with the rectangular window, but can be partially resolved after zero-padding to $N = 256$.

**(d)** Implement Kaiser window computation (using `scipy.special.i0`) and sweep $\beta$ from 0 to 14. Plot peak sidelobe level vs. main-lobe width. Reproduce the three window parameter rows from the table in Section 5.3.

---

**Exercise 5** ** - Parseval, Energy, and the Power Spectral Density

**(a)** Generate a bandlimited noise signal: $x[n] = \sum_{k=10}^{20} \cos(2\pi kn/N + \phi_k)$ for $N = 512$ with random phases $\phi_k \sim \mathcal{U}(0, 2\pi)$. Verify Parseval's theorem numerically: $\sum_n |x[n]|^2 = \frac{1}{N}\sum_k |X[k]|^2$.

**(b)** Compute the one-sided power spectral density (PSD) and identify the occupied bandwidth.

**(c)** Compare the PSD estimated by the DFT (single-frame) vs Welch's method (`scipy.signal.welch`, averaged over overlapping frames). Show that Welch's estimate is less noisy.

**(d)** For white noise $x[n] \sim \mathcal{N}(0,1)$, verify that the expected PSD is approximately flat, and that the variance of the PSD estimate decreases with the number of Welch averages.

---

**Exercise 6** ** - STFT and Spectrogram Analysis

**(a)** Generate a chirp signal: $x[n] = \cos(2\pi(f_0 + \alpha n/N) \cdot n/f_s)$ with $f_0 = 100$ Hz, $\alpha \cdot f_s = 1000$ Hz, $f_s = 8000$ Hz, $N = 8000$ samples (1 second). Compute and display the STFT spectrogram using Hann window of length 256 samples with 50% overlap.

**(b)** Show how the spectrogram changes as you vary the window length: $M \in \{64, 256, 1024\}$. Identify the time-frequency resolution tradeoff.

**(c)** Add a 500 Hz sinusoidal interference to the chirp. Show that a narrow window resolves the interference in time but not frequency, while a wide window resolves it in frequency but not time.

**(d)** Use the STFT to perform simple spectral subtraction denoising: estimate noise power from a "silent" region, subtract from all frames, reconstruct via overlap-add. Report the signal-to-noise ratio improvement.

---

**Exercise 7** *** - Whisper Mel Spectrogram Pipeline

**(a)** Implement the full Whisper mel-spectrogram pipeline from scratch:
  - STFT with Hann window (400 samples), hop 160, FFT size 512
  - Power spectrogram $|X|^2$
  - 80 mel filterbanks covering 0-8000 Hz on the mel scale
  - Log compression with floor at $10^{-10}$
  
Test on a 1-second synthetic signal: a mixture of three sinusoids at 300, 1200, and 3500 Hz.

**(b)** Verify that your mel filterbanks sum to (approximately) the total spectral power: $\sum_{m=1}^{80} \text{mel\_spec}[m, l] \approx \sum_k w_m[k] |X[k, l]|^2$.

**(c)** Apply your pipeline to the test signal and plot the 80100 mel spectrogram. Label the three sinusoid peaks.

**(d)** Analyze the effect of mel filterbank overlap: are the three sinusoids at 300, 1200, and 3500 Hz resolved in separate mel bins? What is the mel-bin width in Hz at each of the three frequencies?

---

**Exercise 8** *** - Fourier Neural Operator: Spectral Convolution Layer

**(a)** Implement the 1D FNO spectral convolution layer:
  ```python
  def spectral_conv1d(x, weights, modes):
      # x: (batch, channels, N)
      # weights: (in_channels, out_channels, modes) complex
      # modes: number of retained Fourier modes
  ```
  The layer: DFT along last axis -> truncate to `modes` -> multiply by `weights` -> zero-pad -> IDFT.

**(b)** Test on the 1D Burgers' equation approximation: generate smooth initial conditions $u_0 \sim \text{GP}(0, (-\partial^2_x + 1)^{-2})$ and apply your spectral layer. Show that retaining $K = 16$ modes captures $> 99\%$ of the energy for smooth functions.

**(c)** Compare the spectral layer against a naive dense convolution layer of comparable parameter count. For $N = 1024$ and $K = 16$: how many parameters does the spectral layer have vs. the dense layer?

**(d)** Implement a 2-layer FNO and train it to approximate the integral operator $\mathcal{G}[a](x) = \int_0^1 e^{-|x-y|} a(y)\,dy$ on discretized grids of size $N \in \{64, 128, 256\}$. Report the relative $L^2$ error and show that the spectral layer generalizes across grid resolutions (resolution-invariance).

---

## 11. Why This Matters for AI (2026 Perspective)

| Concept | AI/ML Impact |
|---------|-------------|
| **FFT O(N log N) complexity** | Foundation of all efficient sequence models: FNet, S4, Mamba, FNO, FlashFFTConv - the difference between feasible and infeasible at 100K+ token lengths |
| **DFT matrix unitarity** | Guarantees that FFT-based feature maps preserve signal energy; relevant to gradient stability in FNO and FNet - unitary transforms avoid the vanishing/exploding gradient problem in deep spectral networks |
| **Spectral truncation** | FNO's key innovation: learning only low-frequency behavior yields generalizable, resolution-invariant operators; matches the inductive bias that real physics is smooth |
| **STFT and mel spectrograms** | The front-end of every modern audio model (Whisper, AudioPaLM, MusicGen, AudioLM, VALL-E 2); getting the FFT pipeline right is prerequisite for any audio AI work |
| **Windowing** | Hann/Hamming windows in Whisper's STFT directly affect spectrogram quality; poor window choice causes leakage artifacts that corrupt speech features used by the ASR transformer |
| **Butterfly matrix structure** | Monarch matrices (Dao et al., 2022) parameterize structured transforms with $O(N\sqrt{N})$ parameters, enabling trainable FFT-like operations inside transformers at 2-4 speedup |
| **Circular convolution theorem** | Makes FFT-based convolution O(N log N) instead of O(N2); used in FlashFFTConv for long-range dependencies; connection to Section 04 where this is developed fully |
| **Nyquist-Shannon theorem** | Governs sample rate requirements for all audio AI; at 16 kHz (Whisper), the Nyquist limit is 8 kHz, matching the mel filterbank design; misunderstanding aliasing causes silent audio processing bugs |
| **Frequency resolution** | In FNO: $K$ retained modes determines the scale of PDE features the network can learn; too few modes = underfitting; too many = overfitting to noise; analogous to bandwidth selection in statistics |
| **Fourier Neural Operator** | State-of-the-art for weather prediction (FourCastNet, GraphCast partially inspired), fluid dynamics, and climate modeling at $O(N\log N)$ cost; represents the convergence of PDE numerics and deep learning |

---

## 12. Conceptual Bridge

### Looking Backward

This section discretizes and algorithmizes the continuous Fourier Transform from [Section 20-02](../02-Fourier-Transform/notes.md). The connection is precise: the DFT evaluates the continuous FT of a sampled signal at $N$ equally-spaced frequencies, with errors bounded by the Nyquist theorem and the windowing artifact. Every property of the continuous FT - linearity, shift, scaling, Plancherel - has a discrete analog, with the crucial difference that shifts are now circular. The $L^2$ inner product of $\mathbb{R}$ becomes the finite inner product on $\mathbb{C}^N$; unitarity of the continuous FT (Plancherel) becomes unitarity of the DFT matrix. The Poisson summation formula from Section 02-7.5 directly explains why sampled spectra are periodic and why aliasing occurs.

The Cooley-Tukey FFT is not a mathematical theorem about the DFT - it is an algorithmic fact about how to implement the DFT matrix-vector product by exploiting the Vandermonde structure of the DFT matrix. The mathematics is in the DFT; the computer science is in the FFT.

### Looking Forward

The immediate next step is [Section 20-04 Convolution Theorem](../04-Convolution-Theorem/notes.md), which proves the central result previewed in Section 3.6: DFT multiplication equals circular convolution. This makes FFT-based convolution $O(N\log N)$ instead of $O(N^2)$, and underlies every CNN, SSM, and FNO architecture. Overlap-add and overlap-save (introduced in Section 5.5) are the standard tools for applying this theorem to long signals.

[Section 20-05 Wavelets](../05-Wavelets/notes.md) addresses the STFT's fundamental limitation: fixed time-frequency resolution. Wavelets use variable window sizes - short windows at high frequencies, long windows at low frequencies - to achieve optimal time-frequency localization simultaneously across all scales. The connection runs through the filter bank theory developed in Section 04: Daubechies wavelets are exactly the solution to the "perfect reconstruction filter bank" problem.

Further afield, the Fourier Neural Operator of Section 8.2 will appear again in advanced PDE-solving contexts, and the spectral graph convolution preview of Section 8.5 connects to [Section 11-04 Spectral Graph Theory](../../11-Graph-Theory/04-Spectral-Methods/notes.md). The mathematical framework of the DFT as a change of basis in a finite-dimensional Hilbert space generalizes to infinite-dimensional Hilbert spaces in [Section 12-02 Hilbert Spaces](../../12-Functional-Analysis/02-Hilbert-Spaces/notes.md), where the orthonormal Fourier basis becomes a complete orthonormal system and Parseval's identity extends to an equality that characterizes separable Hilbert spaces.

```
POSITION IN THE FOURIER ANALYSIS CURRICULUM
========================================================================

  Section 20-01  Fourier Series           periodic functions, L2[-,]
          v  (take period T -> infty)
  Section 20-02  Fourier Transform        continuous FT, Plancherel, uncertainty
          v  (sample at rate fs, observe N points)
  Section 20-03  DFT and FFT              <- YOU ARE HERE
          v  (DFT multiplication = circular convolution)
  Section 20-04  Convolution Theorem      CNNs, WaveNet, S4/Mamba, Hyena
          v  (variable-resolution analysis)
  Section 20-05  Wavelets                 MRA, Daubechies, scattering networks

  Key Connections:
  +-----------------------------------------------------------+
  | DFT = discrete + finite version of the continuous FT      |
  | FFT = O(N log N) algorithm for computing the DFT matrix   |
  | STFT = sliding DFT -> time-frequency representation        |
  | Windowing = control leakage  frequency resolution        |
  | Parseval: x2 = X2/N  (energy conserved by DFT)      |
  | Circular convolution theorem -> leads to Section 04               |
  | Fixed resolution limitation of STFT -> leads to Section 05        |
  +-----------------------------------------------------------+

========================================================================
```

[<- Back to Fourier Analysis](../README.md) | [Previous: Fourier Transform <-](../02-Fourier-Transform/notes.md) | [Next: Convolution Theorem ->](../04-Convolution-Theorem/notes.md)

---

## Appendix A: Extended DFT Theory

### A.1 The DFT as a Projection onto the Fourier Basis

The geometric interpretation of the DFT deserves deeper development. Working in $\mathbb{C}^N$ with the standard Hermitian inner product $\langle \mathbf{u}, \mathbf{v} \rangle = \sum_{n=0}^{N-1} u[n]\overline{v[n]}$, we construct the Fourier basis.

**Orthonormality proof.** Define $\mathbf{e}_k = \frac{1}{\sqrt{N}}(1, \omega_N^k, \omega_N^{2k}, \ldots, \omega_N^{(N-1)k})^\top$. Then:

$$\langle \mathbf{e}_k, \mathbf{e}_l \rangle = \frac{1}{N}\sum_{n=0}^{N-1} \omega_N^{nk}\overline{\omega_N^{nl}} = \frac{1}{N}\sum_{n=0}^{N-1}\omega_N^{n(k-l)} = \delta_{kl}$$

by the geometric sum formula. The normalized DFT matrix $\frac{1}{\sqrt{N}}F_N$ has columns $\mathbf{e}_0, \mathbf{e}_1, \ldots, \mathbf{e}_{N-1}$ (up to conjugation), confirming it is unitary.

**The DFT as expansion coefficients.** The expansion of any $\mathbf{x} \in \mathbb{C}^N$ in the Fourier basis is:

$$\mathbf{x} = \sum_{k=0}^{N-1} \hat{x}_k \mathbf{e}_k, \qquad \hat{x}_k = \langle \mathbf{x}, \mathbf{e}_k \rangle = \frac{1}{\sqrt{N}}\sum_{n=0}^{N-1} x[n]\omega_N^{-nk} = \frac{X[k]}{\sqrt{N}}$$

The DFT coefficient $X[k]$ is $\sqrt{N}$ times the coordinate of $\mathbf{x}$ in the $k$-th Fourier direction.

**Projection interpretation.** $X[k] = \langle \mathbf{x}, \sqrt{N}\mathbf{e}_k \rangle$ measures the "overlap" or "correlation" of $\mathbf{x}$ with the $k$-th complex sinusoid. A large $|X[k]|$ means $\mathbf{x}$ has significant oscillatory component at frequency $k/N$ cycles per sample.

### A.2 Spectral Density and the Periodogram

The **periodogram** is the classical (non-parametric) estimator of the Power Spectral Density (PSD):

$$\hat{S}(f_k) = \frac{1}{N f_s} |X[k]|^2, \quad f_k = k f_s / N$$

The periodogram is an **inconsistent** estimator - its variance does not decrease as $N \to \infty$, even though its mean converges to the true PSD. This is because each DFT coefficient is approximately a chi-squared random variable with 2 degrees of freedom, regardless of $N$.

**Smoothed periodogram estimators:**

1. **Bartlett's method**: average periodograms from non-overlapping blocks of length $M$. Reduces variance by factor $K$ (number of blocks) at the cost of frequency resolution $\Delta f = f_s/M$.

2. **Welch's method**: average periodograms from **overlapping** windowed blocks. With 50% overlap and $K$ blocks: variance reduction $\approx K/1.5$, same frequency resolution as Bartlett.

3. **Multitaper method (Thomson, 1982)**: use multiple orthogonal "Slepian" (DPSS) tapers instead of one window. Optimal bias-variance tradeoff for stationary processes.

**For AI:** Welch's method is used in biomedical signal processing (EEG power spectra), audio analysis, and anomaly detection. Understanding PSD estimation is essential for interpreting spectral features in any time-series ML pipeline.

### A.3 The Discrete-Time Fourier Transform (DTFT) Relation

The **DTFT** of an infinite sequence $x[n]$ is:

$$X(e^{j\omega}) = \sum_{n=-\infty}^{\infty} x[n]\, e^{-j\omega n}, \quad \omega \in [-\pi, \pi)$$

The DTFT produces a **continuous** spectrum over the normalized frequency $\omega = 2\pi f / f_s \in [-\pi, \pi)$.

**DFT as sampled DTFT.** If $x[n]$ is nonzero only for $n = 0, \ldots, N-1$ (or equivalently, if we window to this range), then:

$$X[k] = X(e^{j\omega})\Big|_{\omega = 2\pi k/N} = X\!\left(e^{j2\pi k/N}\right)$$

The DFT is obtained by sampling the DTFT at $N$ equally-spaced points on the unit circle. Denser sampling (larger $N$ or zero-padding) evaluates the DTFT at more points but does not change the underlying DTFT.

**The $z$-transform connection.** The DTFT evaluates the $z$-transform $X(z) = \sum_n x[n]z^{-n}$ on the unit circle $z = e^{j\omega}$. The DFT evaluates it at $N$ equally-spaced points on the unit circle. Poles of $X(z)$ inside the unit circle correspond to stable causal systems.

### A.4 Goertzel Algorithm: Single-Frequency DFT

Computing the entire $N$-point DFT costs $O(N\log N)$. If you need only a **single** DFT coefficient $X[k_0]$ for a specific frequency $k_0$, the FFT is wasteful. The **Goertzel algorithm** computes $X[k_0]$ in $O(N)$ operations using a second-order IIR filter.

**Second-order filter interpretation.** Define $y[n] = x[n] + 2\cos(\omega_0)y[n-1] - y[n-2]$ with $\omega_0 = 2\pi k_0/N$. Then $X[k_0] = y[N] - e^{-j\omega_0}y[N-1]$ (initialized with $y[-1] = y[-2] = 0$).

**Applications:**
- DTMF (Dual-Tone Multi-Frequency) telephone tone decoding: detect specific frequencies (697, 770, 852, 941, 1209, 1336, 1477, 1633 Hz)
- Musical pitch detection when only octave-frequency analysis is needed
- Frequency-selective filtering in embedded systems with limited memory

---

## Appendix B: Advanced FFT Topics

### B.1 In-Place FFT and Memory Efficiency

The butterfly computation in the DIT-FFT can be performed **in-place**: the two output values of each butterfly can overwrite the two input values, because after the butterfly both inputs are no longer needed. This means the entire $N$-point FFT can be computed using only the $N$ complex input/output array (plus $O(\log N)$ twiddle factor storage).

**Implementation sketch:**

```
FOR each stage s = 1, 2, ..., log2(N):
    stride = N / 2^s
    FOR each group g = 0, 1, ..., 2^(s-1) - 1:
        twiddle = omega^{-g * stride}
        FOR each butterfly within group:
            a = x[position]
            b = twiddle * x[position + stride]
            x[position]          = a + b   (overwrite)
            x[position + stride] = a - b   (overwrite)
```

This avoids allocating a second array of size $N$, halving memory usage - critical for embedded systems and for FFTs of very large signals.

### B.2 Multi-Dimensional FFT

The 2D DFT is separable: apply a 1D FFT to each row, then a 1D FFT to each column (or vice versa):

$$X[k_1, k_2] = \sum_{n_1=0}^{N_1-1}\sum_{n_2=0}^{N_2-1} x[n_1, n_2]\,\omega_{N_1}^{-n_1 k_1}\,\omega_{N_2}^{-n_2 k_2}$$

**Computational cost:** $N_2$ FFTs of length $N_1$ (for rows) plus $N_1$ FFTs of length $N_2$ (for columns) = $O(N_1 N_2 \log(N_1 N_2))$.

**Applications in AI:**
- **Image processing**: 2D FFT used for frequency-domain image filtering, noise removal, and texture analysis
- **Convolutional neural networks**: image convolution can be computed via 2D FFT for large kernels ($O(N^2 \log N)$ vs $O(N^2 K^2)$ for direct convolution with kernel size $K$)
- **FNO on 2D domains**: the 2D FNO for Navier-Stokes uses 2D DFT with $K^2$ retained modes

### B.3 Number-Theoretic Transform (NTT)

The **Number-Theoretic Transform** replaces complex arithmetic with arithmetic modulo a prime $p$. The NTT uses a primitive $N$-th root of unity modulo $p$ - an integer $\omega$ such that $\omega^N \equiv 1 \pmod{p}$ and $\omega^k \not\equiv 1 \pmod{p}$ for $k < N$.

**Why NTT matters:** unlike the FFT with floating-point arithmetic, the NTT is exact (integer arithmetic modulo $p$). It is used for:
- **Polynomial multiplication modulo a prime**: key operation in lattice-based cryptography (NTRU, CRYSTALS-Kyber)
- **Large integer multiplication**: GNU MP library uses NTT for multiplying large integers
- **Post-quantum cryptography**: Kyber and Dilithium (NIST post-quantum standards) rely on NTT for efficient polynomial operations

**For AI security:** language model serving with privacy guarantees (homomorphic encryption) requires NTT-based polynomial multiplications. The efficiency of NTT directly affects the feasibility of private inference for LLMs.

---

## Appendix C: Worked Examples

### C.1 Complete DFT Computation: 8-Point Example

Let $\mathbf{x} = (1, 2, 3, 4, 4, 3, 2, 1)$. Compute the 8-point DFT.

**Step 1:** Note that $\mathbf{x}$ is symmetric: $x[n] = x[N-n]$ for $n = 1, \ldots, N-1$. By conjugate symmetry for real input, $X[k] = X[N-k]^*$. For a symmetric real input, all $X[k]$ are real.

**Step 2:** Even and odd subsequences:
- Even: $(x[0], x[2], x[4], x[6]) = (1, 3, 4, 2)$
- Odd: $(x[1], x[3], x[5], x[7]) = (2, 4, 3, 1)$

**Step 3:** 4-point DFT of even subsequence ($\omega_4 = i$):
$$E[0] = 1+3+4+2 = 10, \quad E[1] = 1-3i-4+2i = -3-i$$
$$E[2] = 1-3+4-2 = 0, \quad E[3] = 1+3i-4-2i = -3+i$$

**Step 4:** 4-point DFT of odd subsequence:
$$O[0] = 2+4+3+1 = 10, \quad O[1] = 2-4i-3+i = -1-3i$$
$$O[2] = 2-4+3-1 = 0, \quad O[3] = 2+4i-3-i = -1+3i$$

**Step 5:** Combine with twiddle factors $\omega_8^k = e^{-i\pi k/4}$:
$$X[0] = E[0] + \omega_8^0 O[0] = 10 + 10 = 20$$
$$X[1] = E[1] + \omega_8^1 O[1] = (-3-i) + \frac{1-i}{\sqrt{2}}(-1-3i)$$

Computing $\omega_8^1(-1-3i) = \frac{-1-3i-i+3i^2}{\sqrt{2}} \cdot (-1) = \frac{(1+3i)(1-i)}{\sqrt{2}} \cdot (-1)$... the exact arithmetic is tedious; numerically: $X = (20, -5.657, 0, -0.828, 0, -0.828, 0, -5.657)$.

**Verification:** $X[0] = \sum_n x[n] = 1+2+3+4+4+3+2+1 = 20$ PASS. Parseval: $\sum_n x[n]^2 = 1+4+9+16+16+9+4+1 = 60$; $\frac{1}{8}\sum_k |X[k]|^2 = \frac{1}{8}(400 + 32 + 0 + 0.686 + 0 + 0.686 + 0 + 32) \approx 60$ PASS.

### C.2 Leakage Quantification: Rectangular vs Hann

Signal: $x(t) = \cos(2\pi \cdot 3.5 t)$ sampled at $f_s = 1$ Hz for $N = 16$ samples. The frequency $f_0 = 3.5$ lies exactly halfway between bins 3 and 4.

**Rectangular window:** All 16 DFT values are nonzero. The two closest bins ($k=3$, $k=4$) have magnitude $\approx 7.1$ each (out of a maximum of 8 for a bin-aligned frequency). The farthest bins ($k=8$) have magnitude $\approx 0.5$. Leakage energy (all bins except 3, 4): $\approx 10\%$ of total.

**Hann window:** $w[n] = \frac{1}{2}(1 - \cos(2\pi n/15))$. The main lobe is now 4 bins wide (bins 2, 3, 4, 5 have significant energy). Leakage energy outside the main lobe: $< 0.1\%$ - a 100 reduction in far-field leakage.

**Trade-off:** the Hann window "blurs" the frequency axis (wider main lobe), but dramatically reduces interference from distant frequency components. For audio analysis with multiple simultaneous tones, the Hann window is almost always preferred.

### C.3 FFT for Image Compression (Preview)

A grayscale image $I[m, n]$ of size $256 \times 256$ has $2^{16}$ pixel values. The 2D DFT produces $2^{16}$ complex frequency coefficients.

**Low-frequency concentration:** natural images have most energy in low spatial frequencies - slowly varying regions (sky, smooth surfaces). Empirically, the top-left $16 \times 16$ corner of the DFT magnitude (the lowest 16 frequencies in each direction) typically captures $> 90\%$ of total image energy.

**Compression idea:** retain only $K \times K$ DFT coefficients. Setting all others to zero and applying the IDFT gives a low-frequency reconstruction. For $K = 32$: $1024$ retained coefficients out of $65536$ ($\approx 1.6\%$), with visually acceptable reconstruction for most images.

This is the conceptual predecessor of JPEG, which uses the **Discrete Cosine Transform (DCT)** - the real-valued cousin of the DFT - in $8 \times 8$ blocks, achieving better compression due to avoiding complex arithmetic and reducing boundary artifacts.

---

## Appendix D: Numerical Implementation Notes

### D.1 Choosing FFT Size for Maximum Speed

Different values of $N$ require dramatically different computation times even with the same asymptotic complexity. FFTW benchmarks (in rough order of efficiency):

| $N$ type | Relative speed |
|---|---|
| Powers of 2 ($2^k$) | Fastest |
| Products $2^a 3^b 5^c$ | Very fast |
| Smooth numbers (all prime factors $\leq 7$) | Fast |
| Prime $N$ (uses Bluestein) | Slow - avoid if possible |
| $N = 2p$ for large prime $p$ | Slow |

**Rule of thumb:** always zero-pad to a highly composite number. `scipy.fft.next_fast_len(N)` returns the smallest integer $\geq N$ that has only small prime factors.

### D.2 Avoiding Common Numerical Pitfalls

**Precision requirements:**
- `float32` (single precision): round-off error $\approx 10^{-7}$ per operation. For $N = 2^{20}$: accumulated FFT error $\approx 10^{-7} \times \log_2(2^{20}) \approx 2 \times 10^{-6}$.
- `float64` (double precision): error $\approx 10^{-16}$ per operation. FFT error for $N = 2^{20}$: $\approx 3 \times 10^{-15}$.

For audio processing (16-bit integer input -> 32-bit float): single precision is fine. For scientific computing or precision-sensitive applications: use double precision.

**Large DFT sizes and memory bandwidth:** for $N > 10^6$, the FFT becomes memory-bandwidth-limited rather than compute-limited. The working set (the data being accessed at each butterfly stage) exceeds the CPU cache, and cache misses dominate runtime. For very large FFTs: consider FFT algorithms with cache-oblivious access patterns (e.g., FFTW's recursive divide-and-conquer).

### D.3 GPU FFT Considerations

CUDA FFT (cuFFT) and ROCm FFT achieve peak performance for:
- Power-of-2 sizes on any GPU
- Batch FFTs (many independent FFTs of the same size): significantly faster than sequential single FFTs
- Real-to-complex (rfft) transforms on modern GPUs with Tensor Cores

**For ML training:** when using FNO or FlashFFTConv, always batch the FFT computation across the mini-batch and channel dimensions simultaneously. `torch.fft.fft(x, dim=-1)` can process a (batch, channels, N) tensor with a single cuFFT call, achieving near-peak GPU utilization.

---

## Appendix E: Connections to Other Areas

### E.1 DFT and Polynomial Multiplication

The DFT has an elegant algebraic interpretation: it converts polynomial multiplication in the ring $\mathbb{C}[x]/(x^N - 1)$ into pointwise multiplication of coefficient sequences.

**Fact.** Let $p(x) = \sum_{n=0}^{N-1} a_n x^n$ and $q(x) = \sum_{n=0}^{N-1} b_n x^n$. Their product modulo $x^N - 1$ is the circular convolution $c = a \circledast b$. The DFT converts this to $C[k] = A[k] \cdot B[k]$ (pointwise product). Thus: **DFT diagonalizes the cyclic convolution operator.**

More precisely, the matrix of circular convolution by $\mathbf{a}$ (the circulant matrix $C_a$) has eigenvectors equal to the Fourier basis vectors $\mathbf{f}_k$, with eigenvalues $A[k]$:

$$C_a \mathbf{f}_k = A[k] \mathbf{f}_k$$

The DFT simultaneously diagonalizes all circulant matrices. This is the algebraic foundation of the Convolution Theorem (Section 04).

### E.2 DFT and Number Theory: Quadratic Gauss Sums

The Gauss sum $G = \sum_{n=0}^{N-1} e^{2\pi i n^2/N}$ arises naturally in number theory and signal processing. Its magnitude satisfies $|G| = \sqrt{N}$, which is the basis for several FFT-based tricks.

Bluestein's algorithm uses the identity $nk = \frac{1}{2}[n^2 + k^2 - (n-k)^2]$ to rewrite the DFT as a convolution:

$$X[k] = e^{-i\pi k^2/N} \sum_{n=0}^{N-1} \left(x[n] e^{-i\pi n^2/N}\right) e^{i\pi(n-k)^2/N}$$

This is a convolution of $h[n] = x[n]e^{-i\pi n^2/N}$ with the chirp kernel $f[n] = e^{i\pi n^2/N}$, computable via FFT for any $N$.

### E.3 DFT and the Cooley-Tukey Prime Factor Algorithm

When $N = N_1 N_2$ with $\gcd(N_1, N_2) = 1$, the **Good-Thomas (Prime Factor) Algorithm** maps the 1D DFT to a 2D DFT via the Chinese Remainder Theorem (CRT). This gives a different algorithm from Cooley-Tukey that:
- Requires no twiddle-factor multiplications (only additions)
- Is advantageous when $N$ has many small coprime factors

The CRT mapping is $n \leftrightarrow (n_1, n_2)$ with $n = N_2 n_1 + N_1 n_2 \bmod N$. The DFT factors as:

$$X[k_1 N_1 + k_2 N_2 \bmod N] = \text{DFT}_{N_1} \otimes \text{DFT}_{N_2}$$

**Connection to representation theory:** the Good-Thomas algorithm expresses the DFT on a cyclic group $\mathbb{Z}_N$ in terms of DFTs on subgroups $\mathbb{Z}_{N_1}$ and $\mathbb{Z}_{N_2}$ when $N = N_1 N_2$ is coprime. The general theory of DFT on finite abelian groups is a special case of harmonic analysis on locally compact abelian groups (Pontryagin duality), which is developed in full in [Section 12 Functional Analysis](../../12-Functional-Analysis/README.md).


---

## Appendix F: DFT Properties - Detailed Proofs

### F.1 Time Reversal

If $y[n] = x[-n \bmod N]$, then $Y[k] = X[-k \bmod N] = X[N-k]$.

**Proof.**
$$Y[k] = \sum_{n=0}^{N-1} x[-n \bmod N]\,\omega_N^{-nk} \overset{m=-n \bmod N}{=} \sum_{m=0}^{N-1} x[m]\,\omega_N^{mk} = X[-k \bmod N]$$

**Consequence.** For real $\mathbf{x}$: $X[N-k] = X[-k] = X[k]^*$ (conjugate symmetry, Section 3.4). Time reversal in the time domain maps to frequency reversal (conjugation) in the frequency domain.

### F.2 Duality

If $Y[n] = X[n]$ (i.e., the time-domain signal equals the DFT of another signal), then:

$$\mathcal{F}\{X[n]\}[k] = N\, x[-k \bmod N]$$

**Proof.** From the IDFT: $x[n] = \frac{1}{N}\sum_k X[k]\omega_N^{nk}$. Replace $n \to -k$ and $k \to n$:

$$x[-k] = \frac{1}{N}\sum_{n=0}^{N-1} X[n]\omega_N^{-kn} = \frac{1}{N}\mathcal{F}\{X[n]\}[k]$$

Therefore $\mathcal{F}\{X[n]\}[k] = N\, x[-k \bmod N]$.

**Application.** If $\mathbf{x}$ is a rectangular pulse (1 for $n = 0, \ldots, M-1$, else 0), its DFT $X[k] = \frac{\sin(\pi kM/N)}{\sin(\pi k/N)}e^{-i\pi k(M-1)/N}$ is a Dirichlet kernel. By duality, if the input is a Dirichlet kernel, the DFT is (essentially) a rectangular pulse. This is the discrete analog of the rectangle-sinc FT pair.

### F.3 DFT of Real and Imaginary Parts

For a complex signal $z[n] = x[n] + iy[n]$ with real parts $x[n]$ and $y[n]$:

$$X[k] = \mathcal{F}\{x[n]\}[k] = \frac{Z[k] + Z[N-k]^*}{2}$$
$$Y[k] = \mathcal{F}\{y[n]\}[k] = \frac{Z[k] - Z[N-k]^*}{2i}$$

This allows computing two real DFTs in the time of one complex DFT - an important optimization for processing stereo audio or in-phase/quadrature (IQ) signals.

---

## Appendix G: Windowing Theory - Spectral Analysis Framework

### G.1 The Window Design Problem

Given a finite observation $x_w[n] = x[n] \cdot w[n]$ of an infinite signal $x[n]$, the DFT gives:

$$X_w[k] = \frac{1}{N}(X \circledast W)[k]$$

where $W[k] = \mathcal{F}\{w[n]\}[k]$ is the DFT of the window. The true spectrum $X[k]$ is "blurred" by $W[k]$.

**Ideal window properties (contradictory):**
1. $W[k] = N\delta[k]$ (no leakage - only possible for infinite observation)
2. $w[n] = 1$ for $n \in [0, N-1]$ (rectangular window: maximum time-domain energy)
3. Compact support in both time and frequency (impossible by uncertainty principle)

Any window must trade off between these competing desiderata. The Kaiser window with adjustable $\beta$ offers a continuous tradeoff between resolution and leakage control.

### G.2 Sidelobe Rolloff Rate

The rate at which sidelobes decrease away from the main lobe is called the **rolloff rate** (in dB per octave):

- **Rectangular window**: discontinuities at $n=0$ and $n=N-1$ -> $|W[k]| \sim 1/k$ -> rolloff $\approx -6\,\text{dB/octave}$
- **Hann window**: continuous at endpoints (value 0) but derivative discontinuity -> $|W[k]| \sim 1/k^2$ -> rolloff $\approx -18\,\text{dB/octave}$
- **Blackman window**: zero at endpoints with zero first and second derivatives -> $|W[k]| \sim 1/k^3$ -> rolloff $\approx -18\,\text{dB/octave}$ (but lower peak sidelobe)

**Generalization.** A window with $p-1$ continuous derivatives at the endpoints has sidelobe falloff $O(k^{-(p+1)})$, corresponding to a rolloff rate of $-6(p+1)\,\text{dB/octave}$. This is a direct consequence of the Riemann-Lebesgue lemma applied to the windowed Fourier series.

### G.3 Optimal Windows: Prolate Spheroidal Wave Functions

The **Slepian sequences** (discrete prolate spheroidal wave functions, DPSS) are the theoretically optimal windows in the following sense:

**Problem.** Among all windows $w[n]$ with $\sum_n w[n]^2 = 1$, find the one that maximizes the fraction of energy concentrated in $|\xi| \leq W$ (a specified bandwidth):

$$\text{maximize} \quad \frac{\sum_{|k| \leq KN} |W[k]|^2}{\sum_k |W[k]|^2}$$

The solution is the first Slepian sequence, which is the eigenvector corresponding to the largest eigenvalue of the matrix $A_{nm} = \frac{\sin(2\pi W(n-m))}{\pi(n-m)}$.

Slepian sequences achieve theoretically optimal concentration - the maximum possible energy fraction in a given bandwidth - and are used in **multitaper spectral estimation** (Section A.2).

---

## Appendix H: The FFT in Modern Hardware

### H.1 Vectorization and SIMD

Modern CPUs provide SIMD (Single Instruction Multiple Data) instructions that process multiple values simultaneously:
- SSE2: 2 double-precision complex numbers per operation
- AVX2: 4 double-precision complex numbers per operation
- AVX-512: 8 double-precision complex numbers per operation

FFTW automatically generates SIMD-optimized code for each detected instruction set. For $N = 2^{20}$ at double precision, AVX-512 provides approximately 8 speedup over scalar code.

### H.2 GPU FFT (cuFFT)

NVIDIA's cuFFT library achieves peak performance by:
1. **Tiling**: decomposing the DFT into tiles that fit in shared memory (SRAM)
2. **Warp-level butterfly**: mapping butterfly computations to warp-level shuffle operations
3. **Tensor Core utilization**: for batch FFTs, reformulating twiddle-factor multiplication as GEMM operations executable on Tensor Cores

Peak throughput: NVIDIA H100 achieves $\sim 30\,\text{TFLOPS}$ on batch FFTs of power-of-2 size with `float16` (half precision).

### H.3 Distributed FFT for Very Large Transforms

For transforms exceeding GPU memory (e.g., $N > 10^8$), the DFT can be decomposed across multiple GPUs using the row-column algorithm:

1. Distribute $N = N_1 \times N_2$ points across $P$ GPUs: each GPU holds $N/P$ rows
2. Each GPU applies 1D FFT to its rows ($N_2$-point FFTs)
3. All-to-all communication transposes the data (global shuffle)
4. Each GPU applies 1D FFT to its columns ($N_1$-point FFTs) and applies twiddle factors
5. Final all-to-all restores the layout

Total communication: $O(N/P)$ complex numbers per GPU per transpose step. For $P = 8$ GPUs and $N = 10^9$: $\sim 128\,\text{GB}$ data movement per step.

**Applications:** seismic data processing, climate modeling, and cosmological simulations routinely require distributed FFTs of $10^9$-$10^{12}$ points.

---

## Appendix I: DFT and Signal Reconstruction

### I.1 Bandlimited Interpolation

Given DFT coefficients $X[0], \ldots, X[K]$ (retaining only the lowest $K+1$ frequencies), the bandlimited interpolation of the corresponding signal to any intermediate time $t$ (not necessarily a sample point) is:

$$\hat{x}(t) = \frac{1}{N}\sum_{k=0}^{K} X[k]\, e^{2\pi i k t/N} + \text{conjugate terms}$$

This is the continuous analog of what FNO does (Section 8.2): it parameterizes a function by its low-frequency DFT coefficients, and evaluates it at any desired spatial resolution.

### I.2 Compressed Sensing and Sparse Recovery

**Problem (Compressed Sensing).** A signal $\mathbf{x} \in \mathbb{C}^N$ is $s$-sparse in the DFT domain: only $s \ll N$ Fourier coefficients are nonzero. Can we recover $\mathbf{x}$ from $m \ll N$ measurements?

**Answer (Candes, Romberg, Tao, 2006; Donoho, 2006).** If $m \geq C \cdot s \log(N/s)$ random measurements are taken, $\mathbf{x}$ can be exactly recovered (with high probability) by solving the $\ell^1$ minimization:

$$\min_{\hat{\mathbf{x}}} \lVert \hat{\mathbf{x}} \rVert_1 \quad \text{subject to} \quad A\hat{\mathbf{x}} = \mathbf{b}$$

where $A$ is the random measurement matrix. This is dramatically below the $m \geq N$ requirements of classical sampling.

**For AI:** compressed sensing underlies MRI reconstruction (6-8 speedup by acquiring 1/6 of $k$-space data), compressed attention (attending to sparse subsets of frequency components), and efficient LLM inference when activations are sparse in the Fourier domain. The theoretical guarantees connect directly to the RIP (Restricted Isometry Property) of random DFT matrices.

### I.3 Phase Retrieval

**Problem.** Given only $|X[k]|^2$ (the power spectrum, without phase), recover $\mathbf{x}$.

Phase retrieval is fundamentally ill-posed in 1D (any signal and its time-reversed, time-shifted version have the same power spectrum). In 2D (images), phase retrieval becomes uniquely solvable (generically) due to oversampling constraints.

**Algorithms:** Gerchberg-Saxton (1972), HIO (Fienup, 1982), and modern deep learning approaches (phase retrieval nets).

**For AI:** X-ray crystallography uses phase retrieval to determine protein structure. Modern protein structure prediction (AlphaFold 2, RoseTTAFold) sidesteps phase retrieval by directly predicting 3D coordinates from sequence and co-evolutionary information.


---

## Appendix J: Worked ML Examples

### J.1 Building a Mel Spectrogram from Scratch

This section works through the complete mathematics behind the mel spectrogram used in Whisper, step by step.

**Step 1: Mel scale.** The mel scale converts frequency $f$ (Hz) to perceived pitch $m$ (mels):

$$m(f) = 2595 \log_{10}\!\left(1 + \frac{f}{700}\right), \quad f(m) = 700\left(10^{m/2595} - 1\right)$$

The mel scale is approximately linear below 1 kHz (300 Hz approx 401 mel; 700 Hz approx 811 mel) and logarithmic above 1 kHz, matching the frequency resolution of the human auditory system (the cochlea's place-frequency mapping).

**Step 2: Filterbank design.** For $M = 80$ mel filters covering $[f_{\min}, f_{\max}] = [0, 8000]\,\text{Hz}$:

1. Convert the frequency range to mels: $[m_{\min}, m_{\max}] = [0, 3227]\,\text{mels}$
2. Place $M+2 = 82$ equally-spaced mel points: $m_0 < m_1 < \cdots < m_{81}$
3. Convert back to Hz: $f_k = f(m_k)$
4. For filterbank $m$ ($m = 1, \ldots, 80$), define the triangular filter:
   
$$H_m[k] = \begin{cases}
\dfrac{f_k - f_{m-1}}{f_m - f_{m-1}} & f_{m-1} \leq f_k \leq f_m \\
\dfrac{f_{m+1} - f_k}{f_{m+1} - f_m} & f_m \leq f_k \leq f_{m+1} \\
0 & \text{otherwise}
\end{cases}$$

where $f_k = k f_s / N$ are the DFT bin frequencies.

**Step 3: Power accumulation.** For each STFT frame $l$:

$$\text{mel\_spec}[m, l] = \sum_{k=0}^{N/2} H_m[k] \cdot |X[k, l]|^2$$

**Step 4: Log compression.**

$$\text{log\_mel}[m, l] = \log_{10}\!\left(\max\!\left(\text{mel\_spec}[m, l], 10^{-10}\right)\right)$$

The floor $10^{-10}$ prevents $\log(0)$. The Whisper normalization subtracts the mean of `log_mel` and divides by 4 (an empirically chosen scale factor that centers the values near $[-1, 1]$ for typical speech).

**Why these choices?**
- 80 mel bins: empirically optimal for large-scale ASR (more bins -> marginal improvement; fewer -> information loss)
- 0-8000 Hz: human speech information is concentrated here; extending to 8+ kHz adds noise without significant ASR benefit
- 25 ms window / 10 ms hop: matches the typical duration of speech phonemes (20-200 ms)
- Log compression: matches perceptual loudness; prevents high-amplitude phonemes from dominating the feature representation

### J.2 FNO Architecture: Complete Mathematical Description

The Fourier Neural Operator (Li et al., 2021) architecture for 1D problems:

**Input:** discretized function $\mathbf{a} \in \mathbb{R}^{N \times d_a}$ (input function on $N$ points with $d_a$ channels)
**Output:** discretized function $\mathbf{u} \in \mathbb{R}^{N \times d_u}$ (solution on $N$ points with $d_u$ channels)

**Architecture (4 FNO layers + output projection):**

```
Layer 0: Input lifting
  a  R^{N  d_a}  ->  v  R^{N  d_v}   (learnable MLP)

Layer 1-4: FNO blocks
  v <- FNO-Block(v)

Layer 5: Output projection
  v  R^{N  d_v}  ->  u  R^{N  d_u}   (learnable MLP)
```

**FNO Block:**

```
FNO-Block(v):
    1. Spectral branch:
       V = DFT(v)                  # shape: (N, d_v) complex
       V_trunc = V[:K, :]           # keep lowest K modes
       V_out = R @ V_trunc          # R  C^{K  d_v  d_v}: learned weight
       V_pad = [V_out; 0, ..., 0]   # zero-pad back to N modes
       v_spectral = IDFT(V_pad)     # shape: (N, d_v)
    
    2. Local branch:
       v_local = W @ v              # W  R^{d_v  d_v}: learned weight
    
    3. Combine:
       v_new = (v_spectral + v_local)  #  = GeLU
    
    return v_new
```

**Key design choices:**
- $K = 16$ retained modes: sufficient for smooth PDE solutions; higher $K$ is used for more complex solutions (turbulence: $K = 32$)
- $d_v = 64$ channels: comparable to modern transformer hidden dimension
- Weight sharing: the same $R$ is used across all spatial positions (translation equivariance in the frequency domain)
- Resolution-invariance: the FNO can be trained on $N = 64$ and evaluated on $N = 256$ by changing the DFT size; the spectral weights $R$ and local weights $W$ are independent of $N$

**Parameter count comparison (1D, $N = 1024$, $K = 16$, $d_v = 64$):**

| Layer | Dense global convolution | FNO spectral layer |
|---|---|---|
| Parameters | $N \times d_v^2 = 4.2 \times 10^6$ | $K \times d_v^2 \times 2 = 131,072$ (complex) |
| FLOPs | $O(N^2 d_v) = 4 \times 10^9$ | $O(N d_v \log N + K d_v^2) = 6.7 \times 10^6$ |

The FNO spectral layer has $32\times$ fewer parameters and $600\times$ fewer FLOPs per layer than a fully global convolution.

### J.3 Monarch Matrix Decomposition

A Monarch matrix of size $N = N_1 N_2$ is defined as:

$$M = P L P^\top R$$

where:
- $P \in \{0,1\}^{N \times N}$ is a fixed permutation (the "interleaving permutation")
- $L = \operatorname{diag}(L_0, L_1, \ldots, L_{N_1-1})$ is block-diagonal with $N_1$ blocks of size $N_2 \times N_2$
- $R = \operatorname{diag}(R_0, R_1, \ldots, R_{N_2-1})$ is block-diagonal with $N_2$ blocks of size $N_1 \times N_1$

**Connection to FFT.** The FFT butterfly matrix factors as:

$$F_N = (I_{N_1} \otimes F_{N_2}) \cdot D \cdot (F_{N_1} \otimes I_{N_2}) \cdot P_{\text{bit-reverse}}$$

where $D$ is a diagonal twiddle-factor matrix and $\otimes$ is the Kronecker product. Each butterfly stage is a block-diagonal matrix - exactly the Monarch structure. Thus the FFT is a special Monarch matrix with specific parameter values.

**Monarch parameterization:** instead of fixing the block entries to be DFT matrices, let $L_i$ and $R_j$ be learnable complex matrices. This gives $O(N(N_1 + N_2)) = O(N\sqrt{N})$ parameters (choosing $N_1 = N_2 = \sqrt{N}$).

**Expressiveness.** Monarch matrices can represent any circulant matrix (by choosing $L_i$ and $R_j$ appropriately) and therefore any convolution. They can also represent some attention patterns and are more expressive than diagonal matrices or individual FFT layers.

---

## Appendix K: Reference Summary

### K.1 Key Formulas

| Formula | Expression |
|---------|-----------|
| DFT | $X[k] = \sum_{n=0}^{N-1} x[n]\,\omega_N^{-nk}$, $\quad\omega_N = e^{2\pi i/N}$ |
| IDFT | $x[n] = \frac{1}{N}\sum_{k=0}^{N-1} X[k]\,\omega_N^{nk}$ |
| DFT Matrix | $(F_N)_{kn} = \omega_N^{-kn}$, $\quad F_N F_N^* = N I$ |
| Parseval | $\sum_n \lvert x[n]\rvert^2 = \frac{1}{N}\sum_k \lvert X[k]\rvert^2$ |
| Circular convolution | $(x \circledast y)[n] = \sum_{m=0}^{N-1} x[m]\,y[n-m\bmod N]$ |
| Circular shift | $x[n-m\bmod N] \leftrightarrow \omega_N^{-mk} X[k]$ |
| Conjugate symmetry | $x[n]\in\mathbb{R} \Rightarrow X[N-k] = X[k]^*$ |
| FFT complexity | $\frac{N}{2}\log_2 N$ complex multiplications, $N\log_2 N$ additions |
| Frequency resolution | $\Delta f = f_s/N = 1/T$ |
| Nyquist limit | $f_{\max} = f_s/2$ |
| STFT | $S[l,k] = \sum_{m=0}^{M-1} x[lH+m]\,w[m]\,\omega_N^{-mk}$ |
| Mel scale | $m = 2595\log_{10}(1 + f/700)$ |

### K.2 Quick Comparison: FT vs DFT

| Property | Continuous FT | DFT |
|---|---|---|
| Domain | $L^1(\mathbb{R})$ or $L^2(\mathbb{R})$ | $\mathbb{C}^N$ (finite sequences) |
| Output | Continuous spectrum on $\mathbb{R}$ | $N$ complex values |
| Inversion | Integral formula | Exact, finite sum |
| Shift | Linear: $e^{-2\pi i\xi\tau}$ phase factor | Circular: $\omega_N^{-mk}$ phase factor |
| Convolution | Linear convolution | Circular convolution |
| Parseval | $\int\lvert f\rvert^2 = \int\lvert\hat{f}\rvert^2$ | $\sum\lvert x\rvert^2 = \frac{1}{N}\sum\lvert X\rvert^2$ |
| Unitarity | FT on $L^2$: unitary with factor 1 | $\frac{1}{\sqrt{N}}F_N$ is unitary |
| Computation | Cannot compute exactly | FFT: $O(N\log N)$ |
| Leakage | No finite truncation | Leakage from windowing |


---

## Appendix L: Case Studies

### L.1 Whisper Large-v3: Preprocessing Bug Hunt

A common real-world bug in Whisper-based systems illustrates why understanding the DFT pipeline matters.

**Symptom:** transcription quality degrades for audio from a different microphone than training data, even though the content is identical.

**Root cause:** the new microphone captures at 44.1 kHz (not 16 kHz). Naive resampling using `librosa.load(file, sr=16000)` with default settings may introduce aliasing artifacts or produce a slightly different frequency axis. Specifically:

1. **Aliasing from incorrect anti-aliasing filter**: if the 44.1 kHz signal is downsampled by factor $\approx 2.76$ without applying a proper lowpass filter at 8 kHz (the Nyquist for 16 kHz), frequencies between 8 kHz and 22.05 kHz alias into the 0-8 kHz band, corrupting the mel spectrogram.

2. **Frequency axis mismatch**: if the mel filterbanks were designed for exactly 16 kHz sampling, resampling to 15,994 Hz (a common off-by-one error in resampling libraries) shifts all frequency bin centers by $\approx 0.04\%$ - too small to affect single-frequency detection but enough to corrupt the precise triangular filterbank overlaps.

**Fix:** explicitly verify sample rate after loading (`assert sr == 16000`), use `scipy.signal.resample_poly` with explicit anti-aliasing, and validate the mel filterbank output against a known reference signal.

**Lesson:** every stage of the FFT pipeline - sampling rate, window length, hop size, FFT size, filterbank design - has exact numerical requirements. The DFT is unforgiving of approximations.

### L.2 FNO for Climate Modeling: FourCastNet

FourCastNet (Pathak et al., 2022, NVIDIA) uses a variant of the FNO architecture to predict global weather at 6-hour intervals. It replaces the FFT-based spectral layer with a **Fourier-based AFNO (Adaptive Fourier Neural Operator)** block:

```
AFNO Layer:
  1. Patchify: divide input grid (7201440) into patches -> tokens
  2. DFT along spatial dimensions (2D FFT)
  3. Mix frequencies with learned complex weights (block-diagonal for efficiency)
  4. IDFT -> spatial domain
  5. MLP for channel mixing
```

**Key difference from FNO:** AFNO uses token mixing in frequency space with a block-diagonal structure, reducing parameters from $O(K^2 d^2)$ to $O(K d^2 / B)$ where $B$ is the block size.

**Results:**
- Forecasts 10-day global weather in 2 seconds (vs 4 hours for NWP models like ECMWF)
- Competitive with ECMWF operational forecasts on 2-week horizons for key variables (temperature at 500 hPa, wind at 850 hPa)
- Trained on 40 years of ERA5 reanalysis data (3.8 TB)

**Why frequency domain?** Atmospheric dynamics are dominated by large-scale planetary waves (low spatial frequencies) - the Rossby waves that govern jet streams and storm tracks. Truncating to $K = 32$ modes in each dimension captures $> 95\%$ of the variance in ERA5 data. Higher-frequency turbulence below the grid scale is parameterized separately.

### L.3 STFT Denoising in Production Audio

Modern audio denoising systems (used in Zoom, Teams, Discord) use STFT-based processing:

**Architecture:**
1. STFT with Hann window (1024 samples, hop 256) -> spectrogram $|X[l,k]|$ and phase $\angle X[l,k]$
2. Neural network (small RNN or transformer) estimates the noise mask $M[l,k] \in [0,1]$
3. Apply mask: $\hat{X}[l,k] = M[l,k] \cdot X[l,k]$ (keep speech, suppress noise)
4. Reconstruct waveform via ISTFT with original phase (overlap-add)

**Why preserve phase:** human perception is relatively insensitive to phase errors in speech (the McGurk effect), but phase-inconsistent reconstruction (Griffin-Lim iterations) introduces audible artifacts. Most commercial denoisers use the noisy signal's phase directly - a pragmatic approximation.

**Real-time constraint:** at 48 kHz with 1024-sample windows and 256-sample hop: one frame every 5.33 ms. A DNN processing one frame must complete in $< 5\,\text{ms}$ on the target hardware (often a small DSP or CPU). This limits the network size to $\sim 10^5$ parameters.

**The STFT as a learned feature extractor question:** in principle, one could learn the analysis window jointly with the denoising network (learnable STFT). In practice, fixed Hann windows almost always match or outperform learned windows on denoising benchmarks, because the Hann window's known good time-frequency resolution is hard to improve upon with a finite training set.

---

## Appendix M: Connections to Other Mathematical Areas

### M.1 Harmonic Analysis on Finite Abelian Groups

The DFT on $\mathbb{Z}_N$ (the cyclic group of integers modulo $N$) is a special case of harmonic analysis on locally compact abelian groups (LCA groups). For a finite abelian group $G$, the **Pontryagin dual** $\hat{G}$ is the group of characters (group homomorphisms $\chi: G \to S^1$). For $G = \mathbb{Z}_N$: $\hat{G} \cong \mathbb{Z}_N$ with characters $\chi_k(n) = \omega_N^{kn}$ - exactly the DFT basis.

The **abstract Fourier transform** on $G$ is:

$$\hat{f}(\chi) = \sum_{g \in G} f(g)\,\overline{\chi(g)}$$

For $G = \mathbb{Z}_N$: this recovers the DFT exactly. For $G = \mathbb{R}$: this recovers the continuous FT. The same theorem - Pontryagin duality, Plancherel's theorem, Poisson summation - holds in all cases.

This unification is developed fully in [Section 12 Functional Analysis](../../12-Functional-Analysis/README.md), connecting the DFT to representation theory of compact groups, wavelet theory, and the Fourier analysis of non-commutative groups (non-abelian harmonic analysis).

### M.2 Number-Theoretic Applications

**Quadratic reciprocity via DFT.** The Gauss sum $G(a, N) = \sum_{n=0}^{N-1} e^{2\pi i a n^2/N}$ plays a central role in the proof of quadratic reciprocity. For prime $p$:

$$G(1, p) = \begin{cases} \sqrt{p} & p \equiv 1 \pmod 4 \\ i\sqrt{p} & p \equiv 3 \pmod 4 \end{cases}$$

This is a DFT-like object: $G(a, N) = X[a]$ where $x[n] = 1$ for all $n$ and the kernel is $\omega_N^{an^2}$ instead of $\omega_N^{an}$.

**Integer factorization.** Shor's quantum algorithm (1994) for integer factorization relies on the **quantum Fourier transform (QFT)**: a quantum circuit that applies the DFT to a quantum superposition state in $O((\log N)^2)$ gate operations. Classical FFT requires $O(N \log N)$ operations; QFT requires $O((\log N)^2)$. Shor's algorithm uses the QFT to find the period of $a^k \bmod N$, which reveals the prime factors of $N$ in polynomial time.


---

## Appendix N: Spectral Analysis in Practice - Extended Examples

### N.1 Frequency Estimation: Sub-Bin Accuracy

A key practical problem: given noisy measurements $y[n] = A\cos(2\pi f_0 n/f_s + \phi) + \varepsilon[n]$, estimate $f_0$, $A$, and $\phi$ with sub-bin accuracy.

**Step 1: Coarse estimate.** Compute the DFT magnitude $|Y[k]|$ and find $k_{\max} = \arg\max_k |Y[k]|$. This gives $\hat{f}_0^{(1)} = k_{\max} f_s / N$ with error up to $\pm \Delta f/2 = \pm f_s/(2N)$.

**Step 2: Parabolic interpolation.** Fit a parabola to $|Y[k_{\max}-1]|$, $|Y[k_{\max}]|$, $|Y[k_{\max}+1]|$:

$$\hat{k} = k_{\max} + \frac{|Y[k_{\max}+1]| - |Y[k_{\max}-1]|}{2(2|Y[k_{\max}]| - |Y[k_{\max}-1]| - |Y[k_{\max}+1]|)}$$

This achieves approximately 10 better accuracy than the coarse estimate for SNR $> 20$ dB.

**Step 3: Quinn's Estimator.** For even better accuracy, use the complex DFT values directly:

$$\hat{k} = k_{\max} + \operatorname{Re}\!\left[\frac{Y[k_{\max}+1]/Y[k_{\max}]}{1 - Y[k_{\max}+1]/Y[k_{\max}]}\right]$$

Quinn's estimator is unbiased and achieves near-Cramer-Rao bound accuracy.

**Step 4: Amplitude and phase.** Once $\hat{k}$ is estimated, the amplitude is $\hat{A} = 2|Y[\hat{k}]|/N$ (for a one-sided real spectrum) corrected for the window's coherent gain, and the phase is $\hat{\phi} = \angle Y[\hat{k}]$.

### N.2 Detecting Harmonics in Speech

Speech is approximately **quasi-periodic**: voiced sounds (vowels) have a fundamental frequency $F_0$ (pitch) and harmonics at $2F_0, 3F_0, \ldots$ The harmonic structure is visible in the DFT as a series of peaks at integer multiples of $F_0$.

**HPS (Harmonic Product Spectrum)** algorithm:
1. Compute the DFT magnitude $|X[k]|$
2. Form the product spectrum: $P[k] = \prod_{h=1}^{H} |X[hk]|$
3. Estimate $F_0 = f_s \cdot \arg\max_k P[k] / N$

The HPS effectively downsamples the spectrum by factors $1, 2, \ldots, H$ and takes the product - aligning the harmonic peaks while suppressing noise between them.

**In AI:** pitch estimation (fundamental frequency detection) is a key component of speech processing pipelines for voice synthesis (VALL-E, SoundStorm), emotion recognition, and singing voice synthesis. Modern deep-learning pitch detectors (CREPE, PYIN, PENN) outperform HPS but rely on spectral representations derived from the STFT.

### N.3 The Spectrogram in Transformer-Based Audio Models

The dominant paradigm for audio language models (AudioLM, VALL-E, MusicLM, Stable Audio) is:

```
Audio waveform
    v FFT -> Mel spectrogram (continuous representation)
    v VQ-VAE or EnCodec -> Discrete tokens
    v Language model (transformer) -> Next-token prediction
    v Decoder -> Reconstructed mel spectrogram
    v Vocoder (HiFi-GAN, WaveGrad) -> Audio waveform
```

The FFT and mel spectrogram serve as both the input representation for the encoder and the target representation for the decoder. The transformer operates entirely in the token space derived from the mel spectrogram.

**Why not train end-to-end on waveforms?** The waveform representation at 16-44 kHz requires sequence lengths of $10^4$-$10^5$ samples per second - far beyond current transformer context lengths. The mel spectrogram with 10 ms frame shift compresses by a factor of $\sim 160\times$ (16 kHz) while preserving the perceptually relevant structure. This compression is what makes transformer-based audio generation feasible.

---

## Appendix O: Summary of AI Model Connections

| AI System | DFT Role | Key Parameters |
|---|---|---|
| **OpenAI Whisper** | FFT -> mel spectrogram as input | $f_s=16$kHz, $M=400$, $H=160$, $N=512$, 80 mel bins |
| **FNet (Google)** | Full 2D DFT replaces attention | DFT along sequence AND embedding dims; no learned weights |
| **FNO (CalTech)** | Truncated spectral convolution | $K=16$-$32$ modes; global PDE operator learning |
| **FourCastNet (NVIDIA)** | AFNO block for weather prediction | 2D FFT on $720\times1440$ lat/lon grid; 73 variables |
| **FlashFFTConv** | Butterfly-structured long convolution | Monarch factorization; $O(N\log N)$ with Tensor Cores |
| **Monarch (Together AI)** | FFT butterfly as trainable transform | $O(N\sqrt{N})$ parameters; used as attention surrogate |
| **VALL-E** | Codec (EnCodec) uses STFT loss | Multi-resolution STFT loss for perceptual quality |
| **AudioLM** | Semantic/acoustic tokenization | EnCodec uses mel-scale frequency band processing |
| **CREPE (pitch)** | Spectrogram input to CNN | $n_{\text{fft}}=1024$, $h=10$ms; pitch detection on mel spec |
| **HiFi-GAN (vocoder)** | Multi-period discriminator | Multi-resolution STFT loss; discriminates real vs fake |
| **S4/Mamba** | Spectral view of state space models | SSM recurrence equivalent to causal convolution (FFT in training) |
| **RoPE (LLaMA)** | Frequency-domain position encoding | Complex rotation = modulation in DFT sense (Section 3.3) |


---

## Appendix P: Proofs and Derivations

### P.1 Derivation of Cooley-Tukey for General Radix

For $N = N_1 \cdot N_2$, write $n = N_2 n_1 + n_2$ and $k = N_1 k_2 + k_1$ with $n_1, k_1 \in [N_1]$, $n_2, k_2 \in [N_2]$:

$$X[k] = \sum_{n_1=0}^{N_1-1}\sum_{n_2=0}^{N_2-1} x[N_2 n_1 + n_2]\, \omega_N^{-(N_2 n_1 + n_2)(N_1 k_2 + k_1)}$$

Expanding the exponent: $-(N_2 n_1 + n_2)(N_1 k_2 + k_1) = -N_1 N_2 n_1 k_2 - N_2 n_1 k_1 - n_2 N_1 k_2 - n_2 k_1$

Since $\omega_N^{N_1 N_2} = 1$, the first term vanishes. Grouping:

$$X[N_1 k_2 + k_1] = \sum_{n_2=0}^{N_2-1} \omega_{N_2}^{-n_2 k_2} \left[\underbrace{\omega_N^{-n_2 k_1}}_{\text{twiddle}} \sum_{n_1=0}^{N_1-1} \omega_{N_1}^{-n_1 k_1} x[N_2 n_1 + n_2]\right]$$

The inner sum is an $N_1$-point DFT (for each fixed $n_2, k_1$), and the outer sum is an $N_2$-point DFT (for each fixed $k_1$). The twiddle factors $\omega_N^{-n_2 k_1}$ multiply between the two stages.

**Total operations:** $N_2$ DFTs of size $N_1$ + $N_1$ DFTs of size $N_2$ + $N_1 N_2$ twiddle multiplications = $O(N \log N)$ by the master recurrence.

### P.2 The Uncertainty Principle for the DFT (Donoho-Stark Theorem)

**Theorem (Donoho & Stark, 1989).** For any nonzero $\mathbf{x} \in \mathbb{C}^N$:

$$|\text{supp}(\mathbf{x})| \cdot |\text{supp}(\mathbf{X})| \geq N$$

where $\text{supp}(\mathbf{x}) = \{n : x[n] \neq 0\}$ is the support of $\mathbf{x}$.

**Proof sketch.** Suppose $|\text{supp}(\mathbf{x})| = S$ and $|\text{supp}(\mathbf{X})| = F$. The DFT restricted to $\text{supp}(\mathbf{x})$ and $\text{supp}(\mathbf{X})$ defines an $F \times S$ submatrix $A$ of $\frac{1}{\sqrt{N}}F_N$. Since $\mathbf{x}$ is supported on $S$ points and $\mathbf{X}$ is supported on $F$ points, this submatrix satisfies $A \mathbf{x}_{|S} = \mathbf{X}_{|F}/\sqrt{N}$. A counting argument using the rank of $A$ gives $SF \geq N$.

**Corollary.** A signal cannot have fewer than $\sqrt{N}$ nonzero samples AND fewer than $\sqrt{N}$ nonzero DFT coefficients simultaneously. This is the discrete analog of the Heisenberg uncertainty principle.

**For AI:** compressed sensing exploits this theorem: a signal supported on $S$ DFT bins can be recovered from $m \geq CS\log N$ time-domain measurements - far fewer than $N$. This justifies FNO's use of $K \ll N$ Fourier modes: smooth solutions to PDEs have small Fourier support.

### P.3 Aliasing Formula: Proof via Poisson Summation

When $x(t)$ is sampled at rate $f_s$ (below the Nyquist rate for bandlimited $x$), the DFT of the samples is:

$$X_s[k] = \sum_{n=-\infty}^{\infty} \hat{x}\!\left(\frac{k}{N T_s} - \frac{n}{T_s}\right) = \frac{1}{T_s}\sum_{n=-\infty}^{\infty} \hat{x}(k\Delta f - n f_s)$$

This is the **aliased spectrum**: the true spectrum $\hat{x}$ summed over all frequency shifts by integer multiples of $f_s$. When $\hat{x}$ is nonzero outside $[-f_s/2, f_s/2)$, adjacent copies overlap and contaminate each other - this is aliasing.

**Proof.** Starting from the Poisson summation formula (Section 02-7.5):

$$\sum_{n=-\infty}^{\infty} f(t - nT) = \frac{1}{T}\sum_{k=-\infty}^{\infty} \hat{f}(k/T)\, e^{2\pi i k t/T}$$

Setting $f(t) = x(t) e^{-2\pi i \xi t}$ and $T = 1/f_s$, and evaluating at $t=0$:

$$\sum_n x(n/f_s)\, e^{-2\pi i\xi n/f_s} = f_s \sum_k \hat{x}(\xi - kf_s)$$

The left side is the DTFT of the sampled signal; the right side is the periodized spectrum. The DFT is the sampling of this DTFT at $\xi = k f_s/N$, giving the result.


---

## Appendix Q: Further Reading

### Primary References

1. **Cooley, J.W. and Tukey, J.W.** (1965). "An Algorithm for the Machine Calculation of Complex Fourier Series." *Mathematics of Computation*, 19(90), 297-301. - The original FFT paper; two pages, historically indispensable.

2. **Oppenheim, A.V. and Schafer, R.W.** (2010). *Discrete-Time Signal Processing* (3rd ed.). Prentice Hall. - The standard reference for DSP; chapters 8-10 cover DFT, FFT, and windowing comprehensively.

3. **Strang, G.** (1986). "The Discrete Cosine Transform." *SIAM Review*, 41(1), 135-147. - Beautiful exposition connecting FFT, DCT, and linear algebra.

4. **Frigo, M. and Johnson, S.G.** (2005). "The Design and Implementation of FFTW3." *Proceedings of the IEEE*, 93(2), 216-231. - FFTW's adaptive algorithm selection strategy.

5. **Li, Z. et al.** (2021). "Fourier Neural Operator for Parametric Partial Differential Equations." *ICLR 2021*. - FNO architecture and theory.

6. **Dao, T. et al.** (2022). "Monarch: Expressive Structured Matrices for Efficient and Accurate Training." *ICML 2022*. - Butterfly matrix parameterization.

7. **Radford, A. et al.** (2022). "Robust Speech Recognition via Large-Scale Weak Supervision." *ICML 2023*. - Whisper architecture and mel spectrogram preprocessing.

8. **Donoho, D.L. and Stark, P.B.** (1989). "Uncertainty Principles and Signal Recovery." *SIAM Journal on Applied Mathematics*, 49(3), 906-931. - Discrete uncertainty principle.

### Online Resources

- NumPy FFT documentation: `numpy.fft` module reference with complete API
- SciPy signal processing: `scipy.signal.spectrogram`, `scipy.signal.welch`, `scipy.fft`
- Whisper preprocessing code: `openai/whisper` GitHub repository, `audio.py`
- FNO implementation: `neuraloperator/neuraloperator` GitHub repository


---

## Appendix R: Quick-Reference Formula Sheet

| Symbol | Meaning | Value / Formula |
|--------|---------|-----------------|
| $\omega_N$ | Principal $N$-th root of unity | $e^{2\pi i/N}$ |
| $X[k]$ | $k$-th DFT coefficient | $\sum_{n=0}^{N-1} x[n]\,\omega_N^{-nk}$ |
| $x[n]$ | IDFT reconstruction | $\frac{1}{N}\sum_{k=0}^{N-1} X[k]\,\omega_N^{nk}$ |
| $F_N$ | DFT matrix | $(F_N)_{kn} = \omega_N^{-kn}$, so $F_N F_N^* = NI$ |
| $\Delta f$ | Frequency bin width | $f_s/N = 1/T$ |
| $f_{\text{Nyq}}$ | Nyquist frequency | $f_s/2$ |
| $N_{\min}$ | Minimum $N$ to resolve $\Delta f_0$ | $\lceil f_s/\Delta f_0 \rceil$ |
| $S[l,k]$ | STFT coefficient | $\sum_{m} x[lH+m]\,w[m]\,\omega_N^{-mk}$ |
| COLA | Overlap-add constant | $\sum_l w[n-lH] = C$ for all $n$ |
| $X[N-k]$ | Conjugate symmetry (real $x$) | $X[k]^*$ |

### Window cheat-sheet

| Window | Main lobe (bins) | First sidelobe (dB) | Use case |
|--------|-----------------|---------------------|----------|
| Rectangular | 2 | $-13$ | Broadband noise, transients |
| Hann | 4 | $-31.5$ | General purpose |
| Hamming | 4 | $-42.7$ | Speech processing |
| Blackman | 6 | $-58$ | Weak sinusoids near strong ones |
| Kaiser $\beta=8.6$ | variable | $-69$ | Precision filter design |

### FFT complexity

$$T(N) = 2T(N/2) + O(N) \implies T(N) = O(N \log_2 N)$$

Compared to naive DFT: $N = 10^6$ reduces from $10^{12}$ to $2 \times 10^7$ operations - a $50\,000\times$ speedup.

### Key identities

$$x \circledast y \;\overset{\mathcal{F}}{\longleftrightarrow}\; X[k] \cdot Y[k] \qquad \text{(see Section 04)}$$

$$\sum_{n=0}^{N-1}|x[n]|^2 = \frac{1}{N}\sum_{k=0}^{N-1}|X[k]|^2 \qquad \text{(Parseval)}$$

$$X[k] = X[k+N] \qquad \text{(periodicity in frequency)}$$

$$x[n] = x[n+N] \qquad \text{(periodicity in time, implicit)}$$
