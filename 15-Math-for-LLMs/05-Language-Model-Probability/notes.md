[<- Positional Encodings](../04-Positional-Encodings/notes.md) | [Home](../../README.md) | [Training at Scale ->](../06-Training-at-Scale/notes.md)

---

# Language Model Probability Math

An LLM is a machine for assigning probabilities to token continuations. The transformer stack builds a contextual vector; this section explains how that vector becomes a normalized next-token distribution, how training improves that distribution, and how decoding turns it back into text.

## Overview

Language-model probability is the center of the LLM learning loop. During training, the model receives a true prefix and is penalized when the observed next token has low probability. During evaluation, the same probabilities produce negative log-likelihood, cross-entropy, perplexity, calibration curves, and answer scores. During generation, the probabilities are transformed by decoding rules such as temperature, top-k, top-p, beam search, or constraints.

The most important idea is simple but deep:

$$
P(t_{1:n}) = \\prod_{i=1}^{n} P(t_i \\mid t_{<i}).
$$

This chain-rule factorization is exact. Autoregressive language modeling becomes a practical learning problem because a neural network can estimate each conditional distribution.

## Prerequisites

- Conditional probability and the product rule
- Logarithms, exponentials, and gradients
- Entropy, KL divergence, and cross-entropy
- Vector and matrix shapes from embeddings and attention
- The previous LLM sections on tokenization, embeddings, attention, and positions

## Companion Notebooks

| Notebook | Purpose |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Builds the probability machinery with executable toy LMs, stable softmax, cross-entropy gradients, perplexity, calibration, and decoding demos. |
| [exercises.ipynb](exercises.ipynb) | Gives ten practice problems with scaffolds and complete solutions for sequence probability, loss, decoding, calibration, and conditional scoring. |

## Learning Objectives

After this section, you should be able to:

- Define a language model as a distribution over token sequences with an EOS convention.
- Derive the autoregressive factorization from the probability chain rule.
- Convert hidden states into logits, logits into probabilities, and probabilities into log-likelihoods.
- Implement numerically stable softmax and log-softmax.
- Derive the cross-entropy gradient $p-y$ with respect to logits.
- Compute entropy, KL divergence, cross-entropy, perplexity, bits per token, and length-normalized scores.
- Explain why decoding changes the sampling distribution without changing the trained model.
- Diagnose common probability bugs in LLM training and evaluation code.

## Table of Contents

1. [Language Models as Distributions](#1-language-models-as-distributions)
   - 1.1 [Finite vocabulary and EOS](#11-finite-vocabulary-and-eos)
   - 1.2 [Probability over finite strings](#12-probability-over-finite-strings)
   - 1.3 [Conditional next-token distributions](#13-conditional-nexttoken-distributions)
   - 1.4 [Support and impossible events](#14-support-and-impossible-events)
   - 1.5 [Why next-token prediction is enough](#15-why-nexttoken-prediction-is-enough)
2. [Autoregressive Factorization](#2-autoregressive-factorization)
   - 2.1 [Chain rule derivation](#21-chain-rule-derivation)
   - 2.2 [Teacher forcing](#22-teacher-forcing)
   - 2.3 [Causal masking](#23-causal-masking)
   - 2.4 [Sequence log probability](#24-sequence-log-probability)
   - 2.5 [Prefix scoring](#25-prefix-scoring)
3. [Logits, Softmax, and Log-Sum-Exp](#3-logits-softmax-and-logsumexp)
   - 3.1 [LM head](#31-lm-head)
   - 3.2 [Softmax normalization](#32-softmax-normalization)
   - 3.3 [Shift invariance](#33-shift-invariance)
   - 3.4 [Stable log-softmax](#34-stable-logsoftmax)
   - 3.5 [Temperature](#35-temperature)
4. [Maximum Likelihood and Cross-Entropy](#4-maximum-likelihood-and-crossentropy)
   - 4.1 [Dataset likelihood](#41-dataset-likelihood)
   - 4.2 [Negative log-likelihood](#42-negative-loglikelihood)
   - 4.3 [Cross-entropy](#43-crossentropy)
   - 4.4 [Gradient with respect to logits](#44-gradient-with-respect-to-logits)
   - 4.5 [Padding masks](#45-padding-masks)
5. [Entropy, KL, Perplexity, and Bits](#5-entropy-kl-perplexity-and-bits)
   - 5.1 [Entropy](#51-entropy)
   - 5.2 [Cross-entropy decomposition](#52-crossentropy-decomposition)
   - 5.3 [Perplexity](#53-perplexity)
   - 5.4 [Bits per token](#54-bits-per-token)
   - 5.5 [Tokenizer caveat](#55-tokenizer-caveat)
6. [Sequence Scoring and Calibration](#6-sequence-scoring-and-calibration)
   - 6.1 [Length bias](#61-length-bias)
   - 6.2 [Average log probability](#62-average-log-probability)
   - 6.3 [Calibration](#63-calibration)
   - 6.4 [Expected calibration error](#64-expected-calibration-error)
   - 6.5 [Temperature scaling](#65-temperature-scaling)
7. [Decoding as Distribution Transformation](#7-decoding-as-distribution-transformation)
   - 7.1 [Greedy decoding](#71-greedy-decoding)
   - 7.2 [Sampling](#72-sampling)
   - 7.3 [Top-k filtering](#73-topk-filtering)
   - 7.4 [Nucleus top-p filtering](#74-nucleus-topp-filtering)
   - 7.5 [Beam search and length penalty](#75-beam-search-and-length-penalty)
8. [From Count Models to Neural LMs](#8-from-count-models-to-neural-lms)
   - 8.1 [N-gram models](#81-ngram-models)
   - 8.2 [Smoothing](#82-smoothing)
   - 8.3 [Neural probabilistic LM](#83-neural-probabilistic-lm)
   - 8.4 [Recurrent LM](#84-recurrent-lm)
   - 8.5 [Transformer LM](#85-transformer-lm)
9. [Conditional Generation and Constraints](#9-conditional-generation-and-constraints)
   - 9.1 [Prompt conditioning](#91-prompt-conditioning)
   - 9.2 [RAG conditioning](#92-rag-conditioning)
   - 9.3 [Logit bias and constraints](#93-logit-bias-and-constraints)
   - 9.4 [Guidance by logit mixing](#94-guidance-by-logit-mixing)
   - 9.5 [Safety filters](#95-safety-filters)
10. [Diagnostics and Learning Practice](#10-diagnostics-and-learning-practice)
   - 10.1 [Normalization checks](#101-normalization-checks)
   - 10.2 [Likelihood sanity checks](#102-likelihood-sanity-checks)
   - 10.3 [Distribution shift](#103-distribution-shift)
   - 10.4 [Generation versus evaluation](#104-generation-versus-evaluation)
   - 10.5 [Bridge to training at scale](#105-bridge-to-training-at-scale)

---

## Notation

| Symbol | Meaning |
| --- | --- |
| $V$ | Vocabulary of ordinary tokens |
| $V_\\mathrm{ext}$ | Vocabulary plus special tokens such as EOS |
| $t_i$ | Token at position $i$ |
| $t_{<i}$ | Prefix before position $i$ |
| $h_i$ | Transformer hidden state at position $i$ |
| $z_i$ | Logit vector for the next-token distribution |
| $p_\\theta(\\cdot \\mid t_{<i})$ | Model distribution over the next token |
| $y_i$ | One-hot target vector |
| $m_i$ | Mask indicating whether a token contributes to loss |

The standard decoder-only training path is:

```text
tokens -> embeddings + positions -> causal transformer -> hidden state -> LM head -> logits -> log-softmax -> NLL
```

## 1. Language Models as Distributions

This part studies language models as distributions as an operational object: something an LLM computes for every prefix during training, scoring, and generation. The formulas below are small enough to derive by hand, but they are also exactly the formulas used inside large decoder-only models.

| Subtopic | Core question | Formula |
| --- | --- | --- |
| [Finite vocabulary and EOS](#1-finite-vocabulary-and-eos) | make sequence probability normalize by ending strings explicitly | $V_\mathrm{ext}=V\cup\{\mathrm{EOS}\}$ |
| [Probability over finite strings](#1-probability-over-finite-strings) | a language model assigns mass to every complete token string | $P(t_{1:n},\mathrm{EOS})$ |
| [Conditional next-token distributions](#1-conditional-nexttoken-distributions) | generation uses one normalized categorical distribution at each prefix | $p_\theta(t_{i}\mid t_{<i})$ |
| [Support and impossible events](#1-support-and-impossible-events) | zero probability is a dangerous modeling choice | $p_\theta(v\mid h)>0$ after softmax |
| [Why next-token prediction is enough](#1-why-nexttoken-prediction-is-enough) | the chain rule converts local predictions into a full joint model | $P(t_{1:n})=\prod_i P(t_i\mid t_{<i})$ |

### 1.1 Finite vocabulary and EOS

**Main idea.** Make sequence probability normalize by ending strings explicitly.

The useful formula is:

$$V_\mathrm{ext}=V\cup\{\mathrm{EOS}\}$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 1.2 Probability over finite strings

**Main idea.** A language model assigns mass to every complete token string.

The useful formula is:

$$P(t_{1:n},\mathrm{EOS})$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 1.3 Conditional next-token distributions

**Main idea.** Generation uses one normalized categorical distribution at each prefix.

The useful formula is:

$$p_\theta(t_{i}\mid t_{<i})$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 1.4 Support and impossible events

**Main idea.** Zero probability is a dangerous modeling choice.

The useful formula is:

$$p_\theta(v\mid h)>0$ after softmax$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 1.5 Why next-token prediction is enough

**Main idea.** The chain rule converts local predictions into a full joint model.

The useful formula is:

$$P(t_{1:n})=\prod_i P(t_i\mid t_{<i})$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
## 2. Autoregressive Factorization

This part studies autoregressive factorization as an operational object: something an LLM computes for every prefix during training, scoring, and generation. The formulas below are small enough to derive by hand, but they are also exactly the formulas used inside large decoder-only models.

| Subtopic | Core question | Formula |
| --- | --- | --- |
| [Chain rule derivation](#2-chain-rule-derivation) | no independence assumption is needed | $P(t_{1:n})=P(t_1)\prod_{i=2}^n P(t_i\mid t_{1:i-1})$ |
| [Teacher forcing](#2-teacher-forcing) | training conditions on the true prefix, not sampled model history | $-\log p_\theta(t_i^\star\mid t_{<i}^\star)$ |
| [Causal masking](#2-causal-masking) | attention must not leak future targets into the conditional | $M_{ij}=0$ for $j\le i$ and $-\infty$ otherwise |
| [Sequence log probability](#2-sequence-log-probability) | products become sums for stable scoring | $\log P(t_{1:n})=\sum_i \log p_\theta(t_i\mid t_{<i})$ |
| [Prefix scoring](#2-prefix-scoring) | prompt likelihood and answer likelihood are different objects | $\log p_\theta(y\mid x)=\sum_j \log p_\theta(y_j\mid x,y_{<j})$ |

### 2.1 Chain rule derivation

**Main idea.** No independence assumption is needed.

The useful formula is:

$$P(t_{1:n})=P(t_1)\prod_{i=2}^n P(t_i\mid t_{1:i-1})$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 2.2 Teacher forcing

**Main idea.** Training conditions on the true prefix, not sampled model history.

The useful formula is:

$$-\log p_\theta(t_i^\star\mid t_{<i}^\star)$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** This is why training can be massively parallel while generation is sequential.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 2.3 Causal masking

**Main idea.** Attention must not leak future targets into the conditional.

The useful formula is:

$$M_{ij}=0$ for $j\le i$ and $-\infty$ otherwise$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 2.4 Sequence log probability

**Main idea.** Products become sums for stable scoring.

The useful formula is:

$$\log P(t_{1:n})=\sum_i \log p_\theta(t_i\mid t_{<i})$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 2.5 Prefix scoring

**Main idea.** Prompt likelihood and answer likelihood are different objects.

The useful formula is:

$$\log p_\theta(y\mid x)=\sum_j \log p_\theta(y_j\mid x,y_{<j})$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
## 3. Logits, Softmax, and Log-Sum-Exp

This part studies logits, softmax, and log-sum-exp as an operational object: something an LLM computes for every prefix during training, scoring, and generation. The formulas below are small enough to derive by hand, but they are also exactly the formulas used inside large decoder-only models.

| Subtopic | Core question | Formula |
| --- | --- | --- |
| [LM head](#3-lm-head) | the hidden state becomes one score per vocabulary token | $z=Wh+b$ |
| [Softmax normalization](#3-softmax-normalization) | logits are unnormalized log probabilities | $p_i=\exp(z_i)/\sum_j\exp(z_j)$ |
| [Shift invariance](#3-shift-invariance) | adding a constant to every logit changes no probability | $\mathrm{softmax}(z+c\mathbf{1})=\mathrm{softmax}(z)$ |
| [Stable log-softmax](#3-stable-logsoftmax) | subtract the maximum before exponentiating | $\log p_i=z_i-\operatorname{LSE}(z)$ |
| [Temperature](#3-temperature) | divide logits before softmax to control entropy | $p_i(\tau)=\mathrm{softmax}(z_i/\tau)$ |

### 3.1 LM head

**Main idea.** The hidden state becomes one score per vocabulary token.

The useful formula is:

$$z=Wh+b$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 3.2 Softmax normalization

**Main idea.** Logits are unnormalized log probabilities.

The useful formula is:

$$p_i=\exp(z_i)/\sum_j\exp(z_j)$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** This is the final probability gate between hidden-state geometry and token choice.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 3.3 Shift invariance

**Main idea.** Adding a constant to every logit changes no probability.

The useful formula is:

$$\mathrm{softmax}(z+c\mathbf{1})=\mathrm{softmax}(z)$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 3.4 Stable log-softmax

**Main idea.** Subtract the maximum before exponentiating.

The useful formula is:

$$\log p_i=z_i-\operatorname{LSE}(z)$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 3.5 Temperature

**Main idea.** Divide logits before softmax to control entropy.

The useful formula is:

$$p_i(\tau)=\mathrm{softmax}(z_i/\tau)$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
## 4. Maximum Likelihood and Cross-Entropy

This part studies maximum likelihood and cross-entropy as an operational object: something an LLM computes for every prefix during training, scoring, and generation. The formulas below are small enough to derive by hand, but they are also exactly the formulas used inside large decoder-only models.

| Subtopic | Core question | Formula |
| --- | --- | --- |
| [Dataset likelihood](#4-dataset-likelihood) | training maximizes probability assigned to observed tokens | $\max_\theta \sum_{(x,y)}\log p_\theta(y\mid x)$ |
| [Negative log-likelihood](#4-negative-loglikelihood) | loss is surprise under the model | $\ell=-\log p_\theta(y^\star\mid h)$ |
| [Cross-entropy](#4-crossentropy) | one-hot labels reduce cross-entropy to NLL | $H(q,p_\theta)=-\sum_i q_i\log p_{\theta,i}$ |
| [Gradient with respect to logits](#4-gradient-with-respect-to-logits) | the key derivative is predicted minus target | $\nabla_z \ell=p-y$ |
| [Padding masks](#4-padding-masks) | only real target tokens should contribute to the average loss | $L=\sum_i m_i\ell_i/\sum_i m_i$ |

### 4.1 Dataset likelihood

**Main idea.** Training maximizes probability assigned to observed tokens.

The useful formula is:

$$\max_\theta \sum_{(x,y)}\log p_\theta(y\mid x)$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 4.2 Negative log-likelihood

**Main idea.** Loss is surprise under the model.

The useful formula is:

$$\ell=-\log p_\theta(y^\star\mid h)$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 4.3 Cross-entropy

**Main idea.** One-hot labels reduce cross-entropy to nll.

The useful formula is:

$$H(q,p_\theta)=-\sum_i q_i\log p_{\theta,i}$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 4.4 Gradient with respect to logits

**Main idea.** The key derivative is predicted minus target.

The useful formula is:

$$\nabla_z \ell=p-y$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** This derivative is the reason cross-entropy is so convenient for transformer training.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 4.5 Padding masks

**Main idea.** Only real target tokens should contribute to the average loss.

The useful formula is:

$$L=\sum_i m_i\ell_i/\sum_i m_i$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
## 5. Entropy, KL, Perplexity, and Bits

This part studies entropy, kl, perplexity, and bits as an operational object: something an LLM computes for every prefix during training, scoring, and generation. The formulas below are small enough to derive by hand, but they are also exactly the formulas used inside large decoder-only models.

| Subtopic | Core question | Formula |
| --- | --- | --- |
| [Entropy](#5-entropy) | intrinsic uncertainty of a distribution | $H(p)=-\sum_x p(x)\log p(x)$ |
| [Cross-entropy decomposition](#5-crossentropy-decomposition) | model loss equals data entropy plus mismatch | $H(q,p)=H(q)+D_\mathrm{KL}(q\Vert p)$ |
| [Perplexity](#5-perplexity) | exponentiated average NLL | $\mathrm{PPL}=\exp\left(\frac{1}{N}\sum_i-\log p_i\right)$ |
| [Bits per token](#5-bits-per-token) | use base-2 logs for compression interpretation | $\mathrm{BPT}=\mathrm{NLL}_{\mathrm{nat}}/\log 2$ |
| [Tokenizer caveat](#5-tokenizer-caveat) | perplexity depends on the tokenization unit | $\mathrm{BPB}$ and $\mathrm{BPC}$ are more comparable |

### 5.1 Entropy

**Main idea.** Intrinsic uncertainty of a distribution.

The useful formula is:

$$H(p)=-\sum_x p(x)\log p(x)$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 5.2 Cross-entropy decomposition

**Main idea.** Model loss equals data entropy plus mismatch.

The useful formula is:

$$H(q,p)=H(q)+D_\mathrm{KL}(q\Vert p)$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 5.3 Perplexity

**Main idea.** Exponentiated average nll.

The useful formula is:

$$\mathrm{PPL}=\exp\left(\frac{1}{N}\sum_i-\log p_i\right)$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** This is the most common intrinsic language-model score, but it must be interpreted with tokenizer and dataset context.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 5.4 Bits per token

**Main idea.** Use base-2 logs for compression interpretation.

The useful formula is:

$$\mathrm{BPT}=\mathrm{NLL}_{\mathrm{nat}}/\log 2$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 5.5 Tokenizer caveat

**Main idea.** Perplexity depends on the tokenization unit.

The useful formula is:

$$\mathrm{BPB}$ and $\mathrm{BPC}$ are more comparable$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
## 6. Sequence Scoring and Calibration

This part studies sequence scoring and calibration as an operational object: something an LLM computes for every prefix during training, scoring, and generation. The formulas below are small enough to derive by hand, but they are also exactly the formulas used inside large decoder-only models.

| Subtopic | Core question | Formula |
| --- | --- | --- |
| [Length bias](#6-length-bias) | raw log probability favors shorter strings | $\log p(y\mid x)$ decreases with length |
| [Average log probability](#6-average-log-probability) | length-normalized scores compare answers of different lengths | $\frac{1}{|y|}\sum_j \log p(y_j\mid x,y_{<j})$ |
| [Calibration](#6-calibration) | confidence should match empirical correctness | $P(\mathrm{correct}\mid \hat p=c)\approx c$ |
| [Expected calibration error](#6-expected-calibration-error) | bin confidence gaps to summarize miscalibration | $\mathrm{ECE}=\sum_b\frac{|B_b|}{n}|\mathrm{acc}(B_b)-\mathrm{conf}(B_b)|$ |
| [Temperature scaling](#6-temperature-scaling) | post-hoc calibration rescales logits without changing class order | $z'=z/\tau$ |

### 6.1 Length bias

**Main idea.** Raw log probability favors shorter strings.

The useful formula is:

$$\log p(y\mid x)$ decreases with length$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 6.2 Average log probability

**Main idea.** Length-normalized scores compare answers of different lengths.

The useful formula is:

$$\frac{1}{|y|}\sum_j \log p(y_j\mid x,y_{<j})$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 6.3 Calibration

**Main idea.** Confidence should match empirical correctness.

The useful formula is:

$$P(\mathrm{correct}\mid \hat p=c)\approx c$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 6.4 Expected calibration error

**Main idea.** Bin confidence gaps to summarize miscalibration.

The useful formula is:

$$\mathrm{ECE}=\sum_b\frac{|B_b|}{n}|\mathrm{acc}(B_b)-\mathrm{conf}(B_b)|$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 6.5 Temperature scaling

**Main idea.** Post-hoc calibration rescales logits without changing class order.

The useful formula is:

$$z'=z/\tau$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
## 7. Decoding as Distribution Transformation

This part studies decoding as distribution transformation as an operational object: something an LLM computes for every prefix during training, scoring, and generation. The formulas below are small enough to derive by hand, but they are also exactly the formulas used inside large decoder-only models.

| Subtopic | Core question | Formula |
| --- | --- | --- |
| [Greedy decoding](#7-greedy-decoding) | choose the highest-probability next token | $t_i=\arg\max_v p(v\mid t_{<i})$ |
| [Sampling](#7-sampling) | draw from the categorical distribution | $t_i\sim p_\theta(\cdot\mid t_{<i})$ |
| [Top-k filtering](#7-topk-filtering) | keep only the k highest probability tokens | $S_k=\operatorname{TopK}(p,k)$ |
| [Nucleus top-p filtering](#7-nucleus-topp-filtering) | keep the smallest set whose mass exceeds p | $\sum_{v\in S}p(v)\ge p_\mathrm{nuc}$ |
| [Beam search and length penalty](#7-beam-search-and-length-penalty) | approximate sequence MAP with multiple partial hypotheses | $s(y)=\log p(y\mid x)/|y|^\alpha$ |

### 7.1 Greedy decoding

**Main idea.** Choose the highest-probability next token.

The useful formula is:

$$t_i=\arg\max_v p(v\mid t_{<i})$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 7.2 Sampling

**Main idea.** Draw from the categorical distribution.

The useful formula is:

$$t_i\sim p_\theta(\cdot\mid t_{<i})$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 7.3 Top-k filtering

**Main idea.** Keep only the k highest probability tokens.

The useful formula is:

$$S_k=\operatorname{TopK}(p,k)$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 7.4 Nucleus top-p filtering

**Main idea.** Keep the smallest set whose mass exceeds p.

The useful formula is:

$$\sum_{v\in S}p(v)\ge p_\mathrm{nuc}$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** This is the default practical compromise between deterministic decoding and unrestricted sampling.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 7.5 Beam search and length penalty

**Main idea.** Approximate sequence map with multiple partial hypotheses.

The useful formula is:

$$s(y)=\log p(y\mid x)/|y|^\alpha$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
## 8. From Count Models to Neural LMs

This part studies from count models to neural lms as an operational object: something an LLM computes for every prefix during training, scoring, and generation. The formulas below are small enough to derive by hand, but they are also exactly the formulas used inside large decoder-only models.

| Subtopic | Core question | Formula |
| --- | --- | --- |
| [N-gram models](#8-ngram-models) | approximate history by the last n minus one tokens | $P(t_i\mid t_{<i})\approx P(t_i\mid t_{i-n+1:i-1})$ |
| [Smoothing](#8-smoothing) | reserve probability for unseen events | $P_\alpha(w\mid h)=\frac{c(h,w)+\alpha}{c(h)+\alpha |V|}$ |
| [Neural probabilistic LM](#8-neural-probabilistic-lm) | learn distributed word representations and probability together | $e_w\in\mathbb{R}^d$ |
| [Recurrent LM](#8-recurrent-lm) | compress history into a state | $h_i=f(h_{i-1},x_i)$ |
| [Transformer LM](#8-transformer-lm) | causal self-attention computes context-dependent hidden states in parallel | $p_\theta(t_i\mid t_{<i})=\mathrm{softmax}(Wh_i+b)$ |

### 8.1 N-gram models

**Main idea.** Approximate history by the last n minus one tokens.

The useful formula is:

$$P(t_i\mid t_{<i})\approx P(t_i\mid t_{i-n+1:i-1})$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 8.2 Smoothing

**Main idea.** Reserve probability for unseen events.

The useful formula is:

$$P_\alpha(w\mid h)=\frac{c(h,w)+\alpha}{c(h)+\alpha |V|}$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 8.3 Neural probabilistic LM

**Main idea.** Learn distributed word representations and probability together.

The useful formula is:

$$e_w\in\mathbb{R}^d$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 8.4 Recurrent LM

**Main idea.** Compress history into a state.

The useful formula is:

$$h_i=f(h_{i-1},x_i)$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 8.5 Transformer LM

**Main idea.** Causal self-attention computes context-dependent hidden states in parallel.

The useful formula is:

$$p_\theta(t_i\mid t_{<i})=\mathrm{softmax}(Wh_i+b)$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
## 9. Conditional Generation and Constraints

This part studies conditional generation and constraints as an operational object: something an LLM computes for every prefix during training, scoring, and generation. The formulas below are small enough to derive by hand, but they are also exactly the formulas used inside large decoder-only models.

| Subtopic | Core question | Formula |
| --- | --- | --- |
| [Prompt conditioning](#9-prompt-conditioning) | a prompt is the observed prefix in the conditional | $p_\theta(y\mid x)$ |
| [RAG conditioning](#9-rag-conditioning) | retrieved text changes the conditioning information | $p_\theta(y\mid x,r)$ |
| [Logit bias and constraints](#9-logit-bias-and-constraints) | decoding can restrict or reweight the support | $z'_v=z_v+\beta_v$ |
| [Guidance by logit mixing](#9-guidance-by-logit-mixing) | combine distributions in logit space when a control model exists | $z_\mathrm{guided}=z_\mathrm{base}+\lambda(z_\mathrm{cond}-z_\mathrm{base})$ |
| [Safety filters](#9-safety-filters) | post-processing policies are not the same as model probability | $\Pr(\mathrm{emit})$ may differ from $p_\theta$ |

### 9.1 Prompt conditioning

**Main idea.** A prompt is the observed prefix in the conditional.

The useful formula is:

$$p_\theta(y\mid x)$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 9.2 RAG conditioning

**Main idea.** Retrieved text changes the conditioning information.

The useful formula is:

$$p_\theta(y\mid x,r)$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** Retrieval does not change the probability rules; it changes what information the conditional can see.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 9.3 Logit bias and constraints

**Main idea.** Decoding can restrict or reweight the support.

The useful formula is:

$$z'_v=z_v+\beta_v$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 9.4 Guidance by logit mixing

**Main idea.** Combine distributions in logit space when a control model exists.

The useful formula is:

$$z_\mathrm{guided}=z_\mathrm{base}+\lambda(z_\mathrm{cond}-z_\mathrm{base})$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 9.5 Safety filters

**Main idea.** Post-processing policies are not the same as model probability.

The useful formula is:

$$\Pr(\mathrm{emit})$ may differ from $p_\theta$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
## 10. Diagnostics and Learning Practice

This part studies diagnostics and learning practice as an operational object: something an LLM computes for every prefix during training, scoring, and generation. The formulas below are small enough to derive by hand, but they are also exactly the formulas used inside large decoder-only models.

| Subtopic | Core question | Formula |
| --- | --- | --- |
| [Normalization checks](#10-normalization-checks) | probabilities must sum to one at each prefix | $\sum_v p(v\mid h)=1$ |
| [Likelihood sanity checks](#10-likelihood-sanity-checks) | known continuations should score above random continuations | $\log p(y_\mathrm{good}\mid x)>\log p(y_\mathrm{bad}\mid x)$ |
| [Distribution shift](#10-distribution-shift) | low likelihood can indicate unfamiliar domain or bad tokenization | $p_\mathrm{train}\ne p_\mathrm{test}$ |
| [Generation versus evaluation](#10-generation-versus-evaluation) | sampling quality and held-out NLL measure different properties | $\arg\min \mathrm{NLL}$ is not always best user experience |
| [Bridge to training at scale](#10-bridge-to-training-at-scale) | the same loss drives distributed training and scaling laws | $L(C,N,D)$ |

### 10.1 Normalization checks

**Main idea.** Probabilities must sum to one at each prefix.

The useful formula is:

$$\sum_v p(v\mid h)=1$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 10.2 Likelihood sanity checks

**Main idea.** Known continuations should score above random continuations.

The useful formula is:

$$\log p(y_\mathrm{good}\mid x)>\log p(y_\mathrm{bad}\mid x)$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 10.3 Distribution shift

**Main idea.** Low likelihood can indicate unfamiliar domain or bad tokenization.

The useful formula is:

$$p_\mathrm{train}\ne p_\mathrm{test}$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 10.4 Generation versus evaluation

**Main idea.** Sampling quality and held-out nll measure different properties.

The useful formula is:

$$\arg\min \mathrm{NLL}$ is not always best user experience$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 10.5 Bridge to training at scale

**Main idea.** The same loss drives distributed training and scaling laws.

The useful formula is:

$$L(C,N,D)$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.

---

## Practice Exercises

1. Factorize a three-token sentence probability using the chain rule.
2. Compute stable softmax probabilities from a logit vector by subtracting the maximum.
3. Derive the gradient of one-hot cross-entropy with respect to logits.
4. Compute a masked average loss for a padded batch.
5. Convert average NLL from nats to perplexity and bits per token.
6. Compare raw and length-normalized scores for two candidate answers.
7. Apply temperature, top-k, and top-p filtering to a toy distribution.
8. Compute expected calibration error for binned confidence values.
9. Score a conditional answer $p(y\\mid x)$ without including the prompt tokens in the average answer loss.
10. Write a short checklist for debugging an LM probability implementation.

## Why This Matters for AI

The probability layer is the contract between representation learning and text behavior. If logits are unstable, loss is masked incorrectly, probabilities are compared across incompatible tokenizations, or decoding filters are misunderstood, the model may appear to improve while the measured probability object is wrong. Strong LLM work requires comfort with this layer because it appears in pretraining, supervised fine-tuning, preference optimization, retrieval evaluation, calibration, long-context tests, and deployment decoding.

## Bridge to Training at Scale

The next section studies what happens when the same cross-entropy objective is optimized with billions of tokens, large batches, distributed hardware, mixed precision, gradient accumulation, learning-rate schedules, and checkpointing. Nothing in the training-at-scale story changes the probability target. Scale changes how expensive and fragile it is to estimate the same conditional distributions.

## References

- Claude Shannon, "A Mathematical Theory of Communication", 1948.
- Claude Shannon, "Prediction and Entropy of Printed English", 1951.
- Yoshua Bengio, Rejean Ducharme, Pascal Vincent, and Christian Jauvin, "A Neural Probabilistic Language Model", JMLR, 2003: https://jmlr.csail.mit.edu/papers/v3/bengio03a.html
- Tomas Mikolov, Martin Karafiat, Lukas Burget, Jan Cernocky, and Sanjeev Khudanpur, "Recurrent neural network based language model", Interspeech, 2010: https://www.isca-archive.org/interspeech_2010/mikolov10_interspeech.html
- Dan Jurafsky and James H. Martin, "Speech and Language Processing", Chapter 3 on n-gram language models: https://web.stanford.edu/~jurafsky/slp3/
- Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, and Yonghui Wu, "Exploring the Limits of Language Modeling", 2016: https://arxiv.org/abs/1602.02410
- Ashish Vaswani et al., "Attention Is All You Need", 2017: https://arxiv.org/abs/1706.03762
- Ari Holtzman et al., "The Curious Case of Neural Text Degeneration", 2019: https://arxiv.org/abs/1904.09751
- Ofir Press et al., "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation", 2021: https://arxiv.org/abs/2108.12409
