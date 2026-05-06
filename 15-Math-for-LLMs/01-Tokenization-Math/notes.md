[Back to Math for LLMs](../README.md) | [Next: Embedding Space Math](../02-Embedding-Space-Math/notes.md)

---

# Tokenization Math

> _"Before an LLM predicts text, a tokenizer decides what the text is allowed to be made of."_

## Overview

Tokenization is the first mathematical map in an LLM pipeline. It converts raw text into integer ids, and those ids determine embedding lookups, attention positions, loss targets, context-window usage, retrieval chunks, special-token boundaries, and serving cost.

The key tradeoff is simple but deep: larger vocabularies usually shorten sequences, while smaller vocabularies share pieces more aggressively. BPE, unigram tokenization, WordPiece, byte fallback, and special-token design are different answers to that tradeoff.

This section uses LaTeX Markdown with `$...$` and `$$...$$`. The companion notebooks implement small tokenizers from scratch so the math is visible without external packages.

## Prerequisites

- [Entropy](../../09-Information-Theory/01-Entropy/notes.md)
- [Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md)
- [Embedding Space Math](../02-Embedding-Space-Math/notes.md)
- [Attention Mechanism Math](../03-Attention-Mechanism-Math/notes.md)
- [Language Model Probability](../05-Language-Model-Probability/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Executable BPE, unigram, Viterbi, entropy, fertility, special-token, and context-cost demonstrations. |
| [exercises.ipynb](exercises.ipynb) | Ten checked exercises for tokenizer mechanics and LLM cost reasoning. |

## Learning Objectives

After completing this section, you will be able to:

- Define alphabets, vocabularies, token ids, encoders, and decoders.
- Explain why tokenization is a mathematical compression map, not only text cleanup.
- Train a tiny BPE tokenizer and apply learned merges to new strings.
- Use dynamic programming to find a best unigram segmentation.
- Compare BPE, unigram, and WordPiece segmentation criteria.
- Compute compression ratio, token entropy, context cost, and vocabulary parameter cost.
- Diagnose multilingual fertility and tokenization tax.
- Explain how special tokens affect chat templates, padding masks, tools, and safety boundaries.
- Design round-trip and boundary tests for a tokenizer.
- Explain why tokenizer changes usually require retraining embeddings or careful migration.

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 Text is not the model input](#11-text-is-not-the-model-input)
  - [1.2 The tokenizer as a learned compression map](#12-the-tokenizer-as-a-learned-compression-map)
  - [1.3 Open vocabulary pressure](#13-open-vocabulary-pressure)
  - [1.4 Tokenization tax](#14-tokenization-tax)
  - [1.5 Where tokenization affects LLM behavior](#15-where-tokenization-affects-llm-behavior)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Alphabet strings and byte sequences](#21-alphabet-strings-and-byte-sequences)
  - [2.2 Vocabulary and token ids](#22-vocabulary-and-token-ids)
  - [2.3 Tokenizer and detokenizer functions](#23-tokenizer-and-detokenizer-functions)
  - [2.4 Lossless versus lossy normalization](#24-lossless-versus-lossy-normalization)
  - [2.5 Pre-tokenization boundaries](#25-pretokenization-boundaries)
- [3. Byte Pair Encoding](#3-byte-pair-encoding)
  - [3.1 BPE merge objective](#31-bpe-merge-objective)
  - [3.2 Greedy training loop](#32-greedy-training-loop)
  - [3.3 Encoding with learned merges](#33-encoding-with-learned-merges)
  - [3.4 Byte-level BPE](#34-bytelevel-bpe)
  - [3.5 BPE limitations](#35-bpe-limitations)
- [4. Unigram and SentencePiece](#4-unigram-and-sentencepiece)
  - [4.1 Unigram token probability model](#41-unigram-token-probability-model)
  - [4.2 Viterbi segmentation](#42-viterbi-segmentation)
  - [4.3 Forward probabilities](#43-forward-probabilities)
  - [4.4 EM intuition](#44-em-intuition)
  - [4.5 Subword regularization](#45-subword-regularization)
- [5. WordPiece](#5-wordpiece)
  - [5.1 WordPiece merge score](#51-wordpiece-merge-score)
  - [5.2 Continuation markers](#52-continuation-markers)
  - [5.3 Greedy longest-match encoding](#53-greedy-longestmatch-encoding)
  - [5.4 WordPiece versus BPE](#54-wordpiece-versus-bpe)
  - [5.5 Unknown-token risk](#55-unknowntoken-risk)
- [6. Information and Cost](#6-information-and-cost)
  - [6.1 Compression ratio](#61-compression-ratio)
  - [6.2 Token entropy](#62-token-entropy)
  - [6.3 Vocabulary parameter cost](#63-vocabulary-parameter-cost)
  - [6.4 Context window efficiency](#64-context-window-efficiency)
  - [6.5 Multilingual fertility](#65-multilingual-fertility)
- [7. LLM System Effects](#7-llm-system-effects)
  - [7.1 Special tokens](#71-special-tokens)
  - [7.2 Attention cost](#72-attention-cost)
  - [7.3 Numeracy and spelling](#73-numeracy-and-spelling)
  - [7.4 Retrieval chunking](#74-retrieval-chunking)
  - [7.5 Safety and prompt boundaries](#75-safety-and-prompt-boundaries)
- [8. Evaluation and Diagnostics](#8-evaluation-and-diagnostics)
  - [8.1 Round-trip tests](#81-roundtrip-tests)
  - [8.2 Coverage tests](#82-coverage-tests)
  - [8.3 Fertility dashboards](#83-fertility-dashboards)
  - [8.4 Boundary tests](#84-boundary-tests)
  - [8.5 Tokenizer migration tests](#85-tokenizer-migration-tests)
- [9. Common Mistakes](#9-common-mistakes)
- [10. Exercises](#10-exercises)
- [11. Why This Matters for AI](#11-why-this-matters-for-ai)
- [12. Conceptual Bridge](#12-conceptual-bridge)
- [References](#references)

---

## 1. Intuition

Intuition develops the tokenizer concepts needed before embeddings, attention, language-model probability, and serving tradeoffs can be understood correctly.

### 1.1 Text is not the model input

**Purpose.** Text is not the model input focuses on why neural networks consume integer ids. In LLM systems this choice affects integer ids, sequence length, embedding rows, loss targets, and serving cost.

$$E(x)=(t_1,\ldots,t_n),\qquad t_i\in\{0,\ldots,|\mathcal{V}|-1\}.$$

**Operational definition.**

A transformer receives integer token ids, not raw strings. The tokenizer decides the discrete sequence over which embeddings, attention, and next-token probabilities are defined.

**Worked reading.**

If the text `lowering` becomes `[low, er, ing]`, the model predicts over those pieces. If it becomes `[lower, ing]`, the prediction problem changes.

| Tokenizer object | Mathematical role | LLM consequence |
| --- | --- | --- |
| alphabet $\Sigma$ | atomic input symbols | bytes, characters, or normalized symbols |
| vocabulary $\mathcal{V}$ | finite token set | embedding and output-logit dimensions |
| encoder $E$ | maps text to ids | prompt length, training examples, costs |
| decoder $D$ | maps ids back to text | detokenization and round-trip safety |
| merge/probability table | segmentation rule | subword boundaries and rare-string handling |

Examples:

1. token embeddings.
2. next-token loss.
3. attention over token positions.

Non-examples:

1. raw Unicode text inside a matrix multiply.
2. a word-level assumption for every language.

**Derivation habit.**

1. State the raw alphabet and any normalization step.
2. State whether the model uses BPE, unigram, WordPiece, byte fallback, or a hybrid.
3. Compute token count before making cost, context, or memory claims.
4. Check reversibility with `decode(encode(x))` when the pipeline promises losslessness.
5. Treat special tokens as protected control symbols, not ordinary text pieces.

**Implementation lens.**

A tokenizer is not a preprocessing detail that can be swapped freely. The embedding matrix, output projection, cached datasets, labels, special-token masks, and generation stop conditions all depend on the exact integer ids.

The most useful debugging habit is to print the text, tokens, ids, decoded text, and offsets for edge cases: leading spaces, repeated newlines, numbers, URLs, code, mixed scripts, and delimiter strings used by chat templates.

For model training, the tokenizer changes the effective curriculum. A tokenizer that splits common words into many pieces makes the model spend more positions modeling spelling-level structure. A tokenizer that memorizes very long pieces may save context but increase vocabulary cost and reduce compositional sharing.

For inference, the tokenizer changes both latency and price because attention cost grows with token length. A prompt that is short in words can still be expensive if it tokenizes poorly.

### 1.2 The tokenizer as a learned compression map

**Purpose.** The tokenizer as a learned compression map focuses on why segmentation changes sequence length. In LLM systems this choice affects integer ids, sequence length, embedding rows, loss targets, and serving cost.

$$D(E(x))=x\quad\text{for a lossless tokenizer on its supported domain}.$$

**Operational definition.**

Tokenization is compression under constraints: it trades vocabulary size against sequence length and distribution balance.

**Worked reading.**

A lower token count improves context efficiency, but a huge vocabulary increases embedding and output-layer cost.

| Tokenizer object | Mathematical role | LLM consequence |
| --- | --- | --- |
| alphabet $\Sigma$ | atomic input symbols | bytes, characters, or normalized symbols |
| vocabulary $\mathcal{V}$ | finite token set | embedding and output-logit dimensions |
| encoder $E$ | maps text to ids | prompt length, training examples, costs |
| decoder $D$ | maps ids back to text | detokenization and round-trip safety |
| merge/probability table | segmentation rule | subword boundaries and rare-string handling |

Examples:

1. characters per token.
2. tokens per word.
3. entropy of token frequencies.

Non-examples:

1. judging cost by words alone.
2. ignoring sequence-length effects in attention.

**Derivation habit.**

1. State the raw alphabet and any normalization step.
2. State whether the model uses BPE, unigram, WordPiece, byte fallback, or a hybrid.
3. Compute token count before making cost, context, or memory claims.
4. Check reversibility with `decode(encode(x))` when the pipeline promises losslessness.
5. Treat special tokens as protected control symbols, not ordinary text pieces.

**Implementation lens.**

A tokenizer is not a preprocessing detail that can be swapped freely. The embedding matrix, output projection, cached datasets, labels, special-token masks, and generation stop conditions all depend on the exact integer ids.

The most useful debugging habit is to print the text, tokens, ids, decoded text, and offsets for edge cases: leading spaces, repeated newlines, numbers, URLs, code, mixed scripts, and delimiter strings used by chat templates.

For model training, the tokenizer changes the effective curriculum. A tokenizer that splits common words into many pieces makes the model spend more positions modeling spelling-level structure. A tokenizer that memorizes very long pieces may save context but increase vocabulary cost and reduce compositional sharing.

For inference, the tokenizer changes both latency and price because attention cost grows with token length. A prompt that is short in words can still be expensive if it tokenizes poorly.

### 1.3 Open vocabulary pressure

**Purpose.** Open vocabulary pressure focuses on why word-level vocabularies fail. In LLM systems this choice affects integer ids, sequence length, embedding rows, loss targets, and serving cost.

$$\operatorname{score}_{\mathrm{BPE}}(a,b)=\operatorname{count}(ab).$$

**Operational definition.**

The vocabulary is a finite set of pieces with stable integer ids. Neural embeddings make those ids trainable vectors.

**Worked reading.**

With vocabulary size $|\mathcal{V}|$ and model width $d_{\mathrm{model}}$, the input embedding table has $|\mathcal{V}|d_{\mathrm{model}}$ parameters.

| Tokenizer object | Mathematical role | LLM consequence |
| --- | --- | --- |
| alphabet $\Sigma$ | atomic input symbols | bytes, characters, or normalized symbols |
| vocabulary $\mathcal{V}$ | finite token set | embedding and output-logit dimensions |
| encoder $E$ | maps text to ids | prompt length, training examples, costs |
| decoder $D$ | maps ids back to text | detokenization and round-trip safety |
| merge/probability table | segmentation rule | subword boundaries and rare-string handling |

Examples:

1. embedding lookup.
2. output softmax.
3. reserved special ids.

Non-examples:

1. renumbering tokens after training.
2. adding pieces without resizing embeddings.

**Derivation habit.**

1. State the raw alphabet and any normalization step.
2. State whether the model uses BPE, unigram, WordPiece, byte fallback, or a hybrid.
3. Compute token count before making cost, context, or memory claims.
4. Check reversibility with `decode(encode(x))` when the pipeline promises losslessness.
5. Treat special tokens as protected control symbols, not ordinary text pieces.

**Implementation lens.**

A tokenizer is not a preprocessing detail that can be swapped freely. The embedding matrix, output projection, cached datasets, labels, special-token masks, and generation stop conditions all depend on the exact integer ids.

The most useful debugging habit is to print the text, tokens, ids, decoded text, and offsets for edge cases: leading spaces, repeated newlines, numbers, URLs, code, mixed scripts, and delimiter strings used by chat templates.

For model training, the tokenizer changes the effective curriculum. A tokenizer that splits common words into many pieces makes the model spend more positions modeling spelling-level structure. A tokenizer that memorizes very long pieces may save context but increase vocabulary cost and reduce compositional sharing.

For inference, the tokenizer changes both latency and price because attention cost grows with token length. A prompt that is short in words can still be expensive if it tokenizes poorly.

### 1.4 Tokenization tax

**Purpose.** Tokenization tax focuses on why some languages and strings cost more tokens. In LLM systems this choice affects integer ids, sequence length, embedding rows, loss targets, and serving cost.

$$x^*=\arg\max_{v_1,\ldots,v_k:\,v_1\cdots v_k=x}\sum_{i=1}^k\log p(v_i).$$

**Operational definition.**

This concept controls how raw text becomes the sequence of discrete units optimized by an LLM.

**Worked reading.**

The practical question is always how the choice changes ids, sequence length, reversibility, or downstream loss.

| Tokenizer object | Mathematical role | LLM consequence |
| --- | --- | --- |
| alphabet $\Sigma$ | atomic input symbols | bytes, characters, or normalized symbols |
| vocabulary $\mathcal{V}$ | finite token set | embedding and output-logit dimensions |
| encoder $E$ | maps text to ids | prompt length, training examples, costs |
| decoder $D$ | maps ids back to text | detokenization and round-trip safety |
| merge/probability table | segmentation rule | subword boundaries and rare-string handling |

Examples:

1. BPE pieces.
2. unigram pieces.
3. special tokens.

Non-examples:

1. raw text passed directly to attention.
2. word counts used as token counts.

**Derivation habit.**

1. State the raw alphabet and any normalization step.
2. State whether the model uses BPE, unigram, WordPiece, byte fallback, or a hybrid.
3. Compute token count before making cost, context, or memory claims.
4. Check reversibility with `decode(encode(x))` when the pipeline promises losslessness.
5. Treat special tokens as protected control symbols, not ordinary text pieces.

**Implementation lens.**

A tokenizer is not a preprocessing detail that can be swapped freely. The embedding matrix, output projection, cached datasets, labels, special-token masks, and generation stop conditions all depend on the exact integer ids.

The most useful debugging habit is to print the text, tokens, ids, decoded text, and offsets for edge cases: leading spaces, repeated newlines, numbers, URLs, code, mixed scripts, and delimiter strings used by chat templates.

For model training, the tokenizer changes the effective curriculum. A tokenizer that splits common words into many pieces makes the model spend more positions modeling spelling-level structure. A tokenizer that memorizes very long pieces may save context but increase vocabulary cost and reduce compositional sharing.

For inference, the tokenizer changes both latency and price because attention cost grows with token length. A prompt that is short in words can still be expensive if it tokenizes poorly.

### 1.5 Where tokenization affects LLM behavior

**Purpose.** Where tokenization affects LLM behavior focuses on context, cost, arithmetic, safety, retrieval. In LLM systems this choice affects integer ids, sequence length, embedding rows, loss targets, and serving cost.

$$H(T)=-\sum_{v\in\mathcal{V}}p(v)\log_2 p(v).$$

**Operational definition.**

This concept controls how raw text becomes the sequence of discrete units optimized by an LLM.

**Worked reading.**

The practical question is always how the choice changes ids, sequence length, reversibility, or downstream loss.

| Tokenizer object | Mathematical role | LLM consequence |
| --- | --- | --- |
| alphabet $\Sigma$ | atomic input symbols | bytes, characters, or normalized symbols |
| vocabulary $\mathcal{V}$ | finite token set | embedding and output-logit dimensions |
| encoder $E$ | maps text to ids | prompt length, training examples, costs |
| decoder $D$ | maps ids back to text | detokenization and round-trip safety |
| merge/probability table | segmentation rule | subword boundaries and rare-string handling |

Examples:

1. BPE pieces.
2. unigram pieces.
3. special tokens.

Non-examples:

1. raw text passed directly to attention.
2. word counts used as token counts.

**Derivation habit.**

1. State the raw alphabet and any normalization step.
2. State whether the model uses BPE, unigram, WordPiece, byte fallback, or a hybrid.
3. Compute token count before making cost, context, or memory claims.
4. Check reversibility with `decode(encode(x))` when the pipeline promises losslessness.
5. Treat special tokens as protected control symbols, not ordinary text pieces.

**Implementation lens.**

A tokenizer is not a preprocessing detail that can be swapped freely. The embedding matrix, output projection, cached datasets, labels, special-token masks, and generation stop conditions all depend on the exact integer ids.

The most useful debugging habit is to print the text, tokens, ids, decoded text, and offsets for edge cases: leading spaces, repeated newlines, numbers, URLs, code, mixed scripts, and delimiter strings used by chat templates.

For model training, the tokenizer changes the effective curriculum. A tokenizer that splits common words into many pieces makes the model spend more positions modeling spelling-level structure. A tokenizer that memorizes very long pieces may save context but increase vocabulary cost and reduce compositional sharing.

For inference, the tokenizer changes both latency and price because attention cost grows with token length. A prompt that is short in words can still be expensive if it tokenizes poorly.

## 2. Formal Definitions

Formal Definitions develops the tokenizer concepts needed before embeddings, attention, language-model probability, and serving tradeoffs can be understood correctly.

### 2.1 Alphabet strings and byte sequences

**Purpose.** Alphabet strings and byte sequences focuses on the raw domain $\Sigma^*$. In LLM systems this choice affects integer ids, sequence length, embedding rows, loss targets, and serving cost.

$$D(E(x))=x\quad\text{for a lossless tokenizer on its supported domain}.$$

**Operational definition.**

The raw alphabet determines whether the tokenizer starts from bytes, Unicode code points, characters, or pre-tokenized pieces.

**Worked reading.**

Byte-level systems can represent arbitrary input without an unknown character because every string can be encoded as bytes.

| Tokenizer object | Mathematical role | LLM consequence |
| --- | --- | --- |
| alphabet $\Sigma$ | atomic input symbols | bytes, characters, or normalized symbols |
| vocabulary $\mathcal{V}$ | finite token set | embedding and output-logit dimensions |
| encoder $E$ | maps text to ids | prompt length, training examples, costs |
| decoder $D$ | maps ids back to text | detokenization and round-trip safety |
| merge/probability table | segmentation rule | subword boundaries and rare-string handling |

Examples:

1. UTF-8 byte sequences.
2. ASCII text.
3. mixed code and natural language.

Non-examples:

1. assuming one character equals one byte.
2. dropping unsupported symbols silently.

**Derivation habit.**

1. State the raw alphabet and any normalization step.
2. State whether the model uses BPE, unigram, WordPiece, byte fallback, or a hybrid.
3. Compute token count before making cost, context, or memory claims.
4. Check reversibility with `decode(encode(x))` when the pipeline promises losslessness.
5. Treat special tokens as protected control symbols, not ordinary text pieces.

**Implementation lens.**

A tokenizer is not a preprocessing detail that can be swapped freely. The embedding matrix, output projection, cached datasets, labels, special-token masks, and generation stop conditions all depend on the exact integer ids.

The most useful debugging habit is to print the text, tokens, ids, decoded text, and offsets for edge cases: leading spaces, repeated newlines, numbers, URLs, code, mixed scripts, and delimiter strings used by chat templates.

For model training, the tokenizer changes the effective curriculum. A tokenizer that splits common words into many pieces makes the model spend more positions modeling spelling-level structure. A tokenizer that memorizes very long pieces may save context but increase vocabulary cost and reduce compositional sharing.

For inference, the tokenizer changes both latency and price because attention cost grows with token length. A prompt that is short in words can still be expensive if it tokenizes poorly.

### 2.2 Vocabulary and token ids

**Purpose.** Vocabulary and token ids focuses on the finite set $\mathcal{V}$ and integer map. In LLM systems this choice affects integer ids, sequence length, embedding rows, loss targets, and serving cost.

$$\operatorname{score}_{\mathrm{BPE}}(a,b)=\operatorname{count}(ab).$$

**Operational definition.**

The vocabulary is a finite set of pieces with stable integer ids. Neural embeddings make those ids trainable vectors.

**Worked reading.**

With vocabulary size $|\mathcal{V}|$ and model width $d_{\mathrm{model}}$, the input embedding table has $|\mathcal{V}|d_{\mathrm{model}}$ parameters.

| Tokenizer object | Mathematical role | LLM consequence |
| --- | --- | --- |
| alphabet $\Sigma$ | atomic input symbols | bytes, characters, or normalized symbols |
| vocabulary $\mathcal{V}$ | finite token set | embedding and output-logit dimensions |
| encoder $E$ | maps text to ids | prompt length, training examples, costs |
| decoder $D$ | maps ids back to text | detokenization and round-trip safety |
| merge/probability table | segmentation rule | subword boundaries and rare-string handling |

Examples:

1. embedding lookup.
2. output softmax.
3. reserved special ids.

Non-examples:

1. renumbering tokens after training.
2. adding pieces without resizing embeddings.

**Derivation habit.**

1. State the raw alphabet and any normalization step.
2. State whether the model uses BPE, unigram, WordPiece, byte fallback, or a hybrid.
3. Compute token count before making cost, context, or memory claims.
4. Check reversibility with `decode(encode(x))` when the pipeline promises losslessness.
5. Treat special tokens as protected control symbols, not ordinary text pieces.

**Implementation lens.**

A tokenizer is not a preprocessing detail that can be swapped freely. The embedding matrix, output projection, cached datasets, labels, special-token masks, and generation stop conditions all depend on the exact integer ids.

The most useful debugging habit is to print the text, tokens, ids, decoded text, and offsets for edge cases: leading spaces, repeated newlines, numbers, URLs, code, mixed scripts, and delimiter strings used by chat templates.

For model training, the tokenizer changes the effective curriculum. A tokenizer that splits common words into many pieces makes the model spend more positions modeling spelling-level structure. A tokenizer that memorizes very long pieces may save context but increase vocabulary cost and reduce compositional sharing.

For inference, the tokenizer changes both latency and price because attention cost grows with token length. A prompt that is short in words can still be expensive if it tokenizes poorly.

### 2.3 Tokenizer and detokenizer functions

**Purpose.** Tokenizer and detokenizer functions focuses on the pair $E:\Sigma^*\to [|\mathcal{V}|]^*$ and $D$. In LLM systems this choice affects integer ids, sequence length, embedding rows, loss targets, and serving cost.

$$x^*=\arg\max_{v_1,\ldots,v_k:\,v_1\cdots v_k=x}\sum_{i=1}^k\log p(v_i).$$

**Operational definition.**

This concept controls how raw text becomes the sequence of discrete units optimized by an LLM.

**Worked reading.**

The practical question is always how the choice changes ids, sequence length, reversibility, or downstream loss.

| Tokenizer object | Mathematical role | LLM consequence |
| --- | --- | --- |
| alphabet $\Sigma$ | atomic input symbols | bytes, characters, or normalized symbols |
| vocabulary $\mathcal{V}$ | finite token set | embedding and output-logit dimensions |
| encoder $E$ | maps text to ids | prompt length, training examples, costs |
| decoder $D$ | maps ids back to text | detokenization and round-trip safety |
| merge/probability table | segmentation rule | subword boundaries and rare-string handling |

Examples:

1. BPE pieces.
2. unigram pieces.
3. special tokens.

Non-examples:

1. raw text passed directly to attention.
2. word counts used as token counts.

**Derivation habit.**

1. State the raw alphabet and any normalization step.
2. State whether the model uses BPE, unigram, WordPiece, byte fallback, or a hybrid.
3. Compute token count before making cost, context, or memory claims.
4. Check reversibility with `decode(encode(x))` when the pipeline promises losslessness.
5. Treat special tokens as protected control symbols, not ordinary text pieces.

**Implementation lens.**

A tokenizer is not a preprocessing detail that can be swapped freely. The embedding matrix, output projection, cached datasets, labels, special-token masks, and generation stop conditions all depend on the exact integer ids.

The most useful debugging habit is to print the text, tokens, ids, decoded text, and offsets for edge cases: leading spaces, repeated newlines, numbers, URLs, code, mixed scripts, and delimiter strings used by chat templates.

For model training, the tokenizer changes the effective curriculum. A tokenizer that splits common words into many pieces makes the model spend more positions modeling spelling-level structure. A tokenizer that memorizes very long pieces may save context but increase vocabulary cost and reduce compositional sharing.

For inference, the tokenizer changes both latency and price because attention cost grows with token length. A prompt that is short in words can still be expensive if it tokenizes poorly.

### 2.4 Lossless versus lossy normalization

**Purpose.** Lossless versus lossy normalization focuses on why reversible byte tokenization matters. In LLM systems this choice affects integer ids, sequence length, embedding rows, loss targets, and serving cost.

$$H(T)=-\sum_{v\in\mathcal{V}}p(v)\log_2 p(v).$$

**Operational definition.**

This concept controls how raw text becomes the sequence of discrete units optimized by an LLM.

**Worked reading.**

The practical question is always how the choice changes ids, sequence length, reversibility, or downstream loss.

| Tokenizer object | Mathematical role | LLM consequence |
| --- | --- | --- |
| alphabet $\Sigma$ | atomic input symbols | bytes, characters, or normalized symbols |
| vocabulary $\mathcal{V}$ | finite token set | embedding and output-logit dimensions |
| encoder $E$ | maps text to ids | prompt length, training examples, costs |
| decoder $D$ | maps ids back to text | detokenization and round-trip safety |
| merge/probability table | segmentation rule | subword boundaries and rare-string handling |

Examples:

1. BPE pieces.
2. unigram pieces.
3. special tokens.

Non-examples:

1. raw text passed directly to attention.
2. word counts used as token counts.

**Derivation habit.**

1. State the raw alphabet and any normalization step.
2. State whether the model uses BPE, unigram, WordPiece, byte fallback, or a hybrid.
3. Compute token count before making cost, context, or memory claims.
4. Check reversibility with `decode(encode(x))` when the pipeline promises losslessness.
5. Treat special tokens as protected control symbols, not ordinary text pieces.

**Implementation lens.**

A tokenizer is not a preprocessing detail that can be swapped freely. The embedding matrix, output projection, cached datasets, labels, special-token masks, and generation stop conditions all depend on the exact integer ids.

The most useful debugging habit is to print the text, tokens, ids, decoded text, and offsets for edge cases: leading spaces, repeated newlines, numbers, URLs, code, mixed scripts, and delimiter strings used by chat templates.

For model training, the tokenizer changes the effective curriculum. A tokenizer that splits common words into many pieces makes the model spend more positions modeling spelling-level structure. A tokenizer that memorizes very long pieces may save context but increase vocabulary cost and reduce compositional sharing.

For inference, the tokenizer changes both latency and price because attention cost grows with token length. A prompt that is short in words can still be expensive if it tokenizes poorly.

### 2.5 Pre-tokenization boundaries

**Purpose.** Pre-tokenization boundaries focuses on how whitespace and regex choices constrain merges. In LLM systems this choice affects integer ids, sequence length, embedding rows, loss targets, and serving cost.

$$\operatorname{fertility}(x)=\frac{\#\operatorname{tokens}(x)}{\#\operatorname{words}(x)}.$$

**Operational definition.**

This concept controls how raw text becomes the sequence of discrete units optimized by an LLM.

**Worked reading.**

The practical question is always how the choice changes ids, sequence length, reversibility, or downstream loss.

| Tokenizer object | Mathematical role | LLM consequence |
| --- | --- | --- |
| alphabet $\Sigma$ | atomic input symbols | bytes, characters, or normalized symbols |
| vocabulary $\mathcal{V}$ | finite token set | embedding and output-logit dimensions |
| encoder $E$ | maps text to ids | prompt length, training examples, costs |
| decoder $D$ | maps ids back to text | detokenization and round-trip safety |
| merge/probability table | segmentation rule | subword boundaries and rare-string handling |

Examples:

1. BPE pieces.
2. unigram pieces.
3. special tokens.

Non-examples:

1. raw text passed directly to attention.
2. word counts used as token counts.

**Derivation habit.**

1. State the raw alphabet and any normalization step.
2. State whether the model uses BPE, unigram, WordPiece, byte fallback, or a hybrid.
3. Compute token count before making cost, context, or memory claims.
4. Check reversibility with `decode(encode(x))` when the pipeline promises losslessness.
5. Treat special tokens as protected control symbols, not ordinary text pieces.

**Implementation lens.**

A tokenizer is not a preprocessing detail that can be swapped freely. The embedding matrix, output projection, cached datasets, labels, special-token masks, and generation stop conditions all depend on the exact integer ids.

The most useful debugging habit is to print the text, tokens, ids, decoded text, and offsets for edge cases: leading spaces, repeated newlines, numbers, URLs, code, mixed scripts, and delimiter strings used by chat templates.

For model training, the tokenizer changes the effective curriculum. A tokenizer that splits common words into many pieces makes the model spend more positions modeling spelling-level structure. A tokenizer that memorizes very long pieces may save context but increase vocabulary cost and reduce compositional sharing.

For inference, the tokenizer changes both latency and price because attention cost grows with token length. A prompt that is short in words can still be expensive if it tokenizes poorly.

## 3. Byte Pair Encoding

Byte Pair Encoding develops the tokenizer concepts needed before embeddings, attention, language-model probability, and serving tradeoffs can be understood correctly.

### 3.1 BPE merge objective

**Purpose.** BPE merge objective focuses on frequency-based pair replacement. In LLM systems this choice affects integer ids, sequence length, embedding rows, loss targets, and serving cost.

$$\operatorname{score}_{\mathrm{BPE}}(a,b)=\operatorname{count}(ab).$$

**Operational definition.**

BPE starts from small symbols and repeatedly merges the most frequent adjacent pair. The learned merge order becomes a compression table.

**Worked reading.**

If `l o` is the most frequent pair, BPE creates `lo`; later it may create `low` if `lo w` becomes frequent.

| Tokenizer object | Mathematical role | LLM consequence |
| --- | --- | --- |
| alphabet $\Sigma$ | atomic input symbols | bytes, characters, or normalized symbols |
| vocabulary $\mathcal{V}$ | finite token set | embedding and output-logit dimensions |
| encoder $E$ | maps text to ids | prompt length, training examples, costs |
| decoder $D$ | maps ids back to text | detokenization and round-trip safety |
| merge/probability table | segmentation rule | subword boundaries and rare-string handling |

Examples:

1. GPT-style byte-level tokenizers.
2. subword NMT.
3. domain-specific vocabulary learning.

Non-examples:

1. probabilistically summing all segmentations.
2. longest-match WordPiece with continuation markers.

**Derivation habit.**

1. State the raw alphabet and any normalization step.
2. State whether the model uses BPE, unigram, WordPiece, byte fallback, or a hybrid.
3. Compute token count before making cost, context, or memory claims.
4. Check reversibility with `decode(encode(x))` when the pipeline promises losslessness.
5. Treat special tokens as protected control symbols, not ordinary text pieces.

**Implementation lens.**

A tokenizer is not a preprocessing detail that can be swapped freely. The embedding matrix, output projection, cached datasets, labels, special-token masks, and generation stop conditions all depend on the exact integer ids.

The most useful debugging habit is to print the text, tokens, ids, decoded text, and offsets for edge cases: leading spaces, repeated newlines, numbers, URLs, code, mixed scripts, and delimiter strings used by chat templates.

For model training, the tokenizer changes the effective curriculum. A tokenizer that splits common words into many pieces makes the model spend more positions modeling spelling-level structure. A tokenizer that memorizes very long pieces may save context but increase vocabulary cost and reduce compositional sharing.

For inference, the tokenizer changes both latency and price because attention cost grows with token length. A prompt that is short in words can still be expensive if it tokenizes poorly.

### 3.2 Greedy training loop

**Purpose.** Greedy training loop focuses on why merges are sequential decisions. In LLM systems this choice affects integer ids, sequence length, embedding rows, loss targets, and serving cost.

$$x^*=\arg\max_{v_1,\ldots,v_k:\,v_1\cdots v_k=x}\sum_{i=1}^k\log p(v_i).$$

**Operational definition.**

This concept controls how raw text becomes the sequence of discrete units optimized by an LLM.

**Worked reading.**

The practical question is always how the choice changes ids, sequence length, reversibility, or downstream loss.

| Tokenizer object | Mathematical role | LLM consequence |
| --- | --- | --- |
| alphabet $\Sigma$ | atomic input symbols | bytes, characters, or normalized symbols |
| vocabulary $\mathcal{V}$ | finite token set | embedding and output-logit dimensions |
| encoder $E$ | maps text to ids | prompt length, training examples, costs |
| decoder $D$ | maps ids back to text | detokenization and round-trip safety |
| merge/probability table | segmentation rule | subword boundaries and rare-string handling |

Examples:

1. BPE pieces.
2. unigram pieces.
3. special tokens.

Non-examples:

1. raw text passed directly to attention.
2. word counts used as token counts.

**Derivation habit.**

1. State the raw alphabet and any normalization step.
2. State whether the model uses BPE, unigram, WordPiece, byte fallback, or a hybrid.
3. Compute token count before making cost, context, or memory claims.
4. Check reversibility with `decode(encode(x))` when the pipeline promises losslessness.
5. Treat special tokens as protected control symbols, not ordinary text pieces.

**Implementation lens.**

A tokenizer is not a preprocessing detail that can be swapped freely. The embedding matrix, output projection, cached datasets, labels, special-token masks, and generation stop conditions all depend on the exact integer ids.

The most useful debugging habit is to print the text, tokens, ids, decoded text, and offsets for edge cases: leading spaces, repeated newlines, numbers, URLs, code, mixed scripts, and delimiter strings used by chat templates.

For model training, the tokenizer changes the effective curriculum. A tokenizer that splits common words into many pieces makes the model spend more positions modeling spelling-level structure. A tokenizer that memorizes very long pieces may save context but increase vocabulary cost and reduce compositional sharing.

For inference, the tokenizer changes both latency and price because attention cost grows with token length. A prompt that is short in words can still be expensive if it tokenizes poorly.

### 3.3 Encoding with learned merges

**Purpose.** Encoding with learned merges focuses on applying ranks to new text. In LLM systems this choice affects integer ids, sequence length, embedding rows, loss targets, and serving cost.

$$H(T)=-\sum_{v\in\mathcal{V}}p(v)\log_2 p(v).$$

**Operational definition.**

This concept controls how raw text becomes the sequence of discrete units optimized by an LLM.

**Worked reading.**

The practical question is always how the choice changes ids, sequence length, reversibility, or downstream loss.

| Tokenizer object | Mathematical role | LLM consequence |
| --- | --- | --- |
| alphabet $\Sigma$ | atomic input symbols | bytes, characters, or normalized symbols |
| vocabulary $\mathcal{V}$ | finite token set | embedding and output-logit dimensions |
| encoder $E$ | maps text to ids | prompt length, training examples, costs |
| decoder $D$ | maps ids back to text | detokenization and round-trip safety |
| merge/probability table | segmentation rule | subword boundaries and rare-string handling |

Examples:

1. BPE pieces.
2. unigram pieces.
3. special tokens.

Non-examples:

1. raw text passed directly to attention.
2. word counts used as token counts.

**Derivation habit.**

1. State the raw alphabet and any normalization step.
2. State whether the model uses BPE, unigram, WordPiece, byte fallback, or a hybrid.
3. Compute token count before making cost, context, or memory claims.
4. Check reversibility with `decode(encode(x))` when the pipeline promises losslessness.
5. Treat special tokens as protected control symbols, not ordinary text pieces.

**Implementation lens.**

A tokenizer is not a preprocessing detail that can be swapped freely. The embedding matrix, output projection, cached datasets, labels, special-token masks, and generation stop conditions all depend on the exact integer ids.

The most useful debugging habit is to print the text, tokens, ids, decoded text, and offsets for edge cases: leading spaces, repeated newlines, numbers, URLs, code, mixed scripts, and delimiter strings used by chat templates.

For model training, the tokenizer changes the effective curriculum. A tokenizer that splits common words into many pieces makes the model spend more positions modeling spelling-level structure. A tokenizer that memorizes very long pieces may save context but increase vocabulary cost and reduce compositional sharing.

For inference, the tokenizer changes both latency and price because attention cost grows with token length. A prompt that is short in words can still be expensive if it tokenizes poorly.

### 3.4 Byte-level BPE

**Purpose.** Byte-level BPE focuses on avoiding unknown characters. In LLM systems this choice affects integer ids, sequence length, embedding rows, loss targets, and serving cost.

$$\operatorname{fertility}(x)=\frac{\#\operatorname{tokens}(x)}{\#\operatorname{words}(x)}.$$

**Operational definition.**

The raw alphabet determines whether the tokenizer starts from bytes, Unicode code points, characters, or pre-tokenized pieces.

**Worked reading.**

Byte-level systems can represent arbitrary input without an unknown character because every string can be encoded as bytes.

| Tokenizer object | Mathematical role | LLM consequence |
| --- | --- | --- |
| alphabet $\Sigma$ | atomic input symbols | bytes, characters, or normalized symbols |
| vocabulary $\mathcal{V}$ | finite token set | embedding and output-logit dimensions |
| encoder $E$ | maps text to ids | prompt length, training examples, costs |
| decoder $D$ | maps ids back to text | detokenization and round-trip safety |
| merge/probability table | segmentation rule | subword boundaries and rare-string handling |

Examples:

1. UTF-8 byte sequences.
2. ASCII text.
3. mixed code and natural language.

Non-examples:

1. assuming one character equals one byte.
2. dropping unsupported symbols silently.

**Derivation habit.**

1. State the raw alphabet and any normalization step.
2. State whether the model uses BPE, unigram, WordPiece, byte fallback, or a hybrid.
3. Compute token count before making cost, context, or memory claims.
4. Check reversibility with `decode(encode(x))` when the pipeline promises losslessness.
5. Treat special tokens as protected control symbols, not ordinary text pieces.

**Implementation lens.**

A tokenizer is not a preprocessing detail that can be swapped freely. The embedding matrix, output projection, cached datasets, labels, special-token masks, and generation stop conditions all depend on the exact integer ids.

The most useful debugging habit is to print the text, tokens, ids, decoded text, and offsets for edge cases: leading spaces, repeated newlines, numbers, URLs, code, mixed scripts, and delimiter strings used by chat templates.

For model training, the tokenizer changes the effective curriculum. A tokenizer that splits common words into many pieces makes the model spend more positions modeling spelling-level structure. A tokenizer that memorizes very long pieces may save context but increase vocabulary cost and reduce compositional sharing.

For inference, the tokenizer changes both latency and price because attention cost grows with token length. A prompt that is short in words can still be expensive if it tokenizes poorly.

### 3.5 BPE limitations

**Purpose.** BPE limitations focuses on greedy segmentation and brittle numeric splits. In LLM systems this choice affects integer ids, sequence length, embedding rows, loss targets, and serving cost.

$$\operatorname{params}_{\mathrm{embed}}=|\mathcal{V}|\,d_{\mathrm{model}}.$$

**Operational definition.**

This concept controls how raw text becomes the sequence of discrete units optimized by an LLM.

**Worked reading.**

The practical question is always how the choice changes ids, sequence length, reversibility, or downstream loss.

| Tokenizer object | Mathematical role | LLM consequence |
| --- | --- | --- |
| alphabet $\Sigma$ | atomic input symbols | bytes, characters, or normalized symbols |
| vocabulary $\mathcal{V}$ | finite token set | embedding and output-logit dimensions |
| encoder $E$ | maps text to ids | prompt length, training examples, costs |
| decoder $D$ | maps ids back to text | detokenization and round-trip safety |
| merge/probability table | segmentation rule | subword boundaries and rare-string handling |

Examples:

1. BPE pieces.
2. unigram pieces.
3. special tokens.

Non-examples:

1. raw text passed directly to attention.
2. word counts used as token counts.

**Derivation habit.**

1. State the raw alphabet and any normalization step.
2. State whether the model uses BPE, unigram, WordPiece, byte fallback, or a hybrid.
3. Compute token count before making cost, context, or memory claims.
4. Check reversibility with `decode(encode(x))` when the pipeline promises losslessness.
5. Treat special tokens as protected control symbols, not ordinary text pieces.

**Implementation lens.**

A tokenizer is not a preprocessing detail that can be swapped freely. The embedding matrix, output projection, cached datasets, labels, special-token masks, and generation stop conditions all depend on the exact integer ids.

The most useful debugging habit is to print the text, tokens, ids, decoded text, and offsets for edge cases: leading spaces, repeated newlines, numbers, URLs, code, mixed scripts, and delimiter strings used by chat templates.

For model training, the tokenizer changes the effective curriculum. A tokenizer that splits common words into many pieces makes the model spend more positions modeling spelling-level structure. A tokenizer that memorizes very long pieces may save context but increase vocabulary cost and reduce compositional sharing.

For inference, the tokenizer changes both latency and price because attention cost grows with token length. A prompt that is short in words can still be expensive if it tokenizes poorly.

## 4. Unigram and SentencePiece

Unigram and SentencePiece develops the tokenizer concepts needed before embeddings, attention, language-model probability, and serving tradeoffs can be understood correctly.

### 4.1 Unigram token probability model

**Purpose.** Unigram token probability model focuses on probabilistic segmentation with $p(v)$. In LLM systems this choice affects integer ids, sequence length, embedding rows, loss targets, and serving cost.

$$x^*=\arg\max_{v_1,\ldots,v_k:\,v_1\cdots v_k=x}\sum_{i=1}^k\log p(v_i).$$

**Operational definition.**

This concept controls how raw text becomes the sequence of discrete units optimized by an LLM.

**Worked reading.**

The practical question is always how the choice changes ids, sequence length, reversibility, or downstream loss.

| Tokenizer object | Mathematical role | LLM consequence |
| --- | --- | --- |
| alphabet $\Sigma$ | atomic input symbols | bytes, characters, or normalized symbols |
| vocabulary $\mathcal{V}$ | finite token set | embedding and output-logit dimensions |
| encoder $E$ | maps text to ids | prompt length, training examples, costs |
| decoder $D$ | maps ids back to text | detokenization and round-trip safety |
| merge/probability table | segmentation rule | subword boundaries and rare-string handling |

Examples:

1. BPE pieces.
2. unigram pieces.
3. special tokens.

Non-examples:

1. raw text passed directly to attention.
2. word counts used as token counts.

**Derivation habit.**

1. State the raw alphabet and any normalization step.
2. State whether the model uses BPE, unigram, WordPiece, byte fallback, or a hybrid.
3. Compute token count before making cost, context, or memory claims.
4. Check reversibility with `decode(encode(x))` when the pipeline promises losslessness.
5. Treat special tokens as protected control symbols, not ordinary text pieces.

**Implementation lens.**

A tokenizer is not a preprocessing detail that can be swapped freely. The embedding matrix, output projection, cached datasets, labels, special-token masks, and generation stop conditions all depend on the exact integer ids.

The most useful debugging habit is to print the text, tokens, ids, decoded text, and offsets for edge cases: leading spaces, repeated newlines, numbers, URLs, code, mixed scripts, and delimiter strings used by chat templates.

For model training, the tokenizer changes the effective curriculum. A tokenizer that splits common words into many pieces makes the model spend more positions modeling spelling-level structure. A tokenizer that memorizes very long pieces may save context but increase vocabulary cost and reduce compositional sharing.

For inference, the tokenizer changes both latency and price because attention cost grows with token length. A prompt that is short in words can still be expensive if it tokenizes poorly.

### 4.2 Viterbi segmentation

**Purpose.** Viterbi segmentation focuses on dynamic programming for best token path. In LLM systems this choice affects integer ids, sequence length, embedding rows, loss targets, and serving cost.

$$H(T)=-\sum_{v\in\mathcal{V}}p(v)\log_2 p(v).$$

**Operational definition.**

Given token probabilities, Viterbi dynamic programming finds the highest-probability segmentation of a string.

**Worked reading.**

For `abab`, a unigram model compares `[ab, ab]`, `[a, b, ab]`, `[aba, b]`, and other valid paths by summing log probabilities.

| Tokenizer object | Mathematical role | LLM consequence |
| --- | --- | --- |
| alphabet $\Sigma$ | atomic input symbols | bytes, characters, or normalized symbols |
| vocabulary $\mathcal{V}$ | finite token set | embedding and output-logit dimensions |
| encoder $E$ | maps text to ids | prompt length, training examples, costs |
| decoder $D$ | maps ids back to text | detokenization and round-trip safety |
| merge/probability table | segmentation rule | subword boundaries and rare-string handling |

Examples:

1. SentencePiece unigram decoding.
2. best-path segmentation.
3. subword sampling baseline.

Non-examples:

1. frequency-only pair merging.
2. a regex split with no scoring.

**Derivation habit.**

1. State the raw alphabet and any normalization step.
2. State whether the model uses BPE, unigram, WordPiece, byte fallback, or a hybrid.
3. Compute token count before making cost, context, or memory claims.
4. Check reversibility with `decode(encode(x))` when the pipeline promises losslessness.
5. Treat special tokens as protected control symbols, not ordinary text pieces.

**Implementation lens.**

A tokenizer is not a preprocessing detail that can be swapped freely. The embedding matrix, output projection, cached datasets, labels, special-token masks, and generation stop conditions all depend on the exact integer ids.

The most useful debugging habit is to print the text, tokens, ids, decoded text, and offsets for edge cases: leading spaces, repeated newlines, numbers, URLs, code, mixed scripts, and delimiter strings used by chat templates.

For model training, the tokenizer changes the effective curriculum. A tokenizer that splits common words into many pieces makes the model spend more positions modeling spelling-level structure. A tokenizer that memorizes very long pieces may save context but increase vocabulary cost and reduce compositional sharing.

For inference, the tokenizer changes both latency and price because attention cost grows with token length. A prompt that is short in words can still be expensive if it tokenizes poorly.

### 4.3 Forward probabilities

**Purpose.** Forward probabilities focuses on summing over all segmentations. In LLM systems this choice affects integer ids, sequence length, embedding rows, loss targets, and serving cost.

$$\operatorname{fertility}(x)=\frac{\#\operatorname{tokens}(x)}{\#\operatorname{words}(x)}.$$

**Operational definition.**

This concept controls how raw text becomes the sequence of discrete units optimized by an LLM.

**Worked reading.**

The practical question is always how the choice changes ids, sequence length, reversibility, or downstream loss.

| Tokenizer object | Mathematical role | LLM consequence |
| --- | --- | --- |
| alphabet $\Sigma$ | atomic input symbols | bytes, characters, or normalized symbols |
| vocabulary $\mathcal{V}$ | finite token set | embedding and output-logit dimensions |
| encoder $E$ | maps text to ids | prompt length, training examples, costs |
| decoder $D$ | maps ids back to text | detokenization and round-trip safety |
| merge/probability table | segmentation rule | subword boundaries and rare-string handling |

Examples:

1. BPE pieces.
2. unigram pieces.
3. special tokens.

Non-examples:

1. raw text passed directly to attention.
2. word counts used as token counts.

**Derivation habit.**

1. State the raw alphabet and any normalization step.
2. State whether the model uses BPE, unigram, WordPiece, byte fallback, or a hybrid.
3. Compute token count before making cost, context, or memory claims.
4. Check reversibility with `decode(encode(x))` when the pipeline promises losslessness.
5. Treat special tokens as protected control symbols, not ordinary text pieces.

**Implementation lens.**

A tokenizer is not a preprocessing detail that can be swapped freely. The embedding matrix, output projection, cached datasets, labels, special-token masks, and generation stop conditions all depend on the exact integer ids.

The most useful debugging habit is to print the text, tokens, ids, decoded text, and offsets for edge cases: leading spaces, repeated newlines, numbers, URLs, code, mixed scripts, and delimiter strings used by chat templates.

For model training, the tokenizer changes the effective curriculum. A tokenizer that splits common words into many pieces makes the model spend more positions modeling spelling-level structure. A tokenizer that memorizes very long pieces may save context but increase vocabulary cost and reduce compositional sharing.

For inference, the tokenizer changes both latency and price because attention cost grows with token length. A prompt that is short in words can still be expensive if it tokenizes poorly.

### 4.4 EM intuition

**Purpose.** EM intuition focuses on soft counts for token pieces. In LLM systems this choice affects integer ids, sequence length, embedding rows, loss targets, and serving cost.

$$\operatorname{params}_{\mathrm{embed}}=|\mathcal{V}|\,d_{\mathrm{model}}.$$

**Operational definition.**

This concept controls how raw text becomes the sequence of discrete units optimized by an LLM.

**Worked reading.**

The practical question is always how the choice changes ids, sequence length, reversibility, or downstream loss.

| Tokenizer object | Mathematical role | LLM consequence |
| --- | --- | --- |
| alphabet $\Sigma$ | atomic input symbols | bytes, characters, or normalized symbols |
| vocabulary $\mathcal{V}$ | finite token set | embedding and output-logit dimensions |
| encoder $E$ | maps text to ids | prompt length, training examples, costs |
| decoder $D$ | maps ids back to text | detokenization and round-trip safety |
| merge/probability table | segmentation rule | subword boundaries and rare-string handling |

Examples:

1. BPE pieces.
2. unigram pieces.
3. special tokens.

Non-examples:

1. raw text passed directly to attention.
2. word counts used as token counts.

**Derivation habit.**

1. State the raw alphabet and any normalization step.
2. State whether the model uses BPE, unigram, WordPiece, byte fallback, or a hybrid.
3. Compute token count before making cost, context, or memory claims.
4. Check reversibility with `decode(encode(x))` when the pipeline promises losslessness.
5. Treat special tokens as protected control symbols, not ordinary text pieces.

**Implementation lens.**

A tokenizer is not a preprocessing detail that can be swapped freely. The embedding matrix, output projection, cached datasets, labels, special-token masks, and generation stop conditions all depend on the exact integer ids.

The most useful debugging habit is to print the text, tokens, ids, decoded text, and offsets for edge cases: leading spaces, repeated newlines, numbers, URLs, code, mixed scripts, and delimiter strings used by chat templates.

For model training, the tokenizer changes the effective curriculum. A tokenizer that splits common words into many pieces makes the model spend more positions modeling spelling-level structure. A tokenizer that memorizes very long pieces may save context but increase vocabulary cost and reduce compositional sharing.

For inference, the tokenizer changes both latency and price because attention cost grows with token length. A prompt that is short in words can still be expensive if it tokenizes poorly.

### 4.5 Subword regularization

**Purpose.** Subword regularization focuses on sampling multiple valid segmentations. In LLM systems this choice affects integer ids, sequence length, embedding rows, loss targets, and serving cost.

$$\operatorname{attention\ cost}\propto n_{\mathrm{tokens}}^2.$$

**Operational definition.**

This concept controls how raw text becomes the sequence of discrete units optimized by an LLM.

**Worked reading.**

The practical question is always how the choice changes ids, sequence length, reversibility, or downstream loss.

| Tokenizer object | Mathematical role | LLM consequence |
| --- | --- | --- |
| alphabet $\Sigma$ | atomic input symbols | bytes, characters, or normalized symbols |
| vocabulary $\mathcal{V}$ | finite token set | embedding and output-logit dimensions |
| encoder $E$ | maps text to ids | prompt length, training examples, costs |
| decoder $D$ | maps ids back to text | detokenization and round-trip safety |
| merge/probability table | segmentation rule | subword boundaries and rare-string handling |

Examples:

1. BPE pieces.
2. unigram pieces.
3. special tokens.

Non-examples:

1. raw text passed directly to attention.
2. word counts used as token counts.

**Derivation habit.**

1. State the raw alphabet and any normalization step.
2. State whether the model uses BPE, unigram, WordPiece, byte fallback, or a hybrid.
3. Compute token count before making cost, context, or memory claims.
4. Check reversibility with `decode(encode(x))` when the pipeline promises losslessness.
5. Treat special tokens as protected control symbols, not ordinary text pieces.

**Implementation lens.**

A tokenizer is not a preprocessing detail that can be swapped freely. The embedding matrix, output projection, cached datasets, labels, special-token masks, and generation stop conditions all depend on the exact integer ids.

The most useful debugging habit is to print the text, tokens, ids, decoded text, and offsets for edge cases: leading spaces, repeated newlines, numbers, URLs, code, mixed scripts, and delimiter strings used by chat templates.

For model training, the tokenizer changes the effective curriculum. A tokenizer that splits common words into many pieces makes the model spend more positions modeling spelling-level structure. A tokenizer that memorizes very long pieces may save context but increase vocabulary cost and reduce compositional sharing.

For inference, the tokenizer changes both latency and price because attention cost grows with token length. A prompt that is short in words can still be expensive if it tokenizes poorly.

## 5. WordPiece

WordPiece develops the tokenizer concepts needed before embeddings, attention, language-model probability, and serving tradeoffs can be understood correctly.

### 5.1 WordPiece merge score

**Purpose.** WordPiece merge score focuses on association-style pair scoring. In LLM systems this choice affects integer ids, sequence length, embedding rows, loss targets, and serving cost.

$$H(T)=-\sum_{v\in\mathcal{V}}p(v)\log_2 p(v).$$

**Operational definition.**

WordPiece builds subwords using an association-style score and commonly distinguishes word starts from continuation pieces.

**Worked reading.**

A longest-match encoder chooses the longest valid vocabulary piece at each position, using continuation markers inside words.

| Tokenizer object | Mathematical role | LLM consequence |
| --- | --- | --- |
| alphabet $\Sigma$ | atomic input symbols | bytes, characters, or normalized symbols |
| vocabulary $\mathcal{V}$ | finite token set | embedding and output-logit dimensions |
| encoder $E$ | maps text to ids | prompt length, training examples, costs |
| decoder $D$ | maps ids back to text | detokenization and round-trip safety |
| merge/probability table | segmentation rule | subword boundaries and rare-string handling |

Examples:

1. BERT tokenization.
2. continuation pieces.
3. greedy longest match.

Non-examples:

1. byte fallback.
2. unigram path sampling.

**Derivation habit.**

1. State the raw alphabet and any normalization step.
2. State whether the model uses BPE, unigram, WordPiece, byte fallback, or a hybrid.
3. Compute token count before making cost, context, or memory claims.
4. Check reversibility with `decode(encode(x))` when the pipeline promises losslessness.
5. Treat special tokens as protected control symbols, not ordinary text pieces.

**Implementation lens.**

A tokenizer is not a preprocessing detail that can be swapped freely. The embedding matrix, output projection, cached datasets, labels, special-token masks, and generation stop conditions all depend on the exact integer ids.

The most useful debugging habit is to print the text, tokens, ids, decoded text, and offsets for edge cases: leading spaces, repeated newlines, numbers, URLs, code, mixed scripts, and delimiter strings used by chat templates.

For model training, the tokenizer changes the effective curriculum. A tokenizer that splits common words into many pieces makes the model spend more positions modeling spelling-level structure. A tokenizer that memorizes very long pieces may save context but increase vocabulary cost and reduce compositional sharing.

For inference, the tokenizer changes both latency and price because attention cost grows with token length. A prompt that is short in words can still be expensive if it tokenizes poorly.

### 5.2 Continuation markers

**Purpose.** Continuation markers focuses on why prefixes and inside-word pieces differ. In LLM systems this choice affects integer ids, sequence length, embedding rows, loss targets, and serving cost.

$$\operatorname{fertility}(x)=\frac{\#\operatorname{tokens}(x)}{\#\operatorname{words}(x)}.$$

**Operational definition.**

This concept controls how raw text becomes the sequence of discrete units optimized by an LLM.

**Worked reading.**

The practical question is always how the choice changes ids, sequence length, reversibility, or downstream loss.

| Tokenizer object | Mathematical role | LLM consequence |
| --- | --- | --- |
| alphabet $\Sigma$ | atomic input symbols | bytes, characters, or normalized symbols |
| vocabulary $\mathcal{V}$ | finite token set | embedding and output-logit dimensions |
| encoder $E$ | maps text to ids | prompt length, training examples, costs |
| decoder $D$ | maps ids back to text | detokenization and round-trip safety |
| merge/probability table | segmentation rule | subword boundaries and rare-string handling |

Examples:

1. BPE pieces.
2. unigram pieces.
3. special tokens.

Non-examples:

1. raw text passed directly to attention.
2. word counts used as token counts.

**Derivation habit.**

1. State the raw alphabet and any normalization step.
2. State whether the model uses BPE, unigram, WordPiece, byte fallback, or a hybrid.
3. Compute token count before making cost, context, or memory claims.
4. Check reversibility with `decode(encode(x))` when the pipeline promises losslessness.
5. Treat special tokens as protected control symbols, not ordinary text pieces.

**Implementation lens.**

A tokenizer is not a preprocessing detail that can be swapped freely. The embedding matrix, output projection, cached datasets, labels, special-token masks, and generation stop conditions all depend on the exact integer ids.

The most useful debugging habit is to print the text, tokens, ids, decoded text, and offsets for edge cases: leading spaces, repeated newlines, numbers, URLs, code, mixed scripts, and delimiter strings used by chat templates.

For model training, the tokenizer changes the effective curriculum. A tokenizer that splits common words into many pieces makes the model spend more positions modeling spelling-level structure. A tokenizer that memorizes very long pieces may save context but increase vocabulary cost and reduce compositional sharing.

For inference, the tokenizer changes both latency and price because attention cost grows with token length. A prompt that is short in words can still be expensive if it tokenizes poorly.

### 5.3 Greedy longest-match encoding

**Purpose.** Greedy longest-match encoding focuses on BERT-style deterministic segmentation. In LLM systems this choice affects integer ids, sequence length, embedding rows, loss targets, and serving cost.

$$\operatorname{params}_{\mathrm{embed}}=|\mathcal{V}|\,d_{\mathrm{model}}.$$

**Operational definition.**

This concept controls how raw text becomes the sequence of discrete units optimized by an LLM.

**Worked reading.**

The practical question is always how the choice changes ids, sequence length, reversibility, or downstream loss.

| Tokenizer object | Mathematical role | LLM consequence |
| --- | --- | --- |
| alphabet $\Sigma$ | atomic input symbols | bytes, characters, or normalized symbols |
| vocabulary $\mathcal{V}$ | finite token set | embedding and output-logit dimensions |
| encoder $E$ | maps text to ids | prompt length, training examples, costs |
| decoder $D$ | maps ids back to text | detokenization and round-trip safety |
| merge/probability table | segmentation rule | subword boundaries and rare-string handling |

Examples:

1. BPE pieces.
2. unigram pieces.
3. special tokens.

Non-examples:

1. raw text passed directly to attention.
2. word counts used as token counts.

**Derivation habit.**

1. State the raw alphabet and any normalization step.
2. State whether the model uses BPE, unigram, WordPiece, byte fallback, or a hybrid.
3. Compute token count before making cost, context, or memory claims.
4. Check reversibility with `decode(encode(x))` when the pipeline promises losslessness.
5. Treat special tokens as protected control symbols, not ordinary text pieces.

**Implementation lens.**

A tokenizer is not a preprocessing detail that can be swapped freely. The embedding matrix, output projection, cached datasets, labels, special-token masks, and generation stop conditions all depend on the exact integer ids.

The most useful debugging habit is to print the text, tokens, ids, decoded text, and offsets for edge cases: leading spaces, repeated newlines, numbers, URLs, code, mixed scripts, and delimiter strings used by chat templates.

For model training, the tokenizer changes the effective curriculum. A tokenizer that splits common words into many pieces makes the model spend more positions modeling spelling-level structure. A tokenizer that memorizes very long pieces may save context but increase vocabulary cost and reduce compositional sharing.

For inference, the tokenizer changes both latency and price because attention cost grows with token length. A prompt that is short in words can still be expensive if it tokenizes poorly.

### 5.4 WordPiece versus BPE

**Purpose.** WordPiece versus BPE focuses on score criterion and encoding behavior. In LLM systems this choice affects integer ids, sequence length, embedding rows, loss targets, and serving cost.

$$\operatorname{attention\ cost}\propto n_{\mathrm{tokens}}^2.$$

**Operational definition.**

WordPiece builds subwords using an association-style score and commonly distinguishes word starts from continuation pieces.

**Worked reading.**

A longest-match encoder chooses the longest valid vocabulary piece at each position, using continuation markers inside words.

| Tokenizer object | Mathematical role | LLM consequence |
| --- | --- | --- |
| alphabet $\Sigma$ | atomic input symbols | bytes, characters, or normalized symbols |
| vocabulary $\mathcal{V}$ | finite token set | embedding and output-logit dimensions |
| encoder $E$ | maps text to ids | prompt length, training examples, costs |
| decoder $D$ | maps ids back to text | detokenization and round-trip safety |
| merge/probability table | segmentation rule | subword boundaries and rare-string handling |

Examples:

1. BERT tokenization.
2. continuation pieces.
3. greedy longest match.

Non-examples:

1. byte fallback.
2. unigram path sampling.

**Derivation habit.**

1. State the raw alphabet and any normalization step.
2. State whether the model uses BPE, unigram, WordPiece, byte fallback, or a hybrid.
3. Compute token count before making cost, context, or memory claims.
4. Check reversibility with `decode(encode(x))` when the pipeline promises losslessness.
5. Treat special tokens as protected control symbols, not ordinary text pieces.

**Implementation lens.**

A tokenizer is not a preprocessing detail that can be swapped freely. The embedding matrix, output projection, cached datasets, labels, special-token masks, and generation stop conditions all depend on the exact integer ids.

The most useful debugging habit is to print the text, tokens, ids, decoded text, and offsets for edge cases: leading spaces, repeated newlines, numbers, URLs, code, mixed scripts, and delimiter strings used by chat templates.

For model training, the tokenizer changes the effective curriculum. A tokenizer that splits common words into many pieces makes the model spend more positions modeling spelling-level structure. A tokenizer that memorizes very long pieces may save context but increase vocabulary cost and reduce compositional sharing.

For inference, the tokenizer changes both latency and price because attention cost grows with token length. A prompt that is short in words can still be expensive if it tokenizes poorly.

### 5.5 Unknown-token risk

**Purpose.** Unknown-token risk focuses on why byte fallback changes robustness. In LLM systems this choice affects integer ids, sequence length, embedding rows, loss targets, and serving cost.

$$E(x)=(t_1,\ldots,t_n),\qquad t_i\in\{0,\ldots,|\mathcal{V}|-1\}.$$

**Operational definition.**

This concept controls how raw text becomes the sequence of discrete units optimized by an LLM.

**Worked reading.**

The practical question is always how the choice changes ids, sequence length, reversibility, or downstream loss.

| Tokenizer object | Mathematical role | LLM consequence |
| --- | --- | --- |
| alphabet $\Sigma$ | atomic input symbols | bytes, characters, or normalized symbols |
| vocabulary $\mathcal{V}$ | finite token set | embedding and output-logit dimensions |
| encoder $E$ | maps text to ids | prompt length, training examples, costs |
| decoder $D$ | maps ids back to text | detokenization and round-trip safety |
| merge/probability table | segmentation rule | subword boundaries and rare-string handling |

Examples:

1. BPE pieces.
2. unigram pieces.
3. special tokens.

Non-examples:

1. raw text passed directly to attention.
2. word counts used as token counts.

**Derivation habit.**

1. State the raw alphabet and any normalization step.
2. State whether the model uses BPE, unigram, WordPiece, byte fallback, or a hybrid.
3. Compute token count before making cost, context, or memory claims.
4. Check reversibility with `decode(encode(x))` when the pipeline promises losslessness.
5. Treat special tokens as protected control symbols, not ordinary text pieces.

**Implementation lens.**

A tokenizer is not a preprocessing detail that can be swapped freely. The embedding matrix, output projection, cached datasets, labels, special-token masks, and generation stop conditions all depend on the exact integer ids.

The most useful debugging habit is to print the text, tokens, ids, decoded text, and offsets for edge cases: leading spaces, repeated newlines, numbers, URLs, code, mixed scripts, and delimiter strings used by chat templates.

For model training, the tokenizer changes the effective curriculum. A tokenizer that splits common words into many pieces makes the model spend more positions modeling spelling-level structure. A tokenizer that memorizes very long pieces may save context but increase vocabulary cost and reduce compositional sharing.

For inference, the tokenizer changes both latency and price because attention cost grows with token length. A prompt that is short in words can still be expensive if it tokenizes poorly.

## 6. Information and Cost

Information and Cost develops the tokenizer concepts needed before embeddings, attention, language-model probability, and serving tradeoffs can be understood correctly.

### 6.1 Compression ratio

**Purpose.** Compression ratio focuses on characters per token and bytes per token. In LLM systems this choice affects integer ids, sequence length, embedding rows, loss targets, and serving cost.

$$\operatorname{fertility}(x)=\frac{\#\operatorname{tokens}(x)}{\#\operatorname{words}(x)}.$$

**Operational definition.**

Tokenization is compression under constraints: it trades vocabulary size against sequence length and distribution balance.

**Worked reading.**

A lower token count improves context efficiency, but a huge vocabulary increases embedding and output-layer cost.

| Tokenizer object | Mathematical role | LLM consequence |
| --- | --- | --- |
| alphabet $\Sigma$ | atomic input symbols | bytes, characters, or normalized symbols |
| vocabulary $\mathcal{V}$ | finite token set | embedding and output-logit dimensions |
| encoder $E$ | maps text to ids | prompt length, training examples, costs |
| decoder $D$ | maps ids back to text | detokenization and round-trip safety |
| merge/probability table | segmentation rule | subword boundaries and rare-string handling |

Examples:

1. characters per token.
2. tokens per word.
3. entropy of token frequencies.

Non-examples:

1. judging cost by words alone.
2. ignoring sequence-length effects in attention.

**Derivation habit.**

1. State the raw alphabet and any normalization step.
2. State whether the model uses BPE, unigram, WordPiece, byte fallback, or a hybrid.
3. Compute token count before making cost, context, or memory claims.
4. Check reversibility with `decode(encode(x))` when the pipeline promises losslessness.
5. Treat special tokens as protected control symbols, not ordinary text pieces.

**Implementation lens.**

A tokenizer is not a preprocessing detail that can be swapped freely. The embedding matrix, output projection, cached datasets, labels, special-token masks, and generation stop conditions all depend on the exact integer ids.

The most useful debugging habit is to print the text, tokens, ids, decoded text, and offsets for edge cases: leading spaces, repeated newlines, numbers, URLs, code, mixed scripts, and delimiter strings used by chat templates.

For model training, the tokenizer changes the effective curriculum. A tokenizer that splits common words into many pieces makes the model spend more positions modeling spelling-level structure. A tokenizer that memorizes very long pieces may save context but increase vocabulary cost and reduce compositional sharing.

For inference, the tokenizer changes both latency and price because attention cost grows with token length. A prompt that is short in words can still be expensive if it tokenizes poorly.

### 6.2 Token entropy

**Purpose.** Token entropy focuses on distributional balance of token ids. In LLM systems this choice affects integer ids, sequence length, embedding rows, loss targets, and serving cost.

$$\operatorname{params}_{\mathrm{embed}}=|\mathcal{V}|\,d_{\mathrm{model}}.$$

**Operational definition.**

Tokenization is compression under constraints: it trades vocabulary size against sequence length and distribution balance.

**Worked reading.**

A lower token count improves context efficiency, but a huge vocabulary increases embedding and output-layer cost.

| Tokenizer object | Mathematical role | LLM consequence |
| --- | --- | --- |
| alphabet $\Sigma$ | atomic input symbols | bytes, characters, or normalized symbols |
| vocabulary $\mathcal{V}$ | finite token set | embedding and output-logit dimensions |
| encoder $E$ | maps text to ids | prompt length, training examples, costs |
| decoder $D$ | maps ids back to text | detokenization and round-trip safety |
| merge/probability table | segmentation rule | subword boundaries and rare-string handling |

Examples:

1. characters per token.
2. tokens per word.
3. entropy of token frequencies.

Non-examples:

1. judging cost by words alone.
2. ignoring sequence-length effects in attention.

**Derivation habit.**

1. State the raw alphabet and any normalization step.
2. State whether the model uses BPE, unigram, WordPiece, byte fallback, or a hybrid.
3. Compute token count before making cost, context, or memory claims.
4. Check reversibility with `decode(encode(x))` when the pipeline promises losslessness.
5. Treat special tokens as protected control symbols, not ordinary text pieces.

**Implementation lens.**

A tokenizer is not a preprocessing detail that can be swapped freely. The embedding matrix, output projection, cached datasets, labels, special-token masks, and generation stop conditions all depend on the exact integer ids.

The most useful debugging habit is to print the text, tokens, ids, decoded text, and offsets for edge cases: leading spaces, repeated newlines, numbers, URLs, code, mixed scripts, and delimiter strings used by chat templates.

For model training, the tokenizer changes the effective curriculum. A tokenizer that splits common words into many pieces makes the model spend more positions modeling spelling-level structure. A tokenizer that memorizes very long pieces may save context but increase vocabulary cost and reduce compositional sharing.

For inference, the tokenizer changes both latency and price because attention cost grows with token length. A prompt that is short in words can still be expensive if it tokenizes poorly.

### 6.3 Vocabulary parameter cost

**Purpose.** Vocabulary parameter cost focuses on embedding and output matrix scaling. In LLM systems this choice affects integer ids, sequence length, embedding rows, loss targets, and serving cost.

$$\operatorname{attention\ cost}\propto n_{\mathrm{tokens}}^2.$$

**Operational definition.**

The vocabulary is a finite set of pieces with stable integer ids. Neural embeddings make those ids trainable vectors.

**Worked reading.**

With vocabulary size $|\mathcal{V}|$ and model width $d_{\mathrm{model}}$, the input embedding table has $|\mathcal{V}|d_{\mathrm{model}}$ parameters.

| Tokenizer object | Mathematical role | LLM consequence |
| --- | --- | --- |
| alphabet $\Sigma$ | atomic input symbols | bytes, characters, or normalized symbols |
| vocabulary $\mathcal{V}$ | finite token set | embedding and output-logit dimensions |
| encoder $E$ | maps text to ids | prompt length, training examples, costs |
| decoder $D$ | maps ids back to text | detokenization and round-trip safety |
| merge/probability table | segmentation rule | subword boundaries and rare-string handling |

Examples:

1. embedding lookup.
2. output softmax.
3. reserved special ids.

Non-examples:

1. renumbering tokens after training.
2. adding pieces without resizing embeddings.

**Derivation habit.**

1. State the raw alphabet and any normalization step.
2. State whether the model uses BPE, unigram, WordPiece, byte fallback, or a hybrid.
3. Compute token count before making cost, context, or memory claims.
4. Check reversibility with `decode(encode(x))` when the pipeline promises losslessness.
5. Treat special tokens as protected control symbols, not ordinary text pieces.

**Implementation lens.**

A tokenizer is not a preprocessing detail that can be swapped freely. The embedding matrix, output projection, cached datasets, labels, special-token masks, and generation stop conditions all depend on the exact integer ids.

The most useful debugging habit is to print the text, tokens, ids, decoded text, and offsets for edge cases: leading spaces, repeated newlines, numbers, URLs, code, mixed scripts, and delimiter strings used by chat templates.

For model training, the tokenizer changes the effective curriculum. A tokenizer that splits common words into many pieces makes the model spend more positions modeling spelling-level structure. A tokenizer that memorizes very long pieces may save context but increase vocabulary cost and reduce compositional sharing.

For inference, the tokenizer changes both latency and price because attention cost grows with token length. A prompt that is short in words can still be expensive if it tokenizes poorly.

### 6.4 Context window efficiency

**Purpose.** Context window efficiency focuses on how token count changes usable context. In LLM systems this choice affects integer ids, sequence length, embedding rows, loss targets, and serving cost.

$$E(x)=(t_1,\ldots,t_n),\qquad t_i\in\{0,\ldots,|\mathcal{V}|-1\}.$$

**Operational definition.**

Tokenization is compression under constraints: it trades vocabulary size against sequence length and distribution balance.

**Worked reading.**

A lower token count improves context efficiency, but a huge vocabulary increases embedding and output-layer cost.

| Tokenizer object | Mathematical role | LLM consequence |
| --- | --- | --- |
| alphabet $\Sigma$ | atomic input symbols | bytes, characters, or normalized symbols |
| vocabulary $\mathcal{V}$ | finite token set | embedding and output-logit dimensions |
| encoder $E$ | maps text to ids | prompt length, training examples, costs |
| decoder $D$ | maps ids back to text | detokenization and round-trip safety |
| merge/probability table | segmentation rule | subword boundaries and rare-string handling |

Examples:

1. characters per token.
2. tokens per word.
3. entropy of token frequencies.

Non-examples:

1. judging cost by words alone.
2. ignoring sequence-length effects in attention.

**Derivation habit.**

1. State the raw alphabet and any normalization step.
2. State whether the model uses BPE, unigram, WordPiece, byte fallback, or a hybrid.
3. Compute token count before making cost, context, or memory claims.
4. Check reversibility with `decode(encode(x))` when the pipeline promises losslessness.
5. Treat special tokens as protected control symbols, not ordinary text pieces.

**Implementation lens.**

A tokenizer is not a preprocessing detail that can be swapped freely. The embedding matrix, output projection, cached datasets, labels, special-token masks, and generation stop conditions all depend on the exact integer ids.

The most useful debugging habit is to print the text, tokens, ids, decoded text, and offsets for edge cases: leading spaces, repeated newlines, numbers, URLs, code, mixed scripts, and delimiter strings used by chat templates.

For model training, the tokenizer changes the effective curriculum. A tokenizer that splits common words into many pieces makes the model spend more positions modeling spelling-level structure. A tokenizer that memorizes very long pieces may save context but increase vocabulary cost and reduce compositional sharing.

For inference, the tokenizer changes both latency and price because attention cost grows with token length. A prompt that is short in words can still be expensive if it tokenizes poorly.

### 6.5 Multilingual fertility

**Purpose.** Multilingual fertility focuses on tokens per word across languages or scripts. In LLM systems this choice affects integer ids, sequence length, embedding rows, loss targets, and serving cost.

$$D(E(x))=x\quad\text{for a lossless tokenizer on its supported domain}.$$

**Operational definition.**

Tokenization is compression under constraints: it trades vocabulary size against sequence length and distribution balance.

**Worked reading.**

A lower token count improves context efficiency, but a huge vocabulary increases embedding and output-layer cost.

| Tokenizer object | Mathematical role | LLM consequence |
| --- | --- | --- |
| alphabet $\Sigma$ | atomic input symbols | bytes, characters, or normalized symbols |
| vocabulary $\mathcal{V}$ | finite token set | embedding and output-logit dimensions |
| encoder $E$ | maps text to ids | prompt length, training examples, costs |
| decoder $D$ | maps ids back to text | detokenization and round-trip safety |
| merge/probability table | segmentation rule | subword boundaries and rare-string handling |

Examples:

1. characters per token.
2. tokens per word.
3. entropy of token frequencies.

Non-examples:

1. judging cost by words alone.
2. ignoring sequence-length effects in attention.

**Derivation habit.**

1. State the raw alphabet and any normalization step.
2. State whether the model uses BPE, unigram, WordPiece, byte fallback, or a hybrid.
3. Compute token count before making cost, context, or memory claims.
4. Check reversibility with `decode(encode(x))` when the pipeline promises losslessness.
5. Treat special tokens as protected control symbols, not ordinary text pieces.

**Implementation lens.**

A tokenizer is not a preprocessing detail that can be swapped freely. The embedding matrix, output projection, cached datasets, labels, special-token masks, and generation stop conditions all depend on the exact integer ids.

The most useful debugging habit is to print the text, tokens, ids, decoded text, and offsets for edge cases: leading spaces, repeated newlines, numbers, URLs, code, mixed scripts, and delimiter strings used by chat templates.

For model training, the tokenizer changes the effective curriculum. A tokenizer that splits common words into many pieces makes the model spend more positions modeling spelling-level structure. A tokenizer that memorizes very long pieces may save context but increase vocabulary cost and reduce compositional sharing.

For inference, the tokenizer changes both latency and price because attention cost grows with token length. A prompt that is short in words can still be expensive if it tokenizes poorly.

## 7. LLM System Effects

LLM System Effects develops the tokenizer concepts needed before embeddings, attention, language-model probability, and serving tradeoffs can be understood correctly.

### 7.1 Special tokens

**Purpose.** Special tokens focuses on BOS EOS padding masks roles and tool delimiters. In LLM systems this choice affects integer ids, sequence length, embedding rows, loss targets, and serving cost.

$$\operatorname{params}_{\mathrm{embed}}=|\mathcal{V}|\,d_{\mathrm{model}}.$$

**Operational definition.**

Special tokens are vocabulary entries with control meaning rather than ordinary lexical meaning. They must be protected from accidental splitting.

**Worked reading.**

A chat template may reserve tokens for system, user, assistant, tool call, end-of-message, padding, or beginning-of-sequence boundaries.

| Tokenizer object | Mathematical role | LLM consequence |
| --- | --- | --- |
| alphabet $\Sigma$ | atomic input symbols | bytes, characters, or normalized symbols |
| vocabulary $\mathcal{V}$ | finite token set | embedding and output-logit dimensions |
| encoder $E$ | maps text to ids | prompt length, training examples, costs |
| decoder $D$ | maps ids back to text | detokenization and round-trip safety |
| merge/probability table | segmentation rule | subword boundaries and rare-string handling |

Examples:

1. BOS/EOS.
2. padding and masks.
3. role delimiters in chat models.

Non-examples:

1. ordinary word pieces.
2. strings that can be merged through by BPE.

**Derivation habit.**

1. State the raw alphabet and any normalization step.
2. State whether the model uses BPE, unigram, WordPiece, byte fallback, or a hybrid.
3. Compute token count before making cost, context, or memory claims.
4. Check reversibility with `decode(encode(x))` when the pipeline promises losslessness.
5. Treat special tokens as protected control symbols, not ordinary text pieces.

**Implementation lens.**

A tokenizer is not a preprocessing detail that can be swapped freely. The embedding matrix, output projection, cached datasets, labels, special-token masks, and generation stop conditions all depend on the exact integer ids.

The most useful debugging habit is to print the text, tokens, ids, decoded text, and offsets for edge cases: leading spaces, repeated newlines, numbers, URLs, code, mixed scripts, and delimiter strings used by chat templates.

For model training, the tokenizer changes the effective curriculum. A tokenizer that splits common words into many pieces makes the model spend more positions modeling spelling-level structure. A tokenizer that memorizes very long pieces may save context but increase vocabulary cost and reduce compositional sharing.

For inference, the tokenizer changes both latency and price because attention cost grows with token length. A prompt that is short in words can still be expensive if it tokenizes poorly.

### 7.2 Attention cost

**Purpose.** Attention cost focuses on why token length changes quadratic compute. In LLM systems this choice affects integer ids, sequence length, embedding rows, loss targets, and serving cost.

$$\operatorname{attention\ cost}\propto n_{\mathrm{tokens}}^2.$$

**Operational definition.**

Tokenization is compression under constraints: it trades vocabulary size against sequence length and distribution balance.

**Worked reading.**

A lower token count improves context efficiency, but a huge vocabulary increases embedding and output-layer cost.

| Tokenizer object | Mathematical role | LLM consequence |
| --- | --- | --- |
| alphabet $\Sigma$ | atomic input symbols | bytes, characters, or normalized symbols |
| vocabulary $\mathcal{V}$ | finite token set | embedding and output-logit dimensions |
| encoder $E$ | maps text to ids | prompt length, training examples, costs |
| decoder $D$ | maps ids back to text | detokenization and round-trip safety |
| merge/probability table | segmentation rule | subword boundaries and rare-string handling |

Examples:

1. characters per token.
2. tokens per word.
3. entropy of token frequencies.

Non-examples:

1. judging cost by words alone.
2. ignoring sequence-length effects in attention.

**Derivation habit.**

1. State the raw alphabet and any normalization step.
2. State whether the model uses BPE, unigram, WordPiece, byte fallback, or a hybrid.
3. Compute token count before making cost, context, or memory claims.
4. Check reversibility with `decode(encode(x))` when the pipeline promises losslessness.
5. Treat special tokens as protected control symbols, not ordinary text pieces.

**Implementation lens.**

A tokenizer is not a preprocessing detail that can be swapped freely. The embedding matrix, output projection, cached datasets, labels, special-token masks, and generation stop conditions all depend on the exact integer ids.

The most useful debugging habit is to print the text, tokens, ids, decoded text, and offsets for edge cases: leading spaces, repeated newlines, numbers, URLs, code, mixed scripts, and delimiter strings used by chat templates.

For model training, the tokenizer changes the effective curriculum. A tokenizer that splits common words into many pieces makes the model spend more positions modeling spelling-level structure. A tokenizer that memorizes very long pieces may save context but increase vocabulary cost and reduce compositional sharing.

For inference, the tokenizer changes both latency and price because attention cost grows with token length. A prompt that is short in words can still be expensive if it tokenizes poorly.

### 7.3 Numeracy and spelling

**Purpose.** Numeracy and spelling focuses on why digit and character segmentation matters. In LLM systems this choice affects integer ids, sequence length, embedding rows, loss targets, and serving cost.

$$E(x)=(t_1,\ldots,t_n),\qquad t_i\in\{0,\ldots,|\mathcal{V}|-1\}.$$

**Operational definition.**

This concept controls how raw text becomes the sequence of discrete units optimized by an LLM.

**Worked reading.**

The practical question is always how the choice changes ids, sequence length, reversibility, or downstream loss.

| Tokenizer object | Mathematical role | LLM consequence |
| --- | --- | --- |
| alphabet $\Sigma$ | atomic input symbols | bytes, characters, or normalized symbols |
| vocabulary $\mathcal{V}$ | finite token set | embedding and output-logit dimensions |
| encoder $E$ | maps text to ids | prompt length, training examples, costs |
| decoder $D$ | maps ids back to text | detokenization and round-trip safety |
| merge/probability table | segmentation rule | subword boundaries and rare-string handling |

Examples:

1. BPE pieces.
2. unigram pieces.
3. special tokens.

Non-examples:

1. raw text passed directly to attention.
2. word counts used as token counts.

**Derivation habit.**

1. State the raw alphabet and any normalization step.
2. State whether the model uses BPE, unigram, WordPiece, byte fallback, or a hybrid.
3. Compute token count before making cost, context, or memory claims.
4. Check reversibility with `decode(encode(x))` when the pipeline promises losslessness.
5. Treat special tokens as protected control symbols, not ordinary text pieces.

**Implementation lens.**

A tokenizer is not a preprocessing detail that can be swapped freely. The embedding matrix, output projection, cached datasets, labels, special-token masks, and generation stop conditions all depend on the exact integer ids.

The most useful debugging habit is to print the text, tokens, ids, decoded text, and offsets for edge cases: leading spaces, repeated newlines, numbers, URLs, code, mixed scripts, and delimiter strings used by chat templates.

For model training, the tokenizer changes the effective curriculum. A tokenizer that splits common words into many pieces makes the model spend more positions modeling spelling-level structure. A tokenizer that memorizes very long pieces may save context but increase vocabulary cost and reduce compositional sharing.

For inference, the tokenizer changes both latency and price because attention cost grows with token length. A prompt that is short in words can still be expensive if it tokenizes poorly.

### 7.4 Retrieval chunking

**Purpose.** Retrieval chunking focuses on why chunk size should be token-aware. In LLM systems this choice affects integer ids, sequence length, embedding rows, loss targets, and serving cost.

$$D(E(x))=x\quad\text{for a lossless tokenizer on its supported domain}.$$

**Operational definition.**

This concept controls how raw text becomes the sequence of discrete units optimized by an LLM.

**Worked reading.**

The practical question is always how the choice changes ids, sequence length, reversibility, or downstream loss.

| Tokenizer object | Mathematical role | LLM consequence |
| --- | --- | --- |
| alphabet $\Sigma$ | atomic input symbols | bytes, characters, or normalized symbols |
| vocabulary $\mathcal{V}$ | finite token set | embedding and output-logit dimensions |
| encoder $E$ | maps text to ids | prompt length, training examples, costs |
| decoder $D$ | maps ids back to text | detokenization and round-trip safety |
| merge/probability table | segmentation rule | subword boundaries and rare-string handling |

Examples:

1. BPE pieces.
2. unigram pieces.
3. special tokens.

Non-examples:

1. raw text passed directly to attention.
2. word counts used as token counts.

**Derivation habit.**

1. State the raw alphabet and any normalization step.
2. State whether the model uses BPE, unigram, WordPiece, byte fallback, or a hybrid.
3. Compute token count before making cost, context, or memory claims.
4. Check reversibility with `decode(encode(x))` when the pipeline promises losslessness.
5. Treat special tokens as protected control symbols, not ordinary text pieces.

**Implementation lens.**

A tokenizer is not a preprocessing detail that can be swapped freely. The embedding matrix, output projection, cached datasets, labels, special-token masks, and generation stop conditions all depend on the exact integer ids.

The most useful debugging habit is to print the text, tokens, ids, decoded text, and offsets for edge cases: leading spaces, repeated newlines, numbers, URLs, code, mixed scripts, and delimiter strings used by chat templates.

For model training, the tokenizer changes the effective curriculum. A tokenizer that splits common words into many pieces makes the model spend more positions modeling spelling-level structure. A tokenizer that memorizes very long pieces may save context but increase vocabulary cost and reduce compositional sharing.

For inference, the tokenizer changes both latency and price because attention cost grows with token length. A prompt that is short in words can still be expensive if it tokenizes poorly.

### 7.5 Safety and prompt boundaries

**Purpose.** Safety and prompt boundaries focuses on why control tokens need exact handling. In LLM systems this choice affects integer ids, sequence length, embedding rows, loss targets, and serving cost.

$$\operatorname{score}_{\mathrm{BPE}}(a,b)=\operatorname{count}(ab).$$

**Operational definition.**

This concept controls how raw text becomes the sequence of discrete units optimized by an LLM.

**Worked reading.**

The practical question is always how the choice changes ids, sequence length, reversibility, or downstream loss.

| Tokenizer object | Mathematical role | LLM consequence |
| --- | --- | --- |
| alphabet $\Sigma$ | atomic input symbols | bytes, characters, or normalized symbols |
| vocabulary $\mathcal{V}$ | finite token set | embedding and output-logit dimensions |
| encoder $E$ | maps text to ids | prompt length, training examples, costs |
| decoder $D$ | maps ids back to text | detokenization and round-trip safety |
| merge/probability table | segmentation rule | subword boundaries and rare-string handling |

Examples:

1. BPE pieces.
2. unigram pieces.
3. special tokens.

Non-examples:

1. raw text passed directly to attention.
2. word counts used as token counts.

**Derivation habit.**

1. State the raw alphabet and any normalization step.
2. State whether the model uses BPE, unigram, WordPiece, byte fallback, or a hybrid.
3. Compute token count before making cost, context, or memory claims.
4. Check reversibility with `decode(encode(x))` when the pipeline promises losslessness.
5. Treat special tokens as protected control symbols, not ordinary text pieces.

**Implementation lens.**

A tokenizer is not a preprocessing detail that can be swapped freely. The embedding matrix, output projection, cached datasets, labels, special-token masks, and generation stop conditions all depend on the exact integer ids.

The most useful debugging habit is to print the text, tokens, ids, decoded text, and offsets for edge cases: leading spaces, repeated newlines, numbers, URLs, code, mixed scripts, and delimiter strings used by chat templates.

For model training, the tokenizer changes the effective curriculum. A tokenizer that splits common words into many pieces makes the model spend more positions modeling spelling-level structure. A tokenizer that memorizes very long pieces may save context but increase vocabulary cost and reduce compositional sharing.

For inference, the tokenizer changes both latency and price because attention cost grows with token length. A prompt that is short in words can still be expensive if it tokenizes poorly.

## 8. Evaluation and Diagnostics

Evaluation and Diagnostics develops the tokenizer concepts needed before embeddings, attention, language-model probability, and serving tradeoffs can be understood correctly.

### 8.1 Round-trip tests

**Purpose.** Round-trip tests focuses on checking decode encode identity where promised. In LLM systems this choice affects integer ids, sequence length, embedding rows, loss targets, and serving cost.

$$\operatorname{attention\ cost}\propto n_{\mathrm{tokens}}^2.$$

**Operational definition.**

Tokenizer diagnostics catch failures before training: non-reversible text, unexpected unknowns, costly scripts, and broken control boundaries.

**Worked reading.**

A tokenizer migration changes ids, embeddings, logits, cached data, and often every downstream checkpoint assumption.

| Tokenizer object | Mathematical role | LLM consequence |
| --- | --- | --- |
| alphabet $\Sigma$ | atomic input symbols | bytes, characters, or normalized symbols |
| vocabulary $\mathcal{V}$ | finite token set | embedding and output-logit dimensions |
| encoder $E$ | maps text to ids | prompt length, training examples, costs |
| decoder $D$ | maps ids back to text | detokenization and round-trip safety |
| merge/probability table | segmentation rule | subword boundaries and rare-string handling |

Examples:

1. decode(encode(x)) tests.
2. offset mapping checks.
3. special-token boundary tests.

Non-examples:

1. only checking English prose.
2. changing tokenizers without retraining embeddings.

**Derivation habit.**

1. State the raw alphabet and any normalization step.
2. State whether the model uses BPE, unigram, WordPiece, byte fallback, or a hybrid.
3. Compute token count before making cost, context, or memory claims.
4. Check reversibility with `decode(encode(x))` when the pipeline promises losslessness.
5. Treat special tokens as protected control symbols, not ordinary text pieces.

**Implementation lens.**

A tokenizer is not a preprocessing detail that can be swapped freely. The embedding matrix, output projection, cached datasets, labels, special-token masks, and generation stop conditions all depend on the exact integer ids.

The most useful debugging habit is to print the text, tokens, ids, decoded text, and offsets for edge cases: leading spaces, repeated newlines, numbers, URLs, code, mixed scripts, and delimiter strings used by chat templates.

For model training, the tokenizer changes the effective curriculum. A tokenizer that splits common words into many pieces makes the model spend more positions modeling spelling-level structure. A tokenizer that memorizes very long pieces may save context but increase vocabulary cost and reduce compositional sharing.

For inference, the tokenizer changes both latency and price because attention cost grows with token length. A prompt that is short in words can still be expensive if it tokenizes poorly.

### 8.2 Coverage tests

**Purpose.** Coverage tests focuses on finding unknowns or byte fallback explosions. In LLM systems this choice affects integer ids, sequence length, embedding rows, loss targets, and serving cost.

$$E(x)=(t_1,\ldots,t_n),\qquad t_i\in\{0,\ldots,|\mathcal{V}|-1\}.$$

**Operational definition.**

Tokenizer diagnostics catch failures before training: non-reversible text, unexpected unknowns, costly scripts, and broken control boundaries.

**Worked reading.**

A tokenizer migration changes ids, embeddings, logits, cached data, and often every downstream checkpoint assumption.

| Tokenizer object | Mathematical role | LLM consequence |
| --- | --- | --- |
| alphabet $\Sigma$ | atomic input symbols | bytes, characters, or normalized symbols |
| vocabulary $\mathcal{V}$ | finite token set | embedding and output-logit dimensions |
| encoder $E$ | maps text to ids | prompt length, training examples, costs |
| decoder $D$ | maps ids back to text | detokenization and round-trip safety |
| merge/probability table | segmentation rule | subword boundaries and rare-string handling |

Examples:

1. decode(encode(x)) tests.
2. offset mapping checks.
3. special-token boundary tests.

Non-examples:

1. only checking English prose.
2. changing tokenizers without retraining embeddings.

**Derivation habit.**

1. State the raw alphabet and any normalization step.
2. State whether the model uses BPE, unigram, WordPiece, byte fallback, or a hybrid.
3. Compute token count before making cost, context, or memory claims.
4. Check reversibility with `decode(encode(x))` when the pipeline promises losslessness.
5. Treat special tokens as protected control symbols, not ordinary text pieces.

**Implementation lens.**

A tokenizer is not a preprocessing detail that can be swapped freely. The embedding matrix, output projection, cached datasets, labels, special-token masks, and generation stop conditions all depend on the exact integer ids.

The most useful debugging habit is to print the text, tokens, ids, decoded text, and offsets for edge cases: leading spaces, repeated newlines, numbers, URLs, code, mixed scripts, and delimiter strings used by chat templates.

For model training, the tokenizer changes the effective curriculum. A tokenizer that splits common words into many pieces makes the model spend more positions modeling spelling-level structure. A tokenizer that memorizes very long pieces may save context but increase vocabulary cost and reduce compositional sharing.

For inference, the tokenizer changes both latency and price because attention cost grows with token length. A prompt that is short in words can still be expensive if it tokenizes poorly.

### 8.3 Fertility dashboards

**Purpose.** Fertility dashboards focuses on comparing groups domains and scripts. In LLM systems this choice affects integer ids, sequence length, embedding rows, loss targets, and serving cost.

$$D(E(x))=x\quad\text{for a lossless tokenizer on its supported domain}.$$

**Operational definition.**

Tokenization is compression under constraints: it trades vocabulary size against sequence length and distribution balance.

**Worked reading.**

A lower token count improves context efficiency, but a huge vocabulary increases embedding and output-layer cost.

| Tokenizer object | Mathematical role | LLM consequence |
| --- | --- | --- |
| alphabet $\Sigma$ | atomic input symbols | bytes, characters, or normalized symbols |
| vocabulary $\mathcal{V}$ | finite token set | embedding and output-logit dimensions |
| encoder $E$ | maps text to ids | prompt length, training examples, costs |
| decoder $D$ | maps ids back to text | detokenization and round-trip safety |
| merge/probability table | segmentation rule | subword boundaries and rare-string handling |

Examples:

1. characters per token.
2. tokens per word.
3. entropy of token frequencies.

Non-examples:

1. judging cost by words alone.
2. ignoring sequence-length effects in attention.

**Derivation habit.**

1. State the raw alphabet and any normalization step.
2. State whether the model uses BPE, unigram, WordPiece, byte fallback, or a hybrid.
3. Compute token count before making cost, context, or memory claims.
4. Check reversibility with `decode(encode(x))` when the pipeline promises losslessness.
5. Treat special tokens as protected control symbols, not ordinary text pieces.

**Implementation lens.**

A tokenizer is not a preprocessing detail that can be swapped freely. The embedding matrix, output projection, cached datasets, labels, special-token masks, and generation stop conditions all depend on the exact integer ids.

The most useful debugging habit is to print the text, tokens, ids, decoded text, and offsets for edge cases: leading spaces, repeated newlines, numbers, URLs, code, mixed scripts, and delimiter strings used by chat templates.

For model training, the tokenizer changes the effective curriculum. A tokenizer that splits common words into many pieces makes the model spend more positions modeling spelling-level structure. A tokenizer that memorizes very long pieces may save context but increase vocabulary cost and reduce compositional sharing.

For inference, the tokenizer changes both latency and price because attention cost grows with token length. A prompt that is short in words can still be expensive if it tokenizes poorly.

### 8.4 Boundary tests

**Purpose.** Boundary tests focuses on URLs code numbers whitespace and emoji-like symbols. In LLM systems this choice affects integer ids, sequence length, embedding rows, loss targets, and serving cost.

$$\operatorname{score}_{\mathrm{BPE}}(a,b)=\operatorname{count}(ab).$$

**Operational definition.**

Tokenizer diagnostics catch failures before training: non-reversible text, unexpected unknowns, costly scripts, and broken control boundaries.

**Worked reading.**

A tokenizer migration changes ids, embeddings, logits, cached data, and often every downstream checkpoint assumption.

| Tokenizer object | Mathematical role | LLM consequence |
| --- | --- | --- |
| alphabet $\Sigma$ | atomic input symbols | bytes, characters, or normalized symbols |
| vocabulary $\mathcal{V}$ | finite token set | embedding and output-logit dimensions |
| encoder $E$ | maps text to ids | prompt length, training examples, costs |
| decoder $D$ | maps ids back to text | detokenization and round-trip safety |
| merge/probability table | segmentation rule | subword boundaries and rare-string handling |

Examples:

1. decode(encode(x)) tests.
2. offset mapping checks.
3. special-token boundary tests.

Non-examples:

1. only checking English prose.
2. changing tokenizers without retraining embeddings.

**Derivation habit.**

1. State the raw alphabet and any normalization step.
2. State whether the model uses BPE, unigram, WordPiece, byte fallback, or a hybrid.
3. Compute token count before making cost, context, or memory claims.
4. Check reversibility with `decode(encode(x))` when the pipeline promises losslessness.
5. Treat special tokens as protected control symbols, not ordinary text pieces.

**Implementation lens.**

A tokenizer is not a preprocessing detail that can be swapped freely. The embedding matrix, output projection, cached datasets, labels, special-token masks, and generation stop conditions all depend on the exact integer ids.

The most useful debugging habit is to print the text, tokens, ids, decoded text, and offsets for edge cases: leading spaces, repeated newlines, numbers, URLs, code, mixed scripts, and delimiter strings used by chat templates.

For model training, the tokenizer changes the effective curriculum. A tokenizer that splits common words into many pieces makes the model spend more positions modeling spelling-level structure. A tokenizer that memorizes very long pieces may save context but increase vocabulary cost and reduce compositional sharing.

For inference, the tokenizer changes both latency and price because attention cost grows with token length. A prompt that is short in words can still be expensive if it tokenizes poorly.

### 8.5 Tokenizer migration tests

**Purpose.** Tokenizer migration tests focuses on why changing tokenizers invalidates checkpoints. In LLM systems this choice affects integer ids, sequence length, embedding rows, loss targets, and serving cost.

$$x^*=\arg\max_{v_1,\ldots,v_k:\,v_1\cdots v_k=x}\sum_{i=1}^k\log p(v_i).$$

**Operational definition.**

Tokenizer diagnostics catch failures before training: non-reversible text, unexpected unknowns, costly scripts, and broken control boundaries.

**Worked reading.**

A tokenizer migration changes ids, embeddings, logits, cached data, and often every downstream checkpoint assumption.

| Tokenizer object | Mathematical role | LLM consequence |
| --- | --- | --- |
| alphabet $\Sigma$ | atomic input symbols | bytes, characters, or normalized symbols |
| vocabulary $\mathcal{V}$ | finite token set | embedding and output-logit dimensions |
| encoder $E$ | maps text to ids | prompt length, training examples, costs |
| decoder $D$ | maps ids back to text | detokenization and round-trip safety |
| merge/probability table | segmentation rule | subword boundaries and rare-string handling |

Examples:

1. decode(encode(x)) tests.
2. offset mapping checks.
3. special-token boundary tests.

Non-examples:

1. only checking English prose.
2. changing tokenizers without retraining embeddings.

**Derivation habit.**

1. State the raw alphabet and any normalization step.
2. State whether the model uses BPE, unigram, WordPiece, byte fallback, or a hybrid.
3. Compute token count before making cost, context, or memory claims.
4. Check reversibility with `decode(encode(x))` when the pipeline promises losslessness.
5. Treat special tokens as protected control symbols, not ordinary text pieces.

**Implementation lens.**

A tokenizer is not a preprocessing detail that can be swapped freely. The embedding matrix, output projection, cached datasets, labels, special-token masks, and generation stop conditions all depend on the exact integer ids.

The most useful debugging habit is to print the text, tokens, ids, decoded text, and offsets for edge cases: leading spaces, repeated newlines, numbers, URLs, code, mixed scripts, and delimiter strings used by chat templates.

For model training, the tokenizer changes the effective curriculum. A tokenizer that splits common words into many pieces makes the model spend more positions modeling spelling-level structure. A tokenizer that memorizes very long pieces may save context but increase vocabulary cost and reduce compositional sharing.

For inference, the tokenizer changes both latency and price because attention cost grows with token length. A prompt that is short in words can still be expensive if it tokenizes poorly.

## 9. Common Mistakes

| # | Mistake | Why it is wrong | Fix |
| --- | --- | --- | --- |
| 1 | Assuming words are tokens | Modern LLMs usually use subword or byte-level tokens. | Inspect actual token ids before estimating cost or behavior. |
| 2 | Ignoring reversibility | Normalization can make decode-encode behavior lossy. | State whether the tokenizer is byte-level reversible or normalized. |
| 3 | Changing tokenizers after training | Embeddings and output heads are tied to token ids. | Treat tokenizer choice as part of the checkpoint. |
| 4 | Comparing context windows by characters | Models attend over tokens, not characters. | Measure tokens per sample and fertility. |
| 5 | Forgetting special tokens | Control tokens change sequence boundaries and masks. | Reserve and test special ids explicitly. |
| 6 | Assuming all languages pay the same token cost | Scripts and training data frequency affect fertility. | Audit multilingual fertility and bytes-per-token. |
| 7 | Using unknown tokens silently | UNK loses information and can hide coverage failures. | Prefer byte fallback or explicit coverage reports. |
| 8 | Treating BPE merges as globally optimal | BPE is greedy and merge-order dependent. | Use diagnostics and compare with unigram/WordPiece behavior. |
| 9 | Chunking retrieval by characters only | The model budget is token-limited. | Chunk by token count with overlap in token space. |
| 10 | Ignoring whitespace | Whitespace handling changes ids, offsets, and detokenization. | Test leading spaces, newlines, tabs, and code blocks. |

## 10. Exercises

1. (*) Run two BPE merges by hand on a tiny corpus.
   - (a) State the tokenizer object involved.
   - (b) Compute the small numeric or string example.
   - (c) Explain the LLM training or serving consequence.

2. (*) Compute characters-per-token before and after a merge.
   - (a) State the tokenizer object involved.
   - (b) Compute the small numeric or string example.
   - (c) Explain the LLM training or serving consequence.

3. (*) Compute embedding parameter cost for two vocabulary sizes.
   - (a) State the tokenizer object involved.
   - (b) Compute the small numeric or string example.
   - (c) Explain the LLM training or serving consequence.

4. (**) Find the best unigram segmentation with dynamic programming.
   - (a) State the tokenizer object involved.
   - (b) Compute the small numeric or string example.
   - (c) Explain the LLM training or serving consequence.

5. (**) Compute token entropy from token counts.
   - (a) State the tokenizer object involved.
   - (b) Compute the small numeric or string example.
   - (c) Explain the LLM training or serving consequence.

6. (**) Compare fertility for two short multilingual examples.
   - (a) State the tokenizer object involved.
   - (b) Compute the small numeric or string example.
   - (c) Explain the LLM training or serving consequence.

7. (**) Design a round-trip test for whitespace and code.
   - (a) State the tokenizer object involved.
   - (b) Compute the small numeric or string example.
   - (c) Explain the LLM training or serving consequence.

8. (***) Compute attention-cost growth when token count doubles.
   - (a) State the tokenizer object involved.
   - (b) Compute the small numeric or string example.
   - (c) Explain the LLM training or serving consequence.

9. (***) Identify which strings must be protected as special tokens.
   - (a) State the tokenizer object involved.
   - (b) Compute the small numeric or string example.
   - (c) Explain the LLM training or serving consequence.

10. (***) Explain why tokenizer migration changes a trained checkpoint.
   - (a) State the tokenizer object involved.
   - (b) Compute the small numeric or string example.
   - (c) Explain the LLM training or serving consequence.

## 11. Why This Matters for AI

| Concept | AI impact |
| --- | --- |
| Token ids | Define the rows of embedding matrices and the columns of output logits. |
| Vocabulary size | Controls embedding parameters, softmax cost, and rare-piece coverage. |
| Sequence length | Controls attention compute, memory, context-window utilization, and API cost. |
| Subword segmentation | Determines how words, names, code, numbers, and rare strings are decomposed. |
| Byte fallback | Improves robustness to arbitrary text and reduces unknown-token failures. |
| Special tokens | Encode conversation roles, tools, padding, sequence boundaries, and safety delimiters. |
| Fertility | Reveals fairness and cost differences across languages, domains, and scripts. |
| Round-trip behavior | Protects data pipelines from silent corruption before training or inference. |

## 12. Conceptual Bridge

The backward bridge is information theory: tokenization is compression with a finite codebook, but unlike pure compression it must also support neural prediction, stable ids, and clean detokenization.

The forward bridge is embedding space. Once a tokenizer emits ids, each id selects a row of an embedding matrix. Attention, next-token probability, scaling laws, RAG chunking, and serving cost all inherit the tokenizer's sequence length and boundary choices.

```text
+-----------+      +--------------+      +--------------+      +------------+
| raw text  | ---> | token ids    | ---> | embeddings   | ---> | attention  |
| bytes     |      | finite vocab |      | vector rows  |      | positions  |
+-----------+      +--------------+      +--------------+      +------------+
```

The practical habit is to inspect tokens before trusting intuition. If the model behaves strangely on numbers, names, code, or multilingual text, the tokenizer is one of the first places to look.

## References

- Sennrich, Haddow, Birch. Neural Machine Translation of Rare Words with Subword Units. https://aclanthology.org/P16-1162/
- Kudo and Richardson. SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer. https://arxiv.org/abs/1808.06226
- Google. SentencePiece repository. https://github.com/google/sentencepiece
- Hugging Face. Tokenization algorithms. https://huggingface.co/docs/transformers/tokenizer_summary
- OpenAI. tiktoken tokenizer library. https://github.com/openai/tiktoken
