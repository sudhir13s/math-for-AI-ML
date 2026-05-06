[Previous: Tokenization Math](../01-Tokenization-Math/notes.md) | [Back to Math for LLMs](../README.md) | [Next: Attention Mechanism Math](../03-Attention-Mechanism-Math/notes.md)

---

# Embedding Space Math

> _"An embedding table is where arbitrary ids become trainable geometry."_

## Overview

Embedding space is the bridge between tokenization and transformer computation. Tokenization emits integer ids; the embedding matrix turns those ids into vectors that can be compared, updated, projected into queries and keys, and passed through the residual stream.

The central mathematical idea is that a finite vocabulary can be represented as rows of a matrix. Similarity, analogy, anisotropy, position encoding, output logits, and retrieval all become questions about vector geometry and matrix multiplication.

This section uses LaTeX Markdown with `$...$` and `$$...$$`. The notebooks build small synthetic embedding spaces so every formula can be checked without external model weights.

## Prerequisites

- [Tokenization Math](../01-Tokenization-Math/notes.md)
- [Vectors and Spaces](../../02-Linear-Algebra-Basics/01-Vectors-and-Spaces/notes.md)
- [Matrix Operations](../../02-Linear-Algebra-Basics/02-Matrix-Operations/notes.md)
- [Cosine similarity and norms](../../03-Advanced-Linear-Algebra/06-Matrix-Norms/notes.md)
- [Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Executable demonstrations of lookup, similarity, analogy, anisotropy, gradients, positions, RoPE, ALiBi, PCA, and parameter counts. |
| [exercises.ipynb](exercises.ipynb) | Ten checked practice problems for embedding geometry and LLM systems. |

## Learning Objectives

After completing this section, you will be able to:

- Define the embedding matrix and compute its parameter count.
- Show why one-hot multiplication and row lookup are equivalent.
- Track tensor shapes from token ids to hidden states.
- Compute dot product, cosine similarity, Euclidean distance, and nearest neighbors.
- Explain analogy directions, subspaces, isotropy, and anisotropy.
- Derive the softmax gradient that updates output embedding rows.
- Compare word2vec, GloVe, and transformer embedding training signals.
- Implement sinusoidal, rotary, learned, and ALiBi-style position signals in small examples.
- Explain how embeddings become queries, keys, values, and contextual hidden states.
- Diagnose embedding norms, outliers, special tokens, and tokenizer-embedding compatibility.

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 From token ids to vectors](#11-from-token-ids-to-vectors)
  - [1.2 Why continuous geometry helps](#12-why-continuous-geometry-helps)
  - [1.3 Embedding space as model memory](#13-embedding-space-as-model-memory)
  - [1.4 Static versus contextual meaning](#14-static-versus-contextual-meaning)
  - [1.5 Pipeline position after tokenization](#15-pipeline-position-after-tokenization)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Embedding matrix](#21-embedding-matrix)
  - [2.2 One-hot lookup equivalence](#22-onehot-lookup-equivalence)
  - [2.3 Batch sequence tensor shapes](#23-batch-sequence-tensor-shapes)
  - [2.4 Input output and tied embeddings](#24-input-output-and-tied-embeddings)
  - [2.5 Residual stream initialization](#25-residual-stream-initialization)
- [3. Similarity Geometry](#3-similarity-geometry)
  - [3.1 Dot product](#31-dot-product)
  - [3.2 Cosine similarity](#32-cosine-similarity)
  - [3.3 Euclidean distance](#33-euclidean-distance)
  - [3.4 Norms and frequency effects](#34-norms-and-frequency-effects)
  - [3.5 Nearest neighbors](#35-nearest-neighbors)
- [4. Geometry of Meaning](#4-geometry-of-meaning)
  - [4.1 Analogy directions](#41-analogy-directions)
  - [4.2 Subspaces and probes](#42-subspaces-and-probes)
  - [4.3 Isotropy and anisotropy](#43-isotropy-and-anisotropy)
  - [4.4 Centering and whitening](#44-centering-and-whitening)
  - [4.5 Bias and representation directions](#45-bias-and-representation-directions)
- [5. Training Embeddings](#5-training-embeddings)
  - [5.1 Language-model loss gradients](#51-languagemodel-loss-gradients)
  - [5.2 Word2vec and negative sampling](#52-word2vec-and-negative-sampling)
  - [5.3 GloVe and co-occurrence factorization](#53-glove-and-cooccurrence-factorization)
  - [5.4 Fine-tuning drift](#54-finetuning-drift)
  - [5.5 Vocabulary resizing](#55-vocabulary-resizing)
- [6. Positional Information](#6-positional-information)
  - [6.1 Why positions are needed](#61-why-positions-are-needed)
  - [6.2 Sinusoidal positional encodings](#62-sinusoidal-positional-encodings)
  - [6.3 Learned positional embeddings](#63-learned-positional-embeddings)
  - [6.4 Rotary positional embeddings](#64-rotary-positional-embeddings)
  - [6.5 ALiBi and relative biases](#65-alibi-and-relative-biases)
- [7. Attention Bridge](#7-attention-bridge)
  - [7.1 Query key value projections](#71-query-key-value-projections)
  - [7.2 Attention as soft nearest-neighbor lookup](#72-attention-as-soft-nearestneighbor-lookup)
  - [7.3 Layer-wise contextualization](#73-layerwise-contextualization)
  - [7.4 Dimensionality reduction diagnostics](#74-dimensionality-reduction-diagnostics)
  - [7.5 Embedding geometry in RAG](#75-embedding-geometry-in-rag)
- [8. Scale and Diagnostics](#8-scale-and-diagnostics)
  - [8.1 Parameter counting](#81-parameter-counting)
  - [8.2 Memory and quantization](#82-memory-and-quantization)
  - [8.3 Norm and similarity dashboards](#83-norm-and-similarity-dashboards)
  - [8.4 Outlier tokens and special tokens](#84-outlier-tokens-and-special-tokens)
  - [8.5 Migration and compatibility tests](#85-migration-and-compatibility-tests)
- [9. Common Mistakes](#9-common-mistakes)
- [10. Exercises](#10-exercises)
- [11. Why This Matters for AI](#11-why-this-matters-for-ai)
- [12. Conceptual Bridge](#12-conceptual-bridge)
- [References](#references)

---

## 1. Intuition

Intuition connects token ids to continuous vectors and prepares the exact geometry used by attention, language-model logits, and dense retrieval.

### 1.1 From token ids to vectors

**Purpose.** From token ids to vectors focuses on why lookup tables are the first neural layer. This matters because every later transformer operation starts from these vectors or from hidden states derived from them.

$$E\in\mathbb{R}^{|\mathcal{V}|\times d_{\mathrm{model}}},\qquad \mathbf{x}_t=E_{i_t,:}.$$

**Operational definition.**

Token ids are discrete indices. The embedding matrix turns each id into a trainable vector so gradient-based neural networks can process language.

**Worked reading.**

For ids `[3, 7, 7]`, lookup returns three rows of $E$. Repeated ids reuse the same row before contextual layers make their states differ.

| Object | Shape or formula | Role |
| --- | --- | --- |
| token ids | $B\times T$ | discrete sequence from tokenizer |
| embedding table | $|\mathcal{V}|\times d$ | row lookup for each token id |
| hidden states | $B\times T\times d$ | contextual vectors after lookup and layers |
| LM head | $d\times |\mathcal{V}|$ or tied $E^\top$ | maps hidden states to token logits |
| position signal | vector, rotation, or bias | injects order into attention |

Examples:

1. input embedding lookup.
2. LM-head row selection.
3. token-level residual stream initialization.

Non-examples:

1. raw strings inside attention.
2. one scalar id treated as an ordinal number.

**Derivation habit.**

1. Write the tensor shape before writing the operation.
2. State whether vectors are raw input embeddings, hidden states, output rows, or retrieval embeddings.
3. Choose dot product, cosine similarity, or Euclidean distance deliberately.
4. Check whether position information is additive, rotary, learned, or an attention bias.
5. Track whether input and output embeddings are tied.

**Implementation lens.**

In code, embedding lookup is simple indexing. Conceptually, it is the step where the model stops seeing symbolic ids and starts seeing trainable vectors. That is why tokenizer changes, vocabulary resizing, and special-token handling are not superficial.

When debugging model behavior, inspect embedding norms, nearest neighbors, and the mean direction. Large norms or dominant components can affect logits and similarity search. For retrieval models, normalize vectors before cosine search unless the training objective explicitly uses norm as signal.

For transformer internals, remember that input embeddings are only the first residual-stream state. After attention and MLP layers, a token position's hidden state reflects surrounding context. Static nearest neighbors and contextual hidden-state probes answer different questions.

### 1.2 Why continuous geometry helps

**Purpose.** Why continuous geometry helps focuses on similarity, sharing, and smooth optimization. This matters because every later transformer operation starts from these vectors or from hidden states derived from them.

$$\mathbf{x}_t=\mathbf{e}_{i_t}^{\top}E.$$

**Operational definition.**

This concept explains how discrete language symbols become continuous vectors with trainable geometry.

**Worked reading.**

The operational question is what shape the vector has, how it is compared, and how training changes it.

| Object | Shape or formula | Role |
| --- | --- | --- |
| token ids | $B\times T$ | discrete sequence from tokenizer |
| embedding table | $|\mathcal{V}|\times d$ | row lookup for each token id |
| hidden states | $B\times T\times d$ | contextual vectors after lookup and layers |
| LM head | $d\times |\mathcal{V}|$ or tied $E^\top$ | maps hidden states to token logits |
| position signal | vector, rotation, or bias | injects order into attention |

Examples:

1. embedding rows.
2. hidden states.
3. similarity search.

Non-examples:

1. raw text in linear algebra.
2. ids treated as distances.

**Derivation habit.**

1. Write the tensor shape before writing the operation.
2. State whether vectors are raw input embeddings, hidden states, output rows, or retrieval embeddings.
3. Choose dot product, cosine similarity, or Euclidean distance deliberately.
4. Check whether position information is additive, rotary, learned, or an attention bias.
5. Track whether input and output embeddings are tied.

**Implementation lens.**

In code, embedding lookup is simple indexing. Conceptually, it is the step where the model stops seeing symbolic ids and starts seeing trainable vectors. That is why tokenizer changes, vocabulary resizing, and special-token handling are not superficial.

When debugging model behavior, inspect embedding norms, nearest neighbors, and the mean direction. Large norms or dominant components can affect logits and similarity search. For retrieval models, normalize vectors before cosine search unless the training objective explicitly uses norm as signal.

For transformer internals, remember that input embeddings are only the first residual-stream state. After attention and MLP layers, a token position's hidden state reflects surrounding context. Static nearest neighbors and contextual hidden-state probes answer different questions.

### 1.3 Embedding space as model memory

**Purpose.** Embedding space as model memory focuses on what can be stored in rows and directions. This matters because every later transformer operation starts from these vectors or from hidden states derived from them.

$$\operatorname{cos}(\mathbf{u},\mathbf{v})=\frac{\langle \mathbf{u},\mathbf{v}\rangle}{\lVert\mathbf{u}\rVert_2\lVert\mathbf{v}\rVert_2}.$$

**Operational definition.**

Embedding tables are systems objects too: they consume memory, depend on tokenizer ids, and must handle special rows carefully.

**Worked reading.**

Changing a tokenizer changes which row each token id selects, so old weights no longer mean the same thing without migration.

| Object | Shape or formula | Role |
| --- | --- | --- |
| token ids | $B\times T$ | discrete sequence from tokenizer |
| embedding table | $|\mathcal{V}|\times d$ | row lookup for each token id |
| hidden states | $B\times T\times d$ | contextual vectors after lookup and layers |
| LM head | $d\times |\mathcal{V}|$ or tied $E^\top$ | maps hidden states to token logits |
| position signal | vector, rotation, or bias | injects order into attention |

Examples:

1. vocabulary resizing.
2. special token initialization.
3. embedding quantization.

Non-examples:

1. renaming token ids without changing weights.
2. ignoring padding row behavior.

**Derivation habit.**

1. Write the tensor shape before writing the operation.
2. State whether vectors are raw input embeddings, hidden states, output rows, or retrieval embeddings.
3. Choose dot product, cosine similarity, or Euclidean distance deliberately.
4. Check whether position information is additive, rotary, learned, or an attention bias.
5. Track whether input and output embeddings are tied.

**Implementation lens.**

In code, embedding lookup is simple indexing. Conceptually, it is the step where the model stops seeing symbolic ids and starts seeing trainable vectors. That is why tokenizer changes, vocabulary resizing, and special-token handling are not superficial.

When debugging model behavior, inspect embedding norms, nearest neighbors, and the mean direction. Large norms or dominant components can affect logits and similarity search. For retrieval models, normalize vectors before cosine search unless the training objective explicitly uses norm as signal.

For transformer internals, remember that input embeddings are only the first residual-stream state. After attention and MLP layers, a token position's hidden state reflects surrounding context. Static nearest neighbors and contextual hidden-state probes answer different questions.

### 1.4 Static versus contextual meaning

**Purpose.** Static versus contextual meaning focuses on why a token row is only the starting representation. This matters because every later transformer operation starts from these vectors or from hidden states derived from them.

$$\ell=-\log\frac{\exp(\mathbf{h}^{\top}\mathbf{w}_y)}{\sum_{j\in\mathcal{V}}\exp(\mathbf{h}^{\top}\mathbf{w}_j)}.$$

**Operational definition.**

This concept explains how discrete language symbols become continuous vectors with trainable geometry.

**Worked reading.**

The operational question is what shape the vector has, how it is compared, and how training changes it.

| Object | Shape or formula | Role |
| --- | --- | --- |
| token ids | $B\times T$ | discrete sequence from tokenizer |
| embedding table | $|\mathcal{V}|\times d$ | row lookup for each token id |
| hidden states | $B\times T\times d$ | contextual vectors after lookup and layers |
| LM head | $d\times |\mathcal{V}|$ or tied $E^\top$ | maps hidden states to token logits |
| position signal | vector, rotation, or bias | injects order into attention |

Examples:

1. embedding rows.
2. hidden states.
3. similarity search.

Non-examples:

1. raw text in linear algebra.
2. ids treated as distances.

**Derivation habit.**

1. Write the tensor shape before writing the operation.
2. State whether vectors are raw input embeddings, hidden states, output rows, or retrieval embeddings.
3. Choose dot product, cosine similarity, or Euclidean distance deliberately.
4. Check whether position information is additive, rotary, learned, or an attention bias.
5. Track whether input and output embeddings are tied.

**Implementation lens.**

In code, embedding lookup is simple indexing. Conceptually, it is the step where the model stops seeing symbolic ids and starts seeing trainable vectors. That is why tokenizer changes, vocabulary resizing, and special-token handling are not superficial.

When debugging model behavior, inspect embedding norms, nearest neighbors, and the mean direction. Large norms or dominant components can affect logits and similarity search. For retrieval models, normalize vectors before cosine search unless the training objective explicitly uses norm as signal.

For transformer internals, remember that input embeddings are only the first residual-stream state. After attention and MLP layers, a token position's hidden state reflects surrounding context. Static nearest neighbors and contextual hidden-state probes answer different questions.

### 1.5 Pipeline position after tokenization

**Purpose.** Pipeline position after tokenization focuses on how ids become residual-stream states. This matters because every later transformer operation starts from these vectors or from hidden states derived from them.

$$\nabla_{\mathbf{w}_j}\ell=(p_j-\mathbf{1}\{j=y\})\mathbf{h}.$$

**Operational definition.**

Position mechanisms inject order into otherwise content-based attention. They may be additive vectors, rotations, or attention-score biases.

**Worked reading.**

Sinusoidal encodings add fixed frequency features; RoPE rotates query/key pairs; ALiBi adds a distance-dependent bias to attention logits.

| Object | Shape or formula | Role |
| --- | --- | --- |
| token ids | $B\times T$ | discrete sequence from tokenizer |
| embedding table | $|\mathcal{V}|\times d$ | row lookup for each token id |
| hidden states | $B\times T\times d$ | contextual vectors after lookup and layers |
| LM head | $d\times |\mathcal{V}|$ or tied $E^\top$ | maps hidden states to token logits |
| position signal | vector, rotation, or bias | injects order into attention |

Examples:

1. sinusoidal encodings.
2. learned position rows.
3. RoPE.
4. ALiBi.

Non-examples:

1. bag-of-token attention with no order.
2. token ids used as positions.

**Derivation habit.**

1. Write the tensor shape before writing the operation.
2. State whether vectors are raw input embeddings, hidden states, output rows, or retrieval embeddings.
3. Choose dot product, cosine similarity, or Euclidean distance deliberately.
4. Check whether position information is additive, rotary, learned, or an attention bias.
5. Track whether input and output embeddings are tied.

**Implementation lens.**

In code, embedding lookup is simple indexing. Conceptually, it is the step where the model stops seeing symbolic ids and starts seeing trainable vectors. That is why tokenizer changes, vocabulary resizing, and special-token handling are not superficial.

When debugging model behavior, inspect embedding norms, nearest neighbors, and the mean direction. Large norms or dominant components can affect logits and similarity search. For retrieval models, normalize vectors before cosine search unless the training objective explicitly uses norm as signal.

For transformer internals, remember that input embeddings are only the first residual-stream state. After attention and MLP layers, a token position's hidden state reflects surrounding context. Static nearest neighbors and contextual hidden-state probes answer different questions.

## 2. Formal Definitions

Formal Definitions connects token ids to continuous vectors and prepares the exact geometry used by attention, language-model logits, and dense retrieval.

### 2.1 Embedding matrix

**Purpose.** Embedding matrix focuses on the table $E\in\mathbb{R}^{|\mathcal{V}|\times d}$. This matters because every later transformer operation starts from these vectors or from hidden states derived from them.

$$\mathbf{x}_t=\mathbf{e}_{i_t}^{\top}E.$$

**Operational definition.**

The embedding matrix is a table $E\in\mathbb{R}^{|\mathcal{V}|\times d}$ whose row $E_{i,:}$ is the vector for token id $i$.

**Worked reading.**

A vocabulary of 50,000 and width 4096 already has over 200 million input-embedding parameters if untied from the output head.

| Object | Shape or formula | Role |
| --- | --- | --- |
| token ids | $B\times T$ | discrete sequence from tokenizer |
| embedding table | $|\mathcal{V}|\times d$ | row lookup for each token id |
| hidden states | $B\times T\times d$ | contextual vectors after lookup and layers |
| LM head | $d\times |\mathcal{V}|$ or tied $E^\top$ | maps hidden states to token logits |
| position signal | vector, rotation, or bias | injects order into attention |

Examples:

1. GPT-style token embeddings.
2. BERT wordpiece embeddings.
3. new rows after vocabulary expansion.

Non-examples:

1. a dictionary from words to definitions.
2. a single dense vector for an entire sentence.

**Derivation habit.**

1. Write the tensor shape before writing the operation.
2. State whether vectors are raw input embeddings, hidden states, output rows, or retrieval embeddings.
3. Choose dot product, cosine similarity, or Euclidean distance deliberately.
4. Check whether position information is additive, rotary, learned, or an attention bias.
5. Track whether input and output embeddings are tied.

**Implementation lens.**

In code, embedding lookup is simple indexing. Conceptually, it is the step where the model stops seeing symbolic ids and starts seeing trainable vectors. That is why tokenizer changes, vocabulary resizing, and special-token handling are not superficial.

When debugging model behavior, inspect embedding norms, nearest neighbors, and the mean direction. Large norms or dominant components can affect logits and similarity search. For retrieval models, normalize vectors before cosine search unless the training objective explicitly uses norm as signal.

For transformer internals, remember that input embeddings are only the first residual-stream state. After attention and MLP layers, a token position's hidden state reflects surrounding context. Static nearest neighbors and contextual hidden-state probes answer different questions.

### 2.2 One-hot lookup equivalence

**Purpose.** One-hot lookup equivalence focuses on why $\mathbf{e}_i^\top E$ selects a row. This matters because every later transformer operation starts from these vectors or from hidden states derived from them.

$$\operatorname{cos}(\mathbf{u},\mathbf{v})=\frac{\langle \mathbf{u},\mathbf{v}\rangle}{\lVert\mathbf{u}\rVert_2\lVert\mathbf{v}\rVert_2}.$$

**Operational definition.**

This concept explains how discrete language symbols become continuous vectors with trainable geometry.

**Worked reading.**

The operational question is what shape the vector has, how it is compared, and how training changes it.

| Object | Shape or formula | Role |
| --- | --- | --- |
| token ids | $B\times T$ | discrete sequence from tokenizer |
| embedding table | $|\mathcal{V}|\times d$ | row lookup for each token id |
| hidden states | $B\times T\times d$ | contextual vectors after lookup and layers |
| LM head | $d\times |\mathcal{V}|$ or tied $E^\top$ | maps hidden states to token logits |
| position signal | vector, rotation, or bias | injects order into attention |

Examples:

1. embedding rows.
2. hidden states.
3. similarity search.

Non-examples:

1. raw text in linear algebra.
2. ids treated as distances.

**Derivation habit.**

1. Write the tensor shape before writing the operation.
2. State whether vectors are raw input embeddings, hidden states, output rows, or retrieval embeddings.
3. Choose dot product, cosine similarity, or Euclidean distance deliberately.
4. Check whether position information is additive, rotary, learned, or an attention bias.
5. Track whether input and output embeddings are tied.

**Implementation lens.**

In code, embedding lookup is simple indexing. Conceptually, it is the step where the model stops seeing symbolic ids and starts seeing trainable vectors. That is why tokenizer changes, vocabulary resizing, and special-token handling are not superficial.

When debugging model behavior, inspect embedding norms, nearest neighbors, and the mean direction. Large norms or dominant components can affect logits and similarity search. For retrieval models, normalize vectors before cosine search unless the training objective explicitly uses norm as signal.

For transformer internals, remember that input embeddings are only the first residual-stream state. After attention and MLP layers, a token position's hidden state reflects surrounding context. Static nearest neighbors and contextual hidden-state probes answer different questions.

### 2.3 Batch sequence tensor shapes

**Purpose.** Batch sequence tensor shapes focuses on how ids become $B\times T\times d$ arrays. This matters because every later transformer operation starts from these vectors or from hidden states derived from them.

$$\ell=-\log\frac{\exp(\mathbf{h}^{\top}\mathbf{w}_y)}{\sum_{j\in\mathcal{V}}\exp(\mathbf{h}^{\top}\mathbf{w}_j)}.$$

**Operational definition.**

This concept explains how discrete language symbols become continuous vectors with trainable geometry.

**Worked reading.**

The operational question is what shape the vector has, how it is compared, and how training changes it.

| Object | Shape or formula | Role |
| --- | --- | --- |
| token ids | $B\times T$ | discrete sequence from tokenizer |
| embedding table | $|\mathcal{V}|\times d$ | row lookup for each token id |
| hidden states | $B\times T\times d$ | contextual vectors after lookup and layers |
| LM head | $d\times |\mathcal{V}|$ or tied $E^\top$ | maps hidden states to token logits |
| position signal | vector, rotation, or bias | injects order into attention |

Examples:

1. embedding rows.
2. hidden states.
3. similarity search.

Non-examples:

1. raw text in linear algebra.
2. ids treated as distances.

**Derivation habit.**

1. Write the tensor shape before writing the operation.
2. State whether vectors are raw input embeddings, hidden states, output rows, or retrieval embeddings.
3. Choose dot product, cosine similarity, or Euclidean distance deliberately.
4. Check whether position information is additive, rotary, learned, or an attention bias.
5. Track whether input and output embeddings are tied.

**Implementation lens.**

In code, embedding lookup is simple indexing. Conceptually, it is the step where the model stops seeing symbolic ids and starts seeing trainable vectors. That is why tokenizer changes, vocabulary resizing, and special-token handling are not superficial.

When debugging model behavior, inspect embedding norms, nearest neighbors, and the mean direction. Large norms or dominant components can affect logits and similarity search. For retrieval models, normalize vectors before cosine search unless the training objective explicitly uses norm as signal.

For transformer internals, remember that input embeddings are only the first residual-stream state. After attention and MLP layers, a token position's hidden state reflects surrounding context. Static nearest neighbors and contextual hidden-state probes answer different questions.

### 2.4 Input output and tied embeddings

**Purpose.** Input output and tied embeddings focuses on when the same matrix is reused for logits. This matters because every later transformer operation starts from these vectors or from hidden states derived from them.

$$\nabla_{\mathbf{w}_j}\ell=(p_j-\mathbf{1}\{j=y\})\mathbf{h}.$$

**Operational definition.**

This concept explains how discrete language symbols become continuous vectors with trainable geometry.

**Worked reading.**

The operational question is what shape the vector has, how it is compared, and how training changes it.

| Object | Shape or formula | Role |
| --- | --- | --- |
| token ids | $B\times T$ | discrete sequence from tokenizer |
| embedding table | $|\mathcal{V}|\times d$ | row lookup for each token id |
| hidden states | $B\times T\times d$ | contextual vectors after lookup and layers |
| LM head | $d\times |\mathcal{V}|$ or tied $E^\top$ | maps hidden states to token logits |
| position signal | vector, rotation, or bias | injects order into attention |

Examples:

1. embedding rows.
2. hidden states.
3. similarity search.

Non-examples:

1. raw text in linear algebra.
2. ids treated as distances.

**Derivation habit.**

1. Write the tensor shape before writing the operation.
2. State whether vectors are raw input embeddings, hidden states, output rows, or retrieval embeddings.
3. Choose dot product, cosine similarity, or Euclidean distance deliberately.
4. Check whether position information is additive, rotary, learned, or an attention bias.
5. Track whether input and output embeddings are tied.

**Implementation lens.**

In code, embedding lookup is simple indexing. Conceptually, it is the step where the model stops seeing symbolic ids and starts seeing trainable vectors. That is why tokenizer changes, vocabulary resizing, and special-token handling are not superficial.

When debugging model behavior, inspect embedding norms, nearest neighbors, and the mean direction. Large norms or dominant components can affect logits and similarity search. For retrieval models, normalize vectors before cosine search unless the training objective explicitly uses norm as signal.

For transformer internals, remember that input embeddings are only the first residual-stream state. After attention and MLP layers, a token position's hidden state reflects surrounding context. Static nearest neighbors and contextual hidden-state probes answer different questions.

### 2.5 Residual stream initialization

**Purpose.** Residual stream initialization focuses on embedding plus positional information. This matters because every later transformer operation starts from these vectors or from hidden states derived from them.

$$\operatorname{PE}_{p,2k}=\sin\left(p/10000^{2k/d}\right),\qquad \operatorname{PE}_{p,2k+1}=\cos\left(p/10000^{2k/d}\right).$$

**Operational definition.**

Embeddings become contextual through projections, attention, MLPs, and residual additions. Attention compares projected vectors, not raw token ids.

**Worked reading.**

The query and key projections turn hidden states into vectors whose dot products define attention weights.

| Object | Shape or formula | Role |
| --- | --- | --- |
| token ids | $B\times T$ | discrete sequence from tokenizer |
| embedding table | $|\mathcal{V}|\times d$ | row lookup for each token id |
| hidden states | $B\times T\times d$ | contextual vectors after lookup and layers |
| LM head | $d\times |\mathcal{V}|$ or tied $E^\top$ | maps hidden states to token logits |
| position signal | vector, rotation, or bias | injects order into attention |

Examples:

1. QKV projections.
2. residual stream states.
3. dense retrieval vectors.

Non-examples:

1. nearest neighbors over integer ids.
2. one fixed meaning for a token in every context.

**Derivation habit.**

1. Write the tensor shape before writing the operation.
2. State whether vectors are raw input embeddings, hidden states, output rows, or retrieval embeddings.
3. Choose dot product, cosine similarity, or Euclidean distance deliberately.
4. Check whether position information is additive, rotary, learned, or an attention bias.
5. Track whether input and output embeddings are tied.

**Implementation lens.**

In code, embedding lookup is simple indexing. Conceptually, it is the step where the model stops seeing symbolic ids and starts seeing trainable vectors. That is why tokenizer changes, vocabulary resizing, and special-token handling are not superficial.

When debugging model behavior, inspect embedding norms, nearest neighbors, and the mean direction. Large norms or dominant components can affect logits and similarity search. For retrieval models, normalize vectors before cosine search unless the training objective explicitly uses norm as signal.

For transformer internals, remember that input embeddings are only the first residual-stream state. After attention and MLP layers, a token position's hidden state reflects surrounding context. Static nearest neighbors and contextual hidden-state probes answer different questions.

## 3. Similarity Geometry

Similarity Geometry connects token ids to continuous vectors and prepares the exact geometry used by attention, language-model logits, and dense retrieval.

### 3.1 Dot product

**Purpose.** Dot product focuses on magnitude-sensitive alignment. This matters because every later transformer operation starts from these vectors or from hidden states derived from them.

$$\operatorname{cos}(\mathbf{u},\mathbf{v})=\frac{\langle \mathbf{u},\mathbf{v}\rangle}{\lVert\mathbf{u}\rVert_2\lVert\mathbf{v}\rVert_2}.$$

**Operational definition.**

This concept explains how discrete language symbols become continuous vectors with trainable geometry.

**Worked reading.**

The operational question is what shape the vector has, how it is compared, and how training changes it.

| Object | Shape or formula | Role |
| --- | --- | --- |
| token ids | $B\times T$ | discrete sequence from tokenizer |
| embedding table | $|\mathcal{V}|\times d$ | row lookup for each token id |
| hidden states | $B\times T\times d$ | contextual vectors after lookup and layers |
| LM head | $d\times |\mathcal{V}|$ or tied $E^\top$ | maps hidden states to token logits |
| position signal | vector, rotation, or bias | injects order into attention |

Examples:

1. embedding rows.
2. hidden states.
3. similarity search.

Non-examples:

1. raw text in linear algebra.
2. ids treated as distances.

**Derivation habit.**

1. Write the tensor shape before writing the operation.
2. State whether vectors are raw input embeddings, hidden states, output rows, or retrieval embeddings.
3. Choose dot product, cosine similarity, or Euclidean distance deliberately.
4. Check whether position information is additive, rotary, learned, or an attention bias.
5. Track whether input and output embeddings are tied.

**Implementation lens.**

In code, embedding lookup is simple indexing. Conceptually, it is the step where the model stops seeing symbolic ids and starts seeing trainable vectors. That is why tokenizer changes, vocabulary resizing, and special-token handling are not superficial.

When debugging model behavior, inspect embedding norms, nearest neighbors, and the mean direction. Large norms or dominant components can affect logits and similarity search. For retrieval models, normalize vectors before cosine search unless the training objective explicitly uses norm as signal.

For transformer internals, remember that input embeddings are only the first residual-stream state. After attention and MLP layers, a token position's hidden state reflects surrounding context. Static nearest neighbors and contextual hidden-state probes answer different questions.

### 3.2 Cosine similarity

**Purpose.** Cosine similarity focuses on angle-only semantic comparison. This matters because every later transformer operation starts from these vectors or from hidden states derived from them.

$$\ell=-\log\frac{\exp(\mathbf{h}^{\top}\mathbf{w}_y)}{\sum_{j\in\mathcal{V}}\exp(\mathbf{h}^{\top}\mathbf{w}_j)}.$$

**Operational definition.**

Cosine similarity compares directions rather than magnitudes. It is often better for semantic neighbor queries when norms carry frequency or confidence effects.

**Worked reading.**

If two vectors point in the same direction, their cosine is near 1 even when one has a much larger norm.

| Object | Shape or formula | Role |
| --- | --- | --- |
| token ids | $B\times T$ | discrete sequence from tokenizer |
| embedding table | $|\mathcal{V}|\times d$ | row lookup for each token id |
| hidden states | $B\times T\times d$ | contextual vectors after lookup and layers |
| LM head | $d\times |\mathcal{V}|$ or tied $E^\top$ | maps hidden states to token logits |
| position signal | vector, rotation, or bias | injects order into attention |

Examples:

1. nearest-neighbor search.
2. embedding evaluation.
3. retrieval scoring after normalization.

Non-examples:

1. raw dot-product logits in a trained LM head.
2. Euclidean clustering without normalization.

**Derivation habit.**

1. Write the tensor shape before writing the operation.
2. State whether vectors are raw input embeddings, hidden states, output rows, or retrieval embeddings.
3. Choose dot product, cosine similarity, or Euclidean distance deliberately.
4. Check whether position information is additive, rotary, learned, or an attention bias.
5. Track whether input and output embeddings are tied.

**Implementation lens.**

In code, embedding lookup is simple indexing. Conceptually, it is the step where the model stops seeing symbolic ids and starts seeing trainable vectors. That is why tokenizer changes, vocabulary resizing, and special-token handling are not superficial.

When debugging model behavior, inspect embedding norms, nearest neighbors, and the mean direction. Large norms or dominant components can affect logits and similarity search. For retrieval models, normalize vectors before cosine search unless the training objective explicitly uses norm as signal.

For transformer internals, remember that input embeddings are only the first residual-stream state. After attention and MLP layers, a token position's hidden state reflects surrounding context. Static nearest neighbors and contextual hidden-state probes answer different questions.

### 3.3 Euclidean distance

**Purpose.** Euclidean distance focuses on metric distance and its caveats. This matters because every later transformer operation starts from these vectors or from hidden states derived from them.

$$\nabla_{\mathbf{w}_j}\ell=(p_j-\mathbf{1}\{j=y\})\mathbf{h}.$$

**Operational definition.**

This concept explains how discrete language symbols become continuous vectors with trainable geometry.

**Worked reading.**

The operational question is what shape the vector has, how it is compared, and how training changes it.

| Object | Shape or formula | Role |
| --- | --- | --- |
| token ids | $B\times T$ | discrete sequence from tokenizer |
| embedding table | $|\mathcal{V}|\times d$ | row lookup for each token id |
| hidden states | $B\times T\times d$ | contextual vectors after lookup and layers |
| LM head | $d\times |\mathcal{V}|$ or tied $E^\top$ | maps hidden states to token logits |
| position signal | vector, rotation, or bias | injects order into attention |

Examples:

1. embedding rows.
2. hidden states.
3. similarity search.

Non-examples:

1. raw text in linear algebra.
2. ids treated as distances.

**Derivation habit.**

1. Write the tensor shape before writing the operation.
2. State whether vectors are raw input embeddings, hidden states, output rows, or retrieval embeddings.
3. Choose dot product, cosine similarity, or Euclidean distance deliberately.
4. Check whether position information is additive, rotary, learned, or an attention bias.
5. Track whether input and output embeddings are tied.

**Implementation lens.**

In code, embedding lookup is simple indexing. Conceptually, it is the step where the model stops seeing symbolic ids and starts seeing trainable vectors. That is why tokenizer changes, vocabulary resizing, and special-token handling are not superficial.

When debugging model behavior, inspect embedding norms, nearest neighbors, and the mean direction. Large norms or dominant components can affect logits and similarity search. For retrieval models, normalize vectors before cosine search unless the training objective explicitly uses norm as signal.

For transformer internals, remember that input embeddings are only the first residual-stream state. After attention and MLP layers, a token position's hidden state reflects surrounding context. Static nearest neighbors and contextual hidden-state probes answer different questions.

### 3.4 Norms and frequency effects

**Purpose.** Norms and frequency effects focuses on why common tokens can have large or biased norms. This matters because every later transformer operation starts from these vectors or from hidden states derived from them.

$$\operatorname{PE}_{p,2k}=\sin\left(p/10000^{2k/d}\right),\qquad \operatorname{PE}_{p,2k+1}=\cos\left(p/10000^{2k/d}\right).$$

**Operational definition.**

Embedding spaces have geometry: norms, directions, subspaces, clusters, and dominant components. These structures can encode useful features and dataset artifacts.

**Worked reading.**

Centering an embedding cloud removes the mean direction; whitening rescales dominant axes so cosine neighborhoods are less dominated by global components.

| Object | Shape or formula | Role |
| --- | --- | --- |
| token ids | $B\times T$ | discrete sequence from tokenizer |
| embedding table | $|\mathcal{V}|\times d$ | row lookup for each token id |
| hidden states | $B\times T\times d$ | contextual vectors after lookup and layers |
| LM head | $d\times |\mathcal{V}|$ or tied $E^\top$ | maps hidden states to token logits |
| position signal | vector, rotation, or bias | injects order into attention |

Examples:

1. feature probes.
2. bias directions.
3. PCA diagnostics.

Non-examples:

1. assuming every axis has semantic meaning.
2. judging geometry from one 2D plot only.

**Derivation habit.**

1. Write the tensor shape before writing the operation.
2. State whether vectors are raw input embeddings, hidden states, output rows, or retrieval embeddings.
3. Choose dot product, cosine similarity, or Euclidean distance deliberately.
4. Check whether position information is additive, rotary, learned, or an attention bias.
5. Track whether input and output embeddings are tied.

**Implementation lens.**

In code, embedding lookup is simple indexing. Conceptually, it is the step where the model stops seeing symbolic ids and starts seeing trainable vectors. That is why tokenizer changes, vocabulary resizing, and special-token handling are not superficial.

When debugging model behavior, inspect embedding norms, nearest neighbors, and the mean direction. Large norms or dominant components can affect logits and similarity search. For retrieval models, normalize vectors before cosine search unless the training objective explicitly uses norm as signal.

For transformer internals, remember that input embeddings are only the first residual-stream state. After attention and MLP layers, a token position's hidden state reflects surrounding context. Static nearest neighbors and contextual hidden-state probes answer different questions.

### 3.5 Nearest neighbors

**Purpose.** Nearest neighbors focuses on retrieval in embedding space. This matters because every later transformer operation starts from these vectors or from hidden states derived from them.

$$\operatorname{RoPE}(\mathbf{q},p)=R_p\mathbf{q},\qquad (R_p\mathbf{q})^\top(R_s\mathbf{k})=\mathbf{q}^\top R_{s-p}\mathbf{k}.$$

**Operational definition.**

Embedding spaces have geometry: norms, directions, subspaces, clusters, and dominant components. These structures can encode useful features and dataset artifacts.

**Worked reading.**

Centering an embedding cloud removes the mean direction; whitening rescales dominant axes so cosine neighborhoods are less dominated by global components.

| Object | Shape or formula | Role |
| --- | --- | --- |
| token ids | $B\times T$ | discrete sequence from tokenizer |
| embedding table | $|\mathcal{V}|\times d$ | row lookup for each token id |
| hidden states | $B\times T\times d$ | contextual vectors after lookup and layers |
| LM head | $d\times |\mathcal{V}|$ or tied $E^\top$ | maps hidden states to token logits |
| position signal | vector, rotation, or bias | injects order into attention |

Examples:

1. feature probes.
2. bias directions.
3. PCA diagnostics.

Non-examples:

1. assuming every axis has semantic meaning.
2. judging geometry from one 2D plot only.

**Derivation habit.**

1. Write the tensor shape before writing the operation.
2. State whether vectors are raw input embeddings, hidden states, output rows, or retrieval embeddings.
3. Choose dot product, cosine similarity, or Euclidean distance deliberately.
4. Check whether position information is additive, rotary, learned, or an attention bias.
5. Track whether input and output embeddings are tied.

**Implementation lens.**

In code, embedding lookup is simple indexing. Conceptually, it is the step where the model stops seeing symbolic ids and starts seeing trainable vectors. That is why tokenizer changes, vocabulary resizing, and special-token handling are not superficial.

When debugging model behavior, inspect embedding norms, nearest neighbors, and the mean direction. Large norms or dominant components can affect logits and similarity search. For retrieval models, normalize vectors before cosine search unless the training objective explicitly uses norm as signal.

For transformer internals, remember that input embeddings are only the first residual-stream state. After attention and MLP layers, a token position's hidden state reflects surrounding context. Static nearest neighbors and contextual hidden-state probes answer different questions.

## 4. Geometry of Meaning

Geometry of Meaning connects token ids to continuous vectors and prepares the exact geometry used by attention, language-model logits, and dense retrieval.

### 4.1 Analogy directions

**Purpose.** Analogy directions focuses on linear offsets and relational structure. This matters because every later transformer operation starts from these vectors or from hidden states derived from them.

$$\ell=-\log\frac{\exp(\mathbf{h}^{\top}\mathbf{w}_y)}{\sum_{j\in\mathcal{V}}\exp(\mathbf{h}^{\top}\mathbf{w}_j)}.$$

**Operational definition.**

This concept explains how discrete language symbols become continuous vectors with trainable geometry.

**Worked reading.**

The operational question is what shape the vector has, how it is compared, and how training changes it.

| Object | Shape or formula | Role |
| --- | --- | --- |
| token ids | $B\times T$ | discrete sequence from tokenizer |
| embedding table | $|\mathcal{V}|\times d$ | row lookup for each token id |
| hidden states | $B\times T\times d$ | contextual vectors after lookup and layers |
| LM head | $d\times |\mathcal{V}|$ or tied $E^\top$ | maps hidden states to token logits |
| position signal | vector, rotation, or bias | injects order into attention |

Examples:

1. embedding rows.
2. hidden states.
3. similarity search.

Non-examples:

1. raw text in linear algebra.
2. ids treated as distances.

**Derivation habit.**

1. Write the tensor shape before writing the operation.
2. State whether vectors are raw input embeddings, hidden states, output rows, or retrieval embeddings.
3. Choose dot product, cosine similarity, or Euclidean distance deliberately.
4. Check whether position information is additive, rotary, learned, or an attention bias.
5. Track whether input and output embeddings are tied.

**Implementation lens.**

In code, embedding lookup is simple indexing. Conceptually, it is the step where the model stops seeing symbolic ids and starts seeing trainable vectors. That is why tokenizer changes, vocabulary resizing, and special-token handling are not superficial.

When debugging model behavior, inspect embedding norms, nearest neighbors, and the mean direction. Large norms or dominant components can affect logits and similarity search. For retrieval models, normalize vectors before cosine search unless the training objective explicitly uses norm as signal.

For transformer internals, remember that input embeddings are only the first residual-stream state. After attention and MLP layers, a token position's hidden state reflects surrounding context. Static nearest neighbors and contextual hidden-state probes answer different questions.

### 4.2 Subspaces and probes

**Purpose.** Subspaces and probes focuses on where features can be linearly readable. This matters because every later transformer operation starts from these vectors or from hidden states derived from them.

$$\nabla_{\mathbf{w}_j}\ell=(p_j-\mathbf{1}\{j=y\})\mathbf{h}.$$

**Operational definition.**

Embedding spaces have geometry: norms, directions, subspaces, clusters, and dominant components. These structures can encode useful features and dataset artifacts.

**Worked reading.**

Centering an embedding cloud removes the mean direction; whitening rescales dominant axes so cosine neighborhoods are less dominated by global components.

| Object | Shape or formula | Role |
| --- | --- | --- |
| token ids | $B\times T$ | discrete sequence from tokenizer |
| embedding table | $|\mathcal{V}|\times d$ | row lookup for each token id |
| hidden states | $B\times T\times d$ | contextual vectors after lookup and layers |
| LM head | $d\times |\mathcal{V}|$ or tied $E^\top$ | maps hidden states to token logits |
| position signal | vector, rotation, or bias | injects order into attention |

Examples:

1. feature probes.
2. bias directions.
3. PCA diagnostics.

Non-examples:

1. assuming every axis has semantic meaning.
2. judging geometry from one 2D plot only.

**Derivation habit.**

1. Write the tensor shape before writing the operation.
2. State whether vectors are raw input embeddings, hidden states, output rows, or retrieval embeddings.
3. Choose dot product, cosine similarity, or Euclidean distance deliberately.
4. Check whether position information is additive, rotary, learned, or an attention bias.
5. Track whether input and output embeddings are tied.

**Implementation lens.**

In code, embedding lookup is simple indexing. Conceptually, it is the step where the model stops seeing symbolic ids and starts seeing trainable vectors. That is why tokenizer changes, vocabulary resizing, and special-token handling are not superficial.

When debugging model behavior, inspect embedding norms, nearest neighbors, and the mean direction. Large norms or dominant components can affect logits and similarity search. For retrieval models, normalize vectors before cosine search unless the training objective explicitly uses norm as signal.

For transformer internals, remember that input embeddings are only the first residual-stream state. After attention and MLP layers, a token position's hidden state reflects surrounding context. Static nearest neighbors and contextual hidden-state probes answer different questions.

### 4.3 Isotropy and anisotropy

**Purpose.** Isotropy and anisotropy focuses on why many embeddings collapse into dominant directions. This matters because every later transformer operation starts from these vectors or from hidden states derived from them.

$$\operatorname{PE}_{p,2k}=\sin\left(p/10000^{2k/d}\right),\qquad \operatorname{PE}_{p,2k+1}=\cos\left(p/10000^{2k/d}\right).$$

**Operational definition.**

Embedding spaces have geometry: norms, directions, subspaces, clusters, and dominant components. These structures can encode useful features and dataset artifacts.

**Worked reading.**

Centering an embedding cloud removes the mean direction; whitening rescales dominant axes so cosine neighborhoods are less dominated by global components.

| Object | Shape or formula | Role |
| --- | --- | --- |
| token ids | $B\times T$ | discrete sequence from tokenizer |
| embedding table | $|\mathcal{V}|\times d$ | row lookup for each token id |
| hidden states | $B\times T\times d$ | contextual vectors after lookup and layers |
| LM head | $d\times |\mathcal{V}|$ or tied $E^\top$ | maps hidden states to token logits |
| position signal | vector, rotation, or bias | injects order into attention |

Examples:

1. feature probes.
2. bias directions.
3. PCA diagnostics.

Non-examples:

1. assuming every axis has semantic meaning.
2. judging geometry from one 2D plot only.

**Derivation habit.**

1. Write the tensor shape before writing the operation.
2. State whether vectors are raw input embeddings, hidden states, output rows, or retrieval embeddings.
3. Choose dot product, cosine similarity, or Euclidean distance deliberately.
4. Check whether position information is additive, rotary, learned, or an attention bias.
5. Track whether input and output embeddings are tied.

**Implementation lens.**

In code, embedding lookup is simple indexing. Conceptually, it is the step where the model stops seeing symbolic ids and starts seeing trainable vectors. That is why tokenizer changes, vocabulary resizing, and special-token handling are not superficial.

When debugging model behavior, inspect embedding norms, nearest neighbors, and the mean direction. Large norms or dominant components can affect logits and similarity search. For retrieval models, normalize vectors before cosine search unless the training objective explicitly uses norm as signal.

For transformer internals, remember that input embeddings are only the first residual-stream state. After attention and MLP layers, a token position's hidden state reflects surrounding context. Static nearest neighbors and contextual hidden-state probes answer different questions.

### 4.4 Centering and whitening

**Purpose.** Centering and whitening focuses on simple geometry repairs for similarity search. This matters because every later transformer operation starts from these vectors or from hidden states derived from them.

$$\operatorname{RoPE}(\mathbf{q},p)=R_p\mathbf{q},\qquad (R_p\mathbf{q})^\top(R_s\mathbf{k})=\mathbf{q}^\top R_{s-p}\mathbf{k}.$$

**Operational definition.**

This concept explains how discrete language symbols become continuous vectors with trainable geometry.

**Worked reading.**

The operational question is what shape the vector has, how it is compared, and how training changes it.

| Object | Shape or formula | Role |
| --- | --- | --- |
| token ids | $B\times T$ | discrete sequence from tokenizer |
| embedding table | $|\mathcal{V}|\times d$ | row lookup for each token id |
| hidden states | $B\times T\times d$ | contextual vectors after lookup and layers |
| LM head | $d\times |\mathcal{V}|$ or tied $E^\top$ | maps hidden states to token logits |
| position signal | vector, rotation, or bias | injects order into attention |

Examples:

1. embedding rows.
2. hidden states.
3. similarity search.

Non-examples:

1. raw text in linear algebra.
2. ids treated as distances.

**Derivation habit.**

1. Write the tensor shape before writing the operation.
2. State whether vectors are raw input embeddings, hidden states, output rows, or retrieval embeddings.
3. Choose dot product, cosine similarity, or Euclidean distance deliberately.
4. Check whether position information is additive, rotary, learned, or an attention bias.
5. Track whether input and output embeddings are tied.

**Implementation lens.**

In code, embedding lookup is simple indexing. Conceptually, it is the step where the model stops seeing symbolic ids and starts seeing trainable vectors. That is why tokenizer changes, vocabulary resizing, and special-token handling are not superficial.

When debugging model behavior, inspect embedding norms, nearest neighbors, and the mean direction. Large norms or dominant components can affect logits and similarity search. For retrieval models, normalize vectors before cosine search unless the training objective explicitly uses norm as signal.

For transformer internals, remember that input embeddings are only the first residual-stream state. After attention and MLP layers, a token position's hidden state reflects surrounding context. Static nearest neighbors and contextual hidden-state probes answer different questions.

### 4.5 Bias and representation directions

**Purpose.** Bias and representation directions focuses on why geometry can encode social or dataset artifacts. This matters because every later transformer operation starts from these vectors or from hidden states derived from them.

$$\operatorname{params}_{\mathrm{embed}}=|\mathcal{V}|d_{\mathrm{model}}.$$

**Operational definition.**

Embedding spaces have geometry: norms, directions, subspaces, clusters, and dominant components. These structures can encode useful features and dataset artifacts.

**Worked reading.**

Centering an embedding cloud removes the mean direction; whitening rescales dominant axes so cosine neighborhoods are less dominated by global components.

| Object | Shape or formula | Role |
| --- | --- | --- |
| token ids | $B\times T$ | discrete sequence from tokenizer |
| embedding table | $|\mathcal{V}|\times d$ | row lookup for each token id |
| hidden states | $B\times T\times d$ | contextual vectors after lookup and layers |
| LM head | $d\times |\mathcal{V}|$ or tied $E^\top$ | maps hidden states to token logits |
| position signal | vector, rotation, or bias | injects order into attention |

Examples:

1. feature probes.
2. bias directions.
3. PCA diagnostics.

Non-examples:

1. assuming every axis has semantic meaning.
2. judging geometry from one 2D plot only.

**Derivation habit.**

1. Write the tensor shape before writing the operation.
2. State whether vectors are raw input embeddings, hidden states, output rows, or retrieval embeddings.
3. Choose dot product, cosine similarity, or Euclidean distance deliberately.
4. Check whether position information is additive, rotary, learned, or an attention bias.
5. Track whether input and output embeddings are tied.

**Implementation lens.**

In code, embedding lookup is simple indexing. Conceptually, it is the step where the model stops seeing symbolic ids and starts seeing trainable vectors. That is why tokenizer changes, vocabulary resizing, and special-token handling are not superficial.

When debugging model behavior, inspect embedding norms, nearest neighbors, and the mean direction. Large norms or dominant components can affect logits and similarity search. For retrieval models, normalize vectors before cosine search unless the training objective explicitly uses norm as signal.

For transformer internals, remember that input embeddings are only the first residual-stream state. After attention and MLP layers, a token position's hidden state reflects surrounding context. Static nearest neighbors and contextual hidden-state probes answer different questions.

## 5. Training Embeddings

Training Embeddings connects token ids to continuous vectors and prepares the exact geometry used by attention, language-model logits, and dense retrieval.

### 5.1 Language-model loss gradients

**Purpose.** Language-model loss gradients focuses on how cross-entropy updates rows. This matters because every later transformer operation starts from these vectors or from hidden states derived from them.

$$\nabla_{\mathbf{w}_j}\ell=(p_j-\mathbf{1}\{j=y\})\mathbf{h}.$$

**Operational definition.**

Embedding training moves rows according to prediction errors and co-occurrence structure. Frequent tokens receive many updates, and rare tokens may remain poorly estimated.

**Worked reading.**

For a softmax output row $\mathbf{w}_j$, the gradient is proportional to $(p_j-\mathbf{1}\{j=y\})\mathbf{h}$.

| Object | Shape or formula | Role |
| --- | --- | --- |
| token ids | $B\times T$ | discrete sequence from tokenizer |
| embedding table | $|\mathcal{V}|\times d$ | row lookup for each token id |
| hidden states | $B\times T\times d$ | contextual vectors after lookup and layers |
| LM head | $d\times |\mathcal{V}|$ or tied $E^\top$ | maps hidden states to token logits |
| position signal | vector, rotation, or bias | injects order into attention |

Examples:

1. language-model cross-entropy.
2. negative sampling.
3. co-occurrence factorization.

Non-examples:

1. hand-written semantic coordinates.
2. frozen random rows with no adaptation.

**Derivation habit.**

1. Write the tensor shape before writing the operation.
2. State whether vectors are raw input embeddings, hidden states, output rows, or retrieval embeddings.
3. Choose dot product, cosine similarity, or Euclidean distance deliberately.
4. Check whether position information is additive, rotary, learned, or an attention bias.
5. Track whether input and output embeddings are tied.

**Implementation lens.**

In code, embedding lookup is simple indexing. Conceptually, it is the step where the model stops seeing symbolic ids and starts seeing trainable vectors. That is why tokenizer changes, vocabulary resizing, and special-token handling are not superficial.

When debugging model behavior, inspect embedding norms, nearest neighbors, and the mean direction. Large norms or dominant components can affect logits and similarity search. For retrieval models, normalize vectors before cosine search unless the training objective explicitly uses norm as signal.

For transformer internals, remember that input embeddings are only the first residual-stream state. After attention and MLP layers, a token position's hidden state reflects surrounding context. Static nearest neighbors and contextual hidden-state probes answer different questions.

### 5.2 Word2vec and negative sampling

**Purpose.** Word2vec and negative sampling focuses on predictive embeddings before transformers. This matters because every later transformer operation starts from these vectors or from hidden states derived from them.

$$\operatorname{PE}_{p,2k}=\sin\left(p/10000^{2k/d}\right),\qquad \operatorname{PE}_{p,2k+1}=\cos\left(p/10000^{2k/d}\right).$$

**Operational definition.**

Embedding training moves rows according to prediction errors and co-occurrence structure. Frequent tokens receive many updates, and rare tokens may remain poorly estimated.

**Worked reading.**

For a softmax output row $\mathbf{w}_j$, the gradient is proportional to $(p_j-\mathbf{1}\{j=y\})\mathbf{h}$.

| Object | Shape or formula | Role |
| --- | --- | --- |
| token ids | $B\times T$ | discrete sequence from tokenizer |
| embedding table | $|\mathcal{V}|\times d$ | row lookup for each token id |
| hidden states | $B\times T\times d$ | contextual vectors after lookup and layers |
| LM head | $d\times |\mathcal{V}|$ or tied $E^\top$ | maps hidden states to token logits |
| position signal | vector, rotation, or bias | injects order into attention |

Examples:

1. language-model cross-entropy.
2. negative sampling.
3. co-occurrence factorization.

Non-examples:

1. hand-written semantic coordinates.
2. frozen random rows with no adaptation.

**Derivation habit.**

1. Write the tensor shape before writing the operation.
2. State whether vectors are raw input embeddings, hidden states, output rows, or retrieval embeddings.
3. Choose dot product, cosine similarity, or Euclidean distance deliberately.
4. Check whether position information is additive, rotary, learned, or an attention bias.
5. Track whether input and output embeddings are tied.

**Implementation lens.**

In code, embedding lookup is simple indexing. Conceptually, it is the step where the model stops seeing symbolic ids and starts seeing trainable vectors. That is why tokenizer changes, vocabulary resizing, and special-token handling are not superficial.

When debugging model behavior, inspect embedding norms, nearest neighbors, and the mean direction. Large norms or dominant components can affect logits and similarity search. For retrieval models, normalize vectors before cosine search unless the training objective explicitly uses norm as signal.

For transformer internals, remember that input embeddings are only the first residual-stream state. After attention and MLP layers, a token position's hidden state reflects surrounding context. Static nearest neighbors and contextual hidden-state probes answer different questions.

### 5.3 GloVe and co-occurrence factorization

**Purpose.** GloVe and co-occurrence factorization focuses on global count structure. This matters because every later transformer operation starts from these vectors or from hidden states derived from them.

$$\operatorname{RoPE}(\mathbf{q},p)=R_p\mathbf{q},\qquad (R_p\mathbf{q})^\top(R_s\mathbf{k})=\mathbf{q}^\top R_{s-p}\mathbf{k}.$$

**Operational definition.**

Embedding training moves rows according to prediction errors and co-occurrence structure. Frequent tokens receive many updates, and rare tokens may remain poorly estimated.

**Worked reading.**

For a softmax output row $\mathbf{w}_j$, the gradient is proportional to $(p_j-\mathbf{1}\{j=y\})\mathbf{h}$.

| Object | Shape or formula | Role |
| --- | --- | --- |
| token ids | $B\times T$ | discrete sequence from tokenizer |
| embedding table | $|\mathcal{V}|\times d$ | row lookup for each token id |
| hidden states | $B\times T\times d$ | contextual vectors after lookup and layers |
| LM head | $d\times |\mathcal{V}|$ or tied $E^\top$ | maps hidden states to token logits |
| position signal | vector, rotation, or bias | injects order into attention |

Examples:

1. language-model cross-entropy.
2. negative sampling.
3. co-occurrence factorization.

Non-examples:

1. hand-written semantic coordinates.
2. frozen random rows with no adaptation.

**Derivation habit.**

1. Write the tensor shape before writing the operation.
2. State whether vectors are raw input embeddings, hidden states, output rows, or retrieval embeddings.
3. Choose dot product, cosine similarity, or Euclidean distance deliberately.
4. Check whether position information is additive, rotary, learned, or an attention bias.
5. Track whether input and output embeddings are tied.

**Implementation lens.**

In code, embedding lookup is simple indexing. Conceptually, it is the step where the model stops seeing symbolic ids and starts seeing trainable vectors. That is why tokenizer changes, vocabulary resizing, and special-token handling are not superficial.

When debugging model behavior, inspect embedding norms, nearest neighbors, and the mean direction. Large norms or dominant components can affect logits and similarity search. For retrieval models, normalize vectors before cosine search unless the training objective explicitly uses norm as signal.

For transformer internals, remember that input embeddings are only the first residual-stream state. After attention and MLP layers, a token position's hidden state reflects surrounding context. Static nearest neighbors and contextual hidden-state probes answer different questions.

### 5.4 Fine-tuning drift

**Purpose.** Fine-tuning drift focuses on why embeddings move during adaptation. This matters because every later transformer operation starts from these vectors or from hidden states derived from them.

$$\operatorname{params}_{\mathrm{embed}}=|\mathcal{V}|d_{\mathrm{model}}.$$

**Operational definition.**

This concept explains how discrete language symbols become continuous vectors with trainable geometry.

**Worked reading.**

The operational question is what shape the vector has, how it is compared, and how training changes it.

| Object | Shape or formula | Role |
| --- | --- | --- |
| token ids | $B\times T$ | discrete sequence from tokenizer |
| embedding table | $|\mathcal{V}|\times d$ | row lookup for each token id |
| hidden states | $B\times T\times d$ | contextual vectors after lookup and layers |
| LM head | $d\times |\mathcal{V}|$ or tied $E^\top$ | maps hidden states to token logits |
| position signal | vector, rotation, or bias | injects order into attention |

Examples:

1. embedding rows.
2. hidden states.
3. similarity search.

Non-examples:

1. raw text in linear algebra.
2. ids treated as distances.

**Derivation habit.**

1. Write the tensor shape before writing the operation.
2. State whether vectors are raw input embeddings, hidden states, output rows, or retrieval embeddings.
3. Choose dot product, cosine similarity, or Euclidean distance deliberately.
4. Check whether position information is additive, rotary, learned, or an attention bias.
5. Track whether input and output embeddings are tied.

**Implementation lens.**

In code, embedding lookup is simple indexing. Conceptually, it is the step where the model stops seeing symbolic ids and starts seeing trainable vectors. That is why tokenizer changes, vocabulary resizing, and special-token handling are not superficial.

When debugging model behavior, inspect embedding norms, nearest neighbors, and the mean direction. Large norms or dominant components can affect logits and similarity search. For retrieval models, normalize vectors before cosine search unless the training objective explicitly uses norm as signal.

For transformer internals, remember that input embeddings are only the first residual-stream state. After attention and MLP layers, a token position's hidden state reflects surrounding context. Static nearest neighbors and contextual hidden-state probes answer different questions.

### 5.5 Vocabulary resizing

**Purpose.** Vocabulary resizing focuses on how new rows are initialized and trained. This matters because every later transformer operation starts from these vectors or from hidden states derived from them.

$$E\in\mathbb{R}^{|\mathcal{V}|\times d_{\mathrm{model}}},\qquad \mathbf{x}_t=E_{i_t,:}.$$

**Operational definition.**

This concept explains how discrete language symbols become continuous vectors with trainable geometry.

**Worked reading.**

The operational question is what shape the vector has, how it is compared, and how training changes it.

| Object | Shape or formula | Role |
| --- | --- | --- |
| token ids | $B\times T$ | discrete sequence from tokenizer |
| embedding table | $|\mathcal{V}|\times d$ | row lookup for each token id |
| hidden states | $B\times T\times d$ | contextual vectors after lookup and layers |
| LM head | $d\times |\mathcal{V}|$ or tied $E^\top$ | maps hidden states to token logits |
| position signal | vector, rotation, or bias | injects order into attention |

Examples:

1. embedding rows.
2. hidden states.
3. similarity search.

Non-examples:

1. raw text in linear algebra.
2. ids treated as distances.

**Derivation habit.**

1. Write the tensor shape before writing the operation.
2. State whether vectors are raw input embeddings, hidden states, output rows, or retrieval embeddings.
3. Choose dot product, cosine similarity, or Euclidean distance deliberately.
4. Check whether position information is additive, rotary, learned, or an attention bias.
5. Track whether input and output embeddings are tied.

**Implementation lens.**

In code, embedding lookup is simple indexing. Conceptually, it is the step where the model stops seeing symbolic ids and starts seeing trainable vectors. That is why tokenizer changes, vocabulary resizing, and special-token handling are not superficial.

When debugging model behavior, inspect embedding norms, nearest neighbors, and the mean direction. Large norms or dominant components can affect logits and similarity search. For retrieval models, normalize vectors before cosine search unless the training objective explicitly uses norm as signal.

For transformer internals, remember that input embeddings are only the first residual-stream state. After attention and MLP layers, a token position's hidden state reflects surrounding context. Static nearest neighbors and contextual hidden-state probes answer different questions.

## 6. Positional Information

Positional Information connects token ids to continuous vectors and prepares the exact geometry used by attention, language-model logits, and dense retrieval.

### 6.1 Why positions are needed

**Purpose.** Why positions are needed focuses on self-attention without order is permutation-equivariant. This matters because every later transformer operation starts from these vectors or from hidden states derived from them.

$$\operatorname{PE}_{p,2k}=\sin\left(p/10000^{2k/d}\right),\qquad \operatorname{PE}_{p,2k+1}=\cos\left(p/10000^{2k/d}\right).$$

**Operational definition.**

Position mechanisms inject order into otherwise content-based attention. They may be additive vectors, rotations, or attention-score biases.

**Worked reading.**

Sinusoidal encodings add fixed frequency features; RoPE rotates query/key pairs; ALiBi adds a distance-dependent bias to attention logits.

| Object | Shape or formula | Role |
| --- | --- | --- |
| token ids | $B\times T$ | discrete sequence from tokenizer |
| embedding table | $|\mathcal{V}|\times d$ | row lookup for each token id |
| hidden states | $B\times T\times d$ | contextual vectors after lookup and layers |
| LM head | $d\times |\mathcal{V}|$ or tied $E^\top$ | maps hidden states to token logits |
| position signal | vector, rotation, or bias | injects order into attention |

Examples:

1. sinusoidal encodings.
2. learned position rows.
3. RoPE.
4. ALiBi.

Non-examples:

1. bag-of-token attention with no order.
2. token ids used as positions.

**Derivation habit.**

1. Write the tensor shape before writing the operation.
2. State whether vectors are raw input embeddings, hidden states, output rows, or retrieval embeddings.
3. Choose dot product, cosine similarity, or Euclidean distance deliberately.
4. Check whether position information is additive, rotary, learned, or an attention bias.
5. Track whether input and output embeddings are tied.

**Implementation lens.**

In code, embedding lookup is simple indexing. Conceptually, it is the step where the model stops seeing symbolic ids and starts seeing trainable vectors. That is why tokenizer changes, vocabulary resizing, and special-token handling are not superficial.

When debugging model behavior, inspect embedding norms, nearest neighbors, and the mean direction. Large norms or dominant components can affect logits and similarity search. For retrieval models, normalize vectors before cosine search unless the training objective explicitly uses norm as signal.

For transformer internals, remember that input embeddings are only the first residual-stream state. After attention and MLP layers, a token position's hidden state reflects surrounding context. Static nearest neighbors and contextual hidden-state probes answer different questions.

### 6.2 Sinusoidal positional encodings

**Purpose.** Sinusoidal positional encodings focuses on fixed frequency features. This matters because every later transformer operation starts from these vectors or from hidden states derived from them.

$$\operatorname{RoPE}(\mathbf{q},p)=R_p\mathbf{q},\qquad (R_p\mathbf{q})^\top(R_s\mathbf{k})=\mathbf{q}^\top R_{s-p}\mathbf{k}.$$

**Operational definition.**

Position mechanisms inject order into otherwise content-based attention. They may be additive vectors, rotations, or attention-score biases.

**Worked reading.**

Sinusoidal encodings add fixed frequency features; RoPE rotates query/key pairs; ALiBi adds a distance-dependent bias to attention logits.

| Object | Shape or formula | Role |
| --- | --- | --- |
| token ids | $B\times T$ | discrete sequence from tokenizer |
| embedding table | $|\mathcal{V}|\times d$ | row lookup for each token id |
| hidden states | $B\times T\times d$ | contextual vectors after lookup and layers |
| LM head | $d\times |\mathcal{V}|$ or tied $E^\top$ | maps hidden states to token logits |
| position signal | vector, rotation, or bias | injects order into attention |

Examples:

1. sinusoidal encodings.
2. learned position rows.
3. RoPE.
4. ALiBi.

Non-examples:

1. bag-of-token attention with no order.
2. token ids used as positions.

**Derivation habit.**

1. Write the tensor shape before writing the operation.
2. State whether vectors are raw input embeddings, hidden states, output rows, or retrieval embeddings.
3. Choose dot product, cosine similarity, or Euclidean distance deliberately.
4. Check whether position information is additive, rotary, learned, or an attention bias.
5. Track whether input and output embeddings are tied.

**Implementation lens.**

In code, embedding lookup is simple indexing. Conceptually, it is the step where the model stops seeing symbolic ids and starts seeing trainable vectors. That is why tokenizer changes, vocabulary resizing, and special-token handling are not superficial.

When debugging model behavior, inspect embedding norms, nearest neighbors, and the mean direction. Large norms or dominant components can affect logits and similarity search. For retrieval models, normalize vectors before cosine search unless the training objective explicitly uses norm as signal.

For transformer internals, remember that input embeddings are only the first residual-stream state. After attention and MLP layers, a token position's hidden state reflects surrounding context. Static nearest neighbors and contextual hidden-state probes answer different questions.

### 6.3 Learned positional embeddings

**Purpose.** Learned positional embeddings focuses on trainable position rows. This matters because every later transformer operation starts from these vectors or from hidden states derived from them.

$$\operatorname{params}_{\mathrm{embed}}=|\mathcal{V}|d_{\mathrm{model}}.$$

**Operational definition.**

Position mechanisms inject order into otherwise content-based attention. They may be additive vectors, rotations, or attention-score biases.

**Worked reading.**

Sinusoidal encodings add fixed frequency features; RoPE rotates query/key pairs; ALiBi adds a distance-dependent bias to attention logits.

| Object | Shape or formula | Role |
| --- | --- | --- |
| token ids | $B\times T$ | discrete sequence from tokenizer |
| embedding table | $|\mathcal{V}|\times d$ | row lookup for each token id |
| hidden states | $B\times T\times d$ | contextual vectors after lookup and layers |
| LM head | $d\times |\mathcal{V}|$ or tied $E^\top$ | maps hidden states to token logits |
| position signal | vector, rotation, or bias | injects order into attention |

Examples:

1. sinusoidal encodings.
2. learned position rows.
3. RoPE.
4. ALiBi.

Non-examples:

1. bag-of-token attention with no order.
2. token ids used as positions.

**Derivation habit.**

1. Write the tensor shape before writing the operation.
2. State whether vectors are raw input embeddings, hidden states, output rows, or retrieval embeddings.
3. Choose dot product, cosine similarity, or Euclidean distance deliberately.
4. Check whether position information is additive, rotary, learned, or an attention bias.
5. Track whether input and output embeddings are tied.

**Implementation lens.**

In code, embedding lookup is simple indexing. Conceptually, it is the step where the model stops seeing symbolic ids and starts seeing trainable vectors. That is why tokenizer changes, vocabulary resizing, and special-token handling are not superficial.

When debugging model behavior, inspect embedding norms, nearest neighbors, and the mean direction. Large norms or dominant components can affect logits and similarity search. For retrieval models, normalize vectors before cosine search unless the training objective explicitly uses norm as signal.

For transformer internals, remember that input embeddings are only the first residual-stream state. After attention and MLP layers, a token position's hidden state reflects surrounding context. Static nearest neighbors and contextual hidden-state probes answer different questions.

### 6.4 Rotary positional embeddings

**Purpose.** Rotary positional embeddings focuses on RoPE as complex-plane rotations. This matters because every later transformer operation starts from these vectors or from hidden states derived from them.

$$E\in\mathbb{R}^{|\mathcal{V}|\times d_{\mathrm{model}}},\qquad \mathbf{x}_t=E_{i_t,:}.$$

**Operational definition.**

RoPE encodes position by rotating query and key coordinate pairs. The resulting attention dot product depends on relative offset through the rotation difference.

**Worked reading.**

A vector at position $p$ is rotated by angles tied to $p$ and frequency index, so attention can represent relative distance without adding a separate position vector.

| Object | Shape or formula | Role |
| --- | --- | --- |
| token ids | $B\times T$ | discrete sequence from tokenizer |
| embedding table | $|\mathcal{V}|\times d$ | row lookup for each token id |
| hidden states | $B\times T\times d$ | contextual vectors after lookup and layers |
| LM head | $d\times |\mathcal{V}|$ or tied $E^\top$ | maps hidden states to token logits |
| position signal | vector, rotation, or bias | injects order into attention |

Examples:

1. Llama-style position handling.
2. long-context extrapolation studies.
3. relative offset in attention.

Non-examples:

1. absolute learned position rows.
2. ALiBi scalar bias only.

**Derivation habit.**

1. Write the tensor shape before writing the operation.
2. State whether vectors are raw input embeddings, hidden states, output rows, or retrieval embeddings.
3. Choose dot product, cosine similarity, or Euclidean distance deliberately.
4. Check whether position information is additive, rotary, learned, or an attention bias.
5. Track whether input and output embeddings are tied.

**Implementation lens.**

In code, embedding lookup is simple indexing. Conceptually, it is the step where the model stops seeing symbolic ids and starts seeing trainable vectors. That is why tokenizer changes, vocabulary resizing, and special-token handling are not superficial.

When debugging model behavior, inspect embedding norms, nearest neighbors, and the mean direction. Large norms or dominant components can affect logits and similarity search. For retrieval models, normalize vectors before cosine search unless the training objective explicitly uses norm as signal.

For transformer internals, remember that input embeddings are only the first residual-stream state. After attention and MLP layers, a token position's hidden state reflects surrounding context. Static nearest neighbors and contextual hidden-state probes answer different questions.

### 6.5 ALiBi and relative biases

**Purpose.** ALiBi and relative biases focuses on linear distance penalties in attention scores. This matters because every later transformer operation starts from these vectors or from hidden states derived from them.

$$\mathbf{x}_t=\mathbf{e}_{i_t}^{\top}E.$$

**Operational definition.**

Position mechanisms inject order into otherwise content-based attention. They may be additive vectors, rotations, or attention-score biases.

**Worked reading.**

Sinusoidal encodings add fixed frequency features; RoPE rotates query/key pairs; ALiBi adds a distance-dependent bias to attention logits.

| Object | Shape or formula | Role |
| --- | --- | --- |
| token ids | $B\times T$ | discrete sequence from tokenizer |
| embedding table | $|\mathcal{V}|\times d$ | row lookup for each token id |
| hidden states | $B\times T\times d$ | contextual vectors after lookup and layers |
| LM head | $d\times |\mathcal{V}|$ or tied $E^\top$ | maps hidden states to token logits |
| position signal | vector, rotation, or bias | injects order into attention |

Examples:

1. sinusoidal encodings.
2. learned position rows.
3. RoPE.
4. ALiBi.

Non-examples:

1. bag-of-token attention with no order.
2. token ids used as positions.

**Derivation habit.**

1. Write the tensor shape before writing the operation.
2. State whether vectors are raw input embeddings, hidden states, output rows, or retrieval embeddings.
3. Choose dot product, cosine similarity, or Euclidean distance deliberately.
4. Check whether position information is additive, rotary, learned, or an attention bias.
5. Track whether input and output embeddings are tied.

**Implementation lens.**

In code, embedding lookup is simple indexing. Conceptually, it is the step where the model stops seeing symbolic ids and starts seeing trainable vectors. That is why tokenizer changes, vocabulary resizing, and special-token handling are not superficial.

When debugging model behavior, inspect embedding norms, nearest neighbors, and the mean direction. Large norms or dominant components can affect logits and similarity search. For retrieval models, normalize vectors before cosine search unless the training objective explicitly uses norm as signal.

For transformer internals, remember that input embeddings are only the first residual-stream state. After attention and MLP layers, a token position's hidden state reflects surrounding context. Static nearest neighbors and contextual hidden-state probes answer different questions.

## 7. Attention Bridge

Attention Bridge connects token ids to continuous vectors and prepares the exact geometry used by attention, language-model logits, and dense retrieval.

### 7.1 Query key value projections

**Purpose.** Query key value projections focuses on embedding space after learned linear maps. This matters because every later transformer operation starts from these vectors or from hidden states derived from them.

$$\operatorname{RoPE}(\mathbf{q},p)=R_p\mathbf{q},\qquad (R_p\mathbf{q})^\top(R_s\mathbf{k})=\mathbf{q}^\top R_{s-p}\mathbf{k}.$$

**Operational definition.**

Embeddings become contextual through projections, attention, MLPs, and residual additions. Attention compares projected vectors, not raw token ids.

**Worked reading.**

The query and key projections turn hidden states into vectors whose dot products define attention weights.

| Object | Shape or formula | Role |
| --- | --- | --- |
| token ids | $B\times T$ | discrete sequence from tokenizer |
| embedding table | $|\mathcal{V}|\times d$ | row lookup for each token id |
| hidden states | $B\times T\times d$ | contextual vectors after lookup and layers |
| LM head | $d\times |\mathcal{V}|$ or tied $E^\top$ | maps hidden states to token logits |
| position signal | vector, rotation, or bias | injects order into attention |

Examples:

1. QKV projections.
2. residual stream states.
3. dense retrieval vectors.

Non-examples:

1. nearest neighbors over integer ids.
2. one fixed meaning for a token in every context.

**Derivation habit.**

1. Write the tensor shape before writing the operation.
2. State whether vectors are raw input embeddings, hidden states, output rows, or retrieval embeddings.
3. Choose dot product, cosine similarity, or Euclidean distance deliberately.
4. Check whether position information is additive, rotary, learned, or an attention bias.
5. Track whether input and output embeddings are tied.

**Implementation lens.**

In code, embedding lookup is simple indexing. Conceptually, it is the step where the model stops seeing symbolic ids and starts seeing trainable vectors. That is why tokenizer changes, vocabulary resizing, and special-token handling are not superficial.

When debugging model behavior, inspect embedding norms, nearest neighbors, and the mean direction. Large norms or dominant components can affect logits and similarity search. For retrieval models, normalize vectors before cosine search unless the training objective explicitly uses norm as signal.

For transformer internals, remember that input embeddings are only the first residual-stream state. After attention and MLP layers, a token position's hidden state reflects surrounding context. Static nearest neighbors and contextual hidden-state probes answer different questions.

### 7.2 Attention as soft nearest-neighbor lookup

**Purpose.** Attention as soft nearest-neighbor lookup focuses on dot products over projected embeddings. This matters because every later transformer operation starts from these vectors or from hidden states derived from them.

$$\operatorname{params}_{\mathrm{embed}}=|\mathcal{V}|d_{\mathrm{model}}.$$

**Operational definition.**

Embeddings become contextual through projections, attention, MLPs, and residual additions. Attention compares projected vectors, not raw token ids.

**Worked reading.**

The query and key projections turn hidden states into vectors whose dot products define attention weights.

| Object | Shape or formula | Role |
| --- | --- | --- |
| token ids | $B\times T$ | discrete sequence from tokenizer |
| embedding table | $|\mathcal{V}|\times d$ | row lookup for each token id |
| hidden states | $B\times T\times d$ | contextual vectors after lookup and layers |
| LM head | $d\times |\mathcal{V}|$ or tied $E^\top$ | maps hidden states to token logits |
| position signal | vector, rotation, or bias | injects order into attention |

Examples:

1. QKV projections.
2. residual stream states.
3. dense retrieval vectors.

Non-examples:

1. nearest neighbors over integer ids.
2. one fixed meaning for a token in every context.

**Derivation habit.**

1. Write the tensor shape before writing the operation.
2. State whether vectors are raw input embeddings, hidden states, output rows, or retrieval embeddings.
3. Choose dot product, cosine similarity, or Euclidean distance deliberately.
4. Check whether position information is additive, rotary, learned, or an attention bias.
5. Track whether input and output embeddings are tied.

**Implementation lens.**

In code, embedding lookup is simple indexing. Conceptually, it is the step where the model stops seeing symbolic ids and starts seeing trainable vectors. That is why tokenizer changes, vocabulary resizing, and special-token handling are not superficial.

When debugging model behavior, inspect embedding norms, nearest neighbors, and the mean direction. Large norms or dominant components can affect logits and similarity search. For retrieval models, normalize vectors before cosine search unless the training objective explicitly uses norm as signal.

For transformer internals, remember that input embeddings are only the first residual-stream state. After attention and MLP layers, a token position's hidden state reflects surrounding context. Static nearest neighbors and contextual hidden-state probes answer different questions.

### 7.3 Layer-wise contextualization

**Purpose.** Layer-wise contextualization focuses on representations change through residual blocks. This matters because every later transformer operation starts from these vectors or from hidden states derived from them.

$$E\in\mathbb{R}^{|\mathcal{V}|\times d_{\mathrm{model}}},\qquad \mathbf{x}_t=E_{i_t,:}.$$

**Operational definition.**

This concept explains how discrete language symbols become continuous vectors with trainable geometry.

**Worked reading.**

The operational question is what shape the vector has, how it is compared, and how training changes it.

| Object | Shape or formula | Role |
| --- | --- | --- |
| token ids | $B\times T$ | discrete sequence from tokenizer |
| embedding table | $|\mathcal{V}|\times d$ | row lookup for each token id |
| hidden states | $B\times T\times d$ | contextual vectors after lookup and layers |
| LM head | $d\times |\mathcal{V}|$ or tied $E^\top$ | maps hidden states to token logits |
| position signal | vector, rotation, or bias | injects order into attention |

Examples:

1. embedding rows.
2. hidden states.
3. similarity search.

Non-examples:

1. raw text in linear algebra.
2. ids treated as distances.

**Derivation habit.**

1. Write the tensor shape before writing the operation.
2. State whether vectors are raw input embeddings, hidden states, output rows, or retrieval embeddings.
3. Choose dot product, cosine similarity, or Euclidean distance deliberately.
4. Check whether position information is additive, rotary, learned, or an attention bias.
5. Track whether input and output embeddings are tied.

**Implementation lens.**

In code, embedding lookup is simple indexing. Conceptually, it is the step where the model stops seeing symbolic ids and starts seeing trainable vectors. That is why tokenizer changes, vocabulary resizing, and special-token handling are not superficial.

When debugging model behavior, inspect embedding norms, nearest neighbors, and the mean direction. Large norms or dominant components can affect logits and similarity search. For retrieval models, normalize vectors before cosine search unless the training objective explicitly uses norm as signal.

For transformer internals, remember that input embeddings are only the first residual-stream state. After attention and MLP layers, a token position's hidden state reflects surrounding context. Static nearest neighbors and contextual hidden-state probes answer different questions.

### 7.4 Dimensionality reduction diagnostics

**Purpose.** Dimensionality reduction diagnostics focuses on PCA views of local geometry. This matters because every later transformer operation starts from these vectors or from hidden states derived from them.

$$\mathbf{x}_t=\mathbf{e}_{i_t}^{\top}E.$$

**Operational definition.**

This concept explains how discrete language symbols become continuous vectors with trainable geometry.

**Worked reading.**

The operational question is what shape the vector has, how it is compared, and how training changes it.

| Object | Shape or formula | Role |
| --- | --- | --- |
| token ids | $B\times T$ | discrete sequence from tokenizer |
| embedding table | $|\mathcal{V}|\times d$ | row lookup for each token id |
| hidden states | $B\times T\times d$ | contextual vectors after lookup and layers |
| LM head | $d\times |\mathcal{V}|$ or tied $E^\top$ | maps hidden states to token logits |
| position signal | vector, rotation, or bias | injects order into attention |

Examples:

1. embedding rows.
2. hidden states.
3. similarity search.

Non-examples:

1. raw text in linear algebra.
2. ids treated as distances.

**Derivation habit.**

1. Write the tensor shape before writing the operation.
2. State whether vectors are raw input embeddings, hidden states, output rows, or retrieval embeddings.
3. Choose dot product, cosine similarity, or Euclidean distance deliberately.
4. Check whether position information is additive, rotary, learned, or an attention bias.
5. Track whether input and output embeddings are tied.

**Implementation lens.**

In code, embedding lookup is simple indexing. Conceptually, it is the step where the model stops seeing symbolic ids and starts seeing trainable vectors. That is why tokenizer changes, vocabulary resizing, and special-token handling are not superficial.

When debugging model behavior, inspect embedding norms, nearest neighbors, and the mean direction. Large norms or dominant components can affect logits and similarity search. For retrieval models, normalize vectors before cosine search unless the training objective explicitly uses norm as signal.

For transformer internals, remember that input embeddings are only the first residual-stream state. After attention and MLP layers, a token position's hidden state reflects surrounding context. Static nearest neighbors and contextual hidden-state probes answer different questions.

### 7.5 Embedding geometry in RAG

**Purpose.** Embedding geometry in RAG focuses on dense retrieval and semantic neighborhoods. This matters because every later transformer operation starts from these vectors or from hidden states derived from them.

$$\operatorname{cos}(\mathbf{u},\mathbf{v})=\frac{\langle \mathbf{u},\mathbf{v}\rangle}{\lVert\mathbf{u}\rVert_2\lVert\mathbf{v}\rVert_2}.$$

**Operational definition.**

Embeddings become contextual through projections, attention, MLPs, and residual additions. Attention compares projected vectors, not raw token ids.

**Worked reading.**

The query and key projections turn hidden states into vectors whose dot products define attention weights.

| Object | Shape or formula | Role |
| --- | --- | --- |
| token ids | $B\times T$ | discrete sequence from tokenizer |
| embedding table | $|\mathcal{V}|\times d$ | row lookup for each token id |
| hidden states | $B\times T\times d$ | contextual vectors after lookup and layers |
| LM head | $d\times |\mathcal{V}|$ or tied $E^\top$ | maps hidden states to token logits |
| position signal | vector, rotation, or bias | injects order into attention |

Examples:

1. QKV projections.
2. residual stream states.
3. dense retrieval vectors.

Non-examples:

1. nearest neighbors over integer ids.
2. one fixed meaning for a token in every context.

**Derivation habit.**

1. Write the tensor shape before writing the operation.
2. State whether vectors are raw input embeddings, hidden states, output rows, or retrieval embeddings.
3. Choose dot product, cosine similarity, or Euclidean distance deliberately.
4. Check whether position information is additive, rotary, learned, or an attention bias.
5. Track whether input and output embeddings are tied.

**Implementation lens.**

In code, embedding lookup is simple indexing. Conceptually, it is the step where the model stops seeing symbolic ids and starts seeing trainable vectors. That is why tokenizer changes, vocabulary resizing, and special-token handling are not superficial.

When debugging model behavior, inspect embedding norms, nearest neighbors, and the mean direction. Large norms or dominant components can affect logits and similarity search. For retrieval models, normalize vectors before cosine search unless the training objective explicitly uses norm as signal.

For transformer internals, remember that input embeddings are only the first residual-stream state. After attention and MLP layers, a token position's hidden state reflects surrounding context. Static nearest neighbors and contextual hidden-state probes answer different questions.

## 8. Scale and Diagnostics

Scale and Diagnostics connects token ids to continuous vectors and prepares the exact geometry used by attention, language-model logits, and dense retrieval.

### 8.1 Parameter counting

**Purpose.** Parameter counting focuses on vocabulary size times model width. This matters because every later transformer operation starts from these vectors or from hidden states derived from them.

$$\operatorname{params}_{\mathrm{embed}}=|\mathcal{V}|d_{\mathrm{model}}.$$

**Operational definition.**

Embedding tables are systems objects too: they consume memory, depend on tokenizer ids, and must handle special rows carefully.

**Worked reading.**

Changing a tokenizer changes which row each token id selects, so old weights no longer mean the same thing without migration.

| Object | Shape or formula | Role |
| --- | --- | --- |
| token ids | $B\times T$ | discrete sequence from tokenizer |
| embedding table | $|\mathcal{V}|\times d$ | row lookup for each token id |
| hidden states | $B\times T\times d$ | contextual vectors after lookup and layers |
| LM head | $d\times |\mathcal{V}|$ or tied $E^\top$ | maps hidden states to token logits |
| position signal | vector, rotation, or bias | injects order into attention |

Examples:

1. vocabulary resizing.
2. special token initialization.
3. embedding quantization.

Non-examples:

1. renaming token ids without changing weights.
2. ignoring padding row behavior.

**Derivation habit.**

1. Write the tensor shape before writing the operation.
2. State whether vectors are raw input embeddings, hidden states, output rows, or retrieval embeddings.
3. Choose dot product, cosine similarity, or Euclidean distance deliberately.
4. Check whether position information is additive, rotary, learned, or an attention bias.
5. Track whether input and output embeddings are tied.

**Implementation lens.**

In code, embedding lookup is simple indexing. Conceptually, it is the step where the model stops seeing symbolic ids and starts seeing trainable vectors. That is why tokenizer changes, vocabulary resizing, and special-token handling are not superficial.

When debugging model behavior, inspect embedding norms, nearest neighbors, and the mean direction. Large norms or dominant components can affect logits and similarity search. For retrieval models, normalize vectors before cosine search unless the training objective explicitly uses norm as signal.

For transformer internals, remember that input embeddings are only the first residual-stream state. After attention and MLP layers, a token position's hidden state reflects surrounding context. Static nearest neighbors and contextual hidden-state probes answer different questions.

### 8.2 Memory and quantization

**Purpose.** Memory and quantization focuses on storage cost of embedding rows. This matters because every later transformer operation starts from these vectors or from hidden states derived from them.

$$E\in\mathbb{R}^{|\mathcal{V}|\times d_{\mathrm{model}}},\qquad \mathbf{x}_t=E_{i_t,:}.$$

**Operational definition.**

Embedding tables are systems objects too: they consume memory, depend on tokenizer ids, and must handle special rows carefully.

**Worked reading.**

Changing a tokenizer changes which row each token id selects, so old weights no longer mean the same thing without migration.

| Object | Shape or formula | Role |
| --- | --- | --- |
| token ids | $B\times T$ | discrete sequence from tokenizer |
| embedding table | $|\mathcal{V}|\times d$ | row lookup for each token id |
| hidden states | $B\times T\times d$ | contextual vectors after lookup and layers |
| LM head | $d\times |\mathcal{V}|$ or tied $E^\top$ | maps hidden states to token logits |
| position signal | vector, rotation, or bias | injects order into attention |

Examples:

1. vocabulary resizing.
2. special token initialization.
3. embedding quantization.

Non-examples:

1. renaming token ids without changing weights.
2. ignoring padding row behavior.

**Derivation habit.**

1. Write the tensor shape before writing the operation.
2. State whether vectors are raw input embeddings, hidden states, output rows, or retrieval embeddings.
3. Choose dot product, cosine similarity, or Euclidean distance deliberately.
4. Check whether position information is additive, rotary, learned, or an attention bias.
5. Track whether input and output embeddings are tied.

**Implementation lens.**

In code, embedding lookup is simple indexing. Conceptually, it is the step where the model stops seeing symbolic ids and starts seeing trainable vectors. That is why tokenizer changes, vocabulary resizing, and special-token handling are not superficial.

When debugging model behavior, inspect embedding norms, nearest neighbors, and the mean direction. Large norms or dominant components can affect logits and similarity search. For retrieval models, normalize vectors before cosine search unless the training objective explicitly uses norm as signal.

For transformer internals, remember that input embeddings are only the first residual-stream state. After attention and MLP layers, a token position's hidden state reflects surrounding context. Static nearest neighbors and contextual hidden-state probes answer different questions.

### 8.3 Norm and similarity dashboards

**Purpose.** Norm and similarity dashboards focuses on monitoring geometry during training. This matters because every later transformer operation starts from these vectors or from hidden states derived from them.

$$\mathbf{x}_t=\mathbf{e}_{i_t}^{\top}E.$$

**Operational definition.**

Embedding spaces have geometry: norms, directions, subspaces, clusters, and dominant components. These structures can encode useful features and dataset artifacts.

**Worked reading.**

Centering an embedding cloud removes the mean direction; whitening rescales dominant axes so cosine neighborhoods are less dominated by global components.

| Object | Shape or formula | Role |
| --- | --- | --- |
| token ids | $B\times T$ | discrete sequence from tokenizer |
| embedding table | $|\mathcal{V}|\times d$ | row lookup for each token id |
| hidden states | $B\times T\times d$ | contextual vectors after lookup and layers |
| LM head | $d\times |\mathcal{V}|$ or tied $E^\top$ | maps hidden states to token logits |
| position signal | vector, rotation, or bias | injects order into attention |

Examples:

1. feature probes.
2. bias directions.
3. PCA diagnostics.

Non-examples:

1. assuming every axis has semantic meaning.
2. judging geometry from one 2D plot only.

**Derivation habit.**

1. Write the tensor shape before writing the operation.
2. State whether vectors are raw input embeddings, hidden states, output rows, or retrieval embeddings.
3. Choose dot product, cosine similarity, or Euclidean distance deliberately.
4. Check whether position information is additive, rotary, learned, or an attention bias.
5. Track whether input and output embeddings are tied.

**Implementation lens.**

In code, embedding lookup is simple indexing. Conceptually, it is the step where the model stops seeing symbolic ids and starts seeing trainable vectors. That is why tokenizer changes, vocabulary resizing, and special-token handling are not superficial.

When debugging model behavior, inspect embedding norms, nearest neighbors, and the mean direction. Large norms or dominant components can affect logits and similarity search. For retrieval models, normalize vectors before cosine search unless the training objective explicitly uses norm as signal.

For transformer internals, remember that input embeddings are only the first residual-stream state. After attention and MLP layers, a token position's hidden state reflects surrounding context. Static nearest neighbors and contextual hidden-state probes answer different questions.

### 8.4 Outlier tokens and special tokens

**Purpose.** Outlier tokens and special tokens focuses on why control rows need inspection. This matters because every later transformer operation starts from these vectors or from hidden states derived from them.

$$\operatorname{cos}(\mathbf{u},\mathbf{v})=\frac{\langle \mathbf{u},\mathbf{v}\rangle}{\lVert\mathbf{u}\rVert_2\lVert\mathbf{v}\rVert_2}.$$

**Operational definition.**

Embedding tables are systems objects too: they consume memory, depend on tokenizer ids, and must handle special rows carefully.

**Worked reading.**

Changing a tokenizer changes which row each token id selects, so old weights no longer mean the same thing without migration.

| Object | Shape or formula | Role |
| --- | --- | --- |
| token ids | $B\times T$ | discrete sequence from tokenizer |
| embedding table | $|\mathcal{V}|\times d$ | row lookup for each token id |
| hidden states | $B\times T\times d$ | contextual vectors after lookup and layers |
| LM head | $d\times |\mathcal{V}|$ or tied $E^\top$ | maps hidden states to token logits |
| position signal | vector, rotation, or bias | injects order into attention |

Examples:

1. vocabulary resizing.
2. special token initialization.
3. embedding quantization.

Non-examples:

1. renaming token ids without changing weights.
2. ignoring padding row behavior.

**Derivation habit.**

1. Write the tensor shape before writing the operation.
2. State whether vectors are raw input embeddings, hidden states, output rows, or retrieval embeddings.
3. Choose dot product, cosine similarity, or Euclidean distance deliberately.
4. Check whether position information is additive, rotary, learned, or an attention bias.
5. Track whether input and output embeddings are tied.

**Implementation lens.**

In code, embedding lookup is simple indexing. Conceptually, it is the step where the model stops seeing symbolic ids and starts seeing trainable vectors. That is why tokenizer changes, vocabulary resizing, and special-token handling are not superficial.

When debugging model behavior, inspect embedding norms, nearest neighbors, and the mean direction. Large norms or dominant components can affect logits and similarity search. For retrieval models, normalize vectors before cosine search unless the training objective explicitly uses norm as signal.

For transformer internals, remember that input embeddings are only the first residual-stream state. After attention and MLP layers, a token position's hidden state reflects surrounding context. Static nearest neighbors and contextual hidden-state probes answer different questions.

### 8.5 Migration and compatibility tests

**Purpose.** Migration and compatibility tests focuses on why tokenizer and embedding tables are coupled. This matters because every later transformer operation starts from these vectors or from hidden states derived from them.

$$\ell=-\log\frac{\exp(\mathbf{h}^{\top}\mathbf{w}_y)}{\sum_{j\in\mathcal{V}}\exp(\mathbf{h}^{\top}\mathbf{w}_j)}.$$

**Operational definition.**

Embedding tables are systems objects too: they consume memory, depend on tokenizer ids, and must handle special rows carefully.

**Worked reading.**

Changing a tokenizer changes which row each token id selects, so old weights no longer mean the same thing without migration.

| Object | Shape or formula | Role |
| --- | --- | --- |
| token ids | $B\times T$ | discrete sequence from tokenizer |
| embedding table | $|\mathcal{V}|\times d$ | row lookup for each token id |
| hidden states | $B\times T\times d$ | contextual vectors after lookup and layers |
| LM head | $d\times |\mathcal{V}|$ or tied $E^\top$ | maps hidden states to token logits |
| position signal | vector, rotation, or bias | injects order into attention |

Examples:

1. vocabulary resizing.
2. special token initialization.
3. embedding quantization.

Non-examples:

1. renaming token ids without changing weights.
2. ignoring padding row behavior.

**Derivation habit.**

1. Write the tensor shape before writing the operation.
2. State whether vectors are raw input embeddings, hidden states, output rows, or retrieval embeddings.
3. Choose dot product, cosine similarity, or Euclidean distance deliberately.
4. Check whether position information is additive, rotary, learned, or an attention bias.
5. Track whether input and output embeddings are tied.

**Implementation lens.**

In code, embedding lookup is simple indexing. Conceptually, it is the step where the model stops seeing symbolic ids and starts seeing trainable vectors. That is why tokenizer changes, vocabulary resizing, and special-token handling are not superficial.

When debugging model behavior, inspect embedding norms, nearest neighbors, and the mean direction. Large norms or dominant components can affect logits and similarity search. For retrieval models, normalize vectors before cosine search unless the training objective explicitly uses norm as signal.

For transformer internals, remember that input embeddings are only the first residual-stream state. After attention and MLP layers, a token position's hidden state reflects surrounding context. Static nearest neighbors and contextual hidden-state probes answer different questions.

## 9. Common Mistakes

| # | Mistake | Why it is wrong | Fix |
| --- | --- | --- | --- |
| 1 | Treating token ids as numeric magnitudes | Ids are arbitrary labels, not ordered measurements. | Use embedding lookup or one-hot selection. |
| 2 | Using dot product and cosine interchangeably | Dot product includes norm effects. | State whether magnitude should matter. |
| 3 | Assuming static token rows are final meaning | Transformer layers contextualize representations. | Distinguish input embeddings from hidden states. |
| 4 | Ignoring position information | Self-attention alone is permutation-equivariant. | Add or rotate positional information before attention. |
| 5 | Changing vocabulary without resizing embeddings | New ids need rows and output logits. | Resize, initialize, and train new rows explicitly. |
| 6 | Interpreting PCA plots too literally | Two dimensions can hide high-dimensional structure. | Use PCA as a diagnostic, not proof. |
| 7 | Forgetting anisotropy | Dominant directions can distort cosine neighbors. | Inspect mean vector, norm distribution, and centered similarities. |
| 8 | Assuming analogies always work | Linear offsets are empirical, domain-dependent approximations. | Validate with held-out relations. |
| 9 | Confusing retrieval embeddings with LM token embeddings | Retriever embeddings are usually pooled sequence vectors. | Name the embedding type and training objective. |
| 10 | Ignoring tied embeddings | Input and output tables may share parameters. | Check whether the LM head is tied to token embeddings. |

## 10. Exercises

1. (*) Perform embedding lookup for a batch of token ids.
   - (a) State the shape of every object.
   - (b) Compute the numeric result.
   - (c) Explain the LLM architecture consequence.

2. (*) Show one-hot lookup equivalence.
   - (a) State the shape of every object.
   - (b) Compute the numeric result.
   - (c) Explain the LLM architecture consequence.

3. (*) Compute cosine similarity and nearest neighbors.
   - (a) State the shape of every object.
   - (b) Compute the numeric result.
   - (c) Explain the LLM architecture consequence.

4. (**) Verify a synthetic analogy direction.
   - (a) State the shape of every object.
   - (b) Compute the numeric result.
   - (c) Explain the LLM architecture consequence.

5. (**) Measure anisotropy before and after centering.
   - (a) State the shape of every object.
   - (b) Compute the numeric result.
   - (c) Explain the LLM architecture consequence.

6. (**) Compute a softmax output-row gradient.
   - (a) State the shape of every object.
   - (b) Compute the numeric result.
   - (c) Explain the LLM architecture consequence.

7. (**) Build sinusoidal position encodings.
   - (a) State the shape of every object.
   - (b) Compute the numeric result.
   - (c) Explain the LLM architecture consequence.

8. (***) Apply a RoPE rotation and check norm preservation.
   - (a) State the shape of every object.
   - (b) Compute the numeric result.
   - (c) Explain the LLM architecture consequence.

9. (***) Build an ALiBi bias matrix.
   - (a) State the shape of every object.
   - (b) Compute the numeric result.
   - (c) Explain the LLM architecture consequence.

10. (***) Count embedding parameters and explain tied embeddings.
   - (a) State the shape of every object.
   - (b) Compute the numeric result.
   - (c) Explain the LLM architecture consequence.

## 11. Why This Matters for AI

| Concept | AI impact |
| --- | --- |
| Embedding lookup | Transforms token ids into vectors that the transformer can optimize over. |
| Similarity metrics | Support nearest neighbors, retrieval, clustering, probing, and semantic diagnostics. |
| Analogy directions | Reveal when relational information is approximately linear. |
| Anisotropy | Explains why raw embedding spaces can have poor neighborhood structure. |
| Training gradients | Show how token frequency and prediction errors move embedding rows. |
| Position encodings | Let attention use sequence order and relative distance. |
| QKV projections | Turn embeddings into attention queries, keys, and values. |
| Parameter counts | Tie vocabulary, tokenizer choice, model width, and serving memory together. |

## 12. Conceptual Bridge

The backward bridge is tokenization. Token ids are arbitrary labels until an embedding table gives them trainable vectors. The tokenizer and embedding table therefore form one coupled interface.

The forward bridge is attention. Queries, keys, and values are learned projections of hidden states that begin as embeddings plus position information. Attention is not separate from embedding geometry; it is built on top of it.

```text
+------------+      +------------------+      +-----------------------+
| token ids  | ---> | embedding rows   | ---> | contextual hidden      |
| B x T      |      | B x T x d        |      | states and attention   |
+------------+      +------------------+      +-----------------------+
```

A strong mental model is to treat embeddings as the model's input coordinate system. If that coordinate system is distorted, anisotropic, incompatible with the tokenizer, or poorly initialized for new tokens, every downstream layer inherits the problem.

## References

- Mikolov et al.. Efficient Estimation of Word Representations in Vector Space. https://arxiv.org/abs/1301.3781
- Pennington, Socher, Manning. GloVe: Global Vectors for Word Representation. https://aclanthology.org/D14-1162/
- Vaswani et al.. Attention Is All You Need. https://arxiv.org/abs/1706.03762
- Su et al.. RoFormer: Enhanced Transformer with Rotary Position Embedding. https://arxiv.org/abs/2104.09864
- Press et al.. Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation. https://arxiv.org/abs/2108.12409
