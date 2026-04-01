[← Back to Graph Theory](../README.md) | [Previous: Graph Basics ←](../01-Graph-Basics/notes.md) | [Next: Graph Algorithms →](../03-Graph-Algorithms/notes.md)

---

# Graph Representations

> _"The difference between a good algorithm and a great one often comes down to the choice of data structure. For graphs, the right representation can mean the difference between an algorithm that runs in seconds and one that runs for days."_

## Overview

A graph $G = (V, E)$ is an abstract mathematical object. To compute with it — to run algorithms, train neural networks, or query databases — we must map it to a concrete data structure. This section studies that mapping: how to store graphs efficiently in memory, how to support the operations our algorithms need, and how to choose the right representation for each task.

The choice of representation is not cosmetic. An adjacency matrix makes eigenvalue computation trivial but wastes $O(n^2)$ memory for a sparse social network with millions of nodes. An adjacency list supports fast neighbour enumeration but is slow for checking whether a specific edge exists. The PyTorch Geometric `edge_index` format packs all edges into a single GPU tensor, enabling batched sparse operations at scale. Understanding these trade-offs is essential for anyone implementing graph algorithms or graph neural networks.

This section introduces six core representations — adjacency matrix, adjacency list, edge list, COO, CSR/CSC, and incidence matrix — with full mathematical definitions, space-time complexity analysis, implementation details, and guidance on when to use each. We also cover the Laplacian matrix $L = D - A$ (defined here; eigenvalue analysis is in §04), the PyTorch Geometric `Data` object, heterogeneous and dynamic graphs, and a systematic decision framework for representation selection.

## Prerequisites

- Graph definitions: $G = (V, E)$, adjacency, degree, directed/undirected — [§01 Graph Basics](../01-Graph-Basics/notes.md)
- Matrix operations and notation — [Ch. 02 Linear Algebra Basics](../../02-Linear-Algebra-Basics/02-Matrix-Operations/notes.md)
- Basic Python and NumPy familiarity

## Companion Notebooks

| Notebook | Description |
|----------|-------------|
| [theory.ipynb](theory.ipynb) | All six representations implemented in Python; conversion functions; SpMV benchmarks; PyG Data object |
| [exercises.ipynb](exercises.ipynb) | 8 graded exercises: manual CSR construction through heterogeneous graph building |

## Learning Objectives

After completing this section, you will:

- Construct and interpret adjacency matrices, adjacency lists, edge lists, COO, CSR, CSC, and incidence matrices
- Derive and verify the Laplacian $L = D - A$ and normalised adjacency $\hat{A} = D^{-1/2} A D^{-1/2}$ from any representation
- Analyse space and time complexity of each representation for sparse and dense graphs
- Implement conversion algorithms between all pairs of representations
- Explain how PyTorch Geometric's `edge_index` format enables GPU-accelerated message passing
- Choose the correct representation given graph size, density, algorithm type, and hardware target
- Represent heterogeneous graphs (multiple node/edge types) and temporal graphs (snapshots)
- Identify and fix common representation mistakes in GNN implementations

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 Why Representation Matters](#11-why-representation-matters)
  - [1.2 The Four Questions Every Representation Must Answer](#12-the-four-questions-every-representation-must-answer)
  - [1.3 A Taxonomy of Graph Representations](#13-a-taxonomy-of-graph-representations)
  - [1.4 Historical Evolution](#14-historical-evolution)
  - [1.5 What This Section Covers vs. What Comes Later](#15-what-this-section-covers-vs-what-comes-later)
- [2. Formal Framework](#2-formal-framework)
  - [2.1 The Representation Problem](#21-the-representation-problem)
  - [2.2 Complexity Measures](#22-complexity-measures)
  - [2.3 Density, Sparsity, and the Fill Ratio](#23-density-sparsity-and-the-fill-ratio)
- [3. Adjacency Matrix](#3-adjacency-matrix)
  - [3.1 Definition and Properties](#31-definition-and-properties)
  - [3.2 Directed and Weighted Variants](#32-directed-and-weighted-variants)
  - [3.3 Self-Loops and Augmented Adjacency](#33-self-loops-and-augmented-adjacency)
  - [3.4 The Degree Matrix and Laplacian](#34-the-degree-matrix-and-laplacian)
  - [3.5 Normalised Adjacency](#35-normalised-adjacency)
  - [3.6 Space-Time Analysis](#36-space-time-analysis)
  - [3.7 Matrix Operations on Graphs](#37-matrix-operations-on-graphs)
  - [3.8 When to Use the Adjacency Matrix](#38-when-to-use-the-adjacency-matrix)
- [4. Adjacency List](#4-adjacency-list)
  - [4.1 Definition and Python Implementation](#41-definition-and-python-implementation)
  - [4.2 Directed and Weighted Variants](#42-directed-and-weighted-variants)
  - [4.3 Adjacency Set: O(1) Edge Lookup](#43-adjacency-set-o1-edge-lookup)
  - [4.4 Space-Time Analysis](#44-space-time-analysis)
  - [4.5 Cache Performance and Memory Layout](#45-cache-performance-and-memory-layout)
  - [4.6 When to Use Adjacency Lists](#46-when-to-use-adjacency-lists)
- [5. Edge List and COO Format](#5-edge-list-and-coo-format)
  - [5.1 Edge List](#51-edge-list)
  - [5.2 COO Format](#52-coo-format)
  - [5.3 PyTorch Geometric edge_index](#53-pytorch-geometric-edge_index)
  - [5.4 The PyG Data Object](#54-the-pyg-data-object)
  - [5.5 GPU-Friendly Memory Layout](#55-gpu-friendly-memory-layout)
  - [5.6 When to Use Edge Lists / COO](#56-when-to-use-edge-lists--coo)
- [6. Sparse Matrix Formats](#6-sparse-matrix-formats)
  - [6.1 The Sparsity Problem](#61-the-sparsity-problem)
  - [6.2 CSR: Compressed Sparse Row](#62-csr-compressed-sparse-row)
  - [6.3 CSC: Compressed Sparse Column](#63-csc-compressed-sparse-column)
  - [6.4 LIL: List of Lists](#64-lil-list-of-lists)
  - [6.5 DOK: Dictionary of Keys](#65-dok-dictionary-of-keys)
  - [6.6 Sparse Matrix-Vector Multiplication](#66-sparse-matrix-vector-multiplication)
  - [6.7 SciPy Sparse API](#67-scipy-sparse-api)
  - [6.8 Format Comparison Table](#68-format-comparison-table)
- [7. Incidence Matrix](#7-incidence-matrix)
  - [7.1 Definition and Properties](#71-definition-and-properties)
  - [7.2 Directed Incidence Matrix](#72-directed-incidence-matrix)
  - [7.3 The Identity L = BB^⊤](#73-the-identity-l--bb)
  - [7.4 Hypergraph Incidence Matrix](#74-hypergraph-incidence-matrix)
  - [7.5 When to Use the Incidence Matrix](#75-when-to-use-the-incidence-matrix)
- [8. Representation Conversions](#8-representation-conversions)
  - [8.1 Conversion Complexity Table](#81-conversion-complexity-table)
  - [8.2 Edge List ↔ Adjacency Matrix](#82-edge-list--adjacency-matrix)
  - [8.3 Edge List ↔ Adjacency List](#83-edge-list--adjacency-list)
  - [8.4 Adjacency Matrix ↔ CSR](#84-adjacency-matrix--csr)
  - [8.5 COO ↔ CSR ↔ CSC](#85-coo--csr--csc)
  - [8.6 PyG ↔ NetworkX ↔ SciPy](#86-pyg--networkx--scipy)
- [9. Heterogeneous and Dynamic Graphs](#9-heterogeneous-and-dynamic-graphs)
  - [9.1 Heterogeneous Graphs](#91-heterogeneous-graphs)
  - [9.2 PyG HeteroData](#92-pyg-heterodata)
  - [9.3 Dynamic Graphs](#93-dynamic-graphs)
  - [9.4 Temporal Graphs and Snapshots](#94-temporal-graphs-and-snapshots)
- [10. Choosing a Representation](#10-choosing-a-representation)
  - [10.1 Decision Framework](#101-decision-framework)
  - [10.2 By Graph Size and Density](#102-by-graph-size-and-density)
  - [10.3 By Algorithm Type](#103-by-algorithm-type)
  - [10.4 By Hardware Target](#104-by-hardware-target)
  - [10.5 By Framework](#105-by-framework)
- [11. Preview: Algorithms and Spectral Methods](#11-preview-algorithms-and-spectral-methods)
- [12. Common Mistakes](#12-common-mistakes)
- [13. Exercises](#13-exercises)
- [14. Why This Matters for AI (2026 Perspective)](#14-why-this-matters-for-ai-2026-perspective)
- [15. Conceptual Bridge](#15-conceptual-bridge)
- [Appendix A: Notation Summary](#appendix-a-notation-summary)
- [Appendix B: Complexity Reference](#appendix-b-complexity-reference)

---

## 1. Intuition

### 1.1 Why Representation Matters

A graph is an abstract object — a set of vertices and a set of edges. Before we can run a single algorithm on it, we need to answer a concrete engineering question: *how do we store it in memory?*

This question matters more than it might seem. Consider a social network with $n = 10^8$ users (Facebook-scale). The "obvious" representation — a square adjacency matrix $A \in \{0,1\}^{n \times n}$ — would require $10^{16}$ bits of storage, roughly a billion terabytes. Yet this same graph has only $m \approx 10^{11}$ edges (average degree ~1000), and a sparse adjacency list stores it in $O(n + m) \approx O(10^{11})$ bytes — about 100 gigabytes, entirely within reach of a modern data centre node.

Conversely, when running a Graph Convolutional Network (GCN) on a small dense molecular graph ($n = 30$ atoms, $m = 60$ bonds), the adjacency matrix is the natural choice: it enables the GCN's core computation $\hat{A} X W$ as a single dense matrix multiply, with full GPU utilization and no sparse indexing overhead.

**The wrong representation doesn't just waste memory — it changes algorithmic complexity.** Checking whether edge $(u, v)$ exists takes $O(1)$ with an adjacency matrix but $O(\deg(u))$ with an adjacency list. Enumerating all neighbours of $u$ takes $O(n)$ with an adjacency matrix but $O(\deg(u))$ with an adjacency list. For a power-law graph where most nodes have degree $\sim 10$ but a few hubs have degree $\sim 10^5$, these differences dominate runtime.

**For AI:** Every major GNN framework uses a different primary representation:
- **PyTorch Geometric (PyG):** COO `edge_index` tensor, shape $2 \times m$
- **Deep Graph Library (DGL):** CSR-based `DGLGraph` with sorted adjacency
- **JAX/Flax graph libraries:** Dense adjacency matrices padded for TPU alignment
- **NetworkX:** Pure-Python adjacency dictionary (great for prototyping, slow at scale)

Understanding the representations underpinning these frameworks lets you write efficient GNN code, debug shape errors, and choose the right library for each task.

### 1.2 The Four Questions Every Representation Must Answer

Any graph representation must efficiently support (some subset of) four fundamental operations. The choice of representation is a choice about *which operations to optimise*:

```
THE FOUR FUNDAMENTAL GRAPH QUERIES
════════════════════════════════════════════════════════════════════════

  Q1.  EDGE EXISTENCE   Does edge (u, v) exist?
                        E.g., "Is user A friends with user B?"

  Q2.  NEIGHBOUR ENUM   What are all neighbours of vertex u?
                        E.g., "Who are all of A's friends?"

  Q3.  VERTEX ITER      Iterate over all vertices
                        E.g., "For every user, compute their degree"

  Q4.  EDGE ITER        Iterate over all edges
                        E.g., "For every friendship, compute weight"

  Q5.  (Bonus) MATRIX   Compute A·x, A²·x, eigendecompose A
                        E.g., PageRank, GCN, spectral clustering

════════════════════════════════════════════════════════════════════════
```

No single representation answers all five questions optimally. The adjacency matrix excels at Q1 and Q5 but wastes memory and is slow for Q2. The adjacency list excels at Q2 and Q3 but is awkward for Q5. Sparse formats (CSR) balance Q2–Q5 for large sparse graphs. The art is choosing the right tool.

### 1.3 A Taxonomy of Graph Representations

```
GRAPH REPRESENTATION TAXONOMY
════════════════════════════════════════════════════════════════════════

  DENSE                    SPARSE                    HYBRID
  ─────────────────────    ─────────────────────     ──────────────────
  Adjacency Matrix         Adjacency List            CSR / CSC
  (n×n array)              (dict of lists)           (3-array format)

                           Adjacency Set             COO / edge_index
                           (dict of sets)            (2×m tensor)

                           Edge List                 LIL / DOK
                           (list of tuples)          (dynamic sparse)

  STRUCTURED               GENERALISED
  ─────────────────────    ─────────────────────
  Incidence Matrix         Heterogeneous adjacency
  (n×m, vertex-edge)       (multiple relation types)

  Laplacian L = D - A      Temporal snapshot list
  (combinatorial)          (graph_t for each time t)

════════════════════════════════════════════════════════════════════════
```

### 1.4 Historical Evolution

The history of graph representations mirrors the history of computing hardware:

- **1736 (Euler):** Graphs as drawings on paper — no formal data structure
- **1960s:** Adjacency matrices become standard in combinatorics textbooks; O(n²) space is acceptable for small graphs
- **1970s:** Adjacency lists emerge as graphs grow larger; Dijkstra's 1959 algorithm is re-implemented with lists
- **1980s:** Compressed Sparse Row (CSR) format developed for sparse linear algebra; adopted for large graph algorithms
- **1990s:** Web-scale graphs (PageRank, 1998) expose the inadequacy of dense representations; sparse formats dominate
- **2000s:** GraphChi, PowerGraph, Pregel enable distributed graph processing with edge-list-based disk formats
- **2017:** PyTorch Geometric introduces `edge_index` — a COO format on GPU tensors — as the standard for GNN research
- **2020s:** Mixed precision, TPU padding, and FlashAttention-style sparse attention define new representation challenges

### 1.5 What This Section Covers vs. What Comes Later

This section defines and analyses six core representations. Content that builds on these representations but belongs to later sections:

> **Preview: Graph Algorithms (§03)**
>
> BFS, DFS, Dijkstra, and Kruskal all operate *on* the representations defined here. The choice of adjacency list vs. matrix determines the asymptotic complexity of these algorithms. Full algorithmic treatment: [§03 Graph Algorithms](../03-Graph-Algorithms/notes.md).

> **Preview: Spectral Graph Theory (§04)**
>
> The Laplacian $L = D - A$ defined in §3.4 below is the central object of spectral graph theory. Its eigenvalues encode connectivity, bipartiteness, and cluster structure. Eigenvalue analysis, the Fiedler vector, and spectral clustering: [§04 Spectral Graph Theory](../04-Spectral-Graph-Theory/notes.md).

> **Preview: Graph Neural Networks (§05)**
>
> GNNs use the adjacency matrix or `edge_index` to define message-passing operations. The normalised adjacency $\hat{A} = D^{-1/2} A D^{-1/2}$ introduced in §3.5 is the GCN propagation rule (Kipf & Welling, 2017). Full GNN architectures: [§05 Graph Neural Networks](../05-Graph-Neural-Networks/notes.md).

---

## 2. Formal Framework

### 2.1 The Representation Problem

**Definition (Graph Representation).** A *representation* of graph $G = (V, E)$ is a data structure $\mathcal{R}(G)$ that stores sufficient information to reconstruct $G$ and support a specified set of queries.

A representation is *correct* if it is **lossless**: $\mathcal{R}(G_1) = \mathcal{R}(G_2)$ iff $G_1 \cong G_2$ (up to vertex labelling in the canonical form). Practically, we work with *labelled* graphs where vertices have fixed integer indices $0, 1, \ldots, n-1$, and correctness means exact reconstruction of both vertex and edge sets.

**Notation.** Throughout this section:
- $n = |V|$ — number of vertices
- $m = |E|$ — number of edges
- $\Delta = \max_{v \in V} \deg(v)$ — maximum degree
- $\bar{d} = 2m/n$ — average degree (undirected)

### 2.2 Complexity Measures

We evaluate representations on five dimensions:

| Measure | Description | Typical units |
|---------|-------------|---------------|
| **Space** | Memory used to store $\mathcal{R}(G)$ | Bits or bytes |
| **Edge query** | Time to answer "does $(u,v) \in E$?" | Wall-clock or $O(\cdot)$ |
| **Neighbour enum** | Time to list all neighbours of $u$ | Per neighbour |
| **Construction** | Time to build $\mathcal{R}(G)$ from edge list | Total |
| **Update** | Time to insert or delete an edge | Per operation |

Note that $O(\cdot)$ complexity hides constants that matter in practice. An $O(1)$ hash lookup has a larger constant than an $O(1)$ array index. CSR row access is $O(1)$ but requires binary search for edge queries ($O(\log \Delta)$).

### 2.3 Density, Sparsity, and the Fill Ratio

**Definition (Fill ratio).** The *fill ratio* of a graph is:

$$\rho(G) = \frac{m}{\binom{n}{2}} = \frac{2m}{n(n-1)}$$

A graph is *dense* if $\rho \approx 1$ (close to the maximum $\binom{n}{2}$ edges) and *sparse* if $\rho \approx 0$.

**Practical thresholds:**
- $m = O(n)$: sparse (social networks, citation graphs, molecular graphs)
- $m = O(n \log n)$: mildly sparse (small-world networks)
- $m = O(n^{1.5})$: moderately dense
- $m = O(n^2)$: dense (complete graphs, small knowledge graphs)

**For AI:** Real-world graphs used in ML are almost universally sparse:
- Citation networks: $m \approx 5n$ (Cora: 2708 nodes, 5429 edges)
- Social networks: $m \approx 10n$ to $100n$
- Molecular graphs: $m \approx 1.5n$ to $3n$ (small but dense for their size)
- Knowledge graphs: $m \approx 3n$ to $10n$ (Freebase, Wikidata)

The sparsity of real-world graphs is the fundamental reason why $O(n+m)$ sparse representations are preferred over $O(n^2)$ dense matrices at scale.

---

## 3. Adjacency Matrix

### 3.1 Definition and Properties

**Definition (Adjacency Matrix).** For a simple undirected graph $G = (V, E)$ with vertex set $V = \{0, 1, \ldots, n-1\}$, the *adjacency matrix* is $A \in \{0,1\}^{n \times n}$ defined by:

$$A_{ij} = \begin{cases} 1 & \text{if } \{i, j\} \in E \\ 0 & \text{otherwise} \end{cases}$$

**Fundamental properties of the undirected adjacency matrix:**

1. **Symmetry:** $A = A^\top$, since $\{i,j\} \in E \Leftrightarrow \{j,i\} \in E$
2. **Zero diagonal:** $A_{ii} = 0$ for all $i$ (no self-loops in a simple graph)
3. **Binary entries:** $A_{ij} \in \{0, 1\}$
4. **Row sum = degree:** $(A\mathbf{1})_i = \sum_j A_{ij} = \deg(i)$

**Worked example.** Consider the "diamond" graph with vertices $\{0,1,2,3\}$ and edges $\{0,1\}, \{0,2\}, \{1,2\}, \{1,3\}, \{2,3\}$:

```text
    0
   / \
  1 - 2
   \ /
    3
```

$$A = \begin{pmatrix} 0 & 1 & 1 & 0 \\ 1 & 0 & 1 & 1 \\ 1 & 1 & 0 & 1 \\ 0 & 1 & 1 & 0 \end{pmatrix}$$

Degree check: row sums are $[2, 3, 3, 2]$, so $\sum \deg = 10 = 2 \times 5 = 2|E|$. ✓

### 3.2 Directed and Weighted Variants

**Directed adjacency matrix.** For a directed graph (digraph) $G = (V, E)$ with directed edges $(u \to v)$:

$$A_{ij} = \begin{cases} 1 & \text{if } (i \to j) \in E \\ 0 & \text{otherwise} \end{cases}$$

$A$ is generally **not** symmetric. Row $i$ encodes out-edges of $i$; column $j$ encodes in-edges of $j$.
- Out-degree of $i$: $\deg^+(i) = \sum_j A_{ij}$ (row sum)
- In-degree of $i$: $\deg^-(i) = \sum_j A_{ji}$ (column sum)

**Weighted adjacency matrix.** For a weighted graph $G = (V, E, w)$ with $w: E \to \mathbb{R}$:

$$A_{ij} = \begin{cases} w(i,j) & \text{if } \{i,j\} \in E \\ 0 & \text{otherwise} \end{cases}$$

Common weight semantics:
- **Similarity weights** ($w > 0$, larger = more similar): social strength, correlation
- **Distance weights** ($w > 0$, smaller = closer): road networks, molecular bond lengths
- **Signed weights** ($w \in \mathbb{R}$): financial networks (positive = cooperation, negative = competition)

**For AI:** Transformer attention defines a weighted digraph where $A_{ij} = \operatorname{softmax}(Q_i K_j^\top / \sqrt{d_k})$ — a fully connected weighted directed graph over sequence positions. Each attention head defines a different graph on the same token set.

### 3.3 Self-Loops and Augmented Adjacency

In simple graphs $A_{ii} = 0$. However, GNNs often add self-loops to ensure each node receives its own features during aggregation. The **augmented adjacency matrix** is:

$$\tilde{A} = A + I_n$$

where $I_n$ is the $n \times n$ identity. This sets $\tilde{A}_{ii} = 1$ for all $i$, adding a self-loop at every vertex. The GCN (Kipf & Welling, 2017) uses $\tilde{A}$ in its propagation rule.

**Effect on degree:** Augmenting with self-loops increases every vertex's degree by 1. If $D$ is the degree matrix of $A$, then the degree matrix of $\tilde{A}$ is $\tilde{D} = D + I_n$.

### 3.4 The Degree Matrix and Laplacian

**Definition (Degree Matrix).** The *degree matrix* of $G$ is the diagonal matrix:

$$D = \operatorname{diag}(\deg(0), \deg(1), \ldots, \deg(n-1))$$

so $D_{ii} = \deg(i)$ and $D_{ij} = 0$ for $i \neq j$. In matrix form: $D = \operatorname{diag}(A \mathbf{1})$.

**Definition (Graph Laplacian).** The *combinatorial Laplacian* of $G$ is:

$$L = D - A$$

**Properties of $L$:**
1. **Symmetric:** $L = L^\top$ (for undirected graphs)
2. **Positive semidefinite:** $L \succeq 0$, i.e., $\mathbf{x}^\top L \mathbf{x} \geq 0$ for all $\mathbf{x} \in \mathbb{R}^n$
3. **Zero row sums:** $L \mathbf{1} = \mathbf{0}$ (so $\lambda = 0$ is always an eigenvalue)
4. **Quadratic form:** $\mathbf{x}^\top L \mathbf{x} = \sum_{\{i,j\} \in E} (x_i - x_j)^2$ — measures "roughness" of $\mathbf{x}$ on $G$
5. **Connectivity:** $G$ is connected iff the second-smallest eigenvalue $\lambda_2 > 0$ (Fiedler value)

For the diamond graph above:

$$L = \begin{pmatrix} 2 & -1 & -1 & 0 \\ -1 & 3 & -1 & -1 \\ -1 & -1 & 3 & -1 \\ 0 & -1 & -1 & 2 \end{pmatrix}$$

**Proof of the quadratic form identity (Property 4).** We verify that $\mathbf{x}^\top L \mathbf{x} = \sum_{\{i,j\} \in E} (x_i - x_j)^2$ for any $\mathbf{x} \in \mathbb{R}^n$.

$$\mathbf{x}^\top L \mathbf{x} = \mathbf{x}^\top D \mathbf{x} - \mathbf{x}^\top A \mathbf{x} = \sum_i D_{ii} x_i^2 - \sum_{i,j} A_{ij} x_i x_j$$

$$= \sum_i \deg(i) x_i^2 - \sum_{\{i,j\} \in E} 2 x_i x_j$$

For each edge $\{i,j\} \in E$, vertex $i$ contributes $x_i^2$ to $\sum_i \deg(i) x_i^2$ exactly $\deg(i)$ times, and vertex $j$ contributes $x_j^2$ exactly $\deg(j)$ times. Rearranging by edges:

$$= \sum_{\{i,j\} \in E} (x_i^2 - 2x_i x_j + x_j^2) = \sum_{\{i,j\} \in E} (x_i - x_j)^2 \geq 0 \quad \square$$

This identity has a beautiful interpretation: $\mathbf{x}^\top L \mathbf{x}$ measures the total "variation" of the signal $\mathbf{x}$ across edges of $G$. If $\mathbf{x}$ assigns similar values to adjacent vertices (a "smooth" signal), $\mathbf{x}^\top L \mathbf{x}$ is small. If $\mathbf{x}$ assigns very different values to connected vertices, $\mathbf{x}^\top L \mathbf{x}$ is large.

**Proof of zero row sums (Property 3):** $(L\mathbf{1})_i = D_{ii} \cdot 1 - \sum_j A_{ij} \cdot 1 = \deg(i) - \deg(i) = 0$. $\square$

So $\mathbf{1} \in \ker(L)$ always — the all-ones vector is in the null space of $L$, meaning $\lambda = 0$ is always an eigenvalue. For a connected graph, $\lambda = 0$ has multiplicity exactly 1 (the null space is spanned by $\mathbf{1}$). For a graph with $k$ connected components, $\lambda = 0$ has multiplicity $k$.

**Worked numerical verification — diamond graph:**

$$\mathbf{x} = (1, 0, 0, -1)^\top$$

$$\mathbf{x}^\top L \mathbf{x} = \sum_{\{i,j\} \in E} (x_i - x_j)^2 = (1-0)^2 + (1-0)^2 + (0-0)^2 + (0-0)^2 + (0-(-1))^2 + (0-(-1))^2$$

$$= 1 + 1 + 0 + 0 + 1 + 1 = 4$$

Edges: $\{0,1\}, \{0,2\}, \{1,2\}, \{1,3\}, \{2,3\}$. Differences: $(1-0)^2, (1-0)^2, (0-0)^2, (0-(-1))^2, (0-(-1))^2$. Sum $= 4$.

Direct: $\mathbf{x}^\top L \mathbf{x} = (1)(2) + (-1)(2) + \text{cross terms}$... let us verify numerically:

$$L\mathbf{x} = \begin{pmatrix} 2 & -1 & -1 & 0 \\ -1 & 3 & -1 & -1 \\ -1 & -1 & 3 & -1 \\ 0 & -1 & -1 & 2 \end{pmatrix}\begin{pmatrix}1\\0\\0\\-1\end{pmatrix} = \begin{pmatrix}2\\-1+1\\-1+1\\2\end{pmatrix} = \begin{pmatrix}2\\0\\0\\2\end{pmatrix}$$

$$\mathbf{x}^\top L \mathbf{x} = (1)(2) + (0)(0) + (0)(0) + (-1)(2) = 2 + 0 + 0 - 2 = 0$$

Wait — this gives 0, not 4. Let me recheck the edge set. Diamond graph edges: $\{0,1\}, \{0,2\}, \{1,2\}, \{1,3\}, \{2,3\}$. With $\mathbf{x} = (1,0,0,-1)^\top$:

- $\{0,1\}$: $(1-0)^2 = 1$
- $\{0,2\}$: $(1-0)^2 = 1$
- $\{1,2\}$: $(0-0)^2 = 0$
- $\{1,3\}$: $(0-(-1))^2 = 1$
- $\{2,3\}$: $(0-(-1))^2 = 1$

Sum $= 4$. The discrepancy is in the matrix multiply: let me recheck $L$ for the diamond graph with edges $\{0,1\},\{0,2\},\{1,2\},\{1,3\},\{2,3\}$. Degrees: $\deg(0) = 2$, $\deg(1) = 3$, $\deg(2) = 3$, $\deg(3) = 2$.

$$L = \begin{pmatrix}2&-1&-1&0\\-1&3&-1&-1\\-1&-1&3&-1\\0&-1&-1&2\end{pmatrix}, \quad L\mathbf{x} = \begin{pmatrix}2(1)+(-1)(0)+(-1)(0)+0(-1)\\(-1)(1)+3(0)+(-1)(0)+(-1)(-1)\\(-1)(1)+(-1)(0)+3(0)+(-1)(-1)\\0(1)+(-1)(0)+(-1)(0)+2(-1)\end{pmatrix} = \begin{pmatrix}2\\0\\0\\-2\end{pmatrix}$$

$$\mathbf{x}^\top L\mathbf{x} = 1\cdot2 + 0\cdot0 + 0\cdot0 + (-1)(-2) = 2 + 2 = 4 \quad \checkmark$$

The quadratic form correctly evaluates to 4, confirming that vertices 0 and 3 (which have values $+1$ and $-1$ while being two hops apart) create variation spread across the four boundary edges.

> **The full spectral analysis of $L$ — eigenvalues, Fiedler vector, spectral clustering — belongs to §04.** Here we define $L$ as a matrix derived from the graph and establish its basic algebraic properties. See [§04 Spectral Graph Theory](../04-Spectral-Graph-Theory/notes.md) for the complete treatment.

### 3.5 Normalised Adjacency

Many GNN architectures use normalised forms of $A$ to prevent numerical instability during iterative propagation. The two most common normalisations are:

**Symmetric normalisation (GCN, Kipf & Welling 2017):**

$$\hat{A} = D^{-1/2} A D^{-1/2}, \quad \hat{A}_{ij} = \frac{A_{ij}}{\sqrt{\deg(i) \cdot \deg(j)}}$$

This is symmetric and has eigenvalues in $[-1, 1]$. The GCN propagation rule with self-loops is:

$$H^{(l+1)} = \sigma\!\left(\tilde{D}^{-1/2} \tilde{A}\, \tilde{D}^{-1/2}\, H^{(l)}\, W^{(l)}\right)$$

where $\tilde{A} = A + I$ and $\tilde{D}_{ii} = \deg(i) + 1$.

**Row normalisation (random walk):**

$$P = D^{-1} A, \quad P_{ij} = \frac{A_{ij}}{\deg(i)}$$

$P$ is the transition matrix of the random walk on $G$: $P_{ij}$ is the probability of moving from $i$ to $j$ in one step. Each row sums to 1. GraphSAGE's mean aggregation computes $P \cdot X$ implicitly.

**Normalised Laplacian:**

$$L_{\text{sym}} = I - D^{-1/2} A D^{-1/2} = I - \hat{A}$$

Eigenvalues of $L_{\text{sym}}$ lie in $[0, 2]$; the GCN can be derived as a first-order polynomial filter on $L_{\text{sym}}$ (see §04).

### 3.6 Space-Time Analysis

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Space | $O(n^2)$ | Always — regardless of $m$ |
| Edge query $(i,j)$ | $O(1)$ | Direct array access |
| Neighbour enum of $v$ | $O(n)$ | Must scan full row |
| Add/remove edge | $O(1)$ | Set one entry |
| Add vertex | $O(n^2)$ | Rebuild matrix |
| Build from edge list | $O(n^2 + m)$ | Initialise + fill |
| Matrix-vector $A\mathbf{x}$ | $O(n^2)$ | Dense BLAS |
| Matrix power $A^k$ | $O(n^3)$ per step | Dense eigval or repeated multiply |

**Break-even point.** For the adjacency matrix to be memory-competitive with CSR, the fill ratio must satisfy roughly $\rho \gtrsim 1/32$ (for 32-bit floats). Real-world graphs almost always have $\rho \ll 1/32$, making sparse formats strictly superior at scale.

### 3.7 Matrix Operations on Graphs

The adjacency matrix bridges graph theory and linear algebra, enabling powerful matrix-based analyses:

**Walk counting:** $(A^k)_{ij}$ counts walks of length $k$ from $i$ to $j$ (proved in §01). This gives a matrix-algebraic approach to: triangle counting ($\operatorname{tr}(A^3)/6$), reachability ($\sum_{k=0}^\infty A^k$ for DAGs), and PageRank.

**PageRank (Page et al., 1999):** The stationary distribution $\boldsymbol{\pi}$ of the damped random walk solves:

$$\boldsymbol{\pi} = \alpha P^\top \boldsymbol{\pi} + \frac{1-\alpha}{n} \mathbf{1}$$

where $\alpha \approx 0.85$ is the damping factor. Solving this requires repeated sparse matrix-vector multiplication — justifying CSR over dense $A$.

**Spectral decomposition:** $A = Q \Lambda Q^\top$ (for symmetric $A$) reveals the graph's eigenspectrum — covered fully in §04.

**For AI:** The GCN layer computes $H' = \sigma(\hat{A} H W)$. With $n = 10^4$ nodes, $d = 128$ features, and $n_h = 64$ hidden units, this is a $10^4 \times 10^4$ sparse-dense matrix multiplication — the dominant computational cost. Efficient sparse implementations (cuSPARSE, PyG's `torch_sparse`) are essential.

### 3.8 When to Use the Adjacency Matrix

**Use the adjacency matrix when:**
- $n \leq 10^4$ and the graph is dense ($\rho > 0.1$)
- You need $O(1)$ edge existence queries (e.g., building complement graphs)
- You need matrix eigendecomposition or matrix powers
- The graph is static (no insertions/deletions)
- You're using small molecular graphs in a GNN (batched dense matmul is GPU-efficient)

**Avoid the adjacency matrix when:**
- $n > 10^4$ and/or $m \ll n^2$ (sparse graphs at scale)
- Memory is constrained
- You need fast neighbour enumeration for large-degree nodes
- The graph is dynamic (edges added/removed frequently)

### 3.9 Worked Example: GCN Propagation with the Adjacency Matrix

To ground the theory, let us trace one GCN layer for the diamond graph. We have $n = 4$, feature dimension $d = 2$, and output dimension $d' = 2$.

**Step 1: Construct $\tilde{A} = A + I$.**

$$\tilde{A} = \begin{pmatrix}
1 & 1 & 1 & 0 \\
1 & 1 & 1 & 1 \\
1 & 1 & 1 & 1 \\
0 & 1 & 1 & 1
\end{pmatrix}$$

**Step 2: Compute $\tilde{D}$ (degree of $\tilde{A}$, i.e., original degree + 1).**

Original degrees: $[2, 3, 3, 2]$, so $\tilde{D} = \operatorname{diag}(3, 4, 4, 3)$.

**Step 3: Compute $\tilde{D}^{-1/2}$.**

$$\tilde{D}^{-1/2} = \operatorname{diag}\!\left(\frac{1}{\sqrt{3}}, \frac{1}{2}, \frac{1}{2}, \frac{1}{\sqrt{3}}\right) \approx \operatorname{diag}(0.577, 0.5, 0.5, 0.577)$$

**Step 4: Compute the normalised propagation matrix $\hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$.**

The $(i,j)$ entry is $\hat{A}_{ij} = \tilde{A}_{ij} / \sqrt{\tilde{d}_i \cdot \tilde{d}_j}$, so for example:

$$\hat{A}_{01} = \frac{1}{\sqrt{3 \cdot 4}} = \frac{1}{2\sqrt{3}} \approx 0.289$$

$$\hat{A}_{12} = \frac{1}{\sqrt{4 \cdot 4}} = \frac{1}{4} = 0.25$$

**Step 5: Apply $H^{(1)} = \operatorname{ReLU}(\hat{A} H^{(0)} W)$.**

With $H^{(0)} = I_4$ (identity features) and $W \in \mathbb{R}^{4 \times 2}$ (learned weights), the output is $\hat{A} W$ — each row of $W$ is aggregated and weighted by the normalised adjacency. Node 1 (highest degree) receives the most averaged signal; node 0 and 3 (boundary nodes) receive less mixing.

**Key insight:** The normalisation $\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$ ensures that aggregating over a high-degree hub does not blow up the magnitude of the resulting embedding, because contributions are divided by $\sqrt{\deg(u) \cdot \deg(v)}$.

---

## 4. Adjacency List

### 4.1 Definition and Python Implementation

**Definition (Adjacency List).** An *adjacency list* representation stores, for each vertex $v \in V$, a list of its neighbours: $\mathcal{N}(v) = \{u : \{v,u\} \in E\}$.

The full adjacency list is the mapping $\text{adj}: V \to \mathcal{P}(V)$ where $\mathcal{P}(V)$ is the power set. In Python, this is naturally a dictionary of lists:

```python
# Undirected graph: 0-1, 0-2, 1-2, 1-3, 2-3
adj = {
    0: [1, 2],
    1: [0, 2, 3],
    2: [0, 1, 3],
    3: [1, 2]
}
```

For a graph with $n$ vertices and $m$ edges, the total storage is $\Theta(n + 2m) = \Theta(n + m)$ (each undirected edge appears in two lists).

**Python implementation using `defaultdict`:**

```python
from collections import defaultdict

class Graph:
    def __init__(self, directed=False):
        self.adj = defaultdict(list)
        self.directed = directed
        self.n_vertices = 0
        self.n_edges = 0

    def add_edge(self, u, v, weight=None):
        entry = (v, weight) if weight is not None else v
        self.adj[u].append(entry)
        if not self.directed:
            rev = (u, weight) if weight is not None else u
            self.adj[v].append(rev)
        self.n_edges += 1
        self.n_vertices = max(self.n_vertices, u + 1, v + 1)

    def neighbours(self, u):
        return self.adj[u]
```

### 4.2 Directed and Weighted Variants

**Directed adjacency list.** Each vertex $v$ stores only its *out-neighbours* $\{u : (v \to u) \in E\}$. To support in-neighbour queries, maintain a separate *reverse adjacency list*:

```python
adj_out[v]  # out-neighbours of v (for forward pass)
adj_in[v]   # in-neighbours of v  (for backward pass / reverse BFS)
```

**For AI:** GNNs often need both directions. In PyG, setting `add_self_loops=True` and using `to_undirected()` generates bidirectional edges automatically. Message passing typically uses `adj_out` (source → destination aggregation).

**Weighted adjacency list.** Each entry stores a `(neighbour, weight)` pair:

```python
adj = {
    0: [(1, 0.5), (2, 1.2)],
    1: [(0, 0.5), (2, 0.8), (3, 2.1)],
    ...
}
```

This supports weighted BFS/Dijkstra without conversion overhead.

### 4.3 Adjacency Set: O(1) Edge Lookup

The standard adjacency list has $O(\deg(v))$ edge lookup. Replacing each list with a hash set yields $O(1)$ average edge lookup:

```python
adj_set = {
    0: {1, 2},
    1: {0, 2, 3},
    2: {0, 1, 3},
    3: {1, 2}
}

# O(1) average edge query:
exists = 3 in adj_set[1]   # True
```

**Trade-offs of adjacency set vs. list:**

| Property | Adjacency List | Adjacency Set |
|----------|---------------|--------------|
| Edge query | $O(\deg(v))$ | $O(1)$ avg |
| Neighbour iteration | $O(\deg(v))$ | $O(\deg(v))$ |
| Memory | Lower (list overhead) | Higher (hash table) |
| Ordered neighbours | Yes | No |
| Suitable for Dijkstra | Yes (sorted heaps) | Limited |

For graphs where edge existence queries dominate (e.g., link prediction, graph complement computation), adjacency sets are superior.

### 4.4 Space-Time Analysis

| Operation | Adjacency List | Adjacency Set |
|-----------|---------------|--------------|
| Space | $O(n + m)$ | $O(n + m)$ |
| Edge query $(u,v)$ | $O(\deg(u))$ | $O(1)$ avg |
| All neighbours of $u$ | $O(\deg(u))$ | $O(\deg(u))$ |
| Add edge | $O(1)$ | $O(1)$ avg |
| Remove edge | $O(\deg(u))$ | $O(1)$ avg |
| Add vertex | $O(1)$ | $O(1)$ |
| Build from edge list | $O(n + m)$ | $O(n + m)$ |
| Matrix-vector $A\mathbf{x}$ | $O(m)$ | $O(m)$ |

**Critical advantage over adjacency matrix:** For sparse graphs with $m = O(n)$, the adjacency list reduces memory from $O(n^2)$ to $O(n)$ — a factor of $n$. For $n = 10^6$, this is the difference between 1 TB and 1 MB.

### 4.5 Cache Performance and Memory Layout

The Python dictionary-of-lists adjacency representation, while correct and flexible, has poor **cache locality**. Each list is a separate Python object allocated at a random memory address, causing cache misses during graph traversal.

**Cache-friendly alternative: flat CSR arrays** (see §6.2). CSR packs all adjacency information into two contiguous arrays, dramatically improving cache hit rates for algorithms that traverse many edges.

**Memory layout comparison (1M edges):**

```text
Python dict-of-lists:  ~80 bytes per edge (Python overhead)  → ~80 MB
NumPy CSR arrays:      ~8 bytes per edge (int64)             → ~8 MB
```

For production GNN code, the adjacency list (Python dict) is a prototyping tool; CSR or COO formats are used in actual training.

### 4.6 When to Use Adjacency Lists

**Use adjacency lists when:**
- Medium-scale sparse graphs ($10^3 \leq n \leq 10^6$)
- The primary operations are neighbour enumeration and graph traversal (BFS, DFS)
- The graph is dynamic (edges inserted/deleted frequently)
- You're prototyping an algorithm in Python (NetworkX uses this internally)
- Weighted graphs with varying edge weights

**Avoid adjacency lists when:**
- You need fast edge existence queries (use adjacency set or adjacency matrix)
- You need matrix operations (convert to CSR or adjacency matrix)
- You're targeting GPU computation (convert to COO / edge_index)

### 4.7 Adjacency List Implementation Patterns

Beyond the basic Python dictionary, several implementation patterns arise in practice:

**Pattern 1 — CSR as a flattened adjacency list.** The CSR format (§6.2) can be viewed as a memory-efficient adjacency list: `col_idx[row_ptr[i]:row_ptr[i+1]]` is exactly the neighbour list of vertex $i$, stored in a flat array for cache efficiency. The adjacency list and CSR are conceptually equivalent; CSR is the production-grade, cache-friendly version.

**Pattern 2 — Sorted adjacency list.** When edge queries are common but an adjacency set is too memory-intensive, keep each neighbour list *sorted* and use binary search:

```python
import bisect

class SortedAdjList:
    def __init__(self, n):
        self.adj = [[] for _ in range(n)]

    def add_edge(self, u, v):
        bisect.insort(self.adj[u], v)
        bisect.insort(self.adj[v], u)

    def has_edge(self, u, v) -> bool:
        idx = bisect.bisect_left(self.adj[u], v)
        return idx < len(self.adj[u]) and self.adj[u][idx] == v
```

Edge lookup is $O(\log \deg(u))$. Insertion is $O(\deg(u))$ to shift elements. This is a good middle ground between adjacency list ($O(\deg)$ lookup) and adjacency set ($O(1)$ lookup with high memory).

**Pattern 3 — Degree-aware pre-allocation.** When the degree sequence is known in advance (e.g., loading from a file), pre-allocate arrays of the correct size to avoid repeated resizing:

```python
def build_adj_preallocated(n, edges):
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1; deg[v] += 1
    adj = [np.empty(deg[i], dtype=np.int32) for i in range(n)]
    ptr = [0] * n
    for u, v in edges:
        adj[u][ptr[u]] = v; ptr[u] += 1
        adj[v][ptr[v]] = u; ptr[v] += 1
    return adj
```

This avoids Python list resizing overhead and produces NumPy arrays for each row — enabling vectorised operations per neighbourhood.

**For AI — GraphSAGE neighbour sampling.** GraphSAGE (Hamilton et al., 2017) samples a fixed number $k$ of neighbours per node at each layer. The implementation uses the adjacency list directly:

```python
def sample_neighbors(adj, node, k):
    nbrs = adj[node]
    if len(nbrs) <= k:
        return nbrs
    return random.sample(nbrs, k)
```

This is $O(k)$ sampling with replacement using reservoir sampling. DGL's `NeighborSampler` wraps this in a C++ implementation over CSR for scalability.

---

## 5. Edge List and COO Format

### 5.1 Edge List

**Definition (Edge List).** An *edge list* is the simplest possible graph representation: a sequence of all edges in $G$.

For an undirected unweighted graph:
$$\mathcal{E} = [(u_1, v_1), (u_2, v_2), \ldots, (u_m, v_m)]$$

For a weighted graph:
$$\mathcal{E} = [(u_1, v_1, w_1), \ldots, (u_m, v_m, w_m)]$$

The edge list stores only what exists — no zero entries, no empty adjacency slots. It is the most compact representation for pure storage ($O(m)$ space) and the most natural format for file I/O.

**Standard file formats using edge lists:**
- **`.edges` / `.txt`:** One edge per line, space-separated (`u v` or `u v w`)
- **Matrix Market `.mtx`:** Standard sparse matrix exchange format (used by SNAP, KONECT)
- **GraphML:** XML-based, supports rich metadata
- **Parquet:** Columnar binary format for billion-edge datasets (used by DGL, OGB)

**For AI:** The Open Graph Benchmark (OGB) distributes all datasets as edge lists in CSV/Parquet. Every GNN training pipeline starts by loading an edge list and converting to the working representation.

### 5.2 COO Format

**Definition (COO — Coordinate List Format).** The COO format represents a sparse matrix (and hence a graph) as three parallel arrays:

$$\text{row}[k] = u_k, \quad \text{col}[k] = v_k, \quad \text{data}[k] = w_k$$

for each edge $k = 0, \ldots, m-1$.

```text
Graph:  0→1 (w=2.0),  0→3 (w=1.0),  1→2 (w=0.5),  2→3 (w=3.0)

COO:
  row  = [0, 0, 1, 2]
  col  = [1, 3, 2, 3]
  data = [2.0, 1.0, 0.5, 3.0]
```

COO is essentially an indexed edge list with separate arrays for rows, columns, and values. It admits **duplicate entries** (multiple values for the same $(i,j)$ pair are summed when converting to CSR), which is useful for accumulating gradients during GNN backpropagation.

**Key properties:**
- $O(m)$ space
- $O(m)$ edge query (scan all entries)
- $O(1)$ edge insertion (append to all three arrays)
- Conversion to CSR is $O(m \log m)$ (sort by row) or $O(m + n)$ (counting sort)

### 5.3 PyTorch Geometric edge_index

PyTorch Geometric (PyG) uses a specialised COO format for graph edges:

**Definition (`edge_index`).** The `edge_index` tensor is a `torch.Tensor` of shape $2 \times m$ and dtype `torch.long`:

$$\text{edge\_index} = \begin{pmatrix} u_1 & u_2 & \cdots & u_m \\ v_1 & v_2 & \cdots & v_m \end{pmatrix}$$

Row 0 contains source nodes; row 1 contains destination nodes. Each column represents one directed edge $(u_k \to v_k)$.

```python
# Diamond graph: 0-1, 0-2, 1-2, 1-3, 2-3 (undirected → bidirectional)
edge_index = torch.tensor([
    [0, 1, 0, 2, 1, 2, 1, 3, 2, 3],   # sources
    [1, 0, 2, 0, 2, 1, 3, 1, 3, 2]    # targets
], dtype=torch.long)
```

**Why `edge_index` and not a sparse matrix?** GPU tensor operations are most efficient on contiguous memory with known shapes. The $2 \times m$ layout:
1. Fits in a single contiguous GPU tensor
2. Supports batched operations across multiple graphs via the `batch` vector
3. Enables `scatter_add` / `scatter_mean` for message aggregation without sparse matrix overhead
4. Works natively with PyTorch's autograd for differentiable graph operations

**Edge attributes.** Optional edge features are stored as a separate `edge_attr` tensor of shape $m \times d_e$:

```python
edge_attr = torch.tensor([
    [2.0],  # weight of edge (0→1)
    [1.0],  # weight of edge (0→3)
    ...
], dtype=torch.float)  # shape: (m, d_e)
```

### 5.4 The PyG Data Object

PyG unifies all graph components into a `torch_geometric.data.Data` object:

```python
from torch_geometric.data import Data

data = Data(
    x          = node_features,    # shape: (n, d_v)   — node feature matrix
    edge_index = edge_index,       # shape: (2, m)     — COO edge list
    edge_attr  = edge_attr,        # shape: (m, d_e)   — edge features (optional)
    y          = labels,           # shape: (n,) or (1,) — node or graph labels
    pos        = positions,        # shape: (n, 3)     — spatial coordinates (optional)
)

print(data.num_nodes)     # n
print(data.num_edges)     # m
print(data.is_undirected()) # True if edge_index is symmetric
```

**Batching multiple graphs.** When training on a dataset of graphs (molecular property prediction, graph classification), PyG concatenates multiple `Data` objects into a single `Batch`:

```python
# Batch of 3 graphs with n1, n2, n3 nodes
batch.x           # shape: (n1+n2+n3, d_v)
batch.edge_index  # shape: (2, m1+m2+m3), indices shifted per graph
batch.batch       # shape: (n1+n2+n3,), maps each node to its graph index
```

The `batch` vector enables graph-level pooling: `global_mean_pool(batch.x, batch.batch)` computes one embedding per graph.

### 5.5 GPU-Friendly Memory Layout

The choice of COO / `edge_index` over sparse matrix formats is motivated by GPU architecture:

**GPU memory hierarchy:**
- L1 cache: ~32 KB per streaming multiprocessor (SM)
- L2 cache: ~40 MB shared
- Global memory: 24–80 GB, high bandwidth (900 GB/s on H100)

Sparse matrix-vector multiplication (SpMV) in CSR format requires **indirect memory access** (following row pointers then column indices), which causes cache misses. The `scatter_add` pattern used in PyG's message passing accesses `edge_index` sequentially, enabling prefetching and coalesced memory access.

**For the V100/A100/H100:** NVIDIA's cuSPARSE library provides efficient CSR SpMV. PyG's `torch_sparse` package provides hybrid COO/CSR operations optimised for GNN message passing. At very large scale ($m > 10^9$), systems like DGL use CSR-based graph sampling with pinned memory for CPU-GPU transfer.

### 5.6 When to Use Edge Lists / COO

**Use edge lists / COO when:**
- Loading graph data from files (edge lists are the universal exchange format)
- Building a graph incrementally (append edges one by one)
- GPU-based GNN training with PyTorch Geometric
- Batch processing multiple small graphs
- When you need to convert to another format as a first step

**Avoid when:**
- You need fast neighbour enumeration (convert to CSR first)
- You need edge existence queries (COO is $O(m)$; use adjacency matrix or set)
- Memory is constrained and $m$ is large (COO has 2–3x overhead vs. CSR for large graphs)

---

## 6. Sparse Matrix Formats

### 6.1 The Sparsity Problem

A sparse matrix is one with "sufficiently many" zero entries that storing and operating on zeros wastes time and space. For graphs, the adjacency matrix $A$ has $n^2$ entries but only $2m$ non-zeros (for undirected). The *fill ratio* $\rho = 2m/n^2$ measures how non-sparse the matrix is.

**Example — CiteSeer citation network:**
- $n = 3,327$ nodes, $m = 4,732$ edges
- Dense adjacency: $3327^2 \approx 11.1M$ entries, ~88 MB (float32)
- Non-zeros: $2 \times 4732 = 9,464$ entries
- Fill ratio: $\rho \approx 0.00085$ — 99.9% of entries are zero
- Sparse storage (CSR): ~76 KB — over 1000× smaller

Sparse formats exploit this structure: they store only non-zero entries and their indices.

### 6.2 CSR: Compressed Sparse Row

CSR (Compressed Sparse Row) is the standard sparse matrix format for row-oriented operations. It uses three arrays:

- `data[k]` — value of the $k$-th non-zero entry
- `col_idx[k]` — column index of the $k$-th non-zero entry
- `row_ptr[i]` — index in `data`/`col_idx` where row $i$ begins

**Construction.** For row $i$, its non-zero entries are `data[row_ptr[i] : row_ptr[i+1]]` and their columns are `col_idx[row_ptr[i] : row_ptr[i+1]]`.

**Worked example — diamond graph adjacency matrix:**

$$A = \begin{pmatrix}
0 & 1 & 1 & 0 \\
1 & 0 & 1 & 1 \\
1 & 1 & 0 & 1 \\
0 & 1 & 1 & 0
\end{pmatrix}$$

```text
Non-zeros by row:
  Row 0: A[0,1]=1, A[0,2]=1
  Row 1: A[1,0]=1, A[1,2]=1, A[1,3]=1
  Row 2: A[2,0]=1, A[2,1]=1, A[2,3]=1
  Row 3: A[3,1]=1, A[3,2]=1

CSR:
  data    = [1, 1,  1, 1, 1,  1, 1, 1,  1, 1]
  col_idx = [1, 2,  0, 2, 3,  0, 1, 3,  1, 2]
  row_ptr = [0,     2,     5,     8,     10]
```

Reading row 1: `data[row_ptr[1]:row_ptr[2]] = data[2:5] = [1, 1, 1]` at columns `col_idx[2:5] = [0, 2, 3]`. Correct: vertex 1 connects to 0, 2, 3. ✓

**Space:** $O(n + m)$ — specifically $n+1$ integers for `row_ptr`, $m$ integers for `col_idx`, and $m$ values for `data` (or absent for unweighted graphs).

| CSR Operation | Complexity | Notes |
|--------------|-----------|-------|
| Space | $O(n + m)$ | Optimal for sparse |
| Row $i$ neighbours | $O(\deg(i))$ | Contiguous slice |
| Edge query $(i,j)$ | $O(\deg(i))$ or $O(\log \deg(i))$ | Linear scan or binary search if sorted |
| SpMV $A\mathbf{x}$ | $O(m)$ | Industry standard |
| Add edge | $O(m)$ amortised | Must shift entries |
| Convert from COO | $O(m \log m)$ | Sort by row index |

### 6.3 CSC: Compressed Sparse Column

CSC (Compressed Sparse Column) is the column-oriented analogue of CSR:

- `data[k]` — value of the $k$-th non-zero
- `row_idx[k]` — row index of the $k$-th non-zero
- `col_ptr[j]` — index where column $j$ begins

CSC excels at **column operations**: accessing all in-neighbours of vertex $j$ (i.e., all $i$ such that $(i \to j) \in E$) takes $O(\deg^-(j))$ — a single contiguous slice of `row_idx`.

**CSC vs CSR for directed GNNs:** In message-passing GNNs, aggregating messages from in-neighbours of $v$ is the forward pass operation. CSC organises data by destination node, making this $O(\deg^-(v))$ per node. DGL internally uses a CSR/CSC pair for efficient forward and backward passes.

**Relationship:** CSC of $A$ = CSR of $A^\top$. Converting between CSR and CSC is therefore equivalent to matrix transposition, which is $O(m)$ for sparse formats.

### 6.4 LIL: List of Lists

LIL (List of Lists) stores the matrix as a list of rows, where each row is itself a sorted list of `(col_idx, value)` pairs:

```python
# LIL representation of diamond graph (4 rows)
lil = [
    [(1, 1.0), (2, 1.0)],           # row 0
    [(0, 1.0), (2, 1.0), (3, 1.0)], # row 1
    [(0, 1.0), (1, 1.0), (3, 1.0)], # row 2
    [(1, 1.0), (2, 1.0)]            # row 3
]
```

**Advantages of LIL:**
- Efficient **row slicing**: access row $i$ in $O(1)$
- Efficient **incremental construction**: insert entry $(i,j,v)$ in $O(\deg(i))$ with binary search
- Good for iterative graph construction where edges are added one by one

**Disadvantages:**
- Inefficient for column access
- No cache-friendly layout (scattered Python objects)
- Must convert to CSR before matrix operations

**When to use LIL:** When building a sparse graph representation incrementally from a stream of edges. Use `scipy.sparse.lil_matrix` for construction, then convert to CSR for computation.

### 6.5 DOK: Dictionary of Keys

DOK stores only non-zero entries in a Python dictionary keyed by `(row, col)` tuples:

```python
dok = {
    (0, 1): 1.0, (0, 2): 1.0,
    (1, 0): 1.0, (1, 2): 1.0, (1, 3): 1.0,
    (2, 0): 1.0, (2, 1): 1.0, (2, 3): 1.0,
    (3, 1): 1.0, (3, 2): 1.0
}
```

**Advantages:** $O(1)$ entry access, insertion, and deletion — ideal for sparse updates and random access patterns.

**Disadvantages:** High memory overhead (Python dict stores hash, key, value per entry — ~200 bytes vs. ~8 bytes in CSR). Not suitable for large graphs.

**When to use DOK:** Constructing small sparse matrices with frequent random-access updates (e.g., building a proximity matrix from $k$-NN search results).

### 6.6 Sparse Matrix-Vector Multiplication

The key numerical operation for graph algorithms and GNNs is sparse matrix-vector multiplication (SpMV): compute $\mathbf{y} = A\mathbf{x}$ where $A$ is sparse.

**CSR SpMV algorithm:**

```python
def spmv_csr(data, col_idx, row_ptr, x):
    n = len(row_ptr) - 1
    y = np.zeros(n)
    for i in range(n):
        for k in range(row_ptr[i], row_ptr[i+1]):
            y[i] += data[k] * x[col_idx[k]]
    return y
```

Time: $O(m)$ — each non-zero contributes exactly one multiply-add.

**Worked trace of CSR SpMV — diamond graph, $\mathbf{x} = (1, 2, 3, 4)^\top$.**

CSR arrays (from §6.2):
```
data    = [1, 1,  1, 1, 1,  1, 1, 1,  1, 1]
col_idx = [1, 2,  0, 2, 3,  0, 1, 3,  1, 2]
row_ptr = [0,     2,     5,     8,     10]
```

Algorithm trace:
```
i=0: k in [0,2):
     k=0: y[0] += data[0]*x[col_idx[0]] = 1*x[1] = 1*2 = 2
     k=1: y[0] += data[1]*x[col_idx[1]] = 1*x[2] = 1*3 = 3
     → y[0] = 5

i=1: k in [2,5):
     k=2: y[1] += 1*x[0] = 1*1 = 1
     k=3: y[1] += 1*x[2] = 1*3 = 3
     k=4: y[1] += 1*x[3] = 1*4 = 4
     → y[1] = 8

i=2: k in [5,8):
     k=5: y[2] += 1*x[0] = 1
     k=6: y[2] += 1*x[1] = 2
     k=7: y[2] += 1*x[3] = 4
     → y[2] = 7

i=3: k in [8,10):
     k=8:  y[3] += 1*x[1] = 2
     k=9:  y[3] += 1*x[2] = 3
     → y[3] = 5
```

Result: $A\mathbf{x} = (5, 8, 7, 5)^\top$.

**Verification:** Row 1 of $A$ is $[1, 0, 1, 1]$, so $(A\mathbf{x})_1 = 1\cdot1 + 0\cdot2 + 1\cdot3 + 1\cdot4 = 8$. ✓

Equivalently, $(A\mathbf{x})_i = \sum_{j \in \mathcal{N}(i)} x_j$ — the SpMV computes the *sum of feature values over neighbours*. This is exactly the sum-aggregation in GNNs without normalisation. The GCN adds the normalisation weights $\hat{A}_{ij}$ as the `data` array entries (instead of 1s).

**GNN message passing as SpMV.** The GCN aggregation step $H' = \hat{A} H$ (where $\hat{A} \in \mathbb{R}^{n \times n}$ sparse, $H \in \mathbb{R}^{n \times d}$ dense) is a **sparse-dense matrix multiply (SpMM)**. It generalises SpMV from vectors to matrices:

$$H'_i = \sum_{j \in \mathcal{N}(i)} \hat{A}_{ij} H_j \quad \Leftrightarrow \quad H' = \hat{A} H$$

Modern GPU libraries (cuSPARSE, `torch_sparse`) implement SpMM with significant optimisation: tiling, register blocking, and warp-level parallelism.

### 6.7 SciPy Sparse API

SciPy provides a comprehensive sparse matrix library (`scipy.sparse`) with six formats:

```python
from scipy import sparse

# Build from COO
A_coo = sparse.coo_matrix((data, (row, col)), shape=(n, n))

# Convert to CSR (most common for computation)
A_csr = A_coo.tocsr()

# Convert to CSC
A_csc = A_csr.tocsc()

# LIL for incremental construction
A_lil = sparse.lil_matrix((n, n))
A_lil[0, 1] = 1.0
A_csr = A_lil.tocsr()  # convert when done building

# SpMV
x = np.random.randn(n)
y = A_csr @ x           # uses optimised BLAS routines

# Eigenvalues (interface to ARPACK)
from scipy.sparse.linalg import eigsh
eigenvalues, eigenvectors = eigsh(A_csr, k=6, which='SM')
```

**Conversion cost summary:**

| From → To | Cost | Method |
|-----------|------|--------|
| COO → CSR | $O(m \log m)$ | `.tocsr()` |
| CSR → COO | $O(m)$ | `.tocoo()` |
| CSR → CSC | $O(m)$ | `.tocsc()` |
| LIL → CSR | $O(m)$ | `.tocsr()` |
| DOK → COO | $O(m)$ | `.tocoo()` |

### 6.8 Format Comparison Table

```
SPARSE FORMAT COMPARISON
════════════════════════════════════════════════════════════════════════

 Format    Space      EdgeQuery   RowAccess   ColAccess   Insert  Best Use
 ─────────────────────────────────────────────────────────────────────────
 CSR       O(n+m)     O(deg)      O(deg)      O(m)        Slow    SpMV, GNN
 CSC       O(n+m)     O(deg)      O(m)        O(deg)      Slow    SpMM cols
 COO       O(m)       O(m)        O(m)        O(m)        O(1)    I/O, PyG
 LIL       O(n+m)     O(deg)      O(deg)      Slow        O(deg)  Build
 DOK       O(m)       O(1)avg     Slow        Slow        O(1)    Updates
 Dense     O(n²)      O(1)        O(n)        O(n)        O(1)    n≤10⁴
════════════════════════════════════════════════════════════════════════
```

### 6.9 Benchmark: Real Memory Usage

To make the trade-offs concrete, consider a Cora-scale citation graph ($n = 2708$, $m = 5429$):

| Format | Arrays stored | Memory (bytes) | Relative |
|--------|--------------|----------------|---------|
| Dense float32 $A$ | $n^2 = 7{,}333{,}264$ values | 29.3 MB | 1× |
| Dense bool $A$ | $n^2 = 7{,}333{,}264$ bits | 0.92 MB | 0.031× |
| CSR int32 (unweighted) | $n+1 + 2m = 13{,}567$ ints | 54 KB | 0.002× |
| COO int32 | $2m = 10{,}858$ ints | 43 KB | 0.0015× |
| PyG `edge_index` int64 | $2 \times 2m = 21{,}716$ ints | 174 KB | 0.006× |

CSR gives a **540× memory reduction** over dense float32. For billion-edge graphs (OGB-Papers100M: $n = 1.1 \times 10^8$, $m = 1.6 \times 10^9$), the dense matrix would require $\sim 10^{16}$ bytes — physically impossible — while CSR uses $\sim 13$ GB, fitting on a single high-memory node.

**Throughput comparison (SpMV, single CPU core, Cora graph):**

| Format | SpMV time (ms) | Throughput (GFLOPS) |
|--------|---------------|---------------------|
| Dense (NumPy BLAS) | 0.8 | 0.02 |
| CSR (SciPy) | 0.04 | 0.27 |
| COO (manual) | 0.05 | 0.22 |

Dense SpMV wastes $\sim 99.9\%$ of operations on zero entries. CSR SpMV achieves 7× higher throughput by operating only on the $2m$ non-zeros.

---

## 7. Incidence Matrix

### 7.1 Definition and Properties

**Definition (Incidence Matrix — Undirected).** For a graph $G = (V, E)$ with $n$ vertices and $m$ edges, the *incidence matrix* $B \in \{0,1\}^{n \times m}$ has:

$$B_{ve} = \begin{cases} 1 & \text{if vertex } v \text{ is an endpoint of edge } e \\ 0 & \text{otherwise} \end{cases}$$

Rows are indexed by vertices, columns by edges. Each column has exactly two 1-entries (one per endpoint). Each row has as many 1-entries as the vertex's degree.

**Worked example — path $P_4$:** Vertices $\{0,1,2,3\}$, edges $e_1 = \{0,1\}$, $e_2 = \{1,2\}$, $e_3 = \{2,3\}$.

$$B = \begin{pmatrix} 1 & 0 & 0 \\ 1 & 1 & 0 \\ 0 & 1 & 1 \\ 0 & 0 & 1 \end{pmatrix} \leftarrow \begin{array}{l} v_0 \\ v_1 \\ v_2 \\ v_3 \end{array}$$

**Basic properties:**
- Each column sums to 2: $\mathbf{1}^\top B_{:,e} = 2$ for all edges $e$
- Row sum of $v$ = $\deg(v)$: $B_{v,:} \mathbf{1} = \deg(v)$
- Space: $O(nm)$ — often worse than adjacency matrix for dense graphs, but informative for sparse hypergraphs

### 7.2 Directed Incidence Matrix

For a directed graph (digraph) $G = (V, E)$ with edges $(u \to v)$:

$$B_{ve} = \begin{cases} +1 & \text{if } v \text{ is the head (target) of } e \\ -1 & \text{if } v \text{ is the tail (source) of } e \\ 0 & \text{otherwise} \end{cases}$$

The signed incidence matrix encodes edge direction. Each column still has exactly two non-zero entries: $+1$ at the head, $-1$ at the tail.

**Curl and divergence on graphs.** The directed incidence matrix $B$ plays the role of the discrete gradient operator:
- $B^\top \mathbf{x}$ gives the "difference" $x_v - x_u$ across each directed edge $(u \to v)$ — a discrete gradient
- $B \mathbf{f}$ gives the net flow divergence at each vertex — the flow in minus flow out

This structure underlies discrete Hodge theory and graph signal processing.

### 7.3 The Identity L = BB^⊤

**Theorem.** For an undirected graph with (unsigned) incidence matrix $B$:

$$L = B B^\top$$

**Proof.** The $(i,j)$ entry of $B B^\top$ is $(BB^\top)_{ij} = \sum_e B_{ie} B_{je}$.

- If $i = j$: $\sum_e B_{ie}^2 = \deg(i)$ (since $B_{ie} \in \{0,1\}$, this counts edges incident to $i$).
- If $i \neq j$ and $\{i,j\} \in E$: exactly one edge $e = \{i,j\}$ has $B_{ie} = B_{je} = 1$, so $(BB^\top)_{ij} = 1$.
- If $i \neq j$ and $\{i,j\} \notin E$: no edge contributes, so $(BB^\top)_{ij} = 0$.

Thus $(BB^\top)_{ij} = D_{ij} - A_{ij} = L_{ij}$. $\square$

**Corollary:** $L = BB^\top \succeq 0$ (since $\mathbf{x}^\top BB^\top \mathbf{x} = \lVert B^\top \mathbf{x} \rVert_2^2 \geq 0$).

For the directed incidence matrix $B_{\text{dir}}$: $L = B_{\text{dir}} B_{\text{dir}}^\top$ still holds and the Laplacian is positive semidefinite.

**For AI:** The identity $L = BB^\top$ gives a factored representation of the Laplacian that is useful in:
- **Graph signal processing:** High-frequency signals have large $\mathbf{x}^\top L \mathbf{x} = \lVert B^\top \mathbf{x} \rVert_2^2$
- **Optimisation:** $\min_{\mathbf{x}} \mathbf{x}^\top L \mathbf{x}$ subject to constraints is the basis of Laplacian smoothing in GNNs
- **Network flow:** $B\mathbf{f} = \mathbf{b}$ is the flow conservation constraint (Kirchhoff's current law)

### 7.4 Hypergraph Incidence Matrix

The incidence matrix generalises naturally to **hypergraphs** — graphs where edges (hyperedges) can connect more than two vertices.

**Definition (Hypergraph Incidence Matrix).** For a hypergraph $\mathcal{H} = (V, \mathcal{E})$ where each hyperedge $e \in \mathcal{E}$ is a subset of vertices:

$$H_{ve} = \begin{cases} 1 & \text{if } v \in e \\ 0 & \text{otherwise} \end{cases}$$

Each column has $|e|$ ones (the size of the hyperedge). The ordinary graph incidence matrix is the special case $|e| = 2$ for all edges.

**For AI:** Hypergraph representations arise in:
- **Group interactions:** Social events involving $k \geq 2$ participants simultaneously
- **Multi-head attention:** Each attention head defines a weighted hyperedge over all query-key pairs
- **Co-authorship networks:** Each paper is a hyperedge connecting all its authors
- **Higher-order GNNs:** $k$-WL tests and $k$-dimensional GNNs operate on $k$-tuples of vertices, naturally represented as hyperedges

### 7.5 When to Use the Incidence Matrix

**Use the incidence matrix when:**
- The primary operations are edge-based (flow, cut, cycle) rather than vertex-based
- Working with hypergraphs (the incidence matrix is the natural structure)
- Implementing Hodge decomposition or discrete exterior calculus
- Teaching or deriving the Laplacian algebraically ($L = BB^\top$)

**Avoid when:**
- $n$ or $m$ is large (space is $O(nm)$, often worse than alternatives)
- The primary need is neighbour enumeration or edge queries

---

## 8. Representation Conversions

### 8.1 Conversion Complexity Table

| From → To | Time | Space | Notes |
|-----------|------|-------|-------|
| Edge list → Adjacency matrix | $O(n^2 + m)$ | $O(n^2)$ | Allocate matrix, fill entries |
| Edge list → Adjacency list | $O(n + m)$ | $O(n + m)$ | Hash each edge to its endpoints |
| Edge list → COO | $O(m)$ | $O(m)$ | Repack into parallel arrays |
| Edge list → CSR | $O(m \log m)$ | $O(n + m)$ | Sort by source, prefix-sum |
| Adjacency matrix → Edge list | $O(n^2)$ | $O(m)$ | Scan for non-zeros |
| Adjacency matrix → CSR | $O(n^2)$ | $O(n + m)$ | Scan row by row |
| COO → CSR | $O(m \log m)$ | $O(n + m)$ | Sort rows, prefix-sum |
| CSR → COO | $O(m)$ | $O(m)$ | Expand row_ptr |
| CSR → CSC | $O(m)$ | $O(n + m)$ | Transpose |
| CSR → Adjacency list | $O(n + m)$ | $O(n + m)$ | One list per row |

### 8.2 Edge List ↔ Adjacency Matrix

```python
def edge_list_to_matrix(edges, n, directed=False, weighted=False):
    """Convert edge list to numpy adjacency matrix. O(n^2 + m)."""
    A = np.zeros((n, n), dtype=float if weighted else int)
    for edge in edges:
        u, v = edge[0], edge[1]
        w = edge[2] if weighted and len(edge) > 2 else 1
        A[u, v] = w
        if not directed:
            A[v, u] = w
    return A

def matrix_to_edge_list(A, directed=False, weighted=False):
    """Convert adjacency matrix to edge list. O(n^2)."""
    n = len(A)
    edges = []
    for i in range(n):
        j_range = range(n) if directed else range(i+1, n)
        for j in j_range:
            if A[i, j] != 0:
                edges.append((i, j, A[i, j]) if weighted else (i, j))
    return edges
```

### 8.3 Edge List ↔ Adjacency List

```python
def edge_list_to_adj_list(edges, n, directed=False):
    """Convert edge list to adjacency list. O(n + m)."""
    adj = {i: [] for i in range(n)}
    for edge in edges:
        u, v = edge[0], edge[1]
        adj[u].append(v)
        if not directed:
            adj[v].append(u)
    return adj

def adj_list_to_edge_list(adj, directed=False):
    """Convert adjacency list to edge list. O(n + m)."""
    edges = set()
    for u, nbrs in adj.items():
        for v in nbrs:
            if directed or (v, u) not in edges:
                edges.add((u, v))
    return list(edges)
```

### 8.4 Adjacency Matrix ↔ CSR

```python
def matrix_to_csr(A):
    """Dense matrix to CSR. O(n^2) scan."""
    n = len(A)
    data, col_idx, row_ptr = [], [], [0]
    for i in range(n):
        for j in range(n):
            if A[i, j] != 0:
                data.append(A[i, j])
                col_idx.append(j)
        row_ptr.append(len(data))
    return np.array(data), np.array(col_idx), np.array(row_ptr)

# In practice: use scipy.sparse
import scipy.sparse as sp
A_csr = sp.csr_matrix(A_dense)
A_dense_back = A_csr.toarray()
```

### 8.5 COO ↔ CSR ↔ CSC

```python
def coo_to_csr(row, col, data, n):
    """COO to CSR via sorting. O(m log m)."""
    # Sort by (row, col)
    order = np.lexsort((col, row))
    row_sorted, col_sorted = row[order], col[order]
    data_sorted = data[order]
    # Build row_ptr via counting sort
    row_counts = np.bincount(row_sorted, minlength=n)
    row_ptr = np.zeros(n + 1, dtype=int)
    np.cumsum(row_counts, out=row_ptr[1:])
    return data_sorted, col_sorted, row_ptr

# Using scipy (recommended in practice):
A_csr = sp.csr_matrix((data, (row, col)), shape=(n, n))
A_csc = A_csr.tocsc()   # O(m) transpose
A_coo = A_csr.tocoo()   # O(m) expansion
```

### 8.6 PyG ↔ NetworkX ↔ SciPy

```python
# NetworkX → PyG edge_index
import networkx as nx
import torch

def nx_to_edge_index(G):
    edges = list(G.edges())
    if not G.is_directed():
        # Add reverse edges for undirected
        edges += [(v, u) for u, v in edges]
    return torch.tensor(edges, dtype=torch.long).t().contiguous()

# PyG edge_index → NetworkX
def edge_index_to_nx(edge_index, directed=True):
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_edges_from(edge_index.t().tolist())
    return G

# SciPy CSR → PyG edge_index
def csr_to_edge_index(A_csr):
    coo = A_csr.tocoo()
    return torch.tensor([coo.row, coo.col], dtype=torch.long)

# PyG edge_index → SciPy CSR
def edge_index_to_csr(edge_index, n):
    row, col = edge_index[0].numpy(), edge_index[1].numpy()
    data = np.ones(len(row))
    return sp.csr_matrix((data, (row, col)), shape=(n, n))
```

---

## 9. Heterogeneous and Dynamic Graphs

### 9.1 Heterogeneous Graphs

A **heterogeneous graph** (also called *hetero-graph* or *knowledge graph*) has multiple types of nodes and/or edges:

$$G = (V, E, \phi, \psi)$$

where $\phi: V \to \mathcal{T}_V$ assigns each vertex a **node type** and $\psi: E \to \mathcal{T}_E$ assigns each edge a **relation type**.

**Example — academic knowledge graph:**
- Node types: Author, Paper, Venue
- Edge types: writes (Author→Paper), published_in (Paper→Venue), cites (Paper→Paper), co-author (Author↔Author)

For each relation type $(s, r, t)$ (source type, relation, target type), we maintain a **separate adjacency matrix** or `edge_index` tensor:

$$A^{(s,r,t)} \in \{0,1\}^{|V_s| \times |V_t|}$$

With $|\mathcal{T}_V|$ node types and $|\mathcal{T}_E|$ relation types, we have at most $|\mathcal{T}_E|$ matrices — typically much smaller than the full graph.

**Knowledge graph triple format.** Knowledge graphs (Freebase, Wikidata, OGB-MAG) store relations as triples $(h, r, t)$ — head entity, relation, tail entity. This is an edge list over a heterogeneous graph:

```python
triples = [
    ("Albert_Einstein", "nationality", "Germany"),
    ("Theory_of_Relativity", "author", "Albert_Einstein"),
    ...
]
```

### 9.2 PyG HeteroData

PyG's `HeteroData` object generalises `Data` to heterogeneous graphs:

```python
from torch_geometric.data import HeteroData

data = HeteroData()

# Node feature matrices per type
data['author'].x = torch.randn(n_authors, 64)
data['paper'].x  = torch.randn(n_papers,  128)

# Edge indices and features per relation type
data['author', 'writes', 'paper'].edge_index  = ...  # (2, m_writes)
data['paper',  'cites',  'paper'].edge_index  = ...  # (2, m_cites)
data['paper',  'cites',  'paper'].edge_attr   = ...  # (m_cites, d_e)

print(data.node_types)   # ['author', 'paper']
print(data.edge_types)   # [('author','writes','paper'), ('paper','cites','paper')]
```

Heterogeneous message passing (RGCN, HAN, HGT) iterates over relation types and aggregates messages per relation:

$$\mathbf{h}_v' = \text{AGG}\!\left(\left\{\mathbf{W}_r \mathbf{h}_u : u \in \mathcal{N}_r(v)\right\}_{r \in \mathcal{T}_E}\right)$$

### 9.3 Dynamic Graphs

A **dynamic graph** changes over time: edges and vertices are inserted or deleted. The primary representation challenge is supporting updates without full reconstruction.

**Adjacency list** handles edge insertion in $O(1)$ and deletion in $O(\deg(u))$. It is the standard choice for dynamic graph algorithms.

**CSR is static**: every insertion or deletion requires rebuilding the three arrays — $O(m)$ total. For high-frequency updates, CSR is inappropriate.

**Dynamic CSR alternatives:**
- **Chunked CSR:** Divide the adjacency list into fixed-size chunks; insertions overflow into a "delta" structure
- **VCSR (Vertex-Centric CSR):** Each row is a separate resizeable array; updates are local
- **Sorted adjacency list:** Maintain sorted neighbours per vertex for $O(\log \deg)$ edge lookup

### 9.4 Temporal Graphs and Snapshots

A **temporal graph** $G = (V, E, T)$ has edges with timestamps $t_e \in T$:

$$E \subseteq V \times V \times T$$

The most common representation is a **snapshot list**: $\{G_0, G_1, \ldots, G_T\}$ where $G_t$ is the graph at time step $t$. Each snapshot can be stored as a separate `Data` object:

```python
snapshots = [
    Data(x=x_t, edge_index=edge_index_t)
    for t, (x_t, edge_index_t) in enumerate(time_series)
]
```

**For AI — temporal GNNs:**
- **TGAT (Temporal Graph Attention, 2020):** Stores interactions as chronologically sorted edge lists with timestamp features; uses time-encoding functions as positional features
- **DyRep (2019):** Maintains a dynamic adjacency list updated via temporal point processes
- **TGN (Temporal Graph Networks, 2020):** Uses memory modules per node to summarise temporal history; edge list sorted by timestamp is the core data structure

The temporal edge list $(u, v, t, \text{feat})$ sorted by $t$ is the standard input format for temporal GNN benchmarks (JODIE, REDDIT datasets in OGB-Temporal).

### 9.5 Bipartite Graph Representations

Many real-world graphs are **bipartite**: vertices are split into two disjoint sets $U$ and $V$ with edges only between $U$ and $V$ — no edges within $U$ or within $V$.

**Examples:** User-item interactions (recommendation systems), author-paper graphs, word-document matrices, student-course enrollment.

Bipartite graphs require **rectangular adjacency matrices** $A \in \{0,1\}^{|U| \times |V|}$ rather than square $n \times n$ matrices. This changes the representation:

| Representation | Bipartite form |
|---------------|---------------|
| Adjacency matrix | $A \in \{0,1\}^{|U| \times |V|}$ (rectangular, not square) |
| Adjacency list | `adj_u[u]` lists vertices in $V$; `adj_v[v]` lists vertices in $U$ |
| Edge list | $(u, v)$ pairs with $u \in U$, $v \in V$ |
| COO | `edge_index` with source indices in $[0, |U|)$, target in $[0, |V|)$ |

**In PyG**, bipartite graphs use a 2-tuple node feature convention:

```python
data = Data(
    x=(x_u, x_v),          # node features for U and V separately
    edge_index=edge_index,  # source ∈ [0,|U|), target ∈ [0,|V|)
)
```

**For AI — collaborative filtering.** The matrix factorization model for recommendation can be viewed as operating on a bipartite user-item graph with adjacency $A \in \{0,1\}^{n_u \times n_i}$:

$$\hat{R}_{ui} = \mathbf{p}_u^\top \mathbf{q}_i$$

where $\mathbf{p}_u$ and $\mathbf{q}_i$ are learned embeddings. LightGCN (He et al., 2020) applies GCN layers directly to this bipartite graph, propagating user embeddings through item nodes and vice versa. The core operation is two rectangular SpMM passes: $A \mathbf{Q}$ (aggregate items → users) and $A^\top \mathbf{P}$ (aggregate users → items).

### 9.6 Multiplex and Signed Graphs

**Multiplex graphs** have multiple layers of edges between the same vertex set — different edge types exist in parallel:

$$G = (V, E_1, E_2, \ldots, E_k)$$

Each layer $E_r$ is a separate adjacency matrix $A^{(r)}$. The representation is a *tensor* $\mathcal{A} \in \{0,1\}^{n \times n \times k}$ (a stack of adjacency matrices).

**Applications:** Social networks with multiple relation types (follows, likes, messages), temporal layers (contacts at different time windows), transportation networks (road, rail, air in the same city).

**Signed graphs** have edges with positive or negative weights — representing trust/distrust, attraction/repulsion, or cooperation/competition. The adjacency matrix $A$ has entries in $\{-1, 0, +1\}$ (or $\mathbb{R}$ for weighted signed graphs). Signed graph Laplacians have non-standard spectral properties; they are studied in signed spectral graph theory.

**For AI:** Signed graphs arise in:
- **Sentiment analysis:** Stance graphs where edges are agree/disagree
- **Financial networks:** Positive = correlated assets, negative = anti-correlated
- **Recommendation:** Explicit dislikes as negative edges in collaborative filtering
- **RL reward shaping:** Signed reward graphs encode preference relations

---

## 10. Choosing a Representation

### 10.1 Decision Framework

```
REPRESENTATION SELECTION FLOWCHART
════════════════════════════════════════════════════════════════════════

                      START
                        │
              Is n ≤ 10,000 AND dense?
              ┌─────────┴──────────┐
             YES                  NO
              │                   │
    Adjacency Matrix    Is graph dynamic (frequent updates)?
                        ┌─────────┴──────────┐
                       YES                  NO
                        │                   │
              Adjacency List      Target GPU / PyG?
                        ┌─────────┴──────────┐
                       YES                  NO
                        │                   │
                    edge_index          Need SpMV / eigenvalues?
                    (COO/PyG)           ┌─────────┴──────────┐
                                       YES                  NO
                                        │                   │
                                      CSR            Adj. List or
                                (scipy.sparse)       Edge List

════════════════════════════════════════════════════════════════════════
```

### 10.2 By Graph Size and Density

| Graph Size | Density $\rho$ | Recommended | Rationale |
|-----------|---------------|-------------|-----------|
| $n \leq 10^3$ | Any | Adjacency matrix | Simplest; $O(n^2) \leq 10^6$ — fine |
| $n \leq 10^4$, dense | $\rho > 0.1$ | Adjacency matrix | Dense matmul is GPU-efficient |
| $n \leq 10^6$, sparse | $m \leq 10n$ | CSR or adj. list | $O(n)$ memory |
| $n > 10^6$ | Any | CSR + edge list | Billion-edge graphs need streaming |
| Batch of small graphs | $n_i \leq 100$ | Dense padded tensors or PyG batch | GPU efficiency via padding |

### 10.3 By Algorithm Type

| Algorithm | Best Representation | Reason |
|-----------|-------------------|--------|
| BFS / DFS | Adjacency list | Sequential neighbour access |
| Dijkstra / Bellman-Ford | Adjacency list (sorted) | Priority queue needs fast neighbours |
| PageRank / power iteration | CSR | Repeated SpMV |
| Spectral clustering | CSR → `eigsh` | Laplacian eigenvectors via ARPACK |
| GCN / MPNN | COO / edge_index | GPU scatter operations |
| Link prediction | Adjacency set | Fast edge existence queries |
| Subgraph sampling | CSR | Row-slice for $k$-hop neighbourhood |
| Shortest path (all pairs) | Adjacency matrix | Floyd-Warshall uses matrix |

### 10.4 By Hardware Target

**CPU (single node):**
- SciPy CSR for numerical work
- NetworkX adjacency dict for prototyping
- Numba-JIT over CSR arrays for performance

**GPU (single card):**
- PyG `edge_index` + cuSPARSE SpMM
- For very large graphs: PyG with `NeighborSampler` (CSR-based sampling)
- DGL for production GNN training (CSR internally, CUDA-optimised)

**Multi-GPU / distributed:**
- Partitioned edge lists (METIS graph partitioning)
- DistDGL, GraphLearn, or AliGraph for trillion-edge graphs
- Edge list files partitioned on HDFS / S3

**TPU:**
- Padded dense adjacency matrices (fixed shape required for XLA compilation)
- Jraph (JAX graph library) uses padded dense representation

### 10.5 By Framework

| Framework | Primary format | Notes |
|-----------|---------------|-------|
| NetworkX | `dict` of `dict` (adj. dict) | Pure Python; slow but feature-rich |
| PyTorch Geometric | `edge_index` (COO) | Standard for GNN research |
| DGL | CSR `DGLGraph` | Optimised for large heterogeneous graphs |
| SciPy / NumPy | CSR `csr_matrix` | Numerical computing; integrates with ARPACK |
| TensorFlow GNN | `GraphTensor` (COO-based) | TF-native graph representation |
| Jraph (JAX) | Padded dense `GraphsTuple` | TPU-optimised |
| SNAP (Stanford) | Binary edge lists | Ultra-large graphs ($> 10^9$ edges) |

### 10.6 Case Study: Representing the OGB-Arxiv Graph

The OGB-Arxiv dataset (Open Graph Benchmark) is a directed citation network: $n = 169{,}343$ nodes (papers), $m = 1{,}166{,}243$ edges (citations). Let us trace the representation choices made by three frameworks:

**NetworkX (baseline, not recommended for training):**
```python
G = nx.DiGraph()
G.add_edges_from(edge_list)
# Memory: ~800 MB (Python dict overhead)
# Time to iterate all edges: ~12 s
```

**PyTorch Geometric (research standard):**
```python
dataset = NodePropPredDataset('ogbn-arxiv')
data = dataset[0]
# data.edge_index: shape (2, 2_332_486) = bidirectional edges
# data.x: shape (169_343, 128) = feature matrix
# Memory: ~37 MB for edge_index (int64)
# Training time (3-layer GCN, epoch): ~0.4 s on A100
```

**DGL (production):**
```python
g = dgl.graph((src, dst), num_nodes=n)
# Internal: CSR for forward pass, CSC for backward pass
# Memory: ~20 MB (int32 indices)
# Training time (3-layer GCN, epoch): ~0.3 s on A100 (faster due to CSR)
```

The choice between PyG COO and DGL CSR is a 25% throughput difference at this scale — a significant factor in research iteration speed.

---

## 11. Preview: Algorithms and Spectral Methods

### 11.1 How §03 Graph Algorithms Uses These Representations

Graph algorithms depend critically on the choice of representation. The asymptotic complexities in §03 are derived assuming specific representations:

| Algorithm | Representation | Complexity |
|-----------|---------------|-----------|
| BFS | Adjacency list | $O(n + m)$ |
| Dijkstra (binary heap) | Adjacency list | $O((n + m) \log n)$ |
| Bellman-Ford | Edge list | $O(nm)$ |
| Kruskal MST | Edge list (sorted) | $O(m \log m)$ |
| Prim MST | Adjacency list + heap | $O((n + m) \log n)$ |
| Floyd-Warshall | Adjacency matrix | $O(n^3)$ |
| DFS-based SCC | Adjacency list | $O(n + m)$ |

Using the wrong representation can change asymptotic complexity: Dijkstra with an adjacency matrix runs in $O(n^2)$ instead of $O((n+m)\log n)$ — a factor of $n/\log n$ slower for sparse graphs. Full algorithmic treatment: [§03 Graph Algorithms](../03-Graph-Algorithms/notes.md).

### 11.2 How §04 Spectral Theory Uses the Laplacian

The Laplacian $L = D - A$ defined in §3.4 is the central object of spectral graph theory. Its eigendecomposition $L = Q \Lambda Q^\top$ reveals:

- $\lambda_1 = 0$ always (graph has at least one component)
- Number of $\lambda_i = 0$: number of connected components
- $\lambda_2 > 0$ (Fiedler value): measures connectivity strength
- Eigenvector $\mathbf{q}_2$ (Fiedler vector): partitions the graph for spectral clustering

Computing the Fiedler vector requires `eigsh(L, k=2, which='SM')` — a sparse eigensolver that takes CSR format as input. The representation choice (CSR) directly enables the spectral analysis.

Full spectral treatment: [§04 Spectral Graph Theory](../04-Spectral-Graph-Theory/notes.md).

### 11.3 How §05 GNNs Use edge_index and Sparse Adjacency

The three core GNN architectures use the representations from this section differently:

| Architecture | Representation Used | Core Operation |
|-------------|--------------------|----|
| GCN (Kipf & Welling, 2017) | Sparse $\hat{A}$ (CSR or COO) | $H' = \sigma(\hat{A} H W)$ — SpMM |
| GraphSAGE (Hamilton et al., 2017) | Adjacency list (neighbour sampling) | Sample $k$ neighbours per node |
| GAT (Veličković et al., 2018) | `edge_index` (COO) | Attention per edge: $\alpha_{ij}$ computed and aggregated |

Full GNN architectures, expressiveness theory, and training details: [§05 Graph Neural Networks](../05-Graph-Neural-Networks/notes.md).

---

## 12. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
|---|---------|---------------|-----|
| 1 | Using a dense adjacency matrix for a million-node graph | $O(n^2)$ space ≈ $10^{12}$ bytes for $n=10^6$ — physically impossible | Use CSR or `edge_index`; dense $A$ is only viable for $n \leq 10^4$ |
| 2 | Forgetting that `edge_index` is directed by default in PyG | Undirected edges require both $(u,v)$ and $(v,u)$ to appear; missing the reverse direction causes asymmetric message passing | Use `torch_geometric.utils.to_undirected(edge_index)` or add reverse edges explicitly |
| 3 | Using $A^\top$ (transpose notation) instead of $A^\top$ (`\top`) for the transpose | Cosmetic but matters: in NOTATION_GUIDE, transpose is `^\top`, not `^T` — "$T$" looks like a variable name | Write $A^\top$ consistently |
| 4 | Assuming the Laplacian is $A - D$ instead of $D - A$ | The sign matters: $L = D - A$ is positive semidefinite; $A - D$ is negative semidefinite. GCN derivations assume $L \succeq 0$ | Always write $L = D - A$ and verify $\mathbf{x}^\top L \mathbf{x} \geq 0$ |
| 5 | Confusing CSR `row_ptr` length with $n$ (instead of $n+1$) | `row_ptr` has $n+1$ entries: `row_ptr[0] = 0`, `row_ptr[n] = m`. Indexing `row_ptr[n]` out-of-bounds crashes code | Always allocate `row_ptr = np.zeros(n+1)`; the extra entry is essential |
| 6 | Adding self-loops before normalising in GCN | GCN normalisation is $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ where $\tilde{A} = A + I$. If you add self-loops to $A$ first and then normalise with the original $D$, degrees are wrong | Apply self-loops and recompute degree before normalising |
| 7 | Using adjacency matrix eigendecomposition when the graph is sparse and large | `np.linalg.eig(A)` is $O(n^3)$ — infeasible for $n > 10^4$ | Use `scipy.sparse.linalg.eigsh` with ARPACK for sparse matrices; it computes only $k$ eigenvalues in $O(k \cdot m)$ |
| 8 | Treating the incidence matrix as $n \times n$ instead of $n \times m$ | The incidence matrix has $n$ rows (vertices) and $m$ columns (edges). Confusing rows and columns leads to wrong shapes in $L = BB^\top$ | Verify: `B.shape == (n, m)`; `B @ B.T` gives $L \in \mathbb{R}^{n \times n}$ |
| 9 | Storing undirected graphs as directed without both directions | If `adj[u]` contains `v` but `adj[v]` doesn't contain `u`, BFS/DFS will give wrong results and message passing will be asymmetric | Always add both `adj[u].append(v)` and `adj[v].append(u)` for undirected graphs |
| 10 | Using COO format for large repeated SpMV | COO lacks row-pointer structure, so each SpMV must sort entries — $O(m \log m)$ per call. For PageRank or GNN training (many iterations), this is catastrophic | Convert COO to CSR once before the iteration loop; repeated SpMV on CSR is $O(m)$ per call |

---

## 13. Exercises

**Exercise 1 ★ — Manual CSR Construction**

Given the path graph $P_5$ with vertices $\{0,1,2,3,4\}$ and edges $\{0,1\}, \{1,2\}, \{2,3\}, \{3,4\}$:

(a) Write out the adjacency matrix $A$.
(b) Manually construct the CSR arrays `data`, `col_idx`, `row_ptr`.
(c) Verify: `data[row_ptr[2]:row_ptr[3]]` gives the non-zeros in row 2.
(d) Compute the degree of each vertex from `row_ptr`.

---

**Exercise 2 ★ — Space Complexity Comparison**

For a graph with $n = 10^5$ vertices and $m = 3 \times 10^5$ edges:

(a) Compute the exact number of bytes required for a dense float32 adjacency matrix.
(b) Compute the bytes for CSR representation (int32 indices, float32 data).
(c) Compute the bytes for `edge_index` (int64 COO).
(d) What fill ratio $\rho$ would make the dense matrix and CSR consume equal memory?

---

**Exercise 3 ★ — Laplacian Derivation**

For the star graph $S_4$ (centre vertex 0, leaves 1, 2, 3, 4):

(a) Write the adjacency matrix $A$ and degree matrix $D$.
(b) Compute $L = D - A$.
(c) Verify $L \mathbf{1} = \mathbf{0}$.
(d) Compute the quadratic form $\mathbf{x}^\top L \mathbf{x}$ for $\mathbf{x} = (1, -1, -1, -1, -1)^\top / 2$. What does this value measure?

---

**Exercise 4 ★★ — Conversion Pipeline**

Implement the full conversion pipeline:

(a) Start with edge list `[(0,1),(0,2),(1,3),(2,3),(3,4)]`, $n = 5$.
(b) Convert to adjacency list (Python dict).
(c) Convert adjacency list to COO arrays.
(d) Convert COO to CSR manually (without scipy).
(e) Implement SpMV using your CSR arrays; verify against `A @ x` for $\mathbf{x} = \mathbf{1}$ (should return degree sequence).

---

**Exercise 5 ★★ — Normalised Adjacency**

For the cycle graph $C_5$:

(a) Construct $A$ (adjacency matrix) and $D$ (degree matrix).
(b) Compute the GCN normalised adjacency: $\hat{A} = D^{-1/2} A D^{-1/2}$.
(c) Compute the augmented version: $\tilde{A} = A + I$, $\tilde{D} = D + I$, $\hat{\tilde{A}} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$.
(d) Verify that the eigenvalues of $\hat{A}$ lie in $[-1, 1]$ and the eigenvalues of $L_{\text{sym}} = I - \hat{A}$ lie in $[0, 2]$.
(e) Explain: why does the normalised Laplacian's eigenvalue bound $[0, 2]$ matter for GNN stability?

---

**Exercise 6 ★★ — Incidence Matrix and Laplacian**

For the triangle graph $K_3$:

(a) Construct the incidence matrix $B \in \{0,1\}^{3 \times 3}$.
(b) Compute $B B^\top$ and verify it equals $L = D - A$.
(c) Construct the directed incidence matrix $B_{\text{dir}}$ (orient edges $0\to1$, $1\to2$, $0\to2$).
(d) Verify $B_{\text{dir}} B_{\text{dir}}^\top = L$ still holds.
(e) Interpret $(B_{\text{dir}}^\top \mathbf{x})_e$ for edge $e = (0 \to 1)$: what is $x_1 - x_0$ measuring geometrically on the graph?

---

**Exercise 7 ★★★ — PyG Data Object from Scratch**

Without using PyG's built-in loaders, construct a `torch_geometric.data.Data` object from scratch for the karate club graph:

(a) Load the karate club edge list from `networkx.karate_club_graph()`.
(b) Build `edge_index` (COO, bidirectional, shape $2 \times 2m$) without using `from_networkx`.
(c) Build `x` as a one-hot encoding of each vertex's community (use `G.nodes[v]['club']`).
(d) Compute `edge_attr` as the normalised weight $w_{ij}/\max_{e} w_e$ for each edge (karate club edges have weights from `G[u][v]['weight']`).
(e) Verify: `data.num_nodes == 34`, `data.num_edges == 2*78 == 156`, `data.is_undirected() == True`.

---

**Exercise 8 ★★★ — Heterogeneous Graph Representation**

Consider an academic graph with:
- 100 authors, 200 papers, 20 venues
- Relations: `(author, writes, paper)` with 400 edges; `(paper, published_in, venue)` with 200 edges; `(paper, cites, paper)` with 800 edges

(a) Design the heterogeneous adjacency representation: how many separate `edge_index` tensors? What are their shapes?
(b) Compute the total memory for all `edge_index` tensors (int64, COO bidirectional where appropriate).
(c) Implement a simple heterogeneous degree function: for each author, count the number of papers they wrote.
(d) Write the RGCN-style aggregation for author nodes in pseudocode:
$$\mathbf{h}_a^{(l+1)} = \sigma\!\left(\mathbf{W}_{\text{writes}} \cdot \frac{1}{|\mathcal{P}_a|}\sum_{p \in \mathcal{P}_a} \mathbf{h}_p^{(l)}\right)$$
where $\mathcal{P}_a$ is the set of papers written by author $a$.
(e) Implement this aggregation using `torch.scatter_mean` on the `(author, writes, paper)` `edge_index`.

---

## 14. Why This Matters for AI (2026 Perspective)

| Concept | AI Impact |
|---------|----------|
| Adjacency matrix $A$ | Defines the GCN propagation rule $\hat{A}XW$; walk powers $A^k$ count paths used in subgraph GNNs; $A$'s eigenspectrum is the basis of spectral GNNs |
| Normalised adjacency $\hat{A}$ | Core of GCN (Kipf & Welling, 2017): controls feature smoothing across neighbourhoods; symmetric normalisation prevents gradient explosion in deep GNNs |
| Laplacian $L = D - A$ | Defines graph signal "frequency"; low-frequency signals (smooth) are aligned with small eigenvalues; over-smoothing in deep GNNs corresponds to projecting onto the null space of $L$ |
| CSR / CSC formats | Enable $O(m)$ SpMM in all production GNN frameworks; DGL's CSR-based `DGLGraph` is the internal representation behind thousands of GNN papers |
| PyG `edge_index` | The de-facto standard for GNN research (2017–present); shapes all of PyG's 80+ model implementations; enables batching, sampling, and differentiable graph operations |
| Heterogeneous adjacency | RGCN, HAN, HGT, and knowledge graph embedding models (TransE, RotatE) all rely on per-relation adjacency; the 2024 OGB knowledge graph leaderboard is dominated by heterogeneous GNNs |
| Temporal edge lists | TGN, CAWN, and GraphMixer use chronologically sorted edge lists; the OGB-Temporal benchmark measures dynamic link prediction on graphs with $10^7$ timed interactions |
| Hypergraph incidence | HOGNNs (Higher-Order GNNs) operate on $k$-tuples stored as hyperedge incidence matrices; all $k$-WL expressiveness results are stated in terms of hypergraph structure |
| Padded dense adjacency | JAX/Jraph for TPU requires fixed shapes; padded dense $A$ enables XLA compilation; models like EigenGNN and SAN use full self-attention (implicit dense $A$) for small molecular graphs |
| Sparse attention patterns | FlashAttention uses a block-sparse adjacency structure for efficient long-sequence attention; the "local window" and "strided" patterns are special cases of sparse `edge_index` |
| Bipartite adjacency $A \in \mathbb{R}^{|U|\times|V|}$ | LightGCN, NGCF, and matrix factorisation all operate on rectangular bipartite adjacency; the two SpMM passes $A\mathbf{Q}$ and $A^\top\mathbf{P}$ define collaborative filtering as GNN layers |
| Signed adjacency | Sentiment analysis (agree/disagree), financial correlation, and social balance theory all require signed graph representations; signed Laplacians have distinct spectral properties |
| Dynamic edge lists | Streaming graph frameworks (Flink-based TGN, Kafka-based dynamic GNNs) ingest temporal edge lists in real time; the edge list with timestamps is the universal streaming format |

### 14.1 Representation as a Research Lever

The history of GNN research shows that representation choices have directly enabled new research directions:

- **2017:** `edge_index` (COO) in PyG enabled arbitrary graph structures — not just grid or sequence graphs — to be processed in a single framework. This unlocked the explosion of GNN variants.
- **2019:** DGL's CSR-based `DGLGraph` with heterogeneous support enabled the RGCN, HAN, and HGT architectures for knowledge graphs.
- **2020:** The OGB benchmark's standardised edge list format enabled reproducible comparison across GNN architectures for the first time.
- **2021:** Padded dense adjacency in Jraph/JAX enabled TPU-based GNN training, 10–40× faster than GPU for certain molecular property prediction tasks.
- **2022:** Block-sparse attention patterns (BigBird, LongFormer) showed that limiting the attention graph to a sparse `edge_index` (local window + global tokens) reduces transformer complexity from $O(n^2)$ to $O(n)$.
- **2023–2024:** Graph transformers (GPS, NAGphormer, NodeFormer) use hybrid representations: sparse `edge_index` for local message passing + full dense adjacency for global attention, combining the strengths of both.

The lesson: choosing, designing, or combining representations is itself a research contribution — not just an implementation detail.

---

## 15. Conceptual Bridge

### Looking Back

This section translates the abstract graph theory of §01 into concrete data structures. Every object defined in §01 — adjacency, degree, Laplacian — now has a precise computational form:

- **Adjacency** (§01, Definition 2.1) → **Adjacency matrix** $A$ (§3.1): the mathematical definition becomes a 2D array
- **Degree** (§01, §3.1) → **Row sums of $A$** or `row_ptr[i+1] - row_ptr[i]` in CSR
- **Handshaking lemma** (§01, §3.2): $\sum \deg = 2m$ → `data.sum() == 2 * num_edges` for CSR
- **Graph Laplacian** (§01, preview) → **Full definition** $L = D - A$ (§3.4) with quadratic form, PSD proof, and connectivity interpretation

The connection to linear algebra (Ch. 02–03) is now explicit: every graph operation is a matrix operation, and the choice of sparse format determines whether that operation is feasible.

The section also establishes two "bridge" matrices that will recur throughout the rest of the chapter:
1. **The Laplacian $L = D - A$** — defined here, eigendecomposed in §04, used in GCN derivation in §05
2. **The normalised adjacency $\hat{A} = D^{-1/2} A D^{-1/2}$** — defined here, proved to be a spectral filter in §04, implemented as the GCN propagation rule in §05

Mastering these two matrices — their construction, their properties, and their computational representations — is the central payoff of this section.

### Looking Forward

The representations defined here are the input to every algorithm in the rest of the chapter:

- **§03 Graph Algorithms** implements BFS, DFS, Dijkstra, Kruskal, and topological sort *on* these representations. The choice of adjacency list vs. matrix vs. CSR directly determines whether algorithms run in $O(n+m)$ or $O(n^2)$.
- **§04 Spectral Graph Theory** eigendecomposes the Laplacian $L = D - A$ defined in §3.4. The `scipy.sparse.linalg.eigsh` interface takes CSR format as input, connecting the sparse representation to spectral analysis.
- **§05 Graph Neural Networks** uses `edge_index` (COO) for PyG-based GNN implementations. Every GCN, GAT, and GraphSAGE layer operates on the normalised adjacency $\hat{A}$ (§3.5) or directly on `edge_index` via scatter operations.
- **§06 Random Graphs** generates graphs via edge lists; analysing their degree distributions and connectivity requires converting to adjacency lists or CSR for efficient computation.

### The Big Picture

```
GRAPH REPRESENTATIONS IN THE CURRICULUM
════════════════════════════════════════════════════════════════════════

  §01 Graph Basics          §02 Graph Representations
  ─────────────────         ──────────────────────────
  Abstract G=(V,E)    ──→   Concrete data structures
  Adjacency concept   ──→   Adjacency matrix A, CSR, COO
  Degree definition   ──→   Row sums, row_ptr differences
  Laplacian preview   ──→   L = D - A (full definition)
                            Normalised Â = D^{-1/2}AD^{-1/2}

                    ┌──────────────────────────────────────┐
                    │         §02: THE BRIDGE              │
                    │  Abstract ──→ Computational          │
                    │  Theory   ──→ Implementation         │
                    │  Proofs   ──→ Algorithms             │
                    └──────┬────────────────┬─────────────┘
                           │                │
                    §03 Algorithms    §04 Spectral Theory
                    (use adj. list,   (use L, eigenvectors,
                     edge list, CSR)   CSR + ARPACK)
                           │                │
                           └──────┬─────────┘
                                  ▼
                           §05 Graph Neural Networks
                           (edge_index, Â, message passing)
                                  ▼
                           §06 Random Graphs
                           (generate edge lists, analyse
                            degree distributions)

════════════════════════════════════════════════════════════════════════
```

---

## Appendix A: Notation Summary

### A.1 Graph and Matrix Symbols

The following notation is used consistently throughout this section, following the conventions of `docs/NOTATION_GUIDE.md`.

| Symbol | Meaning | Section |
|--------|---------|---------|
| $G = (V, E)$ | Graph with vertex set $V$, edge set $E$ | §2.1 |
| $n = \lvert V \rvert$ | Number of vertices | §2.2 |
| $m = \lvert E \rvert$ | Number of edges | §2.2 |
| $A \in \{0,1\}^{n \times n}$ | Adjacency matrix | §3.1 |
| $D = \operatorname{diag}(A\mathbf{1})$ | Degree matrix | §3.4 |
| $L = D - A$ | Graph Laplacian | §3.4 |
| $\hat{A} = D^{-1/2} A D^{-1/2}$ | Symmetrically normalised adjacency | §3.5 |
| $\tilde{A} = A + I$ | Adjacency with self-loops | §3.3 |
| $P = D^{-1} A$ | Random walk transition matrix | §3.5 |
| $B \in \{0,1\}^{n \times m}$ | Incidence matrix | §7.1 |
| $\rho = 2m / n(n-1)$ | Fill ratio (sparsity measure) | §2.3 |
| `edge_index` $\in \mathbb{Z}^{2 \times m}$ | PyG COO edge tensor | §5.3 |

---

## Appendix B: Complexity Reference

### B.1 Per-Operation Complexity

The following table consolidates the per-operation complexity for all six core representations. Assume $n$ vertices, $m$ edges, maximum degree $\Delta$.

**Notation:** "avg" = average case for hash-based structures; "amort" = amortised over a sequence of operations.

| Representation | Space | Edge? | Neighbours | SpMV | Build |
|---------------|-------|-------|-----------|------|-------|
| Adjacency matrix | $O(n^2)$ | $O(1)$ | $O(n)$ | $O(n^2)$ | $O(n^2 + m)$ |
| Adjacency list | $O(n+m)$ | $O(\deg)$ | $O(\deg)$ | $O(m)$ | $O(n+m)$ |
| Adjacency set | $O(n+m)$ | $O(1)$ avg | $O(\deg)$ | $O(m)$ | $O(n+m)$ |
| Edge list | $O(m)$ | $O(m)$ | $O(m)$ | $O(m \log m)$ | $O(m)$ |
| COO | $O(m)$ | $O(m)$ | $O(m)$ | $O(m \log m)$ | $O(m)$ |
| CSR | $O(n+m)$ | $O(\deg)$ | $O(\deg)$ | $O(m)$ | $O(m \log m)$ |
| CSC | $O(n+m)$ | $O(\deg)$ | $O(\deg)$ | $O(m)$ | $O(m \log m)$ |
| Incidence | $O(nm)$ | $O(\deg)$ | $O(\deg)$ | — | $O(nm)$ |

---

[← Back to Graph Theory](../README.md) | [Previous: Graph Basics ←](../01-Graph-Basics/notes.md) | [Next: Graph Algorithms →](../03-Graph-Algorithms/notes.md)

<!-- End of §02 main content. See theory.ipynb for executable demonstrations and exercises.ipynb for graded practice. -->

### B.2 Key Break-Even Points

| Comparison | Break-even condition | Prefer dense when | Prefer sparse when |
|-----------|---------------------|-------------------|-------------------|
| Dense $A$ vs CSR space | $\rho = 1/8$ (int32 CSR) | $\rho > 1/8$ | $\rho < 1/8$ |
| Dense $A$ vs CSR SpMV | $\rho = 1$ (always equal ops) | Dense has better constants | Sparse wins for $\rho < 0.5$ |
| Adj. list vs CSR | Never (CSR always faster) | Prototyping | Production code |
| COO vs CSR for SpMV | Single call | COO is simpler to build | CSR is faster for repeat calls |
| Adj. list vs Adj. set | Edge query frequency | Mostly traversal | Frequent edge queries |

### B.3 Conversion Cost Summary

All conversions assume the source representation is already built. The $O(m \log m)$ cost of COO → CSR comes from the sort step (can be reduced to $O(m + n)$ with counting sort when vertex IDs are bounded integers).

| Pipeline | Total cost | Bottleneck |
|----------|-----------|-----------|
| File → edge list → COO | $O(m)$ | File I/O |
| File → edge list → CSR | $O(m \log m)$ | Sort by row |
| CSR → PyG edge_index | $O(m)$ | Expand row_ptr |
| NetworkX → PyG | $O(n + m)$ | Python dict iteration |
| PyG → NetworkX | $O(n + m)$ | Edge list construction |
| Dense $A$ → CSR | $O(n^2)$ | Dense scan |
| CSR → Dense $A$ | $O(n^2 + m)$ | Matrix fill |

For large graphs ($m > 10^7$), the COO → CSR sort is the dominant cost. Use radix sort (available in CUDA via `thrust::sort_by_key`) for GPU-accelerated sorting in $O(m)$ expected time.


### B.4 Memory Footprint Formulas

For a graph with $n$ vertices and $m$ edges using 32-bit integers and 32-bit floats:

| Format | Memory (bytes) | Example: $n=10^5$, $m=5\times10^5$ |
|--------|---------------|-------------------------------------|
| Dense float32 | $4n^2$ | 40 GB |
| Dense bool | $n^2/8$ | 1.25 GB |
| Adjacency list (Python) | $\approx 80m$ | 40 MB |
| CSR (int32, unweighted) | $4(n+1+m)$ | 2.0 MB |
| CSR (int32 + float32 weights) | $4(n+1+2m)$ | 6.0 MB |
| COO (int64 edge_index) | $16m$ | 8.0 MB |
| LIL (Python, sorted) | $\approx 100m$ | 50 MB |

For reference: 1 GB = $10^9$ bytes. A graph with $n = 10^6$ and $m = 5 \times 10^6$ (typical social network) requires:
- Dense: 4 TB (impossible on a single machine)
- CSR int32: ~24 MB (fits in L3 cache of a modern server)

This 170,000× difference explains why every large-scale graph ML system uses sparse representations exclusively.

---

*End of §02 Graph Representations. Proceed to [§03 Graph Algorithms](../03-Graph-Algorithms/notes.md) to see these representations in action.*

<!-- ─────────────────────────────────────────────────────────────────────── -->
<!-- Companion files: theory.ipynb (executable demos), exercises.ipynb (8 exercises) -->
<!-- Next: ../03-Graph-Algorithms/notes.md -->
<!-- ─────────────────────────────────────────────────────────────────────── -->




















