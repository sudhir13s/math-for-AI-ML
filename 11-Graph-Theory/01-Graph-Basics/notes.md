[← Back to Graph Theory](../README.md) | [Next: Graph Representations →](../02-Graph-Representations/notes.md)

---

# Graph Basics

> _"A graph is the mathematician's way of saying: I don't care what the objects are — I care how they're connected. This single abstraction unifies social networks, molecular structures, knowledge bases, parse trees, and the computation graphs inside every neural network."_

## Overview

A graph is a mathematical structure that captures pairwise relationships between objects. Formally, a graph $G = (V, E)$ consists of a set of vertices (or nodes) $V$ and a set of edges $E$ connecting pairs of vertices. This deceptively simple definition gives rise to one of the richest and most applicable branches of mathematics.

For AI and machine learning, graphs are not merely a data structure — they are the native language of relational reasoning. Every knowledge graph (Wikidata, Freebase), every molecular structure (drug discovery, protein folding), every social network (recommendation systems), every dependency parse tree (NLP), and every computation graph (PyTorch, JAX autograd) is a graph. The adjacency matrix of a graph is a matrix, connecting graph theory directly to linear algebra. The eigenvalues of graph matrices reveal community structure, connecting graph theory to spectral methods. And the message-passing paradigm of Graph Neural Networks (Kipf & Welling, 2017; Velickovic et al., 2018) is a direct formalization of how information propagates through graph structure.

This section builds the foundational vocabulary and core theory of graphs from first principles. We define graphs rigorously, develop the theory of degree, paths, connectivity, and special graph families, and introduce graph isomorphism and coloring. Every concept is connected to its role in modern AI systems. Later sections in this chapter build on this foundation: §02 covers how to store graphs computationally, §03 develops classical algorithms, §04 introduces spectral methods, §05 covers Graph Neural Networks, and §06 explores random graph models.

## Prerequisites

- Set notation, functions, and mappings — [Chapter 1: Mathematical Foundations](../../01-Mathematical-Foundations/02-Sets-and-Logic/notes.md)
- Basic matrix operations (for adjacency matrix connections) — [Chapter 2: Linear Algebra Basics](../../02-Linear-Algebra-Basics/02-Matrix-Operations/notes.md)
- Summation notation $\sum$ — [Chapter 1: Summation Notation](../../01-Mathematical-Foundations/04-Summation-Product-Notation/notes.md)

## Companion Notebooks

| Notebook                           | Description                                                                                       |
| ---------------------------------- | ------------------------------------------------------------------------------------------------- |
| [theory.ipynb](theory.ipynb)       | Interactive demonstrations: graph construction, degree analysis, connectivity, coloring, WL test  |
| [exercises.ipynb](exercises.ipynb) | 8 graded exercises: handshaking lemma through Weisfeiler-Leman and real-world graph modeling      |

## Learning Objectives

After completing this section, you will:

- Define graphs formally and distinguish directed, undirected, weighted, and simple variants
- Compute vertex degrees, verify the handshaking lemma, and analyse degree sequences
- Distinguish walks, trails, paths, and cycles and compute graph distance and diameter
- Determine connectivity, identify connected components, and find bridges and articulation points
- Recognise and verify special graph families: complete, bipartite, trees, DAGs, planar
- Construct subgraphs, induced subgraphs, complements, and graph products
- Define graph isomorphism and apply the Weisfeiler-Leman test to distinguish non-isomorphic graphs
- Apply graph coloring and compute chromatic numbers for small graphs
- Connect every graph concept to its role in GNNs, knowledge graphs, molecular modeling, and computation graphs

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 What Is a Graph?](#11-what-is-a-graph)
  - [1.2 Why Graphs Matter for AI](#12-why-graphs-matter-for-ai)
  - [1.3 Graphs Are Everywhere](#13-graphs-are-everywhere)
  - [1.4 The Language of Connections](#14-the-language-of-connections)
  - [1.5 Historical Timeline](#15-historical-timeline)
  - [1.6 What This Section Covers vs. What Comes Later](#16-what-this-section-covers-vs-what-comes-later)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 The Graph G = (V, E)](#21-the-graph-g--v-e)
  - [2.2 Directed vs. Undirected Graphs](#22-directed-vs-undirected-graphs)
  - [2.3 Weighted Graphs](#23-weighted-graphs)
  - [2.4 Simple Graphs, Multigraphs, and Pseudographs](#24-simple-graphs-multigraphs-and-pseudographs)
  - [2.5 Standard Examples and Non-Examples](#25-standard-examples-and-non-examples)
  - [2.6 The Null Graph, Empty Graph, and Trivial Graph](#26-the-null-graph-empty-graph-and-trivial-graph)
- [3. Degree and Degree Sequences](#3-degree-and-degree-sequences)
  - [3.1 Vertex Degree](#31-vertex-degree)
  - [3.2 The Handshaking Lemma](#32-the-handshaking-lemma)
  - [3.3 Degree Sequences](#33-degree-sequences)
  - [3.4 Regular Graphs](#34-regular-graphs)
  - [3.5 Degree Distributions in Real-World Graphs](#35-degree-distributions-in-real-world-graphs)
- [4. Paths, Walks, and Cycles](#4-paths-walks-and-cycles)
  - [4.1 Walks, Trails, and Paths](#41-walks-trails-and-paths)
  - [4.2 Cycles](#42-cycles)
  - [4.3 Shortest Paths and Distance](#43-shortest-paths-and-distance)
  - [4.4 Eulerian and Hamiltonian Paths](#44-eulerian-and-hamiltonian-paths)
  - [4.5 Counting Walks with the Adjacency Matrix](#45-counting-walks-with-the-adjacency-matrix)
- [5. Connectivity](#5-connectivity)
  - [5.1 Connected Graphs and Components](#51-connected-graphs-and-components)
  - [5.2 Strongly and Weakly Connected Digraphs](#52-strongly-and-weakly-connected-digraphs)
  - [5.3 Bridges and Articulation Points](#53-bridges-and-articulation-points)
  - [5.4 Vertex and Edge Connectivity](#54-vertex-and-edge-connectivity)
  - [5.5 Connectivity in AI](#55-connectivity-in-ai)
- [6. Special Graph Families](#6-special-graph-families)
  - [6.1 Complete Graphs](#61-complete-graphs)
  - [6.2 Bipartite Graphs](#62-bipartite-graphs)
  - [6.3 Trees and Forests](#63-trees-and-forests)
  - [6.4 DAGs (Directed Acyclic Graphs)](#64-dags-directed-acyclic-graphs)
  - [6.5 Planar Graphs](#65-planar-graphs)
  - [6.6 Hypergraphs](#66-hypergraphs)
- [7. Subgraphs and Graph Operations](#7-subgraphs-and-graph-operations)
  - [7.1 Subgraphs and Induced Subgraphs](#71-subgraphs-and-induced-subgraphs)
  - [7.2 Graph Complement](#72-graph-complement)
  - [7.3 Graph Union, Intersection, and Product](#73-graph-union-intersection-and-product)
  - [7.4 Graph Minor and Subdivision](#74-graph-minor-and-subdivision)
  - [7.5 Line Graphs](#75-line-graphs)
- [8. Graph Isomorphism and Invariants](#8-graph-isomorphism-and-invariants)
  - [8.1 Graph Isomorphism](#81-graph-isomorphism)
  - [8.2 Graph Invariants](#82-graph-invariants)
  - [8.3 The Weisfeiler-Leman Test](#83-the-weisfeiler-leman-test)
  - [8.4 Graph Automorphisms](#84-graph-automorphisms)
- [9. Graph Coloring](#9-graph-coloring)
  - [9.1 Vertex Coloring and Chromatic Number](#91-vertex-coloring-and-chromatic-number)
  - [9.2 Chromatic Polynomial](#92-chromatic-polynomial)
  - [9.3 The Four Color Theorem](#93-the-four-color-theorem)
  - [9.4 Coloring in AI](#94-coloring-in-ai)
- [10. Preview: Representations, Algorithms, and Spectral Methods](#10-preview-representations-algorithms-and-spectral-methods)
  - [10.1 Graph Representations (Preview)](#101-graph-representations-preview)
  - [10.2 Graph Algorithms (Preview)](#102-graph-algorithms-preview)
  - [10.3 Spectral Graph Theory (Preview)](#103-spectral-graph-theory-preview)
  - [10.4 Graph Neural Networks (Preview)](#104-graph-neural-networks-preview)
  - [10.5 Random Graphs (Preview)](#105-random-graphs-preview)
- [11. Common Mistakes](#11-common-mistakes)
- [12. Exercises](#12-exercises)
- [13. Why This Matters for AI (2026 Perspective)](#13-why-this-matters-for-ai-2026-perspective)
- [14. Conceptual Bridge](#14-conceptual-bridge)

---

## 1. Intuition

### 1.1 What Is a Graph?

At its most elemental, a graph is a collection of **things** and **connections between things**. The "things" are called **vertices** (or nodes), and the connections are called **edges** (or links). This definition is deliberately vague about what the things are — and that is exactly the point. The same mathematical framework that describes friendships between people also describes bonds between atoms, links between web pages, and data flow between operations in a neural network.

Consider a simple social network:

```
SOCIAL NETWORK AS A GRAPH
════════════════════════════════════════════════════════════════════════

  Alice ─────── Bob              Vertices: {Alice, Bob, Carol, Dave}
    │          ╱ │               Edges:    {Alice-Bob, Alice-Carol,
    │        ╱   │                          Bob-Carol, Bob-Dave}
    │      ╱     │
  Carol ─╱     Dave              4 vertices, 4 edges

════════════════════════════════════════════════════════════════════════
```

The graph abstracts away everything except the relational structure: who knows whom. It doesn't matter whether Alice is 25 or 65, whether she lives in Tokyo or Toronto. The graph captures only the connectivity pattern.

**For AI:** This abstraction is precisely why graphs are so powerful in machine learning. A graph neural network processes the *structure* of relationships, not the raw pixel data or text. When DeepMind's AlphaFold predicts protein structure, it represents amino acid residues as vertices and spatial proximity as edges — the graph structure encodes the physics that determines how the protein folds.

### 1.2 Why Graphs Matter for AI

Graphs appear in AI in three fundamentally different roles:

**1. Graphs as input data.** Many real-world datasets are naturally graph-structured: social networks, citation networks, molecular structures, protein interaction networks, scene graphs in computer vision, abstract syntax trees in code analysis. Graph Neural Networks (GNNs) process these directly.

**2. Graphs as computational structure.** Every neural network defines a directed acyclic graph (DAG) — the computation graph. Vertices are operations (matmul, add, activation), edges carry tensors. Backpropagation is reverse topological traversal of this DAG. PyTorch's autograd and JAX's tracing both build and traverse computation graphs.

**3. Graphs as knowledge representation.** Knowledge graphs (Wikidata: 100B+ triples, Google Knowledge Graph) represent entities as vertices and relations as labeled edges. Retrieval-augmented generation (RAG) systems traverse these graphs to ground LLM responses in factual knowledge.

```
THREE ROLES OF GRAPHS IN AI
════════════════════════════════════════════════════════════════════════

  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
  │  GRAPHS AS DATA  │   │ GRAPHS AS COMP. │   │ GRAPHS AS       │
  │                  │   │   STRUCTURE     │   │  KNOWLEDGE      │
  ├──────────────────┤   ├─────────────────┤   ├─────────────────┤
  │ Molecules        │   │ Autograd DAGs   │   │ Wikidata        │
  │ Social networks  │   │ Model arch.     │   │ Google KG       │
  │ Citation graphs  │   │ Dataflow graphs  │   │ ConceptNet      │
  │ Scene graphs     │   │ Compiler IR     │   │ RAG retrieval   │
  └──────────────────┘   └─────────────────┘   └─────────────────┘
       GNNs process           Backprop              LLMs query
       these directly          traverses              these for
                               these                  grounding

════════════════════════════════════════════════════════════════════════
```

### 1.3 Graphs Are Everywhere

To build intuition for the diversity of graphs, consider these concrete examples:

**Chemistry:** A molecule is a graph where atoms are vertices and chemical bonds are edges. Benzene ($C_6H_6$) is a cycle graph $C_6$ with alternating single and double bonds. Drug discovery uses GNNs to predict molecular properties from graph structure (Gilmer et al., 2017).

**Natural Language Processing:** A dependency parse tree is a directed graph where words are vertices and grammatical relationships are edges. The sentence "The cat sat on the mat" produces a tree rooted at "sat" with edges to "cat" (subject) and "on" (preposition). Abstract Meaning Representations (AMRs) are DAGs used in semantic parsing.

**Social Networks:** Facebook's social graph has ~3 billion vertices (users) and hundreds of billions of edges (friendships). The degree distribution follows a power law — most users have few friends, a few "hub" users have thousands.

**The Internet:** The World Wide Web is a directed graph where pages are vertices and hyperlinks are edges. Google's PageRank algorithm (Page et al., 1999) — the foundation of web search — computes the dominant eigenvector of a modified adjacency matrix.

**Biology:** Protein-protein interaction networks, gene regulatory networks, neural connectomes (the wiring diagram of a brain) — all are graphs. The human connectome has ~86 billion vertices (neurons) and ~100 trillion edges (synapses).

### 1.4 The Language of Connections

Graph theory provides a precise vocabulary for describing structural properties that arise across all these domains:

- **How many connections does each object have?** → Degree
- **Can we reach any object from any other?** → Connectivity
- **What is the shortest route between two objects?** → Distance, diameter
- **Are there natural groupings?** → Components, communities
- **Can we split objects into two non-interacting groups?** → Bipartiteness
- **Is there redundancy in connections?** → Cycles
- **Are two networks "the same" structurally?** → Isomorphism

Each of these questions has a rigorous mathematical formalization that we develop in this section. And each has direct consequences for AI: the degree distribution determines how information spreads through a GNN, connectivity determines whether a computation graph can propagate gradients, and the Weisfeiler-Leman isomorphism test is provably equivalent to the expressive power of message-passing GNNs (Xu et al., 2019).

### 1.5 Historical Timeline

| Year | Milestone | Significance |
|------|-----------|-------------|
| 1736 | Euler solves Konigsberg bridge problem | Birth of graph theory — first proof that no Eulerian circuit exists |
| 1852 | Guthrie poses the four color conjecture | Sparked 124 years of research; proved computationally in 1976 |
| 1878 | Cayley enumerates trees | First systematic study of tree structures |
| 1936 | Konig publishes first graph theory textbook | Formalised bipartite matching, Konig's theorem |
| 1959 | Erdos and Renyi introduce random graphs | Foundation of probabilistic graph theory |
| 1962 | Dijkstra publishes shortest-path algorithm | Still used in network routing today |
| 1970s | NP-completeness results (graph coloring, Hamiltonian cycle) | Showed fundamental limits of graph computation |
| 1976 | Appel and Haken prove the four color theorem | First major theorem proved by computer |
| 1998 | Watts and Strogatz model small-world networks | Explained "six degrees of separation" |
| 1999 | Page et al. publish PageRank | Graph eigenvector as the basis of web search |
| 2005 | Semantic Web and knowledge graphs emerge | Graphs as structured knowledge for AI |
| 2017 | Kipf and Welling publish GCN | Spectral graph convolutions for node classification |
| 2018 | Velickovic et al. publish GAT | Attention mechanisms on graphs |
| 2019 | Xu et al. prove GNN-WL equivalence | 1-WL test bounds message-passing GNN expressiveness |
| 2024 | Graph Transformers (GPS, Graphormer) | Combining graph structure with transformer attention |
| 2026 | Heterogeneous graph models at scale | Multi-relational graphs for knowledge-augmented LLMs |

### 1.6 What This Section Covers vs. What Comes Later

This section (§01) is the **vocabulary and core theory** of graph theory. It defines what graphs are, introduces the fundamental properties (degree, paths, connectivity, special families, isomorphism, coloring), and provides the mathematical language used throughout the rest of the chapter.

What comes next builds on this vocabulary:

- **§02 Graph Representations** — How to store graphs in memory (adjacency matrix, adjacency list, CSR format)
- **§03 Graph Algorithms** — Algorithms that compute the properties defined here (BFS, DFS, shortest paths, MST)
- **§04 Spectral Graph Theory** — The eigenvalues of graph matrices reveal structural properties
- **§05 Graph Neural Networks** — Neural architectures that learn from graph structure
- **§06 Random Graphs** — Probabilistic models of graph generation

---

## 2. Formal Definitions

### 2.1 The Graph $G = (V, E)$

**Definition (Undirected Graph).** An *undirected graph* is an ordered pair $G = (V, E)$ where:
- $V$ is a finite, non-empty set called the **vertex set** (or node set)
- $E \subseteq \binom{V}{2}$ is a set of unordered pairs of distinct vertices, called the **edge set**

Here $\binom{V}{2} = \{\{u, v\} : u, v \in V, u \neq v\}$ denotes the set of all 2-element subsets of $V$.

We write $n = |V|$ for the **order** of $G$ (number of vertices) and $m = |E|$ for the **size** of $G$ (number of edges).

**Notation.** An edge between vertices $u$ and $v$ is written as $\{u, v\}$, $uv$, or $(u, v)$ depending on context. In this section we use $\{u, v\}$ for undirected edges and $(u, v)$ for directed edges.

**Example 1: The triangle graph $K_3$.**

$$V = \{1, 2, 3\}, \quad E = \{\{1,2\}, \{1,3\}, \{2,3\}\}$$

```
THE TRIANGLE GRAPH K₃
════════════════════════════════════════════════════════════════════════

      1                 n = 3 (order)
     ╱ ╲                m = 3 (size)
    ╱   ╲               Every pair connected → "complete graph"
   2 ─── 3

════════════════════════════════════════════════════════════════════════
```

**Example 2: The path graph $P_4$.**

$$V = \{1, 2, 3, 4\}, \quad E = \{\{1,2\}, \{2,3\}, \{3,4\}\}$$

```
   1 ── 2 ── 3 ── 4          n = 4, m = 3
```

**Example 3: The cycle graph $C_5$.**

$$V = \{1, 2, 3, 4, 5\}, \quad E = \{\{1,2\}, \{2,3\}, \{3,4\}, \{4,5\}, \{5,1\}\}$$

**For AI:** When a GNN processes a molecular graph, each atom is a vertex in $V$ and each chemical bond is an edge in $E$. The formal definition ensures that the graph is well-defined regardless of how atoms are labeled — only the connection pattern matters.

**Key terminology summary:**

| Term | Symbol | Meaning |
|------|--------|---------|
| Vertex (node) | $v \in V$ | An object in the graph |
| Edge (link) | $e \in E$ | A connection between two vertices |
| Order | $n = \lvert V \rvert$ | Number of vertices |
| Size | $m = \lvert E \rvert$ | Number of edges |
| Adjacent | $u \sim v$ | Vertices $u, v$ connected by an edge |
| Incident | $v \in e$ | Vertex $v$ is an endpoint of edge $e$ |
| Neighbour | $u \in \mathcal{N}(v)$ | $u$ is adjacent to $v$ |
| Neighbourhood | $\mathcal{N}(v)$ | Set of all neighbours of $v$ |
| Endpoint | $u, v$ of $\{u,v\}$ | The two vertices of an edge |

**Maximum number of edges.** A simple undirected graph on $n$ vertices has at most $\binom{n}{2} = \frac{n(n-1)}{2}$ edges (achieved by $K_n$). A simple directed graph has at most $n(n-1)$ arcs. A graph with $m$ close to $\binom{n}{2}$ is called **dense**; a graph with $m = O(n)$ is called **sparse**. Most real-world graphs are sparse.

### 2.2 Directed vs. Undirected Graphs

**Definition (Directed Graph / Digraph).** A *directed graph* (or *digraph*) is an ordered pair $G = (V, E)$ where $E \subseteq V \times V$ is a set of **ordered** pairs of vertices. An edge $(u, v)$ is called an **arc** directed from $u$ (the *tail*) to $v$ (the *head*).

The key distinction: in an undirected graph, $\{u, v\} = \{v, u\}$ — the edge has no direction. In a digraph, $(u, v) \neq (v, u)$ — the arc goes from $u$ to $v$, not the reverse.

```
UNDIRECTED vs. DIRECTED
════════════════════════════════════════════════════════════════════════

  Undirected:              Directed:
   A ─── B                  A ───→ B
   │     │                  ↑       │
   │     │                  │       ↓
   D ─── C                  D ←─── C

  {A,B} = {B,A}            (A,B) ≠ (B,A)
  "A knows B"              "A follows B"
  Symmetric relation        Asymmetric relation

════════════════════════════════════════════════════════════════════════
```

**Example (Undirected):** A social network where friendship is mutual — if Alice is friends with Bob, then Bob is friends with Alice. The edge set is symmetric.

**Example (Directed):** Twitter's follow graph — Alice can follow Bob without Bob following Alice. Citation networks — paper A cites paper B, but B does not cite A.

**Example (Directed, AI-critical):** A computation graph in PyTorch. Each operation node receives input tensors along incoming arcs and produces output tensors along outgoing arcs. The direction encodes dataflow: gradients propagate in the reverse direction during backpropagation.

**Non-example:** A "graph" where edges connect a vertex to itself (self-loop) — this violates the $u \neq v$ requirement for simple graphs. We address self-loops in §2.4.

### 2.3 Weighted Graphs

**Definition (Weighted Graph).** A *weighted graph* is a triple $G = (V, E, w)$ where $(V, E)$ is a graph and $w: E \to \mathbb{R}$ is a **weight function** assigning a real number to each edge.

Common weight interpretations:
- **Distance/cost:** In a road network, $w(\{u,v\})$ is the distance between cities $u$ and $v$
- **Similarity/affinity:** In a $k$-NN graph, $w(\{u,v\}) = \exp(-\lVert \mathbf{x}_u - \mathbf{x}_v \rVert^2 / 2\sigma^2)$ (Gaussian kernel)
- **Capacity:** In a flow network, $w((u,v))$ is the maximum flow along the arc
- **Probability:** In a Markov chain, $w((u,v)) = P(\text{transition } u \to v)$

**For AI:** Attention weights in a transformer define a weighted directed graph over token positions. The attention matrix $\operatorname{softmax}(QK^\top / \sqrt{d_k})$ is precisely the weight matrix of a weighted digraph where $w((i,j))$ is the attention that position $i$ pays to position $j$. Graph Attention Networks (GAT, Velickovic et al., 2018) learn these weights as a function of node features.

**Non-example (not a weighted graph):** A matrix $A \in \mathbb{R}^{n \times n}$ with negative entries — this can be interpreted as a signed weighted graph, but the standard weighted graph definition allows $w: E \to \mathbb{R}$ including negative weights. The distinction matters: negative weights break shortest-path algorithms like Dijkstra but are handled by Bellman-Ford.

### 2.4 Simple Graphs, Multigraphs, and Pseudographs

The formal definition in §2.1 defines a **simple graph**: no self-loops and no parallel edges. Real-world graphs often violate these constraints:

**Definition (Self-loop).** A *self-loop* is an edge from a vertex to itself: $\{v, v\}$ or $(v, v)$.

**Definition (Multigraph).** A *multigraph* allows multiple edges between the same pair of vertices. Formally, $E$ is a multiset rather than a set.

**Definition (Pseudograph).** A *pseudograph* allows both self-loops and multiple edges.

| Graph Type | Self-loops | Parallel Edges | Example |
|-----------|-----------|---------------|---------|
| Simple graph | No | No | Social network (at most one friendship) |
| Multigraph | No | Yes | Transportation network (multiple bus routes) |
| Pseudograph | Yes | Yes | State machine (self-transitions) |

**For AI:** Most GNN frameworks assume simple graphs. Self-loops are added explicitly in GCN: $\tilde{A} = A + I_n$ adds a self-loop to every vertex, ensuring that each node's own features are included in the aggregation step. This is a deliberate design choice, not a property of the input graph.

**Non-example:** A relation "is the same as" on a set — every element is related to itself, producing self-loops on every vertex. This is a pseudograph, not a simple graph.

### 2.5 Standard Examples and Non-Examples

**Example 1: The Petersen graph.** A famous graph with 10 vertices and 15 edges, 3-regular (every vertex has degree 3). It is a counterexample to many graph theory conjectures — it is not Hamiltonian, not edge-4-colorable, and not planar.

**Example 2: The Karate Club graph.** Zachary's karate club (1977) — 34 members, 78 edges representing social interactions. The graph split into two communities when the club divided, making it a standard benchmark for community detection. This is the "MNIST of graph learning."

**Example 3: A molecule.** Water ($H_2O$) has 3 vertices (O, H, H) and 2 edges (O-H bonds). Caffeine ($C_8H_{10}N_4O_2$) has 24 vertices and 25 edges.

**Non-example 1: A set without edges.** The pair $(\{1,2,3\}, \emptyset)$ is technically a valid graph (the "empty graph" on 3 vertices), but it has no edges — no relationships are captured.

**Non-example 2: A "graph" with edges between non-existent vertices.** If $V = \{1,2\}$ and $E = \{\{1,3\}\}$, this is not a valid graph because vertex 3 is not in $V$. Edges must connect vertices in the vertex set.

**Non-example 3: An ordered list.** The sequence $[1, 2, 3, 4]$ is not a graph — it has no explicit edge set. However, we can construct a path graph from it: $V = \{1,2,3,4\}$, $E = \{\{1,2\}, \{2,3\}, \{3,4\}\}$.

### 2.6 The Null Graph, Empty Graph, and Trivial Graph

These edge cases clarify the boundaries of the definition:

**Definition (Trivial Graph).** The graph with exactly one vertex and no edges: $G = (\{v\}, \emptyset)$. This is the simplest possible graph.

**Definition (Empty Graph).** A graph $G = (V, \emptyset)$ with $n \geq 1$ vertices and no edges. Also called an *edgeless graph* or *independent set*. Denoted $\bar{K}_n$ (the complement of the complete graph).

**Definition (Null Graph).** The graph with no vertices and no edges: $G = (\emptyset, \emptyset)$. Some authors exclude this, requiring $V \neq \emptyset$. In this curriculum, we require $|V| \geq 1$.

**For AI:** The empty graph $\bar{K}_n$ appears when constructing GNN inputs: before adding edges (via $k$-NN or radius graphs), the initial graph has $n$ nodes with features but no connections. The GCN self-loop trick $\tilde{A} = A + I$ on an empty graph gives $\tilde{A} = I$ — each node only aggregates its own features, equivalent to a standard MLP.

---

## 3. Degree and Degree Sequences

### 3.1 Vertex Degree

**Definition (Degree).** For an undirected graph $G = (V, E)$, the **degree** of a vertex $v \in V$ is the number of edges incident to $v$:

$$\deg(v) = |\{e \in E : v \in e\}|$$

Equivalently, $\deg(v)$ is the number of neighbours of $v$: the vertices adjacent to $v$.

**Definition (In-degree and Out-degree).** For a directed graph $G = (V, E)$:

$$\deg^+(v) = |\{(v, u) \in E\}| \quad \text{(out-degree: edges leaving } v\text{)}$$

$$\deg^-(v) = |\{(u, v) \in E\}| \quad \text{(in-degree: edges entering } v\text{)}$$

The total degree of a vertex in a digraph is $\deg(v) = \deg^+(v) + \deg^-(v)$.

```
DEGREE EXAMPLES
════════════════════════════════════════════════════════════════════════

  Undirected:                    Directed:
      A ─── B ─── C                 A ──→ B ──→ C
      │           │                 ↑           │
      D ────────── E                D ←──────── E

  deg(A) = 2  (B, D)           deg⁺(A) = 1, deg⁻(A) = 1
  deg(B) = 2  (A, C)           deg⁺(B) = 1, deg⁻(B) = 1
  deg(C) = 2  (B, E)           deg⁺(C) = 0, deg⁻(C) = 1  (sink)
  deg(D) = 2  (A, E)           deg⁺(D) = 1, deg⁻(D) = 1
  deg(E) = 2  (C, D)           deg⁺(E) = 1, deg⁻(E) = 1

════════════════════════════════════════════════════════════════════════
```

**For AI:** In a GNN, a vertex's degree determines how many messages it receives during message passing. Vertices with very high degree (hubs) aggregate information from many neighbours, while vertices with degree 1 (leaves) receive information from a single source. This is why GCN normalises by $1/\sqrt{\deg(u)\deg(v)}$ — to prevent high-degree nodes from dominating the aggregation.

**Notation (Adjacency matrix connection).** If $A$ is the adjacency matrix of an undirected graph, then $\deg(v_i) = \sum_{j=1}^n A_{ij}$ — the row sum. The **degree matrix** $D = \operatorname{diag}(\deg(v_1), \ldots, \deg(v_n))$ is the diagonal matrix of degrees. Both $A$ and $D$ are central to spectral graph theory (§04).

### 3.2 The Handshaking Lemma

**Theorem (Handshaking Lemma).** For any undirected graph $G = (V, E)$:

$$\sum_{v \in V} \deg(v) = 2|E|$$

**Proof.** Each edge $\{u, v\} \in E$ contributes exactly 1 to $\deg(u)$ and exactly 1 to $\deg(v)$. Thus each edge contributes exactly 2 to the total degree sum. Summing over all edges gives $\sum_{v \in V} \deg(v) = 2|E|$. $\square$

**Corollary.** The number of vertices with odd degree is always even. (If there were an odd number of odd-degree vertices, the total degree sum would be odd, contradicting $2|E|$ being even.)

**Directed version.** For a digraph: $\sum_{v \in V} \deg^+(v) = \sum_{v \in V} \deg^-(v) = |E|$. Every arc has exactly one tail and one head.

**Example.** The Petersen graph: 10 vertices, each of degree 3. Total degree sum $= 10 \times 3 = 30 = 2 \times 15$ edges. $\checkmark$

**Example.** The star graph $S_5$ (one center connected to 4 leaves): $\deg(\text{center}) = 4$, $\deg(\text{leaf}_i) = 1$ for $i = 1, \ldots, 4$. Sum $= 4 + 1 + 1 + 1 + 1 = 8 = 2 \times 4$ edges. $\checkmark$

**Application: Average degree.** The **average degree** of a graph is $\bar{d} = \frac{1}{n}\sum_{v \in V}\deg(v) = \frac{2m}{n}$. For sparse graphs ($m = O(n)$), the average degree is $O(1)$. For dense graphs ($m = \Theta(n^2)$), the average degree is $\Theta(n)$.

| Graph | $n$ | $m$ | Average degree $2m/n$ |
|-------|-----|-----|----------------------|
| $K_n$ | $n$ | $n(n-1)/2$ | $n - 1$ |
| $C_n$ | $n$ | $n$ | $2$ |
| Tree on $n$ vertices | $n$ | $n-1$ | $2 - 2/n \approx 2$ |
| $K_{m,n}$ | $m+n$ | $mn$ | $2mn/(m+n)$ |

**For AI:** The handshaking lemma provides a quick sanity check when constructing graphs for GNN input. If you compute degrees from an edge list and the sum is odd, your graph construction has a bug. In large-scale graph datasets (OGB, ogbg-molhiv), this is a standard validation step. The average degree also determines the computational cost of one GNN layer: each message-passing step processes $O(m) = O(n\bar{d}/2)$ edges.

### 3.3 Degree Sequences

**Definition (Degree Sequence).** The **degree sequence** of a graph $G$ is the list of vertex degrees arranged in non-increasing order:

$$d_1 \geq d_2 \geq \cdots \geq d_n$$

**Example.** The complete bipartite graph $K_{2,3}$: vertices on one side have degree 3, vertices on the other have degree 2. Degree sequence: $(3, 3, 2, 2, 2)$.

**Definition (Graphic Sequence).** A non-increasing sequence of non-negative integers $(d_1, \ldots, d_n)$ is *graphic* if there exists a simple graph with that degree sequence.

**Theorem (Erdos-Gallai, 1960).** A non-increasing sequence $(d_1, d_2, \ldots, d_n)$ of non-negative integers with even sum is graphic if and only if for each $k \in \{1, 2, \ldots, n\}$:

$$\sum_{i=1}^k d_i \leq k(k-1) + \sum_{i=k+1}^n \min(d_i, k)$$

**Example.** Is $(3, 3, 2, 2, 2)$ graphic? Sum $= 12$ (even). Check $k = 1$: $3 \leq 0 + \min(3,1) + \min(2,1) + \min(2,1) + \min(2,1) = 0 + 1+1+1+1 = 4$. Yes. (Full check confirms graphicality — it is the degree sequence of $K_{2,3}$.)

**Non-example.** Is $(3, 3, 3, 1)$ graphic? Sum $= 10$ (even), $n = 4$. Check $k = 1$: $3 \leq 0 + \min(3,1) + \min(3,1) + \min(1,1) = 3$. Equality holds. Check $k = 3$: $3+3+3 = 9 \leq 3 \cdot 2 + \min(1,3) = 6 + 1 = 7$. Fails! Not graphic — you cannot build a simple graph on 4 vertices where three vertices have degree 3 and one has degree 1.

### 3.4 Regular Graphs

**Definition ($k$-Regular Graph).** A graph is *$k$-regular* if every vertex has degree $k$: $\deg(v) = k$ for all $v \in V$.

**Examples:**
- **0-regular:** The empty graph $\bar{K}_n$ (no edges)
- **1-regular:** A perfect matching (disjoint edges covering all vertices; requires even $n$)
- **2-regular:** A disjoint union of cycles
- **3-regular (cubic):** The Petersen graph, the complete bipartite graph $K_{3,3}$
- **$(n-1)$-regular:** The complete graph $K_n$ (every vertex connected to every other)

**Proposition.** A $k$-regular graph on $n$ vertices has exactly $m = kn/2$ edges (by the handshaking lemma). This requires $kn$ to be even.

**For AI:** Regular graphs are the "uniform" case in GNN analysis. On a $k$-regular graph, the GCN normalisation factor $1/\sqrt{\deg(u)\deg(v)} = 1/k$ is constant, so GCN reduces to simple averaging of neighbour features. This makes regular graphs the easiest case for theoretical GNN analysis (e.g., proving the WL-equivalence result of Xu et al., 2019).

### 3.5 Degree Distributions in Real-World Graphs

Real-world graphs rarely have uniform degree distributions. Two patterns dominate:

**Poisson degree distribution.** In Erdos-Renyi random graphs $G(n, p)$, the degree of each vertex is approximately $\operatorname{Binomial}(n-1, p) \approx \operatorname{Poisson}(\lambda)$ where $\lambda = (n-1)p$. Most vertices have degree close to $\lambda$.

**Power-law degree distribution.** In many real-world networks (social, citation, web), the fraction of vertices with degree $k$ follows:

$$P(\deg = k) \propto k^{-\gamma}, \quad \gamma \in (2, 3) \text{ typically}$$

This is called a **scale-free** distribution. It means a few "hub" vertices have enormously high degree while most vertices have low degree.

| Network | Type | Typical $\gamma$ |
|---------|------|------------------|
| WWW (out-degree) | Scale-free | $\gamma \approx 2.1$ |
| Citation networks | Scale-free | $\gamma \approx 3.0$ |
| Protein interactions | Scale-free | $\gamma \approx 2.4$ |
| Erdos-Renyi | Poisson | N/A |
| Road networks | Near-uniform | N/A ($\deg \approx 3{-}4$) |

**For AI:** Power-law degree distributions create challenges for GNNs. Hub vertices with degree $>1000$ dominate mini-batch sampling (GraphSAGE, Hamilton et al., 2017) and can cause representation collapse — the "over-squashing" problem. Solutions include degree-based sampling, virtual nodes, and graph transformers that bypass local message passing.

**Clustering coefficient.** Beyond degree, an important local structural measure is the **clustering coefficient**, which quantifies how clustered a vertex's neighbourhood is.

**Definition (Local Clustering Coefficient).** For a vertex $v$ with $\deg(v) \geq 2$:

$$C(v) = \frac{\text{number of edges among neighbours of } v}{\binom{\deg(v)}{2}}$$

This is the fraction of possible edges among $v$'s neighbours that actually exist. $C(v) = 1$ means the neighbourhood is a clique; $C(v) = 0$ means no two neighbours are connected.

**Definition (Global Clustering Coefficient).** The average over all vertices: $C = \frac{1}{n}\sum_{v} C(v)$.

**Alternative: Transitivity.** $T(G) = \frac{3 \times \text{number of triangles}}{\text{number of connected triples}} = \frac{\operatorname{tr}(A^3)}{\mathbf{1}^\top A^2 \mathbf{1} - \operatorname{tr}(A^2)}$.

| Network type | Typical $C$ | Interpretation |
|-------------|-------------|----------------|
| Erdos-Renyi $G(n,p)$ | $\approx p$ | Low, unclustered |
| Social networks | $0.1$–$0.5$ | High, clustered (friends of friends are friends) |
| Protein interactions | $0.1$–$0.3$ | Medium clustering |
| Regular lattice | High | Very clustered but large diameter |
| Small-world (Watts-Strogatz) | High | High clustering AND small diameter |

**For AI:** The clustering coefficient is a standard graph-level feature in GNN benchmarks. High clustering indicates the presence of many triangles — triangle-counting is a canonical task that standard 1-WL GNNs cannot perform exactly (triangles require 3-WL). This motivates higher-order GNN architectures that explicitly count triangular motifs.

> **Note:** The full theory of degree distributions and random graph models is developed in [§06 Random Graphs](../06-Random-Graphs/notes.md). Here we introduce the concept; there we formalise it.

---

## 4. Paths, Walks, and Cycles

### 4.1 Walks, Trails, and Paths

These three concepts form a hierarchy of increasing restriction on vertex/edge repetition:

**Definition (Walk).** A *walk* of length $k$ in $G$ is a sequence of vertices $v_0, v_1, \ldots, v_k$ such that $\{v_{i-1}, v_i\} \in E$ for each $i \in \{1, \ldots, k\}$. Vertices and edges **may repeat**.

**Definition (Trail).** A *trail* is a walk in which no **edge** is repeated (but vertices may repeat).

**Definition (Path).** A *path* is a walk in which no **vertex** is repeated (and hence no edge is repeated).

```
WALK vs. TRAIL vs. PATH
════════════════════════════════════════════════════════════════════════

  Graph:   A ─── B ─── C
           │     │     │
           D ─── E ─── F

  Walk:    A → B → E → B → C          (B visited twice — OK for walk)
  Trail:   A → B → E → D → A → B → C  (no edge repeated, A visited twice)
  Path:    A → B → E → F → C           (no vertex repeated)

  Restriction:   Walk ⊇ Trail ⊇ Path
                 (every path is a trail, every trail is a walk)

════════════════════════════════════════════════════════════════════════
```

**Notation.** The **length** of a walk/trail/path is the number of edges traversed (not vertices). A path from $u$ to $v$ is called a *$u$-$v$ path*.

**Proposition.** If there exists a walk from $u$ to $v$, then there exists a path from $u$ to $v$. (Proof: remove cycles from the walk to obtain a path.)

**For AI:** In a GNN with $k$ message-passing layers, information from vertex $u$ can reach vertex $v$ if and only if there exists a walk of length $\leq k$ from $u$ to $v$. The GNN's **receptive field** at depth $k$ is exactly the set of vertices reachable by walks of length $\leq k$. Note: it is walks (not paths) because GNNs can re-visit vertices through different neighbours at different layers.

### 4.2 Cycles

**Definition (Cycle).** A *cycle* of length $k$ (also called a *$k$-cycle*) is a closed walk $v_0, v_1, \ldots, v_k$ where $v_0 = v_k$, all other vertices are distinct, and $k \geq 3$.

**Definition (Girth).** The *girth* of a graph is the length of its shortest cycle. If the graph has no cycles (is acyclic), the girth is defined as $\infty$.

**Examples:**
- $K_3$ (triangle): girth $= 3$
- $K_4$: girth $= 3$ (contains many triangles)
- $C_n$ (cycle graph): girth $= n$
- Any tree: girth $= \infty$ (acyclic)
- Petersen graph: girth $= 5$ (smallest cycle has 5 vertices)

**Definition (Acyclic Graph).** A graph with no cycles. An undirected acyclic graph is a forest; a connected forest is a tree.

**For AI:** Cycles in computation graphs would create infinite loops during forward pass evaluation. This is why computation graphs are DAGs (directed acyclic graphs) — acyclicity ensures a valid topological ordering for sequential evaluation. Recurrent Neural Networks (RNNs) appear to have cycles, but when "unrolled" over time steps, the computation graph is a DAG.

### 4.3 Shortest Paths and Distance

**Definition (Distance).** The *distance* $d(u, v)$ between vertices $u$ and $v$ is the length of the shortest $u$-$v$ path. If no such path exists (they are in different components), $d(u, v) = \infty$.

**Definition (Eccentricity).** The *eccentricity* of a vertex $v$ is $\varepsilon(v) = \max_{u \in V} d(v, u)$ — the distance to the farthest vertex.

**Definition (Diameter).** The *diameter* of a graph is $\operatorname{diam}(G) = \max_{v \in V} \varepsilon(v) = \max_{u, v \in V} d(u, v)$ — the maximum distance between any two vertices.

**Definition (Radius).** The *radius* of a graph is $\operatorname{rad}(G) = \min_{v \in V} \varepsilon(v)$.

**Definition (Center).** The *center* of a graph is the set of vertices with minimum eccentricity: $\{v : \varepsilon(v) = \operatorname{rad}(G)\}$.

**Examples:**
- Path graph $P_n$: diameter $= n - 1$, radius $= \lfloor(n-1)/2\rfloor$
- Cycle $C_n$: diameter $= \lfloor n/2 \rfloor$
- Complete graph $K_n$: diameter $= 1$ (every pair is adjacent)
- Star graph $S_n$ (one center connected to $n-1$ leaves): diameter $= 2$

**Proposition.** For any connected graph: $\operatorname{rad}(G) \leq \operatorname{diam}(G) \leq 2 \cdot \operatorname{rad}(G)$.

**Proof sketch.** The left inequality is immediate: the max over all vertices is $\geq$ the min. For the right inequality: let $u, v$ achieve the diameter and let $c$ be a center vertex (achieving the radius). Then $d(u,v) \leq d(u,c) + d(c,v) \leq \varepsilon(c) + \varepsilon(c) = 2 \cdot \operatorname{rad}(G)$. $\square$

**Distance properties (metric space).** The graph distance function $d: V \times V \to \mathbb{N} \cup \{0\}$ is a metric:

1. **Non-negativity:** $d(u,v) \geq 0$, with $d(u,v) = 0 \iff u = v$
2. **Symmetry:** $d(u,v) = d(v,u)$ (for undirected graphs)
3. **Triangle inequality:** $d(u,w) \leq d(u,v) + d(v,w)$

This means the vertex set of any connected graph forms a metric space under graph distance. This observation connects graph theory to metric geometry and is used in graph embedding methods (e.g., embedding graph vertices into Euclidean space while approximately preserving distances).

**Distance matrix.** The **distance matrix** $D \in \mathbb{N}^{n \times n}$ has entries $D_{ij} = d(v_i, v_j)$. This is distinct from the degree matrix (also sometimes denoted $D$). The distance matrix contains the full pairwise distance information of the graph and is used as a positional encoding in graph transformers (Li et al., 2020).

```
DISTANCE MATRIX EXAMPLE (P₄: 1─2─3─4)
════════════════════════════════════════════════════════════════════════

  Distance matrix:          Eccentricities:
       1  2  3  4           ε(1) = 3  (farthest: vertex 4)
  1 [  0  1  2  3 ]         ε(2) = 2  (farthest: vertex 4)
  2 [  1  0  1  2 ]         ε(3) = 2  (farthest: vertex 1)
  3 [  2  1  0  1 ]         ε(4) = 3  (farthest: vertex 1)
  4 [  3  2  1  0 ]
                            Diameter = 3, Radius = 2
                            Center = {2, 3}

════════════════════════════════════════════════════════════════════════
```

**For AI:** The diameter of a graph determines the minimum number of GNN layers needed to propagate information between the two most distant vertices. If $\operatorname{diam}(G) = d$, a GNN with fewer than $d$ layers cannot use the full graph structure for prediction. The "small-world" property — most real-world graphs have $\operatorname{diam}(G) = O(\log n)$ — explains why GNNs with 2-3 layers often suffice in practice.

> **Note:** Algorithms for computing shortest paths (BFS for unweighted, Dijkstra for weighted, Bellman-Ford for negative weights) are developed in [§03 Graph Algorithms](../03-Graph-Algorithms/notes.md).

### 4.4 Eulerian and Hamiltonian Paths

**Definition (Eulerian Trail/Circuit).** An *Eulerian trail* is a trail that visits every **edge** exactly once. An *Eulerian circuit* is a closed Eulerian trail (starts and ends at the same vertex).

**Theorem (Euler, 1736).** A connected graph has an Eulerian circuit if and only if every vertex has even degree. It has an Eulerian trail (but not a circuit) if and only if exactly two vertices have odd degree.

This was the first theorem of graph theory, proved by Euler to show that the Konigsberg bridge problem has no solution (the graph of Konigsberg had four vertices, all of odd degree).

```
THE KONIGSBERG BRIDGE PROBLEM
════════════════════════════════════════════════════════════════════════

  The city of Konigsberg (now Kaliningrad) had 7 bridges connecting
  4 landmasses. Can you walk across each bridge exactly once?

  Landmasses as vertices, bridges as edges:

       A ──────── B         deg(A) = 3  (odd)
      ╱│╲        │         deg(B) = 5  (odd)
     ╱ │ ╲       │         deg(C) = 3  (odd)
    ╱  │  ╲      │         deg(D) = 3  (odd)
   C   │   D─────┘
       │                   All degrees odd → no Eulerian trail exists
       └───────────D       (Euler's theorem: need 0 or 2 odd-degree vertices)

════════════════════════════════════════════════════════════════════════
```

**Proof sketch (Euler's theorem — necessity).** If a graph has an Eulerian circuit, every time the walk enters a vertex on one edge it must leave on a different edge. So edges at each vertex are paired: enter-leave, enter-leave, ... This means every vertex uses an even number of edges, so $\deg(v)$ is even. For an Eulerian trail (not circuit), the start and end vertices each use one unpaired edge, so exactly two vertices have odd degree. $\square$

**Definition (Hamiltonian Path/Cycle).** A *Hamiltonian path* is a path that visits every **vertex** exactly once. A *Hamiltonian cycle* is a cycle that visits every vertex exactly once.

**Contrast.** Euler: every edge once. Hamilton: every vertex once.

| Property | Eulerian | Hamiltonian |
|---------|----------|-------------|
| Visits | Every edge once | Every vertex once |
| Existence test | Polynomial (check degrees) | NP-complete |
| $K_n$ | Has Eulerian circuit iff $n$ is odd | Always has Hamiltonian cycle ($n \geq 3$) |
| $K_{m,n}$ | Has Eulerian circuit iff $m, n$ both even | Has Hamiltonian cycle iff $m = n$ |

**For AI:** The Travelling Salesman Problem (TSP) — finding the shortest Hamiltonian cycle in a weighted complete graph — is a canonical NP-hard combinatorial optimization problem. Neural combinatorial optimization (Vinyals et al., 2015; Kool et al., 2019) uses attention-based models to learn heuristic TSP solutions. The graph structure of TSP is $K_n$ with distance weights.

### 4.5 Counting Walks with the Adjacency Matrix

**Theorem.** Let $A$ be the adjacency matrix of a graph $G$. The entry $(A^k)_{ij}$ counts the number of walks of length $k$ from vertex $i$ to vertex $j$.

**Proof sketch.** By induction on $k$. Base case: $A^1 = A$, and $A_{ij} = 1$ iff there is a walk of length 1 (an edge) from $i$ to $j$. Inductive step: $(A^{k+1})_{ij} = \sum_{\ell} (A^k)_{i\ell} \cdot A_{\ell j}$. A walk of length $k+1$ from $i$ to $j$ consists of a walk of length $k$ from $i$ to some $\ell$, followed by an edge from $\ell$ to $j$. Summing over all possible intermediate vertices $\ell$ gives the matrix product. $\square$

**Example.** For the triangle $K_3$ with adjacency matrix:

$$A = \begin{pmatrix} 0 & 1 & 1 \\ 1 & 0 & 1 \\ 1 & 1 & 0 \end{pmatrix}, \quad A^2 = \begin{pmatrix} 2 & 1 & 1 \\ 1 & 2 & 1 \\ 1 & 1 & 2 \end{pmatrix}$$

$(A^2)_{11} = 2$: there are 2 walks of length 2 from vertex 1 to itself ($1 \to 2 \to 1$ and $1 \to 3 \to 1$). The diagonal entries of $A^2$ equal the vertex degrees.

**Corollary.** The number of triangles containing vertex $i$ is $(A^3)_{ii} / 2$ (each triangle is counted twice, once in each direction). The total number of triangles in $G$ is $\operatorname{tr}(A^3) / 6$.

**Example: walk counts in $K_3$.**

$$A^3 = \begin{pmatrix} 2 & 3 & 3 \\ 3 & 2 & 3 \\ 3 & 3 & 2 \end{pmatrix}$$

$(A^3)_{11} = 2$, so vertex 1 is in $2/2 = 1$ triangle. $\operatorname{tr}(A^3) = 6$, so there are $6/6 = 1$ triangle total. Correct for $K_3$.

**Extended corollary: subgraph counting.**

| Subgraph | Formula using $A$ |
|----------|------------------|
| Edges | $\operatorname{tr}(A^2)/2 = m$ |
| Triangles | $\operatorname{tr}(A^3)/6$ |
| Walks of length $k$ (total) | $\mathbf{1}^\top A^k \mathbf{1}$ |
| Paths of length $k$ | Harder — requires inclusion-exclusion |

**Connection to GNN layers.** After $k$ GNN layers, the aggregated representation of vertex $i$ contains information from all vertices reachable by walks of length $\leq k$ from $i$. The contribution of vertex $j$ to vertex $i$'s representation after exactly $k$ layers is proportional to $(A^k)_{ij}$ (in the linear, non-normalised case). The normalised GCN layer uses $\hat{A} = D^{-1/2}AD^{-1/2}$, so the $k$-layer GCN computes $\hat{A}^k X W$ — a matrix whose entries are walks of length $k$ normalised by vertex degrees.

**For AI:** This theorem connects graph theory directly to linear algebra. The spectral decomposition $A = Q\Lambda Q^\top$ gives $A^k = Q\Lambda^k Q^\top$, so the number of walks of length $k$ is controlled by the eigenvalues of $A$. The dominant eigenvalue determines the exponential growth rate of walk counts — this is the mathematical foundation of PageRank (stationary distribution of a random walk = dominant eigenvector of the transition matrix). Full spectral analysis is in [§04 Spectral Graph Theory](../04-Spectral-Graph-Theory/notes.md).

---

## 5. Connectivity

### 5.1 Connected Graphs and Components

**Definition (Connected Graph).** An undirected graph $G$ is *connected* if for every pair of vertices $u, v \in V$, there exists a path from $u$ to $v$.

**Definition (Connected Component).** A *connected component* of $G$ is a maximal connected subgraph — a connected subgraph that is not a proper subgraph of any larger connected subgraph.

Every graph can be uniquely decomposed into its connected components. If $G$ has $c$ connected components, then $c = 1$ means $G$ is connected.

```
CONNECTED COMPONENTS
════════════════════════════════════════════════════════════════════════

  Component 1:    Component 2:    Component 3:
  A ─── B         E ─── F         H
  │   ╱           │
  │ ╱             G
  C ─── D

  3 connected components
  Vertices: {A,B,C,D} ∪ {E,F,G} ∪ {H}

════════════════════════════════════════════════════════════════════════
```

**Proposition.** A graph on $n$ vertices is connected if and only if it has at least $n - 1$ edges (necessary but not sufficient — trees achieve exactly $n - 1$).

**Proposition (Component count from Laplacian).** The number of connected components equals the multiplicity of eigenvalue 0 in the graph Laplacian $L = D - A$. This is a deep result connecting connectivity (combinatorial) to spectrum (algebraic).

**For AI:** If a GNN's input graph has multiple connected components, information cannot flow between components through message passing. Each component is processed independently. In molecular graphs, a molecule with disconnected fragments (e.g., a salt like NaCl in solution) has multiple components — the GNN must handle each separately or use a virtual node to connect them.

### 5.2 Strongly and Weakly Connected Digraphs

For directed graphs, connectivity bifurcates into two notions:

**Definition (Strongly Connected).** A digraph is *strongly connected* if for every pair $u, v$, there exists a directed path from $u$ to $v$ AND a directed path from $v$ to $u$.

**Definition (Weakly Connected).** A digraph is *weakly connected* if the underlying undirected graph (obtained by ignoring edge directions) is connected.

**Definition (Strongly Connected Component / SCC).** A *strongly connected component* is a maximal strongly connected subgraph.

```
STRONG vs. WEAK CONNECTIVITY
════════════════════════════════════════════════════════════════════════

  A ──→ B ──→ C ──→ D        Weakly connected: YES (underlying
  ↑           │                                   undirected graph
  └───────────┘                                   is connected)

  SCCs: {A, B, C} and {D}    Strongly connected: NO (no path D → A)
  
  Within {A, B, C}:          A→B→C→A forms a directed cycle
                              (strongly connected)

════════════════════════════════════════════════════════════════════════
```

**Example:** A citation network is weakly connected (following links in either direction, you can reach most papers) but not strongly connected (you cannot follow citations from a 2024 paper to a 2023 paper — citations only go backward in time).

**Example (Condensation DAG).** Given the SCC decomposition, we can form the **condensation** of a digraph: contract each SCC to a single vertex. The resulting graph is always a DAG (if it had a directed cycle, the SCCs involved would merge into one). This DAG reveals the macroscopic flow structure of the digraph.

**Algorithms for SCCs.** Tarjan's algorithm (1972) and Kosaraju's algorithm both find all SCCs in $O(n + m)$ time using DFS. These are covered in detail in [§03 Graph Algorithms](../03-Graph-Algorithms/notes.md).

**For AI:** The SCC decomposition of a computation graph reveals which operations form feedback loops. In standard feedforward networks, each SCC is a single vertex (the graph is a DAG). In architectures with weight sharing (like RNNs unrolled over time), understanding the SCC structure helps with gradient flow analysis. The condensation DAG of a knowledge graph reveals the hierarchical structure of entity relationships.

### 5.3 Bridges and Articulation Points

**Definition (Bridge).** An edge $e$ is a *bridge* (or *cut edge*) if removing $e$ increases the number of connected components.

**Definition (Articulation Point).** A vertex $v$ is an *articulation point* (or *cut vertex*) if removing $v$ and all its incident edges increases the number of connected components.

**Example:** In the path graph $P_4 = 1{-}2{-}3{-}4$, every edge is a bridge and vertices 2 and 3 are articulation points. Removing edge $\{2,3\}$ disconnects $\{1,2\}$ from $\{3,4\}$.

**Example (non-bridge):** In the cycle $C_4 = 1{-}2{-}3{-}4{-}1$, no edge is a bridge — removing any single edge leaves the graph connected (as a path). No vertex is an articulation point.

**Proposition.** An edge $e = \{u,v\}$ is a bridge if and only if $e$ is not contained in any cycle. Equivalently, $e$ is a bridge iff there is no alternative path from $u$ to $v$.

**Definition (Biconnected Graph).** A connected graph with no articulation points is *biconnected*. Equivalently, between any two vertices there are at least two vertex-disjoint paths.

**For AI:** Bridges and articulation points identify structural vulnerabilities in networks. In a knowledge graph, an articulation point is an entity whose removal disconnects parts of the knowledge base. In network analysis for social media, bridges between communities are the edges that enable information flow between groups.

### 5.4 Vertex and Edge Connectivity

**Definition (Vertex Connectivity).** The *vertex connectivity* $\kappa(G)$ is the minimum number of vertices whose removal disconnects $G$ (or reduces it to a single vertex). A graph is *$k$-connected* if $\kappa(G) \geq k$.

**Definition (Edge Connectivity).** The *edge connectivity* $\lambda(G)$ is the minimum number of edges whose removal disconnects $G$.

**Theorem (Whitney, 1932).** For any graph $G$:

$$\kappa(G) \leq \lambda(G) \leq \delta(G)$$

where $\delta(G) = \min_{v \in V} \deg(v)$ is the minimum degree.

**Theorem (Menger, 1927).** The maximum number of vertex-disjoint $u$-$v$ paths equals the minimum number of vertices whose removal disconnects $u$ from $v$.

This is a max-flow min-cut result: the maximum "flow" of disjoint paths equals the minimum "cut" of vertices. The edge version: the maximum number of edge-disjoint $u$-$v$ paths equals the minimum edge cut separating $u$ from $v$.

**Example of Whitney's inequality.** Consider a graph that is a "book" — two triangles sharing one edge:

```text
  A ─── B         deg(A) = deg(B) = 3
 ╱│╲   ╱│╲        δ(G) = 3  (minimum degree)
C │ D─E │ F       λ(G) = 2  (remove the two edges at the shared edge)
 ╲│╱   ╲│╱        κ(G) = 1  (remove the shared vertex D or E)
  └───────┘
```

So $\kappa = 1 \leq \lambda = 2 \leq \delta = 3$. The inequality is strict.

**Achieving equality.** For $k$-regular graphs: if $G$ is $k$-regular and $k$-edge-connected, then $\kappa(G) = \lambda(G) = k$. Complete graphs $K_n$ achieve $\kappa = \lambda = \delta = n-1$ (maximum possible).

**For AI:** Menger's theorem is the combinatorial version of the max-flow min-cut theorem, which is fundamental to network flow algorithms used in image segmentation (graph cuts), assignment problems, and scheduling. The edge connectivity $\lambda(G)$ measures graph robustness: how many links must fail before the network breaks. This is directly applicable to reliability analysis of distributed systems and communication networks.

### 5.5 Connectivity in AI

**Computation graph connectivity.** A computation graph must be connected (in the weakly-connected sense for digraphs) for gradients to flow from the loss to all parameters. If a parameter's computation is disconnected from the loss, its gradient is zero — it cannot be trained. PyTorch detects this and raises warnings.

**Knowledge graph connectivity.** The "reachability" of entities in a knowledge graph determines what a RAG system can retrieve. If the query entity and the answer entity are in different connected components, no amount of graph traversal will find the answer.

**GNN receptive field.** A GNN with $k$ layers can propagate information across paths of length $\leq k$. If the graph diameter exceeds $k$, some vertex pairs cannot exchange information. This is the "under-reaching" problem, complementary to the "over-smoothing" problem (too many layers cause all representations to converge).

**Molecular connectivity.** In molecular graphs, the number of connected components indicates whether the input is a single molecule or a mixture/complex. Graph-level prediction tasks (e.g., molecular property prediction) typically assume a single connected component.

---

## 6. Special Graph Families

### 6.1 Complete Graphs

**Definition (Complete Graph $K_n$).** The *complete graph* on $n$ vertices is the graph where every pair of distinct vertices is connected by an edge:

$$E = \binom{V}{2}, \quad |E| = \binom{n}{2} = \frac{n(n-1)}{2}$$

**Properties of $K_n$:**
- $(n-1)$-regular (every vertex has degree $n-1$)
- Diameter $= 1$ (every pair is adjacent)
- Number of edges: $\binom{n}{2}$
- Number of triangles: $\binom{n}{3}$
- Number of spanning trees: $n^{n-2}$ (Cayley's formula)
- Chromatic number: $\chi(K_n) = n$
- Hamiltonian for $n \geq 3$, Eulerian for odd $n$

```
COMPLETE GRAPHS
════════════════════════════════════════════════════════════════════════

  K₁    K₂      K₃        K₄           K₅
   •    •──•    •         •───•       •─────•
                │╲        │╲ ╱│      ╱│╲   ╱│
                │ ╲       │ ╳ │    ╱  │ ╲╱  │
                •──•      │╱ ╲│   •───•──•──•
                          •───•       ╲│╱
  m=0   m=1    m=3       m=6     m=10

════════════════════════════════════════════════════════════════════════
```

**For AI:** The fully-connected attention mechanism in transformers computes attention weights over $K_n$ (where $n$ is the sequence length). Every token attends to every other token — this is exactly message passing on the complete graph. The $O(n^2)$ complexity of self-attention is the cost of processing $K_n$.

### 6.2 Bipartite Graphs

**Definition (Bipartite Graph).** A graph $G = (V, E)$ is *bipartite* if $V$ can be partitioned into two disjoint sets $U$ and $W$ such that every edge connects a vertex in $U$ to a vertex in $W$: $E \subseteq U \times W$.

**Theorem (Odd-Cycle Characterisation).** A graph is bipartite if and only if it contains no odd-length cycle.

**Proof sketch.** ($\Rightarrow$) If $G$ is bipartite with parts $U, W$, any cycle must alternate between $U$ and $W$, requiring even length. ($\Leftarrow$) If $G$ has no odd cycle, pick any vertex $v$, colour it by distance: $U = \{u : d(v,u) \text{ even}\}$, $W = \{u : d(v,u) \text{ odd}\}$. No edge connects two vertices at the same distance (that would create an odd cycle). $\square$

**Definition (Complete Bipartite Graph $K_{m,n}$).** The bipartite graph with parts of size $m$ and $n$ where every vertex in one part is connected to every vertex in the other. It has $m + n$ vertices and $mn$ edges.

```
BIPARTITE GRAPH
════════════════════════════════════════════════════════════════════════

  General bipartite:        Complete bipartite K₃,₂:
  U: ● ─── ○              U: ●───○
     │   ╱                    │╲ ╱│
     ● ╱                      │ ╳ │
     │                        │╱ ╲│
  W: ○     ○               W: ●───○───●

  Partition: U (filled), W (open)

════════════════════════════════════════════════════════════════════════
```

**Examples:**
- User-item interaction graphs (recommender systems): users and items are two parts
- Bipartite matching: job assignment (workers ↔ tasks)
- Dependency parse trees are bipartite when arcs alternate between heads and dependents

**Non-example:** $K_3$ (the triangle) is NOT bipartite — it contains an odd cycle of length 3.

**Non-example:** $C_5$ (the pentagon) is NOT bipartite — it is an odd cycle of length 5.

**Testing bipartiteness.** BFS provides an efficient $O(n + m)$ bipartiteness test: start BFS from any vertex, alternating colours between levels. If any edge connects two same-colour vertices, the graph is not bipartite (and that edge, combined with the BFS tree paths, gives an odd cycle certificate). This is covered in [§03 Graph Algorithms](../03-Graph-Algorithms/notes.md).

**Theorem (Konig, 1931).** In a bipartite graph, the size of the maximum matching equals the size of the minimum vertex cover.

This is a foundational result in combinatorial optimisation. A **matching** is a set of edges with no shared vertices. A **vertex cover** is a set of vertices that includes at least one endpoint of every edge. Konig's theorem does NOT hold for non-bipartite graphs (e.g., $K_3$ has matching size 1 but minimum vertex cover size 2).

**Bipartite adjacency matrix structure.** The adjacency matrix of a bipartite graph with parts $U$ ($m$ vertices) and $W$ ($n$ vertices) has a block structure:

$$A = \begin{pmatrix} 0_m & B \\ B^\top & 0_n \end{pmatrix}$$

where $B \in \{0,1\}^{m \times n}$ is the **biadjacency matrix**. This block structure has important spectral consequences: the eigenvalues of $A$ are symmetric about 0 (if $\lambda$ is an eigenvalue, so is $-\lambda$).

**For AI:** Recommender systems model user-item interactions as bipartite graphs. Matrix factorisation for recommendation (the Netflix Prize approach) is equivalent to low-rank approximation of the biadjacency matrix $B$. Knowledge graphs with subject-predicate-object triples can be viewed as bipartite between entities and relations. The block structure of the bipartite adjacency matrix is exploited in bipartite GNN architectures for recommendation (PinSage, Ying et al., 2018).

### 6.3 Trees and Forests

**Definition (Tree).** A *tree* is a connected acyclic graph.

**Definition (Forest).** A *forest* is an acyclic graph (not necessarily connected). Each connected component of a forest is a tree.

**Theorem (Tree Equivalences).** For a graph $G$ on $n$ vertices, the following are equivalent:

1. $G$ is a tree (connected and acyclic)
2. $G$ is connected and has exactly $n - 1$ edges
3. $G$ is acyclic and has exactly $n - 1$ edges
4. There is exactly one path between every pair of vertices
5. $G$ is connected, but removing any edge disconnects it (every edge is a bridge)
6. $G$ is acyclic, but adding any edge creates exactly one cycle

**Proof (1 $\Leftrightarrow$ 2).** ($\Rightarrow$) A connected graph has $\geq n-1$ edges. If it had $\geq n$ edges, by the pigeonhole principle there would be a cycle (contradiction with acyclic). So exactly $n-1$. ($\Leftarrow$) A connected graph with $n-1$ edges has no "spare" edges, so no cycles. $\square$

**Proof (1 $\Rightarrow$ 4).** Suppose $G$ is a tree and there exist two distinct paths $P_1$ and $P_2$ from $u$ to $v$. Since $P_1 \neq P_2$, they diverge at some vertex $w$ and rejoin later at some vertex $x$. The subpaths $w \to x$ via $P_1$ and $w \to x$ via $P_2$ form a cycle — contradicting acyclicity. So the path is unique. $\square$

**Proof (4 $\Rightarrow$ 5).** If there is exactly one path between every pair, then removing any edge $\{u,v\}$ destroys the unique $u$-$v$ path, disconnecting $u$ from $v$. So every edge is a bridge. $\square$

```
TREE EQUIVALENCES — VISUAL SUMMARY
════════════════════════════════════════════════════════════════════════

  Connected + Acyclic        Exactly one       Connected, all
  = TREE                     path between      edges are bridges
    │                        any pair              │
    │    n-1 edges           │                     │
    ▼    (connected)         ▼                     ▼
  ┌──────────────────────────────────────────────────────┐
  │              ALL SIX CONDITIONS EQUIVALENT            │
  │       (any one implies all the others)                │
  └──────────────────────────────────────────────────────┘
    ▲                        ▲                     ▲
    │    n-1 edges           │                     │
    │    (acyclic)      Adding any edge       Maximal acyclic
    │                   creates exactly       = minimal connected
                        one cycle

════════════════════════════════════════════════════════════════════════
```

**Theorem (Cayley's Formula, 1889).** The number of labeled spanning trees of the complete graph $K_n$ is $n^{n-2}$.

| $n$ | $K_n$ | Spanning trees $n^{n-2}$ |
|-----|-------|--------------------------|
| 1 | single vertex | $1$ |
| 2 | single edge | $1$ |
| 3 | triangle | $3$ |
| 4 | $K_4$ | $16$ |
| 5 | $K_5$ | $125$ |

**Theorem (Kirchhoff's Matrix Tree Theorem).** The number of spanning trees of any graph $G$ equals any cofactor of the Laplacian matrix $L = D - A$. Specifically, it equals $\det(L')$ where $L'$ is $L$ with any one row and column deleted.

**Definition (Spanning Tree).** A *spanning tree* of a connected graph $G$ is a tree that includes all vertices of $G$. Every connected graph has at least one spanning tree.

**Definition (Rooted Tree).** A tree with a designated root vertex. Edges are implicitly directed away from (or toward) the root. Rooted trees define a parent-child hierarchy.

**Examples:**
- Parse trees in NLP: dependency trees, constituency trees
- File system directory structures
- Decision trees in machine learning
- Binary search trees, B-trees, tries

**For AI:** Trees are fundamental structures in AI:
- **Decision trees** (CART, Random Forest, XGBoost) — each internal node is a split, each leaf is a prediction
- **Abstract syntax trees (ASTs)** — code represented as trees for program analysis and code generation
- **Spanning trees of graphs** — GNN approaches sometimes use tree decomposition for efficient message passing
- **Hierarchical clustering dendrograms** — trees representing cluster merging

### 6.4 DAGs (Directed Acyclic Graphs)

**Definition (DAG).** A *directed acyclic graph* is a directed graph with no directed cycles.

**Theorem (Topological Ordering).** A directed graph has a topological ordering if and only if it is a DAG.

A **topological ordering** is a linear ordering of vertices such that for every directed edge $(u, v)$, vertex $u$ comes before $v$ in the ordering. A DAG may have multiple valid topological orderings.

```
DAG AND TOPOLOGICAL ORDER
════════════════════════════════════════════════════════════════════════

  DAG:                      Topological orderings:
  A ──→ B ──→ D             A, B, C, D, E
  │     │                   A, C, B, D, E
  ↓     ↓                   A, C, B, E, D   (if C→E exists)
  C ──→ E

  Every directed path respects the ordering

════════════════════════════════════════════════════════════════════════
```

**For AI:** DAGs are arguably the most important graph type in AI:

1. **Computation graphs.** Every feedforward neural network is a DAG. Forward evaluation proceeds in topological order; backpropagation proceeds in reverse topological order.

2. **Bayesian networks.** A Bayesian network is a DAG where vertices represent random variables and edges represent conditional dependencies. The joint distribution factorises according to the DAG structure: $p(x_1, \ldots, x_n) = \prod_i p(x_i \mid \text{parents}(x_i))$.

3. **Causal graphs.** Structural causal models (Pearl, 2009) represent causal relationships as DAGs. The direction of edges encodes causal direction: $A \to B$ means "$A$ causes $B$."

4. **Task scheduling.** Build systems (Make, Bazel), data pipelines (Airflow), and training pipelines all model dependencies as DAGs.

**Topological sort algorithm (Kahn, 1962):**

```text
KAHN'S TOPOLOGICAL SORT
════════════════════════════════════════════════════════════════════════

  Input: DAG G = (V, E)
  1. Compute in-degree of every vertex
  2. Enqueue all vertices with in-degree 0
  3. While queue is non-empty:
       a. Dequeue vertex u, add to output
       b. For each successor v of u:
            reduce in-degree(v) by 1
            if in-degree(v) == 0: enqueue v
  4. If output has fewer than n vertices: G has a cycle (not a DAG)

  Time: O(n + m)     Space: O(n)

════════════════════════════════════════════════════════════════════════
```

**Unique topological ordering.** A DAG has a unique topological ordering if and only if it has a Hamiltonian path — a path visiting all vertices. Equivalently, at every step of Kahn's algorithm, there is exactly one vertex with in-degree 0.

### 6.5 Planar Graphs

**Definition (Planar Graph).** A graph is *planar* if it can be drawn in the plane without edge crossings.

**Theorem (Euler's Formula for Planar Graphs).** For a connected planar graph with $n$ vertices, $m$ edges, and $f$ faces (including the outer face):

$$n - m + f = 2$$

**Proof (by induction on $m$).** Base case: a tree has $n-1$ edges and $f=1$ face (the outer face). So $n - (n-1) + 1 = 2$. $\checkmark$ Inductive step: take a connected planar graph with a cycle. Removing an edge that is on a cycle reduces $m$ by 1 and merges two faces, reducing $f$ by 1. The quantity $n - m + f$ is unchanged. $\square$

**Corollary.** For a simple planar graph with $n \geq 3$: $m \leq 3n - 6$. This follows from Euler's formula plus the fact that each face is bounded by at least 3 edges, and each edge borders at most 2 faces.

**Corollary (bipartite planar).** For a simple bipartite planar graph with $n \geq 3$: $m \leq 2n - 4$. (Each face is bounded by $\geq 4$ edges, since bipartite graphs have no odd cycles.)

**Theorem (Kuratowski, 1930).** A graph is planar if and only if it does not contain a subdivision of $K_5$ or $K_{3,3}$ as a subgraph.

**Examples:**

- $K_4$ is planar (can be drawn as a triangle with a vertex inside). Check: $n=4$, $m=6$, $f=4$. $4 - 6 + 4 = 2$. $\checkmark$
- $K_5$ is NOT planar. Check: $n=5$, $m=10$, but $10 > 3(5)-6 = 9$. Euler's bound violated.
- $K_{3,3}$ is NOT planar. Check: $n=6$, $m=9$, bipartite bound $9 > 2(6)-4 = 8$. Violated.

```text
K₄ PLANAR EMBEDDING
════════════════════════════════════════════════════════════════════════

  1                       Euler check:
  ╱│╲                       n = 4 vertices
 ╱ │ ╲                      m = 6 edges
2──┼──3                     f = 4 faces (3 triangles + outer)
 ╲ │ ╱                      4 - 6 + 4 = 2  ✓
  ╲│╱
   4

════════════════════════════════════════════════════════════════════════
```

**For AI:** Planar graphs arise in geographic information systems (road networks, administrative boundaries), VLSI circuit design (wires must not cross), and mesh-based physics simulations (finite element methods for weather prediction and fluid dynamics). Graph drawing algorithms that minimise edge crossings are used in knowledge graph visualisation. The planarity constraint is also useful as a sparsity prior — planar graphs are sparse ($m \leq 3n-6$), making them computationally tractable for exact GNN inference.

### 6.6 Hypergraphs

**Definition (Hypergraph).** A *hypergraph* is a pair $H = (V, \mathcal{E})$ where $\mathcal{E} \subseteq 2^V$ is a set of **hyperedges** — each hyperedge is a subset of $V$ that may contain more than 2 vertices.

A standard graph is a hypergraph where every hyperedge has exactly 2 vertices.

**Definition ($k$-uniform hypergraph).** A hypergraph where every hyperedge has exactly $k$ vertices. Standard graphs are 2-uniform hypergraphs.

**Example:** In a co-authorship network, a paper with 5 authors creates a 5-element hyperedge connecting all 5 authors simultaneously — this captures the "group" relationship that pairwise edges cannot.

**Bipartite representation.** Any hypergraph $H = (V, \mathcal{E})$ can be represented as a bipartite graph $B = (V \cup \mathcal{E}, \{(v, e) : v \in e\})$ — one part for vertices, one for hyperedges, with edges representing membership. This is the **incidence graph** of the hypergraph and enables standard graph algorithms to operate on hypergraphs.

**For AI:** Hypergraphs naturally model higher-order interactions:
- **Attention mechanisms** compute interactions over sets of tokens — multi-head attention can be viewed as a learned hypergraph where each attention head defines hyperedges
- **Higher-order message passing** in Hypergraph Neural Networks (Feng et al., 2019) propagates messages along hyperedges
- **Set functions** like DeepSets (Zaheer et al., 2017) process unordered sets — each set is a hyperedge

> **Note:** Hypergraph theory is a deep and active research area. This section provides only an introduction. The key takeaway is that standard graphs capture pairwise relationships; hypergraphs capture group relationships.

---

## 7. Subgraphs and Graph Operations

### 7.1 Subgraphs and Induced Subgraphs

**Definition (Subgraph).** A graph $H = (V', E')$ is a *subgraph* of $G = (V, E)$ if $V' \subseteq V$ and $E' \subseteq E$. We write $H \subseteq G$.

**Definition (Induced Subgraph).** Given a vertex subset $S \subseteq V$, the *induced subgraph* $G[S]$ is the subgraph with vertex set $S$ and all edges of $G$ whose both endpoints are in $S$:

$$G[S] = (S, \{e \in E : e \subseteq S\})$$

The key difference: a subgraph can choose any subset of edges, but an induced subgraph must include ALL edges between vertices in $S$.

**Definition (Spanning Subgraph).** A subgraph $H$ is *spanning* if $V' = V$ — it has the same vertex set as $G$ but possibly fewer edges.

```
SUBGRAPH vs. INDUCED SUBGRAPH
════════════════════════════════════════════════════════════════════════

  Original G:           Subgraph H:         Induced G[{A,B,C}]:
  A ─── B               A ─── B             A ─── B
  │╲    │               │                   │╲
  │  ╲  │               │                   │  ╲
  D ─── C               D     C             C
                                             (includes edge A-C
  All edges present      Missing edges OK     because both endpoints
                                              are in {A,B,C})

════════════════════════════════════════════════════════════════════════
```

**For AI:** Mini-batch training for GNNs works by sampling induced subgraphs. GraphSAGE (Hamilton et al., 2017) samples $k$-hop neighbourhoods as induced subgraphs. The choice between subgraph and induced subgraph affects whether all edges within the sample are preserved — induced subgraphs preserve all local structure.

### 7.2 Graph Complement

**Definition (Graph Complement).** The *complement* $\bar{G}$ of a simple graph $G = (V, E)$ is the graph on the same vertex set with complementary edge set:

$$\bar{G} = (V, \binom{V}{2} \setminus E)$$

Two vertices are adjacent in $\bar{G}$ if and only if they are NOT adjacent in $G$.

**Properties:**
- $|E(\bar{G})| = \binom{n}{2} - |E(G)|$
- $G$ and $\bar{G}$ together form $K_n$: $E(G) \cup E(\bar{G}) = E(K_n)$
- $\overline{\bar{G}} = G$ (complement of complement is the original)
- $\overline{K_n} = \bar{K}_n$ (complement of complete graph is empty graph)
- A graph is **self-complementary** if $G \cong \bar{G}$ (e.g., $P_4$, $C_5$)

**For AI:** Complement graphs appear in constraint satisfaction. If $G$ represents conflicts (vertices that cannot be assigned the same resource), then $\bar{G}$ represents compatibility. Finding a maximum clique in $G$ is equivalent to finding a maximum independent set in $\bar{G}$.

### 7.3 Graph Union, Intersection, and Product

**Definition (Graph Union).** For graphs $G_1 = (V_1, E_1)$ and $G_2 = (V_2, E_2)$:

$$G_1 \cup G_2 = (V_1 \cup V_2, E_1 \cup E_2)$$

**Definition (Graph Intersection).** $G_1 \cap G_2 = (V_1 \cap V_2, E_1 \cap E_2)$.

**Definition (Cartesian Product $G_1 \square G_2$).** Vertex set $V_1 \times V_2$. Two vertices $(u_1, u_2)$ and $(v_1, v_2)$ are adjacent if:

- $u_1 = v_1$ and $\{u_2, v_2\} \in E_2$, or
- $u_2 = v_2$ and $\{u_1, v_1\} \in E_1$

**Example:** $P_2 \square P_3$ produces a $2 \times 3$ grid graph. The hypercube $Q_n = K_2 \square K_2 \square \cdots \square K_2$ ($n$ times) has $2^n$ vertices and $n$-bit binary strings as vertex labels.

```text
CARTESIAN PRODUCT P₂ □ P₃  (2×3 grid)
════════════════════════════════════════════════════════════════════════

  P₂: A─B      P₃: 1─2─3

  Product vertices: (A,1), (A,2), (A,3), (B,1), (B,2), (B,3)

  (A,1)─(A,2)─(A,3)    ← copies of P₃ for each vertex of P₂
    │     │     │
  (B,1)─(B,2)─(B,3)    ← copies of P₂ for each vertex of P₃

  Result: 6 vertices, 7 edges (a 2×3 grid)

════════════════════════════════════════════════════════════════════════
```

**Properties of graph products:**

| Product | Symbol | Adjacency condition | $\lvert V \rvert$ | $\lvert E \rvert$ |
|---------|--------|--------------------|-------------------|-------------------|
| Cartesian | $G \square H$ | Same in one coord, adjacent in other | $nm$ | $m_H n + m_G m$ |
| Tensor | $G \times H$ | Adjacent in both coords | $nm$ | $2m_G m_H$ |
| Strong | $G \boxtimes H$ | Cartesian OR tensor | $nm$ | $m_H n + m_G m + 2m_G m_H$ |
| Lexicographic | $G[H]$ | Adjacent in $G$, OR same in $G$ and adjacent in $H$ | $nm$ | $m_G n^2 + n_G m_H$ |

where $n = \lvert V_G \rvert$, $m = \lvert V_H \rvert$, $m_G = \lvert E_G \rvert$, $m_H = \lvert E_H \rvert$.

**Definition (Tensor (Categorical) Product $G_1 \times G_2$).** Vertex set $V_1 \times V_2$. Two vertices $(u_1, u_2)$ and $(v_1, v_2)$ are adjacent if $\{u_1, v_1\} \in E_1$ AND $\{u_2, v_2\} \in E_2$.

**For AI:** Graph products appear in constructing higher-order graph structures. The tensor product of a graph with itself encodes 2-walk patterns: $(A \otimes A)_{(i,j),(k,l)} = A_{ik} \cdot A_{jl}$, counting simultaneous walks. In the theory of GNN expressiveness, the $k$-WL test colours $k$-tuples of vertices — equivalent to operating on the $k$-fold tensor product $G^{\times k}$. Grid graphs (Cartesian products of paths) arise in image processing (pixels as vertices) and in mesh-based physics simulations.

### 7.4 Graph Minor and Subdivision

**Definition (Subdivision).** A *subdivision* of a graph $G$ is obtained by inserting new vertices into edges — replacing edge $\{u,v\}$ with a path $u{-}w_1{-}w_2{-}\cdots{-}v$.

**Definition (Graph Minor).** A graph $H$ is a *minor* of $G$ if $H$ can be obtained from $G$ by a sequence of vertex deletions, edge deletions, and edge contractions. Edge contraction merges two adjacent vertices into one.

**Theorem (Robertson-Seymour Graph Minor Theorem, 2004).** In any infinite sequence of finite graphs, one is a minor of another. Equivalently, every minor-closed graph property can be characterised by a finite set of forbidden minors.

This is one of the deepest results in graph theory, proved over 20 papers spanning 1983-2004. Kuratowski's planarity theorem is a special case: planarity is characterised by two forbidden minors ($K_5$ and $K_{3,3}$).

**For AI:** Graph minor theory is primarily of theoretical interest in AI, but it provides the foundation for treewidth — a measure of how "tree-like" a graph is. Many NP-hard graph problems become polynomial-time on graphs of bounded treewidth, which is exploited in exact inference for probabilistic graphical models.

### 7.5 Line Graphs

**Definition (Line Graph $L(G)$).** The *line graph* $L(G)$ of a graph $G$ has one vertex for each edge of $G$. Two vertices in $L(G)$ are adjacent if the corresponding edges in $G$ share an endpoint.

$$V(L(G)) = E(G), \quad \{e_1, e_2\} \in E(L(G)) \iff e_1 \cap e_2 \neq \emptyset$$

**Example:** If $G$ is the path $1{-}2{-}3{-}4$ with edges $a = \{1,2\}$, $b = \{2,3\}$, $c = \{3,4\}$, then $L(G)$ has vertices $\{a,b,c\}$ with edges $\{a,b\}$ and $\{b,c\}$ — another path!

**Properties:**
- $|V(L(G))| = |E(G)|$
- $\deg_{L(G)}(e) = \deg_G(u) + \deg_G(v) - 2$ where $e = \{u,v\}$
- $L(K_n)$ is the Kneser graph complement / triangular graph with $\binom{n}{2}$ vertices

**For AI:** Line graphs convert edge-level problems into vertex-level problems. Since most GNN frameworks operate on vertices, transforming a graph to its line graph enables edge-centric tasks (link prediction, edge classification) to be solved with standard vertex-classification GNNs.

---

## 8. Graph Isomorphism and Invariants

### 8.1 Graph Isomorphism

**Definition (Graph Isomorphism).** Two graphs $G_1 = (V_1, E_1)$ and $G_2 = (V_2, E_2)$ are *isomorphic*, written $G_1 \cong G_2$, if there exists a bijection $\phi: V_1 \to V_2$ such that:

$$\{u, v\} \in E_1 \iff \{\phi(u), \phi(v)\} \in E_2$$

The bijection $\phi$ is called an *isomorphism*. Intuitively, two graphs are isomorphic if they have the same structure — the same connectivity pattern — just with different vertex labels.

```
GRAPH ISOMORPHISM
════════════════════════════════════════════════════════════════════════

  G₁:             G₂:              Isomorphism φ:
  A ─── B         1 ─── 3          A ↦ 1
  │     │         │     │          B ↦ 3
  │     │         │     │          C ↦ 4
  D ─── C         2 ─── 4          D ↦ 2

  Same structure (cycle C₄), different labels

════════════════════════════════════════════════════════════════════════
```

**Verifying an isomorphism — worked example.**

```text
G₁:           G₂:           Claimed isomorphism φ:
A ─── B        1 ─── 3        A ↦ 1,  B ↦ 3
│     │        │     │        C ↦ 4,  D ↦ 2
│     │        │     │
D ─── C        2 ─── 4
```

Verify: for each edge in $G_1$, check that its image is an edge in $G_2$:

| Edge in $G_1$ | Image under $\phi$ | Edge in $G_2$? |
|--------------|-------------------|----------------|
| $\{A,B\}$ | $\{\phi(A),\phi(B)\} = \{1,3\}$ | Yes |
| $\{B,C\}$ | $\{3,4\}$ | Yes |
| $\{C,D\}$ | $\{4,2\}$ | Yes |
| $\{D,A\}$ | $\{2,1\}$ | Yes |

All 4 edges map correctly and $\phi$ is a bijection. Therefore $G_1 \cong G_2$. Both are $C_4$.

**Proving non-isomorphism.** To show $G_1 \not\cong G_2$, it suffices to find any graph invariant that differs. The quickest method: check the degree sequence. If degree sequences differ, the graphs cannot be isomorphic. If they agree, check further invariants (girth, number of triangles, spectrum).

**Computational complexity.** The graph isomorphism problem is one of the few natural problems in NP that is neither known to be in P nor known to be NP-complete. Babai (2016) showed it can be solved in quasi-polynomial time $\exp(O(\log^c n))$, but no polynomial-time algorithm is known.

**For AI:** Graph isomorphism is central to GNN theory. A GNN that is **invariant** to vertex permutation produces the same output for isomorphic graphs. This is a design requirement: the molecular property of caffeine should not depend on how we number its atoms. Formally, if $f$ is a GNN and $\Pi$ is a permutation matrix, then $f(\Pi A \Pi^\top, \Pi X) = f(A, X)$. **Equivariance** is a related concept: $f(\Pi A \Pi^\top, \Pi X) = \Pi f(A, X)$. Invariance is needed for graph-level tasks; equivariance is needed for node-level tasks (the output at each node should transform consistently with input permutations).

### 8.2 Graph Invariants

A **graph invariant** is a property or quantity that is the same for all isomorphic graphs. Graph invariants provide necessary (but generally not sufficient) conditions for isomorphism.

| Invariant | Description | Distinguishes |
|-----------|-------------|--------------|
| Number of vertices $n$ | $\lvert V \rvert$ | Graphs of different order |
| Number of edges $m$ | $\lvert E \rvert$ | Graphs with different size |
| Degree sequence | Sorted list of degrees | Many non-isomorphic pairs |
| Diameter | Max distance | Some pairs with same degree sequence |
| Girth | Shortest cycle length | Some pairs with same degree/diameter |
| Spectrum | Eigenvalues of $A$ | Most pairs (but not all!) |
| Chromatic number | Min colors for proper coloring | Some pairs |
| Number of triangles | $\operatorname{tr}(A^3)/6$ | Some pairs |

**Non-isomorphic graphs with the same spectrum (cospectral graphs):** There exist non-isomorphic graphs with identical eigenvalue spectra. The smallest example has 6 vertices. This means the spectrum alone does not determine the graph up to isomorphism.

**For AI:** When a GNN computes a graph-level embedding, it is computing a learned graph invariant. The expressiveness question is: which non-isomorphic graphs can the GNN distinguish? This leads directly to the Weisfeiler-Leman connection.

### 8.3 The Weisfeiler-Leman Test

The **Weisfeiler-Leman (WL) test** is a classical algorithm for testing graph isomorphism. It is not always correct (it can fail to distinguish certain non-isomorphic graphs), but it is the theoretical benchmark for GNN expressiveness.

**Algorithm (1-WL / Color Refinement):**

1. **Initialise:** Assign every vertex the same colour $c_0(v) = 1$.
2. **Iterate:** For each vertex, compute a new colour based on its current colour and the **multiset** of its neighbours' colours:
$$c_{t+1}(v) = \operatorname{HASH}\left(c_t(v), \{\!\{c_t(u) : u \in \mathcal{N}(v)\}\!\}\right)$$
3. **Terminate:** When the colour partition stabilises (no further refinement occurs).
4. **Decide:** If the multisets of colours differ between $G_1$ and $G_2$, they are NOT isomorphic. If they are the same, the test is inconclusive.

Here $\{\!\{\cdot\}\!\}$ denotes a multiset (a set with multiplicities).

```
1-WL COLOR REFINEMENT EXAMPLE
════════════════════════════════════════════════════════════════════════

  Graph G:                  Iteration 0:    Iteration 1:
  A ─── B                   All vertices    A: (1, {1,1}) → colour 2
  │     │                   colour 1        B: (1, {1,1}) → colour 2
  │     │                                   C: (1, {1,1}) → colour 2
  D ─── C                                   D: (1, {1,1}) → colour 2

  All vertices get the same colour → 1-WL cannot distinguish
  vertices in a regular graph after iteration 1 (for C₄)

════════════════════════════════════════════════════════════════════════
```

**Theorem (Xu et al., 2019; Morris et al., 2019).** Message-passing GNNs are AT MOST as powerful as the 1-WL test. That is, if 1-WL cannot distinguish two graphs, no message-passing GNN can distinguish them either.

Conversely, there exist GNN architectures (GIN — Graph Isomorphism Network, Xu et al., 2019) that are exactly as powerful as 1-WL.

**The $k$-WL hierarchy.** The $k$-dimensional WL test ($k$-WL) colours $k$-tuples of vertices rather than individual vertices. Higher $k$ gives strictly more distinguishing power:

$$1\text{-WL} < 2\text{-WL} < 3\text{-WL} < \cdots$$

Higher-order GNNs (Morris et al., 2019) achieve $k$-WL power but at $O(n^k)$ computational cost.

**For AI:** The WL-GNN equivalence is one of the most important theoretical results in graph learning. It tells us:

- Message-passing GNNs have a fundamental expressiveness ceiling
- Some graph structures (e.g., distinguishing regular graphs of the same degree) are provably beyond standard GNNs
- To exceed 1-WL power, we need architectural innovations: higher-order message passing, random features, subgraph counting, or graph transformers

**What 1-WL cannot distinguish — a concrete example.**

```text
Graph G₁ (two triangles sharing no vertex):    Graph G₂ (one 6-cycle):

  1 ─ 2       4 ─ 5                             1 ─ 2
  │   │       │   │                             │   │
  3 ──┘       6 ──┘                             6   3
                                                │   │
                                                5 ─ 4
```

Both graphs: 6 vertices, 6 edges, all degrees = 2 (2-regular). The 1-WL test assigns the same colour to all vertices in both graphs at every iteration — it cannot distinguish them. But they are NOT isomorphic: $G_1$ has two components ($C_3 \cup C_3$) while $G_2$ is connected ($C_6$). A standard message-passing GNN also cannot distinguish these two graphs!

**The Graph Isomorphism Network (GIN, Xu et al., 2019).** GIN achieves 1-WL expressiveness by using the aggregation:

$$\mathbf{h}_v^{(k)} = \operatorname{MLP}^{(k)}\!\left((1 + \varepsilon^{(k)}) \cdot \mathbf{h}_v^{(k-1)} + \sum_{u \in \mathcal{N}(v)} \mathbf{h}_u^{(k-1)}\right)$$

The sum aggregator (not mean or max) is critical — it preserves multiset information about neighbour counts, which is equivalent to the multiset hashing in 1-WL.

### 8.4 Graph Automorphisms

**Definition (Automorphism).** An *automorphism* of a graph $G$ is an isomorphism from $G$ to itself: a bijection $\phi: V \to V$ that preserves adjacency.

**Definition (Automorphism Group).** The set of all automorphisms of $G$ forms a group under composition, denoted $\operatorname{Aut}(G)$.

**Examples:**
- $\operatorname{Aut}(K_n) = S_n$ (the symmetric group — any permutation preserves complete adjacency)
- $\operatorname{Aut}(C_n) = D_n$ (the dihedral group — rotations and reflections of the cycle)
- $\operatorname{Aut}(P_n) = \mathbb{Z}_2$ for $n \geq 2$ (only the identity and reversal)

**Definition (Orbit).** The *orbit* of a vertex $v$ under $\operatorname{Aut}(G)$ is $\{\phi(v) : \phi \in \operatorname{Aut}(G)\}$ — the set of vertices that $v$ can be mapped to. Vertices in the same orbit are "structurally equivalent."

**For AI:** Automorphisms represent graph symmetries. A GNN that is permutation-equivariant respects all graph automorphisms: if two vertices are in the same orbit (structurally identical), the GNN assigns them the same representation (in the absence of distinguishing node features). This is why 1-WL (and standard GNNs) assign the same colour to all vertices in a regular graph — they are all in the same automorphism orbit.

### 8.5 Graph Motifs

**Definition (Subgraph Motif).** A *graph motif* is a small, recurring subgraph pattern that appears in a network significantly more often than in a random graph with the same degree sequence. Motifs are the structural "building blocks" of complex networks.

**The canonical small motifs:**

```text
Triangle (K_3)     3-Path (P_3)      4-Cycle (C_4)     Star (S_3)
  1 -- 2             1 - 2 - 3         1 - 2             1
  |  / |                               |   |            /|\
  | /  |                               4 - 3           2 3 4
  3
```

| Motif | Vertices | Edges | Significance |
|-------|----------|-------|-------------|
| Triangle $K_3$ | 3 | 3 | Social triadic closure; clustering coefficient |
| 3-path $P_3$ | 3 | 2 | Shortest paths; betweenness centrality |
| 4-cycle $C_4$ | 4 | 4 | Bipartite structure; absent in trees |
| Star $S_k$ | $k+1$ | $k$ | Hub nodes; scale-free networks |
| Wedge (open triangle) | 3 | 2 | Potential triangle; closure probability |
| 4-clique $K_4$ | 4 | 6 | Dense community core |

**Triangle counting.** The number of triangles in $G$ is:

$$\Delta(G) = \frac{1}{6}\operatorname{tr}(A^3)$$

since $(A^3)_{ii}$ counts closed walks of length 3 starting and ending at vertex $i$, and each triangle is counted twice per vertex (clockwise and counterclockwise), hence the factor $1/6$.

**Local clustering coefficient.** The clustering coefficient of vertex $v$ measures the fraction of $v$'s neighbour pairs that are themselves connected:

$$C(v) = \frac{|\{e_{uw} : u,w \in N(v),\ e_{uw} \in E\}|}{\binom{\deg(v)}{2}}$$

When $\deg(v) \leq 1$, define $C(v) = 0$.  The *global clustering coefficient* (transitivity) is:

$$C_{\text{global}} = \frac{3 \times \text{number of triangles}}{\text{number of connected triples}} = \frac{\operatorname{tr}(A^3)}{\sum_{i}(A^2)_{ii}(\deg(v_i)-1)/\ldots}$$

(A more convenient form: $C_{\text{global}} = \operatorname{tr}(A^3) / \mathbf{1}^\top (A^2 - \operatorname{diag}(A^2))\mathbf{1}$.)

**Motif frequency vector as a graph fingerprint.** The counts of each motif type in a graph form a *motif frequency vector* $\mathbf{m}(G) \in \mathbb{Z}^r$. These vectors:

- Are invariant to vertex relabelling (graph invariants).
- Capture structural information beyond degree sequences.
- Can distinguish many non-isomorphic graphs that stumped simpler invariants.

**For AI:** Motif counting is directly connected to GNN expressiveness. Standard message-passing GNNs (1-WL power) can count *stars* around each node but *cannot* count triangles or 4-cycles in the local neighbourhood without additional architecture. This was a key finding in Bouritsas et al. (2022) *"Improving Graph Neural Network Expressivity via Subgraph Isomorphism Counting"*, which proposed encoding motif counts as additional node features to boost GNN expressiveness beyond 1-WL. Similarly, the GraphSAGE and GAT architectures implicitly learn to weight neighbour aggregations, but are blind to whether those neighbours are mutually connected (triangle detection requires at least a 2-hop view).

> **Practical implication.** When designing a GNN for tasks where triangles matter (social network communities, protein binding sites, knowledge graph triples), consider (a) adding pre-computed triangle counts as node features, (b) using a higher-order GNN (k-WL, $k \geq 2$), or (c) using a subgraph GNN that routes information through induced subgraphs.

---

## 9. Graph Coloring

### 9.1 Vertex Coloring and Chromatic Number

**Definition (Proper Coloring).** A *proper vertex coloring* of a graph $G$ is a function $c: V \to \{1, 2, \ldots, k\}$ such that $c(u) \neq c(v)$ whenever $\{u, v\} \in E$. Adjacent vertices must receive different colours.

**Definition (Chromatic Number).** The *chromatic number* $\chi(G)$ is the minimum number of colours needed for a proper colouring of $G$.

**Computing $\chi(G)$:** Determining the chromatic number is NP-hard in general. For specific graph families:

| Graph | $\chi(G)$ | Reason |
|-------|----------|--------|
| $K_n$ | $n$ | Every pair adjacent — all need different colours |
| $\bar{K}_n$ (empty) | $1$ | No adjacencies — one colour suffices |
| $C_n$ ($n$ even) | $2$ | Bipartite — alternate two colours |
| $C_n$ ($n$ odd) | $3$ | Not bipartite — need one extra colour |
| Tree | $2$ | Bipartite (no odd cycles) |
| Petersen graph | $3$ | 3-regular, girth 5 |
| Planar graph | $\leq 4$ | Four colour theorem |

**Greedy Coloring Algorithm.** Process vertices in some order. Assign each vertex the smallest colour not used by any of its already-coloured neighbours. This uses at most $\Delta(G) + 1$ colours, where $\Delta(G) = \max_v \deg(v)$.

```text
GREEDY COLORING EXAMPLE (path 1─2─3─4─5)
════════════════════════════════════════════════════════════════════════

  Process in order 1,2,3,4,5:
  Vertex 1: neighbours = {}        → colour 1    (first available)
  Vertex 2: neighbours = {1:red}   → colour 2    (red used)
  Vertex 3: neighbours = {2:blue}  → colour 1    (blue used, red free)
  Vertex 4: neighbours = {3:red}   → colour 2    (red used)
  Vertex 5: neighbours = {4:blue}  → colour 1    (blue used, red free)

  Result: 2 colours (optimal — path is bipartite)

════════════════════════════════════════════════════════════════════════
```

**Lower bounds on $\chi(G)$:**

- **Clique number:** $\chi(G) \geq \omega(G)$ where $\omega(G)$ is the size of the largest clique. (A clique requires all-different colours.)
- **Fractional chromatic number:** $\chi(G) \geq \chi_f(G)$ where $\chi_f$ is a relaxation that can be computed in polynomial time.
- **Spectrum:** $\chi(G) \geq 1 - \lambda_{\max}/\lambda_{\min}$ where $\lambda_{\max}, \lambda_{\min}$ are the largest and smallest eigenvalues of $A$.

**Theorem (Brooks, 1941).** For any connected graph $G$ that is not a complete graph or an odd cycle: $\chi(G) \leq \Delta(G)$.

**For AI:** Graph coloring is one of the foundational constraint satisfaction problems (CSPs). Many scheduling, timetabling, and resource allocation problems reduce to graph coloring: if two tasks conflict (share a resource), they must receive different "colours" (time slots). Solving CSPs with neural approaches (e.g., reinforcement learning for combinatorial optimization) often operates on the conflict graph. The clique-cover lower bound $\chi(G) \geq \omega(G)$ connects coloring to clique detection, which is used in feature selection (finding maximally correlated feature groups).

### 9.2 Chromatic Polynomial

**Definition (Chromatic Polynomial).** The *chromatic polynomial* $P(G, k)$ counts the number of proper colourings of $G$ using at most $k$ colours.

**Theorem (Deletion-Contraction).** For any edge $e = \{u,v\}$:

$$P(G, k) = P(G - e, k) - P(G / e, k)$$

where $G - e$ is $G$ with edge $e$ deleted and $G / e$ is $G$ with edge $e$ contracted.

**Examples:**
- $P(K_n, k) = k(k-1)(k-2)\cdots(k-n+1) = k^{(n)}$ (falling factorial)
- $P(\bar{K}_n, k) = k^n$ (no constraints — each vertex independently chooses from $k$ colours)
- $P(T, k) = k(k-1)^{n-1}$ for any tree $T$ on $n$ vertices

**Note:** $\chi(G) = \min\{k : P(G, k) > 0\}$ — the chromatic number is the smallest $k$ for which the chromatic polynomial is positive.

**Worked example — computing $P(C_3, k)$ by deletion-contraction.**

$C_3$ is the triangle graph with vertices $\{1,2,3\}$ and edges $e_{12}, e_{23}, e_{13}$.

*Step 1.* Delete edge $e_{12}$: the resulting graph $C_3 - e_{12}$ is the path $P_3$ (a tree on 3 vertices).

$$P(P_3, k) = k(k-1)^2$$

*Step 2.* Contract edge $e_{12}$: vertices 1 and 2 merge into a single vertex $v^*$, giving $K_2$ (a single edge $\{v^*, 3\}$).

$$P(K_2, k) = k(k-1)$$

*Step 3.* Apply deletion-contraction:

$$P(C_3, k) = P(P_3, k) - P(K_2, k) = k(k-1)^2 - k(k-1) = k(k-1)(k-2)$$

**Verification.** For $k = 3$: $3 \cdot 2 \cdot 1 = 6$ proper 3-colourings of a triangle. Each of the $3! = 6$ permutations of colours gives a valid colouring, confirming the answer.

**Verification.** For $k = 2$: $2 \cdot 1 \cdot 0 = 0$. A triangle cannot be 2-coloured (it contains an odd cycle), consistent with $\chi(C_3) = 3$.

**Coefficients encode structure.** The chromatic polynomial $P(G,k)$ is always a monic polynomial in $k$ of degree $n = |V|$ with integer coefficients. The coefficient of $k^{n-1}$ equals $-|E|$, and the signs alternate. Reading off the polynomial immediately reveals: $\chi(G)$ (smallest positive root), the number of edges, and whether $G$ is bipartite (all positive coefficients $\iff$ bipartite).

| Graph | $P(G, k)$ | $\chi$ |
|-------|-----------|--------|
| $K_1$ | $k$ | $1$ |
| $K_2$ | $k(k-1)$ | $2$ |
| $P_3$ | $k(k-1)^2$ | $2$ |
| $C_3$ | $k(k-1)(k-2)$ | $3$ |
| $C_4$ | $(k-1)^4 + (k-1)$ | $2$ |
| $K_3$ | $k(k-1)(k-2)$ | $3$ |
| $K_4$ | $k(k-1)(k-2)(k-3)$ | $4$ |

> **AI connection.** Chromatic polynomials appear in probabilistic graphical models: the partition function of the zero-temperature Potts model on a graph equals $P(G, q)$. Computing $P(G, k)$ is \#P-hard in general, motivating approximate inference methods (belief propagation, variational methods) that are central to Bayesian deep learning.

### 9.3 The Four Color Theorem

**Theorem (Appel and Haken, 1976).** Every planar graph can be properly coloured with at most 4 colours: $\chi(G) \leq 4$ for all planar $G$.

This theorem is historically significant for two reasons:

1. It was the first major mathematical theorem proved with computer assistance — the proof required checking ~1,500 configurations by computer.
2. It resolved a conjecture open since 1852 (124 years!).

The proof was controversial when published because mathematicians could not verify it by hand. A simplified computer-assisted proof was given by Robertson, Sanders, Seymour, and Thomas (1997), and the result has been formally verified in the Coq proof assistant (Gonthier, 2008).

**Proof idea (Kempe chains, 1879 — flawed but instructive).** Kempe attempted to prove the theorem using "Kempe chains" — maximal connected subgraphs in which only two colours appear. His proof had a flaw (found by Heawood in 1890), but Kempe chains remain a key tool in coloring proofs.

**Five Color Theorem (Kempe/Heawood, 1890).** Every planar graph is 5-colorable. This weaker result is provable by hand:

*Proof sketch.* By induction on $n$. A planar graph on $n \geq 6$ vertices has a vertex $v$ of degree $\leq 5$ (since $m \leq 3n - 6$ implies average degree $< 6$). Remove $v$, 5-color the rest by induction, then try to restore $v$. If its $\leq 5$ neighbours use $\leq 4$ distinct colours, $v$ gets the 5th colour. The key case (5 neighbours using all 5 colours) is handled by a Kempe chain argument. $\square$

**Note:** 4 colours are sometimes necessary — $K_4$ is planar and needs 4 colours. But three colours suffice for most "practical" planar graphs (outerplanar graphs, for example, are always 3-colorable).

### 9.4 Coloring in AI

Graph coloring appears in AI and systems in several important contexts:

**Register allocation in compilers.** Variables that are simultaneously "live" conflict and cannot share a register. The conflict graph has variables as vertices and edges between simultaneously-live variables. Coloring this graph with $k$ colours assigns $k$ registers. The LLVM and GCC compilers use graph coloring for register allocation.

**Frequency assignment.** In wireless networks, adjacent cells cannot use the same frequency (interference). The interference graph's chromatic number determines the minimum number of frequencies needed.

**Map colouring / geographic visualisation.** The original motivation for the four-colour theorem: colouring regions of a map so adjacent regions have different colours. This is equivalent to colouring the dual graph (faces $\to$ vertices, shared borders $\to$ edges).

**Constraint satisfaction.** Many AI planning and scheduling problems reduce to graph coloring. Exam scheduling: students taking the same two courses cannot have those exams at the same time. Course conflict graph $\to$ minimum number of time slots $= \chi(G)$.

**GNN feature augmentation.** Adding graph coloring information (e.g., the colour assigned by a greedy algorithm) as node features can improve GNN expressiveness beyond the 1-WL limit, because coloring breaks vertex symmetry.

---

## 10. Preview: Representations, Algorithms, and Spectral Methods

This section provides brief pointers to the remaining subsections in the Graph Theory chapter. Each topic listed below has its own dedicated section with full treatment.

### 10.1 Graph Representations (Preview)

> **Preview: Graph Representations**
>
> Graphs can be stored in memory using several data structures, each with different
> space-time trade-offs. The **adjacency matrix** $A \in \{0,1\}^{n \times n}$ enables
> matrix operations (spectral methods, GCN) but uses $O(n^2)$ space. The **adjacency list**
> uses $O(n + m)$ space and enables fast neighbour enumeration. **CSR format** (Compressed
> Sparse Row) combines the benefits of both for sparse graphs and is the standard in
> large-scale GNN frameworks.
>
> → _Full treatment: [Graph Representations](../02-Graph-Representations/notes.md)_

### 10.2 Graph Algorithms (Preview)

> **Preview: Graph Algorithms**
>
> Classical algorithms compute the properties defined in this section. **BFS** computes
> shortest paths in unweighted graphs and identifies connected components. **DFS** detects
> cycles, finds bridges, and computes strongly connected components. **Dijkstra's algorithm**
> finds shortest paths in weighted graphs. **Minimum spanning tree** algorithms (Prim, Kruskal)
> find the cheapest connected spanning subgraph.
>
> → _Full treatment: [Graph Algorithms](../03-Graph-Algorithms/notes.md)_

### 10.3 Spectral Graph Theory (Preview)

> **Preview: Spectral Graph Theory**
>
> The eigenvalues and eigenvectors of graph matrices (adjacency matrix $A$, Laplacian
> $L = D - A$, normalised Laplacian $L_{\text{sym}} = I - D^{-1/2}AD^{-1/2}$) reveal deep
> structural properties. The number of zero eigenvalues of $L$ equals the number of connected
> components. The Fiedler vector (second-smallest eigenvector of $L$) provides the optimal
> graph bipartition, forming the basis of spectral clustering. The Cheeger inequality
> connects the algebraic spectrum to combinatorial expansion.
>
> → _Full treatment: [Spectral Graph Theory](../04-Spectral-Graph-Theory/notes.md)_

### 10.4 Graph Neural Networks (Preview)

> **Preview: Graph Neural Networks**
>
> GNNs learn vertex and graph-level representations by iterating a message-passing scheme:
> $\mathbf{h}_v^{(k+1)} = \text{UPDATE}(\mathbf{h}_v^{(k)}, \text{AGG}(\{\mathbf{h}_u^{(k)} : u \in \mathcal{N}(v)\}))$.
> The Graph Convolutional Network (GCN, Kipf & Welling, 2017) uses the normalised adjacency
> matrix: $H^{(l+1)} = \sigma(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}H^{(l)}W^{(l)})$.
> Graph Attention Networks (GAT) learn edge weights dynamically. The expressiveness of
> message-passing GNNs is bounded by the 1-WL test (§8.3).
>
> → _Full treatment: [Graph Neural Networks](../05-Graph-Neural-Networks/notes.md)_

### 10.5 Random Graphs (Preview)

> **Preview: Random Graphs**
>
> Random graph models generate graphs probabilistically. The **Erdos-Renyi model** $G(n, p)$
> includes each edge independently with probability $p$, producing Poisson degree distributions.
> The **Barabasi-Albert model** generates scale-free graphs with power-law degree distributions
> via preferential attachment. The **Watts-Strogatz model** produces small-world graphs with
> high clustering and short diameters.
>
> → _Full treatment: [Random Graphs](../06-Random-Graphs/notes.md)_

---

## 11. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
|---|---------|---------------|-----|
| 1 | Confusing "path" and "walk" | A walk allows vertex repetition; a path does not. Claiming a walk is a path overstates the constraint. | Use "walk" for general traversals, "path" only when no vertex is repeated. |
| 2 | Assuming all graphs are simple | Real-world graphs often have self-loops (user follows self) or parallel edges (multiple bus routes). GCN explicitly adds self-loops. | State "simple graph" explicitly when assuming no self-loops or parallel edges. |
| 3 | Forgetting that edges in undirected graphs are unordered | Writing $(u,v)$ for undirected edges suggests direction. $\{u,v\} = \{v,u\}$ but $(u,v) \neq (v,u)$. | Use $\{u,v\}$ for undirected, $(u,v)$ for directed. |
| 4 | Claiming degree sequence determines the graph | Many non-isomorphic graphs share the same degree sequence. Example: $C_6$ and $K_{3,3} \setminus M$ (perfect matching removed). | Degree sequence is a necessary but not sufficient invariant for isomorphism. |
| 5 | Assuming bipartite $\iff$ 2-colorable | These are actually equivalent! The common mistake is failing to recognise this equivalence and testing bipartiteness separately from 2-coloring. | Bipartite $\Leftrightarrow$ no odd cycle $\Leftrightarrow$ $\chi(G) \leq 2$. Use BFS 2-coloring to test bipartiteness. |
| 6 | Confusing "connected" with "strongly connected" for digraphs | A weakly connected digraph may have vertex pairs with no directed path between them. Strong connectivity requires directed paths in both directions. | Always specify "weakly" or "strongly" for directed graphs. |
| 7 | Treating graph complement as "flipping edges" | The complement removes existing edges AND adds all missing edges. It's not "toggle each edge" unless you mean exactly that (which IS the complement). | $\bar{G}$ has edge $\{u,v\}$ iff $G$ does NOT have edge $\{u,v\}$. |
| 8 | Assuming GNNs can distinguish any two non-isomorphic graphs | Message-passing GNNs are bounded by 1-WL. They fail on regular graphs of the same degree and many other cases. | Use 1-WL as the upper bound. For tasks requiring more power, consider higher-order GNNs or graph transformers. |
| 9 | Counting edges in $K_n$ as $n^2$ | $K_n$ has $\binom{n}{2} = n(n-1)/2$ edges, not $n^2$. The adjacency matrix has $n^2$ entries but is symmetric with zero diagonal. | Use the formula $\binom{n}{2}$ for undirected, $n(n-1)$ for directed complete graphs. |
| 10 | Forgetting the handshaking lemma when debugging graph code | If your computed degree sum is odd, the graph is invalid — odd degree sums are impossible. | Always verify $\sum \deg(v) = 2|E|$ as a sanity check after graph construction. |

---

## 12. Exercises

**Exercise 1** ★ — Handshaking Lemma Verification

Construct three specific graphs (your choice of vertex/edge sets) and verify the handshaking lemma $\sum \deg(v) = 2|E|$ for each.
(a) A graph with 5 vertices and 7 edges.
(b) A directed graph with 4 vertices — verify $\sum \deg^+ = \sum \deg^- = |E|$.
(c) A bipartite graph $K_{3,4}$ — compute degrees of both sides.

**Exercise 2** ★ — Degree Sequences and Graphicality

(a) Determine whether each sequence is graphic (can be realised as a simple graph): $(4, 3, 3, 2, 2)$, $(3, 3, 3, 1)$, $(5, 3, 2, 2, 2, 2)$.
(b) For each graphic sequence, construct a graph that realises it.
(c) Implement the Erdos-Gallai test programmatically.

**Exercise 3** ★ — Paths, Cycles, and Distance

Given the Petersen graph (look up its edge list):
(a) Find all shortest paths from vertex 0 to vertex 5.
(b) Compute the diameter and radius.
(c) Verify the girth is 5 by finding a shortest cycle.
(d) Show that the Petersen graph has no Hamiltonian cycle (explain why, or try exhaustive search).

**Exercise 4** ★★ — Tree Equivalences

(a) Prove that if $G$ is connected and has exactly $n - 1$ edges, then $G$ is acyclic.
(b) Prove that in a tree, there is exactly one path between every pair of vertices.
(c) Compute the number of spanning trees of $K_4$ using Kirchhoff's matrix tree theorem ($\det$ of any cofactor of $L$) and verify against Cayley's formula $4^{4-2} = 16$.

**Exercise 5** ★★ — Bipartiteness Testing

(a) Implement a BFS-based algorithm to test whether a graph is bipartite.
(b) If bipartite, output the bipartition $(U, W)$.
(c) If not bipartite, output an odd cycle as a certificate.
(d) Test on: $C_6$ (bipartite), $C_7$ (not bipartite), Petersen (not bipartite), $K_{3,3}$ (bipartite).

**Exercise 6** ★★ — Graph Distance and Eccentricity

(a) Compute the all-pairs shortest path distance matrix for a given graph using BFS.
(b) From the distance matrix, compute eccentricity, diameter, radius, and center.
(c) Verify: for any connected graph, $\operatorname{rad}(G) \leq \operatorname{diam}(G) \leq 2 \cdot \operatorname{rad}(G)$.

**Exercise 7** ★★★ — Weisfeiler-Leman Test and GNN Expressiveness

(a) Implement the 1-WL color refinement algorithm.
(b) Run it on two non-isomorphic 3-regular graphs on 6 vertices. Does 1-WL distinguish them?
(c) Construct a pair of non-isomorphic graphs that 1-WL CANNOT distinguish (hint: use two regular graphs with the same degree sequence and spectrum).
(d) Explain why this means a standard message-passing GNN cannot distinguish them either.

**Exercise 8** ★★★ — Real-World Graph Analysis

Model a real-world system as a graph and analyse it:
(a) Choose a domain: social network (use Zachary's karate club), citation network, molecular graph, or small knowledge graph.
(b) Compute: number of vertices/edges, degree distribution, diameter, connected components, clustering coefficient.
(c) Test whether the graph is bipartite, planar, or has any special structure.
(d) Discuss: what would a 2-layer GNN "see" on this graph? Which pairs of vertices can exchange information?

---

## 13. Why This Matters for AI (2026 Perspective)

| Concept | AI Impact |
|---------|----------|
| Graph definition $G = (V, E)$ | The foundational data structure for GNNs, knowledge graphs, and computation graphs in all major frameworks (PyTorch Geometric, DGL, JAX-based graph libraries) |
| Directed graphs | Computation graphs (autograd DAGs), causal graphs (structural causal models), citation networks, dependency parse trees |
| Weighted graphs | Attention weights in transformers define weighted digraphs; similarity graphs ($k$-NN) use distance weights; Markov chains use transition probabilities |
| Degree and degree distribution | Determines GNN aggregation balance; power-law distributions cause over-squashing; degree-based normalisation ($D^{-1/2}AD^{-1/2}$) is standard in GCN |
| Paths and distance | GNN receptive field is bounded by path length; graph diameter determines minimum GNN depth; shortest-path features improve GNN expressiveness |
| Connectivity | Disconnected components are processed independently by message-passing GNNs; virtual nodes connect components artificially |
| Trees and DAGs | Decision trees (XGBoost), parse trees (NLP), computation graphs (autograd), Bayesian networks, causal DAGs |
| Bipartite graphs | User-item recommendation (matrix factorization), bipartite matching (assignment), entity-relation graphs |
| Graph isomorphism | GNN permutation invariance; graph-level classification requires isomorphism-invariant readout functions |
| Weisfeiler-Leman test | Provable upper bound on message-passing GNN expressiveness (Xu et al., 2019); drives research into more expressive architectures (Graph Transformers, subgraph GNNs) |
| Graph coloring | Scheduling, resource allocation, register allocation in compilers; CSP solving with neural methods |
| Hypergraphs | Higher-order attention (multi-head attention as learned hyperedges); set function learning (DeepSets); group interaction modeling |
| Planar graphs | Geographic ML, mesh-based physics simulations (weather prediction, fluid dynamics with GNNs) |

---

## 14. Conceptual Bridge

### Looking Back

This section builds on the mathematical foundations established in earlier chapters:

- **Sets and functions** (Ch. 01) provide the language for defining vertex sets, edge sets, and the mappings (isomorphisms, colourings) between them
- **Matrix operations** (Ch. 02) connect through the adjacency matrix $A$ — the handshaking lemma is $\mathbf{1}^\top A \mathbf{1} = 2|E|$, and walk counting uses matrix powers $A^k$
- **Eigenvalues** (Ch. 03) will become central in the next section — the spectrum of $A$ and $L$ encodes deep structural information about the graph

### Looking Forward

The vocabulary and theory developed here is the foundation for the rest of the Graph Theory chapter:

- **§02 Graph Representations** takes the abstract objects defined here (adjacency, degree, connectivity) and asks: how do we store them in memory?
- **§03 Graph Algorithms** provides efficient algorithms for computing the properties defined here: BFS for distance, DFS for connectivity, Dijkstra for weighted shortest paths
- **§04 Spectral Graph Theory** analyses the eigenvalues of $A$, $D$, and $L$ — connecting the combinatorial properties (connectivity, bipartiteness, clustering) to algebraic properties (spectrum)
- **§05 Graph Neural Networks** builds learnable functions on graphs, with the message-passing paradigm directly implementing walk-based aggregation on the structures defined here
- **§06 Random Graphs** asks: what happens when edges are drawn randomly? The degree sequences, connectivity thresholds, and component structure we defined here become probabilistic phenomena

### The Big Picture

```
GRAPH BASICS IN THE CURRICULUM
════════════════════════════════════════════════════════════════════════

  Ch.01 Sets & Logic ──────────→ Vertex sets, edge sets, mappings
           │
  Ch.02 Linear Algebra ────────→ Adjacency matrix, degree matrix
           │
           ▼
  ┌─────────────────────────────────────────────────────────┐
  │                  §01 GRAPH BASICS                        │
  │   Definitions · Degree · Paths · Connectivity · Trees    │
  │   Bipartite · Coloring · Isomorphism · WL Test           │
  └───────┬───────────┬───────────┬───────────┬─────────────┘
          │           │           │           │
          ▼           ▼           ▼           ▼
       §02          §03        §04         §05
    Represent.   Algorithms   Spectral     GNNs
    (storage)   (BFS, DFS)  (eigenvals)  (learning)
                                           │
                                           ▼
                                    §06 Random Graphs
                                    (probabilistic)

════════════════════════════════════════════════════════════════════════
```

---

## 15. Quick Computational Reference

The table below collects the key formulas and algorithms from this section for fast lookup during implementation.

### 15.1 Degree and Edge Counting

| Quantity | Formula | Python (NetworkX) |
|----------|---------|-------------------|
| Total edges | $m = \tfrac{1}{2}\sum_v \deg(v)$ | `G.number_of_edges()` |
| Max degree | $\Delta(G) = \max_v \deg(v)$ | `max(d for _,d in G.degree())` |
| Min degree | $\delta(G) = \min_v \deg(v)$ | `min(d for _,d in G.degree())` |
| Average degree | $\bar{d} = 2m/n$ | `2*G.number_of_edges()/G.number_of_nodes()` |
| Degree sequence | Sorted list of degrees | `sorted([d for _,d in G.degree()], reverse=True)` |

### 15.2 Adjacency Matrix Operations

| Operation | Matrix expression | Interpretation |
|-----------|------------------|----------------|
| Degree of $v_i$ | $(A\mathbf{1})_i$ | Row sum |
| Number of walks of length $k$ | $(A^k)_{ij}$ | Matrix power |
| Number of triangles | $\tfrac{1}{6}\operatorname{tr}(A^3)$ | Trace of cube |
| Graph Laplacian | $L = D - A$ | $D = \operatorname{diag}(A\mathbf{1})$ |
| Normalised Laplacian | $L_{\text{sym}} = I - D^{-1/2}AD^{-1/2}$ | Eigenvalues in $[0, 2]$ |

### 15.3 Graph Properties — Decision Table

| Property | Algorithm | Time complexity |
|----------|-----------|----------------|
| Connectivity | BFS/DFS from any vertex | $O(n + m)$ |
| Strong connectivity | Kosaraju or Tarjan | $O(n + m)$ |
| Bipartiteness | BFS 2-colouring | $O(n + m)$ |
| Euler circuit | Check all degrees even + connected | $O(n + m)$ |
| Spanning tree | BFS/DFS tree | $O(n + m)$ |
| Minimum spanning tree | Kruskal or Prim | $O(m \log n)$ |
| Shortest path (unweighted) | BFS | $O(n + m)$ |
| Shortest path (weighted) | Dijkstra (non-negative) | $O((n+m)\log n)$ |
| Topological sort (DAG) | Kahn's algorithm | $O(n + m)$ |
| Planarity test | Boyer-Myrvold | $O(n)$ |

### 15.4 Graph Invariant Checklist

When you need to test whether two graphs $G_1, G_2$ might be isomorphic, check these invariants in order from cheapest to most expensive:

1. $|V|$ equal? (O(1))
2. $|E|$ equal? (O(1))
3. Degree sequences equal? (O(n log n))
4. Degree distribution equal? (O(n))
5. Number of triangles equal? (O(n^3) or O(n^{2.37}) with fast matrix multiply)
6. Characteristic polynomial of $A$ equal? (O(n^3))
7. Chromatic polynomial equal? (#P-hard in general)
8. Run VF2 or Nauty for exact isomorphism check.

If any invariant differs, the graphs are definitively **not** isomorphic. If all agree, they are *likely* isomorphic but not guaranteed (cospectral non-isomorphic graphs exist).

> **Implementation note.** In practice, NetworkX provides `nx.is_isomorphic(G1, G2)` using VF2, and `nx.graph_atlas_g()` for enumeration. For large graphs (\$n > 10^4\$), use approximate fingerprinting (degree sequence + triangle count + eigenvalue moments) before exact checks.

---

## 16. Section Summary

This section covered the full vocabulary of graph theory needed to read modern GNN papers and implement graph algorithms from scratch.

**Core objects defined:**

- A *graph* $G = (V, E)$ is a pair of a vertex set and an edge set; variants include directed, weighted, simple, multi, and hypergraphs.
- The *degree* of a vertex counts its incident edges; the handshaking lemma $\sum \deg(v) = 2|E|$ is the universal sanity check.
- *Walks*, *trails*, *paths*, and *cycles* form a hierarchy of traversal objects; Eulerian and Hamiltonian structures live at opposite ends of algorithmic tractability.

**Core structural theorems:**

- *Connectivity* is characterised by BFS/DFS reachability; bridges and articulation points are the fragile points; Menger's theorem equates disjoint paths with cuts.
- *Trees* admit six equivalent definitions and are the minimum connected structures; spanning trees compress any connected graph.
- *Bipartite* graphs are exactly those with no odd cycles, equivalent to 2-colourability, and support Konig's matching theorem.
- *Planar* graphs satisfy Euler's formula $n - m + f = 2$ and are characterised by Kuratowski's forbidden minors.

**Core AI connections:**

- The *Weisfeiler-Leman test* (1-WL) is the provable upper bound on the expressiveness of all message-passing GNNs, showing that structural indistinguishability in WL implies indistinguishability by GNNs.
- *Graph motifs* (triangles, stars, paths) are the structural units that standard GNNs cannot count, motivating expressiveness research.
- *Graph isomorphism*, *automorphisms*, and *invariants* formalise what it means for a graph function to be permutation-invariant — the foundational requirement for graph-level prediction.

The **theory notebook** works through all matrix computations, proofs, and visualisations in executable Python. The **exercises notebook** provides graded practice from basic degree calculations to proving WL indistinguishability of specific graph pairs.

**The single most important takeaway:** graphs are simultaneously combinatorial objects (studied via degree, paths, cycles), algebraic objects (studied via matrices and polynomials), and computational objects (studied via algorithms and complexity). GNNs sit at the intersection of all three perspectives — they are learned functions that respect the algebraic symmetries of graphs while computing efficiently on the combinatorial structure. Every concept in this section reappears in that context.

---

<!-- End of §01 Graph Basics main content. See theory.ipynb for executable proofs and exercises.ipynb for graded problems. -->

## 17. Further Reading

### Foundational Texts

- **Diestel, R.** *Graph Theory* (5th ed., 2017) — The standard graduate reference. Free PDF at [diestel-graph-theory.com](http://diestel-graph-theory.com). Chapters 1–2 cover the material in this section with full proofs.
- **West, D.** *Introduction to Graph Theory* (2nd ed., 2001) — Excellent undergraduate text with many exercises. Chapters 1 (Fundamental Concepts) and 2 (Trees and Distance) map directly to §01.
- **Bondy, J. A. and Murty, U. S. R.** *Graph Theory* (2008, Springer GTM) — Comprehensive coverage including extremal graph theory and Ramsey theory.

### For the AI/ML Connection

- **Hamilton, W. L.** *Graph Representation Learning* (2020, Synthesis Lectures) — The definitive ML-focused introduction. Chapter 2 covers graph basics from an ML perspective. Free draft at cs.mcgill.ca/~wlh/grl_book/.
- **Xu, K. et al.** "How Powerful are Graph Neural Networks?" *ICLR 2019* — Establishes the 1-WL upper bound on message-passing GNNs. Essential reading before implementing GNNs.
- **Bronstein, M. et al.** "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges" *arXiv 2021* — Places graphs in the broader context of geometric priors in ML (symmetry, equivariance).

### Historical Papers

- **Euler, L.** "Solutio problematis ad geometriam situs pertinentis" (1736) — The Königsberg bridges paper: the first graph theory result.
- **Erdős, P. and Rényi, A.** "On Random Graphs I" *Publicationes Mathematicae* (1959) — Founding paper of random graph theory; introduces the $G(n,p)$ model studied in §06.
- **Appel, K. and Haken, W.** "Every planar map is four colorable" *Illinois J. Math.* (1977) — The first major computer-assisted proof in mathematics.

---

[← Back to Graph Theory](../README.md) | [Next: Graph Representations →](../02-Graph-Representations/notes.md)

---

## Appendix A: Graph Theory Notation Reference

A quick-reference table for the notation used throughout this section and the rest of the chapter.

### A.1 Graph Notation

| Symbol | Meaning |
|--------|---------|
| $G = (V, E)$ | Graph with vertex set $V$ and edge set $E$ |
| $n = \lvert V \rvert$ | Order (number of vertices) |
| $m = \lvert E \rvert$ | Size (number of edges) |
| $u \sim v$ | Vertices $u$ and $v$ are adjacent |
| $\mathcal{N}(v)$ | Neighbourhood of $v$: all vertices adjacent to $v$ |
| $\mathcal{N}[v]$ | Closed neighbourhood: $\mathcal{N}(v) \cup \{v\}$ |
| $\deg(v)$ | Degree of vertex $v$ in an undirected graph |
| $\deg^+(v)$ | Out-degree of $v$ in a digraph |
| $\deg^-(v)$ | In-degree of $v$ in a digraph |
| $\delta(G)$ | Minimum degree: $\min_{v \in V}\deg(v)$ |
| $\Delta(G)$ | Maximum degree: $\max_{v \in V}\deg(v)$ |
| $d(u, v)$ | Graph distance (shortest path length) from $u$ to $v$ |
| $\operatorname{diam}(G)$ | Diameter: $\max_{u,v} d(u,v)$ |
| $\operatorname{rad}(G)$ | Radius: $\min_v \max_u d(u,v)$ |
| $\chi(G)$ | Chromatic number |
| $\kappa(G)$ | Vertex connectivity |
| $\lambda(G)$ | Edge connectivity |
| $G[S]$ | Induced subgraph on vertex subset $S$ |
| $G - v$ | Graph with vertex $v$ and its incident edges removed |
| $G - e$ | Graph with edge $e$ removed |
| $G / e$ | Graph with edge $e$ contracted |
| $\bar{G}$ | Complement of $G$ |
| $L(G)$ | Line graph of $G$ |
| $\operatorname{Aut}(G)$ | Automorphism group of $G$ |

### A.2 Special Graph Families

| Symbol | Name | Definition |
|--------|------|-----------|
| $K_n$ | Complete graph | $n$ vertices, all $\binom{n}{2}$ edges |
| $K_{m,n}$ | Complete bipartite | $m + n$ vertices, $mn$ edges in bipartite form |
| $C_n$ | Cycle | $n$ vertices in a single cycle |
| $P_n$ | Path | $n$ vertices in a single path |
| $\bar{K}_n$ | Empty graph | $n$ vertices, no edges |
| $S_n$ | Star | One center vertex connected to $n-1$ leaves |
| $W_n$ | Wheel | $C_n$ plus one central vertex connected to all |
| $Q_n$ | Hypercube | $2^n$ vertices, $n$-bit binary strings as vertices |

### A.3 Graph Matrices

| Matrix | Symbol | Definition | Size |
|--------|--------|-----------|------|
| Adjacency | $A$ | $A_{ij} = 1$ if $\{i,j\} \in E$ | $n \times n$ |
| Degree | $D$ | $D = \operatorname{diag}(\deg(v_1), \ldots, \deg(v_n))$ | $n \times n$ diagonal |
| Laplacian | $L$ | $L = D - A$ | $n \times n$ |
| Normalised Laplacian | $L_{\text{sym}}$ | $I - D^{-1/2}AD^{-1/2}$ | $n \times n$ |
| Incidence | $B$ | $B_{ve} = 1$ if $v \in e$ (undirected) | $n \times m$ |
| Distance | $\mathcal{D}$ | $\mathcal{D}_{ij} = d(v_i, v_j)$ | $n \times n$ |

---

## Appendix B: Key Theorems Summary

| Theorem | Statement | Significance |
|---------|-----------|-------------|
| Handshaking Lemma | $\sum_{v}\deg(v) = 2\lvert E \rvert$ | Fundamental constraint; useful for validation |
| Euler's Theorem | Eulerian circuit $\iff$ all degrees even | First theorem of graph theory (1736) |
| Odd-Cycle Theorem | Bipartite $\iff$ no odd cycles | Connects structure to coloring |
| Tree Equivalences | 6 equivalent definitions of a tree | Foundational for spanning trees |
| Cayley's Formula | $n^{n-2}$ spanning trees in $K_n$ | Counts labeled trees |
| Matrix Tree Theorem | Spanning trees = any cofactor of $L$ | Connects combinatorics to linear algebra |
| Euler's Planar Formula | $n - m + f = 2$ | Topological constraint on planar graphs |
| Kuratowski's Theorem | Planar $\iff$ no $K_5$ or $K_{3,3}$ subdivision | Characterises planarity |
| Four Color Theorem | $\chi(G) \leq 4$ for planar $G$ | Computer-assisted proof (1976/2008) |
| Brooks' Theorem | $\chi(G) \leq \Delta(G)$ (non-complete, non-odd-cycle) | Upper bound on chromatic number |
| Konig's Theorem | Max matching = min vertex cover (bipartite) | Foundation of bipartite combinatorics |
| Menger's Theorem | Max disjoint paths = min cut | Max-flow min-cut for graphs |
| WL-GNN Equivalence | MPNN power $\leq$ 1-WL test (Xu et al., 2019) | Expressiveness bound for GNNs |
| Robertson-Seymour | Every minor-closed property has finite forbidden set | Deepest result in structural graph theory |

---

## Appendix C: Common Graph Families — Properties at a Glance

| Graph | $n$ | $m$ | Regular? | Bipartite? | Planar? | $\chi$ | Diameter |
|-------|-----|-----|----------|-----------|---------|--------|---------|
| $K_n$ | $n$ | $n(n{-}1)/2$ | $(n{-}1)$-reg | $n \leq 2$ | $n \leq 4$ | $n$ | $1$ |
| $K_{n,n}$ | $2n$ | $n^2$ | $n$-reg | Yes | $n \leq 2$ | $2$ | $2$ |
| $C_n$ | $n$ | $n$ | $2$-reg | $n$ even | Yes | $\lfloor n/2 \rfloor$ | $\lfloor n/2 \rfloor$ |
| $P_n$ | $n$ | $n{-}1$ | No | Yes | Yes | $2$ | $n{-}1$ |
| Tree (any) | $n$ | $n{-}1$ | No (gen.) | Yes | Yes | $2$ | $\leq n{-}1$ |
| Petersen | $10$ | $15$ | $3$-reg | No | No | $3$ | $2$ |
| $Q_n$ (hypercube) | $2^n$ | $n2^{n-1}$ | $n$-reg | Yes | $n \leq 3$ | $2$ | $n$ |

---

