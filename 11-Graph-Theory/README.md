[← Back to Curriculum Home](../README.md) | [Previous Chapter: Numerical Methods ←](../10-Numerical-Methods/README.md) | [Next Chapter: Functional Analysis →](../12-Functional-Analysis/README.md)

---

# Chapter 11 — Graph Theory

> _"Graphs are the natural language of relationships. Every social network, every molecule, every knowledge base, and every computation graph in a neural network is a graph — and the mathematics of graphs is the mathematics of structure itself."_

## Overview

This chapter develops the mathematics of graphs from first principles through to their modern use in deep learning. It follows a deliberate progression: from the concrete vocabulary of vertices and edges (Graph Basics) through computational representations and classical algorithms, into the spectral theory that connects graph structure to linear algebra, and finally to the graph neural network architectures that have become essential in modern AI.

Each subsection is **self-contained** but **designed to be read in order**. Early sections build the combinatorial and algorithmic intuition; later sections leverage linear algebra and spectral theory to reveal deeper structural properties. This mirrors how graphs appear in ML practice: first as data structures, then as mathematical objects with rich algebraic properties.

---

## Subsection Map

| # | Subsection | What It Covers | Canonical Topics |
|---|-----------|----------------|-----------------|
| 01 | [Graph Basics](01-Graph-Basics/notes.md) | Fundamental definitions, graph types, properties, degree sequences, graph isomorphism, subgraphs | Vertices, edges, directed/undirected graphs, weighted graphs, paths, cycles, connectivity, trees |
| 02 | [Graph Representations](02-Graph-Representations/notes.md) | Data structures for storing graphs; space-time trade-offs for different representations | Adjacency matrix, adjacency list, edge list, incidence matrix, sparse representations |
| 03 | [Graph Algorithms](03-Graph-Algorithms/notes.md) | Classical algorithms for traversal, shortest paths, spanning trees, flow, and connectivity | BFS, DFS, Dijkstra, Bellman-Ford, MST (Prim/Kruskal), max-flow, topological sort |
| 04 | [Spectral Graph Theory](04-Spectral-Graph-Theory/notes.md) | Eigenvalue analysis of graph matrices; spectral clustering; Cheeger inequality | Laplacian matrix, graph spectrum, Fiedler vector, spectral clustering, graph wavelets |
| 05 | [Graph Neural Networks](05-Graph-Neural-Networks/notes.md) | Neural architectures for graph-structured data; message passing; attention on graphs | MPNN framework, GCN, GraphSAGE, GAT, graph pooling, over-smoothing, expressiveness |
| 06 | [Random Graphs](06-Random-Graphs/notes.md) | Probabilistic models of graph generation; phase transitions; network properties | Erdos-Renyi model, Watts-Strogatz, Barabasi-Albert, small-world property, power laws |

---

## Reading Order and Dependencies

```
01-Graph-Basics                 (definitions and vocabulary — start here)
        ↓
02-Graph-Representations        (how to store and compute with graphs)
        ↓
03-Graph-Algorithms             (classical algorithms; uses representations from §02)
        ↓
04-Spectral-Graph-Theory        (eigenvalues of graph matrices; needs linear algebra Ch.02–03)
        ↓
05-Graph-Neural-Networks        (deep learning on graphs; builds on §04 spectral theory)
        ↓
06-Random-Graphs                (probabilistic graph models; uses concepts from §01–§04)
        ↓
12-Functional-Analysis          (Hilbert spaces, kernel methods — next chapter)
```

---

## How the Subsections Relate

**01 vs 02:** Subsection 01 defines graphs mathematically — what vertices, edges, paths, and connectivity mean. Subsection 02 asks the computational question: how do we *store* these objects? The adjacency matrix from §02 becomes the central object in §04 (spectral theory) and §05 (GNNs).

**02 vs 04:** The adjacency matrix and Laplacian matrix introduced as data structures in §02 become the subjects of eigenvalue analysis in §04. Understanding *why* we choose matrix representations connects directly to spectral methods.

**03 vs 05:** Classical graph algorithms (BFS, shortest paths) in §03 are the non-learned predecessors of GNN message passing in §05. The k-hop neighborhood computation in BFS is exactly the receptive field of a k-layer GNN.

**04 vs 05:** Spectral graph theory (§04) provides the theoretical foundation for Graph Convolutional Networks (§05). The GCN layer is derived as a first-order Chebyshev approximation of spectral graph convolution — reading §04 first makes the GCN derivation meaningful.

**06 vs 01–04:** Random graph models (§06) use the vocabulary from §01, can be studied through the representations in §02, and their structural properties (connectivity, diameter) connect to the spectral analysis in §04.

---

## What Belongs Where (Canonical Homes)

| Topic | Canonical Home | Preview In |
|-------|---------------|-----------|
| Vertices, edges, paths, cycles, connectivity | §01 | — |
| Directed vs undirected, weighted graphs | §01 | §02 (as representation variants) |
| Adjacency matrix, adjacency list, edge list | §02 | §01 (informal), §04 (spectral view) |
| Incidence matrix, sparse formats | §02 | — |
| BFS, DFS, traversal | §03 | §05 (as GNN analogy) |
| Shortest paths (Dijkstra, Bellman-Ford) | §03 | — |
| Minimum spanning trees | §03 | — |
| Max-flow, min-cut | §03 | §04 (Cheeger inequality connection) |
| Laplacian matrix, degree matrix | §04 | §02 (definition), §05 (GCN normalization) |
| Graph spectrum, Fiedler vector | §04 | — |
| Spectral clustering | §04 | §05 (learned alternative) |
| Message passing, MPNN framework | §05 | — |
| GCN, GAT, GraphSAGE | §05 | §04 (spectral derivation) |
| Over-smoothing, expressiveness (WL test) | §05 | — |
| Erdos-Renyi, small-world, scale-free models | §06 | §01 (graph properties context) |

---

## Prerequisites

Before starting this chapter, you should be comfortable with:

- **Linear algebra** — Matrix operations, eigenvalues, eigenvectors ([Chapter 2](../02-Linear-Algebra-Basics/README.md) and [Chapter 3](../03-Advanced-Linear-Algebra/README.md))
- **Basic probability** — Random variables, expectation, distributions ([Chapter 6](../06-Probability-Theory/README.md))
- **Calculus** — Derivatives and gradients for the GNN sections ([Chapter 4](../04-Calculus-Fundamentals/README.md))
- **Sets and functions** — From [Chapter 1](../01-Mathematical-Foundations/README.md)

Subsections §01–§03 require only Chapter 1–2 prerequisites. Subsections §04–§06 additionally need eigenvalue theory from Chapter 3.

---

## After This Chapter

This chapter prepares you for:

- **[12-Functional-Analysis](../12-Functional-Analysis/README.md)** — Hilbert spaces and kernel methods generalize the spectral ideas here to infinite dimensions
- **[14-Math-for-Specific-Models](../14-Math-for-Specific-Models/README.md)** — GNN architectures connect to broader neural network mathematics
- **[22-Causal-Inference](../22-Causal-Inference/README.md)** — Causal graphs (DAGs) use the directed graph theory from §01 and §03
- **[21-Statistical-Learning-Theory](../21-Statistical-Learning-Theory/README.md)** — Generalization bounds for GNNs build on the expressiveness results in §05

---

[← Back to Curriculum Home](../README.md) | [Previous Chapter: Numerical Methods ←](../10-Numerical-Methods/README.md) | [Next Chapter: Functional Analysis →](../12-Functional-Analysis/README.md)
