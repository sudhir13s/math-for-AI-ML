[← Back to Graph Theory](../README.md) | [Previous: Graph Representations ←](../02-Graph-Representations/notes.md) | [Next: Spectral Graph Theory →](../04-Spectral-Graph-Theory/notes.md)

---

# Graph Algorithms

> _"An algorithm must be seen to be believed."_ — Donald Knuth

## Overview

Graph algorithms are the computational engine of graph theory. A graph $G = (V, E)$ is an abstract object; a graph algorithm is a procedure that extracts structure, distances, connectivity, or optimal subgraphs from that object in time proportional to the graph's size. The classical algorithms developed between the 1950s and 1980s — BFS, DFS, Dijkstra, Bellman-Ford, Kruskal, Prim, Tarjan, Ford-Fulkerson — remain the workhorses of modern systems, from web crawlers to GPS navigation to protein interaction databases.

For AI and machine learning, graph algorithms are more than historical curiosities. Breadth-first search defines the $k$-hop neighbourhood that a $k$-layer Graph Neural Network can see — the BFS frontier *is* the GNN receptive field. Topological sort on directed acyclic graphs is exactly the forward-pass ordering algorithm in PyTorch and JAX autograd engines. Dijkstra's algorithm reappears as approximate inference in probabilistic graphical models. Minimum spanning trees underpin graph-based clustering and hierarchical agglomeration. Max-flow / min-cut formalises graph partitioning and connects — via the Cheeger inequality — to the spectral clustering methods in §04.

This section covers the full classical canon: traversal (BFS, DFS), shortest paths (Dijkstra, Bellman-Ford, Floyd-Warshall, A\*), minimum spanning trees (Kruskal, Prim), DAG algorithms (topological sort, critical path), strongly connected components (Tarjan, Kosaraju), and maximum flow. Every algorithm is stated with precise pseudocode, complexity analysis, a correctness argument, and an explicit connection to AI practice.

## Prerequisites

- Graph definitions ($G=(V,E)$, adjacency, degree, paths, cycles, DAGs) — [§01 Graph Basics](../01-Graph-Basics/notes.md)
- Graph representations (adjacency list, CSR, edge list) — [§02 Graph Representations](../02-Graph-Representations/notes.md)
- Basic asymptotic notation $O, \Theta, \Omega$ — [Chapter 1](../../01-Mathematical-Foundations/README.md)
- Priority queue / heap data structure (assumed from CS background)
- Union-Find (disjoint-set union) data structure (introduced in §4.2 below)

## Companion Notebooks

| Notebook | Description |
|----------|-------------|
| [theory.ipynb](theory.ipynb) | BFS/DFS with layer visualisation, Dijkstra with priority queue trace, Bellman-Ford DP table, Kruskal with Union-Find, Tarjan SCC, max-flow residual graph animation |
| [exercises.ipynb](exercises.ipynb) | 8 graded exercises: BFS shortest paths through max-flow and GPU graph analytics |

## Learning Objectives

After completing this section, you will:

- Implement BFS and DFS from scratch and use them to compute shortest-hop distances, detect cycles, and classify edges
- Prove correctness of Dijkstra's algorithm via the greedy-choice invariant
- Apply Bellman-Ford for graphs with negative weights and detect negative cycles
- Implement Floyd-Warshall for all-pairs shortest paths and recognise its DP structure
- Describe A\* and construct admissible heuristics for graph search problems
- Implement Kruskal's algorithm with Union-Find and Prim's with a priority queue
- Perform topological sort using both Kahn's and DFS-based algorithms
- Find strongly connected components using Tarjan's single-pass DFS algorithm
- State and apply the Max-Flow Min-Cut theorem
- Map each classical algorithm to its role in a modern AI system (GNN, autograd, clustering, planning)
- Select the right algorithm given graph properties, edge-weight constraints, and hardware

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 What Is a Graph Algorithm?](#11-what-is-a-graph-algorithm)
  - [1.2 The Graph Algorithm Zoo](#12-the-graph-algorithm-zoo)
  - [1.3 Why Graph Algorithms Matter for AI](#13-why-graph-algorithms-matter-for-ai)
  - [1.4 Historical Timeline](#14-historical-timeline)
- [2. Graph Traversal](#2-graph-traversal)
  - [2.1 Breadth-First Search (BFS)](#21-breadth-first-search-bfs)
  - [2.2 BFS Correctness and Shortest-Hop Paths](#22-bfs-correctness-and-shortest-hop-paths)
  - [2.3 Depth-First Search (DFS)](#23-depth-first-search-dfs)
  - [2.4 DFS: Edge Classification](#24-dfs-edge-classification)
  - [2.5 BFS vs DFS — When to Use Which](#25-bfs-vs-dfs--when-to-use-which)
  - [2.6 AI Connection: GNN Receptive Fields](#26-ai-connection-gnn-receptive-fields)
- [3. Shortest Paths](#3-shortest-paths)
  - [3.1 The SSSP Problem](#31-the-sssp-problem)
  - [3.2 Dijkstra's Algorithm](#32-dijkstras-algorithm)
  - [3.3 Bellman-Ford Algorithm](#33-bellman-ford-algorithm)
  - [3.4 Floyd-Warshall: All-Pairs Shortest Paths](#34-floyd-warshall-all-pairs-shortest-paths)
  - [3.5 A* Search](#35-a-search)
  - [3.6 Shortest-Path Complexity Comparison](#36-shortest-path-complexity-comparison)
- [4. Minimum Spanning Trees](#4-minimum-spanning-trees)
  - [4.1 The MST Problem and Key Properties](#41-the-mst-problem-and-key-properties)
  - [4.2 Kruskal's Algorithm](#42-kruskals-algorithm)
  - [4.3 Prim's Algorithm](#43-prims-algorithm)
  - [4.4 AI Applications of MSTs](#44-ai-applications-of-msts)
- [5. Topological Sort and DAG Algorithms](#5-topological-sort-and-dag-algorithms)
  - [5.1 Directed Acyclic Graphs (DAGs)](#51-directed-acyclic-graphs-dags)
  - [5.2 Topological Sort](#52-topological-sort)
  - [5.3 Longest and Shortest Paths in DAGs](#53-longest-and-shortest-paths-in-dags)
  - [5.4 AI Connection: Autograd Computation Graphs](#54-ai-connection-autograd-computation-graphs)
- [6. Strongly Connected Components](#6-strongly-connected-components)
  - [6.1 Connectivity in Directed Graphs](#61-connectivity-in-directed-graphs)
  - [6.2 Kosaraju's Algorithm](#62-kosarajus-algorithm)
  - [6.3 Tarjan's Algorithm](#63-tarjans-algorithm)
  - [6.4 AI Applications of SCCs](#64-ai-applications-of-sccs)
- [7. Maximum Flow and Minimum Cut](#7-maximum-flow-and-minimum-cut)
  - [7.1 Flow Networks](#71-flow-networks)
  - [7.2 Ford-Fulkerson and Augmenting Paths](#72-ford-fulkerson-and-augmenting-paths)
  - [7.3 Edmonds-Karp Algorithm](#73-edmonds-karp-algorithm)
  - [7.4 The Max-Flow Min-Cut Theorem](#74-the-max-flow-min-cut-theorem)
  - [7.5 AI Connections: Graph Cuts and Partitioning](#75-ai-connections-graph-cuts-and-partitioning)
- [8. Advanced Topics](#8-advanced-topics)
  - [8.1 Bidirectional Search](#81-bidirectional-search)
  - [8.2 Johnson's Algorithm](#82-johnsons-algorithm)
  - [8.3 Parallel and GPU Graph Algorithms](#83-parallel-and-gpu-graph-algorithms)
  - [8.4 Approximation Algorithms](#84-approximation-algorithms)
- [9. Algorithm Selection Framework](#9-algorithm-selection-framework)
  - [9.1 Decision Flowchart](#91-decision-flowchart)
  - [9.2 Complexity Summary Table](#92-complexity-summary-table)
  - [9.3 Implementation Pitfalls](#93-implementation-pitfalls)
- [10. Why This Matters for AI (2026 Perspective)](#10-why-this-matters-for-ai-2026-perspective)
- [11. Common Mistakes](#11-common-mistakes)
- [12. Exercises](#12-exercises)
- [13. Conceptual Bridge](#13-conceptual-bridge)

---

## 1. Intuition

### 1.1 What Is a Graph Algorithm?

A graph algorithm is a procedure that takes a graph $G = (V, E)$ (and possibly edge weights or a source node) as input and produces a structured answer about that graph: the shortest path between two nodes, the set of connected components, the minimum total weight spanning tree, the maximum flow that can be routed from source to sink.

Graph algorithms fall into three broad families:

**Traversal algorithms** visit every vertex reachable from a starting point, in a principled order. Breadth-First Search (BFS) visits nodes layer by layer in increasing hop distance; Depth-First Search (DFS) dives as deep as possible before backtracking. Both run in $O(n + m)$ time for a graph with $n$ nodes and $m$ edges — this is optimal, since every vertex and edge must be examined at least once.

**Optimisation algorithms** find a subgraph or path that minimises or maximises some quantity. Dijkstra finds shortest paths minimising total edge weight. Kruskal and Prim find spanning trees minimising total weight. Ford-Fulkerson maximises the flow from source to sink. These algorithms use problem-specific data structures (priority queues, union-find) to achieve better-than-brute-force complexity.

**Structural algorithms** reveal the global organisation of a graph — which vertices belong to the same strongly connected component, whether the graph is a DAG, what its chromatic number is. Tarjan's SCC algorithm and topological sort are the core structural algorithms in this section.

The choice of algorithm depends critically on:
1. **Graph structure:** directed or undirected, weighted or unweighted, dense or sparse
2. **Edge weights:** all positive (Dijkstra), possibly negative (Bellman-Ford), integer capacities (max-flow)
3. **Query type:** single-source vs all-pairs, paths vs trees vs components
4. **Scale:** $n = 10^3$ (any algorithm), $n = 10^6$ (need $O(n \log n)$), $n = 10^9$ (need near-linear or parallel)

### 1.2 The Graph Algorithm Zoo

The table below gives the at-a-glance reference for every algorithm in this section.

| Algorithm | Problem | Time Complexity | Space | Edge Weights |
|-----------|---------|----------------|-------|--------------|
| BFS | SSSP (unweighted), reachability | $O(n+m)$ | $O(n)$ | Unweighted |
| DFS | Cycle detection, topological sort, SCC | $O(n+m)$ | $O(n)$ | Any |
| Dijkstra | SSSP | $O((n+m)\log n)$ | $O(n)$ | Non-negative |
| Bellman-Ford | SSSP, negative-cycle detection | $O(nm)$ | $O(n)$ | Any |
| Floyd-Warshall | APSP | $O(n^3)$ | $O(n^2)$ | Any (no neg cycle) |
| A\* | Single-pair SP with heuristic | $O(m \log n)$ avg | $O(n)$ | Non-negative |
| Kruskal | MST | $O(m \log m)$ | $O(n)$ | Any |
| Prim | MST | $O((n+m)\log n)$ | $O(n)$ | Any |
| Topo Sort (Kahn) | DAG ordering | $O(n+m)$ | $O(n)$ | DAG only |
| Tarjan SCC | Strongly connected components | $O(n+m)$ | $O(n)$ | Directed |
| Kosaraju SCC | Strongly connected components | $O(n+m)$ | $O(n)$ | Directed |
| Ford-Fulkerson | Max flow | $O(Ef)$ | $O(n+m)$ | Non-neg integer cap |
| Edmonds-Karp | Max flow | $O(nm^2)$ | $O(n+m)$ | Non-neg cap |

### 1.3 Why Graph Algorithms Matter for AI

Graph algorithms are not separate from deep learning — they are embedded in it.

**GNN receptive fields.** A $k$-layer Graph Neural Network aggregates information from the $k$-hop neighbourhood of each node. The set of nodes at hop distance $\leq k$ is exactly the BFS frontier after $k$ iterations from that node. The GNN forward pass *is* a parameterised, learnable BFS. See §2.6 for details.

**Autograd engines.** PyTorch's `backward()` call and JAX's `grad()` both operate on a Directed Acyclic Graph (DAG) of operations. Forward pass = topological sort traversal. Backward pass = reverse topological order, accumulating gradients via the chain rule. See §5.4.

**Knowledge graph reasoning.** Multi-hop reasoning over knowledge graphs (e.g., finding paths between entities in Wikidata) uses SSSP and BFS. KGQA (knowledge-graph question answering) systems routinely call shortest-path routines on subgraphs.

**Graph-based clustering.** Spectral clustering (§04) finds the optimal graph cut by computing the Fiedler eigenvector. The Cheeger inequality bounds the spectral gap by the minimum normalised cut — which is a min-cut problem. Graph-cut-based image segmentation (Boykov & Kolmogorov, 2004) uses max-flow to segment images.

**Reinforcement learning.** Bellman-Ford's update rule $d[v] \leftarrow \min_{u:(u,v)\in E}(d[u] + w(u,v))$ is the same recurrence as the Bellman equation in dynamic programming. Value iteration in RL is Bellman-Ford on the state-action graph.

**Embodied AI and planning.** Robot navigation, game playing (MCTS), and task planning all use shortest-path algorithms. A\* and its variants (Theta\*, D\* Lite) are standard in robotics path planning.

### 1.4 Historical Timeline

```
GRAPH ALGORITHM HISTORY
════════════════════════════════════════════════════════════════════════

  1736  Euler — Königsberg Bridge Problem: first graph theory argument
  1847  Kirchhoff — Spanning tree theorem for electrical networks
  1930  Jarník — MST algorithm (rediscovered as Prim 1957)
  1956  Ford & Fulkerson — Max-flow / augmenting path method
  1956  Bellman — Shortest paths via dynamic programming
  1957  Kruskal — MST via sorted edge greedy
  1959  Dijkstra — Shortest path with priority queue
  1962  Floyd / Warshall — All-pairs shortest paths via DP
  1968  Hart, Nilsson, Raphael — A* heuristic search
  1972  Tarjan — Linear-time DFS-based SCC algorithm
  1972  Hopcroft & Karp — Bipartite matching, max-flow
  1975  Kosaraju — Two-pass DFS SCC algorithm
  1987  Goldberg & Tarjan — Push-relabel max-flow O(n²√m)
  2003  Boykov & Kolmogorov — Graph-cuts for image segmentation
  2017  Kipf & Welling — GCN: BFS-structured neural message passing
  2021  GraphBLAS — Sparse linear algebra standard for graph analytics
  2023+ GPU-accelerated BFS for trillion-edge graphs (NVIDIA cuGraph)

════════════════════════════════════════════════════════════════════════
```

---

## 2. Graph Traversal

### 2.1 Breadth-First Search (BFS)

Breadth-First Search answers the question: "Starting from a source node $s$, in what order do we first encounter each reachable vertex, visiting closer vertices before farther ones?"

**Algorithm.** BFS maintains a FIFO queue $Q$ and a distance array $d[\cdot]$ initialised to $\infty$.

```
BFS(G, s):
  for each v in V: d[v] ← ∞, parent[v] ← NIL
  d[s] ← 0
  Q ← empty queue
  enqueue(Q, s)
  while Q not empty:
    u ← dequeue(Q)
    for each neighbour v of u:
      if d[v] = ∞:           ← v not yet visited
        d[v] ← d[u] + 1
        parent[v] ← u
        enqueue(Q, v)
```

**Output:** After BFS, $d[v]$ is the exact shortest number of hops from $s$ to $v$ (or $\infty$ if $v$ is not reachable). The `parent[]` array encodes the **BFS tree** — a spanning tree of the component containing $s$ in which every root-to-leaf path is a shortest path.

**Complexity.** Each vertex is enqueued at most once ($O(n)$ enqueue/dequeue operations). Each edge $(u,v)$ is examined at most twice — once when $u$ is dequeued, once when $v$ is dequeued. Total: $O(n + m)$ time, $O(n)$ space for $d$ and `parent`.

**Layer structure.** BFS naturally partitions $V$ into **layers** $L_0 = \{s\}$, $L_1 = $ neighbours of $s$, $L_2 = $ neighbours of $L_1$ not in $L_0 \cup L_1$, etc. All vertices in $L_k$ are at hop distance exactly $k$ from $s$.

**Example.** For the graph $0 - 1 - 2 - 3$, $1 - 4$, starting at $s = 0$:
- $L_0 = \{0\}$, $d[0]=0$
- $L_1 = \{1\}$, $d[1]=1$
- $L_2 = \{2, 4\}$, $d[2]=d[4]=2$
- $L_3 = \{3\}$, $d[3]=3$


### 2.2 BFS Correctness and Shortest-Hop Paths

**Theorem (BFS Shortest-Hop).** For an unweighted graph $G$ and source $s$, BFS sets $d[v] = \delta(s,v)$ for all $v \in V$, where $\delta(s,v)$ is the minimum hop distance from $s$ to $v$.

**Proof sketch.** We prove by induction on the hop distance $k$ that all vertices at distance $k$ from $s$ are assigned $d[v] = k$. Base: $k=0$, $d[s]=0$ by initialisation. Inductive step: suppose all vertices at distance $\leq k-1$ are correctly assigned. A vertex $v$ at distance $k$ has at least one neighbour $u$ at distance $k-1$. By induction $d[u]=k-1$, so when $u$ is dequeued, $v$ is discovered with $d[v]=k$. No earlier dequeue can discover $v$ with a smaller label, since any earlier dequeued node $u'$ has $d[u'] \leq k-1$, and $v$'s distance from $s$ via $u'$ would be $\geq k$. $\square$

**Corollary.** BFS solves the unweighted SSSP problem in $O(n+m)$ — optimal, since merely reading the graph takes $\Omega(n+m)$.

**Path reconstruction.** Following `parent[]` pointers from $v$ back to $s$ yields a shortest path. The path has exactly $d[v]$ edges.

### 2.3 Depth-First Search (DFS)

DFS explores as far as possible along each branch before backtracking. It assigns two timestamps to each vertex: `disc[v]` (when $v$ is first discovered) and `fin[v]` (when DFS finishes exploring all descendants of $v$).

```
DFS(G):
  for each v in V: colour[v] ← WHITE
  time ← 0
  for each v in V:
    if colour[v] = WHITE:
      DFS-VISIT(G, v)

DFS-VISIT(G, u):
  time ← time + 1; disc[u] ← time
  colour[u] ← GREY              ← discovered, not finished
  for each neighbour v of u:
    if colour[v] = WHITE:
      parent[v] ← u
      DFS-VISIT(G, v)
  colour[u] ← BLACK             ← finished
  time ← time + 1; fin[u] ← time
```

**Three colours** — WHITE (undiscovered), GREY (discovered, stack open), BLACK (finished) — are the classic DFS state machine. A vertex is GREY between its `disc` and `fin` timestamps.

**The parenthesis theorem.** For any two vertices $u, v$: the intervals $[\text{disc}[u], \text{fin}[u]]$ and $[\text{disc}[v], \text{fin}[v]]$ are either disjoint or one contains the other. They cannot partially overlap. This gives DFS a nested interval structure — hence the "parenthesis" name.

**Complexity.** $O(n + m)$ — each vertex transitions WHITE→GREY→BLACK exactly once; each edge is examined twice (once per endpoint in undirected, once in directed).

### 2.4 DFS: Edge Classification

In a DFS on a directed graph, every edge $(u,v)$ falls into exactly one class:

| Edge Type | Condition | Meaning |
|-----------|-----------|---------|
| **Tree edge** | $v$ is WHITE when $(u,v)$ is explored | Edge in DFS forest |
| **Back edge** | $v$ is GREY when $(u,v)$ is explored | $v$ is ancestor of $u$; forms a cycle |
| **Forward edge** | $v$ is BLACK, $\text{disc}[v] > \text{disc}[u]$ | $v$ is descendant of $u$ |
| **Cross edge** | $v$ is BLACK, $\text{disc}[v] < \text{disc}[u]$ | No ancestor/descendant relation |

**Key results from edge classification:**
- **Cycle detection:** A directed graph has a cycle if and only if DFS discovers at least one **back edge**. (An undirected graph has a cycle iff DFS finds a non-tree edge.)
- **DAG test:** A directed graph is a DAG ⟺ DFS finds no back edges.
- **Topological order:** In a DAG, vertices listed in decreasing `fin[]` order form a topological ordering (§5.2).

**For undirected graphs** there are only tree edges and back edges — forward and cross edges cannot occur, because an edge $(u,v)$ explored when $v$ is BLACK would have already been explored in the other direction.

### 2.5 BFS vs DFS — When to Use Which

```
BFS vs DFS: DECISION GUIDE
════════════════════════════════════════════════════════════════════════

  Use BFS when:                        Use DFS when:
  ─────────────                        ────────────
  ✓ Shortest paths (unweighted)        ✓ Cycle detection
  ✓ Level-by-level exploration         ✓ Topological sort
  ✓ GNN k-hop neighbourhood           ✓ Strongly connected components
  ✓ Web crawling (breadth-first)       ✓ Detecting back/forward edges
  ✓ Connected components               ✓ Maze solving (backtracking)
  ✓ Bipartiteness check               ✓ DFS tree for Tarjan SCC
  ✓ Social network proximity          ✓ Memory-efficient deep search

  Memory: O(n) queue (wide graphs     Memory: O(depth) stack (narrow
          can be large!)               graphs can be efficient)

  BFS queue = FIFO (first in, first out)
  DFS stack = LIFO (last in, first out) — implicit via recursion

════════════════════════════════════════════════════════════════════════
```

**Memory consideration.** For a wide, shallow graph (e.g., a star $K_{1,n}$), BFS's queue can hold $O(n)$ nodes simultaneously. For a deep, narrow graph (e.g., a path $P_n$), DFS recursion uses $O(n)$ stack frames. In practice, BFS is preferred when the shortest path or level structure matters; DFS is preferred when structural properties (SCC, cycles) are the goal.

### 2.6 AI Connection: GNN Receptive Fields

> **Preview (full treatment in [§05 Graph Neural Networks](../05-Graph-Neural-Networks/notes.md))**

A Graph Neural Network layer aggregates messages from immediate neighbours. After $k$ layers, each node has aggregated information from all nodes within hop distance $k$ — its **$k$-hop neighbourhood**, or **receptive field**.

This is exactly the BFS $k$-hop set $\Gamma^k(v) = \{u : d(v,u) \leq k\}$.

**GCN layer (Kipf & Welling 2017):**
$$\mathbf{H}^{[l+1]} = \sigma\!\left(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2} \mathbf{H}^{[l]} W^{[l]}\right)$$

where $\tilde{A} = A + I$ (adjacency with self-loops). The matrix multiplication $\tilde{A}\mathbf{H}^{[l]}$ is a **one-step BFS aggregation**: row $v$ of the result is the sum of feature vectors of all neighbours of $v$ (including $v$ itself). Stacking $k$ such layers = BFS $k$ hops, with the degree normalisation $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ playing the role of averaging rather than summing.

**Implication.** The expressiveness of a GNN is bounded by the $k$-WL hierarchy (Weisfeiler-Leman test). Two nodes that BFS cannot distinguish in $k$ hops will be assigned identical embeddings by any $k$-layer GNN with sum/mean aggregation — regardless of the weights $W^{[l]}$.

---

## 3. Shortest Paths

### 3.1 The SSSP Problem

**Definition.** Given a directed or undirected graph $G = (V, E, w)$ with edge weight function $w: E \to \mathbb{R}$ and a source vertex $s \in V$, the **Single-Source Shortest Path (SSSP)** problem asks: for every $v \in V$, find the shortest-weight path from $s$ to $v$.

The **distance** from $s$ to $v$ is:
$$\delta(s, v) = \min_{P: s \leadsto v} \sum_{e \in P} w(e)$$
where the minimum is over all paths $P$ from $s$ to $v$. If no path exists, $\delta(s,v) = +\infty$. If a negative-weight cycle is reachable on some $s \leadsto v$ path, $\delta(s,v) = -\infty$.

**Triangle inequality.** For all $u, v, x \in V$: $\delta(s,v) \leq \delta(s,u) + w(u,v)$. This is the key relaxation invariant used by all SSSP algorithms.

**Relaxation.** The operation of updating a distance estimate:
```
RELAX(u, v, w):
  if d[v] > d[u] + w(u,v):
    d[v] ← d[u] + w(u,v)
    parent[v] ← u
```
All SSSP algorithms consist of choosing which edges to relax, and in what order. BFS is the special case where all weights equal 1 and the queue replaces the priority queue.


### 3.2 Dijkstra's Algorithm

Dijkstra's algorithm solves SSSP for graphs with **non-negative edge weights**. It is a greedy algorithm: at each step it permanently fixes the shortest distance to the closest unvisited vertex.

**Algorithm.**

```
DIJKSTRA(G, s):
  for each v in V: d[v] ← ∞, parent[v] ← NIL
  d[s] ← 0
  Q ← min-priority queue with all vertices keyed by d[·]
  while Q not empty:
    u ← EXTRACT-MIN(Q)           ← vertex with smallest tentative distance
    for each neighbour v of u:
      if d[u] + w(u,v) < d[v]:
        d[v] ← d[u] + w(u,v)
        parent[v] ← u
        DECREASE-KEY(Q, v, d[v]) ← update priority queue
```

**Correctness invariant.** When vertex $u$ is extracted from $Q$, $d[u] = \delta(s,u)$ — it is the exact shortest distance. This holds because all weights are non-negative: any alternative path from $s$ to $u$ passing through an unvisited vertex $x$ has $d[x] \geq d[u]$ (since $u$ was extracted first), so extending that path cannot improve upon $d[u]$.

**Proof sketch.** By induction. Base: $s$ is extracted first with $d[s]=0=\delta(s,s)$. Inductive step: assume all previously extracted vertices have correct distances. Let $u$ be the next extracted vertex. Suppose for contradiction there exists a shorter path $P: s \to \ldots \to y \to x \to \ldots \to u$ with $y$ extracted, $x$ not extracted. Then $d[x] \geq d[u]$ (otherwise $x$ would have been extracted before $u$), and the suffix $x \to \ldots \to u$ has non-negative weight, so the path is no shorter than $d[u]$. Contradiction. $\square$

**Complexity.** With a binary heap: $O((n+m)\log n)$ — $n$ EXTRACT-MIN and up to $m$ DECREASE-KEY operations, each $O(\log n)$. With a Fibonacci heap: $O(m + n \log n)$ — theoretically better for dense graphs; rarely used in practice due to large constants.

**Failure case.** Dijkstra gives incorrect results when any edge has negative weight. Example: $s \to u$ with weight 10, $s \to v$ with weight 2, $v \to u$ with weight $-5$. Dijkstra fixes $d[u] = 10$ when $u$ is extracted, but the true distance is $2 + (-5) = 7$ via $v \to u$.

**For AI.** Dijkstra is used in knowledge graph embedding to find the shortest semantic path between entities. In route planning for embodied agents, Dijkstra on a road graph with positive travel times is the standard baseline. Its priority queue structure mirrors the attention mechanism's softmax weighting of neighbour importance.

### 3.3 Bellman-Ford Algorithm

Bellman-Ford solves SSSP for graphs with **arbitrary edge weights** (including negative), and detects negative-weight cycles.

**Algorithm.**

```
BELLMAN-FORD(G, s):
  for each v in V: d[v] ← ∞, parent[v] ← NIL
  d[s] ← 0
  for i = 1 to n-1:              ← n-1 relaxation rounds
    for each edge (u,v) in E:
      RELAX(u, v, w(u,v))
  for each edge (u,v) in E:      ← negative cycle check
    if d[v] > d[u] + w(u,v):
      return "negative cycle exists"
  return d[]
```

**Why $n-1$ rounds?** Any shortest path that does not repeat a vertex has at most $n-1$ edges. After round $i$, $d[v]$ correctly reflects the shortest path using at most $i$ edges. After $n-1$ rounds, all shortest paths are found (assuming no negative cycle).

**Negative cycle detection.** After $n-1$ rounds, if any edge still allows relaxation, there is a negative-weight cycle reachable from $s$ — the $n$-th round would continue improving distances indefinitely.

**Complexity.** $O(nm)$ — $n-1$ rounds, each examining all $m$ edges. Slower than Dijkstra but correct for negative weights.

**Dynamic programming view.** Let $d_i[v] = $ length of shortest path from $s$ to $v$ using $\leq i$ edges. Recurrence:
$$d_i[v] = \min\!\left(d_{i-1}[v],\; \min_{(u,v)\in E}\!\left(d_{i-1}[u] + w(u,v)\right)\right)$$

This is exactly the Bellman equation in RL:
$$V^*(s) = \max_a \left(R(s,a) + \gamma \max_{s'} V^*(s')\right)$$

with $d_i[v] \leftrightarrow V_i(s)$ (value after $i$ iterations), $w(u,v) \leftrightarrow -R(s,a)$ (negative reward = positive cost), and $\gamma=1$. **Value iteration in RL is Bellman-Ford on the state-action graph.**

### 3.4 Floyd-Warshall: All-Pairs Shortest Paths

Floyd-Warshall solves the **All-Pairs Shortest Path (APSP)** problem: find $\delta(u,v)$ for all pairs $u,v \in V$.

**Algorithm.** Maintain a matrix $D$ where $D[i][j]$ is the shortest known distance from $i$ to $j$.

```
FLOYD-WARSHALL(G):
  D ← n×n matrix with D[i][j] = w(i,j) if (i,j)∈E, 0 if i=j, ∞ otherwise
  for k = 1 to n:
    for i = 1 to n:
      for j = 1 to n:
        D[i][j] ← min(D[i][j], D[i][k] + D[k][j])
  return D
```

**Correctness.** After the outer loop for $k$, $D[i][j]$ holds the shortest path from $i$ to $j$ using only vertices $\{1, \ldots, k\}$ as intermediates. After all $n$ iterations, $D$ holds all-pairs shortest distances. The recurrence:
$$D^{(k)}[i][j] = \min\!\left(D^{(k-1)}[i][j],\; D^{(k-1)}[i][k] + D^{(k-1)}[k][j]\right)$$
is a DP over the choice of whether vertex $k$ is used as an intermediate.

**Transitive closure.** Replace min/+ with Boolean or/and to compute reachability: $T[i][j]=1$ iff a path from $i$ to $j$ exists. Same $O(n^3)$ structure.

**Negative cycles.** After Floyd-Warshall, if $D[i][i] < 0$ for some $i$, vertex $i$ lies on a negative cycle.

**Complexity.** $O(n^3)$ time, $O(n^2)$ space. Practical for $n \leq 1000$; infeasible for large graphs.

**Matrix interpretation.** The Floyd-Warshall update $D[i][j] \leftarrow \min(D[i][j], D[i][k]+D[k][j])$ is the tropical matrix product (min-plus algebra):
$(A \otimes B)[i][j] = \min_k(A[i][k] + B[k][j])$
So $D^{(n)} = W^{\otimes n}$ — the $n$-th tropical power of the weight matrix — gives all-pairs distances. This connects APSP to algebraic path problems over semirings.

### 3.5 A* Search

A\* (Hart, Nilsson & Raphael, 1968) is a best-first search algorithm that guides Dijkstra with a **heuristic function** $h(v)$ estimating the remaining cost from $v$ to the goal $t$.

**Algorithm.** Identical to Dijkstra, but the priority queue key for vertex $v$ is $f(v) = d[v] + h(v)$ instead of just $d[v]$.

```
A-STAR(G, s, t, h):
  for each v: d[v] ← ∞, f[v] ← ∞
  d[s] ← 0; f[s] ← h(s)
  Q ← min-priority queue keyed by f[·]
  enqueue(Q, s)
  while Q not empty:
    u ← EXTRACT-MIN(Q)
    if u = t: return d[t]        ← goal found
    for each neighbour v of u:
      new_d ← d[u] + w(u,v)
      if new_d < d[v]:
        d[v] ← new_d
        f[v] ← new_d + h(v)
        update Q with v
```

**Admissibility.** A heuristic $h$ is **admissible** if $h(v) \leq \delta(v,t)$ for all $v$ — it never overestimates. An admissible heuristic guarantees A\* finds the optimal path.

**Consistency (monotonicity).** $h$ is **consistent** if $h(u) \leq w(u,v) + h(v)$ for all edges $(u,v)$. Consistency implies admissibility and guarantees each vertex is processed at most once (like Dijkstra).

**Examples of admissible heuristics:**
- Grid navigation: $h(v) = $ Euclidean distance to goal (straight-line, never overestimates)
- Grid with 4-directional movement: $h(v) = $ Manhattan distance to goal

**AI relevance.** A\* is the standard algorithm in robotics path planning (ROS navigation stack), game AI (Pathfinder, StarCraft), and any domain where a domain-specific heuristic is available. The trade-off: a tighter heuristic (closer to true distance) leads to fewer node expansions. With $h \equiv 0$, A\* degenerates to Dijkstra. With $h = \delta(\cdot,t)$ (perfect heuristic), A\* expands only nodes on the optimal path.

### 3.6 Shortest-Path Complexity Comparison

| Algorithm | Time | Space | Weights | Use Case |
|-----------|------|-------|---------|----------|
| BFS | $O(n+m)$ | $O(n)$ | Unweighted only | Hop distance, BFS tree |
| Dijkstra (binary heap) | $O((n+m)\log n)$ | $O(n)$ | Non-negative | Standard SP |
| Dijkstra (Fibonacci heap) | $O(m+n\log n)$ | $O(n)$ | Non-negative | Dense graphs, theory |
| Bellman-Ford | $O(nm)$ | $O(n)$ | Any (detects neg cycle) | Negative weights, DP |
| SPFA (Bellman-Ford queue opt.) | $O(nm)$ worst, $O(m)$ avg | $O(n)$ | Any | Practical Bellman-Ford |
| Floyd-Warshall | $O(n^3)$ | $O(n^2)$ | Any (no neg cycle) | All-pairs, small $n$ |
| Johnson's | $O(nm + n^2\log n)$ | $O(n^2)$ | Any | APSP, sparse graphs |
| A\* | $O(m\log n)$ avg | $O(n)$ | Non-negative | Single-pair with heuristic |


---

## 4. Minimum Spanning Trees

### 4.1 The MST Problem and Key Properties

**Definition.** Given a connected, undirected, weighted graph $G = (V, E, w)$, a **spanning tree** $T \subseteq E$ is a subgraph that connects all $n$ vertices with exactly $n-1$ edges (i.e., a connected acyclic spanning subgraph). A **Minimum Spanning Tree (MST)** is a spanning tree minimising total edge weight:
$$\text{MST} = \arg\min_{T \text{ spanning tree}} \sum_{e \in T} w(e)$$

**Existence.** Every connected graph has at least one MST. If all edge weights are distinct, the MST is unique.

**Two key structural theorems** underlie all MST algorithms:

**Cut property.** Let $S \subset V$ be any proper subset of vertices, and let $e = (u,v)$ be the minimum-weight edge crossing the cut $(S, V\setminus S)$ (i.e., $u \in S$, $v \in V\setminus S$). Then $e$ belongs to every MST of $G$.

*Intuition:* If the minimum cut-crossing edge were not in the MST, we could swap it with a heavier cut-crossing edge in the MST and reduce total weight — contradiction.

**Cycle property.** Let $C$ be any cycle in $G$, and let $e$ be the unique maximum-weight edge of $C$. Then $e$ does NOT belong to any MST of $G$.

*Intuition:* If $e$ were in the MST, removing it would disconnect the tree into two components; the other edges of $C$ provide a cheaper reconnection, contradicting the MST assumption.

Both Kruskal and Prim are instantiations of a generic **greedy MST algorithm** that repeatedly adds the minimum-weight safe edge (one that doesn't create a cycle and respects the cut property).

**Total weight lower bound.** The MST weight gives the minimum cost to connect all vertices — any other spanning tree costs at least as much. MST weight is a lower bound for the Travelling Salesman Problem (TSP) on the same metric graph.

### 4.2 Kruskal's Algorithm

Kruskal's algorithm builds the MST by greedily adding edges in increasing weight order, skipping edges that would form a cycle.

```
KRUSKAL(G):
  Sort edges E by weight: e_1 ≤ e_2 ≤ ... ≤ e_m
  T ← ∅
  for each vertex v: MAKE-SET(v)   ← Union-Find initialisation
  for each edge (u,v) in sorted order:
    if FIND(u) ≠ FIND(v):         ← u and v in different components?
      T ← T ∪ {(u,v)}
      UNION(u, v)                  ← merge components
  return T
```

**Union-Find (Disjoint Set Union).** This data structure maintains a collection of disjoint sets and supports:
- `MAKE-SET(x)`: create a singleton set $\{x\}$
- `FIND(x)`: return the representative ("root") of $x$'s set
- `UNION(x, y)`: merge the sets containing $x$ and $y$

With **union by rank** and **path compression**, all operations run in amortised $O(\alpha(n))$ time, where $\alpha$ is the inverse Ackermann function — effectively $O(1)$ for all practical $n$.

**Correctness.** Each added edge is the minimum-weight edge crossing some cut $(S, V\setminus S)$ where $S$ is one of the current components. By the cut property, this edge belongs to the MST.

**Complexity.** Sorting edges: $O(m \log m)$. Union-Find: $O(m \alpha(n))$. Total: $O(m \log m)$, dominated by the sort.

**Example.** Graph with edges $(A,B,1), (A,C,4), (B,C,2), (B,D,5), (C,D,3)$:
- Add $(A,B,1)$: $T = \{AB\}$; components $\{A,B\}, \{C\}, \{D\}$
- Add $(B,C,2)$: $T = \{AB,BC\}$; components $\{A,B,C\}, \{D\}$
- Add $(C,D,3)$: $T = \{AB,BC,CD\}$; all connected — done. MST weight $= 1+2+3=6$
- Skip $(A,C,4)$: would form cycle $A-B-C-A$
- Skip $(B,D,5)$: would form cycle $B-C-D-B$

### 4.3 Prim's Algorithm

Prim's algorithm grows the MST one vertex at a time, always attaching the cheapest edge connecting the current tree to an unvisited vertex.

```
PRIM(G, r):
  for each v: key[v] ← ∞, parent[v] ← NIL, inMST[v] ← FALSE
  key[r] ← 0                    ← start from root r
  Q ← min-priority queue of all vertices keyed by key[·]
  while Q not empty:
    u ← EXTRACT-MIN(Q)
    inMST[u] ← TRUE
    for each neighbour v of u:
      if NOT inMST[v] AND w(u,v) < key[v]:
        key[v] ← w(u,v)
        parent[v] ← u
        DECREASE-KEY(Q, v, key[v])
  return {(v, parent[v]) : v ≠ r}
```

**Analogy with Dijkstra.** Prim and Dijkstra have identical structure — the only difference is the priority key: Dijkstra uses $d[u] + w(u,v)$ (cumulative path cost); Prim uses $w(u,v)$ (edge cost alone). Prim greedily picks the cheapest edge into the tree; Dijkstra greedily picks the cheapest path to any vertex.

**Complexity.** $O((n+m)\log n)$ with a binary heap — same as Dijkstra.

**When to prefer Prim vs Kruskal:**
- Kruskal is simpler to implement and better for sparse graphs ($m$ close to $n$)
- Prim is better for dense graphs ($m$ close to $n^2$) with an adjacency matrix
- Both produce an MST with the same total weight (unique if weights are distinct)

### 4.4 AI Applications of MSTs

**Graph-based clustering.** Remove the $k-1$ heaviest edges from the MST to obtain $k$ clusters. This is **single-linkage hierarchical clustering** — equivalent to building the MST and pruning. Each resulting subtree is a cluster. The method is sensitive to outliers (a long edge can connect two distant clusters) but is $O(m \log m)$ and parameter-free.

**Feature selection.** Build a minimum spanning tree on a graph where nodes are features and edge weights are mutual information between features. The MST structure reveals the most informative non-redundant features.

**Network design.** Connect $n$ nodes (data centres, sensors, brain regions) with minimum total cable cost while ensuring full connectivity.

**Approximate TSP.** The MST provides a 2-approximation for TSP on metric graphs: walk the MST DFS-order and shortcut repeated vertices. This gives a tour of cost $\leq 2 \cdot \text{MST}$. Christofides' algorithm (1976) uses MST + perfect matching for a 1.5-approximation.

**Neuroscience.** Brain connectivity graphs are often analysed via MST to find the backbone connectivity structure with minimal total connection weight.

---

## 5. Topological Sort and DAG Algorithms

### 5.1 Directed Acyclic Graphs (DAGs)

A **Directed Acyclic Graph (DAG)** is a directed graph with no directed cycles. Formally, there is no sequence $v_1 \to v_2 \to \cdots \to v_k \to v_1$ of distinct vertices.

**Recognition.** Run DFS. If DFS finds a **back edge** $(u,v)$ (an edge from $u$ to an ancestor $v$ in the DFS tree), the graph has a cycle and is not a DAG. If DFS completes without finding any back edge, the graph is a DAG.

**Examples of DAGs in AI:**
- **Computation graphs:** every PyTorch/JAX forward pass. Each `op` node has edges to the ops whose outputs it consumes.
- **Causal graphs:** nodes are random variables, edges are causal influences. A structural causal model (SCM) is a DAG.
- **Dependency graphs:** package managers (pip, conda, apt) resolve dependencies on a DAG.
- **Scheduling:** project tasks with precedence constraints.
- **Probabilistic graphical models:** Bayesian networks are DAGs where edges represent conditional dependencies.

**In-degree and out-degree in DAGs.** Every DAG has at least one **source** (in-degree 0) and at least one **sink** (out-degree 0). This follows because if every vertex had a predecessor, we could follow predecessors indefinitely — but the graph is finite and acyclic, so we must eventually reach a vertex with no predecessor.

### 5.2 Topological Sort

A **topological ordering** of a DAG is a linear ordering $v_1, v_2, \ldots, v_n$ of its vertices such that for every directed edge $(v_i, v_j)$, we have $i < j$ (the source comes before the target).

**Existence.** Every DAG has at least one topological ordering. (Proof: repeatedly remove a source vertex and its outgoing edges; by the DAG property, at least one source always exists.)

**Non-uniqueness.** If there are multiple vertices with in-degree 0, different choices yield different topological orderings.

**Algorithm 1: Kahn's Algorithm (BFS-based)**

```
KAHN(G):
  Compute in-degree indeg[v] for each v
  Q ← queue of all vertices with indeg[v] = 0   ← sources
  order ← empty list
  while Q not empty:
    u ← dequeue(Q)
    append u to order
    for each edge (u,v):
      indeg[v] ← indeg[v] - 1
      if indeg[v] = 0: enqueue(Q, v)
  if |order| < n: return "graph has a cycle"
  return order
```

Kahn's algorithm processes vertices in topological order, removing each vertex and its edges. If it processes fewer than $n$ vertices, the remaining vertices form a cycle.

**Algorithm 2: DFS-based**

Run DFS. As each vertex $v$ is finished (colour turns BLACK), prepend $v$ to the result list. The final list (in reverse finish time order) is a valid topological ordering.

*Why?* For any edge $(u,v)$ in a DAG, $\text{fin}[v] < \text{fin}[u]$ (since $v$ finishes before $u$ due to DFS structure). So decreasing finish time = correct topological order.

**Complexity.** Both algorithms run in $O(n+m)$.

### 5.3 Longest and Shortest Paths in DAGs

In general graphs, finding longest paths is NP-hard (it generalises TSP). But in DAGs, both longest and shortest paths can be found in $O(n+m)$ using DP over the topological ordering.

**Algorithm.**

```
DAG-SSSP(G, s):
  d[s] ← 0; d[v] ← ∞ for all v ≠ s
  for each u in topological order of G:
    for each edge (u,v):
      RELAX(u, v, w(u,v))
  return d[]
```

For longest paths, initialise $d[s]=0$, $d[v]=-\infty$ and replace RELAX with a maximisation.

**Critical Path Method.** In project scheduling, nodes are tasks and edges represent precedence. Edge weights are durations. The **critical path** is the longest path in the DAG — it determines the minimum project duration. Earliest/latest start times for each task are computed via two DAG DP passes (forward and backward).

### 5.4 AI Connection: Autograd Computation Graphs

> **This is one of the most important applications of graph algorithms in all of modern AI.**

PyTorch's autograd engine builds a DAG when executing a forward pass. Every tensor operation (`matmul`, `relu`, `softmax`, ...) creates a node; edges connect inputs to outputs.

```
PYTORCH AUTOGRAD DAG (simplified):
════════════════════════════════════════════════════════════════════════

  Forward pass (topological order, sources to sinks):
    x ───matmul──→ z ───relu──→ a ───softmax──→ p ───cross_entropy──→ L
          ↑                         ↑
          W                         (no weight here)

  Backward pass (reverse topological order, sinks to sources):
    ∂L/∂p  ←──── ∂L/∂a ←──── ∂L/∂z ←──── ∂L/∂x, ∂L/∂W

  At each node: multiply incoming gradient by local Jacobian (chain rule)
  Accumulate gradients at leaf nodes (parameters)

════════════════════════════════════════════════════════════════════════
```

**Forward pass = topological traversal** from input tensors (sources) to the loss (sink). Each node applies its operation.

**Backward pass = reverse topological traversal**. Each node receives the upstream gradient $\partial L / \partial \text{output}$ and propagates $\partial L / \partial \text{input} = \partial L / \partial \text{output} \cdot \partial \text{output} / \partial \text{input}$ to its predecessors. The gradient for shared tensors is accumulated (summed) over all incoming edges.

**Why DAG and not a tree?** When a tensor is used multiple times (e.g., weight sharing, skip connections in ResNet), it has multiple out-edges. The backward pass sums gradients from all out-edges — this is the graph-theoretic explanation for gradient accumulation.

**No-cycle guarantee.** The autograd engine assumes the computation graph is a DAG. Recurrent networks (RNNs) are *unrolled* through time into DAGs before differentiation — this is **Backpropagation Through Time (BPTT)**, which is simply differentiation on an unrolled DAG.


---

## 6. Strongly Connected Components

### 6.1 Connectivity in Directed Graphs

In undirected graphs, connectivity is binary: two vertices are either connected or not. In directed graphs, connectivity is richer.

**Definitions.**
- Vertices $u$ and $v$ are **mutually reachable** if there exists a directed path $u \leadsto v$ AND a directed path $v \leadsto u$.
- A **Strongly Connected Component (SCC)** is a maximal subset $C \subseteq V$ such that every pair $u, v \in C$ is mutually reachable.
- Every directed graph has a unique partition into SCCs.
- A vertex with no edges forms its own trivial SCC $\{v\}$.

**Condensation graph.** Contract each SCC to a single node. The result is a DAG — the **condensation** (also called the SCC DAG or meta-graph). This DAG reveals the high-level structure of the directed graph: SCCs with no incoming edges are "sources" (starting points for information flow); SCCs with no outgoing edges are "sinks".

**Example.** For the directed graph $1 \to 2 \to 3 \to 1$ (cycle), $3 \to 4$, $4 \to 5$, $5 \to 4$ (another cycle):
- SCCs: $\{1,2,3\}$ (cycle), $\{4,5\}$ (cycle), each a single strongly connected component
- Condensation: $\{1,2,3\} \to \{4,5\}$ — a DAG with two nodes and one edge

### 6.2 Kosaraju's Algorithm

Kosaraju's algorithm finds all SCCs with two DFS passes.

```
KOSARAJU(G):
  Phase 1: Run DFS on G; push vertices onto stack S in finish order
  Phase 2: Compute G^T (reverse all edges)
           Run DFS on G^T, exploring in decreasing finish order (pop from S)
           Each DFS tree in phase 2 is one SCC
```

**Why it works.** In $G$, a vertex $u$ with a high finish time is in a "high" SCC in the condensation (or is itself a source SCC). In $G^T$, the condensation edges are reversed. A DFS from $u$ in $G^T$ can only reach vertices in $u$'s own SCC (it cannot cross condensation edges to "higher" SCCs, which have been reversed). So each DFS tree in $G^T$ exactly recovers one SCC.

**Complexity.** Two DFS passes: $O(n+m)$ each. Graph reversal: $O(n+m)$. Total: $O(n+m)$.

### 6.3 Tarjan's Algorithm

Tarjan's algorithm finds all SCCs in a single DFS pass using **low-link values**.

**Low-link value.** $\text{low}[v] = \min(\text{disc}[v], \min_{(v,w)\in E}\text{low}[w], \min_{(v,w)\in E, w \in \text{stack}}\text{disc}[w])$ — the minimum discovery time reachable from the subtree rooted at $v$ via back edges.

```
TARJAN(G):
  disc[] ← all -1, low[] ← 0
  onStack[] ← all FALSE
  stack S ← empty, time ← 0, sccs ← []
  
  DFS-VISIT(v):
    disc[v] ← low[v] ← time++
    push v onto S; onStack[v] ← TRUE
    for each edge (v,w):
      if disc[w] = -1:              ← tree edge
        DFS-VISIT(w)
        low[v] ← min(low[v], low[w])
      elif onStack[w]:              ← back edge to stack vertex
        low[v] ← min(low[v], disc[w])
    if low[v] = disc[v]:           ← v is root of an SCC
      pop vertices from S until v, forming one SCC

  for each v with disc[v] = -1: DFS-VISIT(v)
```

**The key insight.** A vertex $v$ is the root of an SCC if and only if no back edge from $v$'s subtree escapes to a vertex discovered before $v$ — i.e., $\text{low}[v] = \text{disc}[v]$. When this condition holds, all vertices on the stack above $v$ (including $v$) form one SCC.

**Complexity.** Single DFS pass: $O(n+m)$.

**Comparison with Kosaraju:**
- Both are $O(n+m)$; Tarjan is faster in practice (one pass vs two, no graph reversal)
- Kosaraju is easier to reason about (correctness follows from DFS finish order properties)
- Tarjan is more memory-efficient (no need to store the reversed graph)

### 6.4 AI Applications of SCCs

**Knowledge graph analysis.** In a directed knowledge graph (entity → relation → entity), SCCs correspond to groups of entities that are mutually related via directed paths. Trivial SCCs (singletons) are "leaf" entities; large SCCs indicate dense relationship clusters.

**Circular dependency detection.** Package dependency graphs should be DAGs. If the package manager finds an SCC with more than one node, there is a circular dependency — a build-system error.

**PageRank on condensation.** Google's original PageRank algorithm first computes the condensation DAG, then computes within-SCC PageRank, then propagates across the condensation edges. This avoids the "dangling node" and "spider trap" problems that arise from cycles.

**Constraint propagation.** In SAT solving (Boolean satisfiability), 2-SAT problems reduce to finding SCCs in an implication graph. A formula is satisfiable iff no variable $x$ and its negation $\neg x$ appear in the same SCC.

---

## 7. Maximum Flow and Minimum Cut

### 7.1 Flow Networks

A **flow network** is a directed graph $G = (V, E)$ with:
- A **source** $s \in V$ (no incoming edges in the abstract model)
- A **sink** $t \in V$ (no outgoing edges)
- A **capacity function** $c: E \to \mathbb{R}_{\geq 0}$ giving the maximum flow on each edge

A **flow** is a function $f: E \to \mathbb{R}_{\geq 0}$ satisfying:
1. **Capacity constraint:** $0 \leq f(u,v) \leq c(u,v)$ for all $(u,v) \in E$
2. **Flow conservation:** For each $v \neq s, t$: $\sum_{u:(u,v)\in E} f(u,v) = \sum_{w:(v,w)\in E} f(v,w)$ (flow in = flow out)

The **value** of flow $f$ is $|f| = \sum_{v:(s,v)\in E} f(s,v)$ — total flow leaving the source.

**Goal:** Find a flow $f^*$ maximising $|f|$.

### 7.2 Ford-Fulkerson and Augmenting Paths

**Residual graph.** Given flow $f$, the **residual graph** $G_f = (V, E_f)$ captures where flow can be added or cancelled:
- For edge $(u,v) \in E$: **forward residual** capacity $c_f(u,v) = c(u,v) - f(u,v) \geq 0$
- For edge $(u,v) \in E$: **backward residual** capacity $c_f(v,u) = f(u,v) \geq 0$ (flow can be cancelled)

An **augmenting path** is any path from $s$ to $t$ in $G_f$ with positive capacity on every edge.

```
FORD-FULKERSON(G, s, t):
  f(u,v) ← 0 for all edges
  while there exists an augmenting path P in G_f:
    δ ← min capacity on P (bottleneck)
    augment flow along P by δ
  return f
```

**Correctness.** The algorithm terminates (for integer capacities; may not terminate for irrational capacities with bad path choices) and returns a maximum flow, by the Max-Flow Min-Cut theorem (§7.4).

**Complexity.** $O(Ef)$ where $E = |E|$ and $f = |f^*|$ is the maximum flow value. This can be exponential if edge capacities are large.

### 7.3 Edmonds-Karp Algorithm

Edmonds-Karp (1972) is Ford-Fulkerson with BFS used to find augmenting paths (shortest path in $G_f$ in terms of number of edges).

**Complexity.** $O(nm^2)$ — polynomial regardless of edge capacities. The key insight: augmenting along a shortest path guarantees that the number of augmentation steps is at most $O(nm)$.

### 7.4 The Max-Flow Min-Cut Theorem

**Cut definition.** An **$s$-$t$ cut** $(S, T)$ partitions $V$ into $S$ (containing $s$) and $T = V \setminus S$ (containing $t$). The **capacity** of the cut is $c(S,T) = \sum_{u \in S, v \in T, (u,v)\in E} c(u,v)$.

**Weak duality.** For any flow $f$ and any $s$-$t$ cut $(S,T)$: $|f| \leq c(S,T)$. (Proof: the flow must cross the cut, and is bounded by cut capacity.)

**Theorem (Max-Flow Min-Cut).** The maximum value of any $s$-$t$ flow equals the minimum capacity of any $s$-$t$ cut:
$$\max_f |f| = \min_{(S,T) \text{ s-t cut}} c(S,T)$$

**Proof sketch.** When Ford-Fulkerson terminates, there is no augmenting path in $G_f$. Let $S = \{v : s \leadsto v \text{ in } G_f\}$. Then $(S, V\setminus S)$ is an $s$-$t$ cut with no residual capacity — meaning every edge $(u,v)$ from $S$ to $V\setminus S$ is saturated ($f(u,v) = c(u,v)$) and every edge from $V\setminus S$ to $S$ carries zero flow. Thus $|f| = c(S,V\setminus S)$. Since $|f| \leq c(S,T)$ for any cut, $f$ is maximum and $(S,V\setminus S)$ is minimum. $\square$

**Integrality theorem.** If all capacities are integers, there exists a maximum flow that is integer-valued on every edge.

### 7.5 AI Connections: Graph Cuts and Partitioning

**Image segmentation.** Model pixels as nodes; adjacent pixels connected by edges weighted by colour similarity (high weight = similar). Source $s$ = "foreground" terminal, sink $t$ = "background" terminal. Add edges from $s$ to pixels likely to be foreground, from pixels likely to be background to $t$. The min-cut partitions pixels into foreground/background — this is the **Boykov-Kolmogorov graph cut** algorithm used throughout computer vision (2001–2010).

**Graph partitioning.** Dividing a graph into $k$ balanced parts with minimal cut capacity — used for distributed computing (partition a computation graph across GPUs), parallel simulation, and community detection.

**Spectral connection (forward reference).** The **normalised cut** (Shi & Malik 2000) minimises $\frac{\text{cut}(S,T)}{\text{vol}(S)} + \frac{\text{cut}(T,S)}{\text{vol}(T)}$, which approximates $h(G)$ — the **Cheeger constant** of the graph. The Cheeger inequality (§04) bounds $h(G)$ in terms of the Fiedler eigenvalue $\lambda_2$: $h(G) \geq \lambda_2/2$. This is how max-flow/min-cut and spectral graph theory connect.

→ _Full treatment of spectral clustering: [§04 Spectral Graph Theory](../04-Spectral-Graph-Theory/notes.md)_


---

## 8. Advanced Topics

### 8.1 Bidirectional Search

**Motivation.** Standard Dijkstra from source $s$ to target $t$ explores a "ball" of radius $d(s,t)$ around $s$. For large graphs, this ball can be enormous.

**Bidirectional Dijkstra** runs two simultaneous Dijkstra searches: one forward from $s$ and one backward from $t$ (on the reversed graph). The searches alternate, and terminate when a vertex is settled by both.

**Speedup.** In the best case, each search explores a ball of radius $d(s,t)/2$, reducing explored area from $\pi r^2$ to $2 \pi (r/2)^2 = \pi r^2 / 2$ — a factor of 2 in practice, but can be much better for graphs with clustered structure.

**ALT (A\* with Landmarks and Triangle inequality).** A practical enhancement: precompute shortest distances from a small set of **landmark** vertices. Use these to compute tight admissible heuristics for A\*. Used in production road routing (OpenStreetMap, Google Maps).

**Contraction Hierarchies.** The leading practical algorithm for road network routing: precompute a hierarchy of "shortcut" edges that compress long paths. Query time is milliseconds for continent-scale road graphs. Used in OSRM and many navigation systems.

### 8.2 Johnson's Algorithm

Johnson's algorithm solves APSP for sparse graphs with negative edge weights (but no negative cycles), more efficiently than running Floyd-Warshall.

**Key idea.** Reweight edges to make all weights non-negative, then run Dijkstra from every source.

```
JOHNSON(G):
  1. Add new vertex q with zero-weight edges to all vertices
  2. Run Bellman-Ford from q to compute h[v] = δ(q,v) for all v
  3. Reweight: w'(u,v) = w(u,v) + h[u] - h[v] ≥ 0
  4. For each source s: run Dijkstra with weights w'
  5. Adjust distances: δ(u,v) = δ'(u,v) - h[u] + h[v]
```

**Complexity.** $O(nm + n^2\log n)$ — one Bellman-Ford ($O(nm)$) plus $n$ Dijkstra runs ($O(n(n+m)\log n)$ total). Better than Floyd-Warshall's $O(n^3)$ for sparse graphs where $m \ll n^2$.

**Correctness of reweighting.** The reweighted distances $w'(u,v)$ are non-negative (by the triangle inequality on Bellman-Ford distances: $h[v] \leq h[u] + w(u,v)$). The reweighting preserves shortest paths: $\delta'(s,t) = \delta(s,t) + h[s] - h[t]$.

### 8.3 Parallel and GPU Graph Algorithms

**Frontier-based parallel BFS.** Standard BFS is sequential (queue pops one vertex at a time). Parallel BFS processes the entire current frontier simultaneously:

```
PARALLEL-BFS(G, s):
  frontier ← {s}
  d[s] ← 0
  while frontier not empty:
    next_frontier ← ∅
    for each u in frontier (in parallel):
      for each neighbour v of u:
        if d[v] = ∞ (using atomic compare-and-swap):
          d[v] ← d[u] + 1
          add v to next_frontier
    frontier ← next_frontier
```

This is called **level-synchronous BFS** or **frontier BFS**. On a GPU with thousands of cores, all vertices in the frontier are processed simultaneously. For billion-node social network graphs, frontier BFS with GPU parallelism achieves throughputs of $10^9$ edges/second (NVIDIA cuGraph, 2023).

**GraphBLAS.** The GraphBLAS standard (2017) expresses graph algorithms as sparse linear algebra over user-defined semirings. BFS becomes matrix-vector multiplication $\mathbf{v}_{k+1} = A \otimes \mathbf{v}_k$ over the (OR, AND) semiring (Boolean frontier propagation). Dijkstra becomes SpMV over the (min, +) tropical semiring. This unification enables highly optimised GPU implementations.

**GNN aggregation as SpMM.** The GNN message-passing step $H^{[l+1]} = \sigma(\hat{A} H^{[l]} W)$ is a **Sparse Matrix times Dense Matrix** (SpMM) operation. Modern frameworks (PyG, DGL) implement this as a single GPU kernel. The time complexity is $O(m \cdot d)$ where $d$ is the feature dimension — identical to running BFS with a $d$-dimensional state vector per node.

### 8.4 Approximation Algorithms

**Metric TSP (2-approximation via MST).**
Given $n$ cities with metric distances (satisfying triangle inequality), find the shortest tour visiting all cities.
- Compute MST $T$ (cost $\leq$ OPT, since deleting any tour edge gives a spanning tree)
- Perform Euler tour (DFS) of $T$, shortcutting repeated vertices
- Result: tour of cost $\leq 2 \cdot \text{MST} \leq 2 \cdot \text{OPT}$

**Greedy max-cut (0.5-approximation).**
Partition $V$ into two sets $S$ and $T$ to maximise the number of edges between $S$ and $T$.
- Greedy: for each vertex, assign it to the side that maximises its cut contribution
- Result: at least $m/2$ edges in the cut (since a random partition cuts each edge in expectation with probability $1/2$)
- Goemans-Williamson SDP relaxation achieves $\approx 0.878$ approximation — a landmark result in approximation algorithms

---

## 9. Algorithm Selection Framework

### 9.1 Decision Flowchart

```
GRAPH ALGORITHM SELECTION
════════════════════════════════════════════════════════════════════════

  What do you need?
  │
  ├─ TRAVERSAL / REACHABILITY
  │   ├─ Shortest hop-distance? ──────────────────────► BFS  O(n+m)
  │   ├─ Detect cycles? ─────────────────────────────► DFS  O(n+m)
  │   ├─ Connected components? ──────────────────────► BFS or DFS
  │   └─ SCCs (directed)? ───────────────────────────► Tarjan  O(n+m)
  │
  ├─ SHORTEST PATHS
  │   ├─ Unweighted graph? ──────────────────────────► BFS  O(n+m)
  │   ├─ Non-negative weights?
  │   │   ├─ Single source ────────────────────────► Dijkstra  O((n+m)logn)
  │   │   ├─ Single pair, heuristic known ─────────► A*  O(m logn)
  │   │   └─ All pairs, dense graph ───────────────► Floyd-Warshall  O(n³)
  │   ├─ Negative weights (no neg cycle)?
  │   │   ├─ Single source ────────────────────────► Bellman-Ford  O(nm)
  │   │   └─ All pairs, sparse graph ──────────────► Johnson's  O(nm + n²logn)
  │   └─ Need negative cycle detection? ────────────► Bellman-Ford
  │
  ├─ SPANNING TREES
  │   ├─ Sparse graph (m close to n) ────────────────► Kruskal  O(m logm)
  │   └─ Dense graph (m close to n²) ───────────────► Prim  O((n+m)logn)
  │
  ├─ DAG PROBLEMS
  │   ├─ Ordering with dependencies ────────────────► Topological Sort  O(n+m)
  │   ├─ Shortest path in DAG ──────────────────────► DAG-DP  O(n+m)
  │   └─ Critical path (project scheduling) ────────► DAG longest path  O(n+m)
  │
  └─ FLOW / CUT
      ├─ Max flow (small graph) ─────────────────────► Ford-Fulkerson
      └─ Max flow (large graph) ─────────────────────► Edmonds-Karp  O(nm²)

════════════════════════════════════════════════════════════════════════
```

### 9.2 Complexity Summary Table

| Problem | Best Algorithm | Time | Notes |
|---------|---------------|------|-------|
| Reachability | BFS/DFS | $O(n+m)$ | Linear optimal |
| Connected components | BFS/DFS | $O(n+m)$ | All components in one pass |
| Bipartiteness | BFS | $O(n+m)$ | 2-colour during BFS |
| SSSP unweighted | BFS | $O(n+m)$ | Exact hop distances |
| SSSP non-neg weights | Dijkstra | $O((n+m)\log n)$ | Binary heap |
| SSSP arbitrary weights | Bellman-Ford | $O(nm)$ | Detects neg cycles |
| SSSP negative cycle check | Bellman-Ford | $O(nm)$ | $n$-th round test |
| APSP dense | Floyd-Warshall | $O(n^3)$ | Simple DP |
| APSP sparse neg weights | Johnson's | $O(nm + n^2\log n)$ | Reweight + Dijkstra |
| MST | Kruskal / Prim | $O(m\log m)$ | Both optimal |
| DAG topological sort | Kahn / DFS | $O(n+m)$ | Linear optimal |
| SCC | Tarjan / Kosaraju | $O(n+m)$ | Linear optimal |
| Max flow | Edmonds-Karp | $O(nm^2)$ | Polynomial guarantee |
| Single-pair SP w/ heuristic | A\* | $O(m\log n)$ avg | Admissible heuristic |

### 9.3 Implementation Pitfalls

Ten common bugs in graph algorithm implementations:

1. **Integer overflow in weights.** Initialising distances to `INT_MAX` and adding a positive weight overflows to a negative number, breaking Dijkstra's invariant. Fix: use `float('inf')` (Python) or `1e18` (C++/Java).

2. **Relaxing undirected edges only once.** In Bellman-Ford on undirected graphs, each undirected edge must be treated as two directed edges. Relaxing only one direction misses valid shorter paths.

3. **Using Dijkstra on negative-weight edges.** Dijkstra is incorrect with negative weights. Symptom: incorrect distances for some vertices. Fix: use Bellman-Ford or reweight with Johnson's.

4. **Missing the negative-cycle check.** Implementing Bellman-Ford without the final check-round misses the case where the answer is $-\infty$. Always run the $(n)$-th relaxation round and check for further improvement.

5. **Off-by-one in Kahn's algorithm.** If the output list has fewer than $n$ vertices, a cycle exists — this is the cycle-detection mechanism. Not checking this means returning a partial topological order silently.

6. **Not resetting the `onStack` flag in Tarjan.** When an SCC is popped, vertices must be marked as not on the stack. Otherwise, later DFS edges to those vertices incorrectly update `low` values.

7. **Forgetting the residual backward edge in max-flow.** Ford-Fulkerson requires adding both forward and backward residual edges. Omitting the backward edge prevents flow cancellation and gives suboptimal results.

8. **BFS on multigraphs with parallel edges.** If multiple edges exist between $u$ and $v$, BFS may mark $v$ as visited via one edge and miss a shorter-weight path via another. For weighted BFS, use Dijkstra.

9. **DFS stack overflow on large graphs.** Recursive DFS overflows the call stack for deep graphs ($n > 10^4$ in Python). Fix: implement iterative DFS with an explicit stack.

10. **Ignoring disconnected graphs.** BFS/DFS from a single source only visits one component. For algorithms requiring all vertices (e.g., Kahn's topological sort, Kosaraju's SCC), always loop over all unvisited sources.


---

## 10. Why This Matters for AI (2026 Perspective)

| Algorithm | Concrete AI/ML Application |
|-----------|---------------------------|
| **BFS** | GNN $k$-hop receptive field; knowledge graph multi-hop QA; web crawling for LLM pre-training data |
| **DFS** | Cycle detection in autograd graphs; dependency resolution for model serving; parse tree traversal in NLP |
| **Dijkstra** | Shortest semantic path in knowledge graphs; road routing for embodied agents; optimal plan length in MCTS |
| **Bellman-Ford** | Value iteration in RL (Bellman equation); credit assignment in temporal-difference learning; negative-reward environments |
| **Floyd-Warshall** | All-pairs semantic distance in knowledge bases; all-pairs token co-occurrence distances; small graph APSP in graph transformers |
| **A\*** | Robot path planning (ROS navigation); game AI search (Alpha*, MCTS + heuristic); structured output generation with heuristic guidance |
| **Kruskal / Prim (MST)** | Single-linkage hierarchical clustering; minimum-cost network design; graph-based feature selection; phylogenetic tree construction |
| **Topological Sort** | PyTorch/JAX autograd forward pass; package dependency resolution (pip, conda); build system (Make, Bazel) ordering |
| **Tarjan SCC** | Circular dependency detection in ML pipelines; knowledge graph cycle analysis; 2-SAT constraint solving in neural architecture search |
| **Ford-Fulkerson / Edmonds-Karp** | Image segmentation (Boykov-Kolmogorov); network reliability; matching in bipartite graphs (Hungarian algorithm) |
| **Max-Flow / Min-Cut** | Graph partitioning for distributed GNN training; community detection; min-cut spectral connection (Cheeger inequality → §04) |
| **Graph cuts (Boykov 2001)** | Semantic segmentation before deep learning (still used as post-processing); MRF energy minimisation |
| **Parallel BFS (GPU)** | Trillion-edge social network analysis; billion-node knowledge graph traversal (NVIDIA cuGraph, GraphBLAS) |
| **SpMM (GNN aggregation)** | Every GNN forward pass: $H^{[l+1]} = \sigma(\hat{A} H^{[l]} W)$ is sparse-dense matrix multiply |

### The GNN as a Parameterised Graph Algorithm

The deepest connection between classical graph algorithms and modern AI is this:

```
GRAPH ALGORITHM → GNN GENERALISATION
════════════════════════════════════════════════════════════════════════

  BFS (hop-distance labels)           → GCN (learned node embeddings)
  BFS frontier = layer k nodes        → GNN layer k receptive field
  "distance from source"              → "representation of k-hop context"
  Fixed aggregation (count)           → Learned aggregation (attention)
  Deterministic (one answer)          → Parameterised (trained on data)

  Dijkstra (shortest path)            → Neural shortest path (learned w)
  Static weights w(u,v)               → Edge features in GNN
  Priority queue extraction           → Attention-weighted aggregation
  Exact optimal solution              → Approximate learned solution

  MST (global structure)              → Graph pooling (global summary)
  Kruskal cut/cycle decisions         → Differentiable graph pooling
  Minimum spanning structure          → Compressed graph representation

════════════════════════════════════════════════════════════════════════
```

Every GNN is a learned, parameterised, differentiable version of a classical graph algorithm. Understanding the classical algorithms provides:
1. **Interpretability:** what the GNN is computing approximations of
2. **Complexity awareness:** how many layers = how many hops the GNN can "see"
3. **Architecture design:** which aggregation functions (sum, mean, max) correspond to which classical problems
4. **Failure case prediction:** when the GNN cannot solve a problem (e.g., WL-indistinguishable graphs)

---

## 11. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
|---|---------|---------------|-----|
| 1 | Using `d[v] = 0` to mark "unvisited" in BFS | Conflicts with actual distance 0 for source; BFS fails on any source other than node 0 | Use `d[v] = ∞` (infinity) for unvisited; only `d[s] = 0` is set at initialisation |
| 2 | Applying Dijkstra to a graph with negative edge weights | Greedy extraction invariant breaks; can return incorrect distances | Use Bellman-Ford for negative weights; or reweight with Johnson's method |
| 3 | Treating DFS finish-order as topological order (not reversing) | DFS finish order is the *reverse* topological order | Reverse the finish order list, or use Kahn's algorithm directly |
| 4 | Confusing BFS shortest paths on weighted graphs | BFS counts hops, not weighted distances; gives wrong answers for weighted SSSP | Use Dijkstra (non-neg weights) or Bellman-Ford (arbitrary weights) |
| 5 | Running Kruskal on a multigraph without handling parallel edges | Parallel edges can all be added if they connect different components; deduplication may be needed | Pre-process: keep only minimum-weight edge between each node pair if MST is the goal |
| 6 | Declaring a graph a DAG after just one DFS component | DFS from one source only checks reachable vertices; cycles elsewhere go undetected | Run DFS from *all* unvisited vertices (outer loop) and check for back edges globally |
| 7 | Floyd-Warshall with `D[i][i] = ∞` initial value | Diagonal should be 0 (distance from a node to itself); incorrect initialisations propagate errors throughout all iterations | Initialise `D[i][i] = 0` for all $i$ |
| 8 | Max-flow without bidirectional residual edges | Cancellation of flow is impossible; algorithm gets stuck at suboptimal solutions | For every edge $(u,v)$ with capacity $c$, add a reverse edge $(v,u)$ with initial capacity 0 in the residual graph |
| 9 | Prim's algorithm applied to a disconnected graph | Prim always starts from one vertex and grows; disconnected vertices are never added to the MST | Verify graph is connected before running Prim, or run Kruskal (which works on disconnected graphs, yielding a minimum spanning forest) |
| 10 | Bellman-Ford stopping early when no relaxation occurs in a round | In the presence of negative edges, relaxation may restart later; early stopping can miss shorter paths | Only apply early stopping if a full round passes with zero relaxations — this is an optimisation, but must complete all $n-1$ rounds first |
| 11 | A\* with an inadmissible heuristic | An overestimating heuristic causes A\* to expand a non-optimal path first, missing the true shortest path | Always verify $h(v) \leq \delta(v,t)$ (admissibility); Euclidean distance is safe for Euclidean graphs |
| 12 | SCCs on an undirected graph using SCC algorithm | Every edge in an undirected graph is a bidirectional path; every connected component is a trivial SCC | For undirected graphs, simply find connected components with BFS/DFS |

---

## 12. Exercises

**Exercise 1 ★ — BFS Shortest Paths**
Consider the undirected graph with adjacency list:
$0: [1, 2]$, $1: [0, 3, 4]$, $2: [0, 4]$, $3: [1, 5]$, $4: [1, 2, 5]$, $5: [3, 4]$.

(a) Run BFS from source $s = 0$. Show the queue state at each step and the resulting distance array $d[\cdot]$.
(b) Find all shortest paths from $0$ to $5$.
(c) How many distinct shortest paths are there from $0$ to $5$? Enumerate them.
(d) Implement BFS in Python and verify your answer.

**Exercise 2 ★ — DFS Edge Classification**
Consider the directed graph with edges: $1 \to 2$, $1 \to 3$, $2 \to 4$, $3 \to 2$, $4 \to 3$, $4 \to 5$.

(a) Run DFS starting from vertex 1. Record disc[] and fin[] timestamps.
(b) Classify each edge as tree, back, forward, or cross edge.
(c) Does this graph contain a cycle? Identify it.
(d) Is a topological ordering possible? Explain why or why not.

**Exercise 3 ★ — Dijkstra's Algorithm**
Run Dijkstra's algorithm on the weighted directed graph with edges:
$(s,a,10), (s,b,3), (b,a,4), (b,c,8), (a,c,1), (c,d,2), (a,d,7)$.

(a) Show the priority queue state after each EXTRACT-MIN.
(b) Give the shortest path tree (parent[] array).
(c) What is the shortest path from $s$ to $d$?
(d) Why would Dijkstra fail if edge $(b,a)$ had weight $-6$? What is the correct answer, and which algorithm finds it?

**Exercise 4 ★★ — Bellman-Ford and DP**
Given the directed graph with edges $(s,u,5), (s,v,8), (u,v,-3), (u,t,6), (v,t,2)$:

(a) Show the Bellman-Ford DP table $d_i[v]$ for $i = 0, 1, 2, 3$ (rows = iterations, columns = vertices).
(b) Verify the final distances match the true shortest paths.
(c) Add edge $(t,u,-10)$. What happens to Bellman-Ford? Show the behaviour in the $n$-th round.
(d) Explain the connection between Bellman-Ford's update rule and the Bellman equation for value iteration in RL.

**Exercise 5 ★★ — Kruskal's MST with Union-Find**
Implement Kruskal's algorithm with Union-Find (union by rank + path compression) for the graph with edges:
$(A,B,4), (A,C,2), (B,C,5), (B,D,10), (C,D,3), (D,E,7), (C,E,8)$.

(a) Sort edges by weight and show each Union-Find decision.
(b) Give the final MST and its total weight.
(c) Verify the MST satisfies the cut property for the cut $(\{A,C\}, \{B,D,E\})$.
(d) How many distinct MSTs does this graph have? (Hint: are all edge weights distinct?)

**Exercise 6 ★★ — Topological Sort and Autograd**
Consider a computation graph (DAG) with operations:
$\text{matmul}_1: \{W_1, x\} \to z_1$, $\text{relu}: \{z_1\} \to a_1$, $\text{matmul}_2: \{W_2, a_1\} \to z_2$, $\text{loss}: \{z_2, y\} \to L$.

(a) Draw the computation DAG (nodes = tensors and operations).
(b) Run Kahn's topological sort. What is the in-degree of each node at the start?
(c) Give the topological order of operations for the forward pass.
(d) Give the reverse topological order for the backward pass.
(e) Which gradients are accumulated (summed) rather than simply propagated? Why?

**Exercise 7 ★★★ — Tarjan's SCC**
Run Tarjan's algorithm on the directed graph with edges:
$0 \to 1$, $1 \to 2$, $2 \to 0$, $1 \to 3$, $3 \to 4$, $4 \to 5$, $5 \to 3$, $2 \to 6$, $6 \to 7$.

(a) Trace disc[] and low[] values throughout the DFS.
(b) Identify when each SCC is detected (when low[v] = disc[v]).
(c) List all SCCs and draw the condensation DAG.
(d) Which SCC is a "source" (in-degree 0 in condensation) and which is a "sink"?

**Exercise 8 ★★★ — Max-Flow and the Min-Cut Theorem**
Consider the flow network with source $s$, sink $t$, and edges:
$(s,a,10), (s,b,8), (a,c,7), (a,d,5), (b,c,6), (b,d,4), (c,t,9), (d,t,6)$.

(a) Find a maximum $s$-$t$ flow using Ford-Fulkerson. Show the residual graph after each augmenting path.
(b) What is the maximum flow value?
(c) Find a minimum $s$-$t$ cut $(S, V\setminus S)$ and verify its capacity equals the max flow.
(d) Describe how this max-flow instance models a data routing problem: $s$ = data centre, $a,b,c,d$ = relay nodes, $t$ = users. What does the min-cut represent physically?

---

## 13. Conceptual Bridge

### Backward: What This Section Builds On

Graph algorithms are the *operational layer* on top of the structures defined in §01 and §02. Every algorithm in this section assumes familiarity with:
- **§01 Graph Basics:** the definitions of paths, cycles, connectivity, and DAGs that make algorithm statements precise
- **§02 Graph Representations:** the choice of adjacency list vs adjacency matrix vs CSR directly determines algorithm complexity — BFS on an adjacency list is $O(n+m)$; on an adjacency matrix it would be $O(n^2)$ even for sparse graphs

The algorithms also draw on core data structures: priority queues (Dijkstra, Prim, A\*), Union-Find (Kruskal), stacks (DFS, Tarjan), and queues (BFS, Kahn). These are assumed from a standard algorithms course.

### Forward: What This Section Enables

**§04 Spectral Graph Theory** takes the graph objects studied here and analyses them through eigenvalues. The Cheeger inequality connects the min-cut (§7.4 above) to the Fiedler eigenvalue $\lambda_2$ of the Laplacian. Spectral clustering finds an approximate min-cut by computing $\lambda_2$'s eigenvector — a bridge from combinatorial algorithms (§03) to algebraic methods (§04).

**§05 Graph Neural Networks** makes the algorithmic connections explicit: GNN layers are parameterised, differentiable generalisations of BFS aggregation. Every classical algorithm in this section has a GNN counterpart: BFS → GCN, Dijkstra → neural shortest paths, MST → differentiable graph pooling. The expressiveness limitations of GNNs (WL test) are precisely characterised in terms of how many BFS hops the network can simulate.

**§22 Causal Inference** uses DAG algorithms — topological sort, d-separation, Markov blanket computation — to reason about causal structure. The do-calculus operates on causal DAGs; interventions modify the DAG structure; counterfactuals involve path analysis.

```
CURRICULUM POSITION: §03 GRAPH ALGORITHMS
════════════════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────────┐
  │  CH. 11  GRAPH THEORY                                          │
  │                                                                 │
  │  §01 Graph Basics        §02 Graph Representations            │
  │  (vocabulary)            (data structures)                     │
  │            ↓                    ↓                             │
  │       ╔══════════════════════════════════╗                    │
  │       ║  §03 GRAPH ALGORITHMS  ← YOU ARE HERE                 ║
  │       ║  BFS · DFS · Dijkstra · Bellman-Ford                  ║
  │       ║  Kruskal · Prim · Topo Sort · SCC                     ║
  │       ║  Ford-Fulkerson · Max-Flow / Min-Cut                   ║
  │       ╚══════════════════════════════════╝                    │
  │                    ↓                                           │
  │       §04 Spectral Graph Theory                               │
  │       (Laplacian eigenvalues, Fiedler, spectral clustering)    │
  │                    ↓                                           │
  │       §05 Graph Neural Networks                               │
  │       (MPNN, GCN, GAT — learned graph algorithms)             │
  │                    ↓                                           │
  │       §06 Random Graphs                                       │
  │       (probabilistic graph models, phase transitions)          │
  └─────────────────────────────────────────────────────────────────┘

  Prerequisite chains into this section:
    §01 ─── definitions, paths, DAGs
    §02 ─── adjacency list (O(n+m) complexity)
    Ch.02 ── matrix multiply (Floyd-Warshall, adjacency matrix BFS)

  Where this section feeds:
    §04 ─── min-cut ↔ Cheeger constant; BFS layers ↔ graph spectrum
    §05 ─── BFS = GNN receptive field; topo sort = autograd ordering
    §22 ─── DAG algorithms for causal inference

════════════════════════════════════════════════════════════════════════
```

### Further Reading

1. **Cormen, Leiserson, Rivest, Stein** — *Introduction to Algorithms* (CLRS), Ch. 22–26: the definitive reference for BFS, DFS, SSSP, MST, max-flow with complete proofs
2. **Kleinberg & Tardos** — *Algorithm Design*, Ch. 3–4, 7: excellent motivating examples for graph algorithms and flows
3. **Sedgewick & Wayne** — *Algorithms* (4th ed.), Part II: practical Java implementations with visualisations
4. **Kepner & Gilbert** — *Graph Algorithms in the Language of Linear Algebra* (2011): GraphBLAS and semiring formulations
5. **Boykov & Kolmogorov** — "An Experimental Comparison of Min-Cut/Max-Flow Algorithms for Energy Minimization in Vision" (PAMI 2004): graph cuts for CV
6. **Leskovec, Rajaraman & Ullman** — *Mining of Massive Datasets*, Ch. 10: graph algorithms at web scale

[← Back to Graph Theory](../README.md) | [Previous: Graph Representations ←](../02-Graph-Representations/notes.md) | [Next: Spectral Graph Theory →](../04-Spectral-Graph-Theory/notes.md)

---

## Appendix A: Worked Traces

### A.1 Complete Dijkstra Trace

**Graph.** Directed, nodes $\{s, a, b, c, d\}$, edges: $(s,a,4)$, $(s,b,2)$, $(b,a,1)$, $(b,c,5)$, $(a,d,3)$, $(c,d,1)$.

**Initialisation:** $d = \{s:0, a:\infty, b:\infty, c:\infty, d:\infty\}$; $Q = \{s,a,b,c,d\}$ keyed by $d$.

| Step | Extract | d[s] | d[a] | d[b] | d[c] | d[d] | Relaxed |
|------|---------|------|------|------|------|------|---------|
| 0 | — | 0 | ∞ | ∞ | ∞ | ∞ | init |
| 1 | s (0) | 0 | 4 | 2 | ∞ | ∞ | s→a=4, s→b=2 |
| 2 | b (2) | 0 | 3 | 2 | 7 | ∞ | b→a: 2+1=3<4 ✓, b→c: 2+5=7 |
| 3 | a (3) | 0 | 3 | 2 | 7 | 6 | a→d: 3+3=6 |
| 4 | c (7) | 0 | 3 | 2 | 7 | 6 | c→d: 7+1=8>6, no update |
| 5 | d (6) | 0 | 3 | 2 | 7 | 6 | no outgoing edges |

**Result:** $d = \{s:0, a:3, b:2, c:7, d:6\}$.
- Shortest path to $d$: $s \to b \to a \to d$ (cost $2+1+3=6$). ✓
- Key observation: $b$ was extracted before $a$ (2 < 4), and when $b$ was processed, it improved $d[a]$ from 4 to 3. This is the power of Dijkstra's greedy approach — later extraction can improve earlier estimates.

### A.2 Complete Bellman-Ford Trace

**Graph.** $n=4$ nodes $\{s,u,v,t\}$, edges: $(s,u,6)$, $(s,v,7)$, $(u,v,8)$, $(u,t,-4)$, $(v,t,9)$, $(u,w,-2)$ where $w=v$, $(v,u,-3)$.

Simplified: nodes $\{s,a,b,t\}$, edges $(s,a,5)$, $(s,b,6)$, $(a,b,-4)$, $(b,a,-3)$ — note: this creates a negative cycle $a \to b \to a$ of weight $-7$.

Let's use a clean no-neg-cycle example: nodes $\{s,a,b,t\}$, edges $(s,a,3)$, $(s,b,6)$, $(a,t,7)$, $(b,a,-2)$, $(b,t,4)$.

| Round | d[s] | d[a] | d[b] | d[t] |
|-------|------|------|------|------|
| 0 (init) | 0 | ∞ | ∞ | ∞ |
| 1 | 0 | 3 | 6 | min(∞,3+7,6+4)=10 |
| 2 | 0 | min(3,6-2)=3 | 6 | min(10,3+7,6+4)=10 |
| 3 | 0 | 3 | 6 | 10 |

**Result:** $d[t] = 10$ via $s \to b \to t$ (cost $6+4$) or $s \to a \to t$ (cost $3+7$) — both give 10. No negative cycle (round 3 makes no changes). ✓

### A.3 Tarjan SCC: Detailed Trace

**Graph.** Nodes $\{0,1,2,3,4\}$, edges: $0 \to 1$, $1 \to 2$, $2 \to 0$ (cycle), $1 \to 3$, $3 \to 4$, $4 \to 1$ (cycle).

DFS from node 0:

```
Visit 0: disc[0]=0, low[0]=0, stack=[0]
  Visit 1: disc[1]=1, low[1]=1, stack=[0,1]
    Visit 2: disc[2]=2, low[2]=2, stack=[0,1,2]
      Edge 2→0: 0 is on stack → low[2] = min(2,disc[0]) = 0
    Finish 2: low[2]=0 ≠ disc[2]=2 → not SCC root
    Back-propagate: low[1] = min(1,low[2]) = 0
    Visit 3: disc[3]=3, low[3]=3, stack=[0,1,2,3]
      Visit 4: disc[4]=4, low[4]=4, stack=[0,1,2,3,4]
        Edge 4→1: 1 is on stack → low[4] = min(4,disc[1]) = 1
      Finish 4: low[4]=1 ≠ disc[4]=4 → not SCC root
      Back-propagate: low[3] = min(3,low[4]) = 1
    Finish 3: low[3]=1 ≠ disc[3]=3 → not SCC root
    Back-propagate: low[1] = min(0,low[3]) = 0
  Finish 1: low[1]=0 ≠ disc[1]=1 → not SCC root
  Back-propagate: low[0] = min(0,low[1]) = 0
Finish 0: low[0]=0 = disc[0]=0 → SCC ROOT!
  Pop stack until 0: {0,1,2,3,4} → but wait...
```

Actually node 0 is the first on the stack. All nodes $\{0,1,2,3,4\}$ pop as one SCC only if they're all mutually reachable. Let's verify: $0 \to 1 \to 3 \to 4 \to 1$ — is 3 reachable from 0? Yes. Is 0 reachable from 3? $3 \to 4 \to 1 \to 2 \to 0$. Yes! So SCCs: $\{0,1,2,3,4\}$ form one SCC in this graph.

For distinct SCCs, use the graph from Exercise 7: $0\to1\to2\to0$ (SCC $\{0,1,2\}$), $3\to4\to5\to3$ (SCC $\{3,4,5\}$), links $2\to6$, $6\to7$ (singletons $\{6\}$, $\{7\}$).

---

## Appendix B: Complexity Deep Dives

### B.1 Why Dijkstra is O((n+m) log n)

The binary-heap-based priority queue supports:
- `INSERT` in $O(\log n)$ — adding $n$ vertices initially: $O(n \log n)$
- `EXTRACT-MIN` in $O(\log n)$ — called $n$ times: $O(n \log n)$
- `DECREASE-KEY` in $O(\log n)$ — called at most once per edge: $O(m \log n)$

Total: $O((n+m)\log n)$.

With a **Fibonacci heap**, DECREASE-KEY runs in $O(1)$ amortised: total $O(m + n\log n)$. Theoretically superior for dense graphs ($m = \Theta(n^2)$), but the large constants make Fibonacci heaps impractical in most implementations.

**Practical consideration.** For road networks (small degree $\approx 2.5$), $m = O(n)$, so Dijkstra is $O(n\log n)$. With contraction hierarchies, query time drops to milliseconds for continent-scale road graphs.

### B.2 Why Floyd-Warshall is Correct: DP Subproblem

Let $d^{(k)}[i][j]$ = length of shortest path from $i$ to $j$ using only vertices $\{1,\ldots,k\}$ as intermediates.

**Base:** $d^{(0)}[i][j] = w(i,j)$ if $(i,j)\in E$, 0 if $i=j$, $\infty$ otherwise.

**Transition:** Either the optimal path from $i$ to $j$ via $\{1,\ldots,k\}$ uses vertex $k$, or it doesn't.
$$d^{(k)}[i][j] = \min\underbrace{d^{(k-1)}[i][j]}_{\text{don't use }k},\; \underbrace{d^{(k-1)}[i][k] + d^{(k-1)}[k][j]}_{\text{use }k}$$

After $n$ iterations (all vertices considered as potential intermediates), $d^{(n)}[i][j] = \delta(i,j)$. $\square$

**In-place update.** The transition $d^{(k)} \to d^{(k+1)}$ can be done in-place (the same matrix) because $d^{(k)}[i][k] = d^{(k-1)}[i][k]$ and $d^{(k)}[k][j] = d^{(k-1)}[k][j]$ — the row $k$ and column $k$ do not change in iteration $k$.

### B.3 Union-Find Complexity: Inverse Ackermann

The **inverse Ackermann function** $\alpha(n)$ grows so slowly that $\alpha(n) \leq 4$ for any $n$ realistically encountered in practice ($\alpha(n) > 4$ only for $n > 2^{2^{2^{65536}}}$). Thus Union-Find with path compression + union by rank runs in $O(\alpha(n))$ amortised per operation — effectively $O(1)$.

**Path compression.** During `FIND(x)`, after finding the root $r$, set the parent of every node on the path $x \to r$ directly to $r$. This flattens the tree for future queries.

**Union by rank.** When merging two trees, attach the shallower tree's root as a child of the taller tree's root. With rank as an upper bound on height, no tree gets taller than $O(\log n)$.

Together, these two optimisations achieve $O(n \alpha(n))$ for any sequence of $n$ operations — proven by Tarjan (1975), completing the complexity analysis of Kruskal's algorithm.

### B.4 Max-Flow Integrality and Network Applications

**Integrality theorem.** If all edge capacities in a flow network are integers, then there exists a maximum flow that is integer-valued on every edge. (Proof: Ford-Fulkerson augments by integer amounts when all capacities and current flows are integers; by induction, the final flow is integer-valued.)

**Bipartite matching via max-flow.** Given a bipartite graph $G = (L \cup R, E)$, add:
- Source $s$ with edge $(s, l, 1)$ for all $l \in L$
- Sink $t$ with edge $(r, t, 1)$ for all $r \in R$
- All original bipartite edges with capacity 1

Maximum flow = maximum matching (by integrality). This reduces bipartite matching to max-flow and solves it in $O(n \cdot m)$ via Ford-Fulkerson, or $O(m\sqrt{n})$ via the specialised Hopcroft-Karp algorithm.

**Hall's theorem connection.** By the max-flow min-cut theorem, a bipartite graph has a perfect matching iff every subset $S \subseteq L$ has $|N(S)| \geq |S|$ (Hall's condition). The min-cut in the flow network corresponds exactly to the violating set $S$.


---

## Appendix C: Connections to Linear Algebra

### C.1 BFS as Matrix-Vector Multiplication

BFS can be reformulated as repeated matrix-vector multiplication over the Boolean semiring (OR, AND).

Let $\mathbf{v} \in \{0,1\}^n$ be the current frontier (1 = in frontier, 0 = not). Let $A$ be the adjacency matrix (Boolean). Then:
$$\mathbf{v}_{k+1} = (A \cdot \mathbf{v}_k) \setminus \bigcup_{i \leq k} \mathbf{v}_i$$
where $\cdot$ is Boolean matrix-vector multiply and $\setminus$ removes already-visited nodes.

This formulation maps directly to the **GraphBLAS** interface: `vxm` (vector-times-matrix) over a user-defined semiring. For BFS, the semiring is (OR, AND). For shortest paths, the semiring is (min, +) — the **tropical semiring**.

### C.2 Shortest Paths via Tropical Algebra

Define the **tropical semiring** $(\mathbb{R}_{\geq 0} \cup \{\infty\}, \oplus, \otimes)$ where:
- $a \oplus b = \min(a, b)$ (tropical addition)
- $a \otimes b = a + b$ (tropical multiplication)
- Identity for $\oplus$: $\infty$; Identity for $\otimes$: 0

The **tropical matrix product** of weight matrices:
$$(A \otimes B)[i][j] = \bigoplus_k A[i][k] \otimes B[k][j] = \min_k (A[i][k] + B[k][j])$$

**Floyd-Warshall** computes the $n$-th power of the weight matrix $W$ in the tropical semiring:
$$W^{\otimes n}[i][j] = \delta(i,j)$$
i.e., the tropical matrix power gives all-pairs shortest paths.

**Bellman-Ford** computes tropical matrix-vector products:
$$\mathbf{d}^{(k)} = W^{\otimes k} \otimes \mathbf{d}^{(0)}$$

This algebraic view unifies shortest-path algorithms under the theory of algebraic path problems. The same framework extends to other graph algorithms by changing the semiring: (max, min) for widest paths, (max, ×) for maximum reliability paths.

### C.3 The Adjacency Matrix Power and Walk Counting

From §02, recall that $(A^k)[i][j]$ counts the number of walks of length exactly $k$ from $i$ to $j$.

**BFS connection.** The first $k$ for which $(A^k)[s][v] > 0$ and $(A^{k-1})[s][v] = 0$ is exactly $d(s,v)$ — the BFS distance from $s$ to $v$.

**Reachability.** Over the Boolean semiring: $(A \vee A^2 \vee \cdots \vee A^n)[i][j] = 1$ iff $j$ is reachable from $i$ in $\leq n$ hops. Computing this is the transitive closure problem, equivalent to Floyd-Warshall over Booleans.

**GNN aggregation.** A $k$-layer GNN with linear activations computes:
$$H^{[k]} = (\hat{A})^k H^{[0]} \prod_{l=0}^{k-1} W^{[l]}$$

The matrix $(\hat{A})^k[i][j]$ measures how much node $j$ contributes to node $i$'s representation — proportional to the number of weighted walks of length $k$ from $j$ to $i$. This is the algebraic content of "receptive field = BFS $k$-hop neighbourhood."

---

## Appendix D: Graph Algorithms in Practice

### D.1 Python Ecosystem

| Library | Algorithms | Strength |
|---------|-----------|---------|
| `networkx` | All classical algorithms | Easy to use, educational; slow for large graphs |
| `igraph` | BFS, DFS, SP, MST, community | C-backed; 10-100× faster than NetworkX |
| `scipy.sparse` | SpMV, matrix-vector BFS | Efficient sparse matrix operations |
| `cuGraph` (NVIDIA) | BFS, SSSP, PageRank, MST | GPU-accelerated; for billion-node graphs |
| `PyG` / `DGL` | GNN message passing (SpMM) | Deep learning on graphs; CUDA support |
| `rustworkx` (IBM) | All classical algorithms | Rust-backed; used in Qiskit circuits |

### D.2 Choosing Data Structure for Each Algorithm

| Algorithm | Best Graph Representation | Why |
|-----------|--------------------------|-----|
| BFS | Adjacency list | $O(1)$ neighbour enumeration |
| DFS | Adjacency list | $O(1)$ neighbour enumeration |
| Dijkstra | Adjacency list + min-heap | Neighbour access + priority queue |
| Bellman-Ford | Edge list | Iterate over all edges directly |
| Floyd-Warshall | Adjacency matrix | Direct index $D[i][j]$ |
| Kruskal | Edge list (sorted) | Sort edges, then Union-Find |
| Prim | Adjacency list + min-heap | Same structure as Dijkstra |
| Topological Sort | Adjacency list + in-degree array | In-degree update per edge |
| Tarjan SCC | Adjacency list | DFS neighbour enumeration |
| Ford-Fulkerson | Residual adjacency list | Forward and backward edge access |

### D.3 Scale Reference

| Scale | Nodes | Edges | Recommended Approach |
|-------|-------|-------|---------------------|
| Tiny | $n < 100$ | $m < 1000$ | Any algorithm, NetworkX |
| Small | $n < 10^4$ | $m < 10^6$ | Python + NumPy/SciPy |
| Medium | $n < 10^6$ | $m < 10^8$ | C++/Rust, igraph, optimised adjacency list |
| Large | $n < 10^9$ | $m < 10^{11}$ | Distributed (GraphX, Pregel), or GPU (cuGraph) |
| Massive | $n > 10^9$ | $m > 10^{11}$ | Streaming algorithms, sketches, sampling |

The Facebook social graph (2012): $n \approx 10^9$ nodes, $m \approx 10^{11}$ edges. BFS on this graph requires distributed computing — each frontier step involves terabytes of edge data.

### D.4 NetworkX Quick Reference

```python
import networkx as nx

# Create graph
G = nx.DiGraph()
G.add_weighted_edges_from([(0,1,4), (0,2,2), (1,3,5), (2,1,1), (2,3,8)])

# BFS
bfs_layers = dict(nx.bfs_layers(G, 0))   # {node: layer}
bfs_tree = nx.bfs_tree(G, 0)

# DFS
dfs_edges = list(nx.dfs_edges(G, 0))
has_cycle = not nx.is_directed_acyclic_graph(G)

# Shortest paths
dist = nx.single_source_dijkstra_path_length(G, 0, weight='weight')
path = nx.dijkstra_path(G, 0, 3, weight='weight')

# MST (undirected only)
Gu = G.to_undirected()
mst = nx.minimum_spanning_tree(Gu, weight='weight')

# Topological sort (DAG only)
topo = list(nx.topological_sort(G))

# SCC
sccs = list(nx.strongly_connected_components(G))

# Max flow
R = nx.maximum_flow(G, 0, 3, capacity='weight')
```


---

## Appendix E: Proofs and Mathematical Details

### E.1 Correctness of Kruskal's Algorithm

**Theorem.** Kruskal's algorithm produces an MST of $G$.

**Proof.** We use the **cut property**: if $e$ is the minimum-weight edge crossing some cut $(S, V\setminus S)$, then $e$ belongs to some MST.

Let $T_K$ be the spanning tree produced by Kruskal. We show $T_K$ is an MST by showing it has the same weight as any MST $T^*$.

Consider the first edge $e = (u,v)$ in Kruskal's sorted order that is in $T_K$ but not $T^*$. Adding $e$ to $T^*$ creates a cycle $C$. Since $T^*$ is a spanning tree, $C$ must contain another edge $e' \neq e$ that crosses the cut $(S, V\setminus S)$ where $S$ is the component containing $u$ when $e$ was processed. Since $e$ was chosen by Kruskal (minimum-weight crossing edge), $w(e) \leq w(e')$. Replace $e'$ with $e$ in $T^*$: the new tree has weight $\leq$ that of $T^*$. Since $T^*$ is an MST, $w(e) = w(e')$, and the new tree is also an MST.

By repeating this argument for each differing edge, we transform $T^*$ into $T_K$ without increasing weight. Hence $T_K$ is an MST. $\square$

### E.2 Correctness of Bellman-Ford

**Theorem.** After $n-1$ iterations of Bellman-Ford from source $s$ in a graph with no negative-weight cycles, $d[v] = \delta(s,v)$ for all $v$.

**Proof.** Let $v$ be any vertex with $\delta(s,v) < \infty$, and let $p = \langle s = v_0, v_1, \ldots, v_k = v \rangle$ be a shortest path (which exists and has $\leq n-1$ edges, since shortest paths are simple for non-negative or no-negative-cycle graphs).

By induction on $i$: after iteration $i$, $d[v_i] = \delta(s, v_i)$.

- $i=0$: $d[s] = 0 = \delta(s,s)$.
- Inductive step: After iteration $i$, $d[v_i] = \delta(s,v_i)$. In iteration $i+1$, edge $(v_i, v_{i+1})$ is relaxed: $d[v_{i+1}] \leftarrow \min(d[v_{i+1}], d[v_i] + w(v_i,v_{i+1})) = \min(d[v_{i+1}], \delta(s,v_i) + w(v_i,v_{i+1})) = \min(d[v_{i+1}], \delta(s,v_{i+1}))$. Since $d[v_{i+1}] \geq \delta(s,v_{i+1})$ always, we get $d[v_{i+1}] = \delta(s,v_{i+1})$.

After $k \leq n-1$ iterations, $d[v] = d[v_k] = \delta(s,v)$. $\square$

### E.3 Max-Flow Min-Cut: Complete Proof

**Theorem.** In a flow network $G$, $\max_f |f| = \min_{(S,T)} c(S,T)$.

**Proof.** We prove three equivalent statements:
1. $f$ is a maximum flow.
2. The residual graph $G_f$ has no augmenting path from $s$ to $t$.
3. $|f| = c(S,T)$ for some $s$-$t$ cut $(S,T)$.

$(1) \Rightarrow (2)$: If there were an augmenting path, we could increase $|f|$, contradicting maximality.

$(2) \Rightarrow (3)$: If no $s \leadsto t$ path in $G_f$, let $S = \{v : s \leadsto v \text{ in } G_f\}$. Then $s \in S$, $t \notin S$, so $(S, V\setminus S)$ is an $s$-$t$ cut. For every edge $(u,v)$ with $u \in S$, $v \notin S$: we need $c_f(u,v) = 0$, i.e., $f(u,v) = c(u,v)$ (saturated). For every edge $(v,u)$ with $v \notin S$, $u \in S$: $c_f(v,u) = 0$, i.e., $f(v,u) = 0$ (no backward cancellation possible). Therefore:
$$|f| = \sum_{u\in S, v\notin S} f(u,v) - \sum_{v\notin S, u\in S} f(v,u) = \sum_{u\in S,v\notin S} c(u,v) - 0 = c(S,T)$$

$(3) \Rightarrow (1)$: By weak duality, $|f| \leq c(S,T)$ for any cut. If $|f| = c(S,T)$, then $f$ achieves the lower bound on cut capacity — no flow can exceed the cut capacity, so $f$ is maximum. $\square$

### E.4 DFS Parenthesis Theorem (Full Proof)

**Theorem.** For any DFS of a graph $G$, and any two vertices $u \neq v$: the intervals $[\text{disc}[u], \text{fin}[u]]$ and $[\text{disc}[v], \text{fin}[v]]$ are either **disjoint** or one **properly contains** the other.

**Proof.** WLOG assume $\text{disc}[u] < \text{disc}[v]$ (u is discovered before v).

**Case 1: $\text{disc}[v] > \text{fin}[u]$** — $v$ is discovered after $u$ finishes. The intervals $[disc[u], fin[u]]$ and $[disc[v], fin[v]]$ are disjoint.

**Case 2: $\text{disc}[v] < \text{fin}[u]$** — $v$ is discovered while $u$ is still on the stack (GREY). Then $v$ must be a descendant of $u$ in the DFS tree (because $u$ is still open when $v$ is opened, meaning $v$ is inside $u$'s subtree). Consequently, $\text{fin}[v] < \text{fin}[u]$ — $v$ finishes before $u$ finishes. So $[\text{disc}[v], \text{fin}[v]] \subset [\text{disc}[u], \text{fin}[u]]$. $\square$

**Corollary.** $v$ is a descendant of $u$ in the DFS tree iff $[\text{disc}[v], \text{fin}[v]] \subset [\text{disc}[u], \text{fin}[u]]$.

---

## Appendix F: Algorithm Summaries (Cheat Sheet)

```
BFS:           Queue + visited set.  O(n+m).  SSSP (unweighted).
DFS:           Stack/recursion + timestamps.  O(n+m).  Cycle detection, SCC.
Dijkstra:      Min-heap + relax.  O((n+m)logn).  SSSP non-neg weights.
Bellman-Ford:  n-1 rounds, relax all edges.  O(nm).  SSSP any weights.
Floyd-Warshall: DP D[i][k]+D[k][j].  O(n³).  APSP.
A*:            Dijkstra + heuristic h(v).  O(m logn).  Single-pair SP.
Kruskal:       Sort edges + Union-Find.  O(m logm).  MST.
Prim:          Min-heap from any root.  O((n+m)logn).  MST.
Kahn:          In-degree queue.  O(n+m).  Topological sort.
Tarjan SCC:    DFS + low-link + stack.  O(n+m).  SCC directed graph.
Ford-Fulkerson: Residual graph + augmenting paths.  O(Ef). Max flow.
Edmonds-Karp:  BFS augmenting paths.  O(nm²).  Max flow poly guarantee.

KEY INVARIANTS:
  Dijkstra:   d[u] = δ(s,u) when u extracted.
  Bellman-Ford: after k rounds, d[v] = δ via ≤k edges.
  Kruskal:    each added edge is min-weight crossing some cut.
  Tarjan:     v is SCC root iff low[v] = disc[v].
  MaxFlow:    optimal iff no augmenting path in residual graph.
```


---

## Appendix G: Extended AI Connections

### G.1 Reinforcement Learning as Graph Search

The core of reinforcement learning can be understood as graph search on the **state-action graph** (Markov Decision Process):
- **Nodes:** states $s \in \mathcal{S}$
- **Edges:** actions $a \in \mathcal{A}$, weighted by $-R(s,a)$ (negative reward = cost)
- **Goal:** find a policy $\pi: \mathcal{S} \to \mathcal{A}$ minimising expected total cost (equivalently, maximising expected reward)

| RL Concept | Graph Algorithm Analogue |
|-----------|-------------------------|
| State $s$ | Node $v$ |
| Action $a$ | Edge $(s, s')$ |
| Reward $R(s,a)$ | Negative edge weight $-w(s,a)$ |
| Value function $V^\pi(s)$ | Shortest path distance $d(s,t)$ |
| Bellman equation (DP) | Bellman-Ford relaxation |
| Value iteration | Bellman-Ford until convergence |
| Policy improvement | Dijkstra (greedy shortest path) |
| $Q$-function $Q(s,a)$ | Edge-based distance labelling |
| Model-based RL | Explicit graph construction + SSSP |
| Model-free RL | Online Bellman-Ford with sampled edges |

**Value iteration** is Bellman-Ford on the MDP graph:
$$V_{k+1}(s) = \max_{a \in \mathcal{A}} \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_k(s') \right]$$
The $\max$ is the Bellman-Ford relaxation; $\gamma < 1$ ensures convergence even without a "goal" node.

### G.2 Transformers and Graph Attention

The Transformer attention mechanism (Vaswani et al., 2017) can be interpreted as a graph algorithm on the **full-attention graph** (complete bipartite graph over tokens).

In self-attention:
- Each token $i$ is a node
- Edge weights $\alpha_{ij} = \text{softmax}(\mathbf{q}_i \cdot \mathbf{k}_j / \sqrt{d})$ are learned attention scores
- Aggregation: $\mathbf{h}_i = \sum_j \alpha_{ij} \mathbf{v}_j$ — a weighted sum over all tokens

This is equivalent to one step of **graph attention** on the complete graph. Sparse transformers (Longformer, BigBird) restrict attention to a sparse graph, reducing complexity from $O(n^2)$ to $O(n \log n)$ or $O(n)$ — precisely the graph-algorithmic idea of working with sparse adjacency structures.

**Graph Attention Networks (GAT, Veličković et al. 2018)** apply the same attention mechanism to graph neighbourhoods:
$$\alpha_{ij} = \text{softmax}_j\left(\text{LeakyReLU}\!\left(\mathbf{a}^\top [\mathbf{W}\mathbf{h}_i \| \mathbf{W}\mathbf{h}_j]\right)\right)$$
$$\mathbf{h}_i' = \sigma\!\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{W} \mathbf{h}_j\right)$$

The support is restricted to the graph's actual edge set — sparse attention on the graph's adjacency structure. The algorithm is BFS aggregation with learned, node-pair-specific weights.

### G.3 Knowledge Graph Reasoning as Shortest Path

Knowledge graphs encode entities and relations as directed, labelled edges: (entity1, relation, entity2). Multi-hop reasoning queries like "Who is the spouse of the president of France?" require traversing multiple edges.

**Rule-based reasoning** finds shortest paths or bounded-length paths in the KG. This is multi-hop BFS: find all paths from "president of France" of length $\leq k$ that end at a person via a "spouse" edge.

**Embedding-based reasoning** (TransE, RotatE, ComplEx) learns entity embeddings $\mathbf{e}_i$ and relation embeddings $\mathbf{r}$ such that $\mathbf{e}_i + \mathbf{r} \approx \mathbf{e}_j$ for true triples $(i,r,j)$. Query answering becomes nearest-neighbour search in embedding space — the graph structure is compiled into the embeddings.

**Neural graph reasoning** (NeuralLP, DRUM, NBFNet) explicitly runs differentiable versions of Bellman-Ford or BFS on the KG, learning edge weights that capture relation semantics. NBFNet (Neural Bellman-Ford Networks, Zhu et al. 2021) initialises boundary conditions at the query entity and propagates generalised Bellman-Ford messages, achieving state-of-the-art on KG link prediction.

### G.4 Causal Graphs and d-Separation

In a Bayesian network or structural causal model, the graph is a DAG where edges represent causal influences $X_i \to X_j$.

**d-separation** (Pearl 1988) is a graph-theoretic criterion for conditional independence: variables $X$ and $Y$ are conditionally independent given $Z$ iff every path between $X$ and $Y$ is "blocked" by $Z$ in the DAG.

A path is blocked if it contains:
- A chain $A \to B \to C$ with $B \in Z$ (conditioning on mediator blocks the path)
- A fork $A \leftarrow B \rightarrow C$ with $B \in Z$ (conditioning on common cause blocks)
- A collider $A \to B \leftarrow C$ with $B \notin Z$ and no descendant of $B$ in $Z$ (collider naturally blocks the path)

**Algorithmic test.** Given a DAG and sets $X$, $Y$, $Z$, determine d-separation by BFS/DFS on the **moral graph** (marry co-parents, drop edge directions) restricted by the Bayes Ball algorithm. This is a graph reachability computation — Dijkstra and BFS on a transformed graph.

**Interventional queries** (do-calculus) modify the DAG by removing incoming edges to the intervened variable and running independence tests on the modified graph. Every do-calculus rule reduces to graph reachability computations.

### G.5 Neural Architecture Search as Graph Problem

Neural Architecture Search (NAS) frames architecture design as optimisation over a **DAG of operations**. A neural architecture is a DAG where:
- Nodes are feature maps (tensors)
- Edges are operations (conv, relu, attention, etc.)
- Multiple paths from input to output = skip connections

**DARTS** (Differentiable Architecture Search, Liu et al. 2019) relaxes the discrete DAG into a continuous optimisation: each edge carries a mixture of all candidate operations, and the mixture weights are jointly optimised with network weights by gradient descent. The final architecture is recovered by picking the argmax operation per edge — a topological sort on the final discrete DAG.

**Cell-based NAS** restricts the search to small DAG templates (cells) that are stacked. The cell DAG is searched over, then replicated. This reduces the search space from all possible architectures to all possible DAGs with $k$ nodes and allowed operations.


---

## Appendix H: Problem Sets and Discussion Questions

### H.1 Conceptual Questions

**Q1.** BFS guarantees shortest hop-distances. Dijkstra guarantees shortest weighted distances. Under what exact condition do these two algorithms give the same answer?

*Answer:* When all edge weights are equal (say $w = 1$). In this case Dijkstra's priority queue always extracts vertices in BFS order, and the two algorithms are equivalent. BFS is preferred in this case since it avoids the $O(\log n)$ overhead of the priority queue.

**Q2.** Why can Bellman-Ford handle negative edge weights but Dijkstra cannot?

*Answer:* Dijkstra's correctness relies on the greedy invariant: once a vertex is extracted from the priority queue, its distance estimate is final and correct. This holds because non-negative edge weights guarantee that any later path through unvisited vertices cannot improve the distance. With negative edge weights, a vertex extracted "early" might later be improved via a path that includes a negative edge — breaking the invariant. Bellman-Ford avoids this by not committing to any distance estimate prematurely: it runs $n-1$ full rounds of relaxation, allowing distances to be improved in any order.

**Q3.** Give an example where DFS-based topological sort and Kahn's algorithm produce different orderings. Which is "correct"?

*Answer:* Any DAG with multiple valid topological orderings will do. For $A \to C$, $B \to C$ (two sources $A, B$): DFS might produce $[B, A, C]$ or $[A, B, C]$ depending on which source is processed first. Kahn's might produce $[A, B, C]$ or $[B, A, C]$ depending on queue order. Both are correct — any topological ordering is valid, and neither algorithm has priority over the other.

**Q4.** In what sense is A\* with $h \equiv 0$ equivalent to Dijkstra? With $h = $ true remaining distance?

*Answer:* With $h \equiv 0$: all vertices are sorted by $f(v) = d[v] + 0 = d[v]$, so A\* extracts vertices in exactly the same order as Dijkstra. They are identical. With $h = \delta(\cdot, t)$ (perfect heuristic): $f(v) = d[v] + \delta(v,t) = \delta(s,v) + \delta(v,t)$ — this equals $\delta(s,t)$ for all vertices on optimal paths. A\* with the perfect heuristic only expands vertices on optimal paths — it finds the answer by exploring the minimum possible number of nodes.

**Q5.** The Ford-Fulkerson algorithm may not terminate if capacities are irrational. Explain the pathological example.

*Answer:* Consider a graph where each augmenting path carries an irrational amount $1/\phi$ (where $\phi = (1+\sqrt{5})/2$ is the golden ratio), and the algorithm always picks a particular augmenting path. The sequence of flow increments converges to a sum that is not the maximum flow — the algorithm oscillates without terminating. Edmonds-Karp avoids this by always using BFS (shortest augmenting path), which guarantees polynomial termination regardless of capacities.

### H.2 Proof Exercises

**P1.** Prove that any graph with $n$ vertices and $m > \binom{n}{2} - n + 1 = \binom{n-1}{2}$ edges must be connected. *(Hint: show a disconnected graph can have at most $\binom{n-1}{2}$ edges by placing all edges in one component.)*

**P2.** Prove the **cycle property**: the unique maximum-weight edge in any cycle is not in any MST. *(Hint: suppose it is in some MST $T$; removing it splits $T$ into two components; the other cycle edges provide a cheaper reconnection.)*

**P3.** Prove that for any DFS tree, a vertex $u$ is an ancestor of $v$ iff $\text{disc}[u] < \text{disc}[v] < \text{fin}[v] < \text{fin}[u]$.

**P4.** Show that the number of SCCs in a directed graph equals the number of source nodes (in-degree 0) in the condensation DAG if and only if the condensation is a path (linear DAG). Give an example where this fails.

### H.3 Implementation Challenges

**I1.** Implement Dijkstra on a graph with $10^5$ nodes and $5 \times 10^5$ edges (random sparse). Compare running time with and without lazy deletion (keep stale entries in the heap and skip them when extracted).

**I2.** Implement Tarjan's SCC iteratively (without recursion, using an explicit stack). Verify against the recursive version on a random directed graph with $n = 10^4$.

**I3.** Implement bidirectional Dijkstra and compare its running time to standard Dijkstra on a $1000 \times 1000$ grid graph. Report the ratio of nodes explored.

**I4.** Implement Ford-Fulkerson with BFS (Edmonds-Karp). Test on a bipartite matching instance: $n$ workers, $n$ jobs, random bipartite graph. Compare max-flow solution to `scipy.optimize.linear_sum_assignment`.


---

## Appendix I: Connections to Other Chapters

### I.1 From Linear Algebra to Graph Algorithms

The adjacency matrix $A \in \{0,1\}^{n \times n}$ introduced in §02 appears in graph algorithms in several ways:

- **Walk counting:** $(A^k)[i][j]$ = number of walks of length $k$ from $i$ to $j$ (§C.3 above)
- **Reachability:** transitive closure $(I \vee A \vee A^2 \vee \cdots \vee A^{n-1})$ — Boolean APSP
- **Tropical matrix power:** shortest paths via $W^{\otimes n}$ in the min-plus semiring (§C.2)
- **Gaussian elimination on graphs:** LU decomposition of the Laplacian relates to Gaussian elimination on the graph (Cholesky = symbolic factorisation of the tree structure)

The **matrix-tree theorem** (Kirchhoff, 1847) connects spanning tree counting to linear algebra:
$$\text{(number of spanning trees of } G) = \text{any cofactor of the Laplacian } L$$
This is proved via the Cauchy-Binet formula and connects the combinatorial (MST, spanning trees) and algebraic (eigenvalues) views of graphs — a bridge to §04.

### I.2 From Probability to Graph Algorithms

Random walks on graphs — the Markov chain perspective — connect graph algorithms to probability theory (Chapter 6):

- **Stationary distribution** of the random walk on a $d$-regular graph = uniform over vertices
- **Mixing time** = how long until the walk is close to stationary ≈ $O(1/\lambda_2)$ where $\lambda_2$ is the Fiedler value (connection to §04)
- **Cover time** = expected steps for the walk to visit all vertices; for any connected graph, cover time $\leq O(n^3)$ (Matthews bound)
- **PageRank** = stationary distribution of a damped random walk: $\mathbf{r} = (1-d)\mathbf{1}/n + d \cdot A^\top D^{-1} \mathbf{r}$ — a fixed-point equation solved by power iteration (an iterative BFS-like process)

### I.3 From Optimisation to Graph Algorithms

Graph algorithms are the foundation of many combinatorial optimisation methods (Chapter 8):

- **Linear programming duality** ↔ max-flow min-cut duality: the LP relaxation of min-cut gives max-flow (strong duality)
- **Integer programming** on graphs: MST, matching, flow are all polynomial due to total unimodularity of their constraint matrices
- **Gradient descent** on the graph Laplacian: $\min_{\mathbf{x}} \mathbf{x}^\top L \mathbf{x}$ subject to constraints — eigenvalue problems (§04)
- **Submodular optimisation:** cut functions are submodular; greedy maximisation of submodular functions on graphs generalises MST

### I.4 Looking Ahead to Chapter 12 (Functional Analysis)

The graph Laplacian $L = D - A$ can be extended to continuous domains: replace the finite vertex set with a manifold, replace the adjacency matrix with a kernel function $k(x,y)$, and take the limit $n \to \infty$. The result is the **Laplace-Beltrami operator** on manifolds — the spectral graph theory of §04 generalises to functional analysis on infinite-dimensional spaces.

Spectral graph algorithms (§04) such as spectral clustering use eigenvectors of $L$ — these correspond to the eigenfunctions of the Laplace-Beltrami operator in the continuous limit. This connection motivates the study of Hilbert spaces and operators in Chapter 12.

---

## Quick Reference: Key Formulas

$$\text{BFS: } d[v] = d[u] + 1 \quad \text{(if } v \text{ discovered via } u\text{)}$$

$$\text{Dijkstra RELAX: } d[v] \leftarrow \min(d[v],\, d[u] + w(u,v))$$

$$\text{Bellman-Ford: } d_k[v] = \min_{(u,v)\in E}\!\left(d_{k-1}[u] + w(u,v)\right) \wedge d_{k-1}[v]$$

$$\text{Floyd-Warshall: } D^{(k)}[i][j] = \min\!\left(D^{(k-1)}[i][j],\; D^{(k-1)}[i][k] + D^{(k-1)}[k][j]\right)$$

$$\text{MST total weight: } W(T) = \sum_{e \in T} w(e) \leq W(T') \;\forall \text{ spanning trees } T'$$

$$\text{Max-flow: } |f^*| = \min_{(S,T) \text{ cut}} c(S,T)$$

$$\text{Kruskal correctness (cut property): } e \in \text{MST} \text{ if } e = \arg\min_{e' \text{ crosses } (S,V\setminus S)} w(e')$$

$$\text{Tarjan SCC root: } \text{low}[v] = \text{disc}[v]$$

$$\text{A}^* \text{ optimality: } h(v) \leq \delta(v,t)\;\forall v \text{ (admissibility)}$$

$$\text{GNN receptive field: } \Gamma^k(v) = \{u : \delta(v,u) \leq k\} = \text{BFS frontier after }k\text{ steps from }v$$

$$\text{Bellman equation (RL): } V^*(s) = \max_a \left[R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^*(s')\right]$$


---

## Appendix J: Graph Algorithms in Large Language Models

### J.1 Retrieval-Augmented Generation and Knowledge Graphs

Large Language Models (LLMs) augmented with external knowledge graphs use graph traversal at inference time:

1. **Entity recognition:** identify entities in the user query (NER)
2. **Graph lookup:** retrieve the entity's node in the KG
3. **Multi-hop traversal:** BFS/DFS to find related facts within $k$ hops
4. **Context injection:** retrieved subgraph facts are appended to the LLM prompt

The KG traversal step is BFS with budget constraints (at most $k$ hops, at most $B$ nodes retrieved). Systems like KAPING (Baek et al. 2023) and Think-on-Graph (Sun et al. 2023) implement this pattern, achieving better factual accuracy than pure parametric LLMs on multi-hop QA.

### J.2 Chain-of-Thought as Topological Sort

When an LLM generates a chain-of-thought (CoT) reasoning trace, it implicitly performs topological ordering over a DAG of reasoning steps:

- Each intermediate conclusion is a node
- Dependencies between conclusions are edges (conclusion A must precede conclusion B)
- The order of generation = a topological sort of the reasoning DAG

**Tree-of-Thought** (Yao et al. 2023) makes this explicit: a search tree of partial solutions is explored using BFS or DFS, with a value function scoring each node. This is A\* or BFS on the reasoning tree — graph algorithms for LLM reasoning.

**Scratchpad / Program-of-Thought** (Chen et al. 2022) generates executable programs (Python, JavaScript) that are then run. A Python program is itself a DAG — statements form a control flow graph. The LLM is generating a topological order of operations, and Python's interpreter executes them in that order.

### J.3 Attention Patterns as Adjacency Matrices

The attention weight matrix in a Transformer layer $\alpha \in \mathbb{R}^{n \times n}$ (for $n$ tokens) is a dense, weighted adjacency matrix of a complete directed graph. Each head computes a different "view" of this graph.

Mechanistic interpretability research (Elhage et al. 2021, Olsson et al. 2022) uses graph-theoretic analysis to understand information flow through attention:

- **Induction heads** implement a 2-hop path: attend to a previous occurrence of the current token, then "copy" the next token — a BFS-2 pattern
- **Circuit analysis** identifies subgraphs of attention heads (circuits) responsible for specific capabilities — analogous to finding subgraphs with specific connectivity properties
- **Attention sink phenomenon** (Xiao et al. 2023): some tokens (e.g., `[BOS]`) receive very high attention from all other tokens — analogous to high-degree "hub" nodes in a graph, or sink nodes that absorb most of the max-flow

### J.4 Structured State Space Models and Path Counting

Structured State Space Models (S4, Mamba) process sequences using linear recurrences:
$$h_t = A h_{t-1} + B x_t, \quad y_t = C h_t$$

The matrix $A$ defines a recurrence over a latent state — equivalent to a weighted adjacency matrix of a directed "state transition graph." The output $y_t$ depends on all past inputs via the series $C A^{t-1} B + C A^{t-2} B + \cdots + CB$ — a sum over all path lengths in the state graph.

This is the tropical algebra perspective again: SSMs compute convolutions in the linear semiring $(+, \times)$, while graph shortest paths use the tropical semiring $(\min, +)$. The mathematical structure is identical; only the semiring changes.

---

## Appendix K: Pseudocode Summary (All Algorithms)

```
══════════════════════════════════════════════════════════════════════
BFS(G, s)
  d[s]=0; d[v]=∞ ∀v≠s; Q=[s]
  while Q: u=dequeue(Q); for v∈N(u): if d[v]=∞: d[v]=d[u]+1; enqueue(Q,v)
──────────────────────────────────────────────────────────────────────
DFS(G)
  for v: DFS-VISIT(v) if v unvisited
  DFS-VISIT(u): disc[u]=time++; GREY; for v∈N(u): if v WHITE: DFS-VISIT(v)
               fin[u]=time++; BLACK
──────────────────────────────────────────────────────────────────────
DIJKSTRA(G, s)
  d[s]=0; d[v]=∞; Q=min-heap(V keyed by d)
  while Q: u=EXTRACT-MIN(Q); for v∈N(u): if d[u]+w<d[v]: DECREASE-KEY
──────────────────────────────────────────────────────────────────────
BELLMAN-FORD(G, s)
  d[s]=0; d[v]=∞
  for i=1..n-1: for (u,v)∈E: RELAX(u,v)
  for (u,v)∈E: if d[u]+w<d[v]: return "neg cycle"
──────────────────────────────────────────────────────────────────────
FLOYD-WARSHALL(G)
  D=weight matrix; D[i][i]=0
  for k=1..n: for i: for j: D[i][j]=min(D[i][j], D[i][k]+D[k][j])
──────────────────────────────────────────────────────────────────────
KRUSKAL(G)
  Sort E by weight; MAKE-SET(v) ∀v
  for (u,v) in sorted E: if FIND(u)≠FIND(v): add edge; UNION(u,v)
──────────────────────────────────────────────────────────────────────
PRIM(G, r)
  key[r]=0; key[v]=∞; Q=min-heap(V keyed by key)
  while Q: u=EXTRACT-MIN; for v∈N(u): if v∈Q and w<key[v]: DECREASE-KEY
──────────────────────────────────────────────────────────────────────
KAHN(G)
  indeg[v]=in-degree ∀v; Q=[v: indeg[v]=0]
  while Q: u=dequeue; order.append(u); for (u,v): indeg[v]--; if 0: enqueue
──────────────────────────────────────────────────────────────────────
TARJAN-SCC(G)
  DFS-VISIT(v): disc[v]=low[v]=time++; stack.push(v); onStack[v]=T
    for (v,w): if disc[w]=-1: visit(w); low[v]=min(low[v],low[w])
               elif onStack[w]: low[v]=min(low[v],disc[w])
    if low[v]=disc[v]: pop SCC from stack until v
──────────────────────────────────────────────────────────────────────
FORD-FULKERSON(G, s, t)
  f=0; while augmenting path P in G_f: δ=bottleneck(P); augment by δ
──────────────────────────────────────────────────────════════════════
```


---

## Appendix L: Worked Example — Full Graph Problem

### L.1 End-to-End: Building and Querying a Road Network

**Problem.** You are given a city road network with 7 intersections and 10 roads (edges). You need to:
(a) Find shortest driving routes (weighted by travel time)
(b) Find which intersections can reach city hall (reachability)
(c) Plan a minimum-cost infrastructure to connect all intersections (MST)
(d) Find the maximum flow of buses from depot to stadium

**Graph.** Intersections $\{A,B,C,D,E,F,G\}$. Roads (undirected, travel time in minutes):

$(A,B,5), (A,C,8), (B,C,3), (B,D,10), (C,D,2), (C,E,7), (D,F,4), (D,G,6), (E,F,9), (F,G,1)$

**Part (a): Dijkstra from $A$**

Sort by extraction order:
- Extract $A(0)$: relax $A\to B(5)$, $A\to C(8)$
- Extract $B(5)$: relax $B\to C: 5+3=8$ (tie), $B\to D: 5+10=15$
- Extract $C(8)$: relax $C\to D: 8+2=10 < 15$ ✓, $C\to E: 8+7=15$
- Extract $D(10)$: relax $D\to F: 10+4=14$, $D\to G: 10+6=16$
- Extract $F(14)$: relax $F\to G: 14+1=15 < 16$ ✓, $F\to E: 14+9=23 > 15$
- Extract $E(15)$: no improvement
- Extract $G(15)$: done

**Result:** $d = \{A:0, B:5, C:8, D:10, E:15, F:14, G:15\}$

**Part (b): BFS reachability (unweighted)**

From $A$: Layer 0: $\{A\}$, Layer 1: $\{B,C\}$, Layer 2: $\{D,E\}$, Layer 3: $\{F,G\}$. All 7 intersections reachable.

**Part (c): Kruskal MST**

Sorted edges: $(F,G,1), (B,C,3), (C,D,2), (A,B,5), (D,F,4), (C,E,7), (A,C,8), (D,G,6), (E,F,9), (B,D,10)$

Re-sorted: $(F,G,1), (C,D,2), (B,C,3), (D,F,4), (A,B,5), (D,G,6), (C,E,7), ...$

- Add $(F,G,1)$: $\{F,G\}$, others singletons. $T = \{FG\}$
- Add $(C,D,2)$: $\{C,D\}$. $T = \{FG, CD\}$
- Add $(B,C,3)$: $\{B,C,D\}$. $T = \{FG, CD, BC\}$
- Add $(D,F,4)$: $\{B,C,D,F,G\}$ (merges). $T = \{FG, CD, BC, DF\}$
- Add $(A,B,5)$: $\{A,B,C,D,F,G\}$ (merges). $T = \{FG, CD, BC, DF, AB\}$
- Add $(D,G,6)$: FIND($D$) = FIND($G$) — same component! Skip.
- Add $(C,E,7)$: $\{A,B,C,D,E,F,G\}$ — all connected! $T = \{FG, CD, BC, DF, AB, CE\}$

**MST edges:** $FG, CD, BC, DF, AB, CE$. **Total weight:** $1+2+3+4+5+7 = 22$ minutes.

**Part (d): Max flow (bus routes with capacities)**

Convert to directed flow network: depot = $A$, stadium = $G$, capacities = edge weights.
Augmenting paths found by BFS:
- Path $A \to B \to C \to D \to F \to G$: bottleneck = min(5,3,2,4,1) = 1
- After augmenting: $d[FG]$ saturated. Next path must avoid $FG$ edge at capacity.
- Path $A \to B \to C \to D \to G$: bottleneck = min(5,3,2,6) = 2; update residual
- Continue until no augmenting path...

**Max flow = 5** (limited by the cut $\{D,F,G\} | \{A,B,C,E\}$ with capacity $c(CD) + c(DG) = 2 + 6 = 8$... 

*[Full trace left as Exercise 8]*

This single graph problem illustrates how all four algorithm families (BFS, SSSP, MST, max-flow) answer different questions about the same underlying graph — each appropriate for a different operational need.


---

## Appendix M: Implementation Notes for ML Engineers

### M.1 Graph Algorithms in PyTorch Geometric

PyTorch Geometric (PyG) is the standard library for graph deep learning, but it also exposes classical graph operations. Understanding how classical algorithms map to PyG's data model is essential for efficient GNN implementation.

**Graph data model:**
```python
from torch_geometric.data import Data
import torch

# A graph with 4 nodes, 4 edges (2 undirected = 4 directed)
edge_index = torch.tensor([[0,1,1,2,2,3,0,3],   # source nodes
                            [1,0,2,1,3,2,3,0]], dtype=torch.long)  # target nodes
x = torch.randn(4, 8)   # 4 nodes, 8 features each
data = Data(x=x, edge_index=edge_index)
```

**BFS in PyG:** not directly built-in, but `torch_geometric.utils.k_hop_subgraph` implements BFS-$k$ neighbourhood extraction:
```python
from torch_geometric.utils import k_hop_subgraph
# Get 2-hop neighbourhood of node 0
subset, edge_index_sub, mapping, mask = k_hop_subgraph(0, num_hops=2, edge_index=edge_index)
```

**Shortest paths via scipy:**
```python
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, connected_components
# Convert PyG edge_index to scipy sparse matrix
A = csr_matrix((torch.ones(edge_index.size(1)), edge_index.numpy()), shape=(n,n))
dist = shortest_path(A, method='D')  # Dijkstra (non-neg weights)
n_comp, labels = connected_components(A)
```

### M.2 Efficient Implementations: Common Patterns

**Pattern 1: Avoid Python loops — use vectorised operations**
```python
# BAD: Python loop BFS (O(n*m) effectively due to Python overhead)
for u in queue:
    for v in adj[u]:
        if d[v] == inf: ...

# GOOD: scipy sparse BFS (C-backed)
from scipy.sparse.csgraph import breadth_first_search
dist = breadth_first_search(A, i_start=0, return_predecessors=False)
```

**Pattern 2: Use indexed edge lists for Bellman-Ford**
```python
# Represent edges as three arrays for vectorised relaxation
src = np.array([0, 0, 1, 2])
dst = np.array([1, 2, 3, 3])
w   = np.array([4., 2., 1., 3.])

d = np.full(n, np.inf); d[s] = 0.
for _ in range(n-1):
    improved = d[src] + w < d[dst]
    d[dst[improved]] = d[src[improved]] + w[improved]
```

**Pattern 3: Union-Find with path compression (NumPy)**
```python
parent = np.arange(n)

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]  # path compression (halving)
        x = parent[x]
    return x

def union(x, y):
    px, py = find(x), find(y)
    if px != py: parent[px] = py
```

### M.3 GPU Graph Algorithms with cuGraph

NVIDIA cuGraph provides GPU-accelerated graph algorithms with a NetworkX-compatible API:

```python
import cudf, cugraph

# Load edge list
edges = cudf.DataFrame({'src': [0,0,1,2], 'dst': [1,2,3,3], 'wt': [4.,2.,1.,3.]})
G = cugraph.Graph()
G.from_cudf_edgelist(edges, source='src', destination='dst', edge_attr='wt')

# GPU BFS
bfs_result = cugraph.bfs(G, start_vertex=0)

# GPU Dijkstra (SSSP)
sssp_result = cugraph.sssp(G, source=0)
```

For billion-node graphs, cuGraph achieves throughputs of $10^9$ edges/second on modern GPUs — three orders of magnitude faster than serial Python implementations.

### M.4 Memory-Efficient Graph Storage

For large graphs that don't fit in GPU memory, consider:

1. **Graph partitioning:** split the graph into $k$ parts using METIS or spectral methods, process each partition on a separate GPU, and exchange boundary information
2. **Graph sampling:** for GNNs, sample subgraphs (GraphSAGE-style) rather than loading the full adjacency
3. **Streaming algorithms:** process edges in order without materialising the full graph; union-find for online connectivity, sliding-window BFS for temporal graphs
4. **External memory algorithms:** if the graph doesn't fit in RAM, use disk-based BFS (level-by-level, one level per disk pass) — used for web-crawl graphs in the early 2000s

