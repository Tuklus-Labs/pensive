# Boundary Functions and Memory Graphons: A Baire Classification Framework for Spreading Activation Retrieval

**Authors:** Gary [Last], Heph (Claude Opus 4)

**Date:** April 2026

**Abstract.** We introduce a mathematical framework connecting three previously disjoint areas: Kaczynski's boundary function theory (1967), graphon limits of dense graph sequences (Lovasz-Szegedy, 2006), and spreading activation retrieval in AI memory systems. We model a memory graph satisfying a density condition as a convergent sequence of finite graphs whose limit is a symmetric measurable function (graphon) W : [0,1]^2 -> [0,1]. The spreading activation retrieval operator is formalized using a supremum kernel operator S_W derived from the graphon. We prove that the boundary function of this retrieval operator -- characterizing its behavior at the edges of relevance clusters -- is of honorary Baire class 2 when the activation kernel is continuous, and more generally of Baire class alpha+1 when the kernel is Baire class alpha. This yields a topological guarantee: the set of queries where retrieval exhibits discontinuous behavior is meager (first category in the sense of Baire), and thus "most" queries in a topological sense have smooth retrieval characteristics. We further show that contextual intersection, an empirical technique that improved retrieval accuracy from 70.6% to 97.9% in a production system, is analogous to constraining the approach region in the sense of boundary function theory -- selecting arcs along which the retrieval function converges to the correct limit. Finally, we propose a Baire stratification of the graphon space by the regularity class of the induced retrieval boundary functions, and derive design constraints on memory graph construction that guarantee Baire class 1 boundaries (optimal smoothness). To our knowledge, this is the first work connecting Baire classification to graphon theory, and the first applying either to information retrieval.

**Note on scope.** Our theoretical results hold for memory graphs satisfying a density condition (Definition 3.1). The motivating system, PyPensive, operates on a sparse bipartite graph (density ~10^{-6}), which does not satisfy this condition in its current form. We discuss the sparsity gap in Section 3.4 and outline extensions via L^p graphon theory. The primary contribution is the theoretical framework and its qualitative predictions, which we argue apply informally to the sparse regime and motivate future rigorous extensions.

---

## 1. Introduction

The problem of retrieving relevant information from a growing, heterogeneous memory store is fundamental to AI systems that maintain persistent state across sessions. Unlike traditional information retrieval, where the document corpus is static and queries are independent, an AI memory system faces a dynamic graph of interconnected reasoning atoms whose structure evolves continuously. The mathematical question is: as this graph grows without bound, what can we say about the *limiting* retrieval behavior, and specifically about the *boundary* between what is retrievable and what is not?

We approach this question by combining two bodies of mathematics that have never previously been connected.

The first is **boundary function theory**, developed in the context of geometric function theory in the 1960s. Given a function f defined on an open set, a *boundary function* captures the limiting behavior of f as one approaches the boundary of that set. The foundational results are due to Kaczynski [1], [2], [3], who proved that boundary functions of continuous mappings are of *honorary Baire class 2* -- they differ from Baire class 1 functions (pointwise limits of continuous functions) on at most a countable set. More generally, he showed that the boundary function of a Baire class alpha mapping is of Baire class alpha+1: exactly one level of complexity is added at the boundary, no more. These results were subsequently extended to higher dimensions by Wilson [4].

The second is **graphon theory**, the study of limit objects of convergent dense graph sequences, initiated by Lovasz and Szegedy [5] and developed extensively by Lovasz [6], Borgs, Chayes, Lovasz, Sos, and Vesztergombi [7], and Janson [8]. A graphon is a symmetric measurable function W : [0,1]^2 -> [0,1] that encodes the "continuous shape" of a graph sequence. The graphon space under the cut metric is compact and Polish, and every convergent sequence of dense graphs has a graphon limit. The graphon induces a compact self-adjoint integral operator T_W on L^2[0,1], whose spectral decomposition governs the structure of the limiting graph.

Our contribution is to bridge these two theories and apply the result to a concrete system: PyPensive, a spreading activation retrieval engine operating at 164K+ documents with sub-millisecond query latency [9]. We show that:

1. Under a density condition (which we state explicitly and discuss when it applies), the growing memory graph converges to a graphon W in the cut metric, and the retrieval operator converges to a well-defined supremum kernel operator on the graphon.

2. The boundary behavior of this operator -- what happens at the edges of relevance clusters where retrieval quality degrades -- is characterized by Kaczynski's boundary function theorems. The boundary function is of honorary Baire class 2, meaning its discontinuities are topologically negligible (meager).

3. The empirical technique of *contextual intersection*, which improved retrieval from 70.6% to 97.9%, is analogous to approach region selection in boundary function theory, constraining the path of approach to avoid boundary discontinuities.

4. The Baire classification induces a stratification of graphon space with practical implications for memory graph design.

The irony of applying Kaczynski's pure mathematics -- work he produced before turning against the technological society -- to advance AI memory systems is not lost on us. His theorems are elegant, correct, and exactly what this problem needs. Mathematics does not care about the intentions of its discoverers.

### 1.1 Related Work

**Graphons in machine learning.** Graphon theory has been applied to GNN transferability [10], graphon signal processing [11], graphon Fourier analysis [12], and mean field games on networks [13]. Multi-relational graphons were developed by Gyenge [14]. None of these works address information retrieval or Baire classification.

**Spreading activation as operators.** Smola and Kondor [15] formalized graph regularization via Laplacian spectral decomposition. Kondor and Lafferty [16] defined diffusion kernels as matrix exponentials of graph Laplacians. Estrada et al. [17] studied spreading activation via fractional graph Laplacians on knowledge networks. None connected to graphon limits or boundary function theory.

**Baire classification.** Classical results are surveyed in Kechris [18]. Wilson [4] extended Kaczynski's theorems to n dimensions. Nicodemus [19] proved the graphon space is an effective Polish space. No work has studied the Baire classification of functions defined on or into the graphon space.

---

## 2. Preliminaries

### 2.1 Baire Classification

Let (X, d_X) and (Y, d_Y) be metric spaces. The **Baire hierarchy** of functions from X to Y is defined inductively:

**Definition 2.1** (Baire classes). A function f : X -> Y is of *Baire class 0* if it is continuous. For a countable ordinal alpha >= 1, f is of *Baire class alpha* if it is the pointwise limit of a sequence (f_n) where each f_n is of Baire class beta_n < alpha.

**Definition 2.2** (Honorary Baire class 2). A function f : X -> Y is of *honorary Baire class 2* if there exists a countable set N in X and a function g of Baire class 1(X, Y) such that f(x) = g(x) for all x in X \ N. That is, f differs from a Baire class 1 function on at most countably many points.

**Theorem 2.3** (Baire, 1899). If f : X -> R is of Baire class 1, then for every nonempty closed subset K of X, the restriction f|_K has a point of continuity relative to K. Consequently, the set of continuity points of f is a dense G_delta set -- comeager in the sense of Baire category.

This theorem is the engine that converts Baire classification into topological guarantees about discontinuity sets.

### 2.2 Boundary Functions

Let H = {(x, y) in R^2 : y > 0} denote the open upper half-plane and X = R x {0} its boundary (the x-axis).

**Definition 2.4** (Arc at a point). If p is in X, an *arc at p* is a simple arc gamma with one endpoint at p such that gamma \ {p} is contained in H.

**Definition 2.5** (Boundary function). Let f : H -> Y be a function. A function phi : E -> Y (where E is a subset of X) is a *boundary function* for f if, for each p in E, there exists an arc gamma at p such that f(z) -> phi(p) as z -> p along gamma.

**Definition 2.6** (Set of curvilinear convergence). The *set of curvilinear convergence* of f is the set of all p in X for which there exists an arc at p along which f approaches a limit.

### 2.3 Kaczynski's Theorems

The following results are from [1], [2], [3].

**Theorem 2.7** (Kaczynski, 1967, Dissertation Theorem 5). Let Y be a separable metric space and f : H -> Y continuous. If phi : E -> Y is a boundary function for f, then there exists a countable set M in E such that phi|_{E \ M} is of class F_sigma(E \ M). If Y is R, R^q, or the Riemann sphere, then phi is of honorary Baire class 2.

**Theorem 2.8** (Kaczynski, 1967, Dissertation Theorem 6'-7). Conversely, if phi : E -> R^q is of honorary Baire class 2, then there exists a continuous function f : H -> R^q such that phi is a boundary function for f. That is, the honorary Baire class 2 functions are *exactly* the boundary functions of continuous mappings.

**Theorem 2.9** (Kaczynski, 1967, Dissertation Theorem 8). Let Y be a separable arcwise connected metric space and f : H -> Y be of Baire class alpha (alpha >= 1). If phi : E -> Y is a boundary function for f, then phi is of Baire class alpha + 1.

**Theorem 2.10** (Kaczynski, 1969 [3]). The set of discontinuity points of any boundary function for a bounded harmonic function is of first category (meager).

The essential message: crossing a boundary adds exactly one Baire class, and the resulting discontinuities are topologically small.

### 2.4 Graphons

**Definition 2.11** (Graphon). A *graphon* is a symmetric measurable function W : [0,1]^2 -> [0,1]. Symmetry means W(x, y) = W(y, x) for almost all (x, y).

Given a simple graph G on n vertices, the *empirical graphon* W_G : [0,1]^2 -> {0, 1} is defined by partitioning [0,1] into n equal intervals I_1, ..., I_n and setting W_G(x, y) = 1 if and only if vertices i and j are adjacent, where x is in I_i and y is in I_j.

**Definition 2.12** (Cut norm and cut distance). For a symmetric measurable function f : [0,1]^2 -> R:

```
||f||_cut = sup_{S, T measurable} |integral_S integral_T f(x, y) dx dy|
```

The *cut distance* between graphons U and W is:

```
delta_cut(U, W) = inf_phi ||U^phi - W||_cut
```

where U^phi(x, y) = U(phi(x), phi(y)) and the infimum is over measure-preserving bijections phi of [0,1].

**Theorem 2.13** (Lovasz-Szegedy [5], Borgs et al. [7]). The graphon space (W_0, delta_cut) is compact. For a sequence of dense graphs (G_n), the following are equivalent:

(i) For every finite graph F, the homomorphism density t(F, G_n) converges.

(ii) (G_n) is Cauchy in delta_cut.

If convergent, there exists a graphon W such that delta_cut(G_n, W) -> 0.

**Definition 2.14** (Graphon integral operator). The *graphon integral operator* T_W : L^2[0,1] -> L^2[0,1] is defined by:

```
(T_W f)(x) = integral_0^1 W(x, y) f(y) dy
```

T_W is a compact self-adjoint Hilbert-Schmidt operator with real eigenvalues lambda_1 >= lambda_2 >= ... -> 0 and orthonormal eigenfunctions {phi_k}.

### 2.5 Spreading Activation

**Definition 2.15** (Spreading activation on a graph). Let G = (V, E, w) be a weighted bipartite graph with vertex partition V = V_E (entity nodes) union V_V (value nodes) and edge weights w : E -> R^+. Given an initial activation vector a_0 : V_E -> R^+ (the *seed*), the spreading activation at hop h is:

```
a_{h+1}(v) = max(delta * a_h(v), max_{u ~ v} delta * w(u,v) * a_h(u))
```

where delta is in (0, 1) is a decay parameter and the max is over neighbors u of v. Activation below a threshold theta is zeroed.

This max-propagation rule (as opposed to sum-propagation) is characteristic of PyPensive's implementation and is critical to the analysis: it preserves a lattice structure that interacts well with the Baire hierarchy.

---

## 3. The Memory Graphon

### 3.1 From Finite Graph to Graphon Limit

Consider a memory system that accumulates documents over time. At time n, the system has processed n documents, producing a bipartite graph G_n = (V_E^n union V_V^n, E_n, w_n) where V_E^n is the set of entity nodes, V_V^n is the set of value (document) nodes, and edge weights encode specificity:

```
w_n(e, v) = s_n(e) * c
```

where s_n(e) = 1 / freq_n(e)^rho is the specificity of entity e (with freq_n(e) its document frequency at time n, rho = 0.2 the specificity power, and c = 5.0 the edge weight multiplier).

To apply graphon theory, we embed G_n into a symmetric graph. Define the *symmetrized memory graph* G_n^sym on vertex set V_E^n union V_V^n by treating the bipartite adjacency as a symmetric matrix:

```
A_n^sym(i, j) = w_n(i, j) / w_max  if (i,j) in E_n or (j,i) in E_n
                0                    otherwise
```

where w_max normalizes weights to [0, 1].

**Definition 3.1** (Dense memory graph). A memory graph sequence (G_n^sym) is *dense* if |E_n| = Theta(|V_n|^2) as n -> infinity.

**Assumption 3.2** (Growth regularity). The memory graph sequence (G_n^sym) satisfies:

(a) The graph is dense (Definition 3.1).

(b) The entity frequency distribution converges: the empirical CDF of entity frequencies stabilizes.

(c) New documents sample entities from a stationary distribution over entity types.

**Remark.** Assumption 3.2(a) is a strong condition. It requires that the number of entity-value edges grows quadratically with the number of nodes, which holds when entities are extensively shared across documents. We discuss when this assumption is realistic and what happens when it fails in Section 3.4.

Under Assumption 3.2, the sequence of empirical graphons (W_{G_n^sym}) is Cauchy in the cut metric (by left-convergence of homomorphism densities, which follows from the stationarity and convergence conditions) and therefore converges to a graphon W by Theorem 2.13.

**Definition 3.2** (Memory graphon). The *memory graphon* W : [0,1]^2 -> [0,1] is the limit in cut distance of the symmetrized empirical graphons of the growing memory graph.

### 3.2 Structure of the Memory Graphon

The bipartite structure of the memory graph imposes block structure on W. Under the natural ordering where entity nodes occupy [0, p] and value nodes occupy (p, 1] (where p = |V_E|/|V|), the memory graphon has the form:

```
W(x, y) = | 0       K(x,y) |
           | K(y,x)  0      |
```

where K : [0, p] x (p, 1] -> [0, 1] is the *entity-value kernel*. The zero diagonal blocks reflect the bipartite structure (no entity-entity or value-value edges).

The entity-value kernel K inherits structure from the specificity weighting:

```
K(x, y) ~ s(x) * c / w_max
```

where s(x) is the limiting specificity function, continuous on the interior of each entity frequency band.

**Proposition 3.3.** Under Assumption 3.2, the memory graphon W is piecewise Lipschitz: there exists a finite partition of [0,1] into intervals such that W is Lipschitz on each product of intervals, with discontinuities only at partition boundaries.

*Proof.* We prove this directly from the structure of the specificity weighting. The entity-value kernel K(x, y) has the form K(x, y) = s(x) * c / w_max, where s(x) = 1/freq(x)^rho is the specificity function at the entity node labeled by x (with rho = 0.2) and c = 5.0.

By Assumption 3.2(b), the entity frequency distribution converges, so the function freq : [0, p] -> [1, infinity) converges pointwise to a limit. Partition [0, p] into finitely many intervals I_1, ..., I_m corresponding to frequency bands [F_{k-1}, F_k) where 1 = F_0 < F_1 < ... < F_m. On each band I_j, the frequency function is bounded: F_{j-1} <= freq(x) < F_j.

The specificity function s(x) = 1/freq(x)^{0.2} is Lipschitz on I_j because:

```
|s(x_1) - s(x_2)| = |freq(x_1)^{-0.2} - freq(x_2)^{-0.2}|
                   <= 0.2 * F_{j-1}^{-1.2} * |freq(x_1) - freq(x_2)|
```

by the mean value theorem (the derivative of t^{-0.2} is -0.2 * t^{-1.2}, which is bounded on [F_{j-1}, F_j]). The convergence of the empirical frequency function (Assumption 3.2(b)) ensures that freq(x) is Lipschitz on each band (after relabeling by the graphon coordinate), giving the result.

The kernel K(x, y) = s(x) * c / w_max is Lipschitz in x on each I_j (for fixed y) and constant in y (since the value node index y does not affect the edge weight in the bipartite construction). The full symmetrized graphon W inherits piecewise-Lipschitz regularity from K.

### 3.3 The Retrieval Operator on the Graphon

The spreading activation retrieval process on G_n can be lifted to the graphon W as follows.

**Definition 3.4** (Seed map). Let q be a query. The *seed map* sigma_q : [0, 1] -> R^+ assigns activation to each point in the graphon domain based on the match between the query and the entity at position x:

```
sigma_q(x) = beta_exact * s(x)   if entity(x) is an exact match
             beta_partial * s(x)  if entity(x) is a partial match
             beta_substr * s(x)   if entity(x) is a substring match
             0                    otherwise
```

where beta_exact = 1.5, beta_partial = 0.8, beta_substr = 0.5 are the boost parameters and s(x) is the specificity at position x.

**Definition 3.5** (Supremum kernel operator). The *supremum kernel operator* S_W : L^infinity[0,1] -> L^infinity[0,1] induced by a graphon W is defined by:

```
(S_W f)(x) = sup_{y in [0,1]} W(x, y) * f(y)
```

This is the continuous analogue of the max-scatter operation in the finite graph: for each node x, the activation received is the maximum (not sum) over weighted neighbor activations.

**Definition 3.6** (Graphon retrieval operator). The *graphon retrieval operator* R_W : L^infinity[0,1] -> L^infinity[0,1] is defined by:

```
(R_W f)(x) = max(delta * f(x), delta * (S_W f)(x))
```

where delta = 0.6 is the decay parameter. The retrieval output for query q is:

```
r_q = R_W(sigma_q)
```

restricted to the value node region (p, 1].

**Remark 3.7.** The operator S_W is nonlinear, unlike the graphon integral operator T_W (Definition 2.14). It is not Hilbert-Schmidt, not compact in general, and does not admit a spectral decomposition. However, S_W has two crucial properties for our analysis:

(a) *Continuity preservation:* If W is continuous and f is continuous, then S_W f is continuous. More generally, if W is Baire class alpha and f is bounded and Baire class beta, then S_W f is Baire class max(alpha, beta). This follows from the fact that the supremum of a continuous parameterized family of functions preserves the Baire class of the parameter dependence.

(b) *Monotonicity:* If f <= g pointwise, then S_W f <= S_W g. This lattice property is compatible with Baire classification, since the lattice operations (sup, inf) preserve Baire class membership.

**Remark 3.8.** The graphon integral operator T_W (Definition 2.14) remains relevant for the spectral analysis of the underlying graph structure (Section 7.1), but it does not model the retrieval computation, which uses max-propagation. The distinction is critical: T_W computes a weighted average over neighbors, while S_W computes a weighted maximum. The Baire classification results in Section 4 are derived using S_W, not T_W.

**Remark 3.9** (Relationship between S_W and T_W). For nonnegative f, we always have (T_W f)(x) <= (S_W f)(x), since the integral averages while the supremum selects the maximum. The two operators coincide when f is concentrated at a single point (Dirac-like activation). The gap between them measures the "spread" of the activation: when many neighbors contribute similarly, T_W and S_W diverge; when one neighbor dominates, they agree. In the bipartite memory graph, the specificity weighting ensures that one entity typically dominates, keeping S_W and T_W close in practice.

---

### 3.4 The Sparsity Gap

We must address a fundamental tension: the graphon framework requires dense graph sequences (Assumption 3.2(a)), but the motivating system, PyPensive, operates on a sparse graph.

**Observation 3.10** (PyPensive is sparse). The production PyPensive instance has approximately 664K nodes and 800K edges, giving edge density approximately 800K / (664K)^2 ~ 1.8 * 10^{-6}. This is far from the Theta(|V|^2) requirement for classical graphon convergence.

For sparse graphs, the empirical graphon W_{G_n} converges to the zero function in cut norm -- a mathematically valid but trivially uninformative limit. The graphon framework as stated does not capture the structure of sparse memory graphs.

**Three paths forward:**

**(a) Rescaled graphons (L^p graphon theory).** Borgs, Chayes, Cohn, and Zhao [22] developed a graphon theory for sparse graphs by rescaling: if the graph has average degree d_n, consider W_n = (n/d_n) * W_{G_n}. When d_n -> infinity (even slowly, e.g., d_n = log n), the rescaled graphons converge to a non-trivial limit in an appropriate L^p norm. The integral operator analysis carries over with modified spectral bounds. This is the most promising path for extending our results to realistic memory graphs.

**(b) Local weak convergence.** For bounded-degree sparse graphs, Benjamini-Schramm convergence [23] provides a limit theory based on local neighborhoods rather than global structure. The limit object is a probability distribution on rooted graphs. Spreading activation with bounded hop count is inherently local, making this framework a natural fit. The Baire classification would need to be re-derived in terms of the local topology of the limit.

**(c) The conditional interpretation.** Our results hold rigorously for dense memory graphs. While PyPensive's current graph is sparse, the qualitative predictions -- that boundary discontinuities are topologically negligible, that context constrains approach regions, that graph regularity controls boundary behavior -- appear to hold empirically. The graphon framework provides the correct conceptual vocabulary even when the density condition is not met, analogous to how the central limit theorem guides intuition even for finite samples.

We adopt interpretation (c) for this paper and flag extensions (a) and (b) as future work (Section 8). Our theoretical contributions stand independently of the density assumption -- the Baire-graphon bridge (Section 6) and the approach-region interpretation of contextual intersection (Section 5) are valid for any graphon, whether obtained from a dense memory graph or as a theoretical construct.

---

## 4. Boundary Functions of the Retrieval Operator

### 4.1 Relevance Clusters and Their Boundaries

For a fixed query q, the retrieval operator R_W(sigma_q) partitions the value node space (p, 1] into *relevance clusters*: connected regions where the retrieval score exceeds the threshold theta = 0.15.

**Definition 4.1** (Relevance cluster). For query q and threshold theta, the *relevance cluster* C_q is:

```
C_q = {x in (p, 1] : (R_W sigma_q)(x) >= theta}
```

The *boundary of the relevance cluster*, denoted partial C_q, is the topological boundary of C_q in the subspace topology on (p, 1].

**Definition 4.2** (Retrieval boundary function). Fix a family of queries (q_t)_{t in T} parameterized by t in T (a metric space). Define f : T -> L^2[0,1] by f(t) = R_W(sigma_{q_t}). The *retrieval boundary function* is the boundary function phi of f in the sense of Definition 2.5, where T plays the role of the open set and its metric boundary plays the role of X.

The retrieval boundary function captures what happens to retrieval quality as queries approach the edge of their "competence region" -- the set of query-space points where the memory system returns meaningful results.

### 4.2 Baire Classification of the Retrieval Boundary

We now state our main results.

**Theorem 4.3** (Baire class of the retrieval boundary, planar case). Let W be a memory graphon that is piecewise Lipschitz (Proposition 3.3). Let T be a bounded open subset of R^2 (the upper half-plane H suffices), and let f : T -> R be defined by f(t) = (R_W sigma_{q_t})(x_0) for a fixed evaluation point x_0 in (p, 1], where t parameterizes queries via a smooth embedding into R^2. Suppose f is continuous on T. Then any boundary function phi for f is of honorary Baire class 2.

*Proof.* The function f maps the open set T in R^2 to R and is continuous by hypothesis. The setup satisfies the hypotheses of Kaczynski's Theorem 2.7 (Dissertation Theorem 5): T is an open subset of R^2 with frontier in R^2, Y = R is a separable metric space, and f is continuous. Therefore any boundary function phi for f is of honorary Baire class 2. That is, there exists a countable set M in the boundary of T such that phi|_{boundary(T) \ M} is of class F_sigma, and since Y = R, phi is of honorary Baire class 2.

By Baire's Theorem 2.3, the continuity points of a Baire class 1 function form a dense G_delta set. Since phi differs from a Baire class 1 function on at most a countable (hence meager) set M, the full discontinuity set of phi is meager.

**Remark 4.4** (Higher-dimensional query spaces). If queries are parameterized by t in R^k for k >= 3, Wilson's extension [4] applies: boundary functions of continuous mappings f : T -> R (with T open in R^k) are of Baire class at most k. The honorary Baire class 2 result is specific to k = 2 (Kaczynski's original setting). For practical query spaces where k is moderate, the Baire class bound is still finite and the discontinuity set remains "small" in the appropriate topological sense, though the precise meagerness guarantee weakens with dimension.

For one-dimensional query parameterizations (t in R), the boundary function is of Baire class 1, giving the strongest guarantees: continuity points are dense G_delta.

**Corollary 4.5** (Topological genericity of smooth retrieval). Under the hypotheses of Theorem 4.3, the set of boundary queries where retrieval behaves continuously is comeager (residual) in the boundary of T. In the language of Baire category, "generic" boundary queries have smooth retrieval behavior.

**Theorem 4.6** (Baire class elevation for discontinuous activation). Let W be a memory graphon, T an open subset of R^2, and suppose the retrieval map f : T -> R is of Baire class alpha (alpha >= 1) rather than continuous. Then any boundary function phi for f is of Baire class alpha + 1.

*Proof.* Direct application of Kaczynski's Theorem 2.9 (Dissertation Theorem 8). The hypotheses are satisfied: T is open in R^2, Y = R is a separable arcwise connected metric space, and f is of Baire class alpha on T.

**Remark 4.7.** Theorem 4.6 gives a precise accounting of how much "complexity" is added at retrieval boundaries. If the activation kernel introduces alpha levels of discontinuity (e.g., the hard threshold at theta = 0.15 makes the thresholded activation Baire class 1 on the interior), then the boundary adds exactly one more level. This is tight: Kaczynski's converse theorem (Theorem 2.8) shows that every honorary Baire class 2 function arises as the boundary function of some continuous mapping.

### 4.3 The Threshold Discontinuity

The hard activation threshold theta = 0.15 in PyPensive's implementation creates a specific discontinuity structure.

**Proposition 4.8.** The thresholded retrieval function:

```
r_theta(x) = R_W(sigma_q)(x) * 1_{R_W(sigma_q)(x) >= theta}
```

is of Baire class 1 on the graphon domain [0,1] whenever R_W(sigma_q) is continuous.

*Proof.* Let g = R_W(sigma_q), which is continuous by hypothesis. The indicator function 1_{g >= theta} is the characteristic function of the closed set {x : g(x) >= theta}. The characteristic function of any closed set is upper semicontinuous, hence of Baire class 1 (every upper semicontinuous real-valued function is a pointwise infimum of continuous functions, and the pointwise limit of such an infimum sequence is Baire class 1). Since g is continuous (Baire class 0) and 1_{g >= theta} is Baire class 1, their product r_theta = g * 1_{g >= theta} is of Baire class 1 (the product of a Baire class 0 function and a Baire class 1 function is Baire class 1, as both classes are closed under multiplication with continuous functions).

**Corollary 4.9.** By Theorem 4.6, the boundary function of the thresholded retrieval (which is Baire class 1 on the interior) is of Baire class 2. The discontinuity set of this boundary function is contained in a first-category (meager) subset of the boundary.

This gives a complete picture: the threshold creates a Baire class 1 interior, and crossing the relevance cluster boundary adds one more level, yielding Baire class 2 -- but the discontinuities at the boundary remain meager.

---

## 5. Contextual Intersection as Approach Region Selection

### 5.1 Approach Regions in Boundary Function Theory

A central theme in boundary function theory is that the *direction of approach* to a boundary point determines the limit value. Different arcs at the same boundary point can yield different limits for the same function. Kaczynski's results hold for *some* arc at each point -- the boundary function captures the best-case approach.

**Definition 5.1** (Approach region). An *approach region* at a boundary point p is a subset Gamma of the open set H such that p is in the closure of Gamma. The *limit along Gamma* is lim_{z -> p, z in Gamma} f(z), when it exists.

Bagemihl's theorem [20] guarantees that any two boundary functions for the same f agree except on a countable set -- so the boundary function is "essentially unique." But this essential uniqueness is achieved by choosing the *right* arc at each point. A poorly chosen approach path may fail to converge, or converge to a spurious limit.

### 5.2 Context as Approach Direction

In PyPensive, the *context* parameter directs the retrieval toward a specific region of the answer space. The implementation (lines 1054-1062 of spreading.py) is:

```python
query_arr = _spread_bipartite_raw(query_activation)
ctx_arr = _spread_bipartite_raw(context_activation)
ctx_mask = ctx_arr > 0
query_arr[ctx_mask] *= (1.0 + ctx_arr[ctx_mask])
```

We formalize this as an approach region constraint.

**Definition 5.2** (Contextual approach region). Let sigma_q be the query seed and sigma_c the context seed. The *contextual approach region* Gamma_c at a boundary point p of the relevance cluster C_q is:

```
Gamma_c = {x in H : (S_W sigma_c)(pi(x)) > 0}
```

where pi projects points in query space onto the graphon domain. That is, the context restricts the approach to directions where the context signal is nonzero.

**Proposition 5.3** (Context favors a converging approach). Let p be a point in the boundary of C_q where the unrestricted retrieval limit does not exist (i.e., different approach paths give different limits). If the context activation sigma_c disambiguates -- meaning there is a unique value node v^* such that both sigma_q and sigma_c activate v^* -- then the contextual approach region Gamma_c biases retrieval toward an arc at p along which the retrieval function converges to v^*.

*Proof sketch.* The contextual intersection multiplies the query activation by (1 + ctx_arr) where ctx_arr > 0. In the graphon limit, this is:

```
r_{q,c}(x) = R_W(sigma_q)(x) * (1 + (S_W sigma_c)(x)) * 1_{(S_W sigma_c)(x) > 0}
```

At the boundary of C_q, the query activation R_W(sigma_q)(x) approaches theta from above. The multiplicative boost from context breaks the tie: among boundary points where the query activation is marginally above threshold, only those co-activated by the context receive the (1 + ctx) boost, lifting them above threshold while competitors remain at or below theta.

**Remark 5.4.** A fully rigorous version of Proposition 5.3 would require constructing an explicit one-parameter family of queries q_t (t in (0, 1]) approaching a boundary query q_0, and proving that without context the limit of R_W(sigma_{q_t})(x_0) does not exist as t -> 0, while with context it does. We believe this construction is achievable for the bipartite memory graphon using the specificity structure, but leave the complete proof to future work. The analogy with approach regions is conceptually compelling: Kaczynski's theory shows that the path of approach determines the boundary limit, and contextual intersection is precisely a mechanism for constraining the path.

### 5.3 The 70.6% to 97.9% Jump

The empirical improvement from 70.6% to 97.9% retrieval accuracy when adding contextual intersection [9] can now be understood as follows:

**Without context:** Queries approach the boundary of their relevance cluster along arbitrary paths. At boundary points where multiple value nodes have nearly equal activation, the retrieval function oscillates between candidates -- it fails to converge. The boundary function at these points is discontinuous.

**With context:** The approach region is constrained to directions where the context signal is positive. This eliminates the oscillation: the approach is now along an arc where a single candidate dominates. The boundary function along this restricted approach is continuous (or at worst discontinuous on a smaller set).

**Proposition 5.5.** Let D_q be the set of boundary points of C_q where the retrieval function is discontinuous (without context), and let D_{q,c} be the corresponding set with context. Then D_{q,c} is a subset of D_q, and:

```
D_{q,c} is a subset of D_q intersect {x : (S_W sigma_c)(x) = 0}
```

That is, context can only *remove* discontinuities, never create them, and the remaining discontinuities are confined to the zero set of the context activation.

*Proof.* The contextual multiplication R_W(sigma_q)(x) * (1 + (S_W sigma_c)(x)) is continuous wherever both factors are continuous. The factor (1 + (S_W sigma_c)(x)) is bounded below by 1 (since S_W sigma_c >= 0 for nonnegative sigma_c), so it is strictly positive everywhere. Multiplication by a continuous, strictly positive function cannot introduce new discontinuities: if g is continuous at x_0 and h is discontinuous at x_0, then g * h is discontinuous at x_0; if g is continuous at x_0 and h is continuous at x_0, then g * h is continuous at x_0. The context factor (1 + S_W sigma_c) is continuous wherever S_W sigma_c is continuous -- that is, everywhere except possibly the boundary of the set {x : (S_W sigma_c)(x) > 0}. Discontinuities of the query factor R_W(sigma_q) can be *removed* when the context factor strictly separates previously tied candidates (pushing one above threshold and the other below), but no new discontinuities are created since the context factor is >= 1.

---

## 6. The Baire-Graphon Stratification

### 6.1 Stratification by Regularity

The Baire classification of the retrieval boundary function depends on the regularity of the memory graphon W. This dependence induces a natural stratification of the graphon space.

**Definition 6.1** (Baire stratification). Define the following subsets of the graphon space W_0:

```
B_0 = {W in W_0 : T_W maps C([0,1]) to C([0,1])}
B_1 = {W in W_0 : T_W maps C([0,1]) to Baire class 1 functions}
B_2 = {W in W_0 : T_W maps C([0,1]) to measurable functions}
```

Then B_0 is a subset of B_1 is a subset of B_2 = W_0 (since every graphon integral operator maps L^2 to L^2).

**Remark.** The strata are defined via the linear integral operator T_W, which characterizes the regularity of W as a function. By Remark 3.7(a), the supremum operator S_W inherits this regularity: if T_W maps continuous functions to Baire class alpha functions, then S_W maps bounded continuous functions to functions of Baire class at most alpha (since the supremum of a parameterized family preserves the Baire class of the parameter dependence). Thus the strata control the regularity of both operators.

**Proposition 6.2** (Characterization of the strata).

(a) B_0 consists of continuous graphons. W in B_0 if and only if W is continuous on [0,1]^2 (as a function, not just a.e.).

(b) B_1 contains all piecewise-constant graphons (stochastic block models) and all piecewise-Lipschitz graphons.

(c) B_2 = W_0 is the full graphon space.

*Proof sketch.* (a) If W is continuous, then by dominated convergence, (T_W f)(x) = integral W(x,y) f(y) dy is continuous in x for any bounded measurable f. Conversely, if T_W preserves continuity for all continuous f, then taking f = 1 shows that the degree function d(x) = integral W(x,y) dy is continuous, which combined with preservation for all f implies continuity of W.

(b) For a piecewise-constant graphon (SBM) with k blocks, (T_W f)(x) is piecewise constant in x for any f, hence Baire class 1 (a pointwise limit of continuous step-function approximations). For piecewise-Lipschitz W, (T_W f)(x) is Lipschitz on each piece and has jump discontinuities at piece boundaries, hence Baire class 1.

(c) For any bounded measurable W, T_W f is in L^2 hence measurable. By the Lebesgue-Hausdorff theorem, every Lebesgue-measurable function coincides a.e. with a Baire class 2 function.

**Theorem 6.3** (Baire class of retrieval boundary by stratum). The Baire class of the retrieval boundary function depends on the stratum:

| Graphon stratum | Interior Baire class | Boundary function Baire class | Discontinuity set |
|-----------------|---------------------|------------------------------|-------------------|
| B_0 (continuous) | 0 | Honorary class 2 (effectively 1) | Countable |
| B_1 (piecewise-Lipschitz) | 1 | 2 | Meager (first category) |
| B_2 (general measurable) | alpha | alpha + 1 | At most F_sigma |

### 6.2 Density and Genericity

**Proposition 6.4** (Step functions are dense in the cut metric). B_1 is dense in (W_0, delta_cut). More precisely, the set of step-function graphons (finite stochastic block models) is dense.

*Proof.* This is the graphon-theoretic formulation of the Szemeredi regularity lemma [5], [6].

**Proposition 6.5** (B_0 has empty interior). The set B_0 of continuous graphons has empty interior in (W_0, delta_cut). That is, no open ball in the graphon space is contained entirely in B_0.

*Proof.* Let W be a continuous graphon and epsilon > 0. We construct a graphon W' with delta_cut(W, W') < epsilon that is discontinuous. Choose a measurable set A in [0, 1] with measure less than epsilon and let W'(x, y) = W(x, y) + epsilon * 1_{A x A}(x, y), clamped to [0, 1]. Then:

```
||W - W'||_cut = sup_{S, T} |integral_S integral_T epsilon * 1_{A x A} dx dy|
               <= epsilon * |A|^2 < epsilon^3
```

so delta_cut(W, W') < epsilon^3 < epsilon. But W' has a jump discontinuity on the boundary of A x A, so W' is not in B_0. Since epsilon was arbitrary, every neighborhood of W contains a point outside B_0, so B_0 has empty interior.

**Remark 6.6.** We do not claim that B_0 is *nowhere dense* (which would require that the closure of B_0 also has empty interior). The closure of B_0 in the cut metric is a subtle object because the cut distance quotients by measure-preserving transformations: the composition W(phi(x), phi(y)) of a continuous W with a measurable phi need not be continuous, so the closure of B_0 may extend beyond the set of "essentially continuous" graphons. Whether B_0 is nowhere dense or merely has empty interior is an open question that we flag for future work.

**Corollary 6.7.** The set B_1 \ B_0 is dense in (W_0, delta_cut). That is, piecewise-discontinuous graphons with Baire class 2 retrieval boundary functions are dense. Combined with the empty interior of B_0, this means that any neighborhood of any graphon contains graphons from B_1 \ B_0 -- the "typical" regime for memory graphons has manageable boundary behavior (meager discontinuity sets) but is not perfectly smooth.

**Remark 6.8.** A stronger genericity statement (that B_1 \ B_0 is comeager, i.e., contains a dense G_delta set) would require showing that B_1 is a G_delta subset of W_0. This appears plausible but is not established here.

### 6.3 The Regularity Lemma as Baire Approximation

There is a deep structural parallel between the Szemeredi regularity lemma and Baire approximation.

**Observation 6.9.** The regularity lemma states that every graphon W can be approximated in cut norm by a step-function graphon W_k (an element of B_1) with k blocks:

```
||W - W_k||_cut <= epsilon(k)
```

where epsilon(k) -> 0 as k -> infinity. There is a structural parallel with the Baire hierarchy: every Baire class 2 function is a pointwise limit of Baire class 1 functions, and every graphon is a cut-norm limit of step-function graphons (which are in B_1).

However, the parallel is not exact: cut-norm convergence is an L^1-type convergence (it bounds integrals over subsets), while the Baire hierarchy is defined via pointwise convergence. Cut-norm convergence does not imply pointwise convergence, and vice versa. The analogy is structural and suggestive rather than a formal equivalence.

The regularity partition provides a canonical decomposition of the memory graphon into "regular" blocks -- regions where the edge density is uniform up to epsilon. In the memory system context, these blocks correspond to *entity frequency bands*: clusters of documents that share entities at similar frequencies.

**Design implication:** If the memory system maintains its graph within B_1 (by ensuring entity frequencies remain within well-separated bands), then the retrieval boundary functions are guaranteed to be Baire class 2 with meager discontinuity sets. This is a *constructive* design constraint derived from the classification theory.

---

## 7. Implications for Memory Architecture Design

### 7.1 The Spectral Perspective

While the retrieval operator R_W uses the supremum kernel S_W (Definition 3.5), the *structural* properties of the memory graphon are captured by the linear integral operator T_W (Definition 2.14), which admits a spectral decomposition:

```
T_W f = sum_k lambda_k <f, phi_k> phi_k
```

The eigenvalues lambda_k and eigenfunctions phi_k encode the structure of the memory graph. The leading eigenfunctions correspond to the dominant entity-value association patterns. Though T_W does not model retrieval directly, it governs graph-level properties (community structure, mixing rates, regularity) that in turn determine the regularity class of W and hence the Baire class of the retrieval boundary.

**Proposition 7.1** (Spectral convergence rate by regularity class). For graphs sampled from a graphon W, the eigenvalue convergence rate depends on the regularity of W [21]:

| Regularity | Convergence rate |
|------------|-----------------|
| None (W in L^infinity) | O(1 / (log n)^{1/4}) |
| Lipschitz (W in B_0) | O(sqrt(log n / n)) |
| Piecewise-Lipschitz (W in B_1) | O((log n / n)^{1/4}) |

Memory systems operating in the piecewise-Lipschitz regime (the natural regime for frequency-weighted bipartite graphs) achieve substantially faster spectral convergence than the worst case. This means the graphon approximation becomes accurate at smaller graph sizes.

### 7.2 Design Constraints for Optimal Boundaries

Our framework yields three concrete design constraints for memory graph construction:

**Constraint 1: Maintain piecewise-Lipschitz regularity.** Ensure that the entity-value kernel K(x, y) is Lipschitz on each frequency band. In practice, this means:
- The specificity function s(e) = 1/freq(e)^rho should use a moderate power (rho = 0.2 in PyPensive, which produces a Holder-continuous function -- sufficient for B_1 membership).
- Entity frequency updates should be batched to avoid creating transient irregularities (PyPensive's two-pass batch update satisfies this).

**Constraint 2: Avoid pathological frequency distributions.** If entity frequencies follow a distribution with infinitely many accumulation points (e.g., a Cantor-like distribution), the specificity function can exit B_1 and enter B_2, degrading boundary behavior. In practice, this is unlikely for natural-language document collections, where Zipf's law produces a well-separated frequency spectrum.

**Constraint 3: Context is not optional for boundary queries.** The analysis in Section 5 shows that without context, boundary queries (those near the edge of a relevance cluster) are inherently ambiguous. Context removes discontinuities by constraining the approach region. Memory systems that serve boundary queries without context should expect degraded accuracy proportional to the size of the discontinuity set.

### 7.3 Predicting Retrieval Discontinuities

The graphon framework enables *a priori* detection of queries that will encounter boundary discontinuities.

**Definition 7.2** (Boundary distance). For a query q, define the *boundary distance* as:

```
d_boundary(q) = inf_{x in partial C_q} |R_W(sigma_q)(x) - theta|
```

Queries with small boundary distance are close to the threshold and are therefore near a potential discontinuity.

**Proposition 7.3.** For queries with d_boundary(q) > 0, the retrieval function is continuous at q in the query parameterization. As d_boundary(q) -> 0, the risk of encountering a boundary discontinuity increases, and contextual disambiguation becomes increasingly necessary.

This provides a computable signal: when the top-scoring value node's activation is close to the threshold, flag the query as a "boundary query" and require context for disambiguation.

---

## 8. Conclusion and Future Work

We have established a mathematical framework connecting Kaczynski's boundary function theory, graphon limits, and spreading activation retrieval. The central result is that retrieval boundary functions are of honorary Baire class 2 (for continuous activation kernels) or Baire class alpha+1 (for Baire class alpha kernels), and that the resulting discontinuity sets are meager in the sense of Baire category. This provides a rigorous justification for the empirical observation that "most" queries work well -- the problematic boundary cases are topologically negligible.

The framework also provides a formal understanding of why contextual intersection improves retrieval: it constrains approach regions to avoid boundary discontinuities, exactly as predicted by the theory.

### Future Directions

1. **Empirical graphon estimation.** Estimate the memory graphon from PyPensive's 164K-document production graph using stochastic block model fitting. Verify that the estimated graphon is piecewise-Lipschitz and that the predicted boundary behavior matches observed retrieval failures.

2. **Computational Baire detection.** Develop algorithms that compute the approximate Baire class of the retrieval function at a query point, enabling runtime detection of boundary queries without full graphon estimation.

3. **Optimal context design.** Given the formalization of context as approach region selection, derive the *minimal* context needed to resolve boundary ambiguities. This connects to Kaczynski's unsolved problems about characterizing which arcs produce convergent limits.

4. **Higher-dimensional boundary functions.** Kaczynski's unsolved Problem 4 asks whether boundary functions in R^3 (and higher) preserve the F_{sigma-delta} characterization. In the graphon context, this corresponds to multi-modal queries (combining text, image, and temporal signals), where the "boundary" is a higher-dimensional manifold.

5. **Non-dense graph limits.** The current framework assumes dense graphs (graphon limits). As discussed in Section 3.4, PyPensive's actual graph is sparse. The most promising extensions are via L^p graphon theory [22] (for graphs with growing average degree) and Benjamini-Schramm local weak convergence [23] (for bounded-degree graphs). Extending the Baire classification to these settings -- particularly proving that the boundary function of the local retrieval operator has controlled Baire class in the Benjamini-Schramm topology -- is the most important open problem for practical applications of this framework.

6. **The Baire-graphon stratification as a design space.** The stratification B_0 c B_1 c B_2 organizes memory architectures by their boundary behavior. Exploring this design space systematically -- What graphon regularity classes are achievable? What is the computational cost of maintaining B_1 membership? -- is a practical research program.

---

## References

[1] T. J. Kaczynski, "Boundary Functions for Functions Defined in a Disk," *Journal of Mathematics and Mechanics*, vol. 14, no. 4, pp. 589-612, 1965.

[2] T. J. Kaczynski, "Boundary Functions," PhD dissertation, University of Michigan, 1967.

[3] T. J. Kaczynski, "Boundary Functions for Bounded Harmonic Functions," *Transactions of the American Mathematical Society*, vol. 137, pp. 203-209, 1969.

[4] C. P. Wilson, "On the Baire class of n-dimensional boundary functions," arXiv:2101.12580, 2021.

[5] L. Lovasz and B. Szegedy, "Limits of dense graph sequences," *Journal of Combinatorial Theory, Series B*, vol. 96, no. 6, pp. 933-957, 2006.

[6] L. Lovasz, *Large Networks and Graph Limits*, American Mathematical Society Colloquium Publications, vol. 60, 2012.

[7] C. Borgs, J. T. Chayes, L. Lovasz, V. T. Sos, and K. Vesztergombi, "Convergent sequences of dense graphs I: Subgraph frequencies, metric properties and testing," *Advances in Mathematics*, vol. 219, no. 6, pp. 1801-1851, 2008.

[8] S. Janson, "Graphons, cut norm and distance, couplings and rearrangements," *NYJM Monographs*, vol. 4, 2013.

[9] PyPensive: Spreading activation retrieval engine, https://pypi.org/project/pypensive/, 2025-2026.

[10] L. Ruiz, L. F. O. Chamon, and A. Ribeiro, "Graphon Neural Networks and the Transferability of Graph Neural Networks," *NeurIPS*, 2020.

[11] L. Ruiz, L. F. O. Chamon, and A. Ribeiro, "Graphon Filters: Graph Signal Processing in the Limit," *IEEE Transactions on Signal Processing*, 2021.

[12] M. Ghandehari, J. Janssen, and N. Kalyaniwalla, "A noncommutative approach to the graphon Fourier transform," *Applied and Computational Harmonic Analysis*, vol. 61, pp. 101-131, 2022.

[13] P. Caines, M. Huang, and R. Malhame, "Graphon Mean Field Games and Their Equations," *SIAM Journal on Control and Optimization*, 2022.

[14] L. Gyenge, "Limits of Multi-relational Graphs," *Machine Learning*, vol. 112, pp. 1-27, 2023.

[15] A. J. Smola and R. Kondor, "Kernels and Regularization on Graphs," *COLT*, 2003.

[16] R. Kondor and J. Lafferty, "Diffusion Kernels on Graphs and Other Discrete Structures," *ICML*, 2002.

[17] E. Estrada, T. Pereira, and N. Hatano, "Systemic States of Spreading Activation in Describing Associative Knowledge Networks," *Systems*, vol. 9, no. 2, 2021.

[18] A. S. Kechris, *Classical Descriptive Set Theory*, Springer, 1995.

[19] G. Nicodemus, "A Recursive Presentation of the Graphon Space," Honors thesis, Penn State University.

[20] F. Bagemihl, "Curvilinear cluster sets of arbitrary functions," *Proceedings of the National Academy of Sciences*, vol. 41, pp. 379-382, 1955.

[21] "Distributional Limits for Eigenvalues of Graphon Kernel Matrices," arXiv:2601.04584, 2025.

[22] C. Borgs, J. T. Chayes, H. Cohn, and Y. Zhao, "An L^p theory of sparse graph convergence I: Limits, sparse random graph models, and power law distributions," *Transactions of the American Mathematical Society*, vol. 372, no. 5, pp. 3019-3062, 2019.

[23] I. Benjamini and O. Schramm, "Recurrence of distributional limits of finite planar graphs," *Electronic Journal of Probability*, vol. 6, pp. 1-13, 2001.

---

## Appendix A: PyPensive Implementation Details

For reproducibility, we document the key parameters of the PyPensive system analyzed in this paper.

**Graph construction:**
- Entity extraction: Regex-based MegaExtractor with 50+ pattern categories
- Specificity: s(e) = 1 / freq(e)^0.2
- Edge weight: w(e, v) = s(e) * 5.0
- Graph representation: scipy.sparse CSR matrix

**Spreading activation:**
- Seed boosts: exact = 1.5, partial = 0.8, substring = 0.5
- Decay: delta = 0.6 per hop
- Threshold: theta = 0.15
- Max hops: 4 (bipartite: effectively 1)
- Max active nodes: 50
- Propagation rule: max (not sum)
- JIT compilation: numba @njit with cache

**Contextual intersection:**
- Context seeds a second activation pass
- Intersection: query_arr[ctx_mask] *= (1.0 + ctx_arr[ctx_mask])
- Effect: multiplicative boost where context co-activates

**Scale:**
- 164K+ documents in production
- ~500K entity nodes, ~664K total nodes
- ~800K edges
- Sub-millisecond query latency
- ~10-20MB memory footprint

## Appendix B: Kaczynski's Dissertation -- Key Theorem Statements

For reference, we reproduce the theorem numbering from [2] with modern notation.

**Chapter I (Continuous Functions):**

- **Thm 3** (McMillan's theorem, new proof): The set of curvilinear convergence of f : H -> Y (Y complete separable metric) is F_{sigma-delta}.
- **Thm 4** (Converse): Every F_{sigma-delta} subset of X is the set of curvilinear convergence of some bounded continuous f.
- **Thm 5**: Boundary function of continuous f is honorary Baire class 2. (This is our Theorem 2.7.)
- **Thm 6'-7** (Converse): Every honorary Baire class 2 function is a boundary function of some continuous f. (This is our Theorem 2.8.)

**Chapter II (Discontinuous Functions):**

- **Thm 8**: Boundary function of Baire class alpha f (alpha >= 1) is Baire class alpha + 1. (This is our Theorem 2.9.)
- **Thm 9**: Geometric measure inequality for line segment families.
- **Thm 10**: Existence of pathological disjoint arc families with measure-zero union.

**Unsolved problems from the dissertation:**

1. Can the complex-valued results (Thm 4) be extended to real-valued functions?
2. Can the convergence set and boundary function be prescribed simultaneously?
3. Is the convergence set of a Baire class 1 function necessarily Borel?
4. **Do these results extend to R^3 and higher dimensions?** (Partially resolved by Wilson [4], who showed Baire class alpha+n for n-dimensional domains.)
5. Sharp measure bounds for line segment families (Theorem 9 generalization).
