# Baire Class 2 Regularity and Path-Dependent Limits in Graphons

## Executive summary

Path-dependent limits at a “node” of a graphon are not a single well-posed phenomenon until you pin down **what a node is** (graphons are defined only up to null sets and measure-preserving relabelings) and **what “approaching” means** (coordinate topology, neighborhood-metric topology, graph-constrained walks, etc.). In the rawest coordinate sense, you can manufacture arbitrarily violent path dependence on a dense set while staying in the *same* graphon equivalence class (change the function on a null set). This is not a paradox; it is an unavoidable consequence of graphons being measure-theoretic objects. citeturn21view0turn15view1turn31view2

Once you choose a genuine topology on the vertex space, path dependence reduces to classical topology: **a limit is path-independent at a point exactly when the function is continuous there in the chosen topology** (or more precisely, when the function has a unique sequential limit along every allowed approach filter). The main graphon-invariant way to put a topology on “nodes” is the **neighborhood (similarity) pseudometric**  
\[
r_W(x,y)=\|W(x,\cdot)-W(y,\cdot)\|_{L^1},
\]
and a standard refinement is to pass to a **pure graphon**, where this becomes a complete metric with full-support measure. citeturn6view0turn31view2turn20view0

For quantifying “how bad” path dependence is, the most robust tools are:
- **local oscillation** (diameter of cluster values near a point), and its constrained variants (only along admissible paths), and
- **ordinal ranks** (oscillation rank \(\beta\), convergence rank \(\gamma\), separation-type ranks), originally developed for Baire-1 and extended to higher Baire classes, including class 2. citeturn12view0turn11view0turn11view1

Baire class 2 is a critical threshold for your question because Baire-2 functions can be **everywhere discontinuous**, so maximal path dependence (two different limits along two sequences converging to the same point) can occur at every point, even on compact metric node spaces (including those coming from pure graphons). citeturn7view0turn7view1turn31view2

## Baire class 2, measurability, and limit constructions

### Baire hierarchy in a form usable for graphons

Let \(X,Y\) be metrizable spaces (typically \(X\) Polish or compact metric, \(Y\subseteq\mathbb R\) or a separable metric space). The **Baire classes** are defined by transfinite iteration of pointwise limits: Baire class 0 means continuous; for \(1<\xi<\omega_1\), Baire class \(\xi\) means “pointwise limit of a sequence of functions of smaller Baire class.” citeturn7view0turn7view1

In particular:
- **Baire class 1**: pointwise limits of continuous functions. citeturn7view0turn11view0  
- **Baire class 2**: pointwise limits of Baire-1 functions (equivalently, Baire class \(\xi\) with \(\xi=2\)). citeturn7view0turn7view1

A central bridge between descriptive set theory and analysis is the **Lebesgue–Hausdorff–Banach characterization**: for separable metrizable \(Y\),  
\[
f\text{ is Baire class }\xi \iff f \text{ is }\Sigma^0_{\xi+1}\text{-measurable}
\]
(preimages of open sets lie in the \((\xi+1)\)-level Borel class), and a function is **Borel** iff it is Baire class \(\xi\) for some countable ordinal \(\xi\ge 1\). citeturn7view1turn7view0

This matters for graphons because a graphon can be taken to be Borel measurable after modifying on a null set (graphon theory is null-set invariant). citeturn21view0turn6view0

### Borel, Lebesgue, and “pointwise limit” closure

There are three “measurability” layers that can get conflated:

- **Borel measurable**: measurable w.r.t. the Borel \(\sigma\)-algebra of the topology.  
- **Lebesgue (completed) measurable**: measurable w.r.t. the completion by null sets; on \([0,1]\) and \([0,1]^2\) this is the usual Lebesgue completion.  
- **Baire class / Baire measurable**: produced by closing continuous functions under pointwise limits through the Baire hierarchy, which (on metrizable spaces) aligns with Borel measurability via the theorem above. citeturn7view1turn21view0turn7view0

For later use: if \(f=\lim_n f_n\) pointwise and each \(f_n\) is \(\Sigma^0_{\xi_n+1}\)-measurable with \(\xi_n<\xi\), then \(f\) is \(\Sigma^0_{\xi+1}\)-measurable, with an explicit “countable union of countable intersections” description of \(f^{-1}(O)\) for open \(O\). citeturn7view1turn7view0

### Quantifying discontinuity severity: oscillation and ranks

For a real-valued function \(f:X\to\mathbb R\) on a metric space \(X\), a base quantitative notion is **oscillation at a point** (here stated in the form used in rank theory): for a closed \(F\subseteq X\),
\[
\omega(f,x,F)=\inf\Bigl\{\sup_{x_1,x_2\in U\cap F}|f(x_1)-f(x_2)|:\ U\text{ open},\,x\in U\Bigr\}. 
\]
From this one defines a derivative operator \(D_{f,\varepsilon}(F)=\{x\in F:\omega(f,x,F)\ge\varepsilon\}\), and the associated **oscillation rank** \(\beta(f)=\sup_{\varepsilon>0}\mathrm{rk}(D_{f,\varepsilon})\), which is countable for Baire-1 functions and is part of a broader trinity of ranks including a convergence rank \(\gamma\). citeturn12view0turn12view1

Even more relevant for your “classify severity” goal: the rank technology extends beyond Baire-1. There are natural rank extensions to **Baire class \(\xi\)** functions (hence to \(\xi=2\)), preserving many structural properties and giving ordinal-valued complexity measures. citeturn11view0turn11view2turn12view1

## Graphons, graph limits, and topologies

### What a graphon is, and why “a node” is slippery

A (standard) **graphon** is a symmetric measurable function
\[
W:\Omega^2\to[0,1],
\]
where \((\Omega,\mathcal F,\mu)\) is a probability space (often \([0,1]\) with Lebesgue measure). citeturn13view0turn21view0

Graphons represent limits of dense graph sequences via **homomorphism densities**. For a finite graph \(F\) with \(k\) vertices,
\[
t(F,W)=\int_{[0,1]^k}\prod_{ij\in E(F)} W(x_i,x_j)\,dx,
\]
and \(t(F,G_n)\to t(F,W)\) is the core “left convergence” notion. citeturn22view0turn23search0turn23search5

Crucially, graphon theory identifies functions that differ by:
1. **null sets**, and
2. **measure-preserving relabelings** (or more generally, appropriate couplings).

One common “isomorphism” notion is: \(W_1\sim W_2\) if \(t(F,W_1)=t(F,W_2)\) for every finite graph \(F\); a theorem (Borgs–Chayes–Lovász) gives an equivalent pullback representation through a common graphon. citeturn22view0turn31view0turn31view2

So, unless you fix a representative and a topology, talking about “the node \(x\)” is like talking about “the 17th raindrop in a cloud.” The object is the distribution, not the label.

### Core norms and metrics on graphons

On a fixed probability space, you can compare labeled graphons with \(L^p\) norms:
\[
\|W\|_p=\Bigl(\int_{\Omega^2}|W|^p\Bigr)^{1/p},\quad 1\le p<\infty,
\]
and \(\|W_1-W_2\|_p\) yields \(L^p\) topology modulo equality almost everywhere. (This is standard functional analysis; graphon papers typically focus on \(p=1,2,\infty\) and the cut norm.) citeturn13view1turn24view3turn21view0

For graph limits, the decisive norm is the **cut norm**. For \(W\in L^1(\Omega^2)\),
\[
\|W\|_{\square,1}=\sup_{S,T\subseteq\Omega}\Bigl|\int_{S\times T}W\,d\mu^{\otimes2}\Bigr|,
\]
equivalently (up to constants) as a dual form over bounded test functions \(f,g\). citeturn13view1turn24view3

A basic inequality that helps compare topologies is:
\[
\int_{\Omega^2} W \le \|W\|_{\square}\le \|W\|_1,
\]
so cut control is weaker than \(L^1\) control (but still strong enough to control subgraph densities). citeturn24view3turn13view1

To compare unlabeled graphons, one uses the **cut metric** \(\delta_\square\), defined by pulling graphons back to a common coupling space and taking the infimum cut norm difference:
\[
\delta_\square(W_1,W_2)=\inf_{\text{couplings }(\varphi_1,\varphi_2)} \bigl\|W_1^{\varphi_1}-W_2^{\varphi_2}\bigr\|_\square.
\]
This is a pseudometric, because relabelings can give distance 0 without pointwise equality. citeturn15view1turn21view0turn22view0

A structural cornerstone is that the graphon space modulo \(\delta_\square=0\) is **compact**, a reformulation closely tied to regularity lemmas. citeturn27view0turn22view0turn27view1

### Pointwise almost-everywhere comparisons

You can speak of \(W_n(x,y)\to W(x,y)\) for almost every \((x,y)\) only after choosing a common labeled domain and versions of the graphons. This is not graphon-invariant: relabelings can destroy coordinatewise convergence without changing the underlying limit object. The standard viewpoint is therefore: a graphon is defined “a.e.,” and the meaningful convergences are \(L^p\)/cut/coupling based. citeturn21view0turn15view1turn31view0

## What it can mean to approach a node in a graphon

There are multiple, inequivalent nonequivalent notions. Your question becomes sharp only after choosing one.

### Coordinate approach versus graphon-invariant approach

If a graphon is represented on \([0,1]\), the coordinate topology gives a naive notion:
\[
x_n\to x \text{ in }([0,1],|\cdot|).
\]
But this is not invariant under measure-preserving transformations \(x\mapsto \phi(x)\): \(\phi\) can be wildly discontinuous as a map of the Euclidean interval while still measure-preserving, and changing \(W\) on a null set changes pointwise behavior without changing the graphon. citeturn15view1turn21view0turn22view0

The standard graphon-invariant fix is to use the **neighborhood pseudometric** (also called similarity distance):
\[
r_W(x,y)=\|W(x,\cdot)-W(y,\cdot)\|_{L^1}=\int_\Omega |W(x,z)-W(y,z)|\,dz,
\]
defined (after harmless null-set edits) for all pairs. citeturn6view0turn31view2turn6view3

If \(r_W(x,y)=0\) for distinct \(x\ne y\), the vertices are **twins** (indistinguishable by their neighborhoods). Passing to a quotient (and then completing) yields a canonical “type space.” citeturn31view2turn6view3

### Pure graphons as the clean node space

A graphon \((J,W)\) is called **pure** if \((J,r_W)\) is a complete metric space and the underlying probability measure has full support (every nonempty open set has positive measure). Every graphon is weakly isomorphic to a pure graphon, so you can assume purity without losing graph-limit information. citeturn31view2turn20view0

Pure graphons are the right setting if you want topological language like “path,” because you now have an honest metric space of nodes.

### Sequences and paths

Let \((J,r_W)\) be the pure node space and \(f:J\to\mathbb R\).

- **Sequential approach**: \(x_n\to x\) if \(r_W(x_n,x)\to 0\).  
- **Path approach (topological)**: a continuous curve \(\gamma:[0,1)\to J\) with \(\gamma(t)\to x\) as \(t\uparrow 1\). (In metric spaces, any such path produces a sequence, and sequences capture continuity.)  
- **Graph-constrained path approach (graph-like)**: pick a threshold \(\varepsilon>0\) and define the support graph \(G_{W,\varepsilon}\) on \(J\) where \((u,v)\) is an edge if \(W(u,v)\ge\varepsilon\). Then a “walk” is a sequence with consecutive edges in \(G_{W,\varepsilon}\). Whether such walks can converge to a given \(x\) in \(r_W\) depends heavily on \(W\) and \(\varepsilon\); this is a genuinely different constraint system than unconstrained convergence. (This is a natural definition, but it is an extra modeling choice, not a built-in part of graphon theory.) citeturn21view0turn31view2

A compact way to visualize the dependency chain:

```mermaid
flowchart TD
  A[Graphon object (equivalence class)] --> B[Choose representative: Borel version / pure graphon]
  B --> C[Choose node topology: coordinate | neighborhood metric r_W]
  C --> D[Choose approach family: all sequences | continuous paths | graph-constrained walks]
  D --> E[Define limits / cluster sets of f along approaches]
  E --> F[Quantify severity: oscillation; oscillation rank β; convergence rank γ; Borel complexity]
```

### Comparison table for “approach” notions

| What is approaching what | Underlying topology/metric | Is it graphon-invariant? | Typical use | What “path dependence” really measures |
|---|---|---|---|---|
| \(x_n\to x\) in \([0,1]\) coordinates | Euclidean \(|x-y|\) | No (relabelings destroy it) | convenience in explicit formulas | continuity failure of a chosen representative |
| \(x_n\to x\) in node space | neighborhood metric \(r_W\) on a pure graphon | Yes (up to weak isomorphism) | intrinsic node geometry / types | continuity failure in the intrinsic node topology |
| \(W_n\to W\) (graphons) | cut metric \(\delta_\square\) | Yes | graph-limit topology, testing, compactness | not a node-level notion; compares whole kernels |
| “walks” \((x_n)\) with \(W(x_n,x_{n+1})\ge\varepsilon\) | constraint + chosen convergence metric | depends on setup | modeling dynamics/random walks | continuity along constrained approach families |

(Definitions of \(\delta_\square\), \(r_W\), and pure graphons are in the cited sources.) citeturn15view1turn31view2turn6view0turn21view0

## Path dependence on graphons: characterization and severity measures

### The basic topological fact

Fix a metric space \((X,d)\) and a function \(f:X\to\mathbb R\). For a point \(x\in X\), the following are equivalent:

- there exists \(L\in\mathbb R\) such that for every sequence \(x_n\to x\), one has \(f(x_n)\to L\);
- the function \(\tilde f\) defined by \(\tilde f(x)=L\) and \(\tilde f=f\) on \(X\setminus\{x\}\) is continuous at \(x\).

So, for “all sequences” as the approach family, **path-independent limits are exactly continuity** (up to defining the value at the limit point appropriately). This is elementary metric topology; the useful takeaway is: once you choose a topology, the question becomes “where is \(f\) continuous?” and “how discontinuous is it?” (The next subparts quantify “how.”)

### Local oscillation and cluster sets

Given an approach family \(\mathcal A(x)\) (for example: all sequences converging to \(x\), or only graph-constrained walks converging to \(x\)), define the **cluster set**
\[
\mathrm{Clust}_{\mathcal A}(f,x)=\{\,\ell:\exists (x_n)\in\mathcal A(x)\text{ with }f(x_n)\to \ell\,\}.
\]

Then:
- path-independent limit exists iff \(\mathrm{Clust}_{\mathcal A}(f,x)\) is a singleton;
- a simple severity proxy is \(\mathrm{diam}(\mathrm{Clust}_{\mathcal A}(f,x))\).

For unconstrained approaches in a metric space, this diameter is controlled by the standard oscillation \(\omega(f,x,X)\) (the infimum, over neighborhoods of \(x\), of the oscillation of \(f\) on that neighborhood), in the sense encoded by the definition used in rank theory. citeturn12view0turn12view1

A graphon-adapted refinement is to define **constrained oscillation** by restricting the supremum in \(\omega(f,x,F)\) to points reachable by admissible paths within the neighborhood. That produces a “directional” or “path-filtered” oscillation index, analogous to studying limits along curves in multivariable calculus, but now on the graphon node space.

### Ordinal ranks as “how discontinuous” scales

For Baire-1 functions, the oscillation derivative \(D_{f,\varepsilon}\) and its transfinite iteration produce a countable ordinal rank \(\beta(f)\) (and similarly a convergence rank \(\gamma(f)\) via ranks of convergent approximating sequences of continuous functions). Larger ordinals correspond to more elaborate discontinuity patterns, even when all discontinuities lie in a meager set. citeturn12view0turn12view1turn11view0

For your focus on **Baire class 2**, you have two complementary approaches:

1. **Descriptive-set complexity**: Baire-2 implies \(\Sigma^0_3\)-measurability, so sets like \(\{x:f(x)>\alpha\}\) have (at worst) \(G_{\delta\sigma}\)-type complexity; this gives an upper bound on the definability complexity of discontinuity and oscillation sets. citeturn7view1turn7view0  
2. **Rank extensions**: the rank framework (originally \(\alpha,\beta,\gamma\) for Baire-1) extends to Baire class \(\xi\), including \(\xi=2\), producing ordinal-valued invariants \(\alpha_\xi^\ast,\beta_\xi^\ast,\gamma_\xi^\ast\) that measure complexity in a way directly aligned with pointwise-limit constructions. citeturn11view0turn11view2turn12view1

### Measurable versus topological notions of “severity”

Graphons are stable under null-set changes; topology is not. A limit pathology that depends on a “thin” set can be totally irrelevant to graph-limit structure.

A useful “graphon-respecting” adjustment is to work with **essential** versions:
- replace sup/inf on neighborhoods by essential sup/inf (ignore sets of measure zero in the neighborhood),
- or define “approximate limits” (limits in measure inside shrinking balls).

The point is not that one is morally better; it is that only some of these are invariant under the graphon equivalences used in graph limit theory. The sources explicitly stress that graphons are defined only up to a.e. equality and that Borel and completed (Lebesgue) versions coincide up to null sets. citeturn21view0turn6view0turn31view2

## Explicit examples and counterexamples on graphons

### A null-set modification that explodes path dependence but changes no graphon

Define on \([0,1]^2\):
\[
W_0(x,y)=xy,\qquad
W(x,y)=\begin{cases}
1,& x\in\mathbb Q\cap[0,1],\ y\in\mathbb Q\cap[0,1],\\
xy,&\text{otherwise.}
\end{cases}
\]

Then \(W=W_0\) almost everywhere because \(\mathbb Q\cap[0,1]\) has Lebesgue measure \(0\), so the modified set \((\mathbb Q\cap[0,1])^2\) is null in \([0,1]^2\). In graphon theory, these are the same kernel (same \(\delta_\square\), same homomorphism densities, etc.). citeturn21view0turn15view1turn22view0

But in the coordinate topology, at any point \((x_*,y_*)\) you can approach along rational-rational sequences (seeing value \(1\) eventually) or irrational-generic sequences (seeing value near \(x_*y_*\)), so the 2D limit behavior as \((x,y)\to(x_*,y_*)\) can become path-dependent everywhere. This example is the canonical warning: **coordinatewise path dependence is not an invariant of a graphon**.

### A Baire class 2 graphon that is discontinuous everywhere yet equivalent to the zero graphon

Let \(Q=\mathbb Q\cap[0,1]\). Define
\[
W_{\oplus}(x,y)=\mathbf 1_{(x\in Q)\ \oplus\ (y\in Q)},
\]
where \(\oplus\) is XOR (“exactly one is rational”).

- The set \(\{(x,y):W_\oplus(x,y)=1\}=(Q\times ([0,1]\setminus Q))\cup(([0,1]\setminus Q)\times Q)\) is Borel (indeed at low Borel complexity), hence \(W_\oplus\) is Borel measurable, thus some Baire class; in fact it is \(\Sigma^0_3\)-measurable and therefore Baire class 2 by the Lebesgue–Hausdorff–Banach theorem. citeturn7view1turn7view0  
- \(W_\oplus\) is discontinuous everywhere in the product coordinate topology, since every neighborhood of \((x,y)\) contains points with either rational/irrational status pattern.  
- Nevertheless, \(W_\oplus=0\) almost everywhere because \(Q\) has measure \(0\), hence this graphon is equivalent (as a graphon object) to the zero graphon. citeturn21view0turn31view2

So, yes: Baire-2 level discontinuity can be “maximal” in the topological sense while being “invisible” to graph-limit topology.

### A pure graphon where the neighborhood metric is exactly Euclidean

Consider the threshold graphon
\[
W_{\ge}(x,y)=\mathbf 1_{x+y\ge 1}.
\]
This is symmetric and measurable, hence a graphon. citeturn21view0turn13view0

For each \(x\), the neighborhood section is \(W_{\ge}(x,\cdot)=\mathbf 1_{[1-x,1]}\). Therefore
\[
r_{W_{\ge}}(x,x')=\|\,\mathbf 1_{[1-x,1]}-\mathbf 1_{[1-x',1]}\,\|_{L^1}=|x-x'|.
\]
So the intrinsic node metric \(r_W\) coincides with the standard metric on \([0,1]\), and \((J,r_W)\) is complete with full-support Lebesgue measure. Hence this is an explicit **pure graphon** in which coordinate and neighborhood notions of approach agree. citeturn31view2turn6view0turn20view0

This example is useful because it lets you transplant classical Baire-2 path phenomena onto an intrinsic graphon node space.

### A Baire class 2 node-signal with maximally path-dependent limits on a pure graphon

On the same node space \([0,1]\) for \(W_{\ge}\), define the node function (“signal”)
\[
f(x)=\mathbf 1_{x\in\mathbb Q}.
\]

By the descriptive-set characterization, \(f\) is Baire class 2 (it is \(\Sigma^0_3\)-measurable), and it is not Baire class 1 because it is discontinuous everywhere, while Baire-1 functions have points of continuity on every nonempty closed set. citeturn7view1turn11view0turn12view0

For every \(x\in[0,1]\):
- choose rationals \(q_n\to x\), then \(f(q_n)=1\to1\);
- choose irrationals \(u_n\to x\), then \(f(u_n)=0\to0\).

Thus \(\mathrm{Clust}(f,x)=\{0,1\}\) for all \(x\), and the oscillation is maximal: \(\omega(f,x,[0,1])=1\). This is the clean archetype of “limit depends on the path” (in the sense “depends on the convergent sequence”), realized on a graphon-invariant node topology because \(r_W=|\cdot|\) here. citeturn12view0turn31view2

### When path-independence is guaranteed: graphon-derived continuous functions in the neighborhood metric

A key counterweight is that many natural functions built from \(W\) are automatically continuous in \(r_W\), hence path-independent along any \(r_W\)-approach.

For a pure graphon \((J,W)\) and any \(g\in L^1(J)\), define
\[
(T_W g)(x)=\int_J W(x,y)g(y)\,dy.
\]
Then \(T_W g\) is continuous on \((J,r_W)\). In particular, the **degree function** \(d_W(x)=\int_J W(x,y)\,dy\) (take \(g\equiv 1\)) is continuous in \(r_W\). citeturn6view3turn31view2turn20view0

Interpretation: if your “function whose limit you care about” is a linear integral functional of a node’s neighborhood profile, then intrinsic path dependence vanishes. If instead you insist on point evaluations (of \(W(x,y_0)\) as \(x\to x_0\), or of a highly non-regular signal), you can get severe path dependence.

### A graphon where the kernel \(W\) itself is not continuous in the intrinsic node metric

Even on pure graphons, \(W\) need not be continuous as a function on \((J,r_W)\times(J,r_W)\). The literature explicitly notes that expecting full continuity of \(W\) in the similarity metric is “too much,” with simple step-type graphons as counterexamples. citeturn6view3turn31view2

For \(W_{\ge}(x,y)=\mathbf 1_{x+y\ge 1}\), the discontinuity along the line \(x+y=1\) persists in the intrinsic metric because \(r_W\) is just Euclidean. Approaching a boundary point \((x_0,1-x_0)\) from the region \(x+y>1\) forces \(W\to1\), while approaching from \(x+y<1\) forces \(W\to0\). This is 2D path dependence at an edge-level point, even though many node-level integral summaries are continuous.

## Literature connections and how to use them for this question

### Core graphon sources that implicitly define the right topological framework

Graphon theory’s standard sources supply exactly the infrastructure needed to interpret “approaching a node” in an invariant way:

- Lovász’s monograph entity["book","Large Networks and Graph Limits","lovasz 2012"] develops the graphon space, proves compactness in the cut metric, and defines the neighborhood pseudometric \(r_W\), pure graphons, and continuity results for integral operators. citeturn27view0turn6view0turn6view3  
- entity["people","Svante Janson","mathematician"] provides an extensive survey of cut norm/metric, couplings, Borel versus Lebesgue graphons, and equivalence theory, emphasizing that Borel and Lebesgue versions coincide up to a.e. changes, and giving clean definitions of \(\delta_\square\). citeturn13view0turn15view1turn21view0  
- entity["people","Balázs Szegedy","mathematician"] together with Lovász formalizes the “pure graphon” viewpoint and defines metrics on node spaces that depend only on weak isomorphism class, exactly the context in which “paths in node space” make sense. citeturn31view2turn20view0  
- entity["people","Persi Diaconis","statistician"] and entity["people","Svante Janson","mathematician"] clarify the probabilistic representation of graphons as kernels generating exchangeable random graphs, reinforcing that node labels are artifacts and invariants are distributional. citeturn30view0

### Baire class theory sources that provide the classification apparatus

On the descriptive set theory side, the minimal toolkit you need is:

- Kechris’s definition of Baire classes via pointwise limits (Baire hierarchy) and the general framework for Baire/Borel measurability. citeturn7view0  
- The Lebesgue–Hausdorff–Banach equivalence between Baire class \(\xi\) and \(\Sigma^0_{\xi+1}\)-measurability (hence Borel iff some Baire class). citeturn7view1  
- The oscillation-rank methodology and its higher-class extensions, which provide ordinal-valued “severity” measures aligned with pointwise approximation structure. citeturn12view0turn11view0turn11view2

### What seems to be missing in the literature, and why it matters

There is plenty of literature on:
- graphon convergence, equivalence, canonical forms, and node metrics (graph limits), citeturn27view0turn21view0turn15view1turn31view2  
- Baire class/rank theory and descriptive complexity of pointwise limit constructions (descriptive set theory). citeturn7view1turn12view0turn11view0  

But there is comparatively little that explicitly asks: “Given the intrinsic node metric of a graphon, what is the Baire-rank profile of naturally arising node observables, and what does that say about path-dependent limits?” So, for your specific question, the best way to proceed is to **compose** the two frameworks:
1. put the graphon into a pure form and work on \((J,r_W)\), citeturn31view2turn6view3  
2. treat the node observable \(f\) as a function on that compact (or at least Polish) metric space, and apply Baire class 2 and rank tools there, citeturn7view0turn11view0turn12view0  
3. if “paths through the graph” impose extra constraints (thresholded edges, random-walk admissibility), analyze constrained oscillation/cluster sets as a refinement of classical oscillation.

That gives a coherent classification scheme that is both graphon-invariant (once you commit to \(r_W\) and purity) and rich enough to quantify severity (oscillation diameters and ordinal ranks).

### Summary chart: approach mode versus what Baire class tells you

| Function regularity on node space \((J,r_W)\) | What you can guarantee about path limits | Typical “size” of bad set | Severity quantifiers that work well |
|---|---|---|---|
| Continuous (Baire 0) | unique limit along any \(r_W\)-approach | none | local oscillation \(=0\) |
| Baire 1 | unique limits on a large set of points; structured discontinuities | discontinuities are describable via oscillation derivatives and have countable-rank structure | oscillation rank \(\beta\), convergence rank \(\gamma\) citeturn12view0turn12view1turn11view0 |
| Baire 2 | can be nowhere continuous, so path dependence can happen everywhere | no “category smallness” guarantee | extended ranks \(\beta^\ast_2,\gamma^\ast_2\) and \(\Sigma^0_3\) definability bounds citeturn7view1turn11view2turn11view0 |
| Merely measurable (mod null sets) without a fixed topology | path dependence can be made arbitrarily bad by null-set edits | can be dense, can be all points, yet graphon-equivalent to a tame kernel | must switch to essential notions (approximate limits) or to intrinsic node topology citeturn21view0turn31view2 |

This is the main conceptual punchline: **Baire class 2 is flexible enough to encode maximal path dependence, and graphons are flexible enough (via null-set invariance) to make path dependence either meaningless or deeply structural depending on the topology you choose.**