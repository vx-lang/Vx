# The Hardware Monad: Categorical Semantics of Topologies and Transfers

This document gives the formal background for Vx's placement discipline — the
rules that decide *where* a value lives, when it may be read, and how it crosses
hardware boundaries. The aim is to show that these rules are not ad-hoc: they are
an instance of structures that are well studied in the semantics literature (a
**graded, parameterised monad** over a **category of located values**), which is
precisely why they compose soundly.

Audience: language designers and compiler engineers. Companions:
[`formal_semantics.md`](formal_semantics.md) (the operational rules, the `Verified`
return-type contract, and the planned `try_pin` acquisition layer),
[`seam_obligations.md`](seam_obligations.md) (the per-hop safety check discharged
by `z3`), and the design record
[`../discussions/brainstorming/hardware_monad_topology.md`](../discussions/brainstorming/hardware_monad_topology.md)
(the open, user-definable topology model this semantics targets).

## 1. Memory spaces form a category

Fix a set of **memory spaces** $\\mathbb{M}$ (`HostDRAM`, `NPUHBM`, `GpuHbm`,
`LocalSRAM`, …). A **transfer** is a directed way of relocating a value from one
space to another. Spaces and transfers form a category $\\mathcal{M}$:

- **objects** — the spaces $P, Q, R \\in \\mathbb{M}$;
- **morphisms** — $P \\xrightarrow{t} Q$, a transfer (a *seam*, in Vx's
  vocabulary);
- **composition** — $t' \\circ t : P \\to R$ is "do $t$, then $t'$" (path
  concatenation);
- **identity** — $\\mathrm{id}\_P : P \\to P$, the no-op / cost-free stay.

This category is exactly the `TransferCostGraph` (`src/arch.rs`): its edges are
the available seams and $\\mathrm{transfer\\\_path}$ (Dijkstra) computes the
composite morphism $P \\to Q$. The categorical laws hold operationally:
associativity of composition is "a multi-hop path is the same however you
parenthesise the hops," and the identity laws are "prefixing or suffixing a
no-op changes nothing."

## 2. Cost is a grading monoid

Each morphism carries a **grade** drawn from an ordered monoid
$(\\mathbb{W}, \\cdot, \\mathbf{1}, \\sqsubseteq)$:

- $\\mathbf{1}$ is the grade of $\\mathrm{id}$ (a stay costs nothing);
- $w \\cdot v$ grades the composite $t' \\circ t$ (grades multiply along a path);
- $\\sqsubseteq$ orders grades so cheaper is comparable to dearer.

In Vx today $\\mathbb{W}$ is the additive monoid of transfer costs
$(\\mathbb{N}, +, 0)$, so Dijkstra minimises $w$ over composites. Nothing in the
theory relies on this choice; $\\mathbb{W}$ is the parameter that makes the monad
below *graded* in the sense of Katsumata's parametric effect monads.

## 3. Located values and the hardware monad

A value "living on" a space is a **located value**, written $A @ p$ and realised
in Vx as `Pinned<A, P>` / `Ref<A, P>`. Relocation is not a plain function
$A @ P \\to A @ Q$: it changes the index $P \\rightsquigarrow Q$ and incurs a
grade. The right structure is a **graded, parameterised (indexed) monad**:

$$
M[P, Q, w, A] \\;\\;-\\;\\; \\text{a computation from location } P \\text{ to } Q,\\ \\text{grade } w,\\ \\text{yielding } A
$$

with three operations,

$$
\\mathrm{return} : A \\to M[P, P, \\mathbf{1}, A]
$$
$$
\\mathrm{bind} : M[P, Q, w, A] \\to (A \\to M[Q, R, v, B]) \\to M[P, R, w \\cdot v, B]
$$
$$
\\mathrm{transfer} : (A @ P) \\to M[P, Q, \\mathrm{cost}(P \\to Q), A @ Q]
$$

Read the indices as a little Hoare logic of place: `return` neither moves nor
costs; `bind` demands the *pre*-index of the continuation meet the *post*-index of
its argument at $Q$, and multiplies the grades; `transfer` is the graded
generator — the only operation that changes the index, and it changes it by
exactly one morphism of $\\mathcal{M}$.

The **monad laws** are graded/indexed versions of the usual three, and each is
operational content in Vx:

$$
\\mathrm{bind}(\\mathrm{return}(a), k) = k(a)
\\qquad (\\text{left identity: } \\mathbf{1}\\cdot v = v)
$$
$$
\\mathrm{bind}(m, \\mathrm{return}) = m
\\qquad (\\text{right identity: } w\\cdot\\mathbf{1} = w)
$$
$$
\\mathrm{bind}(\\mathrm{bind}(m, k), h) = \\mathrm{bind}(m, \\lambda a.\\, \\mathrm{bind}(k(a), h))
\\qquad (\\text{associativity: } (w\\cdot v)\\cdot u = w\\cdot(v\\cdot u))
$$

Associativity is *why a multi-hop transfer may discharge its obligation hop by
hop*: the grade (and consistency effect, §5) of the whole path is the monoid
product of the parts, independent of association. `spawn on(D){…}` is a **Kleisli
arrow** of this monad, and the seam-obligation engine is its **coherence
condition** (§6). This is the design's load-bearing claim: the informal index
tracking already in the checker (`active_topology`, the access check at a use
site) *is* an indexed monad, once `spawn` returns a located result (§4).

> **Placement vs. acquisition — three distinct layers.** This document is about the
> monad of *location* only. Two other layers must not be conflated with it:
> `Verified<T>` is a separate *return-type contract* (statically-proven `assert`s),
> unrelated to placement; and fallible hardware *acquisition* (`try_pin` →
> `HardwareState<T, τ>`, a unit may be saturated) is a *planned, not-yet-implemented*
> orthogonal layer (see [`formal_semantics.md` §2.4](formal_semantics.md)). `spawn`
> itself types to `Pinned<T, τ>` (T-SPAWN); placement soundness (below) is
> independent of both other layers.

## 4. The placement judgment

Write $\\Gamma\\, ;\\, q \\vdash e : A @ p$ — "in context $\\Gamma$, running on
topology $q$, expression $e$ has type $A$ located on $p$." The load-bearing rules:

**Var** — a variable records its location; nothing is checked yet.

$$
\\frac{}{\\Gamma, x : A @ p \\, ;\\, q \\vdash x : A @ p} \\quad (\\text{VAR})
$$

**Use (direct)** — to *read* a $p$-located value while running on $q$, the
location must be **directly visible** from $q$: a cost-0, no-copy relation the
topology declares (unified/coherent memory). The value stays $@p$.

$$
\\frac{\\Gamma\\, ;\\, q \\vdash e : A @ p \\qquad \\mathrm{Visible}(q, p)}
{\\Gamma\\, ;\\, q \\vdash \\mathrm{read}(e) : A @ p} \\quad (\\text{USE-DIRECT})
$$

**Use (needs seam)** — if $p$ is not visible from $q$, USE-DIRECT does not fire.
Under Vx's **explicit-seam** policy the checker does *not* insert a transfer; it
raises a repair-carrying error. $\\mathrm{Reachable}(p, q)$ means a composite
morphism $p \\to q$ exists in $\\mathcal{M}$.

$$
\\frac{\\Gamma\\, ;\\, q \\vdash e : A @ p \\qquad \\lnot\\mathrm{Visible}(q, p) \\qquad \\mathrm{Reachable}(p, q)}
{\\Gamma\\, ;\\, q \\vdash \\mathrm{read}(e) : \\bot \\;\\;[\\text{error: insert } \\mathrm{transfer}(e, q)]} \\quad (\\text{USE-NEEDS-SEAM})
$$

**Spawn** — an index switch whose result is *located* on the target:

$$
\\frac{\\Gamma\\, ;\\, d \\vdash \\mathit{body} : B @ d}
{\\Gamma\\, ;\\, q \\vdash \\mathrm{spawn\\ on}(d)\\{\\ \\mathit{body}\\ \\} : \\mathrm{Pinned}\\langle B, d\\rangle} \\quad (\\text{SPAWN})
$$

The conclusion is $\\mathrm{Pinned}\\langle B, d\\rangle$, **not** a bare $B$.
Returning the located type is what makes reading the result back on $q$
re-trigger USE (hence a visibility check or an explicit transfer). Stripping the
location — silently treating a device kernel's result as host-accessible — is the
classic soundness gap; keeping it is what "makes the monad honest."

**Transfer** — the graded generator; a Kleisli arrow guarded by its obligation:

$$
\\frac{\\Gamma\\, ;\\, q \\vdash x : A @ p \\qquad (p \\to p') \\in \\mathcal{M} \\qquad \\vdash \\mathrm{seam\\\_obligation}(p \\to p')}
{\\Gamma\\, ;\\, q \\vdash \\mathrm{transfer}(x, p') : A @ p'} \\quad (\\text{TRANSFER})
$$

The judgment form $A @ p$ and the "cannot consume a remote value in place" rule
are the modal-logic reading of placement (§7): $@p$ is a modality, USE is its
elimination, and USE-NEEDS-SEAM is the standard restriction that a value
marshalled to another world keeps its home index until an explicit hop rebinds
it.

## 5. Consistency: a second grade component

Memory-ordering strength is a **second grade**. Let $\\mathbb{C}$ be a lattice of
consistency effects — at minimum $\\mathrm{Sync} \\sqsubseteq \\mathrm{Relaxed}$
(more room for the paper's scoped-RC11 grades later). The full grade is the
product monoid $\\mathbb{W} \\times \\mathbb{C}$: a path's cost is the sum of its
hops' costs and its consistency is the meet of its hops' consistencies. A
`_relaxed` transfer sends published buffers to "possibly-stale," and the seam
obligation (`src/hir/seam.rs`) is the proof that the consumer's contract survives
that effect. Consistency is not needed for *compositional* soundness of placement,
so it is a refinement of the grade, not a change to the monad.

The **observation** side — reading a buffer back and asking what is guaranteed
visible — is the *comonadic dual* of transfer, and is where a Galois connection
between placement and read-back would live. It is out of scope for compositional
soundness and is noted here only to mark the boundary.

## 6. Coherence as mechanically-checkable laws

A user-declared topology (`Topology <Name> { memory / visible / transfer … }`) is
**admitted iff** it satisfies coherence obligations, which are exactly the
monad/functor laws made checkable:

- $\\mathrm{default\\\_space} \\in \\mathrm{spaces}$ — the object is inhabited;
- every op the topology claims to support has a lowering — the functor to
  hardware (`VxHardwarePlugin`) is total on the claimed domain;
- declared consistencies form a lattice — $\\mathbb{C}$ is well-formed;
- **dependent-topology index equality**: $\\mathrm{NPU}[i] \\equiv \\mathrm{NPU}[j]$
  iff $i = j$, an equation over a runtime index term.

All of these, together with the per-seam obligation of §5, are discharged by the
**same** `z3` engine (QF_BV / linear-integer queries) — one prover, three uses:
seam consistency, topology well-formedness, and dependent-index equality. The
categorical laws are thereby not decorative; failing one is a compile error
(`E6005` for coherence, `E6004` for a seam, `W1025`–`W1027` for the softer
checks).

## 7. Soundness (informal)

The intended metatheorem, in progress/preservation shape:

> **Placement safety.** If $\\varnothing\\, ;\\, q \\vdash e : A @ p$ and every
> $\\mathrm{transfer}$ in $e$ discharged its seam obligation, then evaluation of
> $e$ never reads a value from a space $p'$ that is neither visible from nor
> reachable-and-transferred-to the topology it is read on. No silent cross-space
> or stale read occurs.

The two ingredients are (i) USE only fires under $\\mathrm{Visible}$, and (ii)
SPAWN returns a located type so an off-topology result cannot be read without
re-entering USE — i.e. the monad's index is never dropped. A full mechanised
proof is future work; the structure above is what makes it tractable, because
each case reduces to a monad law plus a discharged obligation.

## 8. Relationship to established foundations

Vx's design is a systems instantiation of several well-understood lines of work.
Situating it precisely is both intellectual honesty and the source of its
credibility.

- **Parameterised (indexed) monads.** The $M[P, Q, A]$ shape — pre-/post-indices
  that must meet under `bind` — is Atkey's *parameterised notion of computation*
  (Atkey, 2009). Our indices are memory spaces.
- **Graded / parametric effect monads.** The grade $w$ from an ordered monoid is
  Katsumata's *parametric effect monad* (POPL 2014); our grade is transfer cost
  (and consistency).
- **The combination.** A monad that is *both* parameterised *and* graded is the
  category-graded monad of Orchard, Wadler, Eades — *Unifying graded and
  parameterised monads* (2020). $M[P, Q, w, A]$ is an instance; the novelty in Vx
  is the *object category* (an open, SMT-admitted set of hardware topologies),
  not the monad shape.
- **Modal / located types.** The judgment $A @ p$ and the "a value keeps its home
  index until an explicit hop" rule are the modal type theory of ML5 (Murphy,
  Crary, Harper — *Type-safe distributed programming with ML5*, 2008; *Modal
  types for mobile code*): $A @ w$ under a modal (S5-style) discipline.
- **Place / locale / region abstractions.** `spawn on(d)` is X10's `at(p) S`
  place-shift and Chapel's `on` statement; the located-data lineage includes
  X10's dependent *place types*, Legion/Regent *regions*, and Sequoia's explicit
  memory hierarchy. These fix a *closed* set of places; Vx's departure is the
  open, user-declared topology admitted by coherence.
- **Graded modal types.** The pairing of a modality with a grade is the setting of
  Granule (Orchard, Liepelt, Eades), which our (place-modality × cost-grade)
  structure resembles.

What Vx contributes on top of this foundation — an *open* category of topologies
whose admission is a mechanically checked coherence condition, dependent
topologies indexed by runtime values, and a single SMT engine discharging seam,
coherence, and index-equality obligations — is discussed in the design record and
its metatheory notes.

## References

- R. Atkey. *Parameterised Notions of Computation.* JFP, 2009.
- S. Katsumata. *Parametric Effect Monads and Semantics of Effect Systems.*
  POPL 2014.
- D. Orchard, P. Wadler, H. Eades III. *Unifying Graded and Parameterised
  Monads.* 2020 (arXiv:2001.10274).
- T. Murphy VII, K. Crary, R. Harper. *Type-Safe Distributed Programming with
  ML5.* TGC 2008; *Modal Types for Mobile Code* (thesis, CMU, 2008).
- P. Charles et al. *X10: An Object-Oriented Approach to Non-Uniform Cluster
  Computing.* OOPSLA 2005 (places, `at`).
- B. Chamberlain et al. *Parallel Programmability and the Chapel Language.* 2007
  (locales, `on`).
- K. Fatahalian et al. *Sequoia: Programming the Memory Hierarchy.* SC 2006.
- D. Orchard, V.-B. Liepelt, H. Eades III. *Quantitative Program Reasoning with
  Graded Modal Types* (Granule). ICFP 2019.
