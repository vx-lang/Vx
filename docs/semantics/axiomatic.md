# Axiomatic Semantics of Vx

Axiomatic semantics use assertions to mathematically prove that programs satisfy their specifications. Vx extends classical Hoare logic to reason about memory boundaries and hardware topology execution states.

## 1. Context-Aware Hoare Triples

A standard Hoare triple $\{ P \} \ S \ \{ Q \}$ states that if precondition $P$ holds before $S$ executes, then postcondition $Q$ will hold afterward.

Because Vx operates over multiple topologies, assertions must be context-aware. Let $\Omega$ represent the active execution topology context. We define the context-aware triple as:

$$ \{ P \} \ S \ \{ Q \}_\Omega $$

### 1.1 Memory Validity Assertions
We define $\text{Valid}(x, t)$ as an assertion that the reference $x$ is accessible from topology $t$.
- For $x : \text{Ref<Tensor, Memory::NPU\_HBM>}$, $\text{Valid}(x, \text{NPU})$ is True.
- For $x : \text{Ref<Tensor, Memory::Host>}$, $\text{Valid}(x, \text{NPU})$ is False.

## 2. Axiomatic Rules for Hardware State

Vx's `Verified` and `Pinned` types mathematically guarantee type safety, but the axiomatic rules define the spatial safety of hardware execution.

### 2.1 The Spawn Rule

The `spawn on` construct transitions the context $\Omega$ from the Host to the specified topology $t$.

$$
\frac{ \{ P \land \text{Valid}(\text{Env}, t) \} \ S \ \{ Q \}_t }{ \{ P \} \ \text{spawn on}(t) \{ S \} \ \{ Q \}_{Host} } \text{ (Spawn)}
$$

This rule states that if $S$ correctly establishes $Q$ when executed on topology $t$, and the environment guarantees memory validity for topology $t$, then spawning $S$ on $t$ establishes $Q$ from the perspective of the Host.

### 2.2 The Try-Pin Fallback Rule

Because hardware can fail or saturate, we rely on the `HardwareState` monad to branch control flow. The Hoare rule for matching the `HardwareState` enforces that the fallback logic perfectly simulates the intended mathematical operation.

Let $c$ be a computation bounded by $\text{Verified}<T>$, meaning $\{ P \} \ c \ \{ Q \}_{Math}$ is proven. Let $t \in \mathbb{T}$ be the target topology.

$$
\frac{
  \{ P \land \text{Available}(t) \} \ \text{try\_pin}(c, t) \ \{ Q \}_t \quad
  \{ P \land \neg \text{Available}(t) \} \ \text{fallback}(c) \ \{ Q \}_{Fallback}
}{
  \{ P \} \ \text{match try\_pin}(c, t) \{ \dots \} \ \{ Q \}_{Host}
} \text{ (HardwareState Match)}
$$

This rule rigorously enforces **Verified Computation**: No matter if the hardware is available or if execution falls back, the postcondition $Q$ representing the mathematical outcome of the tensor computation is identical.

## 3. Standard Axioms (Extended)

### 3.1 Assignment Rule (Topology Bound)

Tensor assignments are bounded by the topology context.

$$ \{ Q[E/x[\vec{i}]] \land \text{Valid}(x, \Omega) \} \ x[\vec{i}] = E \ \{ Q \}_\Omega $$

This enforces that $x$ cannot be assigned to unless it is valid in the currently executing hardware topology $\Omega$.

### 3.2 Memory Transfer Rule

The `transfer` keyword updates the memory bounds of a reference, making it valid for the destination topology and invalidating it for the source (unless explicitly copied).

$$
\frac{ \{ \text{Valid}(x, t_{src}) \} }{ \{ \text{True} \} \ \text{transfer}(x, \text{Memory::}t_{dst}) \ \{ \text{Valid}(x, t_{dst}) \land \neg \text{Valid}(x, t_{src}) \}_{Host} } \text{ (Transfer-Move)}
$$

### 3.3 For-Loop Rule (Topology Bound)

Because `for` loops in Vx execute fully within the current topology context, the invariant $I$ must hold within $\Omega$.

$$
\frac{ \{ I(i) \land \text{Valid}(\text{Env}, \Omega) \} \ S \ \{ I(i+1) \}_\Omega }{ \{ I(start) \} \ \text{for } i \text{ in } start..end \{ S \} \ \{ I(end) \}_\Omega } \text{ (For-Loop)}
$$
