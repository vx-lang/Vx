# Operational Semantics of Vx

This document details the Structural Operational Semantics (SOS) of the Vx programming language. SOS describes how an Vx program executes step-by-step as a sequence of state transitions.

## 1. The Execution State

Standard imperative languages define state as a tuple $\\langle S, \\sigma \\rangle$ where $S$ is the statement to execute and $\\sigma$ is the memory store.

Vx introduces heterogeneous computing, meaning the active execution state must track which topology is executing the instruction and maintain segmented stores for the Host CPU and discrete Accelerators.

We define the configuration state as:
$$\\langle S, \\sigma\_{Host}, \\tau\_{Acc} \\rangle\_\\Omega$$

Where:

- $S$: The statement or expression to be evaluated.
- $\\sigma\_{Host}$: The standard CPU memory store.
- $\\tau\_{Acc}$: The collective memory store of all active hardware accelerator topologies (e.g., NPU HBM, GPU VRAM).
- $\\Omega \\in \\mathbb{T}$: The active execution context (the topology currently evaluating $S$).

### 1.1 Store Evaluation

If $\\Omega = \\text{Host}$, variables are evaluated against $\\sigma\_{Host}$.
If $\\Omega = \\text{Topo}[i]$, variables are evaluated against $\\tau\_{Acc}\[\\text{Topo}[i]\]$.

## 2. Transition Rules

A transition is denoted as $\\langle S, \\sigma, \\tau \\rangle\_\\Omega \\to \\langle S', \\sigma', \\tau' \\rangle\_{\\Omega'}$.

### 2.1 Sequential Execution (Host Context)

Standard sequential execution occurs when the active context is the Host:

$$
\\frac{ \\langle S_1, \\sigma, \\tau \\rangle\_{Host} \\to \\langle S_1', \\sigma', \\tau' \\rangle\_{Host} }{ \\langle S_1 ; S_2, \\sigma, \\tau \\rangle\_{Host} \\to \\langle S_1' ; S_2, \\sigma', \\tau' \\rangle\_{Host} }
$$

$$
\\frac{ }{ \\langle \\text{skip} ; S_2, \\sigma, \\tau \\rangle\_{Host} \\to \\langle S_2, \\sigma, \\tau \\rangle\_{Host} }
$$

### 2.2 Control Flow (`for` Loops)

Vx supports explicit bounded iteration. Let $\[start, end)$ denote the iteration bounds.

$$
\\frac{
start < end \\quad \\langle S[start/i], \\sigma, \\tau \\rangle\_\\Omega \\to \\langle S', \\sigma', \\tau' \\rangle\_\\Omega
}{
\\langle \\text{for } i \\text{ in } start..end { S }, \\sigma, \\tau \\rangle\_\\Omega \\to \\langle \\text{for } i \\text{ in } (start+1)..end { S }, \\sigma', \\tau' \\rangle\_\\Omega
}
$$

$$
\\frac{
start \\geq end
}{
\\langle \\text{for } i \\text{ in } start..end { S }, \\sigma, \\tau \\rangle\_\\Omega \\to \\langle \\text{skip}, \\sigma, \\tau \\rangle\_\\Omega
}
$$

### 2.3 The `spawn on` Construct

The `spawn on` construct evaluates a block of code strictly within the domain of a specific topology. This transitions the $\\Omega$ context label.

Let $t \\in \\mathbb{T}$ be a valid topology identifier (e.g., `Topology::NPU[0]`).

$$
\\frac{
\\text{HardwareAvailable}(t) \\quad
\\langle S, \\sigma, \\tau \\rangle\_{t} \\to^\* \\langle v, \\sigma', \\tau' \\rangle\_{t}
}{
\\langle \\text{spawn on}(t) { S }, \\sigma, \\tau \\rangle\_{Host} \\to \\langle \\text{Available}(\\text{Pinned}(v, t)), \\sigma', \\tau' \\rangle\_{Host}
} \\text{ (Spawn-Success)}
$$

If the hardware is unavailable, the operational semantic rule evaluates the fallback state instead:

$$
\\frac{
\\neg \\text{HardwareAvailable}(t)
}{
\\langle \\text{spawn on}(t) { S }, \\sigma, \\tau \\rangle\_{Host} \\to \\langle \\text{Saturated}(\\text{Verified}(S)), \\sigma, \\tau \\rangle\_{Host}
} \\text{ (Spawn-Fail)}
$$

### 2.4 Memory Space Tracking (`.with_memory`)

When a tensor is allocated in Host memory, applying `.with_memory(m)` casts it to a bounded reference.

$$
\\frac{
\\langle E, \\sigma, \\tau \\rangle\_\\Omega \\to^\* \\langle l, \\sigma, \\tau \\rangle\_\\Omega \\quad l \\in Loc
}{
\\langle E\\text{.with_memory}(m), \\sigma, \\tau \\rangle\_\\Omega \\to \\langle \\text{Ref}(l, m), \\sigma, \\tau \\rangle\_\\Omega
}
$$

### 2.5 Memory Transfers (`transfer`)

Transfer operations shift values from one topological store to another.

$$
\\frac{
\\rho(x) \\mapsto l\_{src} \\in \\tau\_{Acc}[t\_{src}] \\quad
l\_{dst} \\in \\tau\_{Acc}[t\_{dst}] \\text{ is fresh} \\quad
v = \\tau\_{Acc}[t\_{src}](l_%7Bsrc%7D)
}{
\\langle \\text{transfer}(x, \\text{Memory::}t\_{dst}), \\sigma, \\tau \\rangle\_{\\Omega} \\to \\langle \\text{Ref}(l\_{dst}), \\sigma, \\tau\[t\_{dst} \\mapsto \\tau\_{Acc}[t\_{dst}][l\_{dst} \\mapsto v]\] \\rangle\_{\\Omega}
}
$$

### 2.6 Logical and Relational Operators

Vx supports boolean computation evaluating down to the $\\mathbb{V}\_{Tensor}^{\\text{Bool}}$ domain (or scalar boolean equivalence).

For a relational binary operator $\\oplus \\in { \<, >, \\leq, \\geq, ==, \\neq }$:
$$
\\frac{
\\langle E_1, \\sigma, \\tau \\rangle\_\\Omega \\to^\* \\langle v_1, \\sigma, \\tau \\rangle\_\\Omega \\quad
\\langle E_2, \\sigma, \\tau \\rangle\_\\Omega \\to^\* \\langle v_2, \\sigma, \\tau \\rangle\_\\Omega \\quad
v = v_1 \\oplus v_2
}{
\\langle E_1 \\oplus E_2, \\sigma, \\tau \\rangle\_\\Omega \\to \\langle v, \\sigma, \\tau \\rangle\_\\Omega
}
$$

### 2.7 Vectorized Assignment

Assignment natively checks the element type $\\mathcal{T}$ of the Tensor to ensure soundness.

$$
\\frac{
\\langle E, \\sigma, \\tau \\rangle\_\\Omega \\to^\* \\langle v, \\sigma, \\tau \\rangle\_\\Omega \\quad
\\text{TypeOf}(v) == \\mathcal{T} \\quad
l = \\text{LocOf}(x[\\vec{i}])
}{
\\langle x[\\vec{i}] = E, \\sigma, \\tau \\rangle\_\\Omega \\to \\langle \\text{skip}, \\sigma[l \\mapsto v], \\tau \\rangle\_\\Omega
}
$$
*(Note: store update applies to $\\tau\_{Acc}$ if $\\Omega \\neq \\text{Host}$)*
