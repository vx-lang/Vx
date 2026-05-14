# Denotational Semantics of Akar

This document outlines the denotational semantics of the Akar language. The primary objective is to formalize the mapping of Akar's syntax—specifically heterogeneous features like `Topology`, `Memory`, `Verified`, and `Pinned`—to strict mathematical domains.

## 1. Semantic Domains

Akar requires a dual-domain structure to distinguish between the Host (CPU) memory space and discrete Hardware Topologies (NPUs, GPUs).

### 1.1 Memory and Location Domains
- **$Loc$**: A countably infinite set of memory locations.
- **$\mathbb{M}_{Host}$**: The set of Host memory locations ($Loc_{Host}$).
- **$\mathbb{M}_{Topo}$**: A disjoint set of Hardware memory locations ($Loc_{Topo}$).
- **$\mathbb{M}$**: The universal memory space such that $\mathbb{M} = \mathbb{M}_{Host} \cup \mathbb{M}_{Topo}$.

### 1.2 Topology Domain
- **$\mathbb{T}$**: The set of discrete hardware topologies.
  - $\mathbb{T} = \{ NPU[0], NPU[1], ..., AccCore[0], ... \} \cup \{ Host \}$

### 1.3 Value Domains
Values in Akar include primitive numbers, matrices (Tensors), and specialized hardware-bound states.

- **$\mathbb{N}, \mathbb{R}$**: Standard numerical types (`i64`, `f64`, `bf16`).
- **$\mathbb{V}_{Tensor}$**: A multi-dimensional array mapping indices to numerical types.
  - $\mathbb{V}_{Tensor} = (\mathbb{N} \times \dots \times \mathbb{N}) \to \mathbb{R}$
- **$\mathbb{V}_{Ref}$**: A typed pointer bounding a value to a memory space.
  - $\mathbb{V}_{Ref} = \mathbb{M} \times Type$
- **$\mathbb{V}$**: The domain of all evaluated values: $\mathbb{V} = \mathbb{R} \cup \mathbb{V}_{Tensor} \cup \mathbb{V}_{Ref}$

### 1.4 State Monads
Akar's correctness centers around the hardware execution monad. We formalize this using the `HardwareState` enum and the `Verified` mathematical wrapper.

- **$\mathcal{V}erified[\mathbb{V}]$**: Represents a value computation whose type constraints have been mathematically proven, but whose hardware execution bounds (Topology) have not yet been localized.
- **$\mathcal{P}inned[\mathbb{V}, \mathbb{T}]$**: Represents a verified computation strictly bound to execute on a specific topology $t \in \mathbb{T}$.

The hardware state mathematically bridges these two:
- **$\mathcal{H}state[\mathbb{V}, \mathbb{T}] = \mathcal{P}inned[\mathbb{V}, \mathbb{T}] \oplus \mathcal{V}erified[\mathbb{V}]$**
  - Left mapping ($InL$): Hardware available -> $\mathcal{P}inned$
  - Right mapping ($InR$): Hardware saturated -> $\mathcal{V}erified$

## 2. Environment and Store

- **Environment ($Env$)**: Maps variable identifiers ($Id$) to their evaluated values or locations.
  - $\rho \in Env : Id \to \mathbb{V} \cup Loc$
- **Store ($Store$)**: Maps locations to values. Because Akar has discrete memory topologies, the store is partitioned.
  - $\sigma \in Store : Loc \to \mathbb{V}$
  - $\sigma = \sigma_{Host} \cup \sigma_{Topo}$

## 3. Valuation Functions

The denotation of an expression is a function mapping the current environment and store to a value and an updated store (handling side effects).

$\mathcal{E} \llbracket e \rrbracket : Env \to Store \to (\mathbb{V} \times Store)_\bot$

### 3.1 Constants and Variables
- $\mathcal{E} \llbracket n \rrbracket \rho \sigma = \langle n, \sigma \rangle$
- $\mathcal{E} \llbracket x \rrbracket \rho \sigma = \langle \sigma(\rho(x)), \sigma \rangle$ if $\rho(x) \in Loc$

### 3.2 Tensor Allocation
- $\mathcal{E} \llbracket \text{Tensor}(d_1, d_2) \rrbracket \rho \sigma_{Host} =$
  Let $v \in \mathbb{V}_{Tensor}$ be a zero-initialized matrix of dimension $d_1 \times d_2$.
  Let $l \in Loc_{Host}$ be a fresh memory location.
  $\langle l, \sigma_{Host}[l \mapsto v] \rangle$

### 3.3 Topology Spawn
The `spawn on` construct transitions the evaluation context from $\sigma_{Host}$ to $\sigma_{Topo}$.

Let $t \in \mathbb{T}$ be a target topology.
$\mathcal{E} \llbracket \text{spawn on}(t) \{ S \} \rrbracket \rho \sigma =$
  Let $\sigma' = \mathcal{C} \llbracket S \rrbracket \rho \sigma_{Topo}$.
  If computation is successful, returns $\mathcal{P}inned[\mathbb{V}, t]$.

### 3.4 Hardware Verification (Try Pin)
The denotation of evaluating a hardware state enforces fallback logic:

$\mathcal{E} \llbracket \text{match } \text{try\_pin}(c, t) \rrbracket \rho \sigma =$
  If Topology $t$ is available:
    Return $InL(\mathcal{P}inned[\mathcal{E}\llbracket c \rrbracket, t])$
  Else:
    Return $InR(\mathcal{V}erified[\mathcal{E}\llbracket c \rrbracket])$
