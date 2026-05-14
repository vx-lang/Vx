# Operational Semantics of Akar

This document details the Structural Operational Semantics (SOS) of the Akar programming language. SOS describes how an Akar program executes step-by-step as a sequence of state transitions.

## 1. The Execution State

Standard imperative languages define state as a tuple $\langle S, \sigma \rangle$ where $S$ is the statement to execute and $\sigma$ is the memory store.

Akar introduces heterogeneous computing, meaning the active execution state must track which topology is executing the instruction and maintain segmented stores for the Host CPU and discrete Accelerators.

We define the configuration state as:
$$\langle S, \sigma_{Host}, \tau_{Acc} \rangle_\Omega$$

Where:
- $S$: The statement or expression to be evaluated.
- $\sigma_{Host}$: The standard CPU memory store.
- $\tau_{Acc}$: The collective memory store of all active hardware accelerator topologies (e.g., NPU HBM, GPU VRAM).
- $\Omega \in \mathbb{T}$: The active execution context (the topology currently evaluating $S$).

### 1.1 Store Evaluation
If $\Omega = \text{Host}$, variables are evaluated against $\sigma_{Host}$.
If $\Omega = \text{Topo}[i]$, variables are evaluated against $\tau_{Acc}[\text{Topo}[i]]$.

## 2. Transition Rules

A transition is denoted as $\langle S, \sigma, \tau \rangle_\Omega \to \langle S', \sigma', \tau' \rangle_{\Omega'}$.

### 2.1 Sequential Execution (Host Context)
Standard sequential execution occurs when the active context is the Host:

$$
\frac{ \langle S_1, \sigma, \tau \rangle_{Host} \to \langle S_1', \sigma', \tau' \rangle_{Host} }{ \langle S_1 ; S_2, \sigma, \tau \rangle_{Host} \to \langle S_1' ; S_2, \sigma', \tau' \rangle_{Host} }
$$

$$
\frac{ }{ \langle \text{skip} ; S_2, \sigma, \tau \rangle_{Host} \to \langle S_2, \sigma, \tau \rangle_{Host} }
$$

### 2.2 Control Flow (`for` Loops)
Akar supports explicit bounded iteration. Let $[start, end)$ denote the iteration bounds.

$$
\frac{
  start < end \quad \langle S[start/i], \sigma, \tau \rangle_\Omega \to \langle S', \sigma', \tau' \rangle_\Omega
}{
  \langle \text{for } i \text{ in } start..end \{ S \}, \sigma, \tau \rangle_\Omega \to \langle \text{for } i \text{ in } (start+1)..end \{ S \}, \sigma', \tau' \rangle_\Omega
}
$$

$$
\frac{
  start \geq end
}{
  \langle \text{for } i \text{ in } start..end \{ S \}, \sigma, \tau \rangle_\Omega \to \langle \text{skip}, \sigma, \tau \rangle_\Omega
}
$$

### 2.3 The `spawn on` Construct
The `spawn on` construct evaluates a block of code strictly within the domain of a specific topology. This transitions the $\Omega$ context label.

Let $t \in \mathbb{T}$ be a valid topology identifier (e.g., `Topology::NPU[0]`).

$$
\frac{
  \text{HardwareAvailable}(t) \quad
  \langle S, \sigma, \tau \rangle_{t} \to^* \langle v, \sigma', \tau' \rangle_{t}
}{
  \langle \text{spawn on}(t) \{ S \}, \sigma, \tau \rangle_{Host} \to \langle \text{Available}(\text{Pinned}(v, t)), \sigma', \tau' \rangle_{Host}
} \text{ (Spawn-Success)}
$$

If the hardware is unavailable, the operational semantic rule evaluates the fallback state instead:

$$
\frac{
  \neg \text{HardwareAvailable}(t)
}{
  \langle \text{spawn on}(t) \{ S \}, \sigma, \tau \rangle_{Host} \to \langle \text{Saturated}(\text{Verified}(S)), \sigma, \tau \rangle_{Host}
} \text{ (Spawn-Fail)}
$$

### 2.4 Memory Space Tracking (`.with_memory`)
When a tensor is allocated in Host memory, applying `.with_memory(m)` casts it to a bounded reference.

$$
\frac{
  \langle E, \sigma, \tau \rangle_\Omega \to^* \langle l, \sigma, \tau \rangle_\Omega \quad l \in Loc
}{
  \langle E\text{.with\_memory}(m), \sigma, \tau \rangle_\Omega \to \langle \text{Ref}(l, m), \sigma, \tau \rangle_\Omega
}
$$

### 2.5 Memory Transfers (`transfer`)
Transfer operations shift values from one topological store to another.

$$
\frac{
  \rho(x) \mapsto l_{src} \in \tau_{Acc}[t_{src}] \quad
  l_{dst} \in \tau_{Acc}[t_{dst}] \text{ is fresh} \quad
  v = \tau_{Acc}[t_{src}](l_{src})
}{
  \langle \text{transfer}(x, \text{Memory::}t_{dst}), \sigma, \tau \rangle_{\Omega} \to \langle \text{Ref}(l_{dst}), \sigma, \tau[t_{dst} \mapsto \tau_{Acc}[t_{dst}][l_{dst} \mapsto v]] \rangle_{\Omega}
}
$$

### 2.6 Logical and Relational Operators
Akar supports boolean computation evaluating down to the $\mathbb{V}_{Tensor}^{\text{Bool}}$ domain (or scalar boolean equivalence).

For a relational binary operator $\oplus \in \{ <, >, \leq, \geq, ==, \neq \}$:
$$
\frac{
  \langle E_1, \sigma, \tau \rangle_\Omega \to^* \langle v_1, \sigma, \tau \rangle_\Omega \quad
  \langle E_2, \sigma, \tau \rangle_\Omega \to^* \langle v_2, \sigma, \tau \rangle_\Omega \quad
  v = v_1 \oplus v_2
}{
  \langle E_1 \oplus E_2, \sigma, \tau \rangle_\Omega \to \langle v, \sigma, \tau \rangle_\Omega
}
$$

### 2.7 Vectorized Assignment
Assignment natively checks the element type $\mathcal{T}$ of the Tensor to ensure soundness.

$$
\frac{
  \langle E, \sigma, \tau \rangle_\Omega \to^* \langle v, \sigma, \tau \rangle_\Omega \quad
  \text{TypeOf}(v) == \mathcal{T} \quad
  l = \text{LocOf}(x[\vec{i}])
}{
  \langle x[\vec{i}] = E, \sigma, \tau \rangle_\Omega \to \langle \text{skip}, \sigma[l \mapsto v], \tau \rangle_\Omega
}
$$
*(Note: store update applies to $\tau_{Acc}$ if $\Omega \neq \text{Host}$)*
