# Formal Operational Semantics

This document provides the formal mathematical specification of the Vx language semantics, targeting language designers and compiler engineers. We utilize natural deduction inference rules to specify the type system and small-step operational semantics for Vx's unique spatial and typestate constructs.

## 1. Syntax Domains

Let the following domains be defined:

- $T \\in \\text{Types}$ (e.g., `Tensor<f32>`, `i32`)
- $M \\in \\text{MemorySpaces}$ (e.g., `HostDRAM`, `NPUHBM`)
- $\\tau \\in \\text{Topologies}$ (e.g., `NPU[0]`, `AccCore`)
- $\\Gamma$: The typing context (environment mapping identifiers to types and linearity states)
- $\\Delta$: The spatial context (the hardware topology currently executing the expression)

## 2. Typing Judgments

A typing judgment in Vx takes the form:
$$ \\Gamma, \\Delta \\vdash e : T $$
"Under variable context $\\Gamma$ and executing on topology $\\Delta$, expression $e$ has type $T$."

### 2.1 Spatial Isolation and Memory References

References in Vx encode their physical location. The type `Ref<T, M>` means the value of type $T$ physically resides in memory space $M$.

$$
\\frac{
x : \\text{Ref}\\langle T, M \\rangle \\in \\Gamma \\quad \\text{Accessible}(M, \\Delta)
}{
\\Gamma, \\Delta \\vdash \*x : T
}
\\quad (\\text{T-DEREF})
$$

*Explanation:* A reference can only be safely dereferenced if its memory space $M$ is directly accessible from the current executing topology $\\Delta$.

### 2.2 The Transfer Primitive

Transferring data consumes linear references (if they own the data) and bridges memory spaces.

$$
\\frac{
\\Gamma, \\Delta \\vdash e : \\text{Ref}\\langle T, M\_{src} \\rangle \\quad M\_{tgt} \\in \\text{MemorySpaces}
}{
\\Gamma, \\Delta \\vdash \\text{transfer}(e, M\_{tgt}) : \\text{Ref}\\langle T, M\_{tgt} \\rangle
}
\\quad (\\text{T-TRANSFER})
$$

### 2.3 Spatial Spawning (`spawn on`)

Spawning routes a block of computation to a specific topology $\\tau$.

$$
\\frac{
\\Gamma \\vdash \\tau : \\text{Topology} \\quad \\Gamma, \\tau \\vdash B : T
}{
\\Gamma, \\Delta \\vdash \\text{spawn on}(\\tau) { B } : \\text{Verified}\\langle T \\rangle
}
\\quad (\\text{T-SPAWN})
$$

*Explanation:* The block $B$ is type-checked under the *target* topology context $\\tau$, rather than the current context $\\Delta$. The result of the `spawn` expression is a `Verified<T>` computation graph that has not yet been physically pinned.

### 2.4 Monadic Hardware Pinning (`try_pin`)

$$
\\frac{
\\Gamma, \\Delta \\vdash e : \\text{Verified}\\langle T \\rangle \\quad \\Gamma \\vdash \\tau : \\text{Topology}
}{
\\Gamma, \\Delta \\vdash e.\\text{try_pin}(\\tau) : \\text{HardwareState}\\langle T, \\tau \\rangle
}
\\quad (\\text{T-TRY-PIN})
$$

## 3. Operational Semantics (Small-Step)

We define the operational state as a configuration $\\langle e, \\mu, S \\rangle$, where:

- $e$ is the expression.
- $\\mu$ is the memory state (mapping locations to values).
- $S$ is the state of the hardware topology network (availability of units).

Transitions take the form $\\langle e, \\mu, S \\rangle \\longrightarrow \\langle e', \\mu', S' \\rangle$.

### 3.1 Evaluating `spawn on`

$$
\\frac{
\\text{Enqueue}(B, \\tau, S) = S'
}{
\\langle \\text{spawn on}(\\tau) { B }, \\mu, S \\rangle \\longrightarrow \\langle \\text{TaskHandle}(\\tau), \\mu, S' \\rangle
}
\\quad (\\text{E-SPAWN})
$$

### 3.2 Hardware Fallibility

The `try_pin` method attempts to acquire a lock on the physical hardware $\\tau$.

**Case 1: Hardware is available.**
$$
\\frac{
\\text{Available}(\\tau, S)
}{
\\langle \\text{Verified}(v).\\text{try_pin}(\\tau), \\mu, S \\rangle \\longrightarrow \\langle \\text{Available}(\\text{Pinned}(v, \\tau)), \\mu, \\text{Acquire}(\\tau, S) \\rangle
}
\\quad (\\text{E-PIN-SUCCESS})
$$

**Case 2: Hardware is saturated/unavailable.**
$$
\\frac{
\\neg \\text{Available}(\\tau, S)
}{
\\langle \\text{Verified}(v).\\text{try_pin}(\\tau), \\mu, S \\rangle \\longrightarrow \\langle \\text{Saturated}(\\text{Verified}(v)), \\mu, S \\rangle
}
\\quad (\\text{E-PIN-FAIL})
$$

## 4. Automatic Differentiation (AD)

Let $\\mathcal{J}$ be the Jacobian operator. Vx provides `grad`, `vjp`, and `jvp`.

$$
\\frac{
\\Gamma \\vdash f : \\text{Tensor}\\langle \\vec{x} \\rangle \\rightarrow \\text{Tensor}\\langle y \\rangle
}{
\\Gamma \\vdash \\text{grad}(f) : \\text{Tensor}\\langle \\vec{x} \\rangle \\rightarrow \\text{Tensor}\\langle \\vec{x} \\rangle
}
\\quad (\\text{T-GRAD})
$$

Operational reduction rule for `grad`:
$$
\\langle \\text{grad}(f)(x), \\mu, S \\rangle \\longrightarrow \\langle \\mathcal{J}\_f(x)^\\top \\cdot \\mathbf{1}, \\mu, S \\rangle
$$
*(Where $\\mathcal{J}\_f(x)^\\top$ represents the transpose of the Jacobian of $f$ evaluated at $x$, post-multiplied by a vector of ones, simulating reverse-mode AD).*
