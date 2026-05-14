# Formal Semantics of Akar Generics

Akar's generics are resolved completely statically via monomorphization. We define the semantics of generics as a source-to-source program transformation that occurs prior to runtime operational execution.

## 1. Type Substitution Environment

Let $\Gamma$ be the standard typing environment mapping variables to types.
We introduce $\Delta$, the generic type mapping environment, which maps generic parameter identifiers (e.g., $T$) to concrete Akar types.

A substitution operation $S[\Delta]$ takes an AST node $S$ and replaces all occurrences of $T$ where $T \in \text{dom}(\Delta)$ with $\Delta(T)$.

## 2. Type Unification Rule

When a generic function $f\langle T_1...T_n \rangle(p_1: \tau_1, ..., p_k: \tau_k)$ is invoked with arguments $a_1, ..., a_k$ of concrete types $\sigma_1, ..., \sigma_k$, the compiler performs unification.

We define the unification relation $\simeq$:
$$ \frac{ \Delta \vdash \tau_i \simeq \sigma_i \text{ for all } i \in [1, k] }{ \text{Unify}(\bar{\tau}, \bar{\sigma}) \Rightarrow \Delta } $$

Where individual type matching adheres to structural equivalence:
1. If $\tau_i = T$ (a generic parameter), and $T \notin \text{dom}(\Delta)$, then $\Delta := \Delta \cup \{T \mapsto \sigma_i\}$.
2. If $\tau_i = T$, and $T \in \text{dom}(\Delta)$, then $\Delta(T)$ must strictly equal $\sigma_i$.
3. If $\tau_i = \text{Tensor}\langle T \rangle$ and $\sigma_i = \text{Tensor}\langle \alpha \rangle$, unification proceeds structurally.

## 3. The Monomorphization Transition

Monomorphization behaves as a rewriting step ($\Rightarrow_M$) on the program $P$.

Let $E = f(a_1, ..., a_k)$ be a function call expression, where $f$ is generic.

$$ \frac{
  f \in P_{gen} \quad
  \text{Unify}(\text{params}(f), \text{types}(\bar{a})) \Rightarrow \Delta \quad
  f' = f[\Delta] \quad
  \text{name}(f') = \text{mangle}(\text{name}(f), \Delta)
}{
  \langle P \cup \{E\} \rangle \Rightarrow_M \langle P \cup \{f'\} \cup \{ \text{name}(f')(\bar{a}) \} \rangle
} $$

1. **Deduction:** The compiler extracts $\Delta$.
2. **Instantiation:** A new concrete function $f'$ is forged by applying $\Delta$ to the entire body and signature of $f$.
3. **AST Rewriting:** The generic call expression $E$ is mutated to directly call the mangled concretized function name.

## 4. Interaction with Operational Semantics

Because monomorphization $\Rightarrow_M$ is a total function evaluated completely during Semantic Analysis (compile-time), the runtime Structural Operational Semantics (SOS) described in `operational.md` requires **no modifications**.

The execution engine evaluates $\langle S, \sigma, \tau \rangle_\Omega$ strictly over the concretized program $P'$ where $P_{gen} = \emptyset$. All generic types have been successfully lowered to explicit topological memory allocations or standard host variables.
