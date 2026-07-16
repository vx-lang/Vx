# `stdlib/graph` — a small graph algorithms library in Vx

A multi-module Vx library implementing 10 classic graph algorithms over a dense
adjacency-matrix `Graph` (a weighted directed graph stored row-major in a flat
`Vec<i32>`; `adj[u*n + v]` is the edge weight, `0` = no edge). It doubles as a
realistic, multi-module workload for the parallel compiler.

## Modules

| File | Contents |
|---|---|
| `core.vx` | `Graph` type + constructors/accessors (`new`, `add_edge`, `add_undirected_edge`, `weight`, `has_edge`, `node_count`), and the `graph_inf()` sentinel |
| `traversal.vx` | (1) BFS reachability, (2) DFS reachability |
| `shortest_path.vx` | (3) Dijkstra, (4) Bellman-Ford, (5) Floyd-Warshall |
| `analysis.vx` | (6) connected components, (7) topological sort (Kahn / `is_dag`), (8) directed cycle detection, (9) Prim's MST weight, (10) transitive-closure reachability |
| `tests.vx` | Unit tests: builds small graphs with known answers and checks every algorithm via `assert(...)` |

`import graph::core;` resolves because `stdlib` is a module search root
(`src/module_loader.rs`), alongside `stdlib/std`.

## Building & running

Every module compiles, and `tests.vx` links across all five with `googletest`'s
`expect_eq`:

```
vxc --action emit-mlir stdlib/graph/shortest_path.vx   # any single module
vxc --action emit-mlir stdlib/graph/tests.vx           # whole multi-module program
```

Reaching this exercised two cross-module driver fixes:
[#203](https://github.com/hiraditya/Vx/issues/203) (transitive monomorphization —
a generic like `Vec<i32>::with_capacity` reached only from an imported algorithm
body) and [#204](https://github.com/hiraditya/Vx/issues/204) (imported *generic*
free functions like `expect_eq` were dropped).

**Runtime status:** `tests.vx` compiles but does **not** yet pass under
`--action run-jit` — the algorithms read the adjacency matrix through a `Vec`
field of `&Graph` inside loops, which hits codegen bug
[#205](https://github.com/hiraditya/Vx/issues/205) (a `Vec` field accessed via a
struct reference in a loop returns wrong values). The `expect_eq` checks are
correct and will pass once #205 is fixed; they currently *expose* it.

## Parallel-compiler / ThreadSanitizer workload

`tests/integration_test/graph_workload_test.rs` runs the parallel resolution
phases (`build_symbol_map` + `resolve_names`) over these modules and asserts the
output is invariant to thread count and stable under contention — the intended
ThreadSanitizer payload for [#202](https://github.com/hiraditya/Vx/issues/202)
(run those tests under `-Zsanitizer=thread`).
