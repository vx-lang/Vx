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

Every module compiles, and `tests.vx` links all five and runs all ten unit tests
via `googletest`'s `expect_eq`:

```
vxc --action emit-mlir stdlib/graph/shortest_path.vx   # any single module
vxc --action run-jit   stdlib/graph/tests.vx           # runs all 10 unit tests
```

`tests.vx` JIT-executes cleanly — every `expect_eq` passes. Getting here
exercised three compiler fixes: [#203](https://github.com/hiraditya/Vx/issues/203)
(cross-module *transitive* monomorphization — a generic like
`Vec<i32>::with_capacity` reached only from an imported algorithm body),
[#204](https://github.com/hiraditya/Vx/issues/204) (imported *generic* free
functions like `expect_eq` were dropped), and
[#205](https://github.com/hiraditya/Vx/issues/205) (a struct field assigned
through `&mut self` on a monomorphized generic — e.g. `Vec::push`'s
`self.len = self.len + 1` — was silently dropped, so the algorithms computed
wrong results while appearing to compile).

## Parallel-compiler / ThreadSanitizer workload

`tests/integration_test/graph_workload_test.rs` runs the parallel resolution
phases (`build_symbol_map` + `resolve_names`) over these modules and asserts the
output is invariant to thread count and stable under contention — the intended
ThreadSanitizer payload for [#202](https://github.com/hiraditya/Vx/issues/202)
(run those tests under `-Zsanitizer=thread`).
