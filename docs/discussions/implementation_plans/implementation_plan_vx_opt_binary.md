# Implementation Plan: `vx-opt` Binary and Custom Assembly Formats

## 1. Create `vx-opt` Rust Binary

We will create `src/bin/vx-opt.rs`, a lightweight Rust binary that forwards its CLI arguments directly to the C++ `run_vx_opt` function we implemented in the previous step. This officially creates the `vx-opt` standalone tool.

```rust
// src/bin/vx-opt.rs
extern "C" {
    fn run_vx_opt(argc: std::os::raw::c_int, argv: *const *const std::os::raw::c_char) -> std::os::raw::c_int;
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let c_args: Vec<std::ffi::CString> = args.into_iter().map(|a| std::ffi::CString::new(a).unwrap()).collect();
    let c_ptrs: Vec<*const std::os::raw::c_char> = c_args.iter().map(|a| a.as_ptr()).collect();
    let status = unsafe { run_vx_opt(c_ptrs.len() as std::os::raw::c_int, c_ptrs.as_ptr()) };
    std::process::exit(status);
}
```

## 2. Update Test Framework

We will update `tests/compile_test.rs` so that it parses both `// RUN: vxc %s` and `// RUN: vx-opt %s` lines.

- If it sees `vx-opt`, it will execute the `vx-opt` binary built by Cargo.
- We will update `vectorize.mlr`'s second RUN line to explicitly invoke `vx-opt`.

## 3. Custom MLIR Assembly Formats

Currently, `"vx.spawn"` is in quotes because it relies on the generic MLIR operation syntax. We will add custom declarative assembly formats to `include/VxDialect.td` for `vx.spawn` and `vx.yield`.

```tablegen
def Vx_SpawnOp : Vx_Op<"spawn"> {
    ...
    let assemblyFormat = "`topology` `(` $topology `)` $body attr-dict";
}

def Vx_YieldOp : Vx_Op<"yield", [Terminator]> {
    ...
    let assemblyFormat = "attr-dict";
}
```

This will allow us to write:

```mlir
  vx.spawn topology(0) {
    vx.yield
  }
```

instead of the generic `"vx.spawn"() <{topology = 0 : i32}> (...)` syntax.

## User Feedback

Does this address both of your observations? The standalone binary `vx-opt` will be available and explicitly tested in the `.mlr` files, and `vx.spawn` will have a beautifully formatted, unquoted syntax!
