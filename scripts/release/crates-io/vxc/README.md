# vxc

Vx is a heterogeneous-systems programming language that puts placement and
reachability in the type system.

This crate reserves the name. The compiler is **not installable with
`cargo install`**: it links against LLVM 22 with MLIR, and optionally Enzyme
and z3, which Cargo cannot provide.

Install a released toolchain instead:

```sh
curl -fsSL https://vxlang.org/install.sh | sh
```

- Website: <https://vxlang.org>
- Documentation: <https://vxlang.org/docs/>
- Source: <https://github.com/vx-lang/Vx>
