# Cross-Module Function Calls & Code Generation

This plan addresses how Vx will compile and link multiple files when modules are imported, ensuring that `math.add()` correctly invokes the compiled machine code for `math.vx`'s `add` function without naming collisions.

## User Review Required

> [!WARNING]
> This plan introduces **Name Mangling**. When generating LLVM MLIR, functions in modules will no longer have clean names like `@add`. They will be prefixed (e.g., `@math_ak_add`). The `main.vx` module will retain un-mangled names for ease of execution.

## Open Questions

None at this time. The strategy aligns with standard C++/Rust compilation techniques.

## Proposed Changes

### Module Cache & AST Retention
Right now, `sema.rs` typechecks an imported module and then discards its AST. We need to retain these ASTs so they can be passed to the code generator.
- Introduce `module_asts: HashMap<String, Program>` in `TypeChecker` to cache fully type-checked imported modules.
- `TypeChecker::check_program` will return `Result<(Program, HashMap<String, Program>), Vec<String>>`.

### Name Mangling in Semantic Analysis
To prevent naming collisions (e.g. multiple modules defining `add`), `sema.rs` will perform namespace mangling on imported modules.
- Add `module_prefix: Option<String>` to `TypeChecker`.
- When creating a `sub_checker` for `math.vx`, set `module_prefix = Some("math_ak_".to_string())`.
- When `sub_checker` checks a `Function`, it renames `func.name` to `math_ak_add`.
- When `sub_checker` resolves a `FunctionCall("add")`, it resolves it to `math_ak_add` and updates the AST node.
- `Type::Module` will store the *mangled* name for each exported symbol, alongside its type.
- When `main.vx` processes `math.add()`, it will look up `add` in `Type::Module`, retrieve the mangled name `math_ak_add`, and emit `Expr::FunctionCall("math_ak_add")`.

#### [MODIFY] `src/ast.rs`
- Add mangled name support to `Type::Module` if necessary, or just rely on the `exports` map returning the mangled name string.

#### [MODIFY] `src/sema.rs`
- Add `module_asts: HashMap<String, Program>` to store the parsed modules.
- Add `module_prefix: Option<String>` to `TypeChecker`.
- Implement `mangle_name(name)` which prefixes the name if `module_prefix` is set.
- Update `check_program` to iterate and mangle function/struct names.
- Update `check_expr` for `FunctionCall` and `StructInit` to use mangled names for local module calls.
- Update `check_expr` for `Import` to cache the checked AST in `self.module_asts`.
- Update the return type of `check_program` to `Result<(Program, Vec<Program>), Vec<String>>`.

#### [MODIFY] `src/main.rs`
- Receive the tuple of `(main_program, modules)` from `sema.check_program`.
- Pass both `main_program` and `modules` into `MlirGenerator::generate`.

#### [MODIFY] `src/codegen.rs`
- Update `MlirGenerator::generate` to accept `modules: Vec<Program>`.
- Generate MLIR for all functions in all modules inside the same MLIR `module { }` block before generating `main_program`.

## Verification Plan

### Automated Tests
- Run `vxc tests/frontend/pass/modules_basic/main.vx --emit-mlir`. Verify that `@tests_frontend_pass_modules_basic_math_add` is generated and called correctly.
- Run `vxc tests/frontend/pass/modules_nested/main.vx --run` and verify that the JIT compiler successfully executes the cross-module code and prints the correct exit code or stdout.
- Execute `cargo test` to ensure no regressions in existing AST and parser logic.
