//! Printing values and strings, and aborting with a message.
//!
//! One `impl` block on the per-function emitter; the dispatch that reaches these
//! lives in `step`, next to the struct.

use super::super::*;

impl FnEmit<'_> {
    // Print a value (no result). A tensor is `memref.cast`'d to an unranked memref and passed
    // to the `printMemref*` runtime helper; a scalar goes to `print_*`. These are the same
    // helpers the AST path calls; `emit_module_mlir` prepends their `private` declarations.
    pub(crate) fn op_print(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let arg = self
            .names
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        if let Some(memty) = self
            .mem_of
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone()
        {
            let et = memref_elem(&memty).ok_or(crate::emitter_gap!())?;
            let helper = match et {
                "f32" => "printMemrefF32",
                "f64" => "printMemrefF64",
                "i32" => "printMemrefI32",
                "i64" => "printMemrefI64",
                "bf16" => "printMemrefBF16",
                _ => return Err(crate::emitter_gap!()),
            };
            let c = format!("%pc{idx}");
            self.body += &format!("  {c} = memref.cast {arg} : {memty} to memref<*x{et}>\n");
            self.body += &format!("  func.call @{helper}({c}) : (memref<*x{et}>) -> ()\n");
        } else if let Some(e) = self.elem_at(ins.operand1.0) {
            let et = mlir_scalar(&e).ok_or(crate::emitter_gap!())?;
            // The narrow scalars have no print helper of their own, so widen to one that does.
            // An unsigned value is zero-extended into a signed width that holds all of it --
            // `u32` into `i64` rather than `i32`, since a `u32` above 2^31 does not fit there.
            // A signed narrow width sign-extends, which is the widening the AST path applies.
            let unsigned = matches!(
                e,
                ElementType::U8 | ElementType::U16 | ElementType::U32 | ElementType::U64
            );
            let (arg, et) = match (et, unsigned) {
                ("f16" | "bf16", _) => {
                    let w = format!("%pw{idx}");
                    self.body += &format!("  {w} = arith.extf {arg} : {et} to f32\n");
                    (w, "f32")
                }
                ("i32", true) => {
                    let w = format!("%pw{idx}");
                    self.body += &format!("  {w} = arith.extui {arg} : i32 to i64\n");
                    (w, "i64")
                }
                ("i8" | "i16", true) => {
                    let w = format!("%pw{idx}");
                    self.body += &format!("  {w} = arith.extui {arg} : {et} to i32\n");
                    (w, "i32")
                }
                ("i8" | "i16", false) => {
                    let w = format!("%pw{idx}");
                    self.body += &format!("  {w} = arith.extsi {arg} : {et} to i32\n");
                    (w, "i32")
                }
                _ => (arg, et),
            };
            // `u64` is the one width with nowhere wider to go, so it has a helper of its own.
            let helper = match et {
                "f32" => "print_f32",
                "f64" => "print_f64",
                "i32" => "print_i32",
                "i64" if unsigned && matches!(e, ElementType::U64) => "print_u64",
                "i64" => "print_i64",
                _ => {
                    return Err(Decline::TypeNotModelled {
                        what: "a print of an element type with no helper",
                    })
                }
            };
            let n = format!("%v{idx}");
            self.body += &format!("  {n} = func.call @{helper}({arg}) : ({et}) -> i32\n");
        } else if self
            .ptr_of
            .get(ins.operand1.0 as usize)
            .copied()
            .unwrap_or(false)
        {
            // A pointer value is a C string: the `print_str` helper, as for a literal.
            let n = format!("%v{idx}");
            self.body += &format!("  {n} = func.call @print_str({arg}) : (!llvm.ptr) -> i32\n");
        } else {
            return Err(crate::emitter_gap!());
        }
        Ok(())
    }

    pub(crate) fn op_abort(&mut self, _idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let cond = self
            .names
            .get(ins.operand1.0 as usize)
            .ok_or(crate::emitter_gap!())?
            .clone();
        let msg = self
            .strings
            .get(ins.imm as usize)
            .map(|s| s.as_str())
            .unwrap_or("assertion failed");
        let escaped = msg.replace('\\', "\\\\").replace('"', "\\\"");
        self.body += &format!("  cf.assert {cond}, \"{escaped}\"\n");
        Ok(())
    }

    pub(crate) fn op_print_str(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let n = self.str_base + ins.imm as usize;
        let p = format!("%pstrp{idx}");
        let r = format!("%pstr{idx}");
        self.body += &format!("  {p} = llvm.mlir.addressof @\".str.{n}\" : !llvm.ptr\n");
        self.body += &format!("  {r} = func.call @print_str({p}) : (!llvm.ptr) -> i32\n");
        Ok(())
    }

    // A string literal in value position (#231): take the address of the module-level global
    // (`@".str.<n>"`, `n = str_base + imm`, the same numbering as `PrintStr`) as a first-class
    // `!llvm.ptr` value — what a `let s = "…"` binds or a string argument passes. The global's
    // bytes are emitted by `emit_module_mlir` from the string side table.
    pub(crate) fn op_string_const(&mut self, idx: usize, ins: &HirInstruction) -> Lowered<()> {
        let n = self.str_base + ins.imm as usize;
        let p = format!("%v{idx}");
        self.body += &format!("  {p} = llvm.mlir.addressof @\".str.{n}\" : !llvm.ptr\n");
        self.names[idx] = p;
        self.ptr_of[idx] = true;
        Ok(())
    }
}
