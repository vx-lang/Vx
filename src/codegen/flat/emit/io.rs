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
                _ => return Err(crate::emitter_gap!()),
            };
            let c = format!("%pc{idx}");
            self.body += &format!("  {c} = memref.cast {arg} : {memty} to memref<*x{et}>\n");
            self.body += &format!("  func.call @{helper}({c}) : (memref<*x{et}>) -> ()\n");
        } else if let Some(e) = self.elem_at(ins.operand1.0) {
            let et = mlir_scalar(&e).ok_or(crate::emitter_gap!())?;
            let helper = match et {
                "f32" => "print_f32",
                "f64" => "print_f64",
                "i32" => "print_i32",
                "i64" => "print_i64",
                _ => {
                    return Err(Decline::TypeNotModelled {
                        what: "a print of an element type with no helper",
                    })
                }
            };
            let n = format!("%v{idx}");
            self.body += &format!("  {n} = func.call @{helper}({arg}) : ({et}) -> i32\n");
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
