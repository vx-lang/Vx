//===- generator.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Code generator module orchestrating the compilation pipeline from AST to MLIR to binary.
//
//===----------------------------------------------------------------------===//
use super::*;

use crate::codegen::lower::{LowerError, LowerToMelior};
use crate::syntax;

/// Extract the quoted identifier from an LLVM named-struct MLIR type string, e.g.
/// `!llvm.struct<"Closure_1", ()>` -> `Closure_1`. Returns `None` for anonymous structs or
/// non-struct types. Used to recognize closure-env vs nominal-closure structs during coercion.
fn parse_llvm_struct_name(ty_str: &str) -> Option<String> {
    let rest = ty_str.strip_prefix("!llvm.struct<\"")?;
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

pub struct MeliorGenerator<'c> {
    pub(crate) context: &'c Context,
    pub(crate) module: Module<'c>,
    pub(crate) env: HashMap<crate::symbol::Symbol, (Value<'c, 'c>, Type<'c>)>,
    pub(crate) ast_env: HashMap<crate::symbol::Symbol, syntax::Type>,
    /// Locals bound to a tensor the compiler allocated, as against a view over memory it does
    /// not control. Only these can be filled in place by `c = a @ b` (Vx#391).
    pub(crate) owned_tensors: std::collections::HashSet<crate::symbol::Symbol>,
    /// Declared memory spaces, keyed by space, so a `transfer` can emit the sub-space descriptor
    /// (granule/capacity/scope/parent) as IR metadata for later passes (see subspace_scheduling.md).
    pub(crate) memories: HashMap<syntax::MemorySpace, syntax::MemoryDecl>,
    /// Transfer lowerings by (from, to, topology) -- `impl Transfer<Memory::From,
    /// Memory::To> for Topology::X` -- the single method's body each. Sema records the
    /// key it chose on the `TransferExpr`; the body is inlined at the site in place of
    /// the builtin copy.
    ///
    /// The topology is in the key because the edge alone does not identify a lowering:
    /// an Ampere part and a Hopper part both declare `Memory::L2 -> Memory::SMEM` and
    /// move it with different instructions.
    pub(crate) transfer_impls: HashMap<(String, String, String), syntax::Function>,
    /// Declared topologies by name, so a `Pinned`/tensor value on a declared topology resolves the
    /// memory that topology actually names (`Topology SmemDev { memory: Memory::SMEM }`) when
    /// picking its address space, instead of guessing a like-named space (#258).
    pub(crate) topologies: HashMap<crate::symbol::Symbol, crate::arch::TopologyDescriptor>,
    /// Next free byte per granule'd sub-space — a bump allocator that assigns each tile a
    /// granule-rounded `offset` within its sub-space (SS2). Reset at each function boundary, since
    /// a sub-space (e.g. per-SM SMEM) is reused across kernels.
    pub(crate) subspace_offsets: HashMap<syntax::MemorySpace, u64>,
    pub(crate) structs: HashMap<crate::symbol::Symbol, syntax::StructDecl>,
    #[allow(clippy::type_complexity)]
    pub(crate) enums:
        HashMap<crate::symbol::Symbol, Vec<(crate::symbol::Symbol, Option<Vec<syntax::Type>>)>>,
    pub(crate) functions: HashMap<crate::symbol::Symbol, (Type<'c>, Vec<Type<'c>>)>,
    pub(crate) syntax_functions: HashMap<crate::symbol::Symbol, syntax::Function>,
    pub(crate) enzyme_decls: std::collections::HashSet<String>,
    pub string_counter: usize,
    pub current_return_type: Option<Type<'c>>,
    pub expected_type: Option<Type<'c>>,
    pub in_spawn: bool,
    pub break_blocks: Vec<*const melior::ir::Block<'c>>,
    pub continue_blocks: Vec<*const melior::ir::Block<'c>>,
    pub allocs: std::collections::HashSet<String>,
    pub(crate) is_lvalue_context: bool,
    pub(crate) mlir_block_counter: usize,
    pub(crate) has_returned: bool,
    pub(crate) current_filename: String,
    pub(crate) di_compile_unit: Option<melior::ir::Attribute<'c>>,
    pub(crate) di_subprogram: Option<melior::ir::Attribute<'c>>,
    pub current_span: syntax::types::Span,
    pub(crate) index_ty: Type<'c>,
    pub(crate) i32_ty: Type<'c>,
    pub(crate) i64_ty: Type<'c>,
    pub(crate) f32_ty: Type<'c>,
    pub(crate) f64_ty: Type<'c>,
    pub(crate) f16_ty: Type<'c>,
    pub(crate) bf16_ty: Type<'c>,
    pub(crate) i1_ty: Type<'c>,
    pub(crate) i4_ty: Type<'c>,
    pub(crate) i8_ty: Type<'c>,
    pub(crate) i16_ty: Type<'c>,
    pub(crate) i128_ty: Type<'c>,
    pub(crate) ptr_ty: Type<'c>,
    pub(crate) none_ty: Type<'c>,
    /// Whether to transport host-proven `assert` facts across a device `spawn` seam as
    /// `llvm.intr.assume` certificates (`vxc --emit-seam-certs`). Off by default so
    /// ordinary lowering is unchanged. See `crate::codegen::lower::seam_cert`.
    pub emit_seam_certs: bool,
    /// Host-proven facts (the conditions of `assert`s lowered so far in the current
    /// function) available for transport into a device kernel body. Reset per function.
    pub(crate) assert_facts: Vec<syntax::Expr>,
}

/// `memref<2x3xf32>` -> `memref<?x?xf32>`, keeping the rank and element type.
///
/// Anything carrying a layout or memory space is returned unchanged: there is no shape to erase
/// there without losing the rest of the type with it.
fn erase_memref_extents(s: &str) -> String {
    let Some(inner) = s.strip_prefix("memref<").and_then(|t| t.strip_suffix('>')) else {
        return s.to_string();
    };
    if inner.contains(',') {
        return s.to_string();
    }
    let parts: Vec<&str> = inner.split('x').collect();
    let Some((elem, dims)) = parts.split_last() else {
        return s.to_string();
    };
    if dims.is_empty() || !dims.iter().all(|d| *d == "?" || d.parse::<i64>().is_ok()) {
        return s.to_string();
    }
    let mut out = vec!["?"; dims.len()];
    out.push(elem);
    format!("memref<{}>", out.join("x"))
}

impl<'c> MeliorGenerator<'c> {
    pub fn loc(&self) -> melior::ir::Location<'c> {
        melior::ir::Location::new(
            self.context,
            &self.current_filename,
            self.current_span.line,
            self.current_span.column,
        )
    }

    pub fn is_memref(&self, ty: &Type<'c>) -> bool {
        let ty_str = ty.to_string();
        ty_str.starts_with("memref<")
    }

    pub fn is_llvm_ptr(&self, ty: &Type<'c>) -> bool {
        let ty_str = ty.to_string();
        ty_str.starts_with("!llvm.ptr") || ty_str.starts_with("!llvm.array")
    }

    /// Adapt a closure-literal environment struct (`Closure_N`, the by-value capture record) to a
    /// nominal stdlib closure struct (`Closure0/1/2/3<Args.., Ret>`, laid out `{env: *mut i8, func}`).
    /// A closure passed where an API takes a `ClosureK` (e.g. `VecIter::map`'s `f: Closure1<T,NewItem>`)
    /// reaches codegen as a `Closure_N` value; the two share the `{ptr, ptr}` shape but not the
    /// fields, so we materialize `{env: &spilled_env, func: &Closure_N_call}` here. Returns `None`
    /// when the pair isn't a `Closure_N` -> `ClosureK` adaptation, so the caller falls back to its
    /// normal coercion. See `check_closure_expr` (produces `Closure_N`) and closure.vx (`ClosureK`).
    pub(crate) fn adapt_closure_to_nominal(
        &mut self,
        block: &melior::ir::Block<'c>,
        env_val: Value<'c, 'c>,
        env_ty: Type<'c>,
        target_ty: Type<'c>,
    ) -> Result<Option<Value<'c, 'c>>, crate::codegen::lower::LowerError> {
        let env_str = env_ty.to_string();
        let target_str = target_ty.to_string();
        // Source must be a generated closure-env struct; target a nominal ClosureK (digit after
        // "Closure"). Both are `!llvm.struct<"NAME", (...)>`.
        let env_name = match parse_llvm_struct_name(&env_str) {
            Some(n) if n.starts_with("Closure_") => n,
            _ => return Ok(None),
        };
        let is_nominal_closure = parse_llvm_struct_name(&target_str)
            .map(|n| {
                n.starts_with("Closure")
                    && n[7..].chars().next().is_some_and(|c| c.is_ascii_digit())
            })
            .unwrap_or(false);
        if !is_nominal_closure {
            return Ok(None);
        }

        // The closure's call function: `Closure_N_call(_env, args..) -> ret`.
        let call_fn_name = format!("{}_call", env_name);
        let (ret_ty, arg_types) = self
            .functions
            .get(call_fn_name.as_str())
            .cloned()
            .ok_or_else(|| format!("closure call fn {} not found", call_fn_name))?;
        let fn_ty = melior::ir::r#type::FunctionType::new(self.context, &arg_types, &[ret_ty]);
        let const_op = melior::ir::operation::OperationBuilder::new("func.constant", self.loc())
            .add_attributes(&[(
                melior::ir::Identifier::new(self.context, "value"),
                melior::ir::attribute::FlatSymbolRefAttribute::new(self.context, &call_fn_name)
                    .into(),
            )])
            .add_results(&[fn_ty.into()])
            .build()?;
        let fn_val: Value = block.append_operation(const_op).result(0)?.into();
        // Function value -> opaque ptr for storage in the struct's `func` field.
        let ptr_ty = self.ptr_ty;
        let fn_ptr_op = melior::ir::operation::OperationBuilder::new(
            "builtin.unrealized_conversion_cast",
            self.loc(),
        )
        .add_operands(&[fn_val])
        .add_results(&[ptr_ty])
        .build()?;
        let fn_ptr: Value = block.append_operation(fn_ptr_op).result(0)?.into();

        // Spill the env struct to a stack slot and take its address (the captures live there).
        let i32_ty = self.i32_ty;
        let one_op = melior::ir::operation::OperationBuilder::new("llvm.mlir.constant", self.loc())
            .add_results(&[i32_ty])
            .add_attributes(&[(
                melior::ir::Identifier::new(self.context, "value"),
                melior::ir::attribute::IntegerAttribute::new(i32_ty, 1).into(),
            )])
            .build()?;
        let one: Value = block.append_operation(one_op).result(0)?.into();
        let alloca_op = melior::ir::operation::OperationBuilder::new("llvm.alloca", self.loc())
            .add_operands(&[one])
            .add_results(&[ptr_ty])
            .add_attributes(&[(
                melior::ir::Identifier::new(self.context, "elem_type"),
                melior::ir::attribute::TypeAttribute::new(env_ty).into(),
            )])
            .build()?;
        let env_ptr: Value = block.append_operation(alloca_op).result(0)?.into();
        block.append_operation(
            melior::ir::operation::OperationBuilder::new("llvm.store", self.loc())
                .add_operands(&[env_val, env_ptr])
                .build()?,
        );

        // Build the nominal `{env, func}` struct: field 0 = env ptr, field 1 = func ptr.
        let undef_op = melior::ir::operation::OperationBuilder::new("llvm.mlir.undef", self.loc())
            .add_results(&[target_ty])
            .build()?;
        let mut agg: Value = block.append_operation(undef_op).result(0)?.into();
        for (pos, field_val) in [env_ptr, fn_ptr].into_iter().enumerate() {
            let ins_op =
                melior::ir::operation::OperationBuilder::new("llvm.insertvalue", self.loc())
                    .add_operands(&[agg, field_val])
                    .add_attributes(&[(
                        melior::ir::Identifier::new(self.context, "position"),
                        melior::ir::attribute::DenseI64ArrayAttribute::new(
                            self.context,
                            &[pos as i64],
                        )
                        .into(),
                    )])
                    .add_results(&[target_ty])
                    .build()?;
            agg = block.append_operation(ins_op).result(0)?.into();
        }
        Ok(Some(agg))
    }

    pub fn coerce_type(
        &mut self,
        block: &melior::ir::Block<'c>,
        val: Value<'c, 'c>,
        from_ty: Type<'c>,
        to_ty: Type<'c>,
    ) -> Result<Value<'c, 'c>, crate::codegen::lower::LowerError> {
        if from_ty == to_ty {
            return Ok(val);
        }

        // Scalar -> tensor: broadcast the scalar across a fresh buffer (fill),
        // rather than emitting an invalid f32->memref bitcast. Handles static,
        // identity-layout memrefs whose element type matches the scalar. See #148.
        if self.is_memref(&to_ty) && !self.is_memref(&from_ty) {
            let to_str = to_ty.to_string();
            let el = to_str
                .trim_start_matches("memref<")
                .split('x')
                .next_back()
                .unwrap_or("")
                .trim_end_matches('>');
            if to_str.starts_with("memref<")
                && !to_str.contains('?')
                && !to_str.contains('*')
                && el == from_ty.to_string()
            {
                let alloc_op =
                    melior::ir::operation::OperationBuilder::new("memref.alloc", self.loc())
                        .add_attributes(&[(
                            melior::ir::Identifier::new(self.context, "operandSegmentSizes"),
                            melior::ir::attribute::DenseI32ArrayAttribute::new(
                                self.context,
                                &[0, 0],
                            )
                            .into(),
                        )])
                        .add_results(&[to_ty])
                        .build()
                        .unwrap();
                let dst: Value<'c, 'c> = block.append_operation(alloc_op).result(0).unwrap().into();

                let region = melior::ir::Region::new();
                let fblock =
                    melior::ir::Block::new(&[(from_ty, self.loc()), (from_ty, self.loc())]);
                let yield_op =
                    melior::ir::operation::OperationBuilder::new("linalg.yield", self.loc())
                        .add_operands(&[fblock.argument(0).unwrap().into()])
                        .build()
                        .unwrap();
                fblock.append_operation(yield_op);
                region.append_block(fblock);

                let fill_op =
                    melior::ir::operation::OperationBuilder::new("linalg.fill", self.loc())
                        .add_operands(&[val, dst])
                        .add_attributes(&[(
                            melior::ir::Identifier::new(self.context, "operandSegmentSizes"),
                            melior::ir::attribute::DenseI32ArrayAttribute::new(
                                self.context,
                                &[1, 1],
                            )
                            .into(),
                        )])
                        .add_regions([region])
                        .build()
                        .unwrap();
                block.append_operation(fill_op);
                return Ok(dst);
            }
        }

        if from_ty == self.i32_ty && to_ty == self.i64_ty {
            let cast_op = melior::ir::operation::OperationBuilder::new("arith.extsi", self.loc())
                .add_operands(&[val])
                .add_results(&[to_ty])
                .build()
                .unwrap();
            return Ok(block.append_operation(cast_op).result(0)?.into());
        }
        if from_ty == self.i64_ty && to_ty == self.i32_ty {
            let cast_op = melior::ir::operation::OperationBuilder::new("arith.trunci", self.loc())
                .add_operands(&[val])
                .add_results(&[to_ty])
                .build()
                .unwrap();
            return Ok(block.append_operation(cast_op).result(0)?.into());
        }
        if from_ty == self.f32_ty && to_ty == self.f64_ty {
            let cast_op = melior::ir::operation::OperationBuilder::new("arith.extf", self.loc())
                .add_operands(&[val])
                .add_results(&[to_ty])
                .build()
                .unwrap();
            return Ok(block.append_operation(cast_op).result(0)?.into());
        }
        if from_ty == self.f32_ty && (to_ty == self.bf16_ty || to_ty == self.f16_ty) {
            let cast_op = melior::ir::operation::OperationBuilder::new("arith.truncf", self.loc())
                .add_operands(&[val])
                .add_results(&[to_ty])
                .build()
                .unwrap();
            return Ok(block.append_operation(cast_op).result(0)?.into());
        }
        if (from_ty == self.bf16_ty || from_ty == self.f16_ty) && to_ty == self.f32_ty {
            let cast_op = melior::ir::operation::OperationBuilder::new("arith.extf", self.loc())
                .add_operands(&[val])
                .add_results(&[to_ty])
                .build()
                .unwrap();
            return Ok(block.append_operation(cast_op).result(0)?.into());
        }
        if (from_ty == self.f64_ty || from_ty == self.f32_ty)
            && (to_ty == self.f32_ty || to_ty == self.f16_ty || to_ty == self.bf16_ty)
        {
            let cast_op = melior::ir::operation::OperationBuilder::new("arith.truncf", self.loc())
                .add_operands(&[val])
                .add_results(&[to_ty])
                .build()
                .unwrap();
            return Ok(block.append_operation(cast_op).result(0)?.into());
        }

        if (from_ty == self.bf16_ty || from_ty == self.f16_ty)
            && (to_ty == self.f32_ty || to_ty == self.f64_ty)
        {
            let cast_op = melior::ir::operation::OperationBuilder::new("arith.extf", self.loc())
                .add_operands(&[val])
                .add_results(&[to_ty])
                .build()
                .unwrap();
            return Ok(block.append_operation(cast_op).result(0)?.into());
        }

        let is_float = |t: Type<'c>| {
            t == self.f32_ty || t == self.f64_ty || t == self.f16_ty || t == self.bf16_ty
        };

        if (from_ty == self.i32_ty || from_ty == self.i64_ty) && is_float(to_ty) {
            let cast_op = melior::ir::operation::OperationBuilder::new("arith.sitofp", self.loc())
                .add_operands(&[val])
                .add_results(&[to_ty])
                .build()
                .unwrap();
            return Ok(block.append_operation(cast_op).result(0)?.into());
        }

        if is_float(from_ty) && (to_ty == self.i32_ty || to_ty == self.i64_ty) {
            let cast_op = melior::ir::operation::OperationBuilder::new("arith.fptosi", self.loc())
                .add_operands(&[val])
                .add_results(&[to_ty])
                .build()
                .unwrap();
            return Ok(block.append_operation(cast_op).result(0)?.into());
        }

        // `index` has no direct fp conversion, so route through i64 (#186). This
        // is the loop-index-in-float-context path: `for i in 0..n { i as f32 }`,
        // where `i` is lowered as MLIR `index`. Without this the fallback emits a
        // bitcast (unrealized_conversion_cast index->f32) that survives LLVM
        // lowering as an unreconciled `i64 to index` cast and fails translation.
        if from_ty == self.index_ty && is_float(to_ty) {
            let to_i64 =
                melior::ir::operation::OperationBuilder::new("arith.index_cast", self.loc())
                    .add_operands(&[val])
                    .add_results(&[self.i64_ty])
                    .build()
                    .unwrap();
            let i64_val: Value<'c, 'c> = block.append_operation(to_i64).result(0)?.into();
            let cast_op = melior::ir::operation::OperationBuilder::new("arith.sitofp", self.loc())
                .add_operands(&[i64_val])
                .add_results(&[to_ty])
                .build()
                .unwrap();
            return Ok(block.append_operation(cast_op).result(0)?.into());
        }

        if is_float(from_ty) && to_ty == self.index_ty {
            let to_i64 = melior::ir::operation::OperationBuilder::new("arith.fptosi", self.loc())
                .add_operands(&[val])
                .add_results(&[self.i64_ty])
                .build()
                .unwrap();
            let i64_val: Value<'c, 'c> = block.append_operation(to_i64).result(0)?.into();
            let cast_op =
                melior::ir::operation::OperationBuilder::new("arith.index_cast", self.loc())
                    .add_operands(&[i64_val])
                    .add_results(&[to_ty])
                    .build()
                    .unwrap();
            return Ok(block.append_operation(cast_op).result(0)?.into());
        }

        let from_str = from_ty.to_string();
        let to_str = to_ty.to_string();

        if (from_ty == self.index_ty && (to_str.starts_with("i") || to_str.starts_with("u")))
            || ((from_str.starts_with("i") || from_str.starts_with("u")) && to_ty == self.index_ty)
        {
            let cast_op =
                melior::ir::operation::OperationBuilder::new("arith.index_cast", self.loc())
                    .add_operands(&[val])
                    .add_results(&[to_ty])
                    .build()
                    .unwrap();
            return Ok(block.append_operation(cast_op).result(0)?.into());
        }

        let is_int = |s: &str| s.starts_with("i") || s.starts_with("u");
        if is_int(&from_str) && is_int(&to_str) {
            let from_width: u32 = from_str[1..].parse().unwrap_or(0);
            let to_width: u32 = to_str[1..].parse().unwrap_or(0);
            if from_width > 0 && to_width > 0 {
                let op_name = if from_width > to_width {
                    "arith.trunci"
                } else if from_str.starts_with("u") {
                    "arith.extui"
                } else {
                    "arith.extsi"
                };
                let cast_op = melior::ir::operation::OperationBuilder::new(op_name, self.loc())
                    .add_operands(&[val])
                    .add_results(&[to_ty])
                    .build()
                    .unwrap();
                return Ok(block.append_operation(cast_op).result(0)?.into());
            }
        }

        if val.r#type() == to_ty {
            return Ok(val);
        }

        // A scalar-`memref` reference coerced to a bare `!llvm.ptr` — passing `&x` for a *mutable* scalar
        // local (a `memref<i32>` descriptor) to a `&i32` parameter (which lowers to `!llvm.ptr`). A raw
        // bitcast of the multi-word descriptor to a single pointer produces an `unrealized_conversion_cast`
        // that fails LLVM translation. Extract the memref's aligned data pointer instead, matching how a
        // nested `&mut r` is lowered (#278) — the resulting `!llvm.ptr` aliases the same storage. (#273)
        if to_ty == self.ptr_ty
            && from_str.starts_with("memref<")
            && !from_str.contains('x')
            && !from_str.starts_with("memref<memref<")
        {
            let raw_idx = block
                .append_operation(
                    melior::ir::operation::OperationBuilder::new(
                        "memref.extract_aligned_pointer_as_index",
                        self.loc(),
                    )
                    .add_operands(&[val])
                    .add_results(&[self.index_ty])
                    .build()
                    .unwrap(),
                )
                .result(0)?
                .into();
            let raw_i64 = block
                .append_operation(
                    melior::ir::operation::OperationBuilder::new("arith.index_cast", self.loc())
                        .add_operands(&[raw_idx])
                        .add_results(&[self.i64_ty])
                        .build()
                        .unwrap(),
                )
                .result(0)?
                .into();
            let ptr = block
                .append_operation(
                    melior::ir::operation::OperationBuilder::new("llvm.inttoptr", self.loc())
                        .add_operands(&[raw_i64])
                        .add_results(&[self.ptr_ty])
                        .build()
                        .unwrap(),
                )
                .result(0)?
                .into();
            return Ok(ptr);
        }

        if cfg!(debug_assertions) {
            println!(
                "Warning: Falling back to bitcast from {} to {}\nBacktrace:\n{:?}",
                from_str,
                to_str,
                std::backtrace::Backtrace::force_capture()
            );
        }
        // Default to unrealized_conversion_cast if nothing matches but we need a cast
        let cast_op = melior::ir::operation::OperationBuilder::new(
            "builtin.unrealized_conversion_cast",
            self.loc(),
        )
        .add_operands(&[val])
        .add_results(&[to_ty])
        .build()
        .unwrap();
        Ok(block.append_operation(cast_op).result(0)?.into())
    }

    pub fn new(context: &'c Context, filename: String) -> Self {
        let registry = DialectRegistry::new();
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();

        let location = Location::unknown(context);
        let module = Module::new(location);

        let index_ty = Type::parse(context, "index").unwrap();
        let i32_ty = Type::parse(context, "i32").unwrap();
        let i64_ty = Type::parse(context, "i64").unwrap();
        let f32_ty = Type::parse(context, "f32").unwrap();
        let f64_ty = Type::parse(context, "f64").unwrap();
        let f16_ty = Type::parse(context, "f16").unwrap();
        let bf16_ty = Type::parse(context, "bf16").unwrap();
        let i1_ty = Type::parse(context, "i1").unwrap();
        let i4_ty = Type::parse(context, "i4").unwrap();
        let i8_ty = Type::parse(context, "i8").unwrap();
        let i16_ty = Type::parse(context, "i16").unwrap();
        let i128_ty = Type::parse(context, "i128").unwrap();
        let ptr_ty = Type::parse(context, "!llvm.ptr").unwrap();
        let none_ty = Type::parse(context, "none").unwrap();

        Self {
            context,
            module,
            env: HashMap::new(),
            ast_env: HashMap::new(),
            owned_tensors: std::collections::HashSet::new(),
            memories: HashMap::new(),
            transfer_impls: HashMap::new(),
            topologies: HashMap::new(),
            subspace_offsets: HashMap::new(),
            structs: HashMap::new(),
            enums: HashMap::new(),
            functions: HashMap::new(),
            syntax_functions: HashMap::new(),
            enzyme_decls: std::collections::HashSet::new(),
            string_counter: 0,
            current_return_type: None,
            expected_type: None,
            in_spawn: false,
            break_blocks: Vec::new(),
            continue_blocks: Vec::new(),
            allocs: std::collections::HashSet::new(),
            is_lvalue_context: false,
            mlir_block_counter: 0,
            has_returned: false,
            current_filename: filename,
            di_compile_unit: None,
            di_subprogram: None,
            current_span: syntax::types::Span {
                line: 1,
                column: 1,
                length: 1,
            },
            index_ty,
            i32_ty,
            i64_ty,
            f32_ty,
            f64_ty,
            f16_ty,
            bf16_ty,
            i1_ty,
            i4_ty,
            i8_ty,
            i16_ty,
            i128_ty,
            ptr_ty,
            none_ty,
            emit_seam_certs: false,
            assert_facts: Vec::new(),
        }
    }

    pub fn into_module(self) -> Module<'c> {
        self.module
    }

    pub fn generate(
        &mut self,
        program: &Program,
        modules: &HashMap<crate::symbol::Symbol, Program>,
    ) -> Result<String, LowerError> {
        let location = self.loc();
        self.module = melior::ir::Module::new(location);

        let di_file_str = format!("#llvm.di_file<\"{}\" in \"\">", self.current_filename);
        let di_cu_str = format!(
            "#llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_C, file = {}, producer = \"vx\", isOptimized = false, emissionKind = Full>",
            di_file_str
        );
        self.di_compile_unit = melior::ir::Attribute::parse(self.context, &di_cu_str);

        // Add llvm.module_flags
        let module_flags_attr = melior::ir::Attribute::parse(
            self.context,
            "{\"Dwarf Version\" = 4 : i32, \"Debug Info Version\" = 3 : i32}",
        );
        if let Some(flags) = module_flags_attr {
            use melior::ir::operation::OperationMutLike;
            self.module
                .as_operation_mut()
                .set_attribute("llvm.module_flags", flags);
        }

        self.generate_module(program, modules)?;

        let op = self.module.as_operation();
        let s = format!("{}", op);
        Ok(s)
    }

    pub(crate) fn generate_module(
        &mut self,
        program: &Program,
        modules: &HashMap<crate::symbol::Symbol, Program>,
    ) -> Result<(), LowerError> {
        // Walked in name order rather than straight off `modules.values()`. A HashMap iterates in
        // an order that depends on the process's random hash seed, and the loops below decide the
        // order function bodies and extern declarations are emitted in -- so the same program
        // compiled twice came out with its functions in a different order.
        let sorted_modules: Vec<&Program> = {
            let mut names: Vec<&crate::symbol::Symbol> = modules.keys().collect();
            names.sort();
            names.into_iter().map(|n| &modules[n]).collect()
        };
        for s in &program.structs {
            self.structs.insert(s.name.clone(), s.clone());
        }
        for e in &program.enums {
            self.enums.insert(e.name.clone(), e.variants.clone());
        }
        for m in &program.memories {
            self.memories
                .insert(syntax::MemorySpace::from_name(m.name.as_ref()), m.clone());
        }
        for t in &program.topologies {
            self.topologies.insert(t.name.clone(), t.descriptor.clone());
        }
        for li in &program.transfer_impls {
            if li.methods.len() == 1 {
                self.transfer_impls.insert(
                    (
                        li.from.name(),
                        li.to.name(),
                        li.topology.display_name().to_string(),
                    ),
                    li.methods[0].clone(),
                );
            }
        }
        for module in sorted_modules.iter().copied() {
            for s in &module.structs {
                self.structs.insert(s.name.clone(), s.clone());
            }
            for e in &module.enums {
                self.enums.insert(e.name.clone(), e.variants.clone());
            }
            // Memories and topologies too. `--machine` loads the SKU as a peer
            // *module* (driver.rs), so taking these from the main program alone
            // meant a machine model contributed nothing here: a `vx.transfer`
            // came out carrying `space` and a cost but no `capacity` and no
            // `scope`, because the lookup that adds them found no declaration.
            //
            // Nothing failed. The attributes a device backend reads to place a
            // tile were simply absent, and a lowering keyed on `scope` did the
            // right thing for a program compiled on the flat path -- which does
            // merge them -- and quietly nothing for the same program on the AST
            // path.
            for m in &module.memories {
                self.memories
                    .insert(syntax::MemorySpace::from_name(m.name.as_ref()), m.clone());
            }
            for t in &module.topologies {
                self.topologies.insert(t.name.clone(), t.descriptor.clone());
            }
            for li in &module.transfer_impls {
                if li.methods.len() == 1 {
                    self.transfer_impls.insert(
                        (
                            li.from.name(),
                            li.to.name(),
                            li.topology.display_name().to_string(),
                        ),
                        li.methods[0].clone(),
                    );
                }
            }
        }
        for ext in &program.externs {
            let ret_ty = self.lower_type(&ext.return_type)?;
            let mut arg_tys = Vec::new();
            for (_, ty) in &ext.params {
                arg_tys.push(self.lower_type(ty)?);
            }
            self.functions.insert(ext.name.clone(), (ret_ty, arg_tys));
        }

        // Declare printMemref functions
        for ty_str in &["f32", "f64", "i32", "i64", "bf16"] {
            let func_name = format!("printMemref{}", ty_str.to_uppercase());
            let unranked_memref_ty =
                Type::parse(self.context, &format!("memref<*x{}>", ty_str)).unwrap();

            let func_ty = melior::ir::attribute::TypeAttribute::new(
                Type::parse(self.context, &format!("({unranked_memref_ty}) -> ()")).unwrap(),
            );

            let decl = melior::ir::operation::OperationBuilder::new("func.func", self.loc())
                .add_attributes(&[
                    (
                        melior::ir::Identifier::new(self.context, "sym_name"),
                        melior::ir::attribute::StringAttribute::new(self.context, &func_name)
                            .into(),
                    ),
                    (
                        melior::ir::Identifier::new(self.context, "function_type"),
                        func_ty.into(),
                    ),
                    (
                        melior::ir::Identifier::new(self.context, "sym_visibility"),
                        melior::ir::attribute::StringAttribute::new(self.context, "private").into(),
                    ),
                ])
                .add_regions([melior::ir::Region::new()])
                .build()?;

            self.module.body().append_operation(decl);
        }

        // Declare vx_init_signals
        let sig_init_ty = melior::ir::r#type::FunctionType::new(self.context, &[], &[]);
        let sig_init_decl = melior::ir::operation::OperationBuilder::new("func.func", self.loc())
            .add_attributes(&[
                (
                    melior::ir::Identifier::new(self.context, "sym_name"),
                    melior::ir::attribute::StringAttribute::new(self.context, "vx_init_signals")
                        .into(),
                ),
                (
                    melior::ir::Identifier::new(self.context, "function_type"),
                    melior::ir::attribute::TypeAttribute::new(sig_init_ty.into()).into(),
                ),
                (
                    melior::ir::Identifier::new(self.context, "sym_visibility"),
                    melior::ir::attribute::StringAttribute::new(self.context, "private").into(),
                ),
            ])
            .add_regions([melior::ir::Region::new()])
            .build()?;
        self.module.body().append_operation(sig_init_decl);

        for module_prog in sorted_modules.iter().copied() {
            for ext in &module_prog.externs {
                let ret_ty = self.lower_type(&ext.return_type)?;
                let mut arg_tys = Vec::new();
                for (_, ty) in &ext.params {
                    arg_tys.push(self.lower_type(ty)?);
                }
                self.functions.insert(ext.name.clone(), (ret_ty, arg_tys));
            }
        }

        let mut operations = Vec::new();

        for module_prog in sorted_modules.iter().copied() {
            // Skip generic *templates* — only their concrete instantiations (in `program.functions`,
            // collected as monomorphizations) are codegen'd. Imported modules now retain their
            // generic free functions so the env can instantiate cross-module generic calls (#204).
            for func in module_prog
                .functions
                .iter()
                .filter(|f| f.generics.is_empty())
            {
                let ret_ty = self.lower_type(&func.return_type)?;
                let mut arg_tys = Vec::new();
                for (_, ty) in &func.params {
                    arg_tys.push(self.lower_type(ty)?);
                }
                self.functions.insert(func.name.clone(), (ret_ty, arg_tys));
                self.syntax_functions
                    .insert(func.name.clone(), func.clone());
            }
        }

        for func in &program.functions {
            let ret_ty = self.lower_type(&func.return_type)?;
            let mut arg_tys = Vec::new();
            for (_, ty) in &func.params {
                arg_tys.push(self.lower_type(ty)?);
            }
            self.functions.insert(func.name.clone(), (ret_ty, arg_tys));
            self.syntax_functions
                .insert(func.name.clone(), func.clone());
        }

        // Emit module functions (concrete only; generic templates are skipped — see above).
        for module_prog in sorted_modules.iter().copied() {
            for func in module_prog
                .functions
                .iter()
                .filter(|f| f.generics.is_empty())
            {
                operations.push(self.generate_function(func)?);
            }
        }

        for func in &program.functions {
            operations.push(self.generate_function(func)?);
        }

        let body = self.module.body();

        let mut all_externs = program.externs.clone();
        for module_prog in sorted_modules.iter().copied() {
            all_externs.extend(module_prog.externs.clone());
        }

        let mut seen_externs = std::collections::HashSet::new();
        let mut unique_externs = Vec::new();
        for ext in all_externs {
            let sig = format!("{}: {:?}", ext.name, ext.params);
            if seen_externs.insert(sig) {
                unique_externs.push(ext);
            }
        }

        for ext in &unique_externs {
            let name = &ext.name;
            if name.as_ref() == "printf" || **name == *"vx_internal_printf" {
                continue;
            }
            let (ret_ty, arg_tys) = self.functions.get(name).ok_or_else(|| {
                crate::codegen::lower::LowerError::from(format!("Function not found: {}", name))
            })?;

            let mut actual_ret_tys = Vec::new();
            if ret_ty.to_string() != "none" {
                actual_ret_tys.push(*ret_ty);
            }

            // FunctionType::new takes arg_tys and ret_tys
            let func_type =
                melior::ir::r#type::FunctionType::new(self.context, arg_tys, &actual_ret_tys);

            // Define the string attribute for the function name
            let name_attr = melior::ir::attribute::StringAttribute::new(self.context, name);
            let type_attr = melior::ir::attribute::TypeAttribute::new(func_type.into());

            let builder = melior::ir::operation::OperationBuilder::new("func.func", self.loc())
                .add_attributes(&[
                    (
                        melior::ir::Identifier::new(self.context, "sym_name"),
                        name_attr.into(),
                    ),
                    (
                        melior::ir::Identifier::new(self.context, "function_type"),
                        type_attr.into(),
                    ),
                    (
                        melior::ir::Identifier::new(self.context, "sym_visibility"),
                        melior::ir::attribute::StringAttribute::new(self.context, "private").into(),
                    ),
                ]);

            let region = melior::ir::Region::new();
            let func_op = builder.add_regions([region]).build()?;

            body.append_operation(func_op);
        }

        for op in operations {
            body.append_operation(op);
        }
        Ok(())
    }

    pub(crate) fn generate_function(
        &mut self,
        func: &Function,
    ) -> Result<melior::ir::Operation<'c>, LowerError> {
        self.env.clear();
        self.allocs.clear();
        self.subspace_offsets.clear(); // per-function sub-space bump allocator (SS2)
        let is_main = func.name.as_ref() == "main";
        let true_ret_ty = self.lower_type(&func.return_type)?;
        let ret_ty = if is_main { self.i32_ty } else { true_ret_ty };

        let mut arg_tys = Vec::new();
        for (_, ty) in &func.params {
            arg_tys.push(self.lower_type(ty)?);
        }

        let mut actual_ret_tys = Vec::new();
        if ret_ty.to_string() != "none" {
            actual_ret_tys.push(ret_ty);
        }
        let func_type =
            melior::ir::r#type::FunctionType::new(self.context, &arg_tys, &actual_ret_tys);
        let name_attr = melior::ir::attribute::StringAttribute::new(self.context, &func.name);
        let type_attr = melior::ir::attribute::TypeAttribute::new(func_type.into());
        let di_file_str = format!("#llvm.di_file<\"{}\" in \"\">", self.current_filename);
        let di_cu_str = format!(
            "#llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_C, file = {}, producer = \"vx\", isOptimized = false, emissionKind = Full>",
            di_file_str
        );
        let di_subp_str = format!(
            "#llvm.di_subprogram<id = distinct[1]<>, compileUnit = {}, scope = {}, name = \"{}\", file = {}, subprogramFlags = Definition, type = #llvm.di_subroutine_type<>>",
            di_cu_str, di_file_str, func.name, di_file_str
        );
        self.di_subprogram = melior::ir::Attribute::parse(self.context, &di_subp_str);

        let mlir_str = format!(
            "module {{ func.func private @dummy() loc(fused<{}>[\"{}\":1:1]) }}",
            di_subp_str, self.current_filename
        );
        let dummy_module = melior::ir::Module::parse(self.context, &mlir_str).ok_or_else(|| {
            crate::codegen::lower::LowerError::from(format!("Failed to parse module: {}", mlir_str))
        })?;
        use melior::ir::operation::OperationLike;
        use melior::ir::BlockLike;
        let dummy_op = dummy_module.body().first_operation().ok_or_else(|| {
            crate::codegen::lower::LowerError::from("No first operation".to_string())
        })?;
        let func_loc = dummy_op.location();

        let _region = Region::new();

        let mut block_args = Vec::new();
        for ty in &arg_tys {
            block_args.push((*ty, self.loc()));
        }
        let region = melior::ir::Region::new();
        let mut current_block = region.append_block(Block::new(&block_args));

        // Map arguments into the environment
        for (i, (name, ast_ty)) in func.params.iter().enumerate() {
            let arg_val = current_block.argument(i)?.into();
            self.env
                .insert(name.to_string().into(), (arg_val, arg_tys[i]));
            self.ast_env.insert(name.to_string().into(), ast_ty.clone());
        }

        if is_main {
            // Call vx_init_signals
            let sig_init_call =
                melior::ir::operation::OperationBuilder::new("func.call", self.loc())
                    .add_attributes(&[(
                        melior::ir::Identifier::new(self.context, "callee"),
                        melior::ir::attribute::FlatSymbolRefAttribute::new(
                            self.context,
                            "vx_init_signals",
                        )
                        .into(),
                    )])
                    .add_results(&[])
                    .build()?;
            current_block.append_operation(sig_init_call);
        }

        self.current_return_type = Some(ret_ty);
        // Facts are scoped to the function being lowered so a proven `assert` cannot be
        // transported into a device kernel in an unrelated function (which could name the
        // same variable but bind a different, non-dominating SSA value).
        self.assert_facts.clear();
        let mut terminated = false;
        for stmt in &func.body {
            if let Some(b) = self.generate_statement(stmt, current_block)? {
                current_block = b;
            } else {
                terminated = true;
                break;
            }
        }

        // Append a fall-through terminator only when control actually reaches the
        // end of the body (no explicit `return`/`break` terminated it first).
        if !terminated {
            if is_main {
                // `main` is the C entry point and always lowers to `-> i32`. An
                // explicit `return <expr>` in main is honored like any other
                // function (see #210) -- its i32 value becomes the process exit
                // code -- so we only synthesize `return 0` when the body has no
                // trailing return (e.g. a main that just prints).
                let i32_ty = self.i32_ty;
                let c0_op = current_block.append_operation(
                    melior::ir::operation::OperationBuilder::new("arith.constant", self.loc())
                        .add_results(&[i32_ty])
                        .add_attributes(&[(
                            melior::ir::Identifier::new(self.context, "value"),
                            melior::ir::attribute::IntegerAttribute::new(i32_ty, 0).into(),
                        )])
                        .build()?,
                );
                let c0 = c0_op.result(0)?.into();
                current_block.append_operation(
                    melior::ir::operation::OperationBuilder::new("func.return", self.loc())
                        .add_operands(&[c0])
                        .build()?,
                );
            } else if let syntax::Type::Struct(name, _) = &func.return_type {
                if name.as_ref() == "void" {
                    current_block.append_operation(
                        melior::ir::operation::OperationBuilder::new("func.return", self.loc())
                            .build()?,
                    );
                }
            }
        }

        self.current_return_type = None;

        let mut func_attributes = vec![
            (
                melior::ir::Identifier::new(self.context, "sym_name"),
                name_attr.into(),
            ),
            (
                melior::ir::Identifier::new(self.context, "function_type"),
                type_attr.into(),
            ),
        ];

        if is_main {
            func_attributes.push((
                melior::ir::Identifier::new(self.context, "llvm.emit_c_interface"),
                melior::ir::attribute::Attribute::parse(self.context, "unit").ok_or_else(|| {
                    crate::codegen::lower::LowerError::from(format!(
                        "Failed to parse attribute: {}",
                        "unit"
                    ))
                })?,
            ));
        }

        let func_op = melior::ir::operation::OperationBuilder::new("func.func", func_loc)
            .add_attributes(&func_attributes)
            .add_regions([region])
            .build()?;

        Ok(func_op)
    }

    /// Emit the runtime check for `assert(cond)` / `assert(cond, "msg")` (Vx#361).
    ///
    /// `cf.assert` rather than a hand-rolled branch, because MLIR lowers it for whichever
    /// target the code reaches and both passes are already in the pipelines: on the host
    /// `convert-cf-to-llvm` expands it to a branch onto `puts` + `abort` + `unreachable`, and
    /// INSIDE A KERNEL `convert-gpu-to-nvvm` expands it to `__assertfail` with the message,
    /// file, line and a `noreturn` attribute. `cf` is in `isDeviceLowerableDialect`, so the op
    /// also passes the kernel's device-ready gate.
    ///
    /// That portability is why there is no kernel guard here. An earlier version skipped
    /// emission under `in_spawn`, reasoning that the op would lower to a call to the HOST's
    /// `abort` in PTX -- which was the host pipeline's behaviour applied to device code, and
    /// wrong. A kernel assertion is a real assertion (Vx#362).
    ///
    /// A condition the checker already folded to `true` is dropped rather than emitted: there
    /// is nothing to test at run time, and E8002 has already rejected the false ones.
    fn emit_runtime_assert(
        &mut self,
        s: &syntax::AssertStmt,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Result<Option<melior::ir::BlockRef<'c, 'c>>, LowerError> {
        // A literal `assert(true)` is not worth a branch. Anything less obvious is left to
        // the optimiser, which sees the same constant the checker did; a statically-FALSE
        // condition never reaches codegen at all (E8002 rejects it).
        if matches!(
            &*s.expr,
            syntax::Expr::Identifier(id) if id.name.as_ref() == "true"
        ) {
            return Ok(Some(block));
        }
        let (cond, _ty, block) = self.generate_expr(&s.expr, block)?;
        let msg = s
            .msg
            .clone()
            .unwrap_or_else(|| "assertion failed".to_string());
        let op = melior::ir::operation::OperationBuilder::new("cf.assert", self.loc())
            .add_operands(&[cond])
            .add_attributes(&[(
                melior::ir::Identifier::new(self.context, "msg"),
                melior::ir::attribute::StringAttribute::new(self.context, &msg).into(),
            )])
            .build()?;
        block.append_operation(op);
        Ok(Some(block))
    }

    pub(crate) fn generate_statement(
        &mut self,
        stmt: &Statement,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Result<Option<melior::ir::BlockRef<'c, 'c>>, LowerError> {
        self.current_span = stmt.span();
        match stmt {
            Statement::Return(s) => LowerToMelior::lower(s, self, block),
            Statement::LetDecl(s) => LowerToMelior::lower(s, self, block),
            Statement::Assign(s) => LowerToMelior::lower(s, self, block),
            Statement::CompoundAssign(s) => LowerToMelior::lower(s, self, block),
            Statement::ExprStmt(s) => LowerToMelior::lower(s, self, block),
            Statement::ForLoop(s) => LowerToMelior::lower(s, self, block),
            Statement::Assert(s) => {
                // The condition is also a fact the host proves; record it so it can be
                // transported across a device `spawn` seam as an `llvm.intr.assume`
                // certificate (see `crate::codegen::lower::seam_cert`).
                if self.emit_seam_certs {
                    self.assert_facts.push((*s.expr).clone());
                }
                self.emit_runtime_assert(s, block)
            }
            Statement::Loop(s) => LowerToMelior::lower(s, self, block),
            Statement::Break(s) => LowerToMelior::lower(s, self, block),
            Statement::Continue(s) => LowerToMelior::lower(s, self, block),
            Statement::MacroCall(_) => panic!("Macros should be expanded before codegen"),
            Statement::Error(_) => Ok(None),
        }
    }

    pub(crate) fn generate_expr(
        &mut self,
        expr: &Expr,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Result<(Value<'c, 'c>, Type<'c>, melior::ir::BlockRef<'c, 'c>), LowerError> {
        self.current_span = expr.span();
        match expr {
            Expr::Identifier(e) => LowerToMelior::lower(e, self, block),
            Expr::BinaryOp(e) => LowerToMelior::lower(e, self, block),
            Expr::RelationalOp(e) => LowerToMelior::lower(e, self, block),
            Expr::LogicalOp(e) => LowerToMelior::lower(e, self, block),
            Expr::UnaryOp(e) => LowerToMelior::lower(e, self, block),
            Expr::StructInit(e) => LowerToMelior::lower(e, self, block),
            Expr::MemberAccess(e) => LowerToMelior::lower(e, self, block),
            Expr::IndexAccess(e) => LowerToMelior::lower(e, self, block),
            // `abort()` -- terminate unconditionally, expressed as a conditional abort whose
            // condition is `false`. One form serves both `abort` and `assert`, and it inherits
            // the same target portability: `abort` + `unreachable` on the host, `__assertfail`
            // inside a kernel. Handled here rather than in the general call lowering because it
            // has no Vx-level definition to resolve -- it is a primitive, like `print`. Safe to
            // call; ending a process breaks no memory-safety property.
            Expr::FunctionCall(e) if e.name.as_ref() == "abort" && e.args.is_empty() => {
                let never =
                    melior::ir::operation::OperationBuilder::new("arith.constant", self.loc())
                        .add_attributes(&[(
                            melior::ir::Identifier::new(self.context, "value"),
                            melior::ir::attribute::IntegerAttribute::new(
                                melior::ir::r#type::IntegerType::new(self.context, 1).into(),
                                0,
                            )
                            .into(),
                        )])
                        .add_results(
                            &[melior::ir::r#type::IntegerType::new(self.context, 1).into()],
                        )
                        .build()?;
                let never_v = block.append_operation(never).result(0)?.into();
                let assert_op =
                    melior::ir::operation::OperationBuilder::new("cf.assert", self.loc())
                        .add_operands(&[never_v])
                        .add_attributes(&[(
                            melior::ir::Identifier::new(self.context, "msg"),
                            melior::ir::attribute::StringAttribute::new(
                                self.context,
                                "abort() called",
                            )
                            .into(),
                        )])
                        .build()?;
                block.append_operation(assert_op);
                let zero =
                    melior::ir::operation::OperationBuilder::new("arith.constant", self.loc())
                        .add_attributes(&[(
                            melior::ir::Identifier::new(self.context, "value"),
                            melior::ir::attribute::IntegerAttribute::new(self.i32_ty, 0).into(),
                        )])
                        .add_results(&[self.i32_ty])
                        .build()?;
                Ok((
                    block.append_operation(zero).result(0)?.into(),
                    self.i32_ty,
                    block,
                ))
            }
            Expr::FunctionCall(e) => LowerToMelior::lower(e, self, block),
            Expr::MethodCall(e) => LowerToMelior::lower(e, self, block),
            Expr::SpawnOn(e) => LowerToMelior::lower(e, self, block),
            Expr::Array(e) => LowerToMelior::lower(e, self, block),
            Expr::If(e) => LowerToMelior::lower(e, self, block),
            Expr::EnumVariant(e) => LowerToMelior::lower(e, self, block),
            Expr::Match(e) => LowerToMelior::lower(e, self, block),
            Expr::Number(e) => LowerToMelior::lower(e, self, block),
            Expr::UnsafeBlock(e) => LowerToMelior::lower(e, self, block),
            Expr::Grad(e) => LowerToMelior::lower(e, self, block),
            Expr::Vjp(e) => LowerToMelior::lower(e, self, block),
            Expr::Jvp(e) => LowerToMelior::lower(e, self, block),
            Expr::Transfer(e) => LowerToMelior::lower(e, self, block),
            Expr::Borrow(e) => LowerToMelior::lower(e, self, block),
            Expr::StringLiteral(e) => LowerToMelior::lower(e, self, block),
            Expr::Closure(e) => LowerToMelior::lower(e, self, block),
            Expr::ComptimeBlock(e) => LowerToMelior::lower(e, self, block),
            Expr::Dereference(e) => LowerToMelior::lower(e, self, block),
            Expr::AsCast(e) => LowerToMelior::lower(e, self, block),
            Expr::IndirectCall(e) => LowerToMelior::lower(e, self, block),
            Expr::Print(e) => LowerToMelior::lower(e, self, block),
            Expr::Println(e) => LowerToMelior::lower(e, self, block),
            Expr::InlineMlir(e) => LowerToMelior::lower(e, self, block),
            Expr::Topology(e) => LowerToMelior::lower(e, self, block),
            Expr::SizeOf(e) => LowerToMelior::lower(e, self, block),
            // Constructs with no runtime lowering: report a diagnostic instead of a
            // `todo!()` panic. (Match is exhaustive so a new Expr variant forces a decision.)
            Expr::Range(_) => Err(LowerError::from(
                "a range (`a..b`) is only valid as a `for`-loop bound, not as a value".to_string(),
            )),
            Expr::MemorySpace(_) => Err(LowerError::from(
                "a memory space (`Memory::...`) is not a runtime value".to_string(),
            )),
            Expr::VecMacro(_) => Err(LowerError::from(
                "`vec![...]` is not supported at this codegen position".to_string(),
            )),
            Expr::TransferPredicate(_) => Err(LowerError::from(
                "`Reachable<A, B>` is a comptime predicate; it may only appear as an \
                 `if comptime` condition"
                    .to_string(),
            )),
            Expr::MacroCall(_) => Err(LowerError::from(
                "internal error: macro call reached codegen (macros must be expanded first)"
                    .to_string(),
            )),
        }
    }

    /// Byte size + alignment of a type for `sizeof<T>()`, computed the way the layout pass does
    /// (`crate::layout::LayoutComputer::struct_layout`): a scalar by its element, any pointer/borrow
    /// 8, and a nominal struct (or a monomorphized generic instance of one) as the aligned sum of its
    /// recursively-sized fields. Name-keyed, because the generator holds struct decls by name, not
    /// GID. `None` for an unmodelled type (tensor, closure, generic scalar, data-carrying enum) — the
    /// caller falls back to a conservative default. This replaces `SizeOfExpr::lower`'s old hardcoded
    /// `8` for aggregates, which under-sized a `Vec<struct>`'s element buffer (a `Vec<Vec<T>>` UB). (#242)
    pub(crate) fn type_size_align(&self, ty: &syntax::Type, depth: u32) -> Option<(usize, usize)> {
        use crate::layout::{align_up, scalar_size_align};
        if depth > 32 {
            return None; // guard against a pathological/cyclic nesting
        }
        match ty {
            syntax::Type::Scalar(e) => scalar_size_align(e),
            syntax::Type::Pointer(..) | syntax::Type::Borrow { .. } | syntax::Type::Ref(..) => {
                Some((8, 8))
            }
            syntax::Type::Pinned(inner, _) | syntax::Type::Verified(inner) => {
                self.type_size_align(inner, depth)
            }
            syntax::Type::GenericInstance(base, _) => self.type_size_align(base, depth),
            syntax::Type::Struct(name, _) => {
                let base = name.as_ref().split('<').next().unwrap_or(name.as_ref());
                let decl = self.structs.get(base)?;
                let mut offset = 0usize;
                let mut align = 1usize;
                for (_, fty) in &decl.fields {
                    let (fsize, falign) = self.type_size_align(fty, depth + 1)?;
                    offset = align_up(offset, falign) + fsize;
                    align = align.max(falign);
                }
                Some((align_up(offset, align), align))
            }
            _ => None,
        }
    }

    pub(crate) fn lower_type(
        &self,
        ty: &syntax::Type,
    ) -> Result<Type<'c>, crate::codegen::lower::LowerError> {
        let ty_str = match ty {
            syntax::Type::Tensor(el_ty, dims, top) => {
                return self.lower_tensor_type(el_ty, dims, top, false);
            }
            // A dynamic tensor is the fully dynamic memref. It is the only thing that is now:
            // an empty dimension list on a `Tensor` means rank 0, not unknown (Vx#399).
            syntax::Type::DynTensor(el_ty, top) => {
                return self.lower_tensor_type(el_ty, &[], top, true);
            }
            syntax::Type::Scalar(el_ty) => {
                return Ok(match el_ty {
                    ElementType::F16 => self.f16_ty,
                    ElementType::F32 => self.f32_ty,
                    ElementType::F64 => self.f64_ty,
                    ElementType::BF16 => self.bf16_ty,
                    ElementType::I32 | ElementType::U32 => self.i32_ty,
                    ElementType::I64 | ElementType::U64 => self.i64_ty,
                    ElementType::I4 | ElementType::U4 => self.i4_ty,
                    ElementType::I8 | ElementType::U8 => self.i8_ty,
                    ElementType::I16 | ElementType::U16 => self.i16_ty,
                    ElementType::I128 | ElementType::U128 => self.i128_ty,
                    ElementType::Bool => self.i1_ty,
                    ElementType::F8E4M3 | ElementType::F8E5M2 => {
                        return Err(LowerError::from(
                            "fp8 element types are capacity/declaration-only; fp8 codegen is \
                             tracked in #249"
                                .to_string(),
                        ));
                    }
                    ElementType::Generic(_) => {
                        return Err(LowerError::from(
                            "internal: generic element type reached codegen (monomorphization \
                             should have instantiated it)"
                                .to_string(),
                        ));
                    }
                });
            }
            syntax::Type::Matrix => {
                return Err(LowerError::from(
                    "codegen does not support the `Matrix` type".to_string(),
                ))
            }
            syntax::Type::Ref(inner, _mem) => {
                return self.lower_type(inner);
            }
            syntax::Type::Verified(inner) => return self.lower_type(inner),
            syntax::Type::Pinned(inner, _top) => {
                let inner_ty_str = self.lower_type(inner)?.to_string();
                inner_ty_str
            }
            syntax::Type::Borrow {
                inner,
                mem_space: mem,
                ..
            }
            | syntax::Type::Pointer(inner, mem, _) => {
                let inner_str = self.lower_type_str(inner)?;
                if inner_str.starts_with("memref<") {
                    // A borrowed tensor is the slot holding its descriptor, and every tensor
                    // this backend allocates is dynamically shaped. Naming the static shape
                    // here left the call site casting between two memref element types, which
                    // `memref.cast` cannot do (Vx#417).
                    format!("memref<{}>", erase_memref_extents(&inner_str))
                } else {
                    // A space with no honest target mapping (an undeclared custom space, or one
                    // whose declaration gives no `scope:`) is an error, not a silent fallback --
                    // the old code answered "4", NVPTX read-only constant memory (#258).
                    let addr_space = match mem.as_ref() {
                        None => crate::arch::AddressSpace::Host,
                        Some(m) => crate::arch::declared_address_space(m, self.memories.get(m))
                            .ok_or_else(|| LowerError::from(unmappable_space_message(m)))?,
                    };
                    format!("!llvm.ptr<{}>", addr_space.nvptx_addrspace())
                }
            }
            syntax::Type::Struct(name, _) => {
                if let Some(enum_def) = self.enums.get(name) {
                    if name.starts_with("Option<") {
                        let mut payload_ty_str = "none".to_string();
                        for (v_name, payload) in enum_def {
                            if **v_name == *"Some" {
                                if let Some(types) = payload {
                                    if !types.is_empty() {
                                        let mut lowered = self.lower_type_str(&types[0])?;
                                        if lowered.starts_with("memref<") {
                                            lowered = "!llvm.ptr".to_string();
                                        }
                                        payload_ty_str = lowered;
                                    }
                                }
                            }
                        }
                        return Type::parse(
                            self.context,
                            &format!("!llvm.struct<\"{}\", (i32, {})>", name, payload_ty_str),
                        )
                        .ok_or_else(|| {
                            LowerError::from("failed to parse Option enum layout".to_string())
                        });
                    }
                    return Ok(self.i32_ty);
                }
                if let Some(decl) = self.structs.get(name).cloned() {
                    let mut field_types = Vec::new();
                    for (_, ty) in &decl.fields {
                        let mut lowered = self.lower_type_str(ty)?;
                        if lowered.starts_with("memref<") {
                            lowered = "!llvm.ptr".to_string();
                        }
                        field_types.push(lowered);
                    }
                    format!("!llvm.struct<\"{}\", ({})>", name, field_types.join(","))
                } else if name.as_ref() == "void" {
                    "none".to_string()
                } else if name.contains('<') {
                    // A generic instance that was flattened to a bracketed nominal
                    // name during monomorphization (e.g. "Vec<i32>", "Option<i32>")
                    // — such names arise when a monomorphized value type is
                    // stringified and re-stored as a plain `Struct`. Re-parse it
                    // into a proper `GenericInstance` and lower that, which resolves
                    // to the concrete monomorphized struct body.
                    let mut lexer = crate::lexer::Lexer::new(name);
                    let tokens = lexer.tokenize();
                    let mut parser = crate::parser::Parser::new(&tokens, name);
                    match parser.parse_type() {
                        Ok(parsed @ syntax::Type::GenericInstance(..)) => {
                            return self.lower_type(&parsed);
                        }
                        _ => format!("!llvm.struct<\"{}\">", name),
                    }
                } else {
                    format!("!llvm.struct<\"{}\">", name)
                }
            }
            syntax::Type::GenericInstance(base, args) => {
                if let syntax::Type::Struct(name, _) = &**base {
                    if let Some(decl) = self.structs.get(name).cloned() {
                        let mut field_types = Vec::new();
                        let mut mapping: std::collections::HashMap<
                            crate::symbol::Symbol,
                            syntax::Type,
                        > = std::collections::HashMap::new();
                        for (i, param) in decl.generics.iter().enumerate() {
                            if i >= args.len() {
                                return Err(LowerError::from(format!(
                                    "generic type `{}` expects {} type argument(s), got {}",
                                    name,
                                    decl.generics.len(),
                                    args.len()
                                )));
                            }
                            mapping.insert(param.name().into(), args[i].clone());
                        }
                        for (_, ty) in &decl.fields {
                            let sub_ty = ty.substitute(&mapping);
                            let mut lowered = self.lower_type_str(&sub_ty)?;
                            if lowered.starts_with("memref<") {
                                lowered = "!llvm.ptr".to_string();
                            }
                            field_types.push(lowered);
                        }
                        let args_str: Vec<String> = args
                            .iter()
                            .map(|a| {
                                let lowered = self.lower_type_str(a)?;
                                Ok(lowered
                                    .replace("!", "")
                                    .replace("<", "_")
                                    .replace(">", "_")
                                    .replace(" ", "_")
                                    .replace(",", "_")
                                    .replace("\"", "")
                                    .replace("(", "_")
                                    .replace(")", "_")
                                    .replace(".", "_"))
                            })
                            .collect::<Result<Vec<_>, LowerError>>()?;
                        format!(
                            "!llvm.struct<\"{}_{}\", ({})>",
                            name,
                            args_str.join("_"),
                            field_types.join(", ")
                        )
                    } else if let Some(enum_def) = self.enums.get(name).cloned() {
                        let ty_arg = args.first().ok_or_else(|| {
                            LowerError::from(format!(
                                "generic enum `{}` used with no type argument",
                                name
                            ))
                        })?;
                        let mut payload_ty_str = "none".to_string();
                        for (v_name, payload) in enum_def {
                            if v_name == "Some".into() {
                                if let Some(types) = payload {
                                    if !types.is_empty() {
                                        let mut mapping: std::collections::HashMap<
                                            crate::symbol::Symbol,
                                            syntax::Type,
                                        > = std::collections::HashMap::new();
                                        mapping.insert("T".into(), ty_arg.clone());
                                        let sub_ty = types[0].substitute(&mapping);
                                        let mut lowered = self.lower_type_str(&sub_ty)?;
                                        if lowered.starts_with("memref<") {
                                            lowered = "!llvm.ptr".to_string();
                                        }
                                        payload_ty_str = lowered;
                                    }
                                }
                            }
                        }

                        let args_str: Vec<String> = args
                            .iter()
                            .map(|a| {
                                let lowered = self.lower_type_str(a)?;
                                Ok(lowered
                                    .replace("!", "")
                                    .replace("<", "_")
                                    .replace(">", "_")
                                    .replace(" ", "_")
                                    .replace(",", "_")
                                    .replace("\"", "")
                                    .replace("(", "_")
                                    .replace(")", "_")
                                    .replace(".", "_"))
                            })
                            .collect::<Result<Vec<_>, LowerError>>()?;

                        format!(
                            "!llvm.struct<\"{}_{}\", (i32, {})>",
                            name,
                            args_str.join("_"),
                            payload_ty_str
                        )
                    } else {
                        return Err(LowerError::from(format!(
                            "generic struct or enum `{}` not found during codegen",
                            name
                        )));
                    }
                } else {
                    return Err(LowerError::from(
                        "internal: generic-instance base is not a struct".to_string(),
                    ));
                }
            }
            syntax::Type::Generic(_, _) => {
                return Err(LowerError::from(format!(
                    "internal: generic type reached codegen (should have been monomorphized): {:?}",
                    ty
                )));
            }
            syntax::Type::Simd(el_ty, n) => {
                let ty_str = match el_ty {
                    ElementType::F16 => "f16",
                    ElementType::F32 => "f32",
                    ElementType::F64 => "f64",
                    ElementType::BF16 => "bf16",
                    ElementType::I4 | ElementType::U4 => "i4",
                    ElementType::I8 | ElementType::U8 => "i8",
                    ElementType::I16 | ElementType::U16 => "i16",
                    ElementType::I32 | ElementType::U32 => "i32",
                    ElementType::I64 | ElementType::U64 => "i64",
                    ElementType::I128 | ElementType::U128 => "i128",
                    ElementType::Bool => "i1",
                    ElementType::F8E4M3 | ElementType::F8E5M2 => {
                        return Err(LowerError::from(
                            "fp8 element types are capacity/declaration-only; fp8 codegen is \
                             tracked in #249"
                                .to_string(),
                        ));
                    }
                    ElementType::Generic(_) => {
                        return Err(LowerError::from(
                            "internal: generic element type reached codegen (monomorphization \
                             should have instantiated it)"
                                .to_string(),
                        ));
                    }
                };
                format!("vector<{}x{}>", n, ty_str)
            }
            syntax::Type::Enum(name, _) => {
                if let Some(enum_def) = self.enums.get(name) {
                    if name.starts_with("Option<") {
                        let mut payload_ty_str = "none".to_string();
                        for (v_name, payload) in enum_def {
                            if **v_name == *"Some" {
                                if let Some(types) = payload {
                                    if !types.is_empty() {
                                        let mut lowered = self.lower_type_str(&types[0])?;
                                        if lowered.starts_with("memref<") {
                                            lowered = "!llvm.ptr".to_string();
                                        }
                                        payload_ty_str = lowered;
                                    }
                                }
                            }
                        }
                        return Type::parse(
                            self.context,
                            &format!("!llvm.struct<\"{}\", (i32, {})>", name, payload_ty_str),
                        )
                        .ok_or_else(|| {
                            LowerError::from("failed to parse Option enum layout".to_string())
                        });
                    }
                }
                "i32".to_string()
            }
            syntax::Type::Function(_, _) => {
                return Ok(self.ptr_ty);
            }
            syntax::Type::Closure(_, _) => {
                return Ok(Type::parse(self.context, "!llvm.struct<(ptr, ptr)>").unwrap());
            }
            syntax::Type::Module(..) => "none".to_string(),
            syntax::Type::Const(expr) => {
                return Err(LowerError::from(format!(
                    "cannot lower a const generic argument to an MLIR type: {:?}",
                    expr
                )));
            }
            syntax::Type::Unknown => "unknown".to_string(),
        };

        Type::parse(self.context, &ty_str).ok_or_else(|| {
            LowerError::from(format!("failed to parse lowered MLIR type `{}`", ty_str))
        })
    }

    pub(crate) fn lower_type_str(
        &self,
        ty: &syntax::Type,
    ) -> Result<String, crate::codegen::lower::LowerError> {
        if let syntax::Type::Function(_, _) = ty {
            return Ok("!llvm.ptr".to_string());
        }
        if let syntax::Type::Const(expr) = ty {
            if let syntax::Expr::Number(n) = &**expr {
                return Ok(n.value.to_string());
            } else if let syntax::Expr::StringLiteral(s) = &**expr {
                return Ok(s.value.to_string());
            } else {
                return Ok(format!("{:?}", expr)
                    .replace(" ", "_")
                    .replace("\"", "")
                    .replace("(", "_")
                    .replace(")", "_"));
            }
        }
        let t = self.lower_type(ty)?;
        Ok(t.to_string())
    }

    /// Lowers a Vx `Tensor(ElementType, dims, topology)` to an MLIR `memref<...>` type.
    fn lower_tensor_type(
        &self,
        el_ty: &ElementType,
        dims: &[syntax::Expr],
        top: &Option<syntax::Placement>,
        dynamic: bool,
    ) -> Result<Type<'c>, crate::codegen::lower::LowerError> {
        let ty_str = match el_ty {
            ElementType::F16 => "f16",
            ElementType::F32 => "f32",
            ElementType::F64 => "f64",
            ElementType::BF16 => "bf16",
            ElementType::I4 | ElementType::U4 => "i4",
            ElementType::I8 | ElementType::U8 => "i8",
            ElementType::I16 | ElementType::U16 => "i16",
            ElementType::I32 | ElementType::U32 => "i32",
            ElementType::I64 | ElementType::U64 => "i64",
            ElementType::I128 | ElementType::U128 => "i128",
            ElementType::Bool => "i1",
            ElementType::F8E4M3 | ElementType::F8E5M2 => {
                return Err(LowerError::from(
                    "fp8 element types are capacity/declaration-only; fp8 codegen is \
                     tracked in #249"
                        .to_string(),
                ));
            }
            ElementType::Generic(_) => {
                return Err(LowerError::from(
                    "internal: generic element type reached codegen (monomorphization \
                     should have instantiated it)"
                        .to_string(),
                ));
            }
        };

        // `Tensor<f32, []>` is rank 0 -- `memref<f32>` -- and only a `DynTensor` is the
        // rank-2 dynamic memref. The two used to share the empty dimension list, so a stated
        // rank-0 parameter silently became `memref<?x?xf32>` (Vx#399).
        let mut shape_str = String::new();
        if dims.is_empty() {
            if dynamic {
                shape_str = "?x?".to_string();
            }
        } else {
            for (i, dim) in dims.iter().enumerate() {
                if let syntax::Expr::Number(NumberExpr {
                    value: n_str,
                    ty: _,
                    span: _,
                }) = dim
                {
                    if let Ok(n) = n_str.parse::<f64>() {
                        shape_str.push_str(&format!("{}", n as i64));
                    }
                } else {
                    shape_str.push('?');
                }
                if i < dims.len() - 1 {
                    shape_str.push('x');
                }
            }
        }

        if !shape_str.is_empty() && !shape_str.ends_with('x') {
            shape_str.push('x');
        }

        // Address space follows the topology's default memory space (single source of truth
        // in `arch`), so on-`Topology` and in-`MemorySpace` values of one buffer agree. A
        // topology whose memory has no honest target mapping is an error rather than a silent
        // fallback to constant memory (#258).
        let addr_space = match top.as_ref() {
            None => crate::arch::AddressSpace::Host,
            Some(p) => {
                crate::arch::topology_address_space(&p.topology, &self.memories, &self.topologies)
                    .ok_or_else(|| {
                    LowerError::from(format!(
                    "topology '{}' has no memory space that maps to this target's address spaces; \
                     declare its memory with a `scope:` (device/sm/cta/thread)",
                    p.topology.display_name()
                ))
                })?
            }
        };

        let memref_str = if addr_space != crate::arch::AddressSpace::Host {
            format!(
                "memref<{}{}, {}>",
                shape_str,
                ty_str,
                addr_space.nvptx_addrspace()
            )
        } else {
            format!("memref<{}{}>", shape_str, ty_str)
        };

        Type::parse(self.context, &memref_str).ok_or_else(|| {
            LowerError::from(format!(
                "failed to parse lowered memref type `{}`",
                memref_str
            ))
        })
    }

    pub fn infer_ast_type(&self, expr: &Expr) -> Option<syntax::Type> {
        match expr {
            // A cast's value has the target type by definition (`my_closure as ||->i32`), and the
            // indirect-call path needs it in `ast_env` to rebuild the callee signature.
            Expr::AsCast(c) => Some(c.target_ty.clone()),
            Expr::Identifier(id) => self.ast_env.get(&id.name).cloned().or_else(|| {
                // A bare function name used as a *value* (`let f = probe`) is a function pointer;
                // recover its signature from the function registry so a later indirect call `f(..)`
                // can build the callee type (#268). The frontend already types this via `self.lookup`;
                // this is the codegen counterpart, keeping `ast_env` populated so `lower/expr.rs`'s
                // indirect-call path does not hit "Missing signature for function pointer".
                self.syntax_functions.get(&id.name).map(|f| {
                    syntax::Type::Function(
                        f.params.iter().map(|(_, t)| t.clone()).collect(),
                        Box::new(f.return_type.clone()),
                    )
                })
            }),
            Expr::MemberAccess(ma) => {
                let mut base_ty = self.infer_ast_type(&ma.base)?;
                if let syntax::Type::Borrow { inner, .. } = base_ty {
                    base_ty = *inner;
                }
                let mut generic_args = None;
                if let syntax::Type::GenericInstance(inner, args) = base_ty {
                    base_ty = *inner;
                    generic_args = Some(args);
                }
                if let syntax::Type::Struct(s_name, _) = base_ty {
                    if let Some(decl) = self.structs.get(&s_name) {
                        for (n, t) in &decl.fields {
                            if n == &ma.member {
                                let mut resolved_ty = t.clone();
                                if let Some(args) = generic_args {
                                    let mut mapping: std::collections::HashMap<
                                        crate::symbol::Symbol,
                                        syntax::Type,
                                    > = std::collections::HashMap::new();
                                    for (i, param) in decl.generics.iter().enumerate() {
                                        if i < args.len() {
                                            mapping.insert(param.name().into(), args[i].clone());
                                        }
                                    }
                                    resolved_ty = resolved_ty.substitute(&mapping);
                                }
                                return Some(resolved_ty);
                            }
                        }
                    }
                }
                None
            }
            Expr::FunctionCall(fc) => {
                let name = fc.name.to_string();
                // `Tensor<T>([d0, d1, ...])` constructor: recover the static shape so slice
                // indexing / vectorized reductions know the tile dims (mirrors sema in hir/expr).
                // The newer spelling writes the whole type, placement included, so there is
                // nothing to recover -- hand it back as written.
                if matches!(
                    name.as_str(),
                    "Tensor::new" | "Tensor::uninit" | "DynTensor::new" | "DynTensor::uninit"
                ) {
                    return fc.type_args.as_ref().and_then(|a| a.first()).cloned();
                }
                if name.starts_with("Tensor")
                    && !name.ends_with("::from")
                    && !name.contains('$')
                    && !name.contains("__")
                {
                    let el_ty = match fc.type_args.as_ref().and_then(|a| a.first()) {
                        Some(syntax::Type::Scalar(el)) => el.clone(),
                        _ => syntax::ElementType::F32,
                    };
                    let dims = match fc.args.first() {
                        Some(Expr::Array(arr)) => match arr.initializer_shape() {
                            // Initializer list `Tensor<T>([[..],[..]])`: shape from the nesting.
                            Some(shape) => shape
                                .into_iter()
                                .map(|n| {
                                    Expr::Number(syntax::NumberExpr {
                                        value: n.to_string().into(),
                                        ty: Some(syntax::ElementType::I32),
                                        span: syntax::Span::default(),
                                    })
                                })
                                .collect(),
                            None => arr.elements.clone(),
                        },
                        _ => fc.args.clone(),
                    };
                    return Some(syntax::Type::Tensor(el_ty, dims, None));
                }
                self.syntax_functions
                    .get(&fc.name)
                    .map(|decl| decl.return_type.clone())
            }
            // `transfer(x, space)` relocates x but preserves its shape, so a slice of a
            // transferred tensor (`q[i]` inside a spawn) can still recover its static dims.
            Expr::Transfer(e) => self.infer_ast_type(&e.expr),
            Expr::MethodCall(mc) => {
                let mut base_ty = self.infer_ast_type(&mc.base)?;
                if let syntax::Type::Borrow { inner, .. } = base_ty {
                    base_ty = *inner;
                }
                if let syntax::Type::GenericInstance(inner, _) = base_ty {
                    base_ty = *inner;
                }
                if let syntax::Type::Struct(s_name, _) = base_ty {
                    let mangled = format!("{}_{}", s_name, mc.method_name);
                    self.syntax_functions
                        .get(mangled.as_str())
                        .map(|decl| decl.return_type.clone())
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    pub fn flatten_indices(
        &mut self,
        expr: &Expr,
        block: melior::ir::BlockRef<'c, 'c>,
    ) -> Option<(
        Value<'c, 'c>,
        Type<'c>,
        Vec<Value<'c, 'c>>,
        melior::ir::BlockRef<'c, 'c>,
    )> {
        match expr {
            Expr::IndexAccess(syntax::IndexAccessExpr {
                base,
                index: idx,
                span: _,
            }) => {
                let (base_val, base_ty, mut indices, block) = self.flatten_indices(base, block)?;
                let (idx_val, _, block) = self.generate_expr(idx, block).ok()?;

                let idx_ty_str = idx_val.r#type().to_string();
                let actual_idx = if idx_ty_str != "index" {
                    let cast_op = melior::ir::operation::OperationBuilder::new(
                        "arith.index_cast",
                        self.loc(),
                    )
                    .add_operands(&[idx_val])
                    .add_results(&[self.index_ty])
                    .build()
                    .ok()?;
                    block.append_operation(cast_op).result(0).ok()?.into()
                } else {
                    idx_val
                };

                indices.push(actual_idx);
                Some((base_val, base_ty, indices, block))
            }
            _ => {
                let (val, ty, block) = self.generate_expr(expr, block).ok()?;
                Some((val, ty, Vec::new(), block))
            }
        }
    }
}

/// The diagnostic for a memory space that cannot be mapped to a target address space: an
/// undeclared custom space, or a declaration carrying no `scope:`. Reported instead of silently
/// lowering into NVPTX constant (read-only) memory, which is what every declared space used to
/// get (#258).
fn unmappable_space_message(mem: &syntax::MemorySpace) -> String {
    format!(
        "memory space '{}' has no address space on this target; declare it with a `scope:` \
         (device/sm/cta/thread) so it maps to global/shared/private memory",
        mem.name()
    )
}
