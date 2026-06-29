//===- lower_ast.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Converts the tree-based syntax AST into the flat, arena-based HIR.
//
//===----------------------------------------------------------------------===//

use crate::hir::arena::*;
use crate::syntax::*;

impl HirArena {
    pub fn lower_expr(&mut self, expr: &Expr) -> ExprId {
        let hir_expr = match expr {
            Expr::Identifier(e) => HirExpr::Identifier(self.lower_identifierexpr(e)),
            Expr::EnumVariant(e) => HirExpr::EnumVariant(self.lower_enumvariantexpr(e)),
            Expr::Number(e) => HirExpr::Number(self.lower_numberexpr(e)),
            Expr::StringLiteral(e) => HirExpr::StringLiteral(self.lower_stringliteralexpr(e)),
            Expr::SpawnOn(e) => HirExpr::SpawnOn(self.lower_spawnonexpr(e)),
            Expr::Transfer(e) => HirExpr::Transfer(self.lower_transferexpr(e)),
            Expr::FunctionCall(e) => HirExpr::FunctionCall(self.lower_functioncallexpr(e)),
            Expr::AsCast(e) => HirExpr::AsCast(self.lower_ascastexpr(e)),
            Expr::IndirectCall(e) => HirExpr::IndirectCall(self.lower_indirectcallexpr(e)),
            Expr::Array(e) => HirExpr::Array(self.lower_arrayexpr(e)),
            Expr::MemberAccess(e) => HirExpr::MemberAccess(self.lower_memberaccessexpr(e)),
            Expr::IndexAccess(e) => HirExpr::IndexAccess(self.lower_indexaccessexpr(e)),
            Expr::MethodCall(e) => HirExpr::MethodCall(self.lower_methodcallexpr(e)),
            Expr::BinaryOp(e) => HirExpr::BinaryOp(self.lower_binaryopexpr(e)),
            Expr::RelationalOp(e) => HirExpr::RelationalOp(self.lower_relationalopexpr(e)),
            Expr::LogicalOp(e) => HirExpr::LogicalOp(self.lower_logicalopexpr(e)),
            Expr::UnaryOp(e) => HirExpr::UnaryOp(self.lower_unaryopexpr(e)),
            Expr::Borrow(e) => HirExpr::Borrow(self.lower_borrowexpr(e)),
            Expr::Dereference(e) => HirExpr::Dereference(self.lower_dereferenceexpr(e)),
            Expr::UnsafeBlock(e) => HirExpr::UnsafeBlock(self.lower_unsafeblockexpr(e)),
            Expr::ComptimeBlock(e) => HirExpr::ComptimeBlock(self.lower_comptimeblockexpr(e)),
            Expr::StructInit(e) => HirExpr::StructInit(self.lower_structinitexpr(e)),
            Expr::MemorySpace(e) => HirExpr::MemorySpace(self.lower_memoryspaceexpr(e)),
            Expr::Topology(e) => HirExpr::Topology(self.lower_topologyexpr(e)),
            Expr::Grad(e) => HirExpr::Grad(self.lower_gradexpr(e)),
            Expr::Vjp(e) => HirExpr::Vjp(self.lower_vjpexpr(e)),
            Expr::Jvp(e) => HirExpr::Jvp(self.lower_jvpexpr(e)),
            Expr::If(e) => HirExpr::If(self.lower_ifexpr(e)),
            Expr::Range(e) => HirExpr::Range(self.lower_rangeexpr(e)),
            Expr::Match(e) => HirExpr::Match(self.lower_matchexpr(e)),
            Expr::VecMacro(e) => HirExpr::VecMacro(self.lower_vecmacroexpr(e)),
            Expr::Closure(e) => HirExpr::Closure(self.lower_closureexpr(e)),
            Expr::MacroCall(e) => HirExpr::MacroCall(self.lower_macrocallexpr(e)),
            Expr::Print(e) => HirExpr::Print(self.lower_printexpr(e)),
            Expr::Println(e) => HirExpr::Println(self.lower_printlnexpr(e)),
            Expr::SizeOf(e) => HirExpr::SizeOf(self.lower_sizeofexpr(e)),
            Expr::InlineMlir(e) => HirExpr::InlineMlir(self.lower_inlinemlirexpr(e)),
        };
        self.alloc_expr(hir_expr)
    }

    pub fn lower_stmt(&mut self, stmt: &Statement) -> StmtId {
        let hir_stmt = match stmt {
            Statement::LetDecl(s) => HirStmt::LetDecl(self.lower_letdeclstmt(s)),
            Statement::Return(s) => HirStmt::Return(self.lower_returnstmt(s)),
            Statement::ExprStmt(s) => HirStmt::Expr(self.lower_exprstmtstmt(s)),
            Statement::ForLoop(s) => HirStmt::ForLoop(self.lower_forloopstmt(s)),
            Statement::Assign(s) => HirStmt::Assign(self.lower_assignstmt(s)),
            Statement::CompoundAssign(s) => {
                HirStmt::CompoundAssign(self.lower_compoundassignstmt(s))
            }
            Statement::Assert(s) => HirStmt::Assert(self.lower_assertstmt(s)),
            Statement::Loop(s) => HirStmt::Loop(self.lower_loopstmt(s)),
            Statement::Break(s) => HirStmt::Break(self.lower_breakstmt(s)),
            Statement::Continue(s) => HirStmt::Continue(self.lower_continuestmt(s)),
            Statement::MacroCall(s) => HirStmt::MacroCall(self.lower_macrocallstmt(s)),
            Statement::Error(span) => HirStmt::Error(*span),
        };
        self.alloc_stmt(hir_stmt)
    }
    fn lower_identifierexpr(&mut self, node: &IdentifierExpr) -> HirIdentifierExpr {
        HirIdentifierExpr {
            name: node.name.clone(),
            span: node.span,
        }
    }
    fn lower_enumvariantexpr(&mut self, node: &EnumVariantExpr) -> HirEnumVariantExpr {
        HirEnumVariantExpr {
            enum_name: node.enum_name.clone(),
            variant_name: node.variant_name.clone(),
            payload: node
                .payload
                .as_ref()
                .map(|stmts| stmts.iter().map(|s| self.lower_expr(s)).collect()),
            span: node.span,
        }
    }
    fn lower_numberexpr(&mut self, node: &NumberExpr) -> HirNumberExpr {
        HirNumberExpr {
            value: node.value.clone(),
            ty: node.ty.clone(),
            span: node.span,
        }
    }
    fn lower_stringliteralexpr(&mut self, node: &StringLiteralExpr) -> HirStringLiteralExpr {
        HirStringLiteralExpr {
            value: node.value.clone(),
            span: node.span,
        }
    }
    fn lower_spawnonexpr(&mut self, node: &SpawnOnExpr) -> HirSpawnOnExpr {
        HirSpawnOnExpr {
            top: node.top.clone(),
            stmts: node.stmts.iter().map(|s| self.lower_stmt(s)).collect(),
            ret: node.ret.as_ref().map(|e| self.lower_expr(e)),
            span: node.span,
        }
    }
    fn lower_transferexpr(&mut self, node: &TransferExpr) -> HirTransferExpr {
        HirTransferExpr {
            expr: self.lower_expr(&node.expr),
            space: node.space.clone(),
            cost: node.cost,
            span: node.span,
        }
    }
    fn lower_functioncallexpr(&mut self, node: &FunctionCallExpr) -> HirFunctionCallExpr {
        HirFunctionCallExpr {
            name: node.name.clone(),
            type_args: node.type_args.clone(),
            args: node.args.iter().map(|e| self.lower_expr(e)).collect(),
            span: node.span,
        }
    }
    fn lower_ascastexpr(&mut self, node: &AsCastExpr) -> HirAsCastExpr {
        HirAsCastExpr {
            expr: self.lower_expr(&node.expr),
            target_ty: node.target_ty.clone(),
            source_ty: node.source_ty.clone(),
            span: node.span,
        }
    }
    fn lower_indirectcallexpr(&mut self, node: &IndirectCallExpr) -> HirIndirectCallExpr {
        HirIndirectCallExpr {
            callee: self.lower_expr(&node.callee),
            args: node.args.iter().map(|e| self.lower_expr(e)).collect(),
            target_func_ty: node.target_func_ty.clone(),
            span: node.span,
        }
    }
    fn lower_arrayexpr(&mut self, node: &ArrayExpr) -> HirArrayExpr {
        HirArrayExpr {
            elements: node.elements.iter().map(|e| self.lower_expr(e)).collect(),
            span: node.span,
        }
    }
    fn lower_memberaccessexpr(&mut self, node: &MemberAccessExpr) -> HirMemberAccessExpr {
        HirMemberAccessExpr {
            base: self.lower_expr(&node.base),
            member: node.member.clone(),
            struct_name: node.struct_name.clone(),
            span: node.span,
        }
    }
    fn lower_indexaccessexpr(&mut self, node: &IndexAccessExpr) -> HirIndexAccessExpr {
        HirIndexAccessExpr {
            base: self.lower_expr(&node.base),
            index: self.lower_expr(&node.index),
            span: node.span,
        }
    }
    fn lower_methodcallexpr(&mut self, node: &MethodCallExpr) -> HirMethodCallExpr {
        HirMethodCallExpr {
            base: self.lower_expr(&node.base),
            method_name: node.method_name.clone(),
            type_args: node.type_args.clone(),
            args: node.args.iter().map(|e| self.lower_expr(e)).collect(),
            span: node.span,
        }
    }
    fn lower_binaryopexpr(&mut self, node: &BinaryOpExpr) -> HirBinaryOpExpr {
        HirBinaryOpExpr {
            lhs: self.lower_expr(&node.lhs),
            op: node.op.clone(),
            rhs: self.lower_expr(&node.rhs),
            span: node.span,
        }
    }
    fn lower_relationalopexpr(&mut self, node: &RelationalOpExpr) -> HirRelationalOpExpr {
        HirRelationalOpExpr {
            lhs: self.lower_expr(&node.lhs),
            op: node.op.clone(),
            rhs: self.lower_expr(&node.rhs),
            span: node.span,
        }
    }
    fn lower_logicalopexpr(&mut self, node: &LogicalOpExpr) -> HirLogicalOpExpr {
        HirLogicalOpExpr {
            lhs: self.lower_expr(&node.lhs),
            op: node.op.clone(),
            rhs: self.lower_expr(&node.rhs),
            span: node.span,
        }
    }
    fn lower_unaryopexpr(&mut self, node: &UnaryOpExpr) -> HirUnaryOpExpr {
        HirUnaryOpExpr {
            op: node.op.clone(),
            expr: self.lower_expr(&node.expr),
            span: node.span,
        }
    }
    fn lower_borrowexpr(&mut self, node: &BorrowExpr) -> HirBorrowExpr {
        HirBorrowExpr {
            expr: self.lower_expr(&node.expr),
            is_mut: node.is_mut,
            span: node.span,
        }
    }
    fn lower_dereferenceexpr(&mut self, node: &DereferenceExpr) -> HirDereferenceExpr {
        HirDereferenceExpr {
            expr: self.lower_expr(&node.expr),
            ty: node.ty.clone(),
            span: node.span,
        }
    }
    fn lower_unsafeblockexpr(&mut self, node: &UnsafeBlockExpr) -> HirUnsafeBlockExpr {
        HirUnsafeBlockExpr {
            stmts: node.stmts.iter().map(|s| self.lower_stmt(s)).collect(),
            ret: node.ret.as_ref().map(|e| self.lower_expr(e)),
            span: node.span,
        }
    }
    fn lower_comptimeblockexpr(&mut self, node: &ComptimeBlockExpr) -> HirComptimeBlockExpr {
        HirComptimeBlockExpr {
            stmts: node.stmts.iter().map(|s| self.lower_stmt(s)).collect(),
            ret: node.ret.as_ref().map(|e| self.lower_expr(e)),
            span: node.span,
        }
    }
    fn lower_structinitexpr(&mut self, node: &StructInitExpr) -> HirStructInitExpr {
        HirStructInitExpr {
            name: node.name.clone(),
            fields: node.fields.clone(),
            span: node.span,
        }
    }
    fn lower_memoryspaceexpr(&mut self, node: &MemorySpaceExpr) -> HirMemorySpaceExpr {
        HirMemorySpaceExpr {
            space: node.space.clone(),
            span: node.span,
        }
    }
    fn lower_topologyexpr(&mut self, node: &TopologyExpr) -> HirTopologyExpr {
        HirTopologyExpr {
            top: node.top.clone(),
            span: node.span,
        }
    }
    fn lower_gradexpr(&mut self, node: &GradExpr) -> HirGradExpr {
        HirGradExpr {
            target_fn: node.target_fn.clone(),
            args: node.args.iter().map(|e| self.lower_expr(e)).collect(),
            span: node.span,
        }
    }
    fn lower_vjpexpr(&mut self, node: &VjpExpr) -> HirVjpExpr {
        HirVjpExpr {
            target_fn: node.target_fn.clone(),
            args: node.args.iter().map(|e| self.lower_expr(e)).collect(),
            cotangent: self.lower_expr(&node.cotangent),
            span: node.span,
        }
    }
    fn lower_jvpexpr(&mut self, node: &JvpExpr) -> HirJvpExpr {
        HirJvpExpr {
            target_fn: node.target_fn.clone(),
            args: node.args.iter().map(|e| self.lower_expr(e)).collect(),
            tangent: self.lower_expr(&node.tangent),
            span: node.span,
        }
    }
    fn lower_ifexpr(&mut self, node: &IfExpr) -> HirIfExpr {
        HirIfExpr {
            is_comptime: node.is_comptime,
            cond: self.lower_expr(&node.cond),
            then_block: node.then_block.iter().map(|s| self.lower_stmt(s)).collect(),
            else_block: node
                .else_block
                .as_ref()
                .map(|stmts| stmts.iter().map(|s| self.lower_stmt(s)).collect()),
            span: node.span,
        }
    }
    fn lower_rangeexpr(&mut self, node: &RangeExpr) -> HirRangeExpr {
        HirRangeExpr {
            start: self.lower_expr(&node.start),
            end: self.lower_expr(&node.end),
            span: node.span,
        }
    }
    fn lower_matchexpr(&mut self, node: &MatchExpr) -> HirMatchExpr {
        HirMatchExpr {
            expr: self.lower_expr(&node.expr),
            arms: node.arms.clone(),
            span: node.span,
        }
    }
    fn lower_vecmacroexpr(&mut self, node: &VecMacroExpr) -> HirVecMacroExpr {
        HirVecMacroExpr {
            elements: node.elements.iter().map(|e| self.lower_expr(e)).collect(),
            span: node.span,
        }
    }
    fn lower_closureexpr(&mut self, node: &ClosureExpr) -> HirClosureExpr {
        HirClosureExpr {
            params: node.params.clone(),
            body: self.lower_expr(&node.body),
            captures: node.captures.clone(),
            ret_ty: node.ret_ty.clone(),
            span: node.span,
        }
    }
    fn lower_macrocallexpr(&mut self, node: &MacroCallExpr) -> HirMacroCallExpr {
        HirMacroCallExpr {
            name: node.name.clone(),
            token_tree: node.token_tree.clone(),
            block_tree: node.block_tree.clone(),
            span: node.span,
        }
    }
    fn lower_printexpr(&mut self, node: &PrintExpr) -> HirPrintExpr {
        HirPrintExpr {
            args: node.args.iter().map(|e| self.lower_expr(e)).collect(),
            span: node.span,
        }
    }
    fn lower_printlnexpr(&mut self, node: &PrintlnExpr) -> HirPrintlnExpr {
        HirPrintlnExpr {
            args: node.args.iter().map(|e| self.lower_expr(e)).collect(),
            span: node.span,
        }
    }
    fn lower_sizeofexpr(&mut self, node: &SizeOfExpr) -> HirSizeOfExpr {
        HirSizeOfExpr {
            target_ty: node.target_ty.clone(),
            span: node.span,
        }
    }
    fn lower_inlinemlirexpr(&mut self, node: &InlineMlirExpr) -> HirInlineMlirExpr {
        HirInlineMlirExpr {
            inputs: node.inputs.clone(),
            clobbers: node.clobbers.iter().map(|e| self.lower_expr(e)).collect(),
            returns: node.returns.clone(),
            dialects: node.dialects.clone(),
            block_str: node.block_str.clone(),
            span: node.span,
        }
    }
    fn lower_letdeclstmt(&mut self, node: &LetDeclStmt) -> HirLetDeclStmt {
        HirLetDeclStmt {
            name: node.name.clone(),
            is_mut: node.is_mut,
            ty_ann: node.ty_ann.clone(),
            expr: node.expr.clone(),
            span: node.span,
        }
    }
    fn lower_returnstmt(&mut self, node: &ReturnStmt) -> HirReturnStmt {
        HirReturnStmt {
            expr: node.expr.clone(),
            span: node.span,
        }
    }

    fn lower_macrocallstmt(&mut self, node: &MacroCallStmt) -> HirMacroCallStmt {
        HirMacroCallStmt {
            name: node.name.clone(),
            token_tree: node.token_tree.clone(),
            block_tree: node.block_tree.clone(),
            has_semi: node.has_semi,
            span: node.span,
        }
    }

    fn lower_exprstmtstmt(&mut self, node: &ExprStmtStmt) -> HirExprStmtStmt {
        HirExprStmtStmt {
            expr: node.expr.clone(),
            has_semi: node.has_semi,
            span: node.span,
        }
    }
    fn lower_forloopstmt(&mut self, node: &ForLoopStmt) -> HirForLoopStmt {
        HirForLoopStmt {
            iter: node.iter.clone(),
            iterable: self.lower_expr(&node.iterable),
            invariants: node.invariants.iter().map(|e| self.lower_expr(e)).collect(),
            body: node.body.iter().map(|s| self.lower_stmt(s)).collect(),
            span: node.span,
        }
    }
    fn lower_assignstmt(&mut self, node: &AssignStmt) -> HirAssignStmt {
        HirAssignStmt {
            lhs: node.lhs.clone(),
            rhs: node.rhs.clone(),
            span: node.span,
        }
    }
    fn lower_compoundassignstmt(&mut self, node: &CompoundAssignStmt) -> HirCompoundAssignStmt {
        HirCompoundAssignStmt {
            lhs: node.lhs.clone(),
            op: node.op.clone(),
            rhs: node.rhs.clone(),
            span: node.span,
        }
    }
    fn lower_assertstmt(&mut self, node: &AssertStmt) -> HirAssertStmt {
        HirAssertStmt {
            expr: self.lower_expr(&node.expr),
            msg: node.msg.clone(),
            span: node.span,
        }
    }
    fn lower_loopstmt(&mut self, node: &LoopStmt) -> HirLoopStmt {
        HirLoopStmt {
            invariants: node.invariants.iter().map(|e| self.lower_expr(e)).collect(),
            body: node.body.iter().map(|s| self.lower_stmt(s)).collect(),
            span: node.span,
        }
    }
    fn lower_breakstmt(&mut self, node: &BreakStmt) -> HirBreakStmt {
        HirBreakStmt { span: node.span }
    }
    fn lower_continuestmt(&mut self, node: &ContinueStmt) -> HirContinueStmt {
        HirContinueStmt { span: node.span }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::syntax::{ElementType, Expr, NumberExpr, Span};

    #[test]
    fn test_lower_number_expr() {
        let mut arena = HirArena::new();
        let expr = Expr::Number(NumberExpr {
            value: "42".into(),
            ty: Some(ElementType::I32),
            span: Span::default(),
        });
        let id = arena.lower_expr(&expr);
        assert_eq!(arena.exprs.len(), 1);
        match &arena.exprs[id] {
            HirExpr::Number(n) => assert_eq!(n.value.as_ref(), "42"),
            _ => panic!("Expected Number"),
        }
    }
}
