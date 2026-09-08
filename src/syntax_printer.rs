//===- ast_printer.rs - Vx Compiler ----------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides utility functions for pretty-printing the Abstract Syntax Tree.
// It translates parsed Vx AST nodes back into human-readable string representations,
// which is primarily used for debugging, diagnostic messages, and testing the parser's
// fidelity.
//
//===----------------------------------------------------------------------===//
use crate::syntax::*;
use std::io::Write;

pub struct AstPrinter;

#[derive(Clone, Copy)]
pub struct Indent {
    depth: usize,
    last_mask: u64,
}

impl Default for Indent {
    fn default() -> Self {
        Self::new()
    }
}

impl Indent {
    pub fn new() -> Self {
        Self {
            depth: 0,
            last_mask: 0,
        }
    }

    pub fn child(&self, is_last: bool) -> Self {
        let mut mask = self.last_mask;
        if is_last {
            mask |= 1 << self.depth;
        }
        Self {
            depth: self.depth + 1,
            last_mask: mask,
        }
    }

    pub fn print(&self, w: &mut impl Write) -> std::io::Result<()> {
        for i in 0..self.depth {
            if (self.last_mask & (1 << i)) != 0 {
                write!(w, "   ")?;
            } else {
                write!(w, "│  ")?;
            }
        }
        Ok(())
    }
}

impl AstPrinter {
    fn print_list<T, F>(items: &[T], mut print_fn: F) -> std::io::Result<()>
    where
        F: FnMut(&T, &str, bool) -> std::io::Result<()>,
    {
        for (i, item) in items.iter().enumerate() {
            let is_last = i == items.len() - 1;
            let prefix = if is_last { "└─ " } else { "├─ " };
            print_fn(item, prefix, is_last)?;
        }
        Ok(())
    }

    pub fn print_program(program: &Program, w: &mut impl Write) -> std::io::Result<()> {
        writeln!(w, "Program")?;

        for (i, imp) in program.imports.iter().enumerate() {
            let is_last = i == program.imports.len() - 1
                && program.structs.is_empty()
                && program.functions.is_empty()
                && program.externs.is_empty();
            let prefix = if is_last { "└─ " } else { "├─ " };
            writeln!(w, "{}Import: {}", prefix, imp.path.join("::"))?;
        }

        for (i, struc) in program.structs.iter().enumerate() {
            let is_last = i == program.structs.len() - 1
                && program.functions.is_empty()
                && program.externs.is_empty();
            let prefix = if is_last { "└─ " } else { "├─ " };
            writeln!(w, "{}Struct: {}", prefix, struc.name)?;
            Self::print_fields(w, &struc.fields, &Indent::new().child(is_last))?;
        }

        for (i, func) in program.functions.iter().enumerate() {
            let is_last = i == program.functions.len() - 1 && program.externs.is_empty();
            let prefix = if is_last { "└─ " } else { "├─ " };
            writeln!(w, "{}Function: {}", prefix, func.name)?;
            Self::print_statements(w, &func.body, &Indent::new().child(is_last))?;
        }
        Ok(())
    }

    fn print_fields(
        w: &mut impl Write,
        fields: &[(crate::symbol::Symbol, Type)],
        indent: &Indent,
    ) -> std::io::Result<()> {
        for (i, (name, ty)) in fields.iter().enumerate() {
            let is_last = i == fields.len() - 1;
            let prefix = if is_last { "└─ " } else { "├─ " };
            indent.print(w)?;
            writeln!(w, "{}{}: {:?}", prefix, name, ty)?;
        }
        Ok(())
    }

    fn print_statements(
        w: &mut impl Write,
        stmts: &[Statement],
        indent: &Indent,
    ) -> std::io::Result<()> {
        Self::print_list(stmts, |stmt, prefix, is_last| {
            match stmt {
                Statement::LetDecl(LetDeclStmt {
                    name,
                    is_mut,
                    ty_ann: ty,
                    expr,
                    span: _,
                }) => {
                    indent.print(w)?;
                    writeln!(
                        w,
                        "{}Let {}{}{} = ",
                        prefix,
                        if *is_mut { "mut " } else { "" },
                        name,
                        if ty.is_some() { " (typed)" } else { "" }
                    )?;
                    Self::print_expr(w, expr, &indent.child(is_last), true)?;
                }
                Statement::Assign(AssignStmt { lhs, rhs, span: _ }) => {
                    indent.print(w)?;
                    writeln!(w, "{}Assign", prefix)?;
                    Self::print_expr(w, lhs, &indent.child(is_last), false)?;
                    Self::print_expr(w, rhs, &indent.child(is_last), true)?;
                }
                Statement::Return(ReturnStmt { expr, span: _ }) => {
                    indent.print(w)?;
                    writeln!(w, "{}Return", prefix)?;
                    if let Some(e) = expr {
                        Self::print_expr(w, e, &indent.child(is_last), true)?;
                    }
                }
                Statement::ExprStmt(ExprStmtStmt {
                    expr,
                    has_semi: _,
                    span: _,
                }) => {
                    indent.print(w)?;
                    writeln!(w, "{}ExprStmt", prefix)?;
                    Self::print_expr(w, expr, &indent.child(is_last), true)?;
                }
                _ => {
                    indent.print(w)?;
                    writeln!(w, "{}{:?}", prefix, stmt)?; // Fallback for other statements
                }
            }
            Ok(())
        })
    }

    fn print_expr(
        w: &mut impl Write,
        expr: &Expr,
        indent: &Indent,
        is_last: bool,
    ) -> std::io::Result<()> {
        let prefix = if is_last { "└─ " } else { "├─ " };
        match expr {
            Expr::Identifier(IdentifierExpr { name, span: _ }) => {
                indent.print(w)?;
                writeln!(w, "{}Identifier({})", prefix, name)?;
            }
            Expr::Number(NumberExpr {
                value: val,
                ty: el_ty,
                span: _,
            }) => {
                indent.print(w)?;
                writeln!(w, "{}Number({}{:?})", prefix, val, el_ty)?;
            }
            Expr::StringLiteral(StringLiteralExpr { value: s, span: _ }) => {
                indent.print(w)?;
                writeln!(w, "{}String(\"{}\")", prefix, s)?;
            }
            Expr::BinaryOp(BinaryOpExpr {
                lhs,
                op,
                rhs,
                span: _,
            }) => {
                indent.print(w)?;
                writeln!(w, "{}BinaryOp({:?})", prefix, op)?;
                let new_indent = indent.child(is_last);
                Self::print_expr(w, lhs, &new_indent, false)?;
                Self::print_expr(w, rhs, &new_indent, true)?;
            }
            Expr::FunctionCall(FunctionCallExpr {
                name,
                type_args: _,
                args,
                span: _,
            }) => {
                indent.print(w)?;
                writeln!(w, "{}Call({})", prefix, name)?;
                let new_indent = indent.child(is_last);
                for (i, arg) in args.iter().enumerate() {
                    Self::print_expr(w, arg, &new_indent, i == args.len() - 1)?;
                }
            }
            Expr::MethodCall(MethodCallExpr {
                base: expr,
                method_name: method,
                type_args: _,
                args,
                span: _,
            }) => {
                indent.print(w)?;
                writeln!(w, "{}MethodCall(.'{}')", prefix, method)?;
                let new_indent = indent.child(is_last);
                Self::print_expr(w, expr, &new_indent, args.is_empty())?;
                for (i, arg) in args.iter().enumerate() {
                    Self::print_expr(w, arg, &new_indent, i == args.len() - 1)?;
                }
            }
            Expr::Array(ArrayExpr {
                elements: items,
                span: _,
            }) => {
                indent.print(w)?;
                writeln!(w, "{}Array", prefix)?;
                let new_indent = indent.child(is_last);
                for (i, item) in items.iter().enumerate() {
                    Self::print_expr(w, item, &new_indent, i == items.len() - 1)?;
                }
            }
            Expr::EnumVariant(EnumVariantExpr {
                enum_name,
                variant_name: variant,
                payload: _,
                span: _,
            }) => {
                indent.print(w)?;
                writeln!(w, "{}Enum({}::{})", prefix, enum_name, variant)?;
            }

            Expr::ComptimeBlock(ComptimeBlockExpr {
                stmts,
                ret,
                span: _,
            }) => {
                indent.print(w)?;
                writeln!(w, "{}ComptimeBlock", prefix)?;
                let new_indent = indent.child(is_last);
                Self::print_statements(w, stmts, &new_indent)?;
                if let Some(r) = ret {
                    Self::print_expr(w, r, &new_indent, true)?;
                }
            }
            _ => {
                indent.print(w)?;
                writeln!(w, "{}{:?}", prefix, expr)?;
            }
        }
        Ok(())
    }
}
