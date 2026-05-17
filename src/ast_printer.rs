use crate::ast::*;

pub struct AstPrinter;

impl AstPrinter {
    pub fn print_program(program: &Program) {
        println!("Program");

        for (i, imp) in program.imports.iter().enumerate() {
            let is_last = i == program.imports.len() - 1
                && program.structs.is_empty()
                && program.functions.is_empty()
                && program.externs.is_empty();
            let prefix = if is_last { "└─ " } else { "├─ " };
            println!("{}Import: {}", prefix, imp.path.join("::"));
        }

        for (i, struc) in program.structs.iter().enumerate() {
            let is_last = i == program.structs.len() - 1
                && program.functions.is_empty()
                && program.externs.is_empty();
            let prefix = if is_last { "└─ " } else { "├─ " };
            println!("{}Struct: {}", prefix, struc.name);
            Self::print_fields(&struc.fields, if is_last { "   " } else { "│  " });
        }

        for (i, func) in program.functions.iter().enumerate() {
            let is_last = i == program.functions.len() - 1 && program.externs.is_empty();
            let prefix = if is_last { "└─ " } else { "├─ " };
            println!("{}Function: {}", prefix, func.name);
            Self::print_statements(&func.body, if is_last { "   " } else { "│  " });
        }
    }

    fn print_fields(fields: &[(String, Type)], indent: &str) {
        for (i, (name, ty)) in fields.iter().enumerate() {
            let is_last = i == fields.len() - 1;
            let prefix = if is_last { "└─ " } else { "├─ " };
            println!("{}{}{}: {:?}", indent, prefix, name, ty);
        }
    }

    fn print_statements(stmts: &[Statement], indent: &str) {
        for (i, stmt) in stmts.iter().enumerate() {
            let is_last = i == stmts.len() - 1;
            let prefix = if is_last { "└─ " } else { "├─ " };

            match stmt {
                Statement::LetDecl(name, is_mut, ty, expr, _) => {
                    println!(
                        "{}{}Let {}{}{} = ",
                        indent,
                        prefix,
                        if *is_mut { "mut " } else { "" },
                        name,
                        if ty.is_some() { " (typed)" } else { "" }
                    );
                    Self::print_expr(
                        expr,
                        &format!("{}{}", indent, if is_last { "   " } else { "│  " }),
                        true,
                    );
                }
                Statement::Assign(lhs, rhs, _) => {
                    println!(" {}{}Assign", indent, prefix);
                    Self::print_expr(
                        lhs,
                        &format!("{}{}", indent, if is_last { "   " } else { "│  " }),
                        false,
                    );
                    Self::print_expr(
                        rhs,
                        &format!("{}{}", indent, if is_last { "   " } else { "│  " }),
                        true,
                    );
                }
                Statement::Return(expr, _) => {
                    println!("{}{}Return", indent, prefix);
                    Self::print_expr(
                        expr,
                        &format!("{}{}", indent, if is_last { "   " } else { "│  " }),
                        true,
                    );
                }
                Statement::ExprStmt(expr, _, _) => {
                    println!("{}{}ExprStmt", indent, prefix);
                    Self::print_expr(
                        expr,
                        &format!("{}{}", indent, if is_last { "   " } else { "│  " }),
                        true,
                    );
                }
                _ => {
                    println!("{}{}{:?}", indent, prefix, stmt); // Fallback for other statements
                }
            }
        }
    }

    fn print_expr(expr: &Expr, indent: &str, is_last: bool) {
        let prefix = if is_last { "└─ " } else { "├─ " };
        match expr {
            Expr::Identifier(name, _) => println!("{}{}Identifier({})", indent, prefix, name),
            Expr::Number(val, el_ty, _) => {
                println!("{}{}Number({}{:?})", indent, prefix, val, el_ty)
            }
            Expr::StringLiteral(s, _) => println!("{}{}String(\"{}\")", indent, prefix, s),
            Expr::BinaryOp(lhs, op, rhs, _) => {
                println!("{}{}BinaryOp({:?})", indent, prefix, op);
                let new_indent = format!("{}{}", indent, if is_last { "   " } else { "│  " });
                Self::print_expr(lhs, &new_indent, false);
                Self::print_expr(rhs, &new_indent, true);
            }
            Expr::FunctionCall(name, args, _) => {
                println!("{}{}Call({})", indent, prefix, name);
                let new_indent = format!("{}{}", indent, if is_last { "   " } else { "│  " });
                for (i, arg) in args.iter().enumerate() {
                    Self::print_expr(arg, &new_indent, i == args.len() - 1);
                }
            }
            Expr::MethodCall(expr, method, args, _) => {
                println!("{}{}MethodCall(.'{}')", indent, prefix, method);
                let new_indent = format!("{}{}", indent, if is_last { "   " } else { "│  " });
                Self::print_expr(expr, &new_indent, args.is_empty());
                for (i, arg) in args.iter().enumerate() {
                    Self::print_expr(arg, &new_indent, i == args.len() - 1);
                }
            }
            Expr::Array(items, _) => {
                println!("{}{}Array", indent, prefix);
                let new_indent = format!("{}{}", indent, if is_last { "   " } else { "│  " });
                for (i, item) in items.iter().enumerate() {
                    Self::print_expr(item, &new_indent, i == items.len() - 1);
                }
            }
            Expr::EnumVariant(enum_name, variant, _) => {
                println!("{}{}Enum({}::{})", indent, prefix, enum_name, variant);
            }

            Expr::ComptimeBlock(stmts, ret, _) => {
                println!("{}{}ComptimeBlock", indent, prefix);
                let new_indent = format!("{}{}", indent, if is_last { "   " } else { "│  " });
                Self::print_statements(stmts, &new_indent);
                if let Some(r) = ret {
                    Self::print_expr(r, &new_indent, true);
                }
            }
            _ => println!("{}{}{:?}", indent, prefix, expr), // Fallback
        }
    }
}
