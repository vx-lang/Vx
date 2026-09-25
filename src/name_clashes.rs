//===- name_clashes.rs - Vx Compiler ---------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Two modules of one compile declaring the same top-level name. Every declaration lands in one
// global namespace, so each copy but one is renamed, in the module that declares it and in every
// module that sees it.
//
// A module sees its own declaration of a name first. A module that declares none sees what its
// direct imports see; if they see different declarations, the name is ambiguous there, and an
// error if that module uses it. This is how a local item shadows a glob import in Rust.
//
//===----------------------------------------------------------------------===//
use crate::lexer::{Token, TokenTypeBase};
use crate::symbol::Symbol;
use crate::syntax::Program;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

/// What a name stands for in one module that has to be re-parsed.
#[derive(Debug, Clone, PartialEq)]
pub enum Rename {
    /// A declaration that lost its plain name: every use becomes this.
    To(String),
    /// Two declarations, both seen through imports. Using the name is an error.
    Ambiguous(Vec<Symbol>),
}

/// The renames each module needs, by module name. A module missing here is parsed as written.
pub type Plan = HashMap<Symbol, HashMap<Symbol, Rename>>;

/// Names the compiler looks up by spelling, so the stdlib's declaration cannot be renamed.
const NAMES_THE_COMPILER_USES: &[&str] = &[
    "Option", "Result", "Vec", "String", "Copy", "Closure0", "Closure1", "Closure2", "Closure3",
];

fn module_name(path: &[Symbol]) -> Symbol {
    path.iter()
        .map(|s| s.as_ref())
        .collect::<Vec<_>>()
        .join("::")
        .as_str()
        .into()
}

fn declared_names(p: &Program) -> impl Iterator<Item = &Symbol> {
    p.structs
        .iter()
        .map(|s| &s.name)
        .chain(p.enums.iter().map(|e| &e.name))
        .chain(p.traits.iter().map(|t| &t.name))
        .chain(p.functions.iter().map(|f| &f.name))
}

/// The renamed spelling: the declaring module, then the name. It is still one identifier.
fn renamed(module: &Symbol, name: &Symbol) -> String {
    format!("{}__{}", module.as_ref().replace("::", "_"), name)
}

/// Which modules need re-parsing, and with what. `roots` are the modules nobody imports (the
/// program, `--machine` and `--host` files); one of them keeps its plain name when it declares
/// the clashing name.
pub fn plan(programs: &[Program], roots: &[Symbol]) -> Result<Plan, String> {
    // Sorted, so the copy that keeps its name does not depend on the order modules loaded in.
    let mut declarers: BTreeMap<&Symbol, Vec<&Symbol>> = BTreeMap::new();
    for p in programs {
        for name in declared_names(p).collect::<BTreeSet<_>>() {
            declarers.entry(name).or_default().push(&p.module_path);
        }
    }
    let clashes = declarers.into_iter().filter(|(_, mods)| mods.len() > 1);

    let by_name: HashMap<&Symbol, &Program> =
        programs.iter().map(|p| (&p.module_path, p)).collect();
    let mut plan = Plan::new();
    for (name, mut mods) in clashes {
        mods.sort();
        let root_declarers: Vec<&&Symbol> = mods.iter().filter(|m| roots.contains(m)).collect();
        if root_declarers.len() > 1 {
            // Peer inputs, such as a `--machine` file and the program; the declaration check
            // reports those.
            continue;
        }
        let keeper = root_declarers.first().map_or(mods[0], |m| **m);
        if let Some(lost) = mods.iter().find(|m| **m != keeper) {
            if NAMES_THE_COMPILER_USES.contains(&name.as_ref()) {
                return Err(format!(
                    "`{name}` is declared by both `{keeper}` and `{lost}`; the compiler looks up \
                     the standard library's `{name}` by name, so no other module may declare one"
                ));
            }
        }
        let mut seen: HashMap<&Symbol, BTreeSet<&Symbol>> = HashMap::new();
        for p in programs {
            let declarers = sees(
                &p.module_path,
                name,
                &by_name,
                &mut seen,
                &mut HashSet::new(),
            );
            let rename = match declarers.len() {
                0 => continue,
                1 => {
                    let d = *declarers.iter().next().unwrap();
                    if d == keeper {
                        continue;
                    }
                    Rename::To(renamed(d, name))
                }
                _ => Rename::Ambiguous(declarers.into_iter().cloned().collect()),
            };
            plan.entry(p.module_path.clone())
                .or_default()
                .insert(name.clone(), rename);
        }
    }
    Ok(plan)
}

/// The declarations of `name` that `module` sees.
fn sees<'p>(
    module: &'p Symbol,
    name: &Symbol,
    by_name: &HashMap<&'p Symbol, &'p Program>,
    memo: &mut HashMap<&'p Symbol, BTreeSet<&'p Symbol>>,
    visiting: &mut HashSet<&'p Symbol>,
) -> BTreeSet<&'p Symbol> {
    if let Some(done) = memo.get(module) {
        return done.clone();
    }
    // An interface-only import, or a cycle back to a module still being worked out.
    let Some(p) = by_name.get(module) else {
        return BTreeSet::new();
    };
    if !visiting.insert(module) {
        return BTreeSet::new();
    }
    let out = if declared_names(p).any(|n| n == name) {
        BTreeSet::from([&p.module_path])
    } else {
        let mut out = BTreeSet::new();
        for import in &p.imports {
            let imported = module_name(&import.path);
            if let Some((key, _)) = by_name.get_key_value(&imported) {
                out.extend(sees(key, name, by_name, memo, visiting));
            }
        }
        out
    };
    visiting.remove(module);
    memo.insert(module, out.clone());
    out
}

#[derive(Clone, Copy, PartialEq)]
enum Scope {
    Paren,
    Block,
    /// An `enum` body: a name at this level is a variant.
    EnumBody,
    /// An `impl` or `trait` body: `fn name` here is a method.
    ItemBody,
}

fn is_trivia<S, C>(kind: &TokenTypeBase<S, C>) -> bool {
    matches!(
        kind,
        TokenTypeBase::Comment(_) | TokenTypeBase::DocComment(_) | TokenTypeBase::Whitespace(_)
    )
}

/// Apply one module's renames to its tokens. Only a use of a top-level name is renamed: a field
/// or method (after `.` or `::`), a variant, a method declaration, a struct field and an import
/// path keep their spelling.
pub fn rename_tokens<'a>(
    tokens: &mut [Token<'a>],
    renames: &'a HashMap<Symbol, Rename>,
) -> Result<(), String> {
    let mut scopes: Vec<Scope> = Vec::new();
    let mut next_brace: Option<Scope> = None;
    let mut in_import = false;
    let mut prev: Option<usize> = None;
    for i in 0..tokens.len() {
        if is_trivia(&tokens[i].kind) {
            continue;
        }
        let top_level = scopes.is_empty();
        match &tokens[i].kind {
            TokenTypeBase::Import => in_import = true,
            TokenTypeBase::Semicolon => in_import = false,
            TokenTypeBase::Enum if top_level => next_brace = Some(Scope::EnumBody),
            TokenTypeBase::Impl | TokenTypeBase::Trait if top_level => {
                next_brace = Some(Scope::ItemBody)
            }
            TokenTypeBase::LeftBrace => scopes.push(next_brace.take().unwrap_or(Scope::Block)),
            TokenTypeBase::LeftParen | TokenTypeBase::LeftBracket => scopes.push(Scope::Paren),
            TokenTypeBase::RightBrace | TokenTypeBase::RightParen | TokenTypeBase::RightBracket => {
                scopes.pop();
            }
            TokenTypeBase::Identifier(text) if !in_import => {
                let Some(rename) = renames.get(*text) else {
                    prev = Some(i);
                    continue;
                };
                let prev_kind = prev.map(|p| &tokens[p].kind);
                let next_kind = tokens[i + 1..]
                    .iter()
                    .map(|t| &t.kind)
                    .find(|k| !is_trivia(k));
                let scope = scopes.last().copied();
                let member = matches!(
                    prev_kind,
                    Some(TokenTypeBase::Dot | TokenTypeBase::DoubleColon)
                );
                let variant = scope == Some(Scope::EnumBody);
                let method =
                    scope == Some(Scope::ItemBody) && matches!(prev_kind, Some(TokenTypeBase::Fn));
                let field = scope == Some(Scope::Block)
                    && matches!(next_kind, Some(TokenTypeBase::Colon))
                    && matches!(
                        prev_kind,
                        Some(TokenTypeBase::LeftBrace | TokenTypeBase::Comma)
                    );
                if !(member || variant || method || field) {
                    match rename {
                        Rename::To(new) => tokens[i].kind = TokenTypeBase::Identifier(new),
                        Rename::Ambiguous(from) => {
                            let from: Vec<String> = from.iter().map(|m| format!("`{m}`")).collect();
                            return Err(format!(
                                "line {}: `{text}` is declared by both {}, and this module \
                                 sees both through its imports",
                                tokens[i].line,
                                from.join(" and ")
                            ));
                        }
                    }
                }
            }
            _ => {}
        }
        prev = Some(i);
    }
    Ok(())
}
