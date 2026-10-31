//===- check submodule - Vx Compiler ----------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// `Send` and `Sync`, derived from the fields of a type. The design is S1 to S4 in
// docs/discussions/cpu_tasks_send_sync_and_shared_atomics.md. `Owned` is derived the same way,
// for M5 in docs/discussions/waiting_tasks_channels_and_task_errors.md.
//
//===----------------------------------------------------------------------===//

use super::super::*;
use std::collections::HashMap;

/// The two traits that say how a value may cross tasks.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ThreadTrait {
    /// The value may move into another task.
    Send,
    /// Two tasks may use the value through `&T` at the same time.
    Sync,
    /// The value holds no reference, so it can outlive the locals of the task that made it.
    Owned,
}

impl ThreadTrait {
    pub(crate) fn from_name(name: &str) -> Option<Self> {
        match name {
            "Send" => Some(Self::Send),
            "Sync" => Some(Self::Sync),
            "Owned" => Some(Self::Owned),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Send => "Send",
            Self::Sync => "Sync",
            Self::Owned => "Owned",
        }
    }
}

impl<'a> TypeChecker<'a> {
    /// Whether `ty` is `Send`, `Sync` or `Owned`.
    ///
    /// A number, a `bool`, a tensor and a function are all three. A raw pointer is only `Owned`,
    /// a reference is never `Owned`, and a closure type is none of them. A struct or an enum is
    /// one when each of its fields is, unless an `unsafe impl` says so for it. `SharedCell` is
    /// never `Sync`. A type parameter that is not known yet counts as all three, because a
    /// generic body is checked again with each concrete type.
    pub(crate) fn has_thread_trait(&mut self, ty: &Type, which: ThreadTrait) -> bool {
        self.why_not_thread_trait(ty, which).is_none()
    }

    /// Why `ty` is not `Send` or `Sync`, as the chain of fields that leads to the type that is
    /// not, or `None` when it is.
    pub(crate) fn why_not_thread_trait(&mut self, ty: &Type, which: ThreadTrait) -> Option<String> {
        let mut visiting = Vec::new();
        self.thread_trait_fails(ty, which, &mut visiting)
    }

    fn thread_trait_fails(
        &mut self,
        ty: &Type,
        which: ThreadTrait,
        visiting: &mut Vec<(Type, ThreadTrait)>,
    ) -> Option<String> {
        match ty {
            Type::Scalar(_)
            | Type::Simd(..)
            | Type::Tensor(..)
            | Type::Matrix
            | Type::Function(..)
            | Type::Generic(..)
            | Type::Const(_)
            | Type::Module(..)
            | Type::Unknown => None,
            // Unsafe code answers for what a raw pointer points at.
            Type::Pointer(..) if which == ThreadTrait::Owned => None,
            Type::Pointer(..) => Some(format!(
                "'{ty}' is a raw pointer, which is not `{}`",
                which.name()
            )),
            Type::Closure(..) => Some(format!(
                "'{ty}' is a closure type, which does not say what the closure captures"
            )),
            Type::Borrow { .. } | Type::Ref(..) if which == ThreadTrait::Owned => {
                Some(format!("'{ty}' is a reference, which is not `Owned`"))
            }
            // Two tasks that hold `&T` share the `T`, so `&T` needs `T : Sync` to move.
            Type::Borrow {
                inner,
                is_mut: false,
                ..
            }
            | Type::Ref(inner, _) => self
                .thread_trait_fails(inner, ThreadTrait::Sync, visiting)
                .map(|why| format!("'{ty}' shares a '{inner}', which is not `Sync`: {why}")),
            Type::Borrow {
                inner,
                is_mut: true,
                ..
            } => self.thread_trait_fails(inner, which, visiting),
            Type::Verified(inner) | Type::Pinned(inner, _) => {
                self.thread_trait_fails(inner, which, visiting)
            }
            Type::Struct(..) | Type::Enum(..) | Type::GenericInstance(..) => {
                self.nominal_thread_trait_fails(ty, which, visiting)
            }
        }
    }

    fn nominal_thread_trait_fails(
        &mut self,
        ty: &Type,
        which: ThreadTrait,
        visiting: &mut Vec<(Type, ThreadTrait)>,
    ) -> Option<String> {
        let (name, args) = match ty {
            Type::Struct(name, _) | Type::Enum(name, _) => (name.clone(), Vec::new()),
            Type::GenericInstance(base, args) => match base.as_ref() {
                Type::Struct(name, _) | Type::Enum(name, _) => (name.clone(), args.clone()),
                _ => return None,
            },
            _ => unreachable!("only a struct, an enum or an instance of one reaches here"),
        };
        // A recursive type reaches itself again through a `Box`. The answer for it then depends
        // on the other fields only, so the inner visit assumes that the trait holds.
        if visiting.contains(&(ty.clone(), which)) {
            return None;
        }
        visiting.push((ty.clone(), which));
        let why = self.nominal_fails_unvisited(ty, &name, &args, which, visiting);
        visiting.pop();
        why
    }

    fn nominal_fails_unvisited(
        &mut self,
        ty: &Type,
        name: &crate::symbol::Symbol,
        args: &[Type],
        which: ThreadTrait,
        visiting: &mut Vec<(Type, ThreadTrait)>,
    ) -> Option<String> {
        if let Some(verdict) = self.unsafe_impl_verdict(ty, which, visiting) {
            return verdict;
        }
        // The value in a `SharedCell` can change through `&`, so two tasks cannot share one (U3).
        if which == ThreadTrait::Sync && name.as_ref() == "SharedCell" {
            return Some(format!(
                "'{ty}' is never `Sync`: code can write its value through a shared reference"
            ));
        }
        for (field, held) in self.held_types(name, args) {
            if let Some(inner) = self.thread_trait_fails(&held, which, visiting) {
                return Some(format!("{field} of '{ty}' has type '{held}', and {inner}"));
            }
        }
        None
    }

    /// What an `unsafe impl Send` or `unsafe impl Sync` for `ty` says: `None` when there is no
    /// such impl, `Some(None)` when one applies, and `Some(Some(why))` when one names the type
    /// but a bound on it fails.
    ///
    /// A bound `T : Send` on the impl is checked here, in the same visit, and not through
    /// `bound_holds`. A new visit there would not know the types that this one is in, and
    /// `struct Node { next : Option<Box<Node>> }` would then never end.
    fn unsafe_impl_verdict(
        &mut self,
        ty: &Type,
        which: ThreadTrait,
        visiting: &mut Vec<(Type, ThreadTrait)>,
    ) -> Option<Option<String>> {
        let blocks = self
            .env
            .impls
            .get(which.name())
            .cloned()
            .unwrap_or_default();
        let mut verdict = None;
        for ib in blocks.iter().filter(|ib| ib.is_unsafe) {
            let mut mapping = HashMap::new();
            if !self.unify_types(&ib.target_type, ty, &mut mapping) {
                continue;
            }
            let mut failed = None;
            'params: for g in &ib.generics {
                let decl::GenericParam::Type { name, bounds } = g else {
                    continue;
                };
                let Some(bound_to) = mapping.get(name).cloned() else {
                    continue;
                };
                for b in bounds {
                    let why = match ThreadTrait::from_name(&b.trait_name) {
                        Some(inner) => self.thread_trait_fails(&bound_to, inner, visiting),
                        None if self.bound_holds(b, &bound_to) => None,
                        None => Some(format!("'{bound_to}' does not implement '{b}'")),
                    };
                    if let Some(why) = why {
                        failed = Some(format!(
                            "`unsafe impl {} for {}` needs `{name} : {b}`, and {why}",
                            which.name(),
                            ib.target_type
                        ));
                        break 'params;
                    }
                }
            }
            match failed {
                None => return Some(None),
                Some(why) => verdict = Some(Some(why)),
            }
        }
        verdict
    }

    /// The fields of the struct or enum `name`, each with a name for an error and its type with
    /// `args` put in for the type parameters.
    fn held_types(&self, name: &crate::symbol::Symbol, args: &[Type]) -> Vec<(String, Type)> {
        let substitute = |generics: &[decl::GenericParam], ty: &Type| -> Type {
            let mapping: HashMap<crate::symbol::Symbol, Type> = generics
                .iter()
                .map(|g| crate::symbol::Symbol::from(g.name()))
                .zip(args.iter().cloned())
                .collect();
            ty.substitute(&mapping)
        };
        if let Some(decl) = self.env.structs.get(name.as_ref()) {
            return decl
                .fields
                .iter()
                .map(|(f, t)| (format!("field '{f}'"), substitute(&decl.generics, t)))
                .collect();
        }
        if let Some(decl) = self.env.enums.get(name.as_ref()) {
            let mut held = Vec::new();
            for (variant, payload) in &decl.variants {
                for t in payload.iter().flatten() {
                    held.push((
                        format!("variant '{variant}'"),
                        substitute(&decl.generics, t),
                    ));
                }
            }
            return held;
        }
        Vec::new()
    }
}
