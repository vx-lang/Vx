//===- check submodule - Vx Compiler ----------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A `drop` runs with cancellation off, so a cancel cannot stop a wait in it. A `send` or `recv`
// there waits until the time limit of the `drop` when the other end never acts. W1036 warns
// about such a call, and the fix is `send_for` or `recv_for`, or a call of `within` (F6 and F6a
// in docs/discussions/waiting_tasks_channels_and_task_errors.md). A wait in a closure literal
// written as the body of `within` has the limit of the call, so it gets no warning. A closure in
// a variable gets the warning, because the check cannot see that the variable goes to `within`. `wait_limit` runs before cancellation is off,
// and it should return a time and not wait, so W1036 warns about a wait in it too.
// It is a warning, not an error, because the check does not look into the functions that the
// `drop` calls.
//
//===----------------------------------------------------------------------===//

use super::super::*;

impl TypeChecker<'_> {
    /// Whether `func` is the `drop` of a `Drop` impl: the `drop` that its receiver type uses.
    pub(crate) fn is_drop_impl(&mut self, func: &Function) -> bool {
        if !func.name.ends_with("drop") && !func.name.contains("$drop$") {
            return false;
        }
        self.is_drop_impl_method(func, "drop")
    }

    /// Whether `func` is the `wait_limit` of a `Drop` impl.
    pub(crate) fn is_wait_limit_impl(&mut self, func: &Function) -> bool {
        if !func.name.contains("wait_limit") {
            return false;
        }
        self.is_drop_impl_method(func, "wait_limit")
    }

    /// Whether `func` is the method `method` of the `Drop` impl of its receiver type.
    fn is_drop_impl_method(&mut self, func: &Function, method: &str) -> bool {
        let Some((_, receiver)) = func.params.first() else {
            return false;
        };
        let ty = match receiver {
            Type::Borrow { inner, .. } | Type::Ref(inner, _) => inner.as_ref().clone(),
            ty => ty.clone(),
        };
        self.drops_impl_method(&ty, method)
            .is_some_and(|name| name.as_ref() == func.name.as_ref())
    }

    /// W1036 when the `drop` of a `Drop` impl calls `method` on a value of type `base`, and the
    /// call is a channel wait with no time limit.
    pub(crate) fn warn_wait_in_drop(&mut self, base: &Type, method: &str, span: &Span) {
        if !(self.in_drop || self.in_wait_limit) || self.within_depth > 0 || self.speculating {
            return;
        }
        let timed = match method {
            "send" => "send_for",
            "recv" => "recv_for",
            _ => return,
        };
        let base = match base {
            Type::Borrow { inner, .. } | Type::Ref(inner, _) => inner.as_ref(),
            ty => ty,
        };
        let name = match base {
            Type::Struct(n, _) => n.clone(),
            Type::GenericInstance(b, _) => match b.as_ref() {
                Type::Struct(n, _) => n.clone(),
                _ => return,
            },
            _ => return,
        };
        let channel_end = match method {
            "send" => "Sender",
            _ => "Receiver",
        };
        if name.split('<').next() != Some(channel_end) {
            return;
        }
        if self.in_wait_limit {
            self.errors.warn(
                crate::diagnostic::DiagnosticCode::W1036,
                format!(
                    "'{method}' waits in 'wait_limit'. 'wait_limit' runs before the drop, to say \
                     how long its waits can take, so it should return a time and not wait"
                ),
                Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
            );
            return;
        }
        self.errors.warn(
            crate::diagnostic::DiagnosticCode::W1036,
            format!(
                "'{method}' waits with no time limit of its own in a 'drop'. A drop runs with \
                 cancellation off, so if the other end of the channel never acts, this wait \
                 holds up the drop until the limit of the drop: its type's 'wait_limit', 10 \
                 seconds by default. Use '{timed}' with a time limit, or put the wait in the \
                 body of 'within'"
            ),
            Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
        );
    }
}
