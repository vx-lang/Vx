//===- decl_check submodule - Vx Compiler -----------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Only the file that declares a struct or an enum writes a plain `impl` of it. Another file
// writes `extern impl`, so that a review finds each impl of a type outside its file.
//
//===----------------------------------------------------------------------===//

use super::super::*;

/// The file name of `file`, for a message.
fn file_name(file: &str) -> String {
    let name = std::path::Path::new(file).file_name();
    name.map_or(file.to_string(), |n| n.to_string_lossy().into_owned())
}

/// The file that declares a type: a file of the program, or `core` and `std` for a built-in
/// type.
enum Home {
    File(crate::symbol::Symbol),
    Library,
}

impl TypeChecker<'_> {
    /// E3066 for a plain `impl` outside the file of its type, for an `extern impl` in that file,
    /// and for an `extern impl` that a macro wrote.
    pub fn check_impl_files(&mut self) {
        let mut blocks: Vec<&ImplBlock> = self.env.impls.values().flatten().copied().collect();
        // The impl table is a hash map. Sorted, the errors come in the same order each time.
        blocks.sort_by_key(|ib| {
            (
                ib.place.file.clone(),
                ib.place.span.line,
                ib.place.span.column,
            )
        });
        let outside = self.errors.here();
        for ib in blocks {
            if let Some(message) = self.impl_file_error(ib) {
                self.errors.point_at(&ib.place, &outside);
                self.errors.error_with_code(
                    crate::diagnostic::DiagnosticCode::E3066,
                    message,
                    None,
                );
            }
        }
        self.errors.point_back(outside);
    }

    /// The name of the type that `ty` names, as in `impl X for Name` or `Name<..>`, and the file
    /// that declares it.
    fn home_of(&self, ty: &Type) -> Option<(String, Home)> {
        match ty {
            Type::Struct(name, _) | Type::Enum(name, _) => {
                let place = match self.env.structs.get(name) {
                    Some(decl) => &decl.place,
                    None => &self.env.enums.get(name)?.place,
                };
                Some((name.to_string(), Home::File(place.file.clone()?)))
            }
            Type::GenericInstance(base, _) => self.home_of(base),
            Type::Scalar(_) | Type::Simd(..) => Some((ty.to_string(), Home::Library)),
            Type::Tensor(..) => Some(("Tensor".to_string(), Home::Library)),
            _ => None,
        }
    }

    fn impl_file_error(&self, ib: &ImplBlock) -> Option<String> {
        let target = &ib.target_type;
        let header = match &ib.trait_name {
            Some(t) if t.as_ref() != "_inherent" => format!("impl {t} for {target}"),
            _ => format!("impl {target}"),
        };
        if ib.is_extern && ib.place.span.expansion != 0 {
            return Some(format!(
                "a macro wrote `extern {header}`. A macro cannot write `extern impl`: write it in \
                 the file, so that a review finds it"
            ));
        }
        let file = ib.place.file.as_ref()?;
        let here = file_name(file);
        let (name, home) = self.home_of(target)?;
        match (home, ib.is_extern) {
            (Home::File(home), false) if home != *file => {
                let home = file_name(&home);
                Some(format!(
                    "`{header}` is in `{here}`, and `{home}` declares `{name}`. Move the impl to \
                     `{home}`, or write `extern impl` here"
                ))
            }
            (Home::File(home), true) if home == *file => Some(format!(
                "`{here}` declares `{name}`, so its impl here is a plain `impl`. Remove `extern`"
            )),
            (Home::Library, false) if !self.env.library_files.contains(file) => Some(format!(
                "`{header}` is in `{here}`, and `{name}` is a type of `core` and `std`. Write \
                 `extern impl` here"
            )),
            (Home::Library, true) if self.env.library_files.contains(file) => Some(format!(
                "`core` and `std` declare `{name}`, so its impl in `{here}` is a plain `impl`. \
                 Remove `extern`"
            )),
            _ => None,
        }
    }
}
