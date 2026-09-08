//===- macro_rules.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions for macros and compile-time procedural metaprogramming.
//
//===----------------------------------------------------------------------===//

use crate::lexer::OwnedToken;

#[derive(Debug, Clone, PartialEq)]
pub enum TokenTree {
    Token(OwnedToken),
    Group(Vec<TokenTree>),
    Delimited(Delimiter, Vec<TokenTree>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Delimiter {
    Parenthesis, // ()
    Brace,       // {}
    Bracket,     // []
}
