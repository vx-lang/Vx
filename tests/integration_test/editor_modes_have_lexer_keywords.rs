//===- editor_modes_have_lexer_keywords.rs - modes follow the lexer -*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//! Every keyword of the Vx lexer must also be in the VS Code mode and in the Emacs mode.
//! If this test fails, add the listed words to the mode that does not have them.

use regex::Regex;
use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;
use vxc::lexer::Lexer;

fn read_repository_file(path: &str) -> String {
    let full_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path);
    fs::read_to_string(&full_path).unwrap_or_else(|error| panic!("cannot read {path}: {error}"))
}

fn words_in(text: &str) -> BTreeSet<String> {
    let word = Regex::new(r"[A-Za-z_][A-Za-z0-9_]*").unwrap();
    word.find_iter(text)
        .map(|found| found.as_str().to_string())
        .collect()
}

/// The words in the `m.insert("word", ...)` lines of the keyword table in `src/lexer.rs`.
fn lexer_keywords() -> BTreeSet<String> {
    let source = read_repository_file("src/lexer.rs");
    let insert = Regex::new(r#"m\.insert\("([A-Za-z_][A-Za-z0-9_]*)""#).unwrap();
    let keywords: BTreeSet<String> = insert
        .captures_iter(&source)
        .map(|captures| captures[1].to_string())
        .collect();

    // If the table changes its shape, this test must not pass with an empty list.
    assert!(
        keywords.contains("fn") && keywords.contains("Memory") && keywords.len() > 40,
        "cannot find the keyword table in src/lexer.rs, found only: {keywords:?}"
    );
    for word in &keywords {
        let tokens = Lexer::new(word).tokenize();
        assert!(
            tokens[0].kind.is_keyword(),
            "`{word}` is in the text of the keyword table, but the lexer does not read it as a keyword"
        );
    }
    keywords
}

/// The words in the `"match"` patterns of the VS Code grammar.
fn vscode_words() -> BTreeSet<String> {
    let grammar = read_repository_file("vscode-vx/syntaxes/vx.tmLanguage.json");
    let pattern = Regex::new(r#""match":\s*"((?:[^"\\]|\\.)*)""#).unwrap();
    pattern
        .captures_iter(&grammar)
        .flat_map(|captures| words_in(&captures[1]))
        .collect()
}

/// The words in the lists `vx-keywords` and `vx-builtin-types` of `vx-mode.el`.
fn emacs_words() -> BTreeSet<String> {
    let mode = read_repository_file("emacs-vx/vx-mode.el");
    let mut words = BTreeSet::new();
    for list_name in ["vx-keywords", "vx-builtin-types"] {
        let list = Regex::new(&format!(r"\(defconst {list_name}\s+'\(([^)]*)\)")).unwrap();
        let body = &list
            .captures(&mode)
            .unwrap_or_else(|| panic!("cannot find `{list_name}` in vx-mode.el"))[1];
        words.extend(words_in(body));
    }
    words
}

fn missing_words(keywords: &BTreeSet<String>, mode_words: &BTreeSet<String>) -> Vec<String> {
    keywords.difference(mode_words).cloned().collect()
}

#[test]
fn vscode_mode_has_every_lexer_keyword() {
    let missing = missing_words(&lexer_keywords(), &vscode_words());
    assert!(
        missing.is_empty(),
        "vscode-vx/syntaxes/vx.tmLanguage.json does not have these lexer keywords: {missing:?}"
    );
}

#[test]
fn emacs_mode_has_every_lexer_keyword() {
    let missing = missing_words(&lexer_keywords(), &emacs_words());
    assert!(
        missing.is_empty(),
        "emacs-vx/vx-mode.el does not have these lexer keywords: {missing:?}"
    );
}
