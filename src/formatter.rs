//===- formatter.rs - Vx Compiler ------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides the core logic for the Vx code formatter.
// It iterates over token streams produced by the lexer and applies intelligent
// spacing, indentation, and line-wrapping rules to generate idiomatic and
// readable Vx source code.
//
//===----------------------------------------------------------------------===//
use crate::lexer::{Lexer, Token, TokenType};
use std::collections::{HashMap, HashSet};

pub fn format_file(content: &str, indent_spaces: usize) -> String {
    let mut lexer = Lexer::new_with_comments(content);
    let tokens = lexer.tokenize();

    if let Some(token) = tokens.first() {
        if let TokenType::Comment(c) | TokenType::DocComment(c) = &token.kind {
            if c.contains("//! vx-format: OFF") {
                return content.to_string();
            }
        }
    }

    // A file that does not lex, such as one that ends inside a string, is left as it is: an
    // unknown token is written back as one character, which would lose the rest of its text.
    if tokens
        .iter()
        .any(|t| matches!(t.kind, TokenType::Unknown(_)))
    {
        return content.to_string();
    }

    let tokens = normalize_and_expand_blocks(tokens, indent_spaces);
    let tokens = adjust_spacing(tokens);
    emit_formatted_string(tokens, indent_spaces, content.len())
}

fn normalize_and_expand_blocks(mut tokens: Vec<Token>, indent_spaces: usize) -> Vec<Token> {
    // Pass 1a: Ensure spaces before `{`
    let mut spaced_tokens: Vec<Token> = Vec::with_capacity(tokens.len() + 10);
    for token in &tokens {
        if let TokenType::LeftBrace = token.kind {
            if !spaced_tokens.is_empty()
                && !matches!(spaced_tokens.last().unwrap().kind, TokenType::Whitespace(_))
            {
                spaced_tokens.push(Token {
                    kind: TokenType::Whitespace(" "),
                    line: token.line,
                    column: token.column,
                    length: 1,
                    expansion: token.expansion,
                });
            }
        }
        spaced_tokens.push(token.clone());
    }
    tokens = spaced_tokens;

    // Struct literals, match arms and short if values get their own layout. The braces that
    // must stay on one line skip the block layout below.
    let (laid_out, inline) = lay_out_literals_and_arms(tokens, indent_spaces);
    tokens = laid_out;

    // Pass 1b: Match braces in O(N) using a stack
    let mut stack = Vec::new();
    let mut left_to_right = HashMap::new();
    let mut right_to_left = HashMap::new();
    for (idx, token) in tokens.iter().enumerate() {
        match token.kind {
            TokenType::LeftBrace => stack.push(idx),
            TokenType::RightBrace => {
                if let Some(lb_idx) = stack.pop() {
                    left_to_right.insert(lb_idx, idx);
                    right_to_left.insert(idx, lb_idx);
                }
            }
            _ => {}
        }
    }

    // Pass 1c: Process blocks (collapsing `unsafe` or expanding others)
    let mut new_tokens: Vec<Token> = Vec::with_capacity(tokens.len() + 20);
    let mut i = 0;
    while i < tokens.len() {
        let kept_inline = match tokens[i].kind {
            TokenType::LeftBrace => inline.contains(&i),
            TokenType::RightBrace => right_to_left.get(&i).is_some_and(|lb| inline.contains(lb)),
            _ => false,
        };
        if kept_inline {
            new_tokens.push(tokens[i].clone());
            i += 1;
            continue;
        }
        if let TokenType::LeftBrace = tokens[i].kind {
            if let Some(&rb_idx) = left_to_right.get(&i) {
                let lb_idx = i;
                let has_newline = tokens[(lb_idx + 1)..rb_idx].iter().any(|t| {
                    if let TokenType::Whitespace(ws) = t.kind {
                        ws.contains('\n')
                    } else {
                        false
                    }
                });

                let mut is_unsafe_block = false;
                for j in (0..new_tokens.len()).rev() {
                    match new_tokens[j].kind {
                        TokenType::Whitespace(_)
                        | TokenType::Comment(_)
                        | TokenType::DocComment(_) => continue,
                        TokenType::Unsafe => {
                            is_unsafe_block = true;
                            break;
                        }
                        _ => break,
                    }
                }

                // A line comment inside runs to the end of its line, so a block holding one is
                // never put on one line: the comment would swallow the code after it.
                let holds_comment = tokens[lb_idx..=rb_idx]
                    .iter()
                    .any(|t| matches!(t.kind, TokenType::Comment(_) | TokenType::DocComment(_)));
                if is_unsafe_block && !holds_comment {
                    // Try to measure the total length if we collapsed it to a single line
                    let mut collapsed_len = 0;
                    for j in (0..new_tokens.len()).rev() {
                        if let TokenType::Whitespace(ws) = new_tokens[j].kind {
                            if let Some(pos) = ws.rfind('\n') {
                                collapsed_len += ws.len() - pos - 1;
                                break;
                            } else {
                                collapsed_len += new_tokens[j].length;
                            }
                        } else {
                            collapsed_len += new_tokens[j].length;
                        }
                    }
                    for t in &tokens[lb_idx..=rb_idx] {
                        if let TokenType::Whitespace(ws) = t.kind {
                            if ws.contains('\n') {
                                collapsed_len += 1;
                            } else {
                                collapsed_len += t.length;
                            }
                        } else {
                            collapsed_len += t.length;
                        }
                    }

                    if collapsed_len + 2 <= 80 {
                        // Collapse it! Replace all newlines inside with spaces.
                        new_tokens.push(tokens[lb_idx].clone());

                        if !matches!(tokens[lb_idx + 1].kind, TokenType::Whitespace(_)) {
                            new_tokens.push(Token {
                                kind: TokenType::Whitespace(" "),
                                line: tokens[lb_idx].line,
                                column: tokens[lb_idx].column,
                                length: 1,
                                expansion: tokens[lb_idx].expansion,
                            });
                        }

                        for token in &tokens[(lb_idx + 1)..rb_idx] {
                            let mut t = token.clone();
                            if let TokenType::Whitespace(ref mut ws) = t.kind {
                                if ws.contains('\n') {
                                    *ws = " ";
                                }
                            }
                            new_tokens.push(t);
                        }

                        if !matches!(new_tokens.last().unwrap().kind, TokenType::Whitespace(_)) {
                            new_tokens.push(Token {
                                kind: TokenType::Whitespace(" "),
                                line: tokens[rb_idx].line,
                                column: tokens[rb_idx].column,
                                length: 1,
                                expansion: tokens[rb_idx].expansion,
                            });
                        }

                        new_tokens.push(tokens[rb_idx].clone());
                        i = rb_idx + 1;
                        continue;
                    }
                }

                let has_non_ws = tokens[(lb_idx + 1)..rb_idx].iter().any(|t| {
                    !matches!(
                        t.kind,
                        TokenType::Whitespace(_) | TokenType::Comment(_) | TokenType::DocComment(_)
                    )
                });

                if !has_newline && has_non_ws {
                    new_tokens.push(tokens[lb_idx].clone());

                    if matches!(tokens[lb_idx + 1].kind, TokenType::Whitespace(_)) {
                        let mut t = tokens[lb_idx + 1].clone();
                        t.kind = TokenType::Whitespace("\n");
                        new_tokens.push(t);
                        i = lb_idx + 2;
                    } else {
                        new_tokens.push(Token {
                            kind: TokenType::Whitespace("\n"),
                            line: 0,
                            column: 0,
                            length: 1,
                            expansion: 0,
                        });
                        i = lb_idx + 1;
                    }
                    continue;
                }
            }
        } else if let TokenType::RightBrace = tokens[i].kind {
            if let Some(&lb_idx) = right_to_left.get(&i) {
                let has_newline = tokens[(lb_idx + 1)..i].iter().any(|t| {
                    if let TokenType::Whitespace(ws) = t.kind {
                        ws.contains('\n')
                    } else {
                        false
                    }
                });

                let has_non_ws = tokens[(lb_idx + 1)..i].iter().any(|t| {
                    !matches!(
                        t.kind,
                        TokenType::Whitespace(_) | TokenType::Comment(_) | TokenType::DocComment(_)
                    )
                });

                if !has_newline && has_non_ws {
                    let needs_newline = match new_tokens.last() {
                        Some(t) => {
                            if let TokenType::Whitespace(ws) = t.kind {
                                !ws.contains('\n')
                            } else {
                                true
                            }
                        }
                        None => true,
                    };

                    if needs_newline {
                        if let Some(t) = new_tokens.last_mut() {
                            if let TokenType::Whitespace(_) = t.kind {
                                t.kind = TokenType::Whitespace("\n");
                            } else {
                                new_tokens.push(Token {
                                    kind: TokenType::Whitespace("\n"),
                                    line: 0,
                                    column: 0,
                                    length: 1,
                                    expansion: 0,
                                });
                            }
                        }
                    }
                }
            }
        }

        new_tokens.push(tokens[i].clone());
        i += 1;
    }

    new_tokens
}

/// A struct literal or a short if value stays on one line only when the line fits in this many
/// columns.
const MAX_LINE: usize = 100;
/// A struct literal stays on one line only when its fields take at most this many characters.
const MAX_INLINE_FIELDS: usize = 40;

fn is_trivia(kind: &TokenType) -> bool {
    matches!(
        kind,
        TokenType::Whitespace(_) | TokenType::Comment(_) | TokenType::DocComment(_)
    )
}

fn whitespace<'a>(text: &'static str, near: &Token<'a>) -> Token<'a> {
    Token {
        kind: TokenType::Whitespace(text),
        line: near.line,
        column: near.column,
        length: text.len(),
        expansion: near.expansion,
    }
}

/// What a `{` opens, for the layout of its contents.
#[derive(Clone, Copy, PartialEq)]
enum Group {
    /// `Point { x: 1, y: 2 }`: on one line when short, else one field on each line.
    StructLiteral,
    /// The arms of a `match`: each arm starts on a new line.
    MatchArms,
    Other,
}

/// The brackets of a token stream: where each `(`, `[` and `{` closes, and the innermost one
/// that holds each token.
struct Brackets {
    close: Vec<Option<usize>>,
    open: Vec<Option<usize>>,
    parent: Vec<Option<usize>>,
}

impl Brackets {
    fn of(tokens: &[Token]) -> Self {
        let n = tokens.len();
        let (mut close, mut open, mut parent) = (vec![None; n], vec![None; n], vec![None; n]);
        let mut stack: Vec<usize> = Vec::new();
        for (i, t) in tokens.iter().enumerate() {
            match t.kind {
                TokenType::RightParen | TokenType::RightBracket | TokenType::RightBrace => {
                    if let Some(o) = stack.pop() {
                        close[o] = Some(i);
                        open[i] = Some(o);
                    }
                    parent[i] = stack.last().copied();
                }
                TokenType::LeftParen | TokenType::LeftBracket | TokenType::LeftBrace => {
                    parent[i] = stack.last().copied();
                    stack.push(i);
                }
                _ => parent[i] = stack.last().copied(),
            }
        }
        Brackets {
            close,
            open,
            parent,
        }
    }
}

fn next_sig(tokens: &[Token], from: usize) -> Option<usize> {
    (from..tokens.len()).find(|&k| !is_trivia(&tokens[k].kind))
}

fn prev_sig(tokens: &[Token], before: usize) -> Option<usize> {
    (0..before).rev().find(|&k| !is_trivia(&tokens[k].kind))
}

/// The tokens of the statement header before `at`, nearest first. The scan stops at a `;`, a
/// `}`, a `=>` or a bracket that is still open, and steps over each closed `( )` and `[ ]`.
fn header_before(tokens: &[Token], b: &Brackets, at: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut k = at;
    while let Some(p) = prev_sig(tokens, k) {
        match tokens[p].kind {
            TokenType::Semicolon
            | TokenType::RightBrace
            | TokenType::FatArrow
            | TokenType::LeftParen
            | TokenType::LeftBracket
            | TokenType::LeftBrace => break,
            TokenType::RightParen | TokenType::RightBracket => match b.open[p] {
                Some(o) => k = o,
                None => break,
            },
            _ => {
                out.push(p);
                k = p;
            }
        }
    }
    out
}

fn classify(tokens: &[Token], b: &Brackets, lb: usize) -> Group {
    let Some(rb) = b.close[lb] else {
        return Group::Other;
    };
    let header = header_before(tokens, b, lb);
    if header
        .iter()
        .any(|&k| matches!(tokens[k].kind, TokenType::Match))
    {
        return Group::MatchArms;
    }
    // The first word of a declaration, after `pub`, `unsafe` or `comptime`.
    let first = header.iter().rev().copied().find(|&k| {
        !matches!(
            tokens[k].kind,
            TokenType::Pub | TokenType::Unsafe | TokenType::Comptime
        )
    });
    let declares = first.is_some_and(|k| match tokens[k].kind {
        TokenType::Struct
        | TokenType::Enum
        | TokenType::Trait
        | TokenType::Impl
        | TokenType::Fn
        | TokenType::Extern
        | TokenType::MacroRules
        | TokenType::TypeKw => true,
        // `Memory HBM {` and `Topology Gpu {` in a fleet file.
        TokenType::Memory | TokenType::Topology => next_sig(tokens, k + 1)
            .is_some_and(|n| matches!(tokens[n].kind, TokenType::Identifier(_))),
        _ => false,
    });
    let named = prev_sig(tokens, lb).is_some_and(|p| {
        matches!(
            tokens[p].kind,
            TokenType::Identifier(_) | TokenType::RightAngle
        )
    });
    if declares || !named {
        return Group::Other;
    }
    // The fields: `name: value` or `name,`, with no statement and no match arm among them.
    let Some(f) = next_sig(tokens, lb + 1).filter(|&f| f < rb) else {
        return Group::Other;
    };
    let field = matches!(tokens[f].kind, TokenType::Identifier(_))
        && next_sig(tokens, f + 1)
            .is_some_and(|c| matches!(tokens[c].kind, TokenType::Colon | TokenType::Comma));
    let statements = (lb + 1..rb).any(|k| {
        b.parent[k] == Some(lb)
            && matches!(tokens[k].kind, TokenType::Semicolon | TokenType::FatArrow)
    });
    if field && !statements {
        Group::StructLiteral
    } else {
        Group::Other
    }
}

/// The length of tokens `lo..=hi` on one line, with one space wherever the source has
/// whitespace.
fn inline_len(tokens: &[Token], lo: usize, hi: usize) -> usize {
    tokens[lo..=hi]
        .iter()
        .map(|t| match t.kind {
            TokenType::Whitespace(_) => 1,
            _ => t.length,
        })
        .sum()
}

/// Whether tokens `lo..=hi` can go on one line: no comment, no `;`, and every `{` among them in
/// `allowed`.
fn can_join(tokens: &[Token], lo: usize, hi: usize, allowed: impl Fn(usize) -> bool) -> bool {
    (lo..=hi).all(|k| match tokens[k].kind {
        TokenType::Comment(_) | TokenType::DocComment(_) | TokenType::Semicolon => false,
        TokenType::LeftBrace => allowed(k),
        _ => true,
    })
}

/// The length on one line of the tokens after `at`, up to the end of the line. A `;`, or a `,`
/// of a group in `split`, ends the line too, because a new line follows it.
fn rest_of_line(
    tokens: &[Token],
    b: &Brackets,
    split: &HashSet<usize>,
    in_types: &HashSet<usize>,
    at: usize,
) -> usize {
    let mut len = 0;
    for (k, t) in tokens.iter().enumerate().skip(at) {
        match t.kind {
            TokenType::Whitespace(ws) if ws.contains('\n') => break,
            TokenType::Whitespace(_) => len += 1,
            TokenType::Semicolon => return len + 1,
            TokenType::Comma
                if !in_types.contains(&k) && b.parent[k].is_some_and(|p| split.contains(&p)) =>
            {
                return len + 1
            }
            _ => len += t.length,
        }
    }
    len
}

/// The commas inside type arguments, as in `Result<u64, E>::Ok(n)`. They do not end an item of
/// a group. A `<` right after a name or a `::` opens type arguments. When a token that cannot be
/// in a type comes before the matching `>`, the `<` was a comparison.
fn commas_in_type_arguments(tokens: &[Token]) -> HashSet<usize> {
    let mut found = HashSet::new();
    let mut k = 1;
    while k < tokens.len() {
        let opens = matches!(tokens[k].kind, TokenType::LeftAngle)
            && matches!(
                tokens[k - 1].kind,
                TokenType::Identifier(_) | TokenType::DoubleColon
            );
        if !opens {
            k += 1;
            continue;
        }
        let mut depth = 0usize;
        let mut commas = Vec::new();
        let mut end = None;
        for (j, t) in tokens.iter().enumerate().skip(k) {
            match t.kind {
                TokenType::LeftAngle => depth += 1,
                TokenType::RightAngle => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(j);
                        break;
                    }
                }
                TokenType::Comma => commas.push(j),
                TokenType::Semicolon
                | TokenType::LeftBrace
                | TokenType::RightBrace
                | TokenType::FatArrow
                | TokenType::Equals
                | TokenType::AndAnd
                | TokenType::OrOr => break,
                _ => {}
            }
        }
        match end {
            Some(e) => {
                found.extend(commas);
                k = e + 1;
            }
            None => k += 1,
        }
    }
    found
}

/// Lays out struct literals, the arms of each `match`, and short if values. Returns the tokens,
/// and the `{` of each group that must stay on one line.
fn lay_out_literals_and_arms<'a>(
    tokens: Vec<Token<'a>>,
    indent: usize,
) -> (Vec<Token<'a>>, HashSet<usize>) {
    let b = Brackets::of(&tokens);
    let groups: Vec<Group> = (0..tokens.len())
        .map(|k| match tokens[k].kind {
            TokenType::LeftBrace => classify(&tokens, &b, k),
            _ => Group::Other,
        })
        .collect();
    let in_types = commas_in_type_arguments(&tokens);

    let mut out: Vec<Token<'a>> = Vec::with_capacity(tokens.len() + tokens.len() / 8);
    let mut inline = HashSet::new();
    // The `{` of each group whose items go on lines of their own.
    let mut split: HashSet<usize> = HashSet::new();
    let mut need_newline = false;
    let mut depth = 0usize;
    let mut line_depth = 0usize;

    // The length of the current output line, with the indentation that the emitter will give it.
    let line_len = |out: &[Token], line_depth: usize| -> usize {
        let mut len = line_depth * indent;
        for t in out.iter().rev() {
            match t.kind {
                TokenType::Whitespace(ws) if ws.contains('\n') => break,
                TokenType::Whitespace(ws) => len += ws.len(),
                _ => len += t.length,
            }
        }
        len
    };
    // A line break before the next token, unless one is there already.
    let break_line = |out: &mut Vec<Token<'a>>, near: &Token<'a>| match out.last_mut() {
        Some(Token {
            kind: TokenType::Whitespace(ws),
            ..
        }) if ws.contains('\n') => {}
        Some(
            t @ Token {
                kind: TokenType::Whitespace(_),
                ..
            },
        ) => t.kind = TokenType::Whitespace("\n"),
        Some(Token {
            kind: TokenType::Comment(_) | TokenType::DocComment(_),
            ..
        })
        | None => {}
        Some(_) => out.push(whitespace("\n", near)),
    };
    // Tokens `lo..=hi` on one line: one space wherever the source has whitespace, one space
    // inside each `{ }`, and no comma before a `}`.
    let join = |out: &mut Vec<Token<'a>>, inline: &mut HashSet<usize>, lo: usize, hi: usize| {
        let mut last: Option<usize> = None;
        for k in lo..=hi {
            let t = &tokens[k];
            if matches!(t.kind, TokenType::Whitespace(_)) {
                continue;
            }
            if matches!(t.kind, TokenType::Comma)
                && next_sig(&tokens, k + 1)
                    .is_some_and(|n| n <= hi && matches!(tokens[n].kind, TokenType::RightBrace))
            {
                continue;
            }
            if let Some(p) = last {
                let spaced = matches!(tokens[p].kind, TokenType::LeftBrace)
                    || matches!(t.kind, TokenType::RightBrace)
                    || (p + 1..k).any(|w| matches!(tokens[w].kind, TokenType::Whitespace(_)));
                if spaced {
                    out.push(whitespace(" ", t));
                }
            }
            if matches!(t.kind, TokenType::LeftBrace) {
                inline.insert(out.len());
            }
            out.push(t.clone());
            last = Some(k);
        }
    };

    let mut i = 0;
    while i < tokens.len() {
        let t = &tokens[i];

        if need_newline {
            need_newline = false;
            match t.kind {
                TokenType::Whitespace(ws) => {
                    // A comment after the comma stays on its line.
                    let comment_next = next_sig_or_comment(&tokens, i + 1).is_some_and(|n| {
                        matches!(
                            tokens[n].kind,
                            TokenType::Comment(_) | TokenType::DocComment(_)
                        )
                    });
                    if ws.contains('\n') || comment_next {
                        out.push(t.clone());
                    } else {
                        out.push(whitespace("\n", t));
                    }
                    line_depth = depth;
                    i += 1;
                    continue;
                }
                TokenType::Comment(_) | TokenType::DocComment(_) => {}
                _ => {
                    break_line(&mut out, t);
                    line_depth = depth;
                }
            }
        }

        match t.kind {
            TokenType::If => {
                if let Some(end) = short_if_value(&tokens, &b, i) {
                    let fits = line_len(&out, line_depth)
                        + inline_len(&tokens, i, end)
                        + rest_of_line(&tokens, &b, &split, &in_types, end + 1)
                        <= MAX_LINE;
                    if fits {
                        join(&mut out, &mut inline, i, end);
                        i = end + 1;
                        continue;
                    }
                }
            }
            TokenType::LeftBrace => {
                let rb = b.close[i];
                match (groups[i], rb) {
                    (Group::StructLiteral, Some(rb)) => {
                        let literal = |k: usize| groups[k] == Group::StructLiteral;
                        let fields = inline_len(&tokens, i, rb).saturating_sub(4);
                        let fits = can_join(&tokens, i, rb, literal)
                            && fields <= MAX_INLINE_FIELDS
                            && line_len(&out, line_depth)
                                + inline_len(&tokens, i, rb)
                                + rest_of_line(&tokens, &b, &split, &in_types, rb + 1)
                                <= MAX_LINE;
                        if fits {
                            join(&mut out, &mut inline, i, rb);
                            i = rb + 1;
                            continue;
                        }
                        split.insert(i);
                        need_newline = true;
                    }
                    (Group::MatchArms, Some(_)) => {
                        split.insert(i);
                        need_newline = true;
                    }
                    _ => {}
                }
                depth += 1;
            }
            TokenType::RightBrace => {
                if let Some(lb) = b.open[i] {
                    if split.contains(&lb) {
                        break_line(&mut out, t);
                        line_depth = depth.saturating_sub(1);
                    }
                }
                depth = depth.saturating_sub(1);
            }
            TokenType::Whitespace(ws) if ws.contains('\n') => line_depth = depth,
            _ => {}
        }

        out.push(t.clone());
        // The next item of a split group starts a new line: after a comma, and after the `}`
        // of a match arm with no comma.
        let parent = b.parent[i];
        let item_ends = match t.kind {
            TokenType::Comma => {
                !in_types.contains(&i) && parent.is_some_and(|p| split.contains(&p))
            }
            TokenType::RightBrace => {
                parent.is_some_and(|p| split.contains(&p) && groups[p] == Group::MatchArms)
                    && next_sig(&tokens, i + 1).is_some_and(|n| {
                        !matches!(tokens[n].kind, TokenType::Comma | TokenType::RightBrace)
                    })
            }
            _ => false,
        };
        if item_ends {
            need_newline = true;
        }
        i += 1;
    }
    (out, inline)
}

/// The next token from `from` that is not whitespace.
fn next_sig_or_comment(tokens: &[Token], from: usize) -> Option<usize> {
    (from..tokens.len()).find(|&k| !matches!(tokens[k].kind, TokenType::Whitespace(_)))
}

/// For an `if` used as a value, as in `let x = if c { 1 } else { 2 };`, the index of the last
/// `}` when each branch holds one simple value. `None` for any other `if`.
fn short_if_value(tokens: &[Token], b: &Brackets, at: usize) -> Option<usize> {
    let p = prev_sig(tokens, at)?;
    if !matches!(
        tokens[p].kind,
        TokenType::Equals
            | TokenType::LeftParen
            | TokenType::Comma
            | TokenType::Colon
            | TokenType::Return
            | TokenType::FatArrow
    ) {
        return None;
    }
    let lb1 = (at + 1..tokens.len())
        .find(|&k| matches!(tokens[k].kind, TokenType::LeftBrace | TokenType::Semicolon))?;
    if !matches!(tokens[lb1].kind, TokenType::LeftBrace) {
        return None;
    }
    let rb1 = b.close[lb1]?;
    let e = next_sig(tokens, rb1 + 1).filter(|&e| matches!(tokens[e].kind, TokenType::Else))?;
    let lb2 =
        next_sig(tokens, e + 1).filter(|&k| matches!(tokens[k].kind, TokenType::LeftBrace))?;
    let rb2 = b.close[lb2]?;
    let simple = |lo: usize, hi: usize| {
        (lo + 1..hi).any(|k| !is_trivia(&tokens[k].kind))
            && (lo + 1..hi).all(|k| {
                !matches!(
                    tokens[k].kind,
                    TokenType::Comment(_)
                        | TokenType::DocComment(_)
                        | TokenType::Semicolon
                        | TokenType::LeftBrace
                        | TokenType::RightBrace
                        | TokenType::If
                        | TokenType::Match
                        | TokenType::Return
                )
            })
    };
    // The condition holds no comment either.
    let condition_plain = (at..lb1).all(|k| {
        !matches!(
            tokens[k].kind,
            TokenType::Comment(_) | TokenType::DocComment(_)
        )
    });
    (condition_plain && simple(lb1, rb1) && simple(lb2, rb2)).then_some(rb2)
}

fn adjust_spacing(tokens: Vec<Token>) -> Vec<Token> {
    let mut new_tokens = Vec::with_capacity(tokens.len());
    let mut last_non_ws: Option<TokenType> = None;

    let mut next_non_ws_arr = vec![None; tokens.len()];
    let mut curr_next = None;
    for (i, token) in tokens.iter().enumerate().rev() {
        next_non_ws_arr[i] = curr_next.clone();
        if !matches!(
            token.kind,
            TokenType::Whitespace(_) | TokenType::Comment(_) | TokenType::DocComment(_)
        ) {
            curr_next = Some(token.kind.clone());
        }
    }

    for (i, mut token) in tokens.into_iter().enumerate() {
        if !matches!(
            token.kind,
            TokenType::Whitespace(_) | TokenType::Comment(_) | TokenType::DocComment(_)
        ) {
            last_non_ws = Some(token.kind.clone());
            new_tokens.push(token);
            continue;
        }

        if let TokenType::Whitespace(ref mut ws) = token.kind {
            let next_non_ws = &next_non_ws_arr[i];

            if ws.contains('\n') {
                if let Some(prev) = &last_non_ws {
                    if matches!(prev, TokenType::For | TokenType::If | TokenType::While) {
                        *ws = " ";
                    }
                }
                if let Some(next) = next_non_ws {
                    if matches!(
                        next,
                        TokenType::EqEq
                            | TokenType::NotEq
                            | TokenType::LessEq
                            | TokenType::GreaterEq
                            | TokenType::LeftAngle
                            | TokenType::RightAngle
                            | TokenType::AndAnd
                            | TokenType::OrOr
                    ) {
                        *ws = " ";
                    }
                }
            }

            if let (Some(TokenType::RightBrace), Some(TokenType::Else)) =
                (&last_non_ws, next_non_ws)
            {
                *ws = " ";
            }
            if let (Some(TokenType::Else), Some(TokenType::LeftBrace)) = (&last_non_ws, next_non_ws)
            {
                *ws = " ";
            }
        }

        new_tokens.push(token);
    }

    new_tokens
}

fn emit_formatted_string(tokens: Vec<Token>, indent_spaces: usize, original_len: usize) -> String {
    let mut formatted = String::with_capacity(original_len + original_len / 10);
    let mut indent_level: isize = 0;

    fn write_indent(out: &mut String, level: isize, spaces: usize) {
        if level > 0 {
            for _ in 0..(level as usize * spaces) {
                out.push(' ');
            }
        }
    }

    let mut is_new_line = true;
    let mut format_enabled = true;
    // For each open `[`, whether its line ends after it, so its rows are indented one level.
    let mut brackets: Vec<bool> = Vec::new();

    let mut tokens = tokens.into_iter().peekable();
    while let Some(token) = tokens.next() {
        if token.kind == TokenType::Eof {
            break;
        }

        match token.kind {
            TokenType::LeftBracket => {
                if format_enabled && is_new_line {
                    write_indent(&mut formatted, indent_level, indent_spaces);
                }
                is_new_line = false;
                formatted.push('[');
                let rows_below = matches!(
                    tokens.peek().map(|t| &t.kind),
                    Some(TokenType::Whitespace(ws)) if ws.contains('\n')
                );
                if rows_below {
                    indent_level += 1;
                }
                brackets.push(rows_below);
            }
            TokenType::RightBracket => {
                if brackets.pop() == Some(true) {
                    indent_level = (indent_level - 1).max(0);
                }
                if format_enabled && is_new_line {
                    write_indent(&mut formatted, indent_level, indent_spaces);
                }
                is_new_line = false;
                formatted.push(']');
            }
            TokenType::Whitespace(ws) => {
                if !format_enabled {
                    formatted.push_str(ws);
                    if ws.contains('\n') {
                        is_new_line = true;
                    }
                } else {
                    if ws.contains('\n') {
                        let newlines = ws.chars().filter(|&c| c == '\n').count();
                        for _ in 0..newlines {
                            formatted.push('\n');
                        }
                        is_new_line = true;
                    } else {
                        if !is_new_line {
                            formatted.push_str(ws);
                        }
                    }
                }
            }
            TokenType::RightBrace => {
                indent_level -= 1;
                if indent_level < 0 {
                    indent_level = 0;
                }

                if format_enabled && is_new_line {
                    write_indent(&mut formatted, indent_level, indent_spaces);
                }
                is_new_line = false;
                formatted.push('}');
            }
            TokenType::LeftBrace => {
                if format_enabled && is_new_line {
                    write_indent(&mut formatted, indent_level, indent_spaces);
                }
                is_new_line = false;
                formatted.push('{');
                indent_level += 1;
            }
            TokenType::Comment(c) | TokenType::DocComment(c) => {
                let mut turning_off = false;
                if c.contains("vx-format-begin: OFF") {
                    format_enabled = false;
                    turning_off = true;
                } else if c.contains("vx-format-begin: ON") || c.contains("vx-format-end: OFF") {
                    format_enabled = true;
                }

                if (format_enabled || turning_off) && is_new_line {
                    write_indent(&mut formatted, indent_level, indent_spaces);
                }
                is_new_line = false;
                formatted.push_str(c);
            }
            other => {
                if format_enabled && is_new_line {
                    write_indent(&mut formatted, indent_level, indent_spaces);
                }
                is_new_line = false;
                use std::fmt::Write;
                write!(&mut formatted, "{}", other).unwrap();
            }
        }
    }

    // Strip trailing whitespace from all lines
    let mut final_out = String::with_capacity(formatted.len());
    let mut iter = formatted.split('\n').peekable();
    while let Some(line) = iter.next() {
        final_out.push_str(line.trim_end_matches([' ', '\t']));
        if iter.peek().is_some() {
            final_out.push('\n');
        }
    }

    final_out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_off_top_level() {
        let input = "//! vx-format: OFF\nfn   foo()   {\nlet  x=1;\n}";
        let formatted = format_file(input, 2);
        assert_eq!(formatted, input);
    }

    #[test]
    fn the_rows_of_a_multi_line_array_are_indented() {
        let input = "const T : Tensor<i64, [3]> = [\n1,\n2,\n3\n];\n\nfn main() -> i32 {\n  let a : Tensor<i64, [3]> = [\n  1, 2,\n  3\n  ];\n  let b = [ 1, 2 ];\n  return a[0] + b[1];\n}";
        let expected = "const T : Tensor<i64, [3]> = [\n  1,\n  2,\n  3\n];\n\nfn main() -> i32 {\n  let a : Tensor<i64, [3]> = [\n    1, 2,\n    3\n  ];\n  let b = [ 1, 2 ];\n  return a[0] + b[1];\n}";
        assert_eq!(format_file(input, 2), expected);
    }

    #[test]
    fn escapes_in_strings_are_kept() {
        let input =
            "fn main() -> i32 {\n  print!(\"a\\tb\\n say \\\"hi\\\" \\\\ \");\n  return 0;\n}";
        assert_eq!(format_file(input, 2), input);
    }

    #[test]
    fn a_file_that_ends_inside_a_string_is_left_as_it_is() {
        let input = "fn main() -> i32 {\n    let s = \"abc";
        assert_eq!(format_file(input, 2), input);
    }

    #[test]
    fn a_short_block_with_a_comment_stays_on_several_lines() {
        let input = "fn main() -> i32 {\n  let mut x = 1;\n  unsafe {\n    // set x\n    x = 2;\n  }\n  return x;\n}";
        assert_eq!(format_file(input, 2), input);
    }

    #[test]
    fn test_format_off_block() {
        let input = "fn foo() {\n// vx-format-begin: OFF\n      let   x = 1;\n// vx-format-end: OFF\nlet y = 2;\n}";
        let expected = "fn foo() {\n  // vx-format-begin: OFF\n      let   x = 1;\n  // vx-format-end: OFF\n  let y = 2;\n}";
        let formatted = format_file(input, 2);
        assert_eq!(formatted, expected);
    }

    #[test]
    fn test_format_newline_stripping() {
        let input = "for \n i in 0..10 {\n  if sum_idx\n == 0 { a = 1; }\n}";
        let expected = "for i in 0..10 {\n  if sum_idx == 0 {\n    a = 1;\n  }\n}";
        let formatted = format_file(input, 2);
        assert_eq!(formatted, expected);
    }

    #[test]
    fn test_format_while_loop() {
        let input = "while \n keep_going { step(); }";
        let expected = "while keep_going {\n  step();\n}";
        let formatted = format_file(input, 2);
        assert_eq!(formatted, expected);
        assert_eq!(format_file(&formatted, 2), formatted);
    }

    #[test]
    fn test_format_else_single_line_expansion() {
        let input = "if a > 0 { x = 1; } else {\n  x = 0;\n}";
        let expected = "if a > 0 {\n  x = 1;\n} else {\n  x = 0;\n}";
        let formatted = format_file(input, 2);
        assert_eq!(formatted, expected);
    }

    /// Formats `input`, checks the result against `expected`, and checks that a second run
    /// changes nothing.
    fn formats_to(input: &str, expected: &str) {
        let formatted = format_file(input, 2);
        assert_eq!(formatted, expected);
        assert_eq!(format_file(&formatted, 2), formatted);
    }

    #[test]
    fn a_short_struct_literal_stays_on_one_line() {
        formats_to(
            "fn main() -> i32 {\n  count(Counter {\n    next: Slot::new(0)\n  });\n  let p = Point { x: 1, y: 2 };\n  return p.x;\n}",
            "fn main() -> i32 {\n  count(Counter { next: Slot::new(0) });\n  let p = Point { x: 1, y: 2 };\n  return p.x;\n}",
        );
    }

    #[test]
    fn a_struct_literal_in_a_struct_literal_stays_on_one_line() {
        formats_to(
            "fn main() -> i32 {\n  let q = P { x: 1, y: P { x: 3, y: 4 }.y, };\n  return q.y;\n}",
            "fn main() -> i32 {\n  let q = P { x: 1, y: P { x: 3, y: 4 }.y };\n  return q.y;\n}",
        );
    }

    #[test]
    fn a_long_struct_literal_has_one_field_on_each_line() {
        formats_to(
            "fn f(d : D) -> Wide {\n  return Wide {\n    negative: d.negative, exponent: d.exponent, hi: d.hi, lo: d.lo\n  };\n}",
            "fn f(d : D) -> Wide {\n  return Wide {\n    negative: d.negative,\n    exponent: d.exponent,\n    hi: d.hi,\n    lo: d.lo\n  };\n}",
        );
    }

    #[test]
    fn a_struct_literal_that_would_make_its_line_too_long_has_one_field_on_each_line() {
        formats_to(
            "fn f() -> D {\n  return add_both(DoubleDouble { hi: high_part, lo: low_part }, multiply_add(first_factor, second_factor, atan_of_i));\n}",
            "fn f() -> D {\n  return add_both(DoubleDouble {\n    hi: high_part,\n    lo: low_part\n  }, multiply_add(first_factor, second_factor, atan_of_i));\n}",
        );
    }

    #[test]
    fn a_comment_in_a_struct_literal_stays_after_its_field() {
        formats_to(
            "fn main() -> i32 {\n  let p = Point { x: 1, // left\n    y: 2 };\n  return p.x;\n}",
            "fn main() -> i32 {\n  let p = Point {\n    x: 1, // left\n    y: 2\n  };\n  return p.x;\n}",
        );
    }

    #[test]
    fn a_declaration_is_not_a_struct_literal() {
        formats_to(
            "struct Point { x : i32, y : i32 }\nMemory HBM { capacity: 1 GiB, scope: device }",
            "struct Point {\n  x : i32, y : i32\n}\nMemory HBM {\n  capacity: 1 GiB, scope: device\n}",
        );
    }

    #[test]
    fn a_comma_between_type_arguments_does_not_end_a_match_arm() {
        formats_to(
            "fn main() -> i32 {\n  match r { Result<u64, E>::Ok(n) => { return 1; }, Result<u64, E>::Err(e) => Result<u64, E>::Err(e), x => a < b }\n  return 0;\n}",
            "fn main() -> i32 {\n  match r {\n    Result<u64, E>::Ok(n) => {\n      return 1;\n    },\n    Result<u64, E>::Err(e) => Result<u64, E>::Err(e),\n    x => a < b\n  }\n  return 0;\n}",
        );
    }

    #[test]
    fn each_match_arm_starts_a_line() {
        formats_to(
            "fn main() -> i32 {\n  let mut n = 0;\n  match c { true => { n = 1; }, false => { n = 2; } }\n  match c { true => n = 3, false => n = 4 }\n  return n;\n}",
            "fn main() -> i32 {\n  let mut n = 0;\n  match c {\n    true => {\n      n = 1;\n    },\n    false => {\n      n = 2;\n    }\n  }\n  match c {\n    true => n = 3,\n    false => n = 4\n  }\n  return n;\n}",
        );
    }

    #[test]
    fn a_short_if_value_stays_on_one_line() {
        formats_to(
            "fn main() -> i32 {\n  let a = if c > 0 {\n    1\n  } else {\n    0\n  };\n  if c > 0 { a = 2; }\n  return a;\n}",
            "fn main() -> i32 {\n  let a = if c > 0 { 1 } else { 0 };\n  if c > 0 {\n    a = 2;\n  }\n  return a;\n}",
        );
    }

    #[test]
    fn an_if_value_with_statements_keeps_its_lines() {
        formats_to(
            "fn main() -> i32 {\n  let a = if c > 0 {\n    let b = 1;\n    b\n  } else { 0 };\n  return a;\n}",
            "fn main() -> i32 {\n  let a = if c > 0 {\n    let b = 1;\n    b\n  } else {\n    0\n  };\n  return a;\n}",
        );
    }

    #[test]
    fn test_format_dangling_else() {
        let input = "if a > 0 {\n  x = 1;\n}\n else {\n  x = 0;\n}";
        let expected = "if a > 0 {\n  x = 1;\n} else {\n  x = 0;\n}";
        let formatted = format_file(input, 2);
        assert_eq!(formatted, expected);
    }
}
