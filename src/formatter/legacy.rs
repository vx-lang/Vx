//! Bounded compatibility formatting. This module has no syntax-specific rule dispatch.
use super::source::document::{DelimiterKind, OpenDelimiter, RenderState};
use super::source::ownership::OwnershipMap;
use super::source::SourceFileView;
use crate::lexer::{Token, TokenType};
use std::collections::HashMap;
use std::ops::Range;
use vx_config::FormatConfig;

/// Stands for a unit that the engine formats itself, such as a nested syntax rule. The base
/// formatting sees one token in its place, so it still sees the braces on both sides of it.
const UNIT: &str = "\u{0}formatted elsewhere";
/// The same as `UNIT`, for a unit whose source ends with whitespace, such as a function header
/// before its `{`.
const UNIT_ENDING_IN_SPACE: &str = "\u{0}formatted elsewhere, then a space";
/// The same as `UNIT`, for a macro call on one line. Its width is the width of its text.
const MACRO_CALL: &str = "\u{0}a macro call";
/// The same as `UNIT`, for a macro call that spans several lines. A block that holds it is
/// already on several lines, as it would be with the tokens of the call in its place.
const MACRO_CALL_ON_SEVERAL_LINES: &str = "\u{0}a macro call on several lines";

fn is_unit(kind: &TokenType<'_>) -> bool {
    matches!(kind, TokenType::Identifier(text)
        if *text == UNIT || *text == UNIT_ENDING_IN_SPACE || *text == MACRO_CALL
            || *text == MACRO_CALL_ON_SEVERAL_LINES)
}

/// Whether the unit is code that a rule lays out, whose width is known only after it does.
fn is_rule_unit(kind: &TokenType<'_>) -> bool {
    matches!(kind, TokenType::Identifier(text) if *text == UNIT || *text == UNIT_ENDING_IN_SPACE)
}

/// Gives the width of unit `index` when its rule puts it on one line, and `None` when the rule
/// needs several lines.
pub(super) type MeasureUnit<'a> = dyn Fn(usize) -> Option<usize> + 'a;

fn breaks_a_line(kind: &TokenType<'_>) -> bool {
    match kind {
        TokenType::Whitespace(text) => text.contains('\n'),
        TokenType::Identifier(text) => *text == MACRO_CALL_ON_SEVERAL_LINES,
        _ => false,
    }
}

/// Format `bytes` with the base formatting. Each range in `units` is formatted by `render_unit`,
/// with the unit's index, at its place in the output. `measure_unit` gives the width of a unit
/// that a rule puts on one line.
#[allow(clippy::too_many_arguments)]
pub(super) fn render(
    source: &SourceFileView<'_>,
    ownership: &OwnershipMap,
    bytes: Range<usize>,
    config: &FormatConfig,
    state: &mut RenderState,
    inside_syntax: bool,
    units: &[Range<usize>],
    render_unit: &mut dyn FnMut(usize, &mut RenderState) -> Result<String, String>,
    measure_unit: &MeasureUnit<'_>,
) -> Result<String, String> {
    let mut cursor = bytes.start;
    for unit in units.iter().chain(std::iter::once(&(bytes.end..bytes.end))) {
        if cursor < unit.start {
            if inside_syntax {
                ownership.validate_region_edit(cursor..unit.start)?;
            } else {
                ownership.validate_legacy_edit(cursor..unit.start)?;
            }
        }
        cursor = unit.end;
    }
    let mut input = Vec::new();
    let mut next_unit = 0;
    for original in &source.tokens[source.token_range(&bytes)] {
        if !super::source::overlaps(&original.bytes, &bytes) {
            continue;
        }
        if let Some(unit) = units.get(next_unit) {
            if unit.start <= original.bytes.start {
                if original.bytes.end >= unit.end {
                    let mut token = original.token.clone();
                    let macro_call = source
                        .macros
                        .binary_search_by_key(&unit.start, |view| view.bytes.start)
                        .is_ok_and(|index| source.macros[index].bytes == *unit);
                    token.kind = TokenType::Identifier(
                        if matches!(original.token.kind, TokenType::Whitespace(_)) {
                            UNIT_ENDING_IN_SPACE
                        } else if macro_call && source.text[unit.clone()].contains('\n') {
                            MACRO_CALL_ON_SEVERAL_LINES
                        } else if macro_call {
                            MACRO_CALL
                        } else {
                            UNIT
                        },
                    );
                    token.length = unit.len();
                    // A placeholder has no position of its own, so its column holds the index
                    // of its unit.
                    token.column = next_unit;
                    input.push(token);
                    next_unit += 1;
                }
                continue;
            }
        }
        let start = original.bytes.start.max(bytes.start);
        let end = original.bytes.end.min(bytes.end);
        let mut token = original.token.clone();
        if start != original.bytes.start || end != original.bytes.end {
            if !matches!(token.kind, TokenType::Whitespace(_)) {
                return Err("A legacy interval splits a source token".into());
            }
            token.kind = TokenType::Whitespace(&source.text[start..end]);
            token.length = end - start;
        }
        input.push(token);
    }
    if next_unit != units.len() {
        return Err("A unit does not end at a source token".into());
    }
    let mut unit_index = 0;
    let tokens = adjust_spacing(normalize_and_expand_blocks(
        input,
        state.column,
        state.indent_level,
        config,
        measure_unit,
    ));
    let mut tokens = tokens.iter().peekable();
    let mut output = String::new();
    while let Some(token) = tokens.next() {
        if let TokenType::Whitespace(text) = &token.kind {
            if text.contains('\n') {
                emit(
                    &"\n".repeat(text.bytes().filter(|byte| *byte == b'\n').count()),
                    state,
                    &mut output,
                );
            } else if state.column > 0
                && !(tokens.peek().is_none() && bytes.end == source.text.len())
            {
                emit(text, state, &mut output);
            }
            continue;
        }
        if is_unit(&token.kind) {
            output.push_str(&render_unit(unit_index, state)?);
            unit_index += 1;
            continue;
        }
        let closing = match token.kind {
            TokenType::RightParen => Some(DelimiterKind::Parenthesis),
            TokenType::RightBracket => Some(DelimiterKind::Bracket),
            TokenType::RightBrace => Some(DelimiterKind::Brace),
            _ => None,
        };
        if let Some(kind) = closing {
            let open = state
                .delimiters
                .pop()
                .ok_or("Missing enclosing delimiter in legacy handoff")?;
            if open.kind != kind {
                return Err("Mismatched legacy delimiter state".into());
            }
            state.indent_level = state
                .indent_level
                .checked_sub(usize::from(open.adds_indent))
                .ok_or("Invalid legacy indentation state")?;
        }
        if state.column == 0 {
            emit(
                &" ".repeat(state.indent_level * config.indent),
                state,
                &mut output,
            );
        }
        emit(&token.kind.to_string(), state, &mut output);
        let opening = match token.kind {
            TokenType::LeftParen => Some((DelimiterKind::Parenthesis, false)),
            TokenType::LeftBrace => Some((DelimiterKind::Brace, true)),
            TokenType::LeftBracket => Some((
                DelimiterKind::Bracket,
                tokens.peek().is_some_and(
                    |next| matches!(&next.kind, TokenType::Whitespace(text) if text.contains('\n')),
                ),
            )),
            _ => None,
        };
        if let Some((kind, adds_indent)) = opening {
            state.delimiters.push(OpenDelimiter { kind, adds_indent });
            state.indent_level += usize::from(adds_indent);
        }
    }
    Ok(output)
}

fn emit(text: &str, state: &mut RenderState, output: &mut String) {
    output.push_str(text);
    state.emitted(text);
}

/// Counts the indentation level that `render` gives a line, with the same rules: one level for
/// each open `{`, and for each open `[` whose line ends after it.
struct IndentLevel {
    scanned: usize,
    level: usize,
    brackets: Vec<bool>,
}

impl IndentLevel {
    /// Counts the brackets in `tokens[self.scanned..end]`. The tokens before `end` must not change.
    fn advance(&mut self, tokens: &[Token], end: usize) {
        assert!(
            self.scanned <= end,
            "the indentation was counted past this token"
        );
        for k in self.scanned..end {
            match tokens[k].kind {
                TokenType::LeftBrace => self.level += 1,
                TokenType::RightBrace => self.level = self.level.saturating_sub(1),
                TokenType::LeftBracket => {
                    let rows_below = matches!(
                        tokens.get(k + 1).map(|t| &t.kind),
                        Some(TokenType::Whitespace(ws)) if ws.contains('\n')
                    );
                    self.level += usize::from(rows_below);
                    self.brackets.push(rows_below);
                }
                TokenType::RightBracket => {
                    let rows_below = self.brackets.pop() == Some(true);
                    self.level = self.level.saturating_sub(usize::from(rows_below));
                }
                _ => {}
            }
        }
        self.scanned = end;
    }

    /// The level of a line that starts with `line`, after `advance` counted the lines above it.
    /// A line that starts with a `}`, or with the `]` of a multi-line array, is one level out.
    fn level_of_line(&self, line: &[Token]) -> usize {
        let closes = match line.first().map(|t| &t.kind) {
            Some(TokenType::RightBrace) => true,
            Some(TokenType::RightBracket) => self.brackets.last() == Some(&true),
            _ => false,
        };
        self.level.saturating_sub(usize::from(closes))
    }
}

/// `start_column` is the column of the first token, where the code before it on the line ended.
/// `start_indent` is the indentation level there.
fn normalize_and_expand_blocks<'a>(
    mut tokens: Vec<Token<'a>>,
    start_column: usize,
    start_indent: usize,
    config: &FormatConfig,
    measure_unit: &MeasureUnit<'_>,
) -> Vec<Token<'a>> {
    let max_width = config.max_width;
    // The width of a token on one line. A rule's unit that needs several lines has none.
    let width = |token: &Token| {
        if is_rule_unit(&token.kind) {
            measure_unit(token.column)
        } else {
            Some(token.length)
        }
    };
    // Pass 1a: Ensure spaces before `{`
    let mut spaced_tokens: Vec<Token> = Vec::with_capacity(tokens.len() + 10);
    for token in &tokens {
        if let TokenType::LeftBrace = token.kind {
            if !spaced_tokens.is_empty()
                && !matches!(spaced_tokens.last().unwrap().kind, TokenType::Whitespace(_))
                && !matches!(
                    spaced_tokens.last().unwrap().kind,
                    TokenType::Identifier(text) if text == UNIT_ENDING_IN_SPACE
                )
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
    let mut indent = IndentLevel {
        scanned: 0,
        level: start_indent,
        brackets: Vec::new(),
    };
    let mut i = 0;
    while i < tokens.len() {
        if let TokenType::LeftBrace = tokens[i].kind {
            if let Some(&rb_idx) = left_to_right.get(&i) {
                let lb_idx = i;
                let has_newline = tokens[(lb_idx + 1)..rb_idx]
                    .iter()
                    .any(|t| breaks_a_line(&t.kind));

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
                // A rule's unit can go on the line only when the rule puts it on one line. A
                // macro call on several lines cannot.
                let breaks = tokens[lb_idx..=rb_idx].iter().any(|t| {
                    (breaks_a_line(&t.kind) && is_unit(&t.kind))
                        || (is_unsafe_block && width(t).is_none())
                });
                // With a rule's unit on the line, the whole line must fit in max_width, so that
                // the rule keeps it on one line there too.
                let limit = if tokens[lb_idx..=rb_idx]
                    .iter()
                    .any(|t| is_rule_unit(&t.kind))
                {
                    80.min(max_width)
                } else {
                    80
                };
                if is_unsafe_block && !holds_comment && !breaks {
                    // Measure the line as `render` writes it: with the indentation it gives, not the
                    // indentation of the input. With no line break before the block in these
                    // tokens, the line starts before them, at `start_column`.
                    let line_start = new_tokens
                        .iter()
                        .rposition(
                            |t| matches!(t.kind, TokenType::Whitespace(ws) if ws.contains('\n')),
                        )
                        .map_or(0, |k| k + 1);
                    indent.advance(&new_tokens, line_start);
                    let line = &new_tokens[line_start..];
                    // `render` drops the spaces at the start of a line.
                    let first = line
                        .iter()
                        .position(|t| !matches!(t.kind, TokenType::Whitespace(_)))
                        .unwrap_or(line.len());
                    let mut collapsed_len = if line_start == 0 && start_column > 0 {
                        start_column + line.iter().map(|t| t.length).sum::<usize>()
                            - line[first..].iter().map(|t| t.length).sum::<usize>()
                    } else {
                        indent.level_of_line(&line[first..]) * config.indent
                    };
                    collapsed_len += line[first..]
                        .iter()
                        .map(|t| width(t).unwrap_or(t.length))
                        .sum::<usize>();
                    for t in &tokens[lb_idx..=rb_idx] {
                        if let TokenType::Whitespace(ws) = t.kind {
                            if ws.contains('\n') {
                                collapsed_len += 1;
                            } else {
                                collapsed_len += t.length;
                            }
                        } else {
                            collapsed_len += width(t).unwrap_or(t.length);
                        }
                    }

                    if collapsed_len + 2 <= limit {
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
                let has_newline = tokens[(lb_idx + 1)..i]
                    .iter()
                    .any(|t| breaks_a_line(&t.kind));

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
