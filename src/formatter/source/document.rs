use super::ownership::{OutputSegment, Owner, OwnershipMap};
use super::{SourceFileView, TokenId};
use crate::lexer::TokenType;
use std::ops::Range;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub column: usize,
    pub indent_level: usize,
    pub indent_spaces: usize,
    pub width: usize,
}

impl Shape {
    pub fn remaining(self) -> usize {
        self.width.saturating_sub(self.column)
    }
    pub fn reserve(self, columns: usize) -> Option<Self> {
        (columns <= self.remaining()).then(|| Self {
            width: self.width - columns,
            ..self
        })
    }
    pub fn child_line(self) -> Self {
        Self {
            column: (self.indent_level + 1) * self.indent_spaces,
            indent_level: self.indent_level + 1,
            ..self
        }
    }
}

/// Nested layouts contain source references and whitespace, never arbitrary replacement code.
#[derive(Clone, Debug)]
pub enum Document {
    Token(TokenId),
    Space,
    Line {
        indent_level: usize,
    },
    Concat(Vec<Document>),
    Choice {
        flat: Box<Document>,
        expanded: Box<Document>,
    },
    WithWidth {
        width: usize,
        child: Box<Document>,
    },
    WithShape {
        shape: Shape,
        child: Box<Document>,
    },
    Preserve {
        bytes: Range<usize>,
        keep_column: bool,
    },
}

impl Document {
    /// Select this group's expanded alternative while retaining child constraints.
    pub fn expanded(&self) -> Self {
        match self {
            Self::Choice { expanded, .. } => *expanded.clone(),
            Self::WithShape { shape, child } => Self::WithShape {
                shape: *shape,
                child: Box::new(child.expanded()),
            },
            Self::WithWidth { width, child } => Self::WithWidth {
                width: *width,
                child: Box::new(child.expanded()),
            },
            _ => self.clone(),
        }
    }

    pub fn flat_width(&self, source: &SourceFileView<'_>) -> Option<usize> {
        match self {
            Self::Token(id) => Some(source.token_text(*id).chars().count()),
            Self::Space => Some(1),
            Self::Line { .. } => None,
            Self::Concat(children) => children.iter().try_fold(0usize, |width, child| {
                width.checked_add(child.flat_width(source)?)
            }),
            Self::Choice { flat, .. } => flat.flat_width(source),
            Self::WithWidth { child, .. } | Self::WithShape { child, .. } => {
                child.flat_width(source)
            }
            Self::Preserve { bytes, .. } => {
                let text = &source.text[bytes.clone()];
                (!text.contains(['\r', '\n'])).then(|| text.chars().count())
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DelimiterKind {
    Parenthesis,
    Bracket,
    Brace,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpenDelimiter {
    pub kind: DelimiterKind,
    pub adds_indent: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenderState {
    pub line: usize,
    pub column: usize,
    pub indent_level: usize,
    pub delimiters: Vec<OpenDelimiter>,
}

impl RenderState {
    pub fn new(shape: Shape) -> Self {
        Self {
            line: 1,
            column: shape.column,
            indent_level: shape.indent_level,
            delimiters: Vec::new(),
        }
    }

    /// Advance syntax once for a consumed source span, including unbalanced preserved exclusions.
    pub fn consume_source(
        &mut self,
        source: &SourceFileView<'_>,
        bytes: Range<usize>,
    ) -> Result<(), String> {
        let range = source.token_range(&bytes);
        for (index, token) in source.tokens[range.clone()]
            .iter()
            .enumerate()
            .map(|(index, token)| (index + range.start, token))
            .filter(|(_, token)| bytes.start <= token.bytes.start && token.bytes.end <= bytes.end)
        {
            let opening = match token.token.kind {
                TokenType::LeftParen => Some((DelimiterKind::Parenthesis, false)),
                TokenType::LeftBrace => Some((DelimiterKind::Brace, true)),
                TokenType::LeftBracket => Some((DelimiterKind::Bracket, source.tokens.get(index + 1)
                    .is_some_and(|next| matches!(&next.token.kind, TokenType::Whitespace(text) if text.contains('\n'))))),
                _ => None,
            };
            if let Some((kind, adds_indent)) = opening {
                self.delimiters.push(OpenDelimiter { kind, adds_indent });
                self.indent_level += usize::from(adds_indent);
                continue;
            }
            let closing = match token.token.kind {
                TokenType::RightParen => Some(DelimiterKind::Parenthesis),
                TokenType::RightBrace => Some(DelimiterKind::Brace),
                TokenType::RightBracket => Some(DelimiterKind::Bracket),
                _ => None,
            };
            if let Some(kind) = closing {
                let open = self
                    .delimiters
                    .pop()
                    .ok_or("Cannot establish enclosing delimiter state")?;
                if open.kind != kind {
                    return Err("Mismatched delimiter in source handoff".into());
                }
                self.indent_level = self
                    .indent_level
                    .checked_sub(usize::from(open.adds_indent))
                    .ok_or("Invalid enclosing indentation state")?;
            }
        }
        Ok(())
    }

    pub(crate) fn emitted(&mut self, text: &str) {
        for character in text.chars() {
            if character == '\n' {
                self.line += 1;
                self.column = 0;
            } else {
                self.column += 1;
            }
        }
    }

    pub fn consume_preserved(
        &mut self,
        source: &SourceFileView<'_>,
        bytes: Range<usize>,
    ) -> Result<(), String> {
        self.consume_source(source, bytes.clone())?;
        self.emitted(&source.text[bytes]);
        Ok(())
    }
}

pub struct RenderedDocument {
    pub text: String,
    pub segments: Vec<OutputSegment>,
    pub exit_state: RenderState,
}

pub fn render(
    source: &SourceFileView<'_>,
    ownership: &OwnershipMap,
    document: &Document,
    consumed: Range<usize>,
    shape: Shape,
    mut entry: RenderState,
) -> Result<RenderedDocument, String> {
    if entry.column != shape.column || entry.indent_level != shape.indent_level {
        return Err("Layout shape does not match the enclosing renderer state".into());
    }
    let mut text = String::new();
    let mut segments = Vec::new();
    let mut tokens = Vec::new();
    render_part(
        source,
        ownership,
        document,
        shape,
        &mut entry,
        &mut text,
        &mut segments,
        &mut tokens,
    )?;
    let token_range = source.token_range(&consumed);
    let expected: Vec<_> = source.tokens[token_range.clone()]
        .iter()
        .enumerate()
        .map(|(index, token)| (index + token_range.start, token))
        .filter(|(_, token)| {
            token.bytes.start >= consumed.start
                && token.bytes.end <= consumed.end
                && !matches!(token.token.kind, TokenType::Whitespace(_) | TokenType::Eof)
        })
        .map(|(index, _)| TokenId(index))
        .collect();
    if tokens != expected {
        return Err("Layout removed, duplicated, or reordered source tokens".into());
    }
    source.verify_span_preservation(consumed.clone(), &text)?;
    for span in ownership
        .protected_in(&consumed)
        .filter(|span| super::overlaps(&span.bytes, &consumed))
    {
        if !segments.iter().any(|segment| {
            segment.preserved
                && segment.source.as_ref().is_some_and(|original| {
                    original.start <= span.bytes.start && span.bytes.end <= original.end
                })
        }) {
            return Err("Layout did not preserve a complete opaque or excluded range".into());
        }
    }
    for macro_view in source
        .macros
        .iter()
        .skip(
            source
                .macros
                .partition_point(|view| view.bytes.end <= consumed.start),
        )
        .take_while(|view| view.bytes.start < consumed.end)
        .filter(|view| super::overlaps(&view.bytes, &consumed))
    {
        let segment = segments
            .iter()
            .find(|segment| {
                segment.preserved
                    && segment.source.as_ref().is_some_and(|range| {
                        range.start <= macro_view.bytes.start && macro_view.bytes.end <= range.end
                    })
            })
            .ok_or("Macro token trees require exact source preservation")?;
        let origin = segment
            .source
            .as_ref()
            .expect("preserved segment has original source");
        let offset = segment.output.start + macro_view.bytes.start - origin.start;
        let prefix = &text[..offset];
        let column = prefix.rfind('\n').map_or_else(
            || shape.column + prefix.chars().count(),
            |newline| prefix[newline + 1..].chars().count(),
        );
        if column != macro_view.original_column - 1 {
            return Err("Opaque macro cannot move to a different column".into());
        }
    }
    entry.consume_source(source, consumed)?;
    Ok(RenderedDocument {
        text,
        segments,
        exit_state: entry,
    })
}

#[allow(clippy::too_many_arguments)]
fn render_part(
    source: &SourceFileView<'_>,
    ownership: &OwnershipMap,
    document: &Document,
    shape: Shape,
    state: &mut RenderState,
    text: &mut String,
    segments: &mut Vec<OutputSegment>,
    tokens: &mut Vec<TokenId>,
) -> Result<(), String> {
    match document {
        Document::WithShape {
            shape: child_shape,
            child,
        } => {
            return render_part(
                source,
                ownership,
                child,
                Shape {
                    column: state.column,
                    width: shape
                        .width
                        .min(state.column.saturating_add(child_shape.remaining())),
                    ..*child_shape
                },
                state,
                text,
                segments,
                tokens,
            );
        }
        Document::WithWidth { width, child } => {
            return render_part(
                source,
                ownership,
                child,
                Shape {
                    width: shape.width.min(*width),
                    ..shape
                },
                state,
                text,
                segments,
                tokens,
            );
        }
        Document::Concat(children) => {
            for child in children {
                render_part(
                    source, ownership, child, shape, state, text, segments, tokens,
                )?;
            }
            return Ok(());
        }
        Document::Choice { flat, expanded } => {
            let fits = flat
                .flat_width(source)
                .is_some_and(|width| width <= shape.width.saturating_sub(state.column));
            return render_part(
                source,
                ownership,
                if fits { flat } else { expanded },
                shape,
                state,
                text,
                segments,
                tokens,
            );
        }
        _ => {}
    }
    let (value, origin, owner) = match document {
        Document::Token(id) => {
            let token = &source.tokens[id.0];
            if matches!(token.token.kind, TokenType::Whitespace(_) | TokenType::Eof) {
                return Err("Layout token must reference non-whitespace source".into());
            }
            let owner = ownership.owner_at(token.bytes.start);
            if owner.preserves_bytes() {
                return Err("Opaque source must use an exact preservation layout".into());
            }
            tokens.push(*id);
            (
                source.token_text(*id).to_string(),
                Some(token.bytes.clone()),
                owner,
            )
        }
        Document::Space => (" ".into(), None, Owner::Syntax),
        Document::Line { indent_level } => (
            format!("\n{}", " ".repeat(indent_level * shape.indent_spaces)),
            None,
            Owner::Syntax,
        ),
        Document::Preserve { bytes, keep_column } => {
            if *keep_column {
                let line_start = source.text[..bytes.start]
                    .rfind('\n')
                    .map_or(0, |offset| offset + 1);
                if state.column != source.text[line_start..bytes.start].chars().count() {
                    return Err("Opaque source cannot move to a different column".into());
                }
            }
            let range = source.token_range(bytes);
            for (index, token) in source.tokens[range.clone()]
                .iter()
                .enumerate()
                .map(|(index, token)| (index + range.start, token))
            {
                if bytes.start <= token.bytes.start
                    && token.bytes.end <= bytes.end
                    && !matches!(token.token.kind, TokenType::Whitespace(_) | TokenType::Eof)
                {
                    tokens.push(TokenId(index));
                }
            }
            (
                source.text[bytes.clone()].to_string(),
                Some(bytes.clone()),
                ownership.owner_at(bytes.start),
            )
        }
        _ => unreachable!(),
    };
    let start = text.len();
    text.push_str(&value);
    state.emitted(&value);
    segments.push(OutputSegment {
        output: start..text.len(),
        source: origin,
        owner,
        whitespace_only: value.chars().all(char::is_whitespace),
        preserved: matches!(document, Document::Preserve { .. }),
    });
    Ok(())
}
