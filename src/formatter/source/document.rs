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
    },
    /// Source code that a rule does not lay out itself, such as the value of a struct field.
    /// The engine formats it the normal way: with the rules that own code inside it, and with
    /// the base formatting for the rest. New lines inside it start at `indent_level`.
    Region {
        bytes: Range<usize>,
        indent_level: usize,
    },
    /// `child` must fit on one line of at most `width` columns. Use it inside the one-line
    /// alternative of a `Choice`, which then falls back to its other alternative.
    MaxWidth {
        width: usize,
        child: Box<Document>,
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

    /// Whether the document holds a `Region`, whose width is known only after rendering.
    fn has_region(&self) -> bool {
        match self {
            Self::Region { .. } => true,
            Self::Concat(children) => children.iter().any(Self::has_region),
            Self::Choice { flat, expanded } => flat.has_region() || expanded.has_region(),
            Self::WithWidth { child, .. }
            | Self::WithShape { child, .. }
            | Self::MaxWidth { child, .. } => child.has_region(),
            Self::Token(_) | Self::Space | Self::Line { .. } | Self::Preserve { .. } => false,
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
            Self::MaxWidth { width, child } => child
                .flat_width(source)
                .filter(|child_width| child_width <= width),
            Self::Region { .. } => None,
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

/// Formats the source bytes of a `Document::Region`, starting from the given renderer state.
pub type RegionRenderer<'a> = dyn Fn(Range<usize>, &mut RenderState) -> Result<String, String> + 'a;

/// Render a document that holds no `Region`.
pub fn render(
    source: &SourceFileView<'_>,
    ownership: &OwnershipMap,
    document: &Document,
    consumed: Range<usize>,
    shape: Shape,
    entry: RenderState,
) -> Result<RenderedDocument, String> {
    render_with_regions(
        source,
        ownership,
        document,
        consumed,
        shape,
        entry,
        &|_, _| Err("This layout cannot format a region".into()),
    )
}

pub fn render_with_regions(
    source: &SourceFileView<'_>,
    ownership: &OwnershipMap,
    document: &Document,
    consumed: Range<usize>,
    shape: Shape,
    mut entry: RenderState,
    regions: &RegionRenderer<'_>,
) -> Result<RenderedDocument, String> {
    if entry.column != shape.column || entry.indent_level != shape.indent_level {
        return Err("Layout shape does not match the enclosing renderer state".into());
    }
    let renderer = Renderer {
        source,
        ownership,
        regions,
    };
    let mut output = Output::default();
    renderer.part(document, shape, &mut entry, &mut output)?;
    let Output {
        text,
        segments,
        tokens,
    } = output;
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
        // A macro call may also sit in a region, which formats it the normal way. That keeps
        // its bytes as written and moves only its line.
        let call = span.owner == Owner::OpaqueMacro;
        if !segments.iter().any(|segment| {
            (segment.preserved || call)
                && segment.source.as_ref().is_some_and(|original| {
                    original.start <= span.bytes.start && span.bytes.end <= original.end
                })
        }) {
            return Err(if call {
                "A layout split the tokens of a macro".into()
            } else {
                "Layout did not preserve a complete opaque or excluded range".into()
            });
        }
    }
    entry.consume_source(source, consumed)?;
    Ok(RenderedDocument {
        text,
        segments,
        exit_state: entry,
    })
}

struct Renderer<'a, 'source> {
    source: &'a SourceFileView<'source>,
    ownership: &'a OwnershipMap,
    regions: &'a RegionRenderer<'a>,
}

#[derive(Default)]
struct Output {
    text: String,
    segments: Vec<OutputSegment>,
    tokens: Vec<TokenId>,
}

impl Output {
    fn mark(&self) -> (usize, usize, usize) {
        (self.text.len(), self.segments.len(), self.tokens.len())
    }

    fn reset(&mut self, (text, segments, tokens): (usize, usize, usize)) {
        self.text.truncate(text);
        self.segments.truncate(segments);
        self.tokens.truncate(tokens);
    }
}

impl Renderer<'_, '_> {
    fn part(
        &self,
        document: &Document,
        shape: Shape,
        state: &mut RenderState,
        out: &mut Output,
    ) -> Result<(), String> {
        match document {
            Document::WithShape {
                shape: child_shape,
                child,
            } => {
                return self.part(
                    child,
                    Shape {
                        column: state.column,
                        width: shape
                            .width
                            .min(state.column.saturating_add(child_shape.remaining())),
                        ..*child_shape
                    },
                    state,
                    out,
                );
            }
            Document::WithWidth { width, child } => {
                return self.part(
                    child,
                    Shape {
                        width: shape.width.min(*width),
                        ..shape
                    },
                    state,
                    out,
                );
            }
            Document::Concat(children) => {
                for child in children {
                    self.part(child, shape, state, out)?;
                }
                return Ok(());
            }
            Document::Choice { flat, expanded } if flat.has_region() => {
                // A region's width is known only after it is rendered. Render the one-line
                // alternative, and undo it when it breaks a line or does not fit.
                let mark = out.mark();
                let saved = state.clone();
                let fits = self.part(flat, shape, state, out).is_ok()
                    && !out.text[mark.0..].contains('\n')
                    && state.column <= shape.width;
                if fits {
                    return Ok(());
                }
                out.reset(mark);
                *state = saved;
                return self.part(expanded, shape, state, out);
            }
            Document::Choice { flat, expanded } => {
                let fits = flat
                    .flat_width(self.source)
                    .is_some_and(|width| width <= shape.width.saturating_sub(state.column));
                return self.part(if fits { flat } else { expanded }, shape, state, out);
            }
            Document::MaxWidth { width, child } => {
                let start = out.text.len();
                self.part(child, shape, state, out)?;
                let rendered = &out.text[start..];
                if rendered.contains('\n') || rendered.chars().count() > *width {
                    return Err("A one-line layout does not fit its width".into());
                }
                return Ok(());
            }
            Document::Region {
                bytes,
                indent_level,
            } => {
                let saved_indent = state.indent_level;
                let depth = state.delimiters.len();
                state.indent_level = *indent_level;
                let start = out.text.len();
                let text = (self.regions)(bytes.clone(), state)?;
                if state.delimiters.len() != depth || state.indent_level != *indent_level {
                    return Err("A region must close every delimiter it opens".into());
                }
                state.indent_level = saved_indent;
                out.text.push_str(&text);
                let range = self.source.token_range(bytes);
                out.tokens.extend(
                    self.source.tokens[range.clone()]
                        .iter()
                        .enumerate()
                        .filter(|(_, token)| {
                            bytes.start <= token.bytes.start
                                && token.bytes.end <= bytes.end
                                && !matches!(
                                    token.token.kind,
                                    TokenType::Whitespace(_) | TokenType::Eof
                                )
                        })
                        .map(|(index, _)| TokenId(index + range.start)),
                );
                out.segments.push(OutputSegment {
                    output: start..out.text.len(),
                    source: Some(bytes.clone()),
                    owner: Owner::Legacy,
                    preserved: false,
                });
                return Ok(());
            }
            _ => {}
        }
        let (value, origin, owner) = match document {
            Document::Token(id) => {
                let token = &self.source.tokens[id.0];
                if matches!(token.token.kind, TokenType::Whitespace(_) | TokenType::Eof) {
                    return Err("Layout token must reference non-whitespace source".into());
                }
                let owner = self.ownership.owner_at(token.bytes.start);
                if owner.preserves_bytes() {
                    return Err("Opaque source must use an exact preservation layout".into());
                }
                out.tokens.push(*id);
                (
                    self.source.token_text(*id).to_string(),
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
            Document::Preserve { bytes } => {
                let range = self.source.token_range(bytes);
                for (index, token) in self.source.tokens[range.clone()]
                    .iter()
                    .enumerate()
                    .map(|(index, token)| (index + range.start, token))
                {
                    if bytes.start <= token.bytes.start
                        && token.bytes.end <= bytes.end
                        && !matches!(token.token.kind, TokenType::Whitespace(_) | TokenType::Eof)
                    {
                        out.tokens.push(TokenId(index));
                    }
                }
                (
                    self.source.text[bytes.clone()].to_string(),
                    Some(bytes.clone()),
                    self.ownership.owner_at(bytes.start),
                )
            }
            _ => unreachable!("composite documents return above"),
        };
        let start = out.text.len();
        out.text.push_str(&value);
        state.emitted(&value);
        out.segments.push(OutputSegment {
            output: start..out.text.len(),
            source: origin,
            owner,
            preserved: matches!(document, Document::Preserve { .. }),
        });
        Ok(())
    }
}
