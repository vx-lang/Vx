//! The `struct_literals` rule: `Point { x: 1, y: 2 }` stays on one line when it is short, and
//! has one field on each line when it is not.

use super::dispatch::{NodeContext, NodeFormatter, Rule, SyntaxKind, SyntaxNode};
use super::document::{Document, Shape};
use super::{CoverageWarning, SourceFileView, StructLiteralView};
use crate::lexer::TokenType;
use std::ops::Range;

pub(super) struct StructLiteralFormatter;

impl NodeFormatter for StructLiteralFormatter {
    fn kind(&self) -> SyntaxKind {
        SyntaxKind::StructLiteral
    }

    fn rule(&self) -> Option<Rule> {
        Some(Rule {
            name: "struct_literals",
            enabled_by_default: |_| true,
        })
    }

    fn roots<'a>(&self, source: &'a SourceFileView<'_>) -> Vec<SyntaxNode<'a>> {
        source
            .struct_literals
            .iter()
            .map(SyntaxNode::StructLiteral)
            .collect()
    }

    fn format(
        &self,
        node: SyntaxNode<'_>,
        context: &NodeContext<'_, '_, '_>,
        shape: Shape,
    ) -> Result<Document, CoverageWarning> {
        let SyntaxNode::StructLiteral(view) = node else {
            return Err(CoverageWarning {
                bytes: context.source.token_bytes(node.tokens()),
                message: "Unsupported syntax ownership; preserve the struct literal".into(),
            });
        };
        let region = |tokens: Range<usize>, indent_level: usize| Document::Region {
            bytes: context.source.token_bytes(tokens),
            indent_level,
        };
        if view.fields.is_empty() {
            return Ok(region(view.tokens.clone(), shape.indent_level));
        }
        let name = region(view.name.clone(), shape.indent_level);
        let inner = shape.indent_level + 1;
        let source = context.source;
        // The comments between `from` and `to`, outside the fields. A comment stays on its line:
        // after the code before it, or on a line of its own.
        let comments = |from: usize, to: usize, indent_level: usize| {
            let mut documents = Vec::new();
            for index in from..to {
                if !is_comment(&source.tokens[index].token.kind) {
                    continue;
                }
                let own_line = matches!(
                    &source.tokens[index - 1].token.kind,
                    TokenType::Whitespace(text) if text.contains('\n')
                );
                documents.push(if own_line {
                    Document::Line { indent_level }
                } else {
                    Document::Space
                });
                documents.push(Document::Token(super::TokenId(index)));
            }
            documents
        };

        // A comment before a comma, or before the `{`, has no place in the layout below.
        let mut gaps: Vec<Range<usize>> = Vec::new();
        gaps.push(view.open.0 + 1..view.fields[0].start);
        for (index, (field, comma)) in view.fields.iter().zip(&view.commas).enumerate() {
            let after = comma.map_or(field.end, |comma| comma.0 + 1);
            let next = view
                .fields
                .get(index + 1)
                .map_or(view.close.0, |next| next.start);
            gaps.push(after..next);
        }
        let misplaced = view.tokens.clone().any(|index| {
            is_comment(&source.tokens[index].token.kind)
                && !view.fields.iter().any(|field| field.contains(&index))
                && !gaps.iter().any(|gap| gap.contains(&index))
        });
        if misplaced {
            return Ok(region(view.tokens.clone(), shape.indent_level));
        }

        let mut expanded = vec![name.clone(), Document::Space, Document::Token(view.open)];
        let mut gap_start = view.open.0 + 1;
        for (field, comma) in view.fields.iter().zip(&view.commas) {
            expanded.extend(comments(gap_start, field.start, inner));
            expanded.push(Document::Line {
                indent_level: inner,
            });
            expanded.push(region(field.clone(), inner));
            expanded.extend(comma.map(Document::Token));
            gap_start = comma.map_or(field.end, |comma| comma.0 + 1);
        }
        expanded.extend(comments(gap_start, view.close.0, inner));
        expanded.push(Document::Line {
            indent_level: shape.indent_level,
        });
        expanded.push(Document::Token(view.close));
        let expanded = Document::Concat(expanded);
        // A line comment ends its line, so a literal with a comment between its fields cannot be
        // on one line.
        if has_comment_between_fields(source, view) || holds_a_block(source, view) {
            return Ok(expanded);
        }

        // On one line, the literal has no comma before its `}`. With one field on each line, it
        // keeps the comma that it has.
        let last = view.fields.len() - 1;
        let mut fields = Vec::new();
        for (index, (field, comma)) in view.fields.iter().zip(&view.commas).enumerate() {
            if index > 0 {
                fields.push(Document::Space);
            }
            fields.push(region(field.clone(), shape.indent_level));
            fields.extend(comma.map(if index == last {
                Document::Remove
            } else {
                Document::Token
            }));
        }
        let flat = Document::Concat(vec![
            name,
            Document::Space,
            Document::Token(view.open),
            Document::Space,
            Document::MaxWidth {
                width: context.config.struct_lit_width,
                child: Box::new(Document::Concat(fields)),
            },
            Document::Space,
            Document::Token(view.close),
        ]);
        Ok(Document::Choice {
            flat: Box::new(flat),
            expanded: Box::new(expanded),
        })
    }
}

fn is_comment(kind: &TokenType<'_>) -> bool {
    matches!(kind, TokenType::Comment(_) | TokenType::DocComment(_))
}

/// Whether a `{` inside the literal opens something other than a struct literal with fields,
/// such as a block or `PhantomData<T> {}`. Such a literal keeps one field on each line.
fn holds_a_block(source: &SourceFileView<'_>, view: &StructLiteralView) -> bool {
    (view.open.0 + 1..view.close.0).any(|index| match source.tokens[index].token.kind {
        TokenType::LeftBrace => !source
            .struct_literals
            .iter()
            .any(|other| other.open.0 == index && !other.fields.is_empty()),
        TokenType::Semicolon => true,
        _ => false,
    })
}

fn has_comment_between_fields(source: &SourceFileView<'_>, view: &StructLiteralView) -> bool {
    view.tokens.clone().any(|index| {
        is_comment(&source.tokens[index].token.kind)
            && !view.fields.iter().any(|field| field.contains(&index))
    })
}
