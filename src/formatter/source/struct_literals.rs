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
        // A comment between the fields ends its line, so the rule cannot move what follows it.
        // Leave such a literal, and one with no fields, to the base formatting.
        if view.fields.is_empty() || has_comment_between_fields(context.source, view) {
            return Ok(region(view.tokens.clone(), shape.indent_level));
        }
        let name = region(view.name.clone(), shape.indent_level);
        let inner = shape.indent_level + 1;

        let mut expanded = vec![name.clone(), Document::Space, Document::Token(view.open)];
        for (field, comma) in view.fields.iter().zip(&view.commas) {
            expanded.push(Document::Line {
                indent_level: inner,
            });
            expanded.push(region(field.clone(), inner));
            expanded.extend(comma.map(Document::Token));
        }
        expanded.push(Document::Line {
            indent_level: shape.indent_level,
        });
        expanded.push(Document::Token(view.close));
        let expanded = Document::Concat(expanded);
        // The rule keeps every token, so a trailing comma stays. Only the layout with one field
        // on each line has a place for it.
        if view.commas.last().is_some_and(Option::is_some) {
            return Ok(expanded);
        }

        let mut fields = Vec::new();
        for (index, (field, comma)) in view.fields.iter().zip(&view.commas).enumerate() {
            if index > 0 {
                fields.push(Document::Space);
            }
            fields.push(region(field.clone(), shape.indent_level));
            fields.extend(comma.map(Document::Token));
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

fn has_comment_between_fields(source: &SourceFileView<'_>, view: &StructLiteralView) -> bool {
    view.tokens.clone().any(|index| {
        matches!(
            source.tokens[index].token.kind,
            TokenType::Comment(_) | TokenType::DocComment(_)
        ) && !view.fields.iter().any(|field| field.contains(&index))
    })
}
