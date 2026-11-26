//! The `if_values` rule: a short `if` value, such as `let a = if c { 1 } else { 0 };`, stays on
//! one line.

use super::dispatch::{NodeContext, NodeFormatter, Rule, SyntaxKind, SyntaxNode};
use super::document::{Document, Shape};
use super::{CoverageWarning, SourceFileView};
use crate::lexer::TokenType;
use std::ops::Range;

pub(super) struct IfValueFormatter;

impl NodeFormatter for IfValueFormatter {
    fn kind(&self) -> SyntaxKind {
        SyntaxKind::IfValue
    }

    fn rule(&self) -> Option<Rule> {
        Some(Rule {
            name: "if_values",
            enabled_by_default: |_| true,
        })
    }

    fn roots<'a>(&self, source: &'a SourceFileView<'_>) -> Vec<SyntaxNode<'a>> {
        source.if_values.iter().map(SyntaxNode::IfValue).collect()
    }

    fn format(
        &self,
        node: SyntaxNode<'_>,
        context: &NodeContext<'_, '_, '_>,
        shape: Shape,
    ) -> Result<Document, CoverageWarning> {
        let source = context.source;
        let SyntaxNode::IfValue(view) = node else {
            return Err(CoverageWarning {
                bytes: source.token_bytes(node.tokens()),
                message: "Unsupported syntax ownership; preserve the if value".into(),
            });
        };
        let region = |tokens: Range<usize>| Document::Region {
            bytes: source.token_bytes(tokens),
            indent_level: shape.indent_level,
        };
        // The base formatting gives each branch lines of its own.
        let expanded = region(view.tokens.clone());
        // A comment ends its line, so an `if` that holds one cannot be on one line. A branch
        // with a block, a statement, an `if`, a `match` or a `return` keeps its lines too.
        let comment = |index: usize| {
            matches!(
                source.tokens[index].token.kind,
                TokenType::Comment(_) | TokenType::DocComment(_)
            )
        };
        let complex = |index: usize| {
            matches!(
                source.tokens[index].token.kind,
                TokenType::LeftBrace
                    | TokenType::RightBrace
                    | TokenType::Semicolon
                    | TokenType::If
                    | TokenType::Match
                    | TokenType::Return
            )
        };
        if view.tokens.clone().any(comment)
            || view
                .branches
                .iter()
                .any(|(_, value, _)| value.clone().any(complex))
        {
            return Ok(expanded);
        }
        let mut flat = vec![
            Document::Token(super::TokenId(view.tokens.start)),
            Document::Space,
            region(view.condition.clone()),
        ];
        for (index, (open, value, close)) in view.branches.iter().enumerate() {
            if index == 1 {
                flat.extend([Document::Space, Document::Token(view.else_keyword)]);
            }
            flat.extend([
                Document::Space,
                Document::Token(*open),
                Document::Space,
                region(value.clone()),
                Document::Space,
                Document::Token(*close),
            ]);
        }
        Ok(Document::Choice {
            flat: Box::new(Document::MaxWidth {
                width: context.config.single_line_if_else_max_width,
                child: Box::new(Document::Concat(flat)),
            }),
            expanded: Box::new(expanded),
        })
    }
}
