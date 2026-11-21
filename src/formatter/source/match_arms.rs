//! The `match_arms` rule: each arm of a `match` starts a line of its own.

use super::dispatch::{NodeContext, NodeFormatter, Rule, SyntaxKind, SyntaxNode};
use super::document::{Document, Shape};
use super::{CoverageWarning, SourceFileView, TokenId};
use crate::lexer::TokenType;
use std::ops::Range;

pub(super) struct MatchArmsFormatter;

impl NodeFormatter for MatchArmsFormatter {
    fn kind(&self) -> SyntaxKind {
        SyntaxKind::Match
    }

    fn rule(&self) -> Option<Rule> {
        Some(Rule {
            name: "match_arms",
            enabled_by_default: |_| true,
        })
    }

    fn roots<'a>(&self, source: &'a SourceFileView<'_>) -> Vec<SyntaxNode<'a>> {
        source.matches.iter().map(SyntaxNode::Match).collect()
    }

    fn format(
        &self,
        node: SyntaxNode<'_>,
        context: &NodeContext<'_, '_, '_>,
        shape: Shape,
    ) -> Result<Document, CoverageWarning> {
        let source = context.source;
        let SyntaxNode::Match(view) = node else {
            return Err(CoverageWarning {
                bytes: source.token_bytes(node.tokens()),
                message: "Unsupported syntax ownership; preserve the match".into(),
            });
        };
        let region = |tokens: Range<usize>, indent_level: usize| Document::Region {
            bytes: source.token_bytes(tokens),
            indent_level,
        };
        // A comment next to the value or next to a comma has no clear line of its own. Leave
        // such a match to the base formatting.
        let unusual_comment = has_comment(source, view.keyword.0 + 1..view.scrutinee.start)
            || has_comment(source, view.scrutinee.end..view.open.0)
            || view.arms.iter().zip(&view.commas).any(|(arm, comma)| {
                comma.is_some_and(|comma| has_comment(source, arm.end..comma.0))
            });
        if unusual_comment {
            return Ok(region(view.tokens.clone(), shape.indent_level));
        }
        let outer = shape.indent_level;
        let inner = outer + 1;
        let mut document = vec![
            Document::Token(view.keyword),
            Document::Space,
            region(view.scrutinee.clone(), outer),
            Document::Space,
            Document::Token(view.open),
        ];
        let mut previous_end = view.open.0 + 1;
        for (index, (arm, comma)) in view.arms.iter().zip(&view.commas).enumerate() {
            let gap = gap(source, previous_end..arm.start);
            push_comments(&mut document, &gap.comments, inner);
            // Keep one blank line between arms, but none after the `{`.
            if gap.newlines_after_comments >= 2 && (index > 0 || !gap.comments.is_empty()) {
                document.push(Document::Line { indent_level: 0 });
            }
            document.push(Document::Line {
                indent_level: inner,
            });
            document.push(region(arm.clone(), inner));
            document.extend(comma.map(Document::Token));
            previous_end = comma.map_or(arm.end, |comma| comma.0 + 1);
        }
        let last = gap(source, previous_end..view.close.0);
        push_comments(&mut document, &last.comments, inner);
        document.push(Document::Line {
            indent_level: outer,
        });
        document.push(Document::Token(view.close));
        Ok(Document::Concat(document))
    }
}

/// The comments in the space between two parts of a match.
struct Gap {
    /// Each comment, with the number of line breaks before it.
    comments: Vec<(TokenId, usize)>,
    /// The number of line breaks after the last comment, or in the whole gap without one.
    newlines_after_comments: usize,
}

fn gap(source: &SourceFileView<'_>, tokens: Range<usize>) -> Gap {
    let mut comments = Vec::new();
    let mut newlines = 0;
    for index in tokens {
        match &source.tokens[index].token.kind {
            TokenType::Whitespace(text) => newlines += text.matches('\n').count(),
            TokenType::Comment(_) | TokenType::DocComment(_) => {
                comments.push((TokenId(index), newlines));
                newlines = 0;
            }
            other => unreachable!("a gap between match arms holds {other:?}"),
        }
    }
    Gap {
        comments,
        newlines_after_comments: newlines,
    }
}

/// A comment on the line of the code before it stays there. Any other comment gets a line of
/// its own, and keeps one blank line before it.
fn push_comments(document: &mut Vec<Document>, comments: &[(TokenId, usize)], indent_level: usize) {
    for (comment, newlines) in comments {
        if *newlines == 0 {
            document.push(Document::Space);
        } else {
            if *newlines >= 2 {
                document.push(Document::Line { indent_level: 0 });
            }
            document.push(Document::Line { indent_level });
        }
        document.push(Document::Token(*comment));
    }
}

fn has_comment(source: &SourceFileView<'_>, tokens: Range<usize>) -> bool {
    tokens.into_iter().any(|index| {
        matches!(
            source.tokens[index].token.kind,
            TokenType::Comment(_) | TokenType::DocComment(_)
        )
    })
}
