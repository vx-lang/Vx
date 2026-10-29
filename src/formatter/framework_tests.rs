use super::context::{FormatContext, SyntaxContext};
use super::layout::{LayoutEdit, LayoutPiece, LayoutPlan};
use super::registry::{FormatRule, RuleRegistry};
use crate::lexer::{Lexer, TokenType};
use std::sync::Mutex;
use vx_config::FormatConfig;

fn with_context(source: &str, check: impl FnOnce(&FormatContext<'_, '_>)) {
    let tokens = Lexer::new_with_comments(source).tokenize();
    let syntax = SyntaxContext::new(&tokens);
    check(&FormatContext {
        tokens: &tokens,
        syntax: &syntax,
        config: &FormatConfig::default(),
        index: 0,
        column: 0,
        indent_level: 0,
    });
}

/// An independent rule proves the registry needs no function-specific dispatch.
struct SeparateTokens(&'static str);

impl FormatRule for SeparateTokens {
    fn name(&self) -> &'static str {
        self.0
    }

    fn apply(&self, context: &FormatContext<'_, '_>, layout: &mut LayoutPlan) {
        let end = context.tokens.len() - 1;
        let mut pieces = Vec::new();
        for index in 0..end {
            if !matches!(context.tokens[index].kind, TokenType::Whitespace(_)) {
                if !pieces.is_empty() {
                    pieces.push(LayoutPiece::Space);
                }
                pieces.push(LayoutPiece::Token(index));
            }
        }
        layout.propose(
            self.name(),
            context,
            LayoutEdit {
                range: 0..end,
                pieces,
            },
        );
    }
}

#[test]
fn a_new_rule_can_format_through_the_registry_and_shared_renderer() {
    let source = "let value=1;";
    let tokens = Lexer::new_with_comments(source).tokenize();
    let rule = SeparateTokens("separate_tokens");
    let rules: &[&dyn FormatRule] = &[&rule];
    let output = super::emit_with_rules(
        tokens,
        &FormatConfig::default(),
        source.len(),
        &RuleRegistry::new(rules),
    );
    assert_eq!(output, "let value = 1 ;");
}

#[test]
fn shared_context_matches_nested_delimiters() {
    with_context("([])", |context| {
        assert_eq!(context.syntax.matching_delimiter(0), Some(3));
        assert_eq!(context.syntax.matching_delimiter(1), Some(2));
    });
}

#[test]
fn a_rule_cannot_replace_a_range_containing_a_disabled_region() {
    with_context(
        "let value=1;\n// vx-format-begin: OFF\nlet other=2;\n// vx-format-end: OFF",
        |context| {
            let rule = SeparateTokens("separate_tokens");
            assert!(RuleRegistry::new(&[&rule])
                .plan(context)
                .into_edit()
                .is_none());
        },
    );
}

#[test]
fn a_rule_cannot_format_a_disabled_file() {
    with_context("//! vx-format: OFF\nlet value=1;", |context| {
        let rule = SeparateTokens("separate_tokens");
        assert!(RuleRegistry::new(&[&rule])
            .plan(context)
            .into_edit()
            .is_none());
    });
}

struct RecordOrder<'a> {
    name: &'static str,
    enabled: bool,
    calls: &'a Mutex<Vec<&'static str>>,
}

impl FormatRule for RecordOrder<'_> {
    fn name(&self) -> &'static str {
        self.name
    }
    fn enabled(&self, _: &FormatConfig) -> bool {
        self.enabled
    }
    fn apply(&self, _: &FormatContext<'_, '_>, _: &mut LayoutPlan) {
        self.calls.lock().unwrap().push(self.name);
    }
}

#[test]
fn rules_run_in_registration_order_and_disabled_rules_do_not_run() {
    with_context("let value=1;", |context| {
        let calls = Mutex::new(Vec::new());
        let first = RecordOrder {
            name: "first",
            enabled: true,
            calls: &calls,
        };
        let disabled = RecordOrder {
            name: "disabled",
            enabled: false,
            calls: &calls,
        };
        let last = RecordOrder {
            name: "last",
            enabled: true,
            calls: &calls,
        };
        RuleRegistry::new(&[&first, &disabled, &last]).plan(context);
        assert_eq!(*calls.lock().unwrap(), vec!["first", "last"]);
    });
}

#[test]
#[should_panic(expected = "duplicate format rule")]
fn duplicate_rule_names_are_rejected() {
    let first = SeparateTokens("same_name");
    let second = SeparateTokens("same_name");
    RuleRegistry::new(&[&first, &second]);
}

#[test]
#[should_panic(expected = "overlapping layouts")]
fn competing_layouts_are_rejected() {
    with_context("let value=1;", |context| {
        let first = SeparateTokens("first");
        let second = SeparateTokens("second");
        RuleRegistry::new(&[&first, &second]).plan(context);
    });
}

#[test]
#[should_panic(expected = "removed source tokens")]
fn layouts_cannot_remove_tokens() {
    with_context("value;", |context| {
        LayoutPlan::default().propose(
            "broken_rule",
            context,
            LayoutEdit {
                range: 0..2,
                pieces: vec![LayoutPiece::Token(0)],
            },
        );
    });
}

#[test]
#[should_panic(expected = "changed source token order")]
fn layouts_cannot_reorder_tokens() {
    with_context("value;", |context| {
        LayoutPlan::default().propose(
            "broken_rule",
            context,
            LayoutEdit {
                range: 0..2,
                pieces: vec![LayoutPiece::Token(1), LayoutPiece::Token(0)],
            },
        );
    });
}

#[test]
#[should_panic(expected = "complete delimiter pairs")]
fn layouts_cannot_consume_only_half_of_a_delimiter_pair() {
    with_context("(value)", |context| {
        LayoutPlan::default().propose(
            "broken_rule",
            context,
            LayoutEdit {
                range: 0..2,
                pieces: vec![LayoutPiece::Token(0), LayoutPiece::Token(1)],
            },
        );
    });
}

struct MergeReferences;

impl FormatRule for MergeReferences {
    fn name(&self) -> &'static str {
        "merge_references"
    }
    fn starts_at(&self, token: &TokenType<'_>) -> bool {
        matches!(token, TokenType::Ampersand)
    }
    fn apply(&self, context: &FormatContext<'_, '_>, layout: &mut LayoutPlan) {
        layout.propose(
            self.name(),
            context,
            LayoutEdit {
                range: 0..3,
                pieces: vec![LayoutPiece::Token(0), LayoutPiece::Token(2)],
            },
        );
    }
}

#[test]
#[should_panic(expected = "changed the source tokens")]
fn whitespace_changes_cannot_merge_tokens_at_replacement_boundaries() {
    let source = "& &value";
    let tokens = Lexer::new_with_comments(source).tokenize();
    super::emit_with_rules(
        tokens,
        &FormatConfig::default(),
        source.len(),
        &RuleRegistry::new(&[&MergeReferences]),
    );
}
