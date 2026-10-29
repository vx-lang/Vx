use super::context::FormatContext;
use super::layout::LayoutPlan;
use super::rules::function_signatures::FunctionSignatures;
use crate::lexer::TokenType;
use std::collections::HashSet;
use vx_config::FormatConfig;

pub(super) trait FormatRule: Sync {
    fn name(&self) -> &'static str;

    fn enabled(&self, _config: &FormatConfig) -> bool {
        true
    }

    fn starts_at(&self, _token: &TokenType<'_>) -> bool {
        true
    }

    fn apply(&self, context: &FormatContext<'_, '_>, layout: &mut LayoutPlan);
}

/// Registration order is explicit. Competing proposals at the same position are errors.
pub(super) struct RuleRegistry<'a> {
    rules: &'a [&'a dyn FormatRule],
}

impl<'a> RuleRegistry<'a> {
    pub fn new(rules: &'a [&'a dyn FormatRule]) -> Self {
        let mut names = HashSet::new();
        for rule in rules {
            assert!(!rule.name().is_empty(), "format rules must have names");
            assert!(
                names.insert(rule.name()),
                "duplicate format rule {}",
                rule.name()
            );
        }
        Self { rules }
    }

    pub fn plan(&self, context: &FormatContext<'_, '_>) -> LayoutPlan {
        let mut layout = LayoutPlan::default();
        if !context.can_format(context.index..context.index + 1) {
            return layout;
        }
        for rule in self.rules {
            if rule.enabled(context.config) && rule.starts_at(&context.tokens[context.index].kind) {
                rule.apply(context, &mut layout);
            }
        }
        layout
    }

    pub fn has_candidate(&self, token: &TokenType<'_>) -> bool {
        self.rules.iter().any(|rule| rule.starts_at(token))
    }
}

pub(super) fn builtin_rules() -> RuleRegistry<'static> {
    static RULES: &[&dyn FormatRule] = &[&FunctionSignatures];
    RuleRegistry::new(RULES)
}
