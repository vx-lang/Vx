use super::document::{Document, Shape};
use super::{
    CoverageWarning, FunctionHeaderView, IfValueView, MatchView, ParameterView, SourceFileView,
    StructLiteralView, TypeView,
};
use std::collections::HashSet;
use std::ops::Range;
use vx_config::{FormatConfig, StyleEdition};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SyntaxKind {
    Header,
    ParameterList,
    Parameter,
    Type,
    StructLiteral,
    Match,
    IfValue,
}

#[derive(Clone, Copy)]
pub enum SyntaxNode<'a> {
    Header(&'a FunctionHeaderView),
    ParameterList(&'a FunctionHeaderView),
    Parameter(&'a ParameterView),
    Type(&'a TypeView),
    StructLiteral(&'a StructLiteralView),
    Match(&'a MatchView),
    IfValue(&'a IfValueView),
}

impl SyntaxNode<'_> {
    pub fn kind(self) -> SyntaxKind {
        match self {
            Self::Header(_) => SyntaxKind::Header,
            Self::ParameterList(_) => SyntaxKind::ParameterList,
            Self::Parameter(_) => SyntaxKind::Parameter,
            Self::Type(_) => SyntaxKind::Type,
            Self::StructLiteral(_) => SyntaxKind::StructLiteral,
            Self::Match(_) => SyntaxKind::Match,
            Self::IfValue(_) => SyntaxKind::IfValue,
        }
    }
    pub fn tokens(self) -> Range<usize> {
        match self {
            Self::Header(view) => view.tokens.clone(),
            Self::ParameterList(view) => view.parameter_open.0..view.parameter_close.0 + 1,
            Self::Parameter(view) => view.tokens.clone(),
            Self::Type(view) => view.tokens.clone(),
            Self::StructLiteral(view) => view.tokens.clone(),
            Self::Match(view) => view.tokens.clone(),
            Self::IfValue(view) => view.tokens.clone(),
        }
    }
}

/// A formatting rule that a project can switch on or off in `[format.rules]`.
#[derive(Clone, Copy, Debug)]
pub struct Rule {
    /// The name in `[format.rules]`.
    pub name: &'static str,
    /// Whether the rule is on when the configuration does not name it.
    pub enabled_by_default: fn(StyleEdition) -> bool,
}

pub trait NodeFormatter: Sync {
    fn kind(&self) -> SyntaxKind;
    /// The rule that switches this formatter on or off. `None` for a formatter that only
    /// formats parts of a larger node: its parent's rule decides.
    fn rule(&self) -> Option<Rule> {
        None
    }
    fn roots<'a>(&self, _source: &'a SourceFileView<'_>) -> Vec<SyntaxNode<'a>> {
        Vec::new()
    }
    /// The code that this formatter's rule cannot format yet. The engine keeps it as written.
    fn warnings(&self, _source: &SourceFileView<'_>) -> Vec<CoverageWarning> {
        Vec::new()
    }
    fn format(
        &self,
        node: SyntaxNode<'_>,
        context: &NodeContext<'_, '_, '_>,
        shape: Shape,
    ) -> Result<Document, CoverageWarning>;
}

pub struct NodeRegistry<'a> {
    formatters: &'a [&'a dyn NodeFormatter],
}

impl<'a> NodeRegistry<'a> {
    /// The roots of the formatters whose rule `config` switches on. The code of a switched-off
    /// rule gets the formatting it would get if the rule did not exist.
    pub fn enabled_roots<'source>(
        &self,
        source: &'source SourceFileView<'_>,
        config: &FormatConfig,
    ) -> Vec<SyntaxNode<'source>> {
        self.formatters
            .iter()
            .filter(|formatter| enabled(**formatter, config))
            .flat_map(|formatter| {
                let nodes = formatter.roots(source);
                assert!(
                    nodes.iter().all(|node| node.kind() == formatter.kind()),
                    "A root node must belong to its registered syntax formatter"
                );
                nodes
            })
            .collect()
    }

    /// The warnings of the formatters whose rule `config` switches on.
    pub fn enabled_warnings(
        &self,
        source: &SourceFileView<'_>,
        config: &FormatConfig,
    ) -> Vec<CoverageWarning> {
        self.formatters
            .iter()
            .filter(|formatter| enabled(**formatter, config))
            .flat_map(|formatter| formatter.warnings(source))
            .collect()
    }

    pub fn new(formatters: &'a [&'a dyn NodeFormatter]) -> Result<Self, String> {
        let mut kinds = HashSet::new();
        let mut rules = HashSet::new();
        for formatter in formatters {
            if !kinds.insert(formatter.kind()) {
                return Err("A syntax kind has more than one owning formatter".into());
            }
            if let Some(rule) = formatter.rule() {
                if !rules.insert(rule.name) {
                    return Err(format!(
                        "Two formatters share the rule name `{}`",
                        rule.name
                    ));
                }
            }
        }
        Ok(Self { formatters })
    }

    /// The rules of the registered formatters, in registration order.
    pub fn rules(&self) -> Vec<Rule> {
        self.formatters
            .iter()
            .filter_map(|formatter| formatter.rule())
            .collect()
    }

    /// An error for each rule that `config` names but no formatter has.
    pub fn check_rule_names(&self, config: &FormatConfig) -> Result<(), String> {
        let known = self.rules();
        for name in config.rules.keys() {
            if !known.iter().any(|rule| rule.name == name) {
                let names: Vec<_> = known.iter().map(|rule| rule.name).collect();
                return Err(format!(
                    "unknown formatting rule `{name}` in [format.rules]; the known rules are: {}",
                    names.join(", ")
                ));
            }
        }
        Ok(())
    }

    pub fn format(
        &self,
        node: SyntaxNode<'_>,
        source: &SourceFileView<'_>,
        config: &FormatConfig,
        shape: Shape,
    ) -> Result<Document, CoverageWarning> {
        let Some(formatter) = self
            .formatters
            .iter()
            .find(|formatter| formatter.kind() == node.kind())
        else {
            return Err(CoverageWarning {
                bytes: source.token_bytes(node.tokens()),
                message: "Unsupported syntax kind; preserve the enclosing structure".into(),
            });
        };
        if !enabled(*formatter, config) {
            return Ok(Document::Region {
                bytes: source.token_bytes(node.tokens()),
                indent_level: shape.indent_level,
            });
        }
        formatter.format(
            node,
            &NodeContext {
                source,
                config,
                registry: self,
            },
            shape,
        )
    }
}

fn enabled(formatter: &dyn NodeFormatter, config: &FormatConfig) -> bool {
    formatter.rule().is_none_or(|rule| {
        config.rule_enabled(rule.name, (rule.enabled_by_default)(config.style_edition))
    })
}

pub struct NodeContext<'a, 'source, 'rules> {
    pub source: &'a SourceFileView<'source>,
    pub config: &'a FormatConfig,
    pub registry: &'a NodeRegistry<'rules>,
}

impl NodeContext<'_, '_, '_> {
    pub fn child(&self, node: SyntaxNode<'_>, shape: Shape) -> Result<Document, CoverageWarning> {
        self.registry
            .format(node, self.source, self.config, shape)
            .map(|child| Document::WithShape {
                shape,
                child: Box::new(child),
            })
    }
}

pub struct FormatOutcome {
    pub text: String,
    /// Code that the formatter kept as written because it does not support it yet.
    pub warnings: Vec<CoverageWarning>,
}
