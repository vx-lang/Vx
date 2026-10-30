use super::document::{Document, Shape};
use super::{
    CoverageWarning, FunctionDeclView, FunctionHeaderView, ParameterView, SourceFileView, TypeView,
};
use std::collections::HashSet;
use std::ops::Range;
use vx_config::FormatConfig;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SyntaxKind {
    Function,
    Header,
    ParameterList,
    Parameter,
    Type,
}

#[derive(Clone, Copy)]
pub enum SyntaxNode<'a> {
    Function(&'a FunctionDeclView),
    Header(&'a FunctionHeaderView),
    ParameterList(&'a FunctionHeaderView),
    Parameter(&'a ParameterView),
    Type(&'a TypeView),
}

impl SyntaxNode<'_> {
    pub fn kind(self) -> SyntaxKind {
        match self {
            Self::Function(_) => SyntaxKind::Function,
            Self::Header(_) => SyntaxKind::Header,
            Self::ParameterList(_) => SyntaxKind::ParameterList,
            Self::Parameter(_) => SyntaxKind::Parameter,
            Self::Type(_) => SyntaxKind::Type,
        }
    }
    pub fn tokens(self) -> Range<usize> {
        match self {
            Self::Function(view) => view.tokens.clone(),
            Self::Header(view) => view.tokens.clone(),
            Self::ParameterList(view) => view.parameter_open.0..view.parameter_close.0 + 1,
            Self::Parameter(view) => view.tokens.clone(),
            Self::Type(view) => view.tokens.clone(),
        }
    }
}

pub trait NodeFormatter: Sync {
    fn kind(&self) -> SyntaxKind;
    fn enabled(&self, _config: &FormatConfig) -> bool {
        true
    }
    fn roots<'a>(&self, _source: &'a SourceFileView<'_>) -> Vec<SyntaxNode<'a>> {
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
    pub fn roots<'source>(&self, source: &'source SourceFileView<'_>) -> Vec<SyntaxNode<'source>> {
        self.formatters
            .iter()
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

    pub fn new(formatters: &'a [&'a dyn NodeFormatter]) -> Result<Self, String> {
        let mut kinds = HashSet::new();
        for formatter in formatters {
            if !kinds.insert(formatter.kind()) {
                return Err("A syntax kind has more than one owning formatter".into());
            }
        }
        Ok(Self { formatters })
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
        if !formatter.enabled(config) {
            return Ok(Document::Preserve {
                bytes: source.token_bytes(node.tokens()),
                keep_column: false,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Coverage {
    Complete,
    Partial,
}

pub struct FormatOutcome {
    pub text: String,
    pub coverage: Coverage,
    pub warnings: Vec<CoverageWarning>,
}

impl FormatOutcome {
    /// Warnings never change the exit status. Input and configuration failures remain caller errors.
    pub fn check_status(&self, original: &str) -> u8 {
        u8::from(self.text != original)
    }
}
