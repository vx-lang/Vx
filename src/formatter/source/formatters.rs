use super::dispatch::{
    Coverage, FormatOutcome, NodeContext, NodeFormatter, NodeRegistry, SyntaxKind, SyntaxNode,
};
use super::document::{self, Document, RenderState, Shape};
use super::ownership::{Owner, OwnershipMap};
use super::{CoverageWarning, GapConstraint, SourceFileView, TokenId};
use crate::lexer::TokenType;
use std::ops::Range;
use vx_config::FormatConfig;

struct FunctionFormatter;
struct HeaderFormatter;
struct ParametersFormatter;
struct ParameterFormatter;
struct TypeFormatter;

pub fn builtin_registry() -> NodeRegistry<'static> {
    static FORMATTERS: &[&dyn NodeFormatter] = &[
        &FunctionFormatter,
        &HeaderFormatter,
        &ParametersFormatter,
        &ParameterFormatter,
        &TypeFormatter,
    ];
    NodeRegistry::new(FORMATTERS).expect("built-in syntax owners must be unique")
}

/// Format one recognized declaration. Failures preserve its original bytes and become warnings.
pub fn format_declaration(
    source: &SourceFileView<'_>,
    index: usize,
    config: &FormatConfig,
    entry: RenderState,
) -> FormatOutcome {
    let declaration = &source.functions[index];
    let original = &source.text[declaration.bytes.clone()];
    let attempt = || -> Result<String, CoverageWarning> {
        if let Some(warning) = source
            .warnings
            .iter()
            .find(|warning| super::overlaps(&warning.bytes, &declaration.bytes))
        {
            return Err(warning.clone());
        }
        let mut ownership = OwnershipMap::from_source(source);
        ownership
            .claim(declaration.bytes.clone(), Owner::Syntax)
            .map_err(|message| CoverageWarning {
                bytes: declaration.bytes.clone(),
                message,
            })?;
        let shape = Shape {
            column: entry.column,
            indent_level: entry.indent_level,
            indent_spaces: config.indent,
            width: config.fn_signature_width,
        };
        let layout =
            builtin_registry().format(SyntaxNode::Function(declaration), source, config, shape)?;
        document::render(
            source,
            &ownership,
            &layout,
            declaration.bytes.clone(),
            shape,
            entry.clone(),
        )
        .map(|rendered| rendered.text)
        .map_err(|message| CoverageWarning {
            bytes: declaration.bytes.clone(),
            message,
        })
    };
    match attempt() {
        Ok(text) => FormatOutcome {
            text,
            coverage: Coverage::Complete,
            warnings: Vec::new(),
        },
        Err(warning) => FormatOutcome {
            text: original.to_string(),
            coverage: Coverage::Partial,
            warnings: vec![warning],
        },
    }
}

fn unsupported(context: &NodeContext<'_, '_, '_>, node: SyntaxNode<'_>) -> CoverageWarning {
    CoverageWarning {
        bytes: context.source.token_bytes(node.tokens()),
        message: "Unsupported syntax ownership; preserve the enclosing declaration".into(),
    }
}

impl NodeFormatter for FunctionFormatter {
    fn kind(&self) -> SyntaxKind {
        SyntaxKind::Function
    }
    fn format(
        &self,
        node: SyntaxNode<'_>,
        context: &NodeContext<'_, '_, '_>,
        shape: Shape,
    ) -> Result<Document, CoverageWarning> {
        let SyntaxNode::Function(view) = node else {
            return Err(unsupported(context, node));
        };
        let header = context.child(SyntaxNode::Header(&view.header), shape)?;
        let body = context
            .source
            .token_bytes(view.header.terminator.0..view.tokens.end);
        Ok(Document::Concat(vec![
            header,
            Document::Preserve {
                bytes: body,
                keep_column: false,
            },
        ]))
    }
}

impl NodeFormatter for HeaderFormatter {
    fn enabled(&self, config: &FormatConfig) -> bool {
        config.format_function_headers
    }

    fn kind(&self) -> SyntaxKind {
        SyntaxKind::Header
    }
    fn roots<'a>(&self, source: &'a SourceFileView<'_>) -> Vec<SyntaxNode<'a>> {
        source
            .functions
            .iter()
            .map(|function| SyntaxNode::Header(&function.header))
            .collect()
    }
    fn format(
        &self,
        node: SyntaxNode<'_>,
        context: &NodeContext<'_, '_, '_>,
        shape: Shape,
    ) -> Result<Document, CoverageWarning> {
        let SyntaxNode::Header(view) = node else {
            return Err(unsupported(context, node));
        };
        let source = context.source;
        let bytes = source.token_bytes(view.tokens.clone());
        if source.tokens[view.tokens.clone()].iter().any(|token| {
            matches!(
                token.token.kind,
                TokenType::Comment(_) | TokenType::DocComment(_)
            )
        }) {
            return Ok(Document::Preserve {
                bytes,
                keep_column: false,
            });
        }
        if source.is_excluded(&bytes)
            || source
                .macros
                .get(
                    source
                        .macros
                        .partition_point(|view| view.bytes.end <= bytes.start),
                )
                .is_some_and(|view| super::overlaps(&view.bytes, &bytes))
        {
            let end = source.tokens[view.terminator.0].bytes.start;
            return Ok(Document::Preserve {
                bytes: bytes.start..end,
                keep_column: true,
            });
        }
        let (suffix, reserved) = header_suffix(context, view);
        let signature_shape = Shape {
            width: shape.width.saturating_sub(reserved),
            ..shape
        };
        let prefix = compact(context, view.tokens.start..view.parameter_open.0)?;
        let parameter_shape = Shape {
            column: shape.column
                + prefix
                    .flat_width(source)
                    .ok_or_else(|| unsupported(context, node))?,
            ..shape
        };
        let parameters = context.child(SyntaxNode::ParameterList(view), parameter_shape)?;
        let expanded_parameters = parameters.expanded();
        let flat_end = parameter_shape.column
            + parameters
                .flat_width(source)
                .ok_or_else(|| unsupported(context, node))?;
        let flat = Document::Concat(vec![
            prefix.clone(),
            parameters,
            header_tail(context, view, signature_shape, flat_end)?,
        ]);
        if view.parameters.is_empty() {
            return Ok(Document::Concat(vec![flat, suffix]));
        }
        let expanded_end = shape.indent_level * shape.indent_spaces + 1;
        let expanded = Document::Concat(vec![
            prefix,
            expanded_parameters,
            header_tail(context, view, signature_shape, expanded_end)?,
        ]);
        Ok(Document::Concat(vec![
            Document::WithWidth {
                width: signature_shape.width,
                child: Box::new(Document::Choice {
                    flat: Box::new(flat),
                    expanded: Box::new(expanded),
                }),
            },
            suffix,
        ]))
    }
}

fn header_suffix(
    context: &NodeContext<'_, '_, '_>,
    view: &super::FunctionHeaderView,
) -> (Document, usize) {
    let source = context.source;
    let has_contracts = view
        .clauses
        .iter()
        .any(|clause| clause.tokens.start >= view.return_type.tokens.end);
    if has_contracts {
        let start = source.tokens[view.return_type.tokens.end - 1].bytes.end;
        let end = source.tokens[view.terminator.0].bytes.start;
        let text = &source.text[start..end];
        let first_line = text.split('\n').next().unwrap_or_default();
        let reserved = first_line.chars().count() + usize::from(!text.contains('\n'));
        return (
            Document::Preserve {
                bytes: start..end,
                keep_column: false,
            },
            reserved,
        );
    }
    let brace = matches!(
        source.tokens[view.terminator.0].token.kind,
        TokenType::LeftBrace
    );
    let suffix = if brace {
        Document::Space
    } else {
        Document::Concat(Vec::new())
    };
    // Reserve the following delimiter, which lies outside the header's source range.
    (suffix, 1 + usize::from(brace))
}

fn header_tail(
    context: &NodeContext<'_, '_, '_>,
    view: &super::FunctionHeaderView,
    shape: Shape,
    column: usize,
) -> Result<Document, CoverageWarning> {
    if view.return_type.tokens.is_empty() {
        return Ok(Document::Concat(Vec::new()));
    }
    let before = compact(
        context,
        view.parameter_close.0 + 1..view.return_type.tokens.start,
    )?;
    let before_width = before
        .flat_width(context.source)
        .ok_or_else(|| unsupported(context, SyntaxNode::Header(view)))?;
    let return_shape = Shape {
        column: column + before_width + 2,
        ..shape
    };
    let return_type = context.child(SyntaxNode::Type(&view.return_type), return_shape)?;
    Ok(Document::Concat(vec![
        Document::Space,
        before,
        Document::Space,
        return_type,
    ]))
}

impl NodeFormatter for ParametersFormatter {
    fn kind(&self) -> SyntaxKind {
        SyntaxKind::ParameterList
    }
    fn format(
        &self,
        node: SyntaxNode<'_>,
        context: &NodeContext<'_, '_, '_>,
        shape: Shape,
    ) -> Result<Document, CoverageWarning> {
        let SyntaxNode::ParameterList(view) = node else {
            return Err(unsupported(context, node));
        };
        let mut flat = vec![Document::Token(view.parameter_open)];
        let mut expanded = vec![Document::Token(view.parameter_open)];
        let mut column = shape.column + 1;
        for (index, parameter) in view.parameters.iter().enumerate() {
            let inline =
                context.child(SyntaxNode::Parameter(parameter), Shape { column, ..shape })?;
            column += inline
                .flat_width(context.source)
                .ok_or_else(|| unsupported(context, node))?;
            flat.push(inline);
            expanded.push(Document::Line {
                indent_level: shape.indent_level + 1,
            });
            expanded.push(context.child(SyntaxNode::Parameter(parameter), shape.child_line())?);
            if index + 1 < view.parameters.len() {
                let comma = (parameter.tokens.end..view.parameters[index + 1].tokens.start)
                    .find(|token| {
                        matches!(context.source.tokens[*token].token.kind, TokenType::Comma)
                    })
                    .ok_or_else(|| unsupported(context, node))?;
                flat.extend([Document::Token(TokenId(comma)), Document::Space]);
                expanded.push(Document::Token(TokenId(comma)));
                column += 2;
            }
        }
        flat.push(Document::Token(view.parameter_close));
        if view.parameters.is_empty() {
            return Ok(Document::Concat(flat));
        }
        expanded.push(Document::Line {
            indent_level: shape.indent_level,
        });
        expanded.push(Document::Token(view.parameter_close));
        Ok(Document::Choice {
            flat: Box::new(Document::Concat(flat)),
            expanded: Box::new(Document::Concat(expanded)),
        })
    }
}

impl NodeFormatter for ParameterFormatter {
    fn kind(&self) -> SyntaxKind {
        SyntaxKind::Parameter
    }
    fn format(
        &self,
        node: SyntaxNode<'_>,
        context: &NodeContext<'_, '_, '_>,
        shape: Shape,
    ) -> Result<Document, CoverageWarning> {
        let SyntaxNode::Parameter(view) = node else {
            return Err(unsupported(context, node));
        };
        let prefix = compact(context, view.tokens.start..view.ty.tokens.start)?;
        let child_shape = Shape {
            column: shape.column + prefix.flat_width(context.source).unwrap_or(0) + 1,
            ..shape
        };
        let ty = context.child(SyntaxNode::Type(&view.ty), child_shape)?;
        Ok(Document::Concat(vec![prefix, Document::Space, ty]))
    }
}

impl NodeFormatter for TypeFormatter {
    fn kind(&self) -> SyntaxKind {
        SyntaxKind::Type
    }
    fn format(
        &self,
        node: SyntaxNode<'_>,
        context: &NodeContext<'_, '_, '_>,
        _: Shape,
    ) -> Result<Document, CoverageWarning> {
        let SyntaxNode::Type(view) = node else {
            return Err(unsupported(context, node));
        };
        compact(context, view.tokens.clone())
    }
}

fn compact(
    context: &NodeContext<'_, '_, '_>,
    range: Range<usize>,
) -> Result<Document, CoverageWarning> {
    let source = context.source;
    let mut children = Vec::new();
    let mut previous: Option<usize> = None;
    let mut whitespace = false;
    for index in range.clone() {
        let kind = &source.tokens[index].token.kind;
        if matches!(kind, TokenType::Whitespace(_)) {
            whitespace = true;
            continue;
        }
        if matches!(
            kind,
            TokenType::Comment(_) | TokenType::DocComment(_) | TokenType::Eof
        ) {
            return Err(CoverageWarning {
                bytes: source.token_bytes(range),
                message: "Unsupported comment layout; preserve the enclosing declaration".into(),
            });
        }
        if let Some(before) = previous {
            let previous_kind = &source.tokens[before].token.kind;
            let tight = matches!(
                kind,
                TokenType::Comma
                    | TokenType::Colon
                    | TokenType::DoubleColon
                    | TokenType::RightParen
                    | TokenType::RightBracket
                    | TokenType::RightAngle
            ) || (matches!(
                kind,
                TokenType::LeftParen | TokenType::LeftBracket | TokenType::LeftAngle
            ) && !matches!(
                previous_kind,
                TokenType::Colon | TokenType::Comma | TokenType::Arrow
            )) || matches!(
                previous_kind,
                TokenType::LeftParen
                    | TokenType::LeftBracket
                    | TokenType::LeftAngle
                    | TokenType::DoubleColon
                    | TokenType::Ampersand
                    | TokenType::Star
            );
            let spaced = whitespace
                || matches!(
                    previous_kind,
                    TokenType::Fn | TokenType::Comma | TokenType::Colon | TokenType::Arrow
                )
                || matches!(kind, TokenType::Arrow)
                || (matches!(previous_kind, TokenType::Ampersand)
                    && matches!(kind, TokenType::Ampersand));
            let constrained = source
                .gaps
                .get(source.gaps.partition_point(|gap| gap.left.0 < before))
                .filter(|gap| gap.left.0 == before && gap.right.0 == index)
                .map(|gap| gap.constraint);
            let insert_space = match constrained {
                Some(GapConstraint::MustTouch) => false,
                Some(GapConstraint::MustSeparate) => true,
                Some(GapConstraint::Flexible) | None if matches!(kind, TokenType::Colon) => {
                    context.config.space_before_colon
                }
                Some(GapConstraint::Flexible) | None => {
                    (spaced && !tight)
                        || (matches!(previous_kind, TokenType::Ampersand)
                            && matches!(kind, TokenType::Ampersand))
                }
            };
            if insert_space {
                children.push(Document::Space);
            }
        }
        children.push(Document::Token(TokenId(index)));
        previous = Some(index);
        whitespace = false;
    }
    Ok(Document::Concat(children))
}
