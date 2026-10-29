use super::dispatch::{Coverage, FormatOutcome, NodeRegistry, SyntaxNode};
use super::document::{self, RenderState, Shape};
use super::ownership::{Owner, OwnershipMap};
use super::{builtin_registry, overlaps, CoverageWarning, SourceFileView};
use crate::lexer::TokenType;
use std::ops::Range;
use vx_config::FormatConfig;

#[derive(Clone)]
enum Work<'a> {
    Syntax(SyntaxNode<'a>),
    Preserve,
}

#[derive(Clone)]
struct Unit<'a> {
    bytes: Range<usize>,
    work: Work<'a>,
}

pub(crate) fn format_source(text: &str, config: &FormatConfig) -> FormatOutcome {
    format_source_with_registry(text, config, &builtin_registry())
}

pub(crate) fn format_source_with_registry(
    text: &str,
    config: &FormatConfig,
    registry: &NodeRegistry<'_>,
) -> FormatOutcome {
    let source = SourceFileView::parse(text);
    if source.exclusions.contains(&(0..text.len())) {
        return FormatOutcome {
            text: text.to_string(),
            coverage: Coverage::Complete,
            warnings: Vec::new(),
        };
    }
    let mut warnings = source.warnings.clone();
    if warnings
        .iter()
        .any(|warning| warning.bytes == (0..text.len()))
    {
        return preserved(text, warnings);
    }
    let result = render_file(&source, config, registry, &mut warnings);
    match result {
        Ok(output) => FormatOutcome {
            text: output,
            coverage: if warnings.is_empty() {
                Coverage::Complete
            } else {
                Coverage::Partial
            },
            warnings,
        },
        Err(message) => {
            warnings.push(CoverageWarning {
                bytes: 0..text.len(),
                message: format!("{message}; preserved the file unchanged"),
            });
            preserved(text, warnings)
        }
    }
}

fn preserved(text: &str, mut warnings: Vec<CoverageWarning>) -> FormatOutcome {
    warnings.sort_by_key(|warning| (warning.bytes.start, warning.bytes.end));
    warnings.dedup();
    FormatOutcome {
        text: text.to_string(),
        coverage: Coverage::Partial,
        warnings,
    }
}

fn line_envelope(source: &SourceFileView<'_>, bytes: &Range<usize>) -> Range<usize> {
    let mut start = source.text[..bytes.start]
        .rfind('\n')
        .map_or(0, |offset| offset + 1);
    let mut end = source.text[bytes.end..]
        .find('\n')
        .map_or(source.text.len(), |offset| bytes.end + offset + 1);
    // A multiline literal can cross a line boundary. Never split its token.
    for token in &source.tokens {
        if matches!(token.token.kind, TokenType::Whitespace(_)) {
            continue;
        }
        if token.bytes.start < start && start < token.bytes.end {
            start = token.bytes.start;
        }
        if token.bytes.start < end && end < token.bytes.end {
            end = token.bytes.end;
        }
    }
    start..end
}

fn render_file(
    source: &SourceFileView<'_>,
    config: &FormatConfig,
    registry: &NodeRegistry<'_>,
    warnings: &mut Vec<CoverageWarning>,
) -> Result<String, String> {
    let mut raw: Vec<_> = source.exclusions.clone();
    raw.extend(
        source
            .macros
            .iter()
            .map(|view| line_envelope(source, &view.bytes)),
    );
    raw.extend(
        warnings
            .iter()
            .map(|warning| line_envelope(source, &warning.bytes)),
    );
    raw.sort_by_key(|range| (range.start, range.end));
    let mut merged: Vec<Range<usize>> = Vec::new();
    for range in raw {
        if let Some(previous) = merged.last_mut() {
            if range.start <= previous.end {
                previous.end = previous.end.max(range.end);
                continue;
            }
        }
        merged.push(range);
    }
    let mut ownership = OwnershipMap::from_source(source);
    // Additional protection is installed in a fresh map to avoid overlapping opaque envelopes.
    ownership.protect_union(&merged)?;
    let mut units: Vec<_> = merged
        .iter()
        .cloned()
        .map(|bytes| Unit {
            bytes,
            work: Work::Preserve,
        })
        .collect();
    let mut roots = registry.roots(source);
    roots.sort_by_key(|node| {
        let bytes = source.token_bytes(node.tokens());
        (bytes.start, std::cmp::Reverse(bytes.end))
    });
    let mut previous_root: Option<Range<usize>> = None;
    for node in roots {
        let bytes = source.token_bytes(node.tokens());
        if let Some(previous) = &previous_root {
            if bytes == *previous {
                return Err("Competing syntax owners for one source range".into());
            }
            if previous.start <= bytes.start && bytes.end <= previous.end {
                continue;
            }
            if overlaps(previous, &bytes) {
                return Err("Partially overlapping syntax roots".into());
            }
        }
        previous_root = Some(bytes.clone());
        if bytes.is_empty() || merged.iter().any(|range| overlaps(range, &bytes)) {
            continue;
        }
        ownership.claim(bytes.clone(), Owner::Syntax)?;
        units.push(Unit {
            bytes,
            work: Work::Syntax(node),
        });
    }
    units.sort_by_key(|unit| unit.bytes.start);
    let mut output = String::new();
    let shape = Shape {
        column: 0,
        indent_level: 0,
        indent_spaces: config.indent,
        width: config.fn_signature_width,
    };
    let mut state = RenderState::new(shape);
    let mut cursor = 0;
    for unit in units {
        if unit.bytes.start < cursor {
            return Err("Overlapping formatter ownership".into());
        }
        if cursor < unit.bytes.start {
            output.push_str(&super::super::legacy::render(
                source,
                &ownership,
                cursor..unit.bytes.start,
                config,
                &mut state,
            )?);
        }
        match unit.work {
            Work::Preserve => {
                output.push_str(&source.text[unit.bytes.clone()]);
                state.consume_preserved(source, unit.bytes.clone())?;
            }
            Work::Syntax(node) => {
                if state.column == 0 {
                    let indent = " ".repeat(state.indent_level * config.indent);
                    state.emitted(&indent);
                    output.push_str(&indent);
                }
                let shape = Shape {
                    column: state.column,
                    indent_level: state.indent_level,
                    ..shape
                };
                match registry.format(node, source, config, shape) {
                    Ok(layout) => {
                        let rendered = document::render(
                            source,
                            &ownership,
                            &layout,
                            unit.bytes.clone(),
                            shape,
                            state.clone(),
                        )?;
                        output.push_str(&rendered.text);
                        state = rendered.exit_state;
                    }
                    Err(warning) => {
                        // A failed owner preserves its source rather than guessing a layout.
                        warnings.push(warning);
                        output.push_str(&source.text[unit.bytes.clone()]);
                        state.consume_preserved(source, unit.bytes.clone())?;
                    }
                }
            }
        }
        cursor = unit.bytes.end;
    }
    if cursor < source.text.len() {
        output.push_str(&super::super::legacy::render(
            source,
            &ownership,
            cursor..source.text.len(),
            config,
            &mut state,
        )?);
    }
    if !state.delimiters.is_empty() {
        return Err("Incomplete renderer state after formatting".into());
    }
    source.verify_preservation(&output)?;
    Ok(output)
}
