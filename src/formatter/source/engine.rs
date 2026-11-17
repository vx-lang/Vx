use super::dispatch::{Coverage, FormatOutcome, NodeRegistry, SyntaxNode};
use super::document::{self, RenderState, Shape};
use super::ownership::{Owner, OwnershipMap};
use super::{builtin_registry, overlaps, CoverageWarning, SourceFileView};
use crate::lexer::TokenType;
use std::cell::RefCell;
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
    for token in &source.tokens[source.token_range(&(start..end))] {
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
    let mut protected: Vec<Range<usize>> = Vec::new();
    for range in raw {
        if let Some(previous) = protected.last_mut() {
            if range.start <= previous.end {
                previous.end = previous.end.max(range.end);
                continue;
            }
        }
        protected.push(range);
    }
    let mut ownership = OwnershipMap::from_source(source);
    // Additional protection is installed in a fresh map to avoid overlapping opaque envelopes.
    ownership.protect_union(&protected)?;
    let mut roots: Vec<_> = registry
        .enabled_roots(source, config)
        .into_iter()
        .map(|node| (source.token_bytes(node.tokens()), node))
        .filter(|(bytes, _)| !bytes.is_empty())
        .collect();
    roots.sort_by_key(|(bytes, _)| (bytes.start, std::cmp::Reverse(bytes.end)));
    // Roots may nest, but two roots never share a range or overlap partly.
    let mut open: Vec<Range<usize>> = Vec::new();
    for (bytes, _) in &roots {
        while open.last().is_some_and(|outer| outer.end <= bytes.start) {
            open.pop();
        }
        if let Some(outer) = open.last() {
            if outer == bytes {
                return Err("Competing syntax owners for one source range".into());
            }
            if bytes.end > outer.end {
                return Err("Partially overlapping syntax roots".into());
            }
        }
        open.push(bytes.clone());
    }
    let mut renderer = FileRenderer {
        source,
        config,
        registry,
        ownership,
        protected,
        roots,
        warnings: RefCell::new(std::mem::take(warnings)),
    };
    for unit in renderer.units(0..source.text.len(), None) {
        if let Work::Syntax(_) = unit.work {
            renderer.ownership.claim(unit.bytes, Owner::Syntax)?;
        }
    }
    let mut state = RenderState::new(Shape {
        column: 0,
        indent_level: 0,
        indent_spaces: config.indent,
        width: config.max_width,
    });
    let output = renderer.render_range(0..source.text.len(), None, &mut state, false);
    *warnings = renderer.warnings.into_inner();
    let output = output?;
    if !state.delimiters.is_empty() {
        return Err("Incomplete renderer state after formatting".into());
    }
    source.verify_preservation(&output)?;
    Ok(output)
}

/// Formats one file: the syntax roots of the switched-on rules, the protected ranges, and the
/// base formatting for the code between them.
struct FileRenderer<'a, 'source> {
    source: &'a SourceFileView<'source>,
    config: &'a FormatConfig,
    registry: &'a NodeRegistry<'a>,
    ownership: OwnershipMap,
    /// Ranges kept exactly as written: macros, exclusions, and code with a coverage warning.
    protected: Vec<Range<usize>>,
    /// The roots of the switched-on rules, sorted by start, each outer root before its inner ones.
    roots: Vec<(Range<usize>, SyntaxNode<'a>)>,
    warnings: RefCell<Vec<CoverageWarning>>,
}

impl<'a> FileRenderer<'a, '_> {
    /// The protected ranges and the outermost roots inside `bytes`, sorted by start. `skip` is a
    /// root that hands its own range back to normal formatting, so it is not a unit again.
    fn units(&self, bytes: Range<usize>, skip: Option<&Range<usize>>) -> Vec<Unit<'a>> {
        let inside = |range: &Range<usize>| bytes.start <= range.start && range.end <= bytes.end;
        let mut units: Vec<Unit<'a>> = self
            .protected
            .iter()
            .filter(|range| inside(range))
            .map(|range| Unit {
                bytes: range.clone(),
                work: Work::Preserve,
            })
            .collect();
        let first = self
            .roots
            .partition_point(|(range, _)| range.start < bytes.start);
        let mut covered_until = bytes.start;
        for (range, node) in &self.roots[first..] {
            if range.start >= bytes.end {
                break;
            }
            if !inside(range) || range.start < covered_until || Some(range) == skip {
                continue;
            }
            let protected = self
                .protected
                .partition_point(|other| other.end <= range.start);
            if self
                .protected
                .get(protected)
                .is_some_and(|other| overlaps(other, range))
            {
                continue;
            }
            covered_until = range.end;
            units.push(Unit {
                bytes: range.clone(),
                work: Work::Syntax(*node),
            });
        }
        units.sort_by_key(|unit| unit.bytes.start);
        units
    }

    /// Format `bytes`, starting from `state`. `skip` is the root whose layout asked for this
    /// range, when the range is a region of that layout.
    fn render_range(
        &self,
        bytes: Range<usize>,
        skip: Option<&Range<usize>>,
        state: &mut RenderState,
        inside_syntax: bool,
    ) -> Result<String, String> {
        let mut output = String::new();
        let mut cursor = bytes.start;
        for unit in self.units(bytes.clone(), skip) {
            if unit.bytes.start < cursor {
                return Err("Overlapping formatter ownership".into());
            }
            if cursor < unit.bytes.start {
                output.push_str(&super::super::legacy::render(
                    self.source,
                    &self.ownership,
                    cursor..unit.bytes.start,
                    self.config,
                    state,
                    inside_syntax,
                )?);
            }
            match unit.work {
                Work::Preserve => {
                    output.push_str(&self.source.text[unit.bytes.clone()]);
                    state.consume_preserved(self.source, unit.bytes.clone())?;
                }
                Work::Syntax(node) => {
                    output.push_str(&self.render_root(node, &unit.bytes, state)?)
                }
            }
            cursor = unit.bytes.end;
        }
        if cursor < bytes.end {
            output.push_str(&super::super::legacy::render(
                self.source,
                &self.ownership,
                cursor..bytes.end,
                self.config,
                state,
                inside_syntax,
            )?);
        }
        Ok(output)
    }

    fn render_root(
        &self,
        node: SyntaxNode<'_>,
        bytes: &Range<usize>,
        state: &mut RenderState,
    ) -> Result<String, String> {
        let mut output = String::new();
        if state.column == 0 {
            let indent = " ".repeat(state.indent_level * self.config.indent);
            state.emitted(&indent);
            output.push_str(&indent);
        }
        let shape = Shape {
            column: state.column,
            indent_level: state.indent_level,
            indent_spaces: self.config.indent,
            width: self.config.max_width,
        };
        match self.registry.format(node, self.source, self.config, shape) {
            Ok(layout) => {
                let regions = |region: Range<usize>, state: &mut RenderState| {
                    self.render_range(region, Some(bytes), state, true)
                };
                let rendered = document::render_with_regions(
                    self.source,
                    &self.ownership,
                    &layout,
                    bytes.clone(),
                    shape,
                    state.clone(),
                    &regions,
                )?;
                output.push_str(&rendered.text);
                *state = rendered.exit_state;
            }
            Err(warning) => {
                // A failed owner preserves its source rather than guessing a layout.
                self.warnings.borrow_mut().push(warning);
                output.push_str(&self.source.text[bytes.clone()]);
                state.consume_preserved(self.source, bytes.clone())?;
            }
        }
        Ok(output)
    }
}
