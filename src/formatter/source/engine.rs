use super::dispatch::{FormatOutcome, NodeRegistry, SyntaxKind, SyntaxNode};
use super::document::{self, RenderState, Shape};
use super::ownership::{Owner, OwnershipMap};
use super::{builtin_registry, overlaps, CoverageWarning, MacroKind, SourceFileView};
use crate::lexer::TokenType;
use std::cell::RefCell;
use std::ops::Range;
use vx_config::FormatConfig;

#[derive(Clone, Copy)]
enum Work<'a> {
    Syntax(SyntaxNode<'a>),
    /// A macro call: its bytes stay as written, but its line takes the indentation.
    MacroCall,
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
    let unchanged = |warnings: Vec<CoverageWarning>| FormatOutcome {
        text: text.to_string(),
        warnings,
    };
    if source.exclusions.contains(&(0..text.len())) {
        return unchanged(Vec::new());
    }
    let mut warnings = source.warnings.clone();
    warnings.extend(registry.enabled_warnings(&source, config));
    warnings.sort_by_key(|warning| (warning.bytes.start, warning.bytes.end));
    warnings.dedup();
    if warnings
        .iter()
        .any(|warning| warning.bytes == (0..text.len()))
    {
        return unchanged(warnings);
    }
    // An error here is a bug in the formatter, not a problem in the program. Stop, so that
    // `--check` cannot pass on a file that the formatter did not check.
    let output = render_file(&source, config, registry, &mut warnings)
        .unwrap_or_else(|message| panic!("internal formatter error: {message}"));
    FormatOutcome {
        text: output,
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
    // Keep whole lines as written around excluded code, macro definitions and code with a
    // coverage warning. A macro call keeps only its own bytes, so its line can still move.
    let mut raw: Vec<_> = source.exclusions.clone();
    raw.extend(
        source
            .macros
            .iter()
            .filter(|view| view.kind == MacroKind::Definition)
            .map(|view| line_envelope(source, &view.bytes)),
    );
    raw.extend(
        warnings
            .iter()
            .map(|warning| line_envelope(source, &warning.bytes)),
    );
    // A protected range that cuts through a macro call protects the whole call.
    let calls: Vec<Range<usize>> = source
        .macros
        .iter()
        .filter(|view| view.kind == MacroKind::Invocation)
        .map(|view| view.bytes.clone())
        .collect();
    let cut_calls: Vec<_> = calls
        .iter()
        .filter(|call| raw.iter().any(|range| overlaps(range, call)))
        .cloned()
        .collect();
    raw.extend(cut_calls);
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
    let inside_protected = |bytes: &Range<usize>| {
        protected
            .iter()
            .any(|range| range.start <= bytes.start && bytes.end <= range.end)
    };
    let calls: Vec<_> = calls
        .into_iter()
        .filter(|call| !inside_protected(call))
        .collect();
    let mut ownership = OwnershipMap::from_source(source);
    // Additional protection is installed in a fresh map to avoid overlapping opaque envelopes.
    ownership.protect_union(&protected)?;
    // A root inside a macro call is not Vx code that the formatter may move.
    let mut roots: Vec<_> = registry
        .enabled_roots(source, config)
        .into_iter()
        .map(|node| (source.token_bytes(node.tokens()), node))
        .filter(|(bytes, _)| !bytes.is_empty())
        .filter(|(bytes, _)| {
            !calls.iter().any(|call| {
                overlaps(call, bytes) && !(bytes.start <= call.start && call.end <= bytes.end)
            })
        })
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
        calls,
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
    /// Whole lines kept exactly as written: exclusions, macro definitions, and code with a
    /// coverage warning.
    protected: Vec<Range<usize>>,
    /// The macro calls outside `protected`, sorted by start. Each keeps its bytes as written.
    calls: Vec<Range<usize>>,
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
        // A macro call inside a root belongs to that root, which formats it in a region.
        let calls: Vec<_> = self
            .calls
            .iter()
            .filter(|call| inside(call))
            .filter(|call| {
                !units
                    .iter()
                    .any(|unit| matches!(unit.work, Work::Syntax(_)) && overlaps(&unit.bytes, call))
            })
            .map(|call| Unit {
                bytes: call.clone(),
                work: Work::MacroCall,
            })
            .collect();
        units.extend(calls);
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
        // The base formatting formats the code between protected ranges in one piece, with
        // each syntax root as one placeholder, so that it sees whole blocks around the roots.
        let mut roots: Vec<(Range<usize>, Work<'a>)> = Vec::new();
        let units = self.units(bytes.clone(), skip);
        for unit in units.iter().map(Some).chain(std::iter::once(None)) {
            if let Some(Unit {
                bytes: root,
                work: work @ (Work::Syntax(_) | Work::MacroCall),
            }) = unit
            {
                roots.push((root.clone(), *work));
                continue;
            }
            let end = unit.map_or(bytes.end, |unit| unit.bytes.start);
            if end < cursor {
                return Err("Overlapping formatter ownership".into());
            }
            if cursor < end {
                let ranges: Vec<_> = roots.iter().map(|(root, _)| root.clone()).collect();
                output.push_str(&super::super::legacy::render(
                    self.source,
                    &self.ownership,
                    cursor..end,
                    self.config,
                    state,
                    inside_syntax,
                    &ranges,
                    &mut |index, state| match roots[index].1 {
                        Work::Syntax(node) => {
                            let root = &roots[index].0;
                            let reserved = self.rest_of_line(root, &bytes, inside_syntax);
                            self.render_root(node, root, reserved, state)
                        }
                        _ => self.render_macro_call(&roots[index].0, state),
                    },
                    &|index| match roots[index].1 {
                        Work::Syntax(node) => self.one_line_width(node, &roots[index].0),
                        _ => None,
                    },
                )?);
            }
            roots.clear();
            if let Some(unit) = unit {
                output.push_str(&self.source.text[unit.bytes.clone()]);
                state.consume_preserved(self.source, unit.bytes.clone())?;
                cursor = unit.bytes.end;
            }
        }
        Ok(output)
    }

    fn render_macro_call(
        &self,
        bytes: &Range<usize>,
        state: &mut RenderState,
    ) -> Result<String, String> {
        let mut output = String::new();
        if state.column == 0 {
            let indent = " ".repeat(state.indent_level * self.config.indent);
            state.emitted(&indent);
            output.push_str(&indent);
        }
        output.push_str(&self.source.text[bytes.clone()]);
        state.consume_preserved(self.source, bytes.clone())?;
        Ok(output)
    }

    /// The width of the code after `bytes` that stays on its line: the source up to a line
    /// break, a comment, or the end of `range`, the range that the root is formatted in. Inside
    /// a region of another rule, that rule puts at most a comma after the region on the line.
    fn rest_of_line(
        &self,
        bytes: &Range<usize>,
        range: &Range<usize>,
        inside_syntax: bool,
    ) -> usize {
        let mut width = 0;
        let mut pending_space = 0;
        for token in &self.source.tokens[self.source.token_range(&(bytes.end..range.end))] {
            if token.bytes.end > range.end {
                break;
            }
            match token.token.kind {
                TokenType::Whitespace(text) if text.contains('\n') => return width,
                TokenType::Whitespace(text) => pending_space += text.chars().count(),
                TokenType::Comment(_) | TokenType::DocComment(_) | TokenType::Eof => return width,
                _ => {
                    width += pending_space + self.source.text[token.bytes.clone()].chars().count();
                    pending_space = 0;
                }
            }
        }
        let comma_follows = self.source.tokens[self
            .source
            .token_range(&(range.end..self.source.text.len()))]
        .iter()
        .find(|token| !matches!(token.token.kind, TokenType::Whitespace(_)))
        .is_some_and(|token| matches!(token.token.kind, TokenType::Comma));
        width + usize::from(inside_syntax && comma_follows)
    }

    /// The width of a root when its rule puts it on one line, laid out on its own. `None` when
    /// the rule needs several lines. Nothing that this layout does stays: not its warnings either.
    fn one_line_width(&self, node: SyntaxNode<'_>, bytes: &Range<usize>) -> Option<usize> {
        let warnings = self.warnings.borrow().len();
        let mut state = RenderState::new(Shape {
            column: 1,
            indent_level: 0,
            indent_spaces: self.config.indent,
            width: self.config.max_width,
        });
        let text = self.render_root(node, bytes, 0, &mut state);
        self.warnings.borrow_mut().truncate(warnings);
        text.ok()
            .filter(|text| !text.contains('\n'))
            .map(|text| text.chars().count())
    }

    fn render_root(
        &self,
        node: SyntaxNode<'_>,
        bytes: &Range<usize>,
        rest_of_line: usize,
        state: &mut RenderState,
    ) -> Result<String, String> {
        let mut output = String::new();
        if state.column == 0 {
            let indent = " ".repeat(state.indent_level * self.config.indent);
            state.emitted(&indent);
            output.push_str(&indent);
        }
        // A function header already reserves the `{` or `;` after it.
        let reserved = match node.kind() {
            SyntaxKind::StructLiteral | SyntaxKind::IfValue => rest_of_line,
            _ => 0,
        };
        let shape = Shape {
            column: state.column,
            indent_level: state.indent_level,
            indent_spaces: self.config.indent,
            width: self.config.max_width.saturating_sub(reserved),
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
