use super::*;
use source::dispatch::{NodeContext, NodeFormatter, NodeRegistry, SyntaxKind, SyntaxNode};
use source::document::{self, Document, RenderState, Shape};
use source::ownership::OwnershipMap;
use source::{CoverageWarning, SourceFileView, TokenId};

fn checked(input: &str, width: usize) -> FormatOutcome {
    format_file_with_report(
        input,
        &FormatConfig {
            indent: 2,
            fn_signature_width: width,
            ..FormatConfig::default()
        },
    )
}

#[test]
fn headers_and_bodies_are_formatted_in_one_stable_pass() {
    let input = "fn add(\nfirst: i32,\nsecond: i32\n) -> i32 {\nreturn first + second;\n}";
    let result = checked(input, 100);
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    assert_eq!(
        result.text,
        "fn add(first: i32, second: i32) -> i32 {\n  return first + second;\n}"
    );
    let repeated = checked(&result.text, 100);
    assert!(repeated.warnings.is_empty(), "{:?}", repeated.warnings);
    assert_eq!(repeated.text, result.text);
}

#[test]
fn a_header_inside_an_impl_is_indented_with_the_impl() {
    let input = "struct Counter { value: i32 }\nimpl Counter {\nfn add(first: i32, second: i32) -> i32 {\nfirst + second\n}\n}";
    let result = checked(input, 39);
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    assert!(
        result.text.contains(
            "  fn add(\n    first: i32,\n    second: i32\n  ) -> i32 {\n    first + second\n  }"
        ),
        "{}",
        result.text
    );
    assert_eq!(checked(&result.text, 39).text, result.text);
}

#[test]
fn impl_generics_remain_available_to_tensor_parameter_types() {
    let input = "impl<T> Holder<T> {\nfn read(value: Tensor<T, [2, 2]>) -> T {\nvalue[0][0]\n}\n}";
    let result = checked(input, 100);
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    assert!(
        result
            .text
            .contains("  fn read(value: Tensor<T, [2, 2]>) -> T {\n    value[0][0]"),
        "{}",
        result.text
    );
}

#[test]
fn a_clause_on_an_extern_function_preserves_that_declaration() {
    let input = "extern {\nfn future(value: i32) -> i32 ensures value > 0;\nfn known(value: i32) -> i32;\n}";
    let result = checked(input, 20);
    assert!(!result.warnings.is_empty());
    assert!(result
        .text
        .contains("fn future(value: i32) -> i32 ensures value > 0;"));
    assert!(
        result
            .text
            .contains("fn known(\n    value: i32\n  ) -> i32;"),
        "{}",
        result.text
    );
}

#[test]
fn a_clause_on_an_extern_function_is_formatted_normally_when_headers_are_off() {
    let input = "extern {\nfn future(value: i32) -> i32 ensures value > 0;\n}";
    let config = FormatConfig::default().with_rule("function_headers", false);
    let result = format_file_with_report(input, &config);
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    assert_eq!(
        result.text,
        "extern {\n  fn future(value: i32) -> i32 ensures value > 0;\n}"
    );
}

#[test]
fn every_kind_of_header_clause_is_kept() {
    let input = "fn compute(value: i32) on Topology::CPU -> i32 requires value > 0 ensures value > 0 where Reachable<CPU, GPU> {\nvalue\n}";
    let result = checked(input, 40);
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    assert!(result.text.contains("fn compute(\n  value: i32\n) on Topology::CPU -> i32 requires value > 0 ensures value > 0 where Reachable<CPU, GPU>"), "{}", result.text);
    assert!(crate::parse_module(&result.text).is_ok());
}

#[test]
fn macros_keep_their_bytes_and_their_lines_take_the_indentation() {
    let input = "macro_rules! make { ($x:expr) => { fn   fake(x:i32)->i32 {$x} }; }\nfn real(value: i32) -> i32 {\nlet result = mlir!(inputs: ()) {\n%x = operation\n};\nprintln!(\"{}\",   value);\nreturn value;\n}";
    let source = SourceFileView::parse(input);
    let result = checked(input, 100);
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    assert_eq!(
        result.text,
        "macro_rules! make { ($x:expr) => { fn   fake(x:i32)->i32 {$x} }; }\nfn real(value: i32) -> i32 {\n  let result = mlir!(inputs: ()) {\n%x = operation\n};\n  println!(\"{}\",   value);\n  return value;\n}"
    );
    assert!(source.verify_preservation(&result.text).is_ok());
    assert_eq!(checked(&result.text, 100).text, result.text);
}

#[test]
fn a_rule_formats_code_that_holds_a_macro_call() {
    let input = "fn main() -> i32 {\nlet p = Point {   x: one!(),   y: 2 };\nreturn 0;\n}";
    let result = checked(input, 100);
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    assert_eq!(
        result.text,
        "fn main() -> i32 {\n  let p = Point { x: one!(), y: 2 };\n  return 0;\n}"
    );
}

#[test]
fn braces_in_code_with_formatting_off_still_count_for_indentation() {
    let input = "fn read() -> i32 {\n// vx-format-begin: OFF\nif true {\n// vx-format-end: OFF\nreturn 1;\n}\nreturn 0;\n}";
    let result = checked(input, 100);
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    assert_eq!(result.text, "fn read() -> i32 {\n// vx-format-begin: OFF\nif true {\n// vx-format-end: OFF\n    return 1;\n  }\n  return 0;\n}");
}

#[test]
fn cleanup_never_trims_comment_or_string_contents() {
    let input = "fn text() -> i32 {\n// keep these spaces   \nlet text = \"first  \nsecond  \";\nreturn 0;\n}";
    let result = checked(input, 100);
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    assert!(result.text.contains("// keep these spaces   \n"));
    assert!(result.text.contains("\"first  \nsecond  \""));
    assert!(SourceFileView::parse(input)
        .verify_preservation(&result.text)
        .is_ok());
}

#[test]
fn unknown_top_level_syntax_warns_and_keeps_the_file() {
    let input = "future_binding value=1;\n";
    let result = checked(input, 100);
    assert_eq!(result.text, input);
    assert!(!result.warnings.is_empty());
    assert_eq!(result.text, input);
}

#[test]
fn unsupported_bodies_do_not_hide_supported_formatting_differences() {
    let input =
        "fn future() -> i32 {\nfuture_binding value;\n}\nfn known(value: i32) -> i32 {\nvalue\n}";
    let result = checked(input, 100);
    assert!(!result.warnings.is_empty());
    assert!(result
        .text
        .starts_with("fn future() -> i32 {\nfuture_binding value;\n}\n"));
    assert!(
        result
            .text
            .ends_with("fn known(value: i32) -> i32 {\n  value\n}"),
        "{}",
        result.text
    );
    assert_ne!(result.text, input);
}

#[test]
fn nested_generic_closers_join_but_a_shift_in_a_contract_stays() {
    let input =
        "fn read(value: Vec<Vec<i32> >, count: i32) -> i32 requires (count >> 1) > 0 {\ncount\n}";
    let result = checked(input, 100);
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    assert!(result.text.contains("Vec<Vec<i32>>"), "{}", result.text);
    assert!(result.text.contains(">>"));
    assert!(crate::parse_module(&result.text).is_ok());
    assert!(SourceFileView::parse(input)
        .verify_preservation(&result.text)
        .is_ok());
}

#[test]
fn comments_between_parameters_keep_their_header_but_allow_body_formatting() {
    let input =
        "fn add(\nfirst: i32, // keep this comment\nsecond: i32\n) -> i32 {\nfirst + second\n}";
    let result = checked(input, 20);
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    assert_eq!(
        result.text,
        "fn add(\nfirst: i32, // keep this comment\nsecond: i32\n) -> i32 {\n  first + second\n}"
    );
}

struct RegisteredType;

impl NodeFormatter for RegisteredType {
    fn kind(&self) -> SyntaxKind {
        SyntaxKind::Type
    }
    fn roots<'a>(&self, source: &'a SourceFileView<'_>) -> Vec<SyntaxNode<'a>> {
        source
            .functions
            .iter()
            .map(|function| SyntaxNode::Type(&function.header.return_type))
            .collect()
    }
    fn format(
        &self,
        node: SyntaxNode<'_>,
        _: &NodeContext<'_, '_, '_>,
        _: Shape,
    ) -> Result<Document, CoverageWarning> {
        Ok(Document::Concat(vec![
            Document::Line { indent_level: 0 },
            Document::Token(TokenId(node.tokens().start)),
        ]))
    }
}

#[test]
fn a_registered_formatter_formats_code_in_a_whole_file() {
    let owner = RegisteredType;
    let owners: &[&dyn NodeFormatter] = &[&owner];
    let registry = NodeRegistry::new(owners).unwrap();
    let result = format_file_with_registry(
        "fn value(value: i32) -> i32 { value }",
        &FormatConfig::default(),
        &registry,
    );
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    assert_eq!(result.text, "fn value(value: i32) -> \ni32 {\n  value\n}");
}

#[test]
fn layouts_reject_dropped_and_reordered_tokens_before_rendering_completes() {
    let source = SourceFileView::parse("value;");
    let ownership = OwnershipMap::from_source(&source);
    let shape = Shape {
        column: 0,
        indent_level: 0,
        indent_spaces: 2,
        width: 100,
    };
    for layout in [
        Document::Token(TokenId(0)),
        Document::Concat(vec![
            Document::Token(TokenId(1)),
            Document::Token(TokenId(0)),
        ]),
    ] {
        let result = document::render(
            &source,
            &ownership,
            &layout,
            0..source.text.len(),
            shape,
            RenderState::new(shape),
        );
        assert_eq!(
            result.err().as_deref(),
            Some("Layout removed, duplicated, or reordered source tokens")
        );
    }
}

#[test]
fn a_space_between_a_macro_name_and_its_bang_is_rejected() {
    assert!(SourceFileView::parse("name!")
        .verify_preservation("name !")
        .is_err());
}

struct BudgetHeader;
struct BudgetType;

impl NodeFormatter for BudgetHeader {
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
        let SyntaxNode::Header(header) = node else {
            unreachable!()
        };
        let all = context.source.token_bytes(header.tokens.clone());
        let ty = context
            .source
            .token_bytes(header.return_type.tokens.clone());
        let prefix = all.start..ty.start;
        let column = shape.column + context.source.text[prefix.clone()].chars().count();
        let child = context.child(
            SyntaxNode::Type(&header.return_type),
            Shape {
                column,
                width: column,
                ..shape
            },
        )?;
        Ok(Document::Concat(vec![
            Document::Preserve { bytes: prefix },
            child,
            Document::Preserve {
                bytes: ty.end..all.end,
            },
        ]))
    }
}

impl NodeFormatter for BudgetType {
    fn kind(&self) -> SyntaxKind {
        SyntaxKind::Type
    }
    fn format(
        &self,
        node: SyntaxNode<'_>,
        context: &NodeContext<'_, '_, '_>,
        shape: Shape,
    ) -> Result<Document, CoverageWarning> {
        assert_eq!(shape.remaining(), 0);
        let flat = Document::Preserve {
            bytes: context.source.token_bytes(node.tokens()),
        };
        Ok(Document::Choice {
            flat: Box::new(flat.clone()),
            expanded: Box::new(Document::Concat(vec![
                Document::Line { indent_level: 0 },
                flat,
            ])),
        })
    }
}

#[test]
fn a_child_layout_gets_the_width_that_its_parent_left() {
    let owners: &[&dyn NodeFormatter] = &[&BudgetHeader, &BudgetType];
    let registry = NodeRegistry::new(owners).unwrap();
    let result = format_file_with_registry("fn f() -> i32 {}", &FormatConfig::default(), &registry);
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    assert_eq!(result.text, "fn f() -> \ni32 {}");
}

#[test]
fn excluded_root_syntax_does_not_disable_independent_declarations() {
    let excluded = "// vx-format-begin: OFF\nfuture_binding value;\n// vx-format-end: OFF\n";
    let input = format!("{excluded}fn   known(value:i32)->i32 {{\nvalue\n}}\n");
    let result = checked(&input, 100);
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    assert_eq!(
        result.text,
        format!("{excluded}fn known(value: i32) -> i32 {{\n  value\n}}\n")
    );
    assert_ne!(result.text, input);
}

#[test]
fn excluded_statements_do_not_disable_the_rest_of_the_body() {
    let excluded = "// vx-format-begin: OFF\nfuture_binding value;\n// vx-format-end: OFF\n";
    let input = format!("fn   known(value:i32)->i32 {{\n{excluded}return value;\n}}\n");
    let result = checked(&input, 100);
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    assert_eq!(
        result.text,
        format!("fn known(value: i32) -> i32 {{\n{excluded}  return value;\n}}\n")
    );
    assert_eq!(checked(&result.text, 100).text, result.text);
}

#[test]
fn a_byte_range_maps_to_the_tokens_inside_it() {
    let source = SourceFileView::parse("fn f() -> i32 {}\nfn g() -> i32 {}\n");
    let bytes = source.functions[1].bytes.clone();
    let range = source.token_range(&bytes);
    assert!(range.start > 0);
    assert_eq!(source.tokens[range.start].bytes.start, bytes.start);
    assert_eq!(source.tokens[range.end - 1].bytes.end, bytes.end);
}

#[test]
fn disabled_function_headers_preserve_layout_types_and_contracts() {
    let header = "fn   long_name(a: Tensor< f32, [2, 2] >, b : i32) -> i32\nrequires b > 0\nensures return > 0 ";
    let input = format!("{header}{{\nreturn b;\n}}\n");
    let config = FormatConfig {
        fn_signature_width: 20,
        space_before_colon: true,
        ..FormatConfig::default()
    }
    .with_rule("function_headers", false);
    let result = format_file_with_report(&input, &config);
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    assert_eq!(result.text, format!("{header}{{\n  return b;\n}}\n"));
    assert_eq!(
        format_file_with_report(&result.text, &config).text,
        result.text
    );
}

#[test]
fn unknown_tokens_in_macro_definitions_do_not_disable_surrounding_formatting() {
    let opaque = "macro_rules! make { (# $x:expr) => { %acc#0 = operation }; }\n";
    let input = format!("{opaque}fn real(value: i32) -> i32 {{\nreturn value;\n}}\n");
    let source = SourceFileView::parse(&input);
    assert!(source
        .tokens
        .iter()
        .any(|token| matches!(token.token.kind, crate::lexer::TokenType::Unknown('#'))));
    let result = checked(&input, 100);
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    assert_eq!(
        result.text,
        format!("{opaque}fn real(value: i32) -> i32 {{\n  return value;\n}}\n")
    );
    assert!(source.verify_preservation(&result.text).is_ok());
}

#[test]
fn unknown_tokens_in_mlir_blocks_keep_their_bytes() {
    let opaque = "let result = mlir!(inputs: ()) {\n%acc#0 = operation\n};\n";
    let input = format!("fn real(value: i32) -> i32 {{\n{opaque}return value;\n}}\n");
    let result = checked(&input, 100);
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    assert_eq!(
        result.text,
        format!("fn real(value: i32) -> i32 {{\n  {opaque}  return value;\n}}\n")
    );
    assert_eq!(checked(&result.text, 100).text, result.text);
}

#[test]
fn unknown_tokens_outside_macros_still_warn_and_preserve_the_file() {
    let input = "fn real(value: i32) -> i32 {\nreturn value;\n}\n#\n";
    let result = checked(input, 100);
    assert!(result
        .warnings
        .iter()
        .any(|warning| warning.message.contains("Unrecognized lexer token")));
    assert_eq!(result.text, input);
    assert_eq!(result.text, input);
}

#[test]
fn inline_contracts_keep_expression_spacing_while_parameters_format() {
    let input =
        "fn   check(n : i32) -> i32 requires n > 0 && n < 100 ensures return > 0 {\nreturn n;\n}\n";
    let result = checked(input, 100);
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    assert_eq!(
        result.text,
        "fn check(n: i32) -> i32 requires n > 0 && n < 100 ensures return > 0 {\n  return n;\n}\n"
    );
    assert!(crate::parse_module(&result.text).is_ok());
    assert_eq!(checked(&result.text, 100).text, result.text);
}

#[test]
fn multiline_contracts_do_not_force_short_parameters_to_wrap() {
    let clauses = "\n  requires n > 0 && n < 100 && n < 1000\n  requires n << 1 > 0\n  ensures return > 0\n  where Reachable<CPU, GPU>\n";
    let input = format!("fn   check(n : i32) -> i32{clauses}{{\nreturn n;\n}}\n");
    let result = checked(&input, 28);
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    assert_eq!(
        result.text,
        format!("fn check(n: i32) -> i32{clauses}{{\n  return n;\n}}\n")
    );
    assert!(crate::parse_module(&result.text).is_ok());
    assert_eq!(checked(&result.text, 28).text, result.text);
}

#[test]
fn wrapped_parameters_keep_contracts_on_their_original_lines() {
    let input = "fn   merge(first : i32, second : i32) -> i32\nrequires first > second\nensures return > 0 {\nreturn first;\n}\n";
    let result = checked(input, 25);
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    assert_eq!(result.text, "fn merge(\n  first: i32,\n  second: i32\n) -> i32\nrequires first > second\nensures return > 0 {\n  return first;\n}\n");
    assert_eq!(checked(&result.text, 25).text, result.text);
}

#[test]
fn inline_contract_width_reserves_the_expression_and_body_delimiter() {
    let input = "fn f(a: i32) -> i32 requires a > 0 {\n  return a;\n}\n";
    let width = input.lines().next().unwrap().chars().count();
    let fits = checked(input, width);
    assert!(fits.warnings.is_empty(), "{:?}", fits.warnings);
    assert_eq!(fits.text, input);
    let wraps = checked(input, width - 1);
    assert!(wraps.warnings.is_empty(), "{:?}", wraps.warnings);
    assert_eq!(
        wraps.text,
        "fn f(\n  a: i32\n) -> i32 requires a > 0 {\n  return a;\n}\n"
    );
}

#[test]
fn a_name_defined_twice_does_not_stop_formatting() {
    // A second definition is a compile error, not a syntax error, so the file is still formatted.
    let input = "struct Point { x: i32 }\nstruct Point { x: i32 }\nfn twice(a : i32) -> i32 {\nreturn a;\n}\nfn twice(a : i32) -> i32 {\nreturn a;\n}";
    let result = checked(input, 100);
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    assert_eq!(
        result
            .text
            .matches("fn twice(a: i32) -> i32 {\n  return a;\n}")
            .count(),
        2,
        "{}",
        result.text
    );
}

#[test]
fn a_header_on_any_topology_is_formatted() {
    let input = "fn clamp(v : i32, lo : i32) on Topology::Any -> i32 {\nreturn v;\n}";
    let result = checked(input, 100);
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    assert_eq!(
        result.text,
        "fn clamp(v: i32, lo: i32) on Topology::Any -> i32 {\n  return v;\n}"
    );
}

#[test]
fn pub_belongs_to_the_function_header() {
    let input = "pub\nfn add(a : i32) -> i32 {\nreturn a;\n}";
    let result = checked(input, 100);
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    assert_eq!(result.text, "pub fn add(a: i32) -> i32 {\n  return a;\n}");
}

/// A header layout whose one-line alternative is the base formatting of the header, and whose
/// other alternative keeps the header as written.
struct RegionOrOriginalHeader;

impl NodeFormatter for RegionOrOriginalHeader {
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
        let bytes = context.source.token_bytes(node.tokens());
        Ok(Document::Choice {
            flat: Box::new(Document::Region {
                bytes: bytes.clone(),
                indent_level: shape.indent_level,
            }),
            expanded: Box::new(Document::Preserve { bytes }),
        })
    }
}

#[test]
fn a_one_line_layout_that_breaks_a_line_falls_back_to_the_other_layout() {
    // The base formatting re-indents the second line of the header, so the one-line alternative
    // holds a line break. The renderer must reject it and keep the header as written.
    let owners: &[&dyn NodeFormatter] = &[&RegionOrOriginalHeader];
    let registry = NodeRegistry::new(owners).unwrap();
    let input = "fn f(a: i32,\n        b: i32) -> i32 {\n  a\n}\n";
    let result = format_file_with_registry(input, &FormatConfig::default(), &registry);
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    assert_eq!(result.text, input);
}

#[test]
fn a_block_that_holds_a_macro_call_on_several_lines_keeps_its_lines() {
    let input = "fn sum() -> i32 {\n  return unsafe { mlir!(returns: i32) {\n    %x = operation\n  } };\n}\n";
    let result = checked(input, 100);
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    assert_eq!(result.text, input);
}
