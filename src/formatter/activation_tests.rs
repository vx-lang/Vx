use super::*;
use source::dispatch::{NodeContext, NodeFormatter, NodeRegistry, SyntaxKind, SyntaxNode};
use source::document::{self, Document, RenderState, Shape};
use source::ownership::OwnershipMap;
use source::{CoverageWarning, SourceFileView, TokenId};
use std::sync::atomic::{AtomicUsize, Ordering};

fn checked(input: &str, width: usize) -> FormatOutcome {
    format_file_with_report(
        input,
        &FormatConfig {
            indent: 2,
            fn_signature_width: width,
        },
    )
}

#[test]
fn active_pipeline_formats_headers_and_bodies_in_one_stable_pass() {
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
fn nested_headers_keep_the_parent_renderer_state() {
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
fn unrecognized_clauses_in_foreign_signatures_preserve_that_declaration() {
    let input = "extern {\nfn future(value: i32) -> i32 requires value > 0;\nfn known(value: i32) -> i32;\n}";
    let result = checked(input, 20);
    assert!(!result.warnings.is_empty());
    assert!(result
        .text
        .contains("fn future(value: i32) -> i32 requires value > 0;"));
    assert!(
        result
            .text
            .contains("fn known(\n    value: i32\n  ) -> i32;"),
        "{}",
        result.text
    );
}

#[test]
fn all_header_clauses_survive_the_active_path() {
    let input = "fn compute(value: i32) on Topology::CPU -> i32 requires value > 0 ensures value > 0 where Reachable<CPU, GPU> {\nvalue\n}";
    let result = checked(input, 40);
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    assert!(result.text.contains("fn compute(\n  value: i32\n) on Topology::CPU -> i32 requires value> 0 ensures value> 0 where Reachable<CPU, GPU>"), "{}", result.text);
    assert!(crate::parse_module(&result.text).is_ok());
}

#[test]
fn opaque_macros_keep_bytes_columns_and_embedded_declarations() {
    let input = "macro_rules! make { ($x:expr) => { fn   fake(x:i32)->i32 {$x} }; }\nfn real(value: i32) -> i32 {\nlet result = mlir!(inputs: ()) {\n%x = operation\n};\nreturn value;\n}";
    let source = SourceFileView::parse(input);
    let result = checked(input, 100);
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    assert!(result
        .text
        .starts_with("macro_rules! make { ($x:expr) => { fn   fake(x:i32)->i32 {$x} }; }\n"));
    assert!(result
        .text
        .contains("let result = mlir!(inputs: ()) {\n%x = operation\n};\n"));
    assert!(result.text.contains("  return value;"), "{}", result.text);
    assert!(source.verify_preservation(&result.text).is_ok());
    assert_eq!(checked(&result.text, 100).text, result.text);
}

#[test]
fn excluded_delimiters_update_the_enclosing_state_without_rewriting_bytes() {
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
fn unknown_root_syntax_warns_and_preserves_the_file() {
    let input = "future_binding value=1;\n";
    let result = checked(input, 100);
    assert_eq!(result.text, input);
    assert!(!result.warnings.is_empty());
    assert_eq!(result.check_status(input), 0);
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
    assert_eq!(result.check_status(input), 1);
}

#[test]
fn generic_closers_can_compact_without_separating_contract_shifts() {
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

struct RegisteredType<'a>(&'a AtomicUsize);

impl NodeFormatter for RegisteredType<'_> {
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
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(Document::Token(TokenId(node.tokens().start)))
    }
}

#[test]
fn registering_a_syntax_owner_is_enough_to_enter_the_real_pipeline() {
    let calls = AtomicUsize::new(0);
    let owner = RegisteredType(&calls);
    let owners: &[&dyn NodeFormatter] = &[&owner];
    let registry = NodeRegistry::new(owners).unwrap();
    let result = format_file_with_registry(
        "fn value(value: i32) -> i32 { value }",
        &FormatConfig::default(),
        &registry,
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    assert_eq!(result.text, "fn value(value: i32) -> i32 {\n  value\n}");
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
fn macro_name_adjacency_is_checked_without_opaque_tree_fallback() {
    assert!(SourceFileView::parse("name!")
        .verify_preservation("name !")
        .is_err());
}
