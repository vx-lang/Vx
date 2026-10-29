use super::dispatch::{
    Coverage, FormatOutcome, NodeContext, NodeFormatter, NodeRegistry, SyntaxKind, SyntaxNode,
};
use super::document::{self, Document, RenderState, Shape};
use super::ownership::{OutputSegment, Owner, OwnershipMap};
use super::*;
use vx_config::FormatConfig;

fn shape(width: usize) -> Shape {
    Shape {
        column: 0,
        indent_level: 0,
        indent_spaces: 2,
        width,
    }
}

#[test]
fn source_tokens_keep_original_unicode_bytes_and_line_endings() {
    let text = "// café\r\nfn été(value: i32) -> i32 { value }\r\n";
    let source = SourceFileView::parse(text);
    let name = source
        .tokens
        .iter()
        .enumerate()
        .find(|(_, token)| matches!(token.token.kind, TokenType::Identifier("été")))
        .unwrap()
        .0;
    assert_eq!(source.token_text(TokenId(name)), "été");
    assert_eq!(source.text, text);
    assert!(source.verify_preservation(text).is_ok());
    assert_eq!(source.functions.len(), 1);
}

#[test]
fn complete_headers_include_every_clause_before_the_real_body() {
    let text = "fn compute<T>(value: T) on Topology::CPU -> i32 requires value > 0 requires value < 100 ensures value > 0 where Reachable<CPU, GPU>, Reachable<GPU, CPU> { value }";
    let source = SourceFileView::parse(text);
    assert!(source.warnings.is_empty(), "{:?}", source.warnings);
    assert_eq!(source.functions.len(), 1);
    let header = &source.functions[0].header;
    let kinds: Vec<_> = header.clauses.iter().map(|clause| clause.kind).collect();
    assert_eq!(
        kinds,
        vec![
            HeaderClauseKind::On,
            HeaderClauseKind::Requires,
            HeaderClauseKind::Requires,
            HeaderClauseKind::Ensures,
            HeaderClauseKind::Where
        ]
    );
    assert_eq!(source.token_text(header.terminator), "{");
    assert_eq!(
        &source.text[source.token_bytes(header.return_type.tokens.clone())],
        "i32"
    );
}

#[test]
fn an_unknown_header_suffix_does_not_produce_a_partial_function_view() {
    let source = SourceFileView::parse("fn add(value: i32) -> i32 future_clause value { value }");
    assert!(source.functions.is_empty());
    assert!(!source.warnings.is_empty());
}

#[test]
fn macro_envelopes_hide_declarations_in_arguments_matchers_and_transcribers() {
    let text = "macro_rules! make { ($x:expr) => { fn fake(value: i32) -> i32 { value } }; }\nmlir!(inputs: ()) { %x = fn fake() }\nfn real(value: i32) -> i32 { value }";
    let source = SourceFileView::parse(text);
    assert_eq!(source.macros.len(), 2);
    assert_eq!(source.macros[0].kind, MacroKind::Definition);
    assert_eq!(source.macros[0].token_trees.len(), 2);
    assert_eq!(source.macros[1].token_trees.len(), 2);
    assert_eq!(source.functions.len(), 1);
    assert_eq!(source.token_text(source.functions[0].header.name), "real");
}

#[test]
fn incomplete_macro_contents_do_not_become_ordinary_declarations() {
    let source = SourceFileView::parse("unknown!({ fn fake(value: i32) -> i32 { value }");
    assert!(source.functions.is_empty());
    assert!(!source.warnings.is_empty());
}

#[test]
fn an_invalid_separated_macro_does_not_expose_its_contents_as_vx() {
    let source = SourceFileView::parse("unknown !{ fn fake(value: i32) -> i32 { value } }");
    assert!(source.functions.is_empty());
    assert!(!source.warnings.is_empty());
}

#[test]
fn formatter_directives_inside_macros_do_not_escape_the_macro() {
    let text =
        "unknown! {\n// vx-format-begin: OFF\nfn fake()\n}\nfn real(value: i32) -> i32 { value }";
    let source = SourceFileView::parse(text);
    assert!(source.exclusions.is_empty());
    assert_eq!(source.functions.len(), 1);
}

#[test]
fn token_preservation_checks_parser_sensitive_adjacency() {
    for (input, output) in [
        ("value >> count", "value > > count"),
        ("value << count", "value < < count"),
        ("value >>= count", "value > >= count"),
        ("name!(value)", "name !(value)"),
        ("value > > count", "value >> count"),
    ] {
        assert!(
            SourceFileView::parse(input)
                .verify_preservation(output)
                .is_err(),
            "accepted {input:?} -> {output:?}"
        );
    }
}

#[test]
fn opaque_macro_preservation_is_stronger_than_vx_token_equality() {
    let input = "mlir!(inputs: ()) { %x = operation }";
    let changed = "mlir!(inputs: ()) { % x = operation }";
    assert!(SourceFileView::parse(input)
        .verify_preservation(changed)
        .is_err());
    assert!(SourceFileView::parse(input)
        .verify_preservation(input)
        .is_ok());
}

#[test]
fn excluded_whitespace_survives_final_preservation_checks() {
    let input = "// vx-format-begin: OFF\nlet   value = 1;  \n// vx-format-end: OFF";
    let output = input.replace("1;  \n", "1;\n");
    assert!(SourceFileView::parse(input)
        .verify_preservation(&output)
        .is_err());
}

#[test]
fn legacy_edits_cannot_cross_new_or_preserved_ownership() {
    let source = SourceFileView::parse("fn add(value: i32) -> i32 { unknown!(value) }");
    let mut ownership = OwnershipMap::from_source(&source);
    ownership
        .claim(source.functions[0].bytes.clone(), Owner::Syntax)
        .unwrap();
    assert!(ownership.validate_legacy_edit(0..2).is_err());
    assert!(ownership
        .validate_legacy_edit(source.macros[0].bytes.clone())
        .is_err());
    assert_eq!(
        ownership.owner_at(source.macros[0].bytes.start),
        Owner::OpaqueMacro
    );
}

#[test]
fn preserved_fragments_keep_enclosing_delimiter_and_indentation_state() {
    let source = SourceFileView::parse("{\n[\nvalue\n]\n}");
    let mut state = RenderState::new(shape(100));
    state.consume_preserved(&source, 0..2).unwrap();
    assert_eq!(state.indent_level, 1);
    state.consume_preserved(&source, 2..4).unwrap();
    assert_eq!(state.indent_level, 2);
    assert_eq!(state.delimiters.len(), 2);
    state
        .consume_preserved(&source, 4..source.text.len())
        .unwrap();
    assert_eq!(state.indent_level, 0);
    assert!(state.delimiters.is_empty());
    assert_eq!(state.column, 1);
}

struct TypeOwner;

impl NodeFormatter for TypeOwner {
    fn kind(&self) -> SyntaxKind {
        SyntaxKind::Type
    }
    fn format(
        &self,
        node: SyntaxNode<'_>,
        _: &NodeContext<'_, '_, '_>,
        _: Shape,
    ) -> Result<Document, CoverageWarning> {
        Ok(Document::Token(TokenId(node.tokens().start)))
    }
}

#[test]
fn syntax_registry_rejects_competing_owners() {
    let first = TypeOwner;
    let second = TypeOwner;
    assert!(NodeRegistry::new(&[&first, &second]).is_err());
}

#[test]
fn hierarchical_formatting_keeps_type_normalization_and_header_width() {
    let text = "fn apply(value: Tensor< f32, [2, 2] >, count: i32) -> i32 { value }";
    let source = SourceFileView::parse(text);
    assert_eq!(source.functions.len(), 1);
    let config = FormatConfig {
        indent: 2,
        fn_signature_width: 40,
    };
    let result = format_declaration(&source, 0, &config, RenderState::new(shape(40)));
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    assert_eq!(
        result.text,
        "fn apply(\n  value: Tensor<f32, [2, 2]>,\n  count: i32\n) -> i32 { value }"
    );
    assert!(source.verify_preservation(&result.text).is_ok());
    let repeated = SourceFileView::parse(&result.text);
    let result_again = format_declaration(&repeated, 0, &config, RenderState::new(shape(40)));
    assert_eq!(result_again.text, result.text);
}

#[test]
fn short_multiline_headers_become_single_line() {
    let source = SourceFileView::parse("fn add(\nvalue: i32\n) -> i32 { value }");
    let result = format_declaration(
        &source,
        0,
        &FormatConfig::default(),
        RenderState::new(shape(100)),
    );
    assert_eq!(result.text, "fn add(value: i32) -> i32 { value }");
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
}

#[test]
fn source_layouts_cannot_move_opaque_macros_to_another_column() {
    let source = SourceFileView::parse("unknown!(value)");
    let layout = Document::Concat(vec![
        Document::Space,
        Document::Preserve {
            bytes: 0..source.text.len(),
            keep_column: false,
        },
    ]);
    let ownership = OwnershipMap::from_source(&source);
    assert!(document::render(
        &source,
        &ownership,
        &layout,
        0..source.text.len(),
        shape(100),
        RenderState::new(shape(100))
    )
    .is_err());
}

#[test]
fn declaration_rendering_preserves_enclosing_impl_state() {
    let source = SourceFileView::parse(
        "impl Counter {\n  fn add(first: i32, second: i32) -> i32 { first }\n}",
    );
    assert_eq!(source.functions.len(), 1);
    let declaration = &source.functions[0];
    let config = FormatConfig {
        indent: 2,
        fn_signature_width: 39,
    };
    let mut entry = RenderState::new(shape(39));
    entry
        .consume_preserved(&source, 0..declaration.bytes.start)
        .unwrap();
    assert_eq!(entry.indent_level, 1);
    let node_shape = Shape {
        column: entry.column,
        indent_level: entry.indent_level,
        ..shape(39)
    };
    let layout = builtin_registry()
        .format(
            SyntaxNode::Function(declaration),
            &source,
            &config,
            node_shape,
        )
        .unwrap();
    let mut ownership = OwnershipMap::from_source(&source);
    ownership
        .claim(declaration.bytes.clone(), Owner::Syntax)
        .unwrap();
    let result = document::render(
        &source,
        &ownership,
        &layout,
        declaration.bytes.clone(),
        node_shape,
        entry,
    )
    .unwrap();
    assert_eq!(
        result.text,
        "fn add(\n    first: i32,\n    second: i32\n  ) -> i32 { first }"
    );
    assert_eq!(result.exit_state.indent_level, 1);
    assert_eq!(result.exit_state.delimiters.len(), 1);
}

#[test]
fn warning_only_results_do_not_fail_check_mode() {
    let result = FormatOutcome {
        text: "unchanged".into(),
        coverage: Coverage::Partial,
        warnings: vec![CoverageWarning {
            bytes: 0..9,
            message: "unsupported syntax; preserved the file".into(),
        }],
    };
    assert_eq!(result.check_status("unchanged"), 0);
    assert_eq!(result.check_status("different"), 1);
    assert!(result.warnings[0]
        .render("program.vx", "unchanged")
        .starts_with("WARN: program.vx:1:"));
}

#[test]
fn child_shapes_reserve_parent_width_and_indent_once() {
    let parent = Shape {
        column: 8,
        indent_level: 1,
        indent_spaces: 4,
        width: 40,
    };
    assert_eq!(parent.reserve(5).unwrap().remaining(), 27);
    assert_eq!(parent.child_line().column, 8);
    assert!(parent.reserve(33).is_none());
}

#[test]
fn cleanup_cannot_remove_parser_sensitive_separation() {
    let source = SourceFileView::parse("value > > count");
    let gap = &source.gaps[0];
    let bytes = source.tokens[gap.left.0].bytes.end..source.tokens[gap.right.0].bytes.start;
    let segment = OutputSegment {
        output: bytes.clone(),
        source: Some(bytes),
        owner: Owner::Legacy,
        whitespace_only: true,
        preserved: false,
    };
    assert!(!segment.allows_legacy_cleanup(&OwnershipMap::from_source(&source), &source));
}
