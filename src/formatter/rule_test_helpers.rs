//! Helpers for the tests of the rules that format code inside function bodies.

use super::*;

/// Format `body` inside `fn main`, with `config`, and check that a second run changes nothing.
pub(super) fn format_body(body: &str, config: &FormatConfig) -> String {
    let result = format_file_with_report(&in_main(body), config);
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    let again = format_file_with_report(&result.text, config);
    assert_eq!(again.text, result.text, "a second run changed the output");
    result.text
}

/// `body` inside `fn main`.
pub(super) fn in_main(body: &str) -> String {
    format!("fn main() -> i32 {{\n{body}\n}}\n")
}
