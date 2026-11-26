//! Tests of the `if_values` rule.

use super::rule_test_helpers::{format_body, in_main};
use super::*;

#[test]
fn a_short_if_value_is_joined_on_one_line() {
    let body = "  let a = if c > 0 {\n    1\n  } else {\n    0\n  };";
    assert_eq!(
        format_body(body, &FormatConfig::default()),
        in_main("  let a = if c > 0 { 1 } else { 0 };")
    );
}

#[test]
fn an_if_statement_keeps_its_lines() {
    let body = "  if c > 0 { a = 2; } else { a = 3; }";
    assert_eq!(
        format_body(body, &FormatConfig::default()),
        in_main("  if c > 0 {\n    a = 2;\n  } else {\n    a = 3;\n  }")
    );
}

#[test]
fn an_if_value_with_a_statement_keeps_its_lines() {
    let body = "  let b = if c > 0 {\n    let d = 1;\n    d\n  } else { 0 };";
    assert_eq!(
        format_body(body, &FormatConfig::default()),
        in_main("  let b = if c > 0 {\n    let d = 1;\n    d\n  } else {\n    0\n  };")
    );
}

#[test]
fn an_if_value_wider_than_its_limit_keeps_its_lines() {
    let body = "  let w = if a_long_condition > another_one { first_value } else { second };";
    let expected = in_main(
        "  let w = if a_long_condition > another_one {\n    first_value\n  } else {\n    second\n  };",
    );
    assert_eq!(format_body(body, &FormatConfig::default()), expected);
    let wide = FormatConfig {
        single_line_if_else_max_width: 80,
        ..FormatConfig::default()
    };
    assert_eq!(
        format_body(body, &wide),
        in_main("  let w = if a_long_condition > another_one { first_value } else { second };")
    );
}

#[test]
fn an_if_value_in_a_call_and_in_a_struct_field_is_joined() {
    let body = "  let p = Point { x: if c {\n    1\n  } else {\n    2\n  } };\n  return pick(if c { 1 } else { 2 });";
    assert_eq!(
        format_body(body, &FormatConfig::default()),
        in_main(
            "  let p = Point {\n    x: if c { 1 } else { 2 }\n  };\n  return pick(if c { 1 } else { 2 });"
        )
    );
}

#[test]
fn an_if_value_with_a_comment_keeps_its_lines() {
    let body = "  let a = if c { 1 } else { 0 // never\n  };";
    let switched_off = FormatConfig::default().with_rule("if_values", false);
    assert_eq!(
        format_body(body, &FormatConfig::default()),
        format_body(body, &switched_off)
    );
}

#[test]
fn a_switched_off_rule_gives_the_base_formatting() {
    let body = "  let a = if c { 1 } else { 0 };";
    let switched_off = FormatConfig::default().with_rule("if_values", false);
    assert_eq!(
        format_body(body, &switched_off),
        in_main("  let a = if c {\n    1\n  } else {\n    0\n  };")
    );
}

#[test]
fn an_if_statement_with_one_call_in_each_branch_keeps_its_lines() {
    // Each branch holds one value, but the `if` is a statement, not a value.
    let body = "  if c { print(1) } else { print(2) }\n  return 0;";
    assert_eq!(
        format_body(body, &FormatConfig::default()),
        in_main("  if c {\n    print(1)\n  } else {\n    print(2)\n  }\n  return 0;")
    );
}

#[test]
fn an_if_value_with_a_statement_on_one_line_gets_the_base_formatting() {
    let body = "  let b = if c { let d = 1; d } else { 0 };";
    let switched_off = FormatConfig::default().with_rule("if_values", false);
    assert_eq!(
        format_body(body, &FormatConfig::default()),
        format_body(body, &switched_off)
    );
}

#[test]
fn the_code_after_an_if_value_counts_toward_the_line_width() {
    // On one line, the `if` ends at column 33, but the line ends at column 52.
    let body = "  let v = f(if c { 1 } else { 2 }, 123456789012345);";
    let config = FormatConfig {
        max_width: 40,
        fn_signature_width: 40,
        ..FormatConfig::default()
    };
    assert_eq!(
        format_body(body, &config),
        in_main("  let v = f(if c {\n    1\n  } else {\n    2\n  }, 123456789012345);")
    );
}
