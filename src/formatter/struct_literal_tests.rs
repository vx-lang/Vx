//! Tests of the `struct_literals` rule. Each input sits in a function body, where struct literals
//! are written.

use super::rule_test_helpers::{format_body, in_main};
use super::*;

#[test]
fn a_short_struct_literal_is_joined_on_one_line() {
    let body = "  count(Counter {\n    next: Slot::new(0)\n  });\n  let p = Point { x: 1, y: 2 };";
    assert_eq!(
        format_body(body, &FormatConfig::default()),
        in_main("  count(Counter { next: Slot::new(0) });\n  let p = Point { x: 1, y: 2 };")
    );
}

#[test]
fn a_struct_literal_with_wide_fields_has_one_field_on_each_line() {
    let body = "  return Wide {\n    negative: d.negative, exponent: d.exponent\n  };";
    assert_eq!(
        format_body(body, &FormatConfig::default()),
        in_main("  return Wide {\n    negative: d.negative,\n    exponent: d.exponent\n  };")
    );
}

#[test]
fn struct_lit_width_sets_how_wide_the_fields_may_be() {
    let body = "  return Wide {\n    negative: d.negative, exponent: d.exponent\n  };";
    let config = FormatConfig {
        struct_lit_width: 50,
        ..FormatConfig::default()
    };
    assert_eq!(
        format_body(body, &config),
        in_main("  return Wide { negative: d.negative, exponent: d.exponent };")
    );
}

#[test]
fn a_struct_literal_that_does_not_fit_the_line_has_one_field_on_each_line() {
    let body = "  let point = make_a_point_from(Point { x: 1, y: 2 });";
    let config = FormatConfig {
        max_width: 40,
        fn_signature_width: 40,
        ..FormatConfig::default()
    };
    assert_eq!(
        format_body(body, &config),
        in_main("  let point = make_a_point_from(Point {\n    x: 1,\n    y: 2\n  });")
    );
}

#[test]
fn a_struct_literal_in_a_field_value_gets_its_own_layout() {
    let body = "  let p = P {\n    x: 1, y: Q {\n      a: 2\n    }\n  };";
    assert_eq!(
        format_body(body, &FormatConfig::default()),
        in_main("  let p = P {\n    x: 1,\n    y: Q { a: 2 }\n  };")
    );
}

#[test]
fn a_trailing_comma_keeps_one_field_on_each_line() {
    let body = "  let p = Point { x: 1, y: 2, };";
    assert_eq!(
        format_body(body, &FormatConfig::default()),
        in_main("  let p = Point {\n    x: 1,\n    y: 2,\n  };")
    );
}

#[test]
fn a_field_value_with_a_block_is_indented_under_its_field() {
    let body = "  let h = Holder { run: |v| { v + 1 }, n: 2 };";
    assert_eq!(
        format_body(body, &FormatConfig::default()),
        in_main("  let h = Holder {\n    run: |v| {\n      v + 1\n    },\n    n: 2\n  };")
    );
}

#[test]
fn a_comment_between_fields_leaves_the_literal_to_the_base_formatting() {
    let body = "  let p = Point { x: 1, // left\n    y: 2 };";
    let switched_off = FormatConfig::default().with_rule("struct_literals", false);
    assert_eq!(
        format_body(body, &FormatConfig::default()),
        format_body(body, &switched_off)
    );
}

#[test]
fn a_switched_off_rule_gives_the_base_formatting() {
    let body = "  let p = Point { x: 1, y: 2 };";
    let switched_off = FormatConfig::default().with_rule("struct_literals", false);
    assert_eq!(
        format_body(body, &switched_off),
        in_main("  let p = Point {\n    x: 1, y: 2\n  };")
    );
}

#[test]
fn an_unsafe_block_that_holds_a_struct_literal_keeps_its_layout_in_a_second_run() {
    // The base formatting puts a short `unsafe` block on one line, but it cannot measure the
    // literal, whose rule may need several lines. So the block gets lines of its own.
    let body = "  unsafe { write(slot, Triple { a : k, b : 10 * k, c : 100 * k }); }";
    assert_eq!(
        format_body(body, &FormatConfig::default()),
        in_main(
            "  unsafe {\n    write(slot, Triple {\n      a : k,\n      b : 10 * k,\n      c : 100 * k\n    });\n  }"
        )
    );
}

#[test]
fn the_code_after_a_struct_literal_counts_toward_the_line_width() {
    // On one line, the literal ends at column 25, but the line ends at column 43.
    let body = "  mk(Point { x: 1, y: 2 }, 12345678901234);";
    let config = FormatConfig {
        max_width: 40,
        fn_signature_width: 40,
        ..FormatConfig::default()
    };
    assert_eq!(
        format_body(body, &config),
        in_main("  mk(Point {\n    x: 1,\n    y: 2\n  }, 12345678901234);")
    );
}
