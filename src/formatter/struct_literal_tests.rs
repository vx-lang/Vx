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
    // The fields of `P` take more than 40 characters, but the fields of `Q` do not.
    let body =
        "  let p = P {\n    first: 1, y: Q {\n      a: 2\n    }, last: a_long_value_name\n  };";
    assert_eq!(
        format_body(body, &FormatConfig::default()),
        in_main(
            "  let p = P {\n    first: 1,\n    y: Q { a: 2 },\n    last: a_long_value_name\n  };"
        )
    );
}

#[test]
fn a_struct_literal_on_one_line_has_no_comma_before_its_brace() {
    let body = "  let p = Point { x: 1, y: 2, };";
    assert_eq!(
        format_body(body, &FormatConfig::default()),
        in_main("  let p = Point { x: 1, y: 2 };")
    );
}

#[test]
fn a_struct_literal_with_one_field_on_each_line_keeps_its_last_comma() {
    let body = "  return Wide { negative: d.negative, exponent: d.exponent, };";
    assert_eq!(
        format_body(body, &FormatConfig::default()),
        in_main("  return Wide {\n    negative: d.negative,\n    exponent: d.exponent,\n  };")
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
fn a_comment_between_fields_stays_after_its_field() {
    let body = "  let p = Point { x: 1, // left\n    y: 2 };";
    assert_eq!(
        format_body(body, &FormatConfig::default()),
        in_main("  let p = Point {\n    x: 1, // left\n    y: 2\n  };")
    );
}

#[test]
fn a_comment_on_a_line_of_its_own_keeps_its_line() {
    let body = "  let p = Point { // first\n    x: 1,\n    // about y\n    y: 2 // last\n  };";
    assert_eq!(
        format_body(body, &FormatConfig::default()),
        in_main("  let p = Point { // first\n    x: 1,\n    // about y\n    y: 2 // last\n  };")
    );
}

#[test]
fn a_comment_before_a_comma_leaves_the_literal_to_the_base_formatting() {
    let body = "  let p = Point { x: 1 // one\n    , y: 2 };";
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
    let narrow = FormatConfig {
        struct_lit_width: 18,
        ..FormatConfig::default()
    };
    assert_eq!(
        format_body(body, &narrow),
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

#[test]
fn a_struct_literal_that_holds_a_block_keeps_one_field_on_each_line() {
    let body = "  let i = Instant {\n    ns: unsafe { clock_ns() }\n  };\n  let a = Arc<T> {\n    count: count,\n    value: PhantomData<T> {}\n  };";
    assert_eq!(format_body(body, &FormatConfig::default()), in_main(body));
}

#[test]
fn a_short_unsafe_block_that_holds_a_short_struct_literal_stays_on_one_line() {
    let body = "  unsafe { *p = Pair { a : 1, b : 2 }; }";
    assert_eq!(format_body(body, &FormatConfig::default()), in_main(body));
    let written_on_lines = "  unsafe {\n    *p = Pair {\n      a : 1,\n      b : 2\n    };\n  }";
    assert_eq!(
        format_body(written_on_lines, &FormatConfig::default()),
        in_main(body)
    );
}
