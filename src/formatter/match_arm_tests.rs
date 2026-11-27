//! Tests of the `match_arms` rule.

use super::rule_test_helpers::{format_body, in_main};
use super::*;

#[test]
fn each_arm_starts_a_line() {
    let body = "  match dot { Shape::Dot(d) => { n = d.x; }, Shape::Line(u, _) => { n = u.y; } }";
    assert_eq!(
        format_body(body, &FormatConfig::default()),
        in_main(
            "  match dot {\n    Shape::Dot(d) => {\n      n = d.x;\n    },\n    Shape::Line(u, _) => {\n      n = u.y;\n    }\n  }"
        )
    );
}

#[test]
fn arms_without_commas_also_start_a_line() {
    let body = "  match c { true => { n = 3; } false => { n = 4; } }";
    assert_eq!(
        format_body(body, &FormatConfig::default()),
        in_main(
            "  match c {\n    true => {\n      n = 3;\n    }\n    false => {\n      n = 4;\n    }\n  }"
        )
    );
}

#[test]
fn comments_and_one_blank_line_between_arms_stay() {
    let body = "  let v = match c {\n    // the first case\n    A => { 1 }, // one\n\n\n    B => {\n      2\n    }\n    // the end\n  };";
    assert_eq!(
        format_body(body, &FormatConfig::default()),
        in_main(
            "  let v = match c {\n    // the first case\n    A => {\n      1\n    }, // one\n\n    B => {\n      2\n    }\n    // the end\n  };"
        )
    );
}

#[test]
fn a_match_inside_an_arm_gets_its_own_layout() {
    let body =
        "  match a { X => { match b { Y => { n = 1; }, Z => { n = 2; } } }, W => { n = 3; } }";
    assert_eq!(
        format_body(body, &FormatConfig::default()),
        in_main(
            "  match a {\n    X => {\n      match b {\n        Y => {\n          n = 1;\n        },\n        Z => {\n          n = 2;\n        }\n      }\n    },\n    W => {\n      n = 3;\n    }\n  }"
        )
    );
}

#[test]
fn a_comment_before_the_brace_leaves_the_match_to_the_base_formatting() {
    let body = "  match c // the flag\n  { A => { n = 1; }, B => { n = 2; } }";
    let switched_off = FormatConfig::default().with_rule("match_arms", false);
    assert_eq!(
        format_body(body, &FormatConfig::default()),
        format_body(body, &switched_off)
    );
}

#[test]
fn a_switched_off_rule_gives_the_base_formatting() {
    let body = "  match c { A => { n = 1; }, B => { n = 2; } }";
    let switched_off = FormatConfig::default().with_rule("match_arms", false);
    assert_eq!(
        format_body(body, &switched_off),
        in_main("  match c {\n    A => {\n      n = 1;\n    }, B => {\n      n = 2;\n    }\n  }")
    );
}

#[test]
fn no_blank_line_follows_the_opening_brace() {
    let body = "  match c {\n\n    A => { n = 1; }\n  }";
    assert_eq!(
        format_body(body, &FormatConfig::default()),
        in_main("  match c {\n    A => {\n      n = 1;\n    }\n  }")
    );
}

#[test]
fn a_comma_on_the_line_after_an_arm_that_ends_with_a_semicolon_stays_there() {
    let body = "  match x {\n    1 => out = 10;\n    ,\n    _ => out = 20;\n    ,\n  }";
    assert_eq!(format_body(body, &FormatConfig::default()), in_main(body));
}
