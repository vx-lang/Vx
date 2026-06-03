import os

TEST_DIR = "tests/backend/pass"

math_tests_f32 = {
    "math_sin_f32": ("0.0", "x.sin()", "0.0"),
    "math_cos_f32": ("0.0", "x.cos()", "1.0"),
    "math_tan_f32": ("0.0", "x.tan()", "0.0"),
    "math_abs_f32_pos": ("1.5", "x.abs()", "1.5"),
    "math_abs_f32_neg": ("(-1.5)", "x.abs()", "1.5"),
    "math_sqrt_f32": ("4.0", "x.sqrt()", "2.0"),
    "math_exp_f32": ("0.0", "x.exp()", "1.0"),
    "math_log2_f32": ("4.0", "x.log2()", "2.0"),
    "math_log10_f32": ("100.0", "x.log10()", "2.0"),
    "math_asin_f32": ("0.0", "x.asin()", "0.0"),
    "math_acos_f32": ("1.0", "x.acos()", "0.0"),
    "math_atan_f32": ("0.0", "x.atan()", "0.0"),
}

math_tests_f64 = {
    "math_sin_f64": ("0.0", "x.sin()", "0.0"),
    "math_cos_f64": ("0.0", "x.cos()", "1.0"),
    "math_tan_f64": ("0.0", "x.tan()", "0.0"),
    "math_abs_f64_pos": ("1.5", "x.abs()", "1.5"),
    "math_abs_f64_neg": ("(-1.5)", "x.abs()", "1.5"),
    "math_sqrt_f64": ("4.0", "x.sqrt()", "2.0"),
    "math_exp_f64": ("0.0", "x.exp()", "1.0"),
    "math_log2_f64": ("4.0", "x.log2()", "2.0"),
    "math_log10_f64": ("100.0", "x.log10()", "2.0"),
    "math_asin_f64": ("0.0", "x.asin()", "0.0"),
    "math_acos_f64": ("1.0", "x.acos()", "0.0"),
    "math_atan_f64": ("0.0", "x.atan()", "0.0"),
}

def generate_file(name, t_type, init, expr, expected):
    content = f"""//===- {name}.vx - Vx Compiler -------------------*- Vx -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
import std::math;

fn main() -> i32 {{
    let x: {t_type} = {init};
    let res: {t_type} = {expr};
    let expected: {t_type} = {expected};
    
    // allow small floating point difference
    let diff: {t_type} = (res - expected).abs();
    let mut ret: i32 = 1;
    if diff < 0.001 {{
        ret = 0;
    }}
    return ret;
}}
"""
    path = os.path.join(TEST_DIR, f"{name}.vx")
    with open(path, "w") as f:
        f.write(content)
    print(f"Generated {path}")

for name, (init, expr, expected) in math_tests_f32.items():
    generate_file(name, "f32", init, expr, expected)

for name, (init, expr, expected) in math_tests_f64.items():
    generate_file(name, "f64", init, expr, expected)
