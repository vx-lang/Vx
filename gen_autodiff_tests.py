import os

TEST_DIR = "tests/backend/pass/autodiff"

math_grad_tests = {
    "autodiff_math_sin": ("sin", "x.sin()", "0.0", "1.0"),       # d/dx sin(0) = cos(0) = 1.0
    "autodiff_math_cos": ("cos", "x.cos()", "0.0", "0.0"),       # d/dx cos(0) = -sin(0) = 0.0
    "autodiff_math_exp": ("exp", "x.exp()", "1.0", "2.718281"),  # d/dx exp(1) = exp(1) approx 2.718
    "autodiff_math_sqrt": ("sqrt", "x.sqrt()", "4.0", "0.25"),   # d/dx sqrt(4) = 0.5 / sqrt(4) = 0.25
    "autodiff_math_abs_pos": ("abs_pos", "x.abs()", "2.0", "1.0"),   # d/dx abs(2) = 1.0
    "autodiff_math_abs_neg": ("abs_neg", "x.abs()", "-2.0", "-1.0"), # d/dx abs(-2) = -1.0
}

def generate_file(name, func_name, expr, eval_at, expected):
    content = f"""//===- {name}.vx - Vx Compiler -------------------*- Vx -*-===//
import std::math;

fn test_func(x: f32) -> f32 {{
    return {expr};
}}

fn main() -> i32 {{
    let x: f32 = {eval_at};
    let dx: f32 = grad(test_func, x);
    let expected: f32 = {expected};
    
    let diff: f32 = (dx - expected).abs();
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

for name, (func_name, expr, eval_at, expected) in math_grad_tests.items():
    generate_file(name, func_name, expr, eval_at, expected)
