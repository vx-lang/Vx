import os
import sys
import argparse

try:
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
except ImportError:
    print("Warning: coremltools not installed. Skipping CoreML model generation.")
    sys.exit(0)

def build_matmul(output_dir, dim):
    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(dim, dim)), # w: (d, n)
            mb.TensorSpec(shape=(dim, dim))  # x: (n, d)
        ]
    )
    def matmul_prog(w, x):
        res = mb.matmul(x=w, y=x, transpose_x=False, transpose_y=False)
        return res

    mlmodel = ct.convert(
        matmul_prog,
        source="milinternal",
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.ALL
    )
    out_path = os.path.join(output_dir, f"matmul_{dim}x{dim}.mlpackage")
    mlmodel.save(out_path)
    print(f"Saved matmul to {out_path}")

def build_affine(output_dir, dim):
    # c[i] = a[i] * alpha + beta
    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(dim,)), # a
            mb.TensorSpec(shape=(1,)), # alpha
            mb.TensorSpec(shape=(1,))  # beta
        ]
    )
    def affine_prog(a, alpha, beta):
        mul_res = mb.mul(x=a, y=alpha)
        res = mb.add(x=mul_res, y=beta)
        return res

    mlmodel = ct.convert(
        affine_prog,
        source="milinternal",
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.ALL
    )
    out_path = os.path.join(output_dir, f"affine_{dim}.mlpackage")
    mlmodel.save(out_path)
    print(f"Saved affine to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=".", help="Output directory")
    parser.add_argument("--dim", type=int, default=4, help="Tensor dimension")
    args = parser.parse_args()

    print(f"Generating ANE primitive models (dim={args.dim}) in {args.out_dir}...")
    build_matmul(args.out_dir, args.dim)
    build_affine(args.out_dir, args.dim)
    print("Done!")
