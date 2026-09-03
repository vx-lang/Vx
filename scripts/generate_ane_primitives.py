import os
import sys
import argparse

try:
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
except ImportError:
    print("Warning: coremltools not installed. Skipping CoreML model generation.")
    sys.exit(0)

def build_matmul(output_dir, dim, precision="fp32"):
    """One square matmul primitive.

    `precision` decides which hardware CoreML will actually use, and it is not a
    tuning knob. Asked via MLComputePlan which device it prefers, CoreML answers
    CPU for every fp32 matmul at every size tried, up to 512x512. In fp16 it
    answers CPU up to 384 and Neural Engine at 512. So an fp32 primitive never
    reaches the ANE however large it is, and a small fp16 one does not either.
    """
    fp16 = precision == "fp16"

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
        compute_precision=ct.precision.FLOAT16 if fp16 else ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.ALL
    )
    suffix = "_fp16" if fp16 else ""
    out_path = os.path.join(output_dir, f"matmul_{dim}x{dim}{suffix}.mlpackage")
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

def build_attention(output_dir, sq, sk, hd):
    """Fused attention, as one CoreML operation.

    `scaled_dot_product_attention` rather than a matmul/softmax/matmul chain on
    purpose: fused attention exists so that the score matrix is never
    materialised, and decomposing it into GEMM calls defeats that (Vx#319).

    fp16 and roughly 2M elements per operand are both required. Asked through
    MLComputePlan which device it prefers, CoreML answers CPU for every fp32
    attention tried, and for fp16 below about 2M elements; at 2M and above it
    answers Neural Engine. Total size is what matters rather than head count --
    a single-head (1, 16384, 128) is preferred there just as a 32-head one is.
    """
    q_shape = (1, sq, hd)
    kv_shape = (1, sk, hd)

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=q_shape),
            mb.TensorSpec(shape=kv_shape),
            mb.TensorSpec(shape=kv_shape),
        ],
        opset_version=ct.target.iOS18,
    )
    def attn_prog(q, k, v):
        return mb.scaled_dot_product_attention(query=q, key=k, value=v)

    mlmodel = ct.convert(
        attn_prog,
        source="milinternal",
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
    )
    out_path = os.path.join(
        output_dir, f"attention_{sq}x{sk}x{hd}_fp16.mlpackage"
    )
    mlmodel.save(out_path)
    print(f"Saved attention to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=".", help="Output directory")
    parser.add_argument("--dim", type=int, default=4, help="Tensor dimension")
    parser.add_argument(
        "--ane-dims",
        default="512",
        help="Comma-separated square sizes to build fp16 matmul primitives for. "
        "512 is the smallest size CoreML prefers on the Neural Engine for this "
        "operation; at 384 and below it picks the CPU, so a smaller primitive "
        "would add a model that runs on the host.",
    )
    parser.add_argument(
        "--attn-shapes",
        default="256x256x8192",
        help="Comma-separated <sq>x<sk>x<hd> fused-attention primitives to build. "
        "Wide and short is deliberate: the ANE wants ~2M elements per operand, "
        "and the host fallback for attention costs sq*sk*hd, so a long sequence "
        "makes the non-ANE path impractical.",
    )
    args = parser.parse_args()

    print(f"Generating ANE primitive models (dim={args.dim}) in {args.out_dir}...")
    build_matmul(args.out_dir, args.dim)
    build_affine(args.out_dir, args.dim)
    # The ones the Neural Engine will actually take. The 4x4 fp32 pair above is
    # kept because the affine path and the existing tests are written to it.
    for d in [int(x) for x in args.ane_dims.split(",") if x.strip()]:
        build_matmul(args.out_dir, d, precision="fp16")
    for spec in [x for x in args.attn_shapes.split(",") if x.strip()]:
        sq, sk, hd = (int(v) for v in spec.split("x"))
        build_attention(args.out_dir, sq, sk, hd)
    print("Done!")
