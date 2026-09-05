#!/usr/bin/env python3
"""Which graphs does CoreML actually place on the Apple Neural Engine.

A `.vx` test cannot answer this. It sees the numbers a kernel produced, and the
same numbers come back whether the work ran on the Neural Engine, on a CoreML
CPU path, or on the portable host shim. Asking the dispatcher does not answer it
either: `MLComputeUnitsAll` lets CoreML choose, so a log line saying "dispatched"
says nothing about where. This asks CoreML itself, through `MLComputePlan`.

What it pins, and why the negative cases are here
------------------------------------------------

Every case below asserts two things: the placement of *every* operation in the
graph, and the numbers the graph produces. A positive case passes only when
nothing fell back -- one op on the CPU fails it, because a graph split across
two devices is not "running on the ANE" -- and only when the output matches a
reference computed independently in NumPy.

The inputs are pseudo-random and never zero. A zero-filled operand makes a
matmul return zeros, a softmax return a uniform distribution, and a transposed
or misindexed operand indistinguishable from a correct one; three separate
defects in earlier ANE work were invisible until the inputs varied.

The negative cases are what make the positive ones worth reading. A probe that
answered ANE unconditionally would pass every positive case and mean nothing, so
the same assertion runs against graphs that must *not* be ANE-placed: an fp32
matmul, and a chain of ops with no matmul in it.

The finding these encode
------------------------

Device choice is a property of the graph, not of the operation. The identical
f16 softmax on the identical shape is CPU-placed alone and ANE-placed when a
matmul feeds it, and a four-op chain with no matmul stays on the CPU however
many ops it has. The matmul is the anchor; its neighbours come with it. That is
why a per-operation capability table is the wrong thing to build, and why these
cases are whole graphs.

`MLComputePlan` reports the *preferred* compute device -- CoreML's own plan for
the model, which is the strongest evidence the public API offers. It is a plan
rather than an execution trace, and there is no public per-op runtime
attribution to check it against.

Usage:
  scripts/ane_device_check.py [--out-dir DIR] [--list]

Requires macOS, coremltools that can build a model, and `coremlc` from Xcode.
Exits 0 when every case matches its expectation, 1 otherwise, and 77 (the
autotools "skipped" convention) when the toolchain is not present.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from math import erf

import numpy as np

warnings.filterwarnings("ignore")
# coremltools prints a progress bar per conversion pipeline, which buries the results.
os.environ.setdefault("TQDM_DISABLE", "1")
import logging  # noqa: E402

SKIP = 77


def _need(cond, why):
    if not cond:
        print(f"SKIP: {why}")
        sys.exit(SKIP)


try:
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.models.compute_plan import MLComputePlan
    from coremltools.models.compute_device import (
        MLCPUComputeDevice,
        MLGPUComputeDevice,
        MLNeuralEngineComputeDevice,
    )
except Exception as e:  # noqa: BLE001 - any import failure is a skip, not a failure
    _need(False, f"coremltools unavailable ({e})")

logging.getLogger("coremltools").setLevel(logging.ERROR)


def device_name(d):
    if isinstance(d, MLNeuralEngineComputeDevice):
        return "ANE"
    if isinstance(d, MLGPUComputeDevice):
        return "GPU"
    if isinstance(d, MLCPUComputeDevice):
        return "CPU"
    return "UNKNOWN"


def compile_and_plan(model, name, out_dir):
    """Save, compile with coremlc, and return {op_name: device} for the graph."""
    pkg = os.path.join(out_dir, name + ".mlpackage")
    model.save(pkg)
    mlmodelc = os.path.join(out_dir, name + ".mlmodelc")
    if os.path.isdir(mlmodelc):
        shutil.rmtree(mlmodelc)
    r = subprocess.run(
        ["xcrun", "coremlc", "compile", pkg, out_dir],
        capture_output=True,
        text=True,
    )
    if not os.path.isdir(mlmodelc):
        raise RuntimeError(f"coremlc did not produce {name}.mlmodelc: {r.stderr[-400:]}")
    plan = MLComputePlan.load_from_path(mlmodelc, compute_units=ct.ComputeUnit.ALL)
    main = plan.model_structure.program.functions["main"]
    placement = {}
    for op in main.block.operations:
        if op.operator_name == "const":
            continue
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        short = op.operator_name.split(".")[-1]
        dev = device_name(usage.preferred_compute_device) if usage else "UNKNOWN"
        # A repeated op has to agree with itself: record the first disagreement.
        if short in placement and placement[short] != dev:
            placement[short] = f"{placement[short]}/{dev}"
        else:
            placement.setdefault(short, dev)
    return placement


def rnd(*shape, seed=0, lo=-1.0, hi=1.0):
    """Deterministic non-zero inputs. Seeded per operand so a case replays exactly."""
    g = np.random.default_rng(1000 + seed)
    a = g.uniform(lo, hi, size=shape).astype(np.float32)
    # Nothing may be exactly zero: a zero lane hides an indexing error.
    a[np.abs(a) < 1e-3] = 0.5
    return a


def np_softmax(x):
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


_erf = np.vectorize(erf)


def np_gelu(x):
    # CoreML's default gelu is the exact (erf) form, not the tanh approximation.
    return 0.5 * x * (1.0 + _erf(x / np.sqrt(2.0)))


def np_layer_norm(x, eps=1e-5):
    m = x.mean(axis=-1, keepdims=True)
    v = x.var(axis=-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps)


def convert(prog, target, precision):
    return ct.convert(
        prog,
        source="milinternal",
        convert_to="mlprogram",
        minimum_deployment_target=target,
        compute_precision=precision,
        compute_units=ct.ComputeUnit.ALL,
    )


T17 = None
T18 = None


def build_cases():
    """Each case is (name, expected_device, note, thunk).

    The thunk returns (model, inputs, reference): the CoreML model, the named
    non-zero inputs to run it on, and the array those inputs should produce.
    """
    global T17, T18
    T17 = ct.target.iOS17
    T18 = ct.target.iOS18
    F16 = ct.precision.FLOAT16
    F32 = ct.precision.FLOAT32
    cases = []

    def case(name, expect, note, tol=2e-2):
        def deco(fn):
            cases.append((name, expect, note, fn, tol))
            return fn

        return deco

    def spec(*shape):
        return mb.TensorSpec(shape=shape)

    # --- The anchor itself ------------------------------------------------

    @case("matmul_f16_512", "ANE", "the smallest square f16 matmul CoreML prefers on the ANE")
    def _():
        @mb.program(input_specs=[spec(512, 512), spec(512, 512)], opset_version=T17)
        def p(a, b):
            return mb.matmul(x=a, y=b)

        a, b = rnd(512, 512, seed=1), rnd(512, 512, seed=2)
        return convert(p, T17, F16), {"a": a, "b": b}, a @ b

    @case("matmul_f16_1024", "ANE", "and it stays there as the square grows")
    def _():
        @mb.program(input_specs=[spec(1024, 1024), spec(1024, 1024)], opset_version=T17)
        def p(a, b):
            return mb.matmul(x=a, y=b)

        a, b = rnd(1024, 1024, seed=3), rnd(1024, 1024, seed=4)
        return convert(p, T17, F16), {"a": a, "b": b}, a @ b

    @case("transpose_f16", "ANE", "a data movement the ANE takes on its own", tol=1e-3)
    def _():
        @mb.program(input_specs=[spec(1, 512, 512)], opset_version=T17)
        def p(x):
            return mb.transpose(x=x, perm=[0, 2, 1])

        x = rnd(1, 512, 512, seed=5)
        return convert(p, T17, F16), {"x": x}, x.transpose(0, 2, 1)

    # --- Elementwise, which needs size on its own -------------------------

    @case("add_f16_2M", "ANE", "elementwise reaches the ANE alone once it is large enough", tol=1e-3)
    def _():
        @mb.program(input_specs=[spec(1, 2048, 1024), spec(1, 2048, 1024)], opset_version=T17)
        def p(a, b):
            return mb.add(x=a, y=b)

        a, b = rnd(1, 2048, 1024, seed=6), rnd(1, 2048, 1024, seed=7)
        return convert(p, T17, F16), {"a": a, "b": b}, a + b

    @case("mul_f16_2M", "ANE", "the same for multiply", tol=1e-3)
    def _():
        @mb.program(input_specs=[spec(1, 2048, 1024), spec(1, 2048, 1024)], opset_version=T17)
        def p(a, b):
            return mb.mul(x=a, y=b)

        a, b = rnd(1, 2048, 1024, seed=8), rnd(1, 2048, 1024, seed=9)
        return convert(p, T17, F16), {"a": a, "b": b}, a * b

    @case("softmax_f16_16M", "ANE", "softmax alone does reach the ANE, given enough work", tol=5e-3)
    def _():
        @mb.program(input_specs=[spec(1, 4096, 4096)], opset_version=T17)
        def p(x):
            return mb.softmax(x=x, axis=-1)

        x = rnd(1, 4096, 4096, seed=10)
        return convert(p, T17, F16), {"x": x}, np_softmax(x)

    # --- The anchor effect: the point of the whole file --------------------

    @case(
        "softmax_f16_512_after_matmul",
        "ANE",
        "the softmax that is CPU-placed alone, ANE-placed when a matmul feeds it",
    )
    def _():
        @mb.program(input_specs=[spec(1, 512, 512), spec(1, 512, 512)], opset_version=T17)
        def p(a, b):
            s = mb.matmul(x=a, y=b)
            return mb.softmax(x=s, axis=-1)

        a, b = rnd(1, 512, 512, seed=11), rnd(1, 512, 512, seed=12)
        return convert(p, T17, F16), {"a": a, "b": b}, np_softmax(a @ b)

    @case(
        "transformer_block_after_matmul",
        "ANE",
        "layer_norm, gelu and softmax all follow the matmul onto the ANE",
    )
    def _():
        @mb.program(input_specs=[spec(1, 512, 512), spec(1, 512, 512)], opset_version=T17)
        def p(a, b):
            s = mb.matmul(x=a, y=b)
            n = mb.layer_norm(x=s, axes=[-1])
            g = mb.gelu(x=n)
            return mb.softmax(x=g, axis=-1)

        a, b = rnd(1, 512, 512, seed=13), rnd(1, 512, 512, seed=14)
        ref = np_softmax(np_gelu(np_layer_norm(a @ b)))
        return convert(p, T17, F16), {"a": a, "b": b}, ref

    # --- Attention, fused and written out ---------------------------------

    @case("attention_fused", "ANE", "one CoreML operation for the whole of attention")
    def _():
        @mb.program(input_specs=[spec(1, 256, 8192) for _ in range(3)], opset_version=T18)
        def p(q, k, v):
            return mb.scaled_dot_product_attention(query=q, key=k, value=v)

        q, k, v = rnd(1, 256, 8192, seed=15), rnd(1, 256, 8192, seed=16), rnd(1, 256, 8192, seed=17)
        # CoreML's SDPA takes no scale argument and always divides by sqrt(head_dim).
        scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(8192.0)
        return convert(p, T18, F16), {"q": q, "k": k, "v": v}, np_softmax(scores) @ v

    @case(
        "attention_decomposed",
        "ANE",
        "and the same mathematics written as transpose/matmul/softmax/matmul",
    )
    def _():
        @mb.program(input_specs=[spec(1, 256, 8192) for _ in range(3)], opset_version=T18)
        def p(q, k, v):
            kt = mb.transpose(x=k, perm=[0, 2, 1])
            s = mb.matmul(x=q, y=kt)
            # The same 1/sqrt(head_dim) the fused op applies internally. Without
            # it the scores reach ~30 over 8192 dimensions and exponentiating
            # their differences is hostile in fp16 -- which would make this a
            # test of fp16 range rather than of where the graph runs.
            sc = mb.mul(x=s, y=1.0 / float(np.sqrt(8192.0)))
            w = mb.softmax(x=sc, axis=-1)
            return mb.matmul(x=w, y=v)

        q, k, v = rnd(1, 256, 8192, seed=18), rnd(1, 256, 8192, seed=19), rnd(1, 256, 8192, seed=20)
        scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(8192.0)
        return convert(p, T18, F16), {"q": q, "k": k, "v": v}, np_softmax(scores) @ v

    # --- Negative controls -------------------------------------------------

    @case("matmul_fp32_4x4", "CPU", "a small fp32 matmul stays on the CPU", tol=1e-5)
    def _():
        @mb.program(input_specs=[spec(4, 4), spec(4, 4)], opset_version=T17)
        def p(a, b):
            return mb.matmul(x=a, y=b)

        a, b = rnd(4, 4, seed=21), rnd(4, 4, seed=22)
        return convert(p, T17, F32), {"a": a, "b": b}, a @ b

    @case(
        "matmul_fp32_1024",
        "GPU",
        "a large one goes to the GPU -- fp32 never reaches the ANE, but it does not mean CPU",
        tol=1e-4,
    )
    def _():
        @mb.program(input_specs=[spec(1024, 1024), spec(1024, 1024)], opset_version=T17)
        def p(a, b):
            return mb.matmul(x=a, y=b)

        a, b = rnd(1024, 1024, seed=23), rnd(1024, 1024, seed=24)
        return convert(p, T17, F32), {"a": a, "b": b}, a @ b

    @case(
        "softmax_f16_512_alone",
        "CPU",
        "the same softmax as above, with nothing to anchor it",
        tol=1e-3,
    )
    def _():
        @mb.program(input_specs=[spec(1, 512, 512)], opset_version=T17)
        def p(x):
            return mb.softmax(x=x, axis=-1)

        x = rnd(1, 512, 512, seed=25)
        return convert(p, T17, F16), {"x": x}, np_softmax(x)

    @case(
        "chain_without_matmul",
        "CPU",
        "four ops and no matmul: op count is not what moves a graph",
    )
    def _():
        @mb.program(input_specs=[spec(1, 512, 512)], opset_version=T17)
        def p(x):
            n = mb.layer_norm(x=x, axes=[-1])
            g = mb.gelu(x=n)
            e = mb.mul(x=g, y=g)
            return mb.softmax(x=e, axis=-1)

        x = rnd(1, 512, 512, seed=26)
        g = np_gelu(np_layer_norm(x))
        return convert(p, T17, F16), {"x": x}, np_softmax(g * g)

    return cases


def check_numerics(model, inputs, reference, tol):
    """Run the model on the given inputs and compare against the reference.

    Returns (ok, detail). The tolerance is relative to the reference's own
    scale, because fp16 compute over a long reduction has an absolute error
    proportional to the magnitudes involved.
    """
    out = model.predict(inputs)
    got = np.asarray(next(iter(out.values())), dtype=np.float32)
    ref = np.asarray(reference, dtype=np.float32)
    if got.shape != ref.shape:
        return False, f"shape {got.shape} != reference {ref.shape}"
    scale = max(1e-6, float(np.abs(ref).max()))
    err = float(np.abs(got - ref).max()) / scale
    # A correct-looking but constant output would pass a loose tolerance, so the
    # spread is checked too: a reference that varies must not come back flat.
    if float(ref.std()) > 1e-6 and float(got.std()) <= 1e-9:
        return False, "output is constant where the reference varies"
    return err <= tol, f"rel err {err:.2e} (tol {tol:.0e})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None, help="where to build models (default: a temp dir)")
    ap.add_argument("--list", action="store_true", help="list the cases and exit")
    args = ap.parse_args()

    _need(sys.platform == "darwin", "not macOS")
    _need(shutil.which("xcrun") is not None, "xcrun not found")

    cases = build_cases()
    if args.list:
        for name, expect, note, _, _ in cases:
            print(f"{name:34s} expect {expect:3s}  {note}")
        return 0

    # coremlc ships with Xcode rather than the Command Line Tools, and `xcrun`
    # finds it only when DEVELOPER_DIR points at a full Xcode.
    if subprocess.run(["xcrun", "--find", "coremlc"], capture_output=True).returncode != 0:
        os.environ.setdefault("DEVELOPER_DIR", "/Applications/Xcode.app/Contents/Developer")
        _need(
            subprocess.run(["xcrun", "--find", "coremlc"], capture_output=True).returncode == 0,
            "coremlc not found (needs Xcode, not just the Command Line Tools)",
        )

    tmp = None
    out_dir = args.out_dir
    if out_dir is None:
        tmp = tempfile.mkdtemp(prefix="vx-ane-check-")
        out_dir = tmp
    os.makedirs(out_dir, exist_ok=True)

    failures = []
    try:
        for name, expect, note, build, tol in cases:
            try:
                model, inputs, reference = build()
                placement = compile_and_plan(model, name, out_dir)
                num_ok, detail = check_numerics(model, inputs, reference, tol)
            except Exception as e:  # noqa: BLE001
                failures.append((name, expect, f"error: {e}"))
                print(f"FAIL {name:34s} could not be built, planned or run: {e}")
                continue
            devices = set(placement.values())
            place_ok = devices == {expect}
            ok = place_ok and num_ok
            status = "ok  " if ok else "FAIL"
            print(
                f"{status} {name:34s} expect {expect:3s} got {sorted(devices)}  {detail}  {placement}"
            )
            if not ok:
                why = []
                if not place_ok:
                    why.append(f"placement {sorted(devices)}")
                if not num_ok:
                    why.append(f"numerics {detail}")
                failures.append((name, expect, "; ".join(why)))
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)

    print()
    if failures:
        print(f"{len(failures)} of {len(cases)} cases disagree with what was measured:")
        for name, expect, got in failures:
            print(f"  {name}: expected every op on {expect}, {got}")
        return 1
    print(f"all {len(cases)} cases match: placement and numbers both as measured")
    return 0


if __name__ == "__main__":
    sys.exit(main())
