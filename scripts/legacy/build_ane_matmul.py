import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil.types.symbolic import any_symbolic

# We want to perform: out = w * x
# where w is (D, N) and x is (N, 1)

@mb.program(
    input_specs=[
        mb.TensorSpec(shape=(any_symbolic, any_symbolic)), # w: (d, n)
        mb.TensorSpec(shape=(any_symbolic, 1))             # x: (n, 1)
    ]
)
def matmul_prog(w, x):
    # Perform matrix multiplication
    # CoreML matmul does (D, N) x (N, 1) -> (D, 1)
    res = mb.matmul(x=w, y=x, transpose_x=False, transpose_y=False)
    return res

if __name__ == "__main__":
    print("Converting to MLProgram...")
    mlmodel = ct.convert(
        matmul_prog,
        source="milinternal",
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.ALL
    )
    
    print("Saving to matmul.mlpackage...")
    mlmodel.save("matmul.mlpackage")
    print("Successfully built matmul.mlpackage!")
