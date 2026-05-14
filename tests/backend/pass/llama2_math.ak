struct Config {
    dim: i32,
    hidden_dim: i32,
    n_layers: i32,
    n_heads: i32,
    n_kv_heads: i32,
    vocab_size: i32,
    seq_len: i32
}

struct TransformerWeights {
    token_embedding_table: *mut f32,
    rms_att_weight: *mut f32,
    rms_ffn_weight: *mut f32,
    wq: *mut f32,
    wk: *mut f32,
    wv: *mut f32,
    wo: *mut f32,
    w1: *mut f32,
    w2: *mut f32,
    w3: *mut f32,
    rms_final_weight: *mut f32,
    wcls: *mut f32
}

extern {
    fn malloc(size: i64) -> *mut f32;
    fn free(ptr: *mut f32) -> i32;
    fn printf(format: *const i8, val: f32) -> i32;
    fn akar_sqrtf(x: f32) -> f32;
}

fn rmsnorm(o: *mut f32, x: *mut f32, weight: *mut f32, size: i32) -> i32 {
    let mut ss: f32 = 0.0;
    for j in 0..size {
        let val: f32 = x[j];
        ss += val * val;
    }
    
    let size_f: f32 = 0.0; // Needs cast or intrinsic for proper size conversion
    // For simplicity in test, let's hardcode size_f logic later or assume input size_f
    // ss = ss / size_f + 1e-5;
    // ss = 1.0 / akar_sqrtf(ss);
    
    return 0;
}

fn main() -> i32 {
    let config = Config {
        dim: 256,
        hidden_dim: 512,
        n_layers: 4,
        n_heads: 8,
        n_kv_heads: 8,
        vocab_size: 32000,
        seq_len: 1024
    };
    
    let dim_f: f32 = 256.0;
    unsafe {
        printf("Dim: %f\n", dim_f);
    }
    return 0;
}

// CHECK: func.func @main() -> i32 {
// CHECK: = llvm.mlir.undef : !llvm.struct<"Config", (i32, i32, i32, i32, i32, i32, i32)>
