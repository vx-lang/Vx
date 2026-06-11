# Attention Rewrite: FlashAttention & GQA

This plan outlines the refactoring of the Llama2 attention mechanism across our codebase to implement FlashAttention (single-pass online softmax) and formalize Grouped Query Attention (GQA).

## Current State & Bottlenecks

Currently, attention is calculated inline inside the `transformer` loops via a multi-pass approach over the context window (`pos`):

1. **Pass 1:** Calculate all `pos` dot-products and store in `s.att`
1. **Pass 2:** Find `max` over `s.att`
1. **Pass 3:** Calculate `exp` and running sum
1. **Pass 4:** Normalize `s.att`
1. **Pass 5:** Multiply normalized weights with `v_cache`

This requires `O(seq_len)` memory per head (the `s.att` buffer) and wastes massive memory bandwidth by reading/writing the `s.att` tensor 5 times.

Furthermore, while the index math `h_kv_mul = h / (n_heads / n_kv_heads)` already technically achieves GQA routing, the logic is deeply tangled inside the forward pass.

## Proposed Changes

### 1. Factor out `attn` Function

We will create a standalone `attn` function that can be easily invoked from any Llama2 benchmark or example script:

```rust
fn attn(
  xb: *mut f32,          // Output buffer
  q: *mut f32,           // Query vector
  key_cache: *mut f32,   // K cache 
  value_cache: *mut f32, // V cache
  seq_len: i32,
  pos: i32,
  n_heads: i32,
  n_kv_heads: i32,
  head_size: i32,
  kv_dim: i32,
  layer_offset: i32      // Precomputed l * seq_len * kv_dim
) -> i32
```

### 2. Implement FlashAttention (Online Softmax)

We will replace the 5-pass algorithm with a single-pass fused loop for exact Softmax Attention (often referred to as FlashAttention for decoding). For each query head:

```python
# Initialize running statistics
m = -inf 
l = 0.0  
out = zeros(head_size)

for t in 0..pos+1:
    # 1. Compute score
    score = dot(q, k_cache[t]) / sqrt(head_size)
    
    # 2. Update running max
    m_new = max(m, score)
    
    # 3. Compute correction factor for previous running stats
    correction = exp(m - m_new)
    
    # 4. Compute exp for current score
    exp_val = exp(score - m_new)
    
    # 5. Update running sum
    l = l * correction + exp_val
    
    # 6. Update running output vector
    for i in 0..head_size:
        out[i] = out[i] * correction + exp_val * v_cache[t][i]

# 7. Final normalization
for i in 0..head_size:
    xb[i] = out[i] / l
```

### 3. Formalize GQA

Inside the `attn` loop, the `k_cache` and `v_cache` offsets will elegantly use `n_kv_heads` to load the correct group vector without repeating data, fully embracing GQA logic:

```rust
let kv_mul : i32 = n_heads / n_kv_heads;
let kv_head_idx : i32 = h / kv_mul;
```

### Target Files

We will apply this rewrite to:

- `benchmarks/llama2_100.vx`
- `benchmarks/llama2_scaling.vx`
- `examples/llama.vx` (which uses high-level Tensors but needs the same algorithmic update)

## User Review Required

> [!IMPORTANT]
> The single-pass FlashAttention requires updating local vector states (`out[i]`) iteratively. This is highly optimized for CPUs/NPUs due to L1 cache locality, but completely drops the `s.att` array. I will remove `s.att` from `RunState`. Does this sound good to you?
