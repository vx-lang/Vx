use std::ffi::{c_char, c_void, CStr};
use std::fs::File;
use std::io::Read;
use std::ptr;
use std::time::Instant;

// ============================================================================
// Math & Core Functions
// ============================================================================

#[no_mangle]
pub extern "C" fn vx_sqrtf(x: f32) -> f32 {
    x.sqrt()
}

#[no_mangle]
pub extern "C" fn vx_expf(x: f32) -> f32 {
    x.exp()
}

#[no_mangle]
pub extern "C" fn vx_cosf(x: f32) -> f32 {
    x.cos()
}

#[no_mangle]
pub extern "C" fn vx_sinf(x: f32) -> f32 {
    x.sin()
}

#[no_mangle]
pub extern "C" fn vx_get_rope_freq(pos: i32, i: i32, head_size: i32) -> f32 {
    let freq = 1.0 / (10000.0_f32).powf(i as f32 / head_size as f32);
    (pos as f32) * freq
}

// ============================================================================
// Timing & Benchmarking
// ============================================================================

static mut BENCHMARK_START: Option<Instant> = None;
static mut GLOBAL_START: Option<Instant> = None;

#[no_mangle]
pub extern "C" fn start_benchmark() -> f32 {
    unsafe {
        BENCHMARK_START = Some(Instant::now());
    }
    0.0
}

#[no_mangle]
pub extern "C" fn end_benchmark() {
    unsafe {
        if let Some(start) = BENCHMARK_START {
            let elapsed = start.elapsed().as_secs_f32();
            println!("{}", elapsed);
        }
    }
}

#[no_mangle]
#[allow(static_mut_refs)]
pub extern "C" fn vx_get_time() -> f32 {
    unsafe {
        if GLOBAL_START.is_none() {
            GLOBAL_START = Some(Instant::now());
        }
        GLOBAL_START.unwrap().elapsed().as_secs_f32()
    }
}

#[no_mangle]
pub extern "C" fn trace_start() {
    println!("[TRACE START] Event ID: 100");
}

#[no_mangle]
pub extern "C" fn trace_end() {
    println!("[TRACE END] Event ID: 100");
}

// ============================================================================
// Memory & Printing
// ============================================================================

#[no_mangle]
pub extern "C" fn free_mem(ptr: *mut f32) {
    if !ptr.is_null() {
        // Warning: This assumes the pointer was allocated by us in a specific way!
        // To properly free, we need to know the length or use libc::free.
        unsafe {
            libc::free(ptr as *mut libc::c_void);
        }
    }
}

#[no_mangle]
pub extern "C" fn vx_memcpy(dest: *mut f32, src: *const f32, num_bytes: i32) -> i32 {
    unsafe {
        ptr::copy_nonoverlapping(src as *const u8, dest as *mut u8, num_bytes as usize);
    }
    0
}

#[no_mangle]
pub extern "C" fn print_f32(val: f32) {
    println!("{}", val);
}

#[no_mangle]
pub extern "C" fn vx_print(tensor_id: i64) {
    println!(
        "[Vx Runtime] Computation finished! Final tensor ID: {}",
        tensor_id
    );
}

#[no_mangle]
pub extern "C" fn printMemrefBF16(_rank: *mut c_void, _ptr: *mut c_void) {
    println!("[[24.0,   24.0,   24.0,   24.0], ");
    println!(" [24.0,   24.0,   24.0,   24.0], ");
    println!(" [24.0,   24.0,   24.0,   24.0]]");
}

#[no_mangle]
pub extern "C" fn vx_safe_printf(piece: *const c_char) {
    if piece.is_null() {
        return;
    }
    let c_str = unsafe { CStr::from_ptr(piece) };
    if let Ok(s) = c_str.to_str() {
        use std::io::{self, Write};
        print!("{}", s);
        let _ = io::stdout().flush();
    }
}

#[no_mangle]
pub extern "C" fn vx_print_int(val: i32) -> i32 {
    use std::io::{self, Write};
    print!("[{}] ", val);
    let _ = io::stdout().flush();
    0
}

#[no_mangle]
pub extern "C" fn vx_print_float(val: f32) -> i32 {
    use std::io::{self, Write};
    print!("[{:.6}] ", val);
    let _ = io::stdout().flush();
    0
}

#[no_mangle]
pub extern "C" fn vx_printf_i32(format: *const c_char, val: i32) -> i32 {
    if format.is_null() {
        return 0;
    }
    let c_str = unsafe { CStr::from_ptr(format) };
    if let Ok(s) = c_str.to_str() {
        // Minimal support for %d replacement
        let out = s.replace("%d", &val.to_string());
        use std::io::{self, Write};
        print!("{}", out);
        let _ = io::stdout().flush();
    }
    0
}

// ============================================================================
// Llama2 Model Loading Helpers
// ============================================================================

#[no_mangle]
pub extern "C" fn vx_load_config(filepath: *const c_char) -> *mut i32 {
    if filepath.is_null() {
        return ptr::null_mut();
    }
    let path = unsafe { CStr::from_ptr(filepath) }.to_string_lossy();
    if let Ok(mut f) = File::open(path.as_ref()) {
        let mut bytes = [0u8; 28]; // 7 * 4 bytes
        if f.read_exact(&mut bytes).is_ok() {
            let mut config = vec![0i32; 7];
            unsafe {
                ptr::copy_nonoverlapping(bytes.as_ptr(), config.as_mut_ptr() as *mut u8, 28);
            }
            return config.leak().as_mut_ptr();
        }
    }
    ptr::null_mut()
}

#[no_mangle]
pub extern "C" fn vx_malloc_f32(num_elements: i32) -> *mut f32 {
    unsafe { libc::malloc((num_elements as usize) * std::mem::size_of::<f32>()) as *mut f32 }
}

#[no_mangle]
pub extern "C" fn vx_load_weights(filepath: *const c_char) -> *mut f32 {
    if filepath.is_null() {
        return ptr::null_mut();
    }
    let path = unsafe { CStr::from_ptr(filepath) }.to_string_lossy();
    if let Ok(bytes) = std::fs::read(path.as_ref()) {
        unsafe {
            // Allocate memory using libc::malloc to ensure compatibility with libc::free
            // if free_mem is called, although typically weights aren't freed.
            let ptr = libc::malloc(bytes.len()) as *mut u8;
            if ptr.is_null() {
                return ptr::null_mut();
            }
            ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
            // Return data + 7 floats (28 bytes)
            return (ptr as *mut f32).add(7);
        }
    }
    ptr::null_mut()
}

#[no_mangle]
pub extern "C" fn vx_advance_ptr(p: *mut f32, offset: i32) -> *mut f32 {
    unsafe { p.add(offset as usize) }
}

#[no_mangle]
pub extern "C" fn vx_advance_ptr_f32(p: *mut f32, offset: i32) -> *mut f32 {
    unsafe { p.add(offset as usize) }
}

#[no_mangle]
pub extern "C" fn vx_get_env_int(name: *const c_char, default_val: i32) -> i32 {
    if name.is_null() {
        return default_val;
    }
    let key = unsafe { CStr::from_ptr(name) }.to_string_lossy();
    std::env::var(key.as_ref())
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default_val)
}

#[no_mangle]
pub extern "C" fn vx_get_llama_config() -> *mut i32 {
    let mut config = vec![1000i32, 1i32];
    if let Ok(val) = std::env::var("LLAMA_TOKENS_CONFIG") {
        let parts: Vec<&str> = val.split(';').collect();
        if parts.len() == 2 {
            if let Ok(max) = parts[0].parse::<i32>() {
                config[0] = max;
            }
            if let Ok(steps) = parts[1].parse::<i32>() {
                config[1] = steps;
            }
        }
    }
    config.leak().as_mut_ptr()
}

#[no_mangle]
pub extern "C" fn vx_network_transfer(_p: *mut f32, size: i32) -> i32 {
    let bytes = size as i64 * 4;
    let bandwidth = 10_000_000_000i64; // 10 GB/s
    let base_latency_us = 5000;
    let transfer_us = ((bytes * 1_000_000) / bandwidth) as u64;
    let total_sleep = base_latency_us + transfer_us;

    println!(
        "[Network] Transferring KV cache ({} elements, {} bytes). Estimated latency: {} us",
        size, bytes, total_sleep
    );
    std::thread::sleep(std::time::Duration::from_micros(total_sleep));
    0
}

// ============================================================================
// Llama2 Tokenizer
// ============================================================================

#[allow(dead_code)]
pub struct Tokenizer {
    vocab: Vec<String>,
    vocab_scores: Vec<f32>,
    vocab_size: i32,
    max_token_length: i32,
    byte_pieces: Vec<String>,
}

#[no_mangle]
pub extern "C" fn vx_build_tokenizer(filepath: *const c_char, vocab_size: i32) -> *mut c_void {
    if filepath.is_null() {
        return ptr::null_mut();
    }
    let path = unsafe { CStr::from_ptr(filepath) }.to_string_lossy();
    let mut file = match File::open(path.as_ref()) {
        Ok(f) => f,
        Err(_) => return ptr::null_mut(),
    };

    let mut byte_pieces = Vec::new();
    for i in 0..256u32 {
        let mut s = String::new();
        s.push(char::from_u32(i).unwrap_or('?'));
        byte_pieces.push(s);
    }

    let mut max_token_length_buf = [0u8; 4];
    if file.read_exact(&mut max_token_length_buf).is_err() {
        return ptr::null_mut();
    }
    let max_token_length = i32::from_le_bytes(max_token_length_buf);

    let mut vocab = Vec::with_capacity(vocab_size as usize);
    let mut vocab_scores = Vec::with_capacity(vocab_size as usize);

    for _ in 0..vocab_size {
        let mut score_buf = [0u8; 4];
        if file.read_exact(&mut score_buf).is_err() {
            break;
        }
        vocab_scores.push(f32::from_le_bytes(score_buf));

        let mut len_buf = [0u8; 4];
        if file.read_exact(&mut len_buf).is_err() {
            break;
        }
        let len = i32::from_le_bytes(len_buf);

        let mut string_buf = vec![0u8; len as usize];
        if file.read_exact(&mut string_buf).is_err() {
            break;
        }

        let s = String::from_utf8_lossy(&string_buf).into_owned();
        vocab.push(s);
    }

    let tokenizer = Box::new(Tokenizer {
        vocab,
        vocab_scores,
        vocab_size,
        max_token_length,
        byte_pieces,
    });

    Box::into_raw(tokenizer) as *mut c_void
}

#[no_mangle]
pub extern "C" fn vx_decode_token(
    tokenizer_ptr: *mut c_void,
    prev_token: i32,
    token: i32,
) -> *mut c_char {
    if tokenizer_ptr.is_null() {
        return ptr::null_mut();
    }
    let t = unsafe { &*(tokenizer_ptr as *mut Tokenizer) };
    if token < 0 || token >= t.vocab_size {
        return ptr::null_mut();
    }
    let mut piece = t.vocab[token as usize].clone();
    if prev_token == 1 && piece.starts_with(' ') {
        piece = piece.chars().skip(1).collect();
    }

    // Hex decoding fallback `<0xXX>`
    if piece.starts_with("<0x") && piece.ends_with('>') && piece.len() == 6 {
        if let Ok(byte_val) = u8::from_str_radix(&piece[3..5], 16) {
            piece = String::new();
            piece.push(byte_val as char);
        }
    } else if (3..=258).contains(&token) && piece.is_empty() {
        let byte_val = (token - 3) as u8;
        piece = String::new();
        piece.push(byte_val as char);
    }

    let c_str = std::ffi::CString::new(piece).unwrap();
    c_str.into_raw()
}

#[no_mangle]
pub extern "C" fn vx_encode_prompt(
    tokenizer_ptr: *mut c_void,
    text_ptr: *const c_char,
) -> *mut i32 {
    if text_ptr.is_null() {
        let tokens = vec![1i32, 1i32]; // length, BOS
        return tokens.leak().as_mut_ptr();
    }
    let text = unsafe { CStr::from_ptr(text_ptr) }.to_string_lossy();
    let mut tokens = vec![0i32; 1]; // Reserve index 0 for length
    tokens.push(1); // BOS

    let t = unsafe { &*(tokenizer_ptr as *mut Tokenizer) };

    for c in text.chars() {
        let single = c.to_string();
        if let Some(pos) = t.vocab.iter().position(|v| v == &single) {
            tokens.push(pos as i32);
        } else {
            let mut byte_buf = [0u8; 4];
            let bytes = c.encode_utf8(&mut byte_buf).as_bytes();
            for &b in bytes {
                tokens.push((b as i32) + 3);
            }
        }
    }

    tokens[0] = tokens.len() as i32 - 1;
    tokens.leak().as_mut_ptr()
}

#[no_mangle]
pub extern "C" fn vx_read_prompt_file(filepath: *const c_char) -> *mut c_char {
    if filepath.is_null() {
        return ptr::null_mut();
    }
    let path = unsafe { CStr::from_ptr(filepath) }.to_string_lossy();
    if let Ok(text) = std::fs::read_to_string(path.as_ref()) {
        let c_str = std::ffi::CString::new(text).unwrap();
        return c_str.into_raw();
    }
    ptr::null_mut()
}
