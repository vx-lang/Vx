// RUN: akarc %s

// Expect execution trap/segfault when trying to run an NPU execution block on host JIT
// The goal is to expose that our compiler JIT backend does not yet sandbox NPU pointers,
// allowing undefined behavior.
fn strip_pin_and_deref(t: Tensor<f32>) -> Tensor<f32> {
  unsafe {
    let ptr: *const Tensor<f32> = t.as_ptr();
    let val: f32 = *ptr; // Dereferencing an NPU pointer on CPU!
  }
  return t;
}

fn main() -> Tensor<f32> {
  let t_host: Tensor<f32> = Tensor([1, 1]);
  let t_device: Tensor<f32> = t_host.to_device();

  // Pass the NPU tensor to a Host function to strip the Pinned wrapper
  let result = strip_pin_and_deref(t_device);

  return result;
}

// EXPECT: 0
