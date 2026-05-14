struct Dummy { val: Tensor<f32> }

fn test_invalid_method(d: Dummy) -> Tensor<i32> {
  let ptr = d.as_ptr(); // ERROR: Cannot call as_ptr on Struct
  return 0;
}
