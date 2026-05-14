// EXPECT:  [24,   24,   24,   24],
// EXPECT:  [24,   24,   24,   24],
// EXPECT:  [24,   24,   24,   24]]

fn custom_matmul(a: Ref<Tensor<f64>, Memory::NPU_HBM>, b: Ref<Tensor<f64>, Memory::NPU_HBM>) -> Verified<Tensor<f64>> {
  spawn on(Topology::NPU[0]) {
    let mut result = Tensor<f64>([4, 4]).with_memory(Memory::NPU_HBM);
    for i in 0..4 {
      for j in 0..4 {
        result[i][j] = 0.0;
        for k in 0..4 {
          result[i][j] = result[i][j] + a[i][k] * b[k][j];
        }
      }
    }
    return Verified(result);
  }
}

fn main() -> Verified<Tensor<f64>> {
  let mut a = Tensor<f64>([4, 4]).with_memory(Memory::NPU_HBM);
  let mut b = Tensor<f64>([4, 4]).with_memory(Memory::NPU_HBM);

  for i in 0..4 {
    for j in 0..4 {
      a[i][j] = 2.0;
      b[i][j] = 3.0;
    }
  }

  let result = custom_matmul(a, b);
  print(result);
  return result;
}
