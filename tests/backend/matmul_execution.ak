// EXPECT: Unranked Memref base@ =
// EXPECT: rank = 2 offset = 0 sizes = [4, 4] strides = [4, 1] data =
// EXPECT: [[24,   24,   24,   24],
// EXPECT:  [24,   24,   24,   24],
// EXPECT:  [24,   24,   24,   24],
// EXPECT:  [24,   24,   24,   24]]

fn custom_matmul(a: Ref<Tensor, Memory::NPU_HBM>, b: Ref<Tensor, Memory::NPU_HBM>) -> Verified<Tensor> {
  spawn on(Topology::NPU[0]) {
    let mut result = Tensor([4, 4]).with_memory(Memory::NPU_HBM);
    for i in 0..4 {
      for j in 0..4 {
        result[i][j] = 0;
        for k in 0..4 {
          result[i][j] = result[i][j] + a[i][k] * b[k][j];
        }
      }
    }
    return Verified(result);
  }
}

fn main() -> Verified<Tensor> {
  let mut a = Tensor([4, 4]).with_memory(Memory::NPU_HBM);
  let mut b = Tensor([4, 4]).with_memory(Memory::NPU_HBM);

  for i in 0..4 {
    for j in 0..4 {
      a[i][j] = 2;
      b[i][j] = 3;
    }
  }

  let result = custom_matmul(a, b);
  print(result);
  return result;
}
