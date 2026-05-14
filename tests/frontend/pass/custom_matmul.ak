fn custom_matmul(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32> {
  let mut result: Tensor<f32> = Tensor([a.shape[0], b.shape[1]]);
  for i in 0..a.shape[0] {
    for j in 0..b.shape[1] {
      result[i][j] = 0;
      for k in 0..a.shape[1] {
        result[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  return result;
}

fn distributed_matmul(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32> {
  let local_a = a.to_device();
  let local_b = b.to_device();
  spawn on(Topology::NPU[0]) {
    let result = custom_matmul(local_a, local_b);
    return result;
  }
}
